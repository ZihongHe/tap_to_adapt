"""Server-authoritative study phase and session-state transitions.

StudyState validates events against the participant's protocol, device role,
active block, and completion state. Breaks and incomplete sessions retain
their distinct semantics for timing eligibility and outcome aggregation.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .db import StudyDB
from .operator_resume import keep_milestone_event, load_operator_resume
from .pipeline_guard import requires_locked_pipeline
from .protocol import Protocol


COMPLETE_STATES = {"completed", "ended"}
TERMINAL_STATES = COMPLETE_STATES | {"aborted"}


class ProtocolViolation(ValueError):
    def __init__(self, code: str, message: str, checks: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.checks = checks or {}

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "checks": self.checks}


class StudyState:
    def __init__(self, db: StudyDB, protocol: Protocol):
        self.db = db
        self.protocol = protocol

    def assert_can_start(
        self,
        participant: dict[str, Any],
        day: int,
        role: str,
        *,
        allow_aborted_restart: bool = False,
        allow_interrupt_reopen: bool = False,
    ) -> None:
        try:
            study = self.protocol.study_for(participant)
            rules = self.protocol.rules_for(participant)
        except ValueError as exc:
            raise ProtocolViolation(
                "protocol_version_mismatch", "该编号不属于当前实验协议，请联系研究团队。"
            ) from exc
        total_sessions = int(study.get("session_design", {}).get("total_sessions", 3))
        if day < 1 or day > total_sessions:
            raise ProtocolViolation(
                "session_out_of_range",
                "该编号只对应一次完整实验，不能创建额外实验日。",
                {"requested_session": day, "total_sessions": total_sessions},
            )
        sessions = self.db.sessions(participant["id"])
        same = next(
            (row for row in sessions if int(row["day"]) == day and row["client_role"] == role),
            None,
        )
        if same is not None:
            if same["state"] in TERMINAL_STATES:
                if same["state"] == "aborted" and (
                    allow_aborted_restart or allow_interrupt_reopen
                ):
                    return
                raise ProtocolViolation(
                    "day_already_closed",
                    f"第 {day} 天已经结束，不能重新建立会话。",
                    {"day": day, "state": same["state"]},
                )
            return

        if role == "companion":
            study = next(
                (
                    row
                    for row in sessions
                    if int(row["day"]) == day and row["client_role"] == "study"
                ),
                None,
            )
            if study is None or study["state"] in TERMINAL_STATES:
                if (
                    allow_interrupt_reopen
                    and study is not None
                    and str(study.get("state") or "") == "aborted"
                ):
                    return
                raise ProtocolViolation(
                    "study_session_required",
                    "请先在电脑端建立同一天的实验会话。",
                    {"day": day},
                )
            return

        if any(int(row["day"]) > day for row in sessions):
            raise ProtocolViolation(
                "day_order_invalid", "实验日顺序异常，请联系研究团队。", {"day": day}
            )
        if day == 1:
            return

        previous = next(
            (
                row
                for row in sessions
                if int(row["day"]) == day - 1 and row["client_role"] == "study"
            ),
            None,
        )
        if previous is None or previous["state"] not in COMPLETE_STATES or previous["ended_ms"] is None:
            raise ProtocolViolation(
                "previous_day_incomplete",
                f"必须先完整完成第 {day - 1} 天，才能开始第 {day} 天。",
                {"required_day": day - 1},
            )

        now_ms = int(time.time() * 1000)
        gap_ms = now_ms - int(previous["ended_ms"])
        required_gap_ms = int(float(rules["min_day_gap_seconds"]) * 1000)
        checks = {
            "previous_day_completed": True,
            "gap_ms": gap_ms,
            "required_gap_ms": required_gap_ms,
        }
        if gap_ms < required_gap_ms:
            raise ProtocolViolation(
                "day_gap_too_short",
                "本次实验需要跨日完成，现在距离上一日结束时间仍不足。",
                checks,
            )
        if rules.get("require_later_local_date"):
            timezone = ZoneInfo(str(study["timezone"]))
            previous_date = datetime.fromtimestamp(int(previous["ended_ms"]) / 1000, timezone).date()
            current_date = datetime.fromtimestamp(now_ms / 1000, timezone).date()
            checks.update(
                {"previous_local_date": str(previous_date), "current_local_date": str(current_date)}
            )
            if current_date <= previous_date:
                raise ProtocolViolation(
                    "later_local_date_required",
                    "下一实验日必须在新的自然日开始。",
                    checks,
                )

    def assert_can_activate(self, participant: dict[str, Any], session_id: str) -> None:
        try:
            study = self.protocol.study_for(participant)
            self.protocol.rules_for(participant)
        except ValueError as exc:
            raise ProtocolViolation(
                "protocol_version_mismatch", "该编号不属于当前实验协议，请联系研究团队。"
            ) from exc
        snapshot = self.db.protocol_snapshot(participant["id"], session_id)
        session = snapshot["session"]
        if session is None or session["client_role"] != "study":
            raise ProtocolViolation("study_session_required", "只能启动电脑端实验会话。")
        if session["state"] == "active" or session.get("activated_ms") is not None:
            raise ProtocolViolation(
                "session_already_active",
                "今天的实验流程已经启动，请勿重复点击；若页面曾刷新，请联系研究团队重新安排。",
            )
        if int(session["day"]) == 1 and not self.db.response_exists(
            participant["id"], session_id, "baseline"
        ):
            raise ProtocolViolation(
                "baseline_incomplete",
                "请先完成第 1 天开始前信息与操作理解检查。",
            )
        latest: dict[str, dict[str, Any]] = {}
        for row in snapshot["quality"]:
            latest.setdefault(str(row["kind"]), row)
        missing = [
            kind
            for kind in study["required_quality_checks"]
            if kind not in latest or not bool(latest[kind]["passed"])
        ]
        if missing:
            raise ProtocolViolation(
                "quality_checks_incomplete",
                "网络、麦克风、反应时和浏览器检测全部通过后才能开始。",
                {"missing_or_failed": missing},
            )
        if participant.get("data_origin") == "formal":
            browser = latest.get("browser")
            browser_metrics = json.loads(browser["metrics"]) if browser else {}
            device_id = str(browser_metrics.get("device_id") or "")
            if not device_id:
                raise ProtocolViolation(
                    "device_identity_missing",
                    "浏览器设备标识缺失，请刷新后重新完成浏览器检测。",
                )
            day = int(session["day"])
            if day > 1:
                day_one_ids = []
                for row in self.db.quality_history(participant["id"], "browser"):
                    if int(row.get("day") or 0) != 1 or not bool(row["passed"]):
                        continue
                    metrics = json.loads(row["metrics"])
                    if metrics.get("device_id"):
                        day_one_ids.append(str(metrics["device_id"]))
                if not day_one_ids or device_id != day_one_ids[0]:
                    raise ProtocolViolation(
                        "study_device_changed",
                        "请使用第 1 天同一台电脑和同一浏览器继续；若设备损坏，请联系研究团队记录更换原因。",
                    )

    def assert_events_allowed(
        self,
        participant: dict[str, Any],
        session_id: str,
        events: list[dict[str, Any]],
    ) -> None:
        study = self.protocol.study_for(participant)
        snapshot = self.db.protocol_snapshot(participant["id"], session_id)
        session = snapshot["session"]
        if session is None or session["client_role"] != "study":
            raise ProtocolViolation("study_session_required", "实验事件必须来自电脑端会话。")
        topic_ids = {
            str(prompt["id"])
            for category in self.protocol.topics.get("categories", [])
            for prompt in category.get("prompts", [])
        }
        if study.get("topic_gate", {}).get("allow_free_topic"):
            topic_ids.add("free_topic")
        existing_topic_events = [
            row
            for row in snapshot["events"]
            if row["event_type"] in {"topic.selected", "topic.changed"}
        ]
        topic_seen = bool(existing_topic_events)
        current_topic_id = None
        if existing_topic_events:
            current_topic_id = json.loads(existing_topic_events[-1]["payload"]).get(
                "topic_id"
            )
        server_only_events = {
            "agent.auto_activate",
            "chat.turn",
            "pair.qualified_start",
            "phase.server_committed",
            "tap.interrupt",
            "tap.manual_activate",
            "tap.tap_burst_ignored",
            "tts.generated",
        }
        for event in events:
            event_type = str(event["event_type"])
            payload = event.get("payload") or {}
            if event_type in server_only_events:
                raise ProtocolViolation(
                    "server_event_required",
                    f"事件 {event_type} 只能由实验服务器生成。",
                )
            if event_type == "topic.selected":
                if topic_seen:
                    raise ProtocolViolation(
                        "topic_already_selected",
                        "今天的起始话题已经保存；如需切换，请使用页面内的更换话题操作。",
                    )
                topic_id = str(payload.get("topic_id") or "")
                if topic_id not in topic_ids:
                    raise ProtocolViolation(
                        "topic_invalid", "请选择页面话题库中的话题，或选择自由话题。"
                    )
                if not isinstance(payload.get("reused_topic_id"), bool):
                    raise ProtocolViolation(
                        "topic_reuse_flag_missing", "话题选择记录缺少重复使用标记，请刷新页面。"
                    )
                topic_seen = True
                current_topic_id = topic_id
            if event_type == "topic.changed":
                if not study.get("topic_gate", {}).get(
                    "allow_explicit_changes", False
                ):
                    raise ProtocolViolation(
                        "topic_change_disabled", "当前实验协议不允许中途更换话题。"
                    )
                if not topic_seen:
                    raise ProtocolViolation(
                        "topic_required", "请先保存今天的起始话题。"
                    )
                topic_id = str(payload.get("topic_id") or "")
                if topic_id not in topic_ids:
                    raise ProtocolViolation(
                        "topic_invalid", "请选择页面话题库中的话题，或选择自由话题。"
                    )
                if str(payload.get("from_topic_id") or "") != str(
                    current_topic_id or ""
                ):
                    raise ProtocolViolation(
                        "topic_change_stale", "话题已经在另一页面中更新，请刷新后重试。"
                    )
                if topic_id == current_topic_id:
                    raise ProtocolViolation(
                        "topic_change_duplicate", "当前已经是这个话题。"
                    )
                if payload.get("user_initiated") is not True:
                    raise ProtocolViolation(
                        "topic_change_intent_missing", "更换话题必须由参与者主动确认。"
                    )
                if not isinstance(payload.get("reused_topic_id"), bool):
                    raise ProtocolViolation(
                        "topic_reuse_flag_missing", "话题更换记录缺少重复使用标记。"
                    )
                current_topic_id = topic_id
            if event_type == "phase.start":
                if session["state"] != "active" or session.get("activated_ms") is None:
                    raise ProtocolViolation(
                        "session_not_active", "双端会话尚未激活，不能开始实验阶段。"
                    )
                if (
                    study.get("topic_gate", {}).get(
                        "required_before_first_phase", False
                    )
                    and not topic_seen
                ):
                    raise ProtocolViolation(
                        "topic_required", "请先选择今天准备交流的话题，再开始计时。"
                    )

    def assert_primary_chat_allowed(
        self, participant: dict[str, Any], session_id: str
    ) -> None:
        snapshot = self.db.protocol_snapshot(participant["id"], session_id)
        if not any(
            row["event_type"] in {"topic.selected", "topic.changed"}
            for row in snapshot["events"]
        ):
            raise ProtocolViolation(
                "topic_required", "请先选择今天准备交流的话题，再请求语音回复。"
            )

    def completion_report(
        self,
        participant: dict[str, Any],
        session_id: str,
        now_ms: int | None = None,
        *,
        require_final_response: bool = True,
    ) -> dict[str, Any]:
        now_ms = now_ms or int(time.time() * 1000)
        study = self.protocol.study_for(participant)
        snapshot = self.db.protocol_snapshot(participant["id"], session_id)
        session = snapshot["session"]
        if session is None or session["client_role"] != "study":
            raise ProtocolViolation("study_session_required", "只能完成电脑端实验会话。")
        rules = self.protocol.rules_for(participant)
        day = int(session["day"])
        checks: dict[str, Any] = {
            "protocol_version": participant["protocol_version"],
            "protocol_mode": participant["protocol_mode"],
            "data_origin": participant["data_origin"],
            "day": day,
        }

        activated_ms = session.get("activated_ms")
        completed = session["state"] == "completed" and session.get("ended_ms") is not None
        reference_ms = int(session["ended_ms"]) if completed else now_ms
        elapsed_ms = reference_ms - int(activated_ms) if activated_ms is not None else 0
        required_session_ms = int(float(rules["min_session_seconds"]) * 1000)
        checks["session"] = {
            "state": session["state"],
            "valid_for_completion": session["state"] in {
                "active",
                "questionnaire",
                "completed",
            },
            "elapsed_ms": elapsed_ms,
            "required_ms": required_session_ms,
        }

        pair_overlap_ms = 0
        open_pair = False
        paired_at_completion = False
        qualified_intervals = [
            interval for interval in snapshot["pair_intervals"] if interval.get("qualified")
        ]
        for interval in qualified_intervals:
            start = int(interval["started_ms"])
            if activated_ms is not None:
                start = max(start, int(activated_ms))
            interval_end = (
                int(interval["ended_ms"])
                if interval["ended_ms"] is not None
                else reference_ms
            )
            end = min(interval_end, reference_ms)
            pair_overlap_ms += max(0, end - start)
            open_pair = open_pair or interval["ended_ms"] is None
            paired_at_completion = paired_at_completion or (
                int(interval["started_ms"]) <= reference_ms
                and (
                    interval["ended_ms"] is None
                    or int(interval["ended_ms"]) >= reference_ms
                )
            )
        required_pair_ms = int(float(rules["min_pair_overlap_seconds"]) * 1000)
        checks["pair"] = {
            "currently_online": open_pair,
            "paired_at_completion": paired_at_completion,
            "online_required_at_completion": False,
            "overlap_ms": pair_overlap_ms,
            "required_ms": required_pair_ms,
            "attempt_count": len(snapshot["pair_intervals"]),
            "qualified_interval_count": len(qualified_intervals),
            "qualified_pair_count": len(
                {
                    (
                        str(interval["study_session_id"]),
                        str(interval["companion_session_id"]),
                    )
                    for interval in qualified_intervals
                }
            ),
        }

        latest_quality: dict[str, dict[str, Any]] = {}
        for row in snapshot["quality"]:
            latest_quality.setdefault(str(row["kind"]), row)
        quality_status = {
            kind: bool(latest_quality.get(kind, {}).get("passed"))
            for kind in study["required_quality_checks"]
        }
        checks["quality"] = quality_status

        session_design = study.get("session_design", {})
        final_day = int(session_design.get("total_sessions", 3))
        require_final = bool(session_design.get("require_final_questionnaire", True))
        response_names = {str(row["instrument"]) for row in snapshot["responses"]}
        required_responses = set()
        if final_day == 1:
            if require_final and require_final_response:
                required_responses.add("final")
        else:
            required_responses.add(f"daily_day_{day}")
        if day == 1:
            required_responses.add("baseline")
        if (
            day == final_day
            and final_day > 1
            and require_final
            and require_final_response
        ):
            required_responses.add("final")
        checks["responses"] = {
            "required": sorted(required_responses),
            "present": sorted(required_responses & response_names),
            "final_required_for_this_report": bool(require_final_response),
        }

        events = []
        for row in snapshot["events"]:
            events.append({**row, "payload": json.loads(row["payload"])})
        topic_events = [
            row
            for row in events
            if row["event_type"] in {"topic.selected", "topic.changed"}
        ]
        initial_topic_events = [
            row for row in topic_events if row["event_type"] == "topic.selected"
        ]
        topic_change_events = [
            row for row in topic_events if row["event_type"] == "topic.changed"
        ]
        primary_turns = [
            row
            for row in events
            if row["event_type"] == "chat.turn"
            and row["payload"].get("source") == "primary"
        ]
        minimum_turns = int(
            study.get("topic_gate", {}).get(
                "minimum_primary_chat_turns_per_day", 0
            )
        )
        checks["topic"] = {
            "selected": bool(topic_events),
            "selection_count": len(initial_topic_events),
            "change_count": len(topic_change_events),
            "current_topic_id": (
                topic_events[-1]["payload"].get("topic_id") if topic_events else None
            ),
            "primary_chat_turns": len(primary_turns),
            "required_primary_chat_turns": minimum_turns,
        }
        browser_tts = [
            row
            for row in events
            if row["event_type"] == "tts.browser_start"
            or (
                row["event_type"] == "tts.playback_start"
                and row["payload"].get("fallback")
            )
            or (
                row["event_type"] == "companion.chat_state"
                and row["payload"].get("mode") == "browser_fallback"
            )
        ]
        study_browser_primary = sum(
            1
            for row in events
            if row["event_type"] == "tts.playback_start"
            and row["payload"].get("provider") == "speechSynthesis"
            and not row["payload"].get("fallback")
        )
        tts_failures = [
            row
            for row in events
            if row["event_type"] in {"tts.server_failed", "tts.playback_failed"}
            or (
                row["event_type"] == "companion.chat_state"
                and row["payload"].get("action") == "tts_failed"
            )
        ]
        checks["tts"] = {
            "server_only_required": bool(
                study.get("tts_gate", {}).get("server_only")
            ),
            "browser_fallbacks": len(browser_tts),
            "failures": len(tts_failures),
            "server_generations": sum(
                row["event_type"] == "tts.generated" for row in events
            ),
            "browser_primary_generations": study_browser_primary,
            "tts_observations": sum(
                row["event_type"] == "tts.generated" for row in events
            )
            + study_browser_primary,
            "required_server_generations": int(
                study.get("tts_gate", {}).get(
                    "minimum_server_generations_per_day", 0
                )
            ),
        }
        checks["phases"] = self._phase_report(participant, day, events, rules)
        event_counts: dict[str, int] = {}
        for row in events:
            event_counts[row["event_type"]] = event_counts.get(row["event_type"], 0) + 1
        valid_predictions = [
            row
            for row in events
            if row["event_type"] == "model.prediction"
            and not row["payload"].get("skipped")
            and isinstance(row["payload"].get("probability"), (int, float))
            and bool(row["payload"].get("policy"))
            and bool(row["payload"].get("policy_effective"))
        ]
        checks["timing_stream"] = {
            "model_ready": event_counts.get("model.ready", 0) >= 1
            or (
                participant["protocol_mode"] == "compressed"
                and not requires_locked_pipeline(participant)
            ),
            "model_predictions": len(valid_predictions),
            "required_predictions": int(rules["min_model_predictions"]),
        }
        frozen_like = requires_locked_pipeline(participant) and participant[
            "group_name"
        ].lower() in {"frozen", "f", "holdout", "holdout_evaluator"}
        ready_fingerprints = [
            row["payload"].get("model_fingerprint", {})
            for row in events
            if row["event_type"] == "model.ready"
        ]
        prior = (
            self.db.shared_prior(str(participant.get("prior_id")))
            if participant.get("prior_id")
            else None
        )
        expected_fingerprint = (
            {
                "encoder_sha256": prior["encoder_sha256"],
                "main_sha256": prior["main_sha256"],
            }
            if prior
            else None
        )
        checks["frozen_model_identity"] = {
            "required": frozen_like,
            "ready_events": len(ready_fingerprints),
            "matches_locked_prior": bool(
                not frozen_like
                or (
                    expected_fingerprint
                    and ready_fingerprints
                    and all(
                        fingerprint == expected_fingerprint
                        for fingerprint in ready_fingerprints
                    )
                )
            ),
        }
        policy_mismatches = []
        forbidden_updates = []
        cross_phase_activations = []
        for row in valid_predictions:
            payload = row["payload"]
            expected_policy = payload.get("policy")
            effective = payload.get("policy_effective")
            if effective != expected_policy:
                policy_mismatches.append(
                    {"policy": expected_policy, "effective": effective}
                )
        for row in events:
            payload = row["payload"]
            if row["event_type"].startswith("tap.") and payload.get("training_eligible"):
                if (
                    payload.get("activation_phase") != "learning"
                    or payload.get("activation_policy") != "current"
                    or payload.get("phase") != payload.get("activation_phase")
                    or not payload.get("activation_id")
                    or participant["group_name"].lower()
                    in {"frozen", "f", "holdout", "holdout_evaluator"}
                ):
                    forbidden_updates.append(row["event_type"])
            if (
                row["event_type"] in {"tap.interrupt", "tap.result"}
                and payload.get("action") in {None, "interrupt"}
                and payload.get("activation_phase")
                and payload.get("tap_phase")
                and payload.get("activation_phase") != payload.get("tap_phase")
            ):
                cross_phase_activations.append(
                    {
                        "activation_id": payload.get("activation_id"),
                        "activation_phase": payload.get("activation_phase"),
                        "tap_phase": payload.get("tap_phase"),
                    }
                )
        checks["policy_execution"] = {
            "mismatch_count": len(policy_mismatches),
            "examples": policy_mismatches[:10],
            "forbidden_update_count": len(forbidden_updates),
            "cross_phase_activation_count": len(cross_phase_activations),
            "cross_phase_examples": cross_phase_activations[:10],
        }
        server_tap_rows = [
            row
            for row in events
            if row["event_type"].startswith("tap.")
            and row["event_type"] not in {"tap.pointer", "tap.result"}
        ]
        after_complete_tap_ids = {
            str(row["payload"].get("tap_id") or "")
            for row in server_tap_rows
            if row["payload"].get("phase") == "complete"
            or row["payload"].get("tap_phase") == "complete"
            or row["payload"].get("action") == "after_complete_ignored"
        }
        after_complete_tap_ids.discard("")
        accepted_taps = [
            str(row["payload"].get("tap_id") or "")
            for row in events
            if row["event_type"] == "tap.pointer" and row["payload"].get("accepted") is True
            and str(row["payload"].get("tap_id") or "") not in after_complete_tap_ids
        ]
        server_taps = [
            str(row["payload"].get("tap_id") or "")
            for row in server_tap_rows
            if str(row["payload"].get("tap_id") or "") not in after_complete_tap_ids
        ]
        acknowledged_taps = [
            str(row["payload"].get("tap_id") or "")
            for row in events
            if row["event_type"] == "tap.result"
            and str(row["payload"].get("tap_id") or "") not in after_complete_tap_ids
        ]
        receipt_complete = (
            all(accepted_taps)
            and all(server_taps)
            and all(acknowledged_taps)
            and len(accepted_taps) == len(set(accepted_taps))
            and len(server_taps) == len(set(server_taps))
            and len(acknowledged_taps) == len(set(acknowledged_taps))
            and set(accepted_taps) == set(server_taps) == set(acknowledged_taps)
        )
        checks["tap_receipts"] = {
            "accepted": len(accepted_taps),
            "server_recorded": len(server_taps),
            "client_acknowledged": len(acknowledged_taps),
            "complete": receipt_complete,
            "after_complete_ignored": len(after_complete_tap_ids),
        }
        pipeline = study.get("pipeline", {})
        budget_kind = str(pipeline.get("budget_kind") or "activation_opportunity")
        q_key = "q_feedback_levels" if budget_kind == "eligible_feedback" else "q_opportunity_levels"
        max_q = max((int(value) for value in pipeline.get(q_key, [])), default=0)
        q_levels = [int(value) for value in pipeline.get(q_key, [])]
        budget_semantics = str(pipeline.get("budget_semantics") or "exact_count")
        require_max_q_attainment = bool(
            pipeline.get("require_max_q_attainment", True)
        )
        checks["donor_budget"] = {
            "required": 0,
            "observed": 0,
            "budget_semantics": budget_semantics,
            "snapshots": [],
            "eligible_for_prior": True,
        }
        if (
            requires_locked_pipeline(participant)
            and participant.get("cohort") == "study_a"
            and participant.get("group_name")
            in {"donor", "seed_donor", "evaluator_donor"}
            and day == int(study.get("session_design", {}).get("total_sessions", 3))
        ):
            feedback_counts = self.db.feedback_counts(participant["id"])
            observed = (
                feedback_counts["eligible_feedbacks"]
                if budget_kind == "eligible_feedback"
                else self.db.learning_opportunity_count(participant["id"])
            )
            snapshots = self.db.model_snapshots(
                participant["run_id"],
                source="donor_shadow",
                budget_kind=budget_kind,
            )
            snapshot_budgets = sorted(
                int(row["opportunity_budget"])
                for row in snapshots
                if row["participant_id"] == participant["id"]
            )
            donor_snapshots = {
                int(row["opportunity_budget"]): row
                for row in snapshots
                if row["participant_id"] == participant["id"]
            }
            invalid_caps = [
                budget
                for budget, row in donor_snapshots.items()
                if budget in q_levels
                and budget_semantics == "per_donor_cap"
                and int(row["eligible_feedbacks"]) != min(observed, budget)
            ]
            checks["donor_budget"] = {
                "required": max_q,
                "observed": observed,
                "budget_kind": budget_kind,
                "budget_semantics": budget_semantics,
                "max_q_attainment_required": require_max_q_attainment,
                **feedback_counts,
                "snapshots": snapshot_budgets,
                "cap_attainment": {
                    str(budget): observed >= budget for budget in q_levels
                },
                "realized_feedback_by_cap": {
                    str(budget): (
                        int(donor_snapshots[budget]["eligible_feedbacks"])
                        if budget in donor_snapshots
                        else None
                    )
                    for budget in q_levels
                },
                "invalid_caps": invalid_caps,
                "eligible_for_prior": set(q_levels).issubset(snapshot_budgets)
                and not invalid_caps
                and (not require_max_q_attainment or observed >= max_q),
            }

        failures: list[str] = []
        if not checks["session"]["valid_for_completion"]:
            failures.append("session_not_completable")
        if elapsed_ms < required_session_ms:
            failures.append("minimum_session_duration")
        # The phone is mandatory during the timed interaction, but the final
        # questionnaire is completed on the study computer after interaction
        # has ended.  Closing the phone while answering must not invalidate an
        # otherwise complete session.  Pair overlap and identity checks below
        # continue to enforce that a qualified phone participated.
        if pair_overlap_ms < required_pair_ms:
            failures.append("minimum_pair_overlap")
        if (
            requires_locked_pipeline(participant)
            and checks["pair"]["qualified_interval_count"] < 1
        ):
            failures.append("qualified_pair_interval")
        if (
            requires_locked_pipeline(participant)
            and checks["pair"]["qualified_pair_count"] != 1
        ):
            failures.append("qualified_pair_identity")
        if not all(quality_status.values()):
            failures.append("quality_checks")
        if not checks["topic"]["selected"]:
            failures.append("topic_not_selected")
        if checks["topic"]["primary_chat_turns"] < minimum_turns:
            failures.append("insufficient_primary_chat_turns")
        browser_fallback_allowed = bool(
            study.get("tts_gate", {}).get("browser_fallback_allowed", False)
        )
        if (
            checks["tts"]["server_only_required"]
            and checks["tts"]["browser_fallbacks"]
            and not browser_fallback_allowed
        ):
            failures.append("server_tts_integrity")
        if (
            checks["tts"]["tts_observations"]
            < checks["tts"]["required_server_generations"]
        ):
            failures.append("server_tts_not_observed")
        if not required_responses.issubset(response_names):
            failures.append("questionnaires")
        if not checks["phases"]["valid"]:
            failures.append("phase_sequence_or_duration")
        if not checks["timing_stream"]["model_ready"]:
            failures.append("timing_stream_not_ready")
        if len(valid_predictions) < int(rules["min_model_predictions"]):
            failures.append("insufficient_model_predictions")
        if requires_locked_pipeline(participant) and policy_mismatches:
            failures.append("timing_policy_fallback_or_mismatch")
        if requires_locked_pipeline(participant) and not receipt_complete:
            failures.append("tap_receipt_mismatch")
        if forbidden_updates:
            failures.append("forbidden_model_update")
        if not checks["frozen_model_identity"]["matches_locked_prior"]:
            failures.append("frozen_model_hash_changed")
        if cross_phase_activations:
            failures.append("activation_crossed_phase_boundary")
        checks["failures"] = failures
        checks["complete"] = not failures
        return checks

    def _phase_report(
        self,
        participant: dict[str, Any],
        day: int,
        events: list[dict[str, Any]],
        rules: dict[str, Any],
    ) -> dict[str, Any]:
        block_schedule = rules.get("block_schedule", {})
        if block_schedule.get("enabled"):
            return self._block_phase_report(participant, events, rules)
        relevant = [
            row
            for row in events
            if row["event_type"]
            in {"phase.start", "phase.end", "companion.window_start", "companion.window_end"}
        ]
        order = json.loads(participant["probe_schedule"])[day - 1]
        policy_map = {
            "A": "current",
            "B": "base" if participant.get("cohort") == "study_a" else "matched_other",
            "C": "vad",
        }
        policies = [policy_map[letter] for letter in order]
        candidate_assignments = (
            self.db.prior_probe_assignments(
                participant_id=participant["id"], day=day
            )
            if day == 1
            else []
        )
        candidate_tokens = [
            str(row["probe_token"]) for row in candidate_assignments
        ]

        expected: list[tuple[str, Callable[[dict[str, Any]], bool], str]] = []

        def phase_match(
            phase: str, policy_token: str | None = None
        ) -> Callable[[dict[str, Any]], bool]:
            return lambda payload: payload.get("phase") == phase and (
                policy_token is None or payload.get("policy_token") == policy_token
            )

        def window_match(expected_window: int) -> Callable[[dict[str, Any]], bool]:
            return lambda payload: payload.get("window") == expected_window

        if candidate_tokens:
            for index, probe_token in enumerate(candidate_tokens, start=1):
                expected.extend(
                    [
                        (
                            "phase.start",
                            phase_match("candidate_probe", probe_token),
                            f"candidate.{index}.start",
                        ),
                        (
                            "phase.end",
                            phase_match("candidate_probe", probe_token),
                            f"candidate.{index}.end",
                        ),
                    ]
                )
        else:
            expected.extend(
                [
                    ("phase.start", phase_match("frozen", "A"), "frozen.start"),
                    ("phase.end", phase_match("frozen"), "frozen.end"),
                ]
            )
        for index in range(3):
            expected.extend(
                [
                    ("phase.start", phase_match("learning", "A"), f"learning.{index + 1}.start"),
                    ("phase.end", phase_match("learning"), f"learning.{index + 1}.end"),
                    (
                        "companion.window_start",
                        window_match(index + 1),
                        f"companion.{index + 1}.start",
                    ),
                    (
                        "companion.window_end",
                        window_match(index + 1),
                        f"companion.{index + 1}.end",
                    ),
                ]
            )
        expected.extend(
            [
                ("phase.start", phase_match("learning", "A"), "learning.4.start"),
                ("phase.end", phase_match("learning"), "learning.4.end"),
            ]
        )
        for index, policy_token in enumerate(order, start=1):
            expected.extend(
                [
                    (
                        "phase.start",
                        phase_match("micro_probe", policy_token),
                        f"micro.{index}.start",
                    ),
                    (
                        "phase.end",
                        phase_match("micro_probe", policy_token),
                        f"micro.{index}.end",
                    ),
                ]
            )

        actual_labels: list[str] = []
        sequence_valid = len(relevant) == len(expected)
        if sequence_valid:
            for row, (event_type, predicate, label) in zip(relevant, expected):
                if row["event_type"] != event_type or not predicate(row["payload"]):
                    sequence_valid = False
                    break
                actual_labels.append(label)

        duration_failures: list[str] = []
        if sequence_valid and participant["protocol_mode"] == "strict":
            phase_pairs = [(relevant[index], relevant[index + 1]) for index in range(0, len(relevant), 2)]
            initial_planned = (
                [float(rules["candidate_probe_seconds"])] * len(candidate_tokens)
                if candidate_tokens
                else [float(rules["frozen"]["min_seconds"])]
            )
            initial_labels = (
                [f"candidate.{index}" for index in range(1, len(candidate_tokens) + 1)]
                if candidate_tokens
                else ["frozen"]
            )
            planned = [
                *initial_planned,
                float(rules["learning_seconds"][0]),
                float(rules["companion_seconds"][0]),
                float(rules["learning_seconds"][1]),
                float(rules["companion_seconds"][1]),
                float(rules["learning_seconds"][2]),
                float(rules["companion_seconds"][2]),
                float(rules["learning_seconds"][3]),
                *[float(value) for value in rules["micro_probe_seconds"]],
            ]
            labels = [
                *initial_labels,
                "learning.1",
                "companion.1",
                "learning.2",
                "companion.2",
                "learning.3",
                "companion.3",
                "learning.4",
                "micro.1",
                "micro.2",
                "micro.3",
            ]
            for (start, end), seconds, label in zip(phase_pairs, planned, labels):
                server_seconds = (int(end["server_ms"]) - int(start["server_ms"])) / 1000
                reported = end["payload"].get(
                    "elapsed_seconds", end["payload"].get("completed_seconds", 0)
                )
                if server_seconds < seconds * float(rules["min_phase_server_ratio"]):
                    duration_failures.append(f"{label}.server_time")
                if float(reported or 0) < seconds * float(rules["min_phase_reported_ratio"]):
                    duration_failures.append(f"{label}.reported_time")
            if not candidate_tokens:
                frozen_end = phase_pairs[0][1]["payload"]
                opportunities = int(frozen_end.get("opportunities", 0))
                frozen_elapsed = float(frozen_end.get("elapsed_seconds", 0))
                if (
                    opportunities < int(rules["frozen"]["target_opportunities"])
                    and frozen_elapsed
                    < float(rules["frozen"]["max_seconds"])
                    * float(rules["min_phase_reported_ratio"])
                ):
                    duration_failures.append("frozen.opportunities")

        return {
            "valid": sequence_valid and not duration_failures,
            "expected_milestones": len(expected),
            "observed_milestones": len(relevant),
            "matched": actual_labels,
            "duration_failures": duration_failures,
            "probe_order": policies,
            "candidate_probe_count": len(candidate_tokens),
        }

    def _block_phase_report(
        self,
        participant: dict[str, Any],
        events: list[dict[str, Any]],
        rules: dict[str, Any],
    ) -> dict[str, Any]:
        schedule = rules["block_schedule"]
        blocks = int(schedule["active_blocks"])
        active_phase = (
            "frozen"
            if str(participant.get("group_name") or "").strip().lower()
            in {"frozen", "f", "holdout", "holdout_evaluator"}
            else "learning"
        )
        relevant_types = {
            "phase.start",
            "phase.end",
            "companion.window_start",
            "companion.window_end",
            "rest.start",
            "rest.end",
        }
        recovery_events = [
            row
            for row in events
            if row["event_type"] in {"session.restart_required", "session.interrupted"}
        ]
        all_relevant = [row for row in events if row["event_type"] in relevant_types]
        resume = participant.get("operator_resume") if isinstance(participant, dict) else None
        if resume is None:
            settings = getattr(getattr(self, "db", None), "settings", None)
            data_dir = getattr(settings, "data_dir", None)
            resume = load_operator_resume(
                data_dir, str((participant or {}).get("id") or "")
            )
        if resume:
            all_relevant = [
                row for row in all_relevant if keep_milestone_event(resume, row)
            ]
        expected: list[tuple[str, int, str]] = []
        include_final_rest = bool(schedule.get("rest_after_final_block"))
        for block in range(1, blocks + 1):
            expected.extend(
                [
                    ("phase.start", block, active_phase),
                    ("phase.end", block, active_phase),
                    ("companion.window_start", block, "companion"),
                    ("companion.window_end", block, "companion"),
                ]
            )
            if block < blocks or include_final_rest:
                expected.extend(
                    [
                        ("rest.start", block, "rest"),
                        ("rest.end", block, "rest"),
                    ]
                )

        def milestone_sequence_matches(candidate: list[dict[str, Any]]) -> bool:
            if len(candidate) != len(expected):
                return False
            for row, (event_type, block, phase) in zip(candidate, expected):
                payload = row["payload"]
                observed_block = payload.get(
                    "block", payload.get("window", payload.get("active_block_index"))
                )
                observed_phase = (
                    payload.get("phase") if event_type.startswith("phase.") else phase
                )
                if (
                    row["event_type"] != event_type
                    or int(observed_block or -1) != block
                    or observed_phase != phase
                    or (
                        event_type.startswith("rest.")
                        and (
                            payload.get("excluded_from_analysis") is not True
                            or payload.get("analysis_eligible") is not False
                        )
                    )
                    or (
                        not event_type.startswith("rest.")
                        and payload.get("analysis_eligible") is False
                    )
                ):
                    return False
            return True

        def slot_matches(row: dict[str, Any], event_type: str, block: int, phase: str) -> bool:
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            raw_block = payload.get(
                "block",
                payload.get("window", payload.get("active_block_index")),
            )
            try:
                observed_block = int(raw_block)
            except (TypeError, ValueError):
                observed_block = -1
            observed_phase = (
                payload.get("phase") if event_type.startswith("phase.") else phase
            )
            if str(row.get("event_type") or "") != event_type:
                return False
            if observed_block != block or observed_phase != phase:
                return False
            if event_type.startswith("rest."):
                return (
                    payload.get("excluded_from_analysis") is True
                    and payload.get("analysis_eligible") is False
                )
            return payload.get("analysis_eligible") is not False

        relevant = all_relevant
        # After an interrupt continue, only the unfinished stage is replayed.
        # Keep the latest start/end pair that actually met the planned
        # duration, so a later full companion window replaces a short one.
        if recovery_events:
            planned_seconds = {
                active_phase: float(schedule["primary_seconds"]),
                "companion": float(schedule["companion_seconds"]),
                "rest": float(schedule["rest_seconds"]),
            }
            server_ratio = float(rules["min_phase_server_ratio"])
            reported_ratio = float(rules["min_phase_reported_ratio"])

            def pair_meets_duration(
                start: dict[str, Any], end: dict[str, Any], phase_name: str
            ) -> bool:
                planned = planned_seconds[phase_name]
                server_seconds = (
                    int(end.get("server_ms") or 0) - int(start.get("server_ms") or 0)
                ) / 1000
                payload = end.get("payload") if isinstance(end.get("payload"), dict) else {}
                reported = payload.get(
                    "elapsed_seconds", payload.get("completed_seconds", 0)
                )
                return (
                    server_seconds >= planned * server_ratio
                    and float(reported or 0) >= planned * reported_ratio
                )

            latest_slots: list[dict[str, Any]] = []
            for index in range(0, len(expected), 2):
                start_type, start_block, start_phase = expected[index]
                end_type, end_block, end_phase = expected[index + 1]
                phase_name = (
                    start_phase
                    if start_type.startswith("phase.")
                    else ("companion" if start_type.startswith("companion.") else "rest")
                )
                starts = [
                    row
                    for row in all_relevant
                    if slot_matches(row, start_type, start_block, start_phase)
                ]
                ends = [
                    row
                    for row in all_relevant
                    if slot_matches(row, end_type, end_block, end_phase)
                ]
                ok_pairs = [
                    (start, end)
                    for start in starts
                    for end in ends
                    if int(end.get("server_ms") or 0) >= int(start.get("server_ms") or 0)
                    and pair_meets_duration(start, end, phase_name)
                ]
                if not ok_pairs:
                    latest_slots = []
                    break
                latest_slots.extend(
                    max(
                        ok_pairs,
                        key=lambda pair: (
                            int(pair[1].get("server_ms") or 0),
                            int(pair[0].get("server_ms") or 0),
                            int(pair[1].get("seq") or 0),
                            int(pair[0].get("seq") or 0),
                        ),
                    )
                )
            if latest_slots and milestone_sequence_matches(latest_slots):
                slot_times = [int(row.get("server_ms") or 0) for row in latest_slots]
                if slot_times == sorted(slot_times):
                    relevant = latest_slots
            if relevant is all_relevant and len(all_relevant) >= len(expected):
                for start in range(len(all_relevant) - len(expected), -1, -1):
                    candidate = all_relevant[start : start + len(expected)]
                    if milestone_sequence_matches(candidate):
                        relevant = candidate
                        break
        first_relevant_seq = int(relevant[0].get("seq") or 0) if relevant else 0
        prior_recovery_events = [
            row
            for row in recovery_events
            if int(row.get("seq") or 0) < first_relevant_seq
        ]
        restart_boundary_seq = (
            max(int(row.get("seq") or 0) for row in prior_recovery_events)
            if prior_recovery_events
            else None
        )

        matched: list[str] = []
        sequence_valid = milestone_sequence_matches(relevant)
        if sequence_valid:
            for row, (event_type, block, phase) in zip(relevant, expected):
                matched.append(f"block.{block}.{phase}.{event_type.rsplit('.', 1)[-1]}")

        duration_failures: list[str] = []
        if sequence_valid and participant["protocol_mode"] == "strict":
            expected_seconds = {
                active_phase: float(schedule["primary_seconds"]),
                "companion": float(schedule["companion_seconds"]),
                "rest": float(schedule["rest_seconds"]),
            }
            for index in range(0, len(relevant), 2):
                start, end = relevant[index], relevant[index + 1]
                event_prefix = start["event_type"].split(".", 1)[0]
                phase = (
                    str(start["payload"].get("phase"))
                    if event_prefix == "phase"
                    else ("companion" if event_prefix == "companion" else "rest")
                )
                block = int(
                    start["payload"].get(
                        "block",
                        start["payload"].get(
                            "window",
                            start["payload"].get("active_block_index", -1),
                        ),
                    )
                )
                planned = expected_seconds[phase]
                server_seconds = (
                    int(end["server_ms"]) - int(start["server_ms"])
                ) / 1000
                reported = end["payload"].get(
                    "elapsed_seconds",
                    end["payload"].get("completed_seconds", 0),
                )
                label = f"block.{block}.{phase}"
                if server_seconds < planned * float(rules["min_phase_server_ratio"]):
                    duration_failures.append(f"{label}.server_time")
                if float(reported or 0) < planned * float(
                    rules["min_phase_reported_ratio"]
                ):
                    duration_failures.append(f"{label}.reported_time")

        return {
            "valid": sequence_valid and not duration_failures,
            "expected_milestones": len(expected),
            "observed_milestones": len(relevant),
            "discarded_pre_restart_milestones": len(all_relevant) - len(relevant),
            "restart_boundary_seq": restart_boundary_seq,
            "matched": matched,
            "duration_failures": duration_failures,
            "probe_order": [],
            "candidate_probe_count": 0,
            "active_blocks": blocks,
            "rest_seconds": int(schedule["rest_seconds"]),
            "rest_excluded_from_analysis": True,
        }
