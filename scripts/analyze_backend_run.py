"""Read-only aggregate audit for one study run.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Iterable


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not data:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(data),
        "min": round(min(data), 4),
        "median": round(statistics.median(data), 4),
        "mean": round(statistics.mean(data), 4),
        "max": round(max(data), 4),
    }


def nested_counter(counter: collections.Counter[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        values = key if isinstance(key, tuple) else (key,)
        rows.append({"key": list(values), "count": count})
    return rows


def participant_number(assignment_id: str | None) -> int | None:
    match = re.fullmatch(r"A(\d+)", assignment_id or "")
    return int(match.group(1)) if match else None


def verify_run_event_chains(
    connection: sqlite3.Connection, run_id: str, participant_ids: list[str]
) -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    for participant_id in participant_ids:
        previous = "0" * 64
        expected_seq = 1
        rows = connection.execute(
            "SELECT * FROM events WHERE run_id=? AND participant_id=? ORDER BY seq",
            (run_id, participant_id),
        ).fetchall()
        for row in rows:
            material = "|".join(
                [
                    previous,
                    row["participant_id"],
                    str(row["session_id"] or ""),
                    str(row["seq"]),
                    row["event_type"],
                    str(row["server_ms"]),
                    row["payload"],
                ]
            )
            digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if row["seq"] != expected_seq:
                errors.append(
                    f"{participant_id}: expected seq {expected_seq}, got {row['seq']}"
                )
            if row["prev_hash"] != previous:
                errors.append(f"{participant_id} seq {row['seq']}: previous hash mismatch")
            if row["event_hash"] != digest:
                errors.append(f"{participant_id} seq {row['seq']}: event hash mismatch")
            previous = row["event_hash"]
            expected_seq += 1
            checked += 1
    return {"checked": checked, "errors": errors, "passed": not errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only aggregate audit for one study run.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--consent-database", type=Path)
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget-kind",
        choices=("auto", "activation_opportunity", "eligible_feedback"),
        default="auto",
    )
    parser.add_argument(
        "--q-levels",
        default="",
        help="Comma-separated locked q levels. Defaults to 2,4,8 for v6 and 8,16,32 otherwise.",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    run = connection.execute(
        "SELECT * FROM study_runs WHERE slug=?", (args.run_slug,)
    ).fetchone()
    if run is None:
        raise SystemExit(f"Run not found: {args.run_slug}")
    run_id = str(run["id"])
    protocol_snapshot = json.loads(str(run["protocol_json"] or "{}"))
    pipeline_rules = protocol_snapshot.get("pipeline", {})
    protocol_budget_kind = str(
        pipeline_rules.get("budget_kind") or "activation_opportunity"
    )
    budget_kind = (
        protocol_budget_kind
    ) if args.budget_kind == "auto" else args.budget_kind
    budget_semantics = str(
        pipeline_rules.get("budget_semantics") or "exact_count"
    )
    protocol_q_levels = pipeline_rules.get(
        "q_feedback_levels"
        if budget_kind == "eligible_feedback"
        else "q_opportunity_levels"
    )
    q_levels = (
        [int(value) for value in args.q_levels.split(",") if value.strip()]
        if args.q_levels
        else [
            int(value)
            for value in (
                protocol_q_levels
                or ([2, 4, 8] if budget_kind == "eligible_feedback" else [8, 16, 32])
            )
        ]
    )
    if not q_levels or any(value <= 0 for value in q_levels):
        raise SystemExit("q levels must be positive integers")

    all_participants = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM participants WHERE run_id=?", (run_id,)
        )
    ]
    sessions = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM sessions WHERE run_id=?", (run_id,)
        )
    ]
    active_ids = sorted({str(row["participant_id"]) for row in sessions})
    participants = [row for row in all_participants if row["id"] in active_ids]
    participant_by_id = {str(row["id"]): row for row in all_participants}
    session_by_id = {str(row["id"]): row for row in sessions}
    candidate_probe_participants = {
        str(row["participant_id"])
        for row in connection.execute(
            "SELECT DISTINCT participant_id FROM prior_probe_assignments WHERE run_id=?",
            (run_id,),
        )
    }

    session_counts: collections.Counter[Any] = collections.Counter()
    session_durations: dict[tuple[int, str], list[float]] = collections.defaultdict(list)
    study_sessions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sessions:
        session_counts[(row["day"], row["client_role"], row["state"])] += 1
        duration_start = row.get("activated_ms") or row.get("started_ms")
        if duration_start is not None and row["ended_ms"] is not None:
            session_durations[(row["day"], row["client_role"])].append(
                (row["ended_ms"] - duration_start) / 1000.0
            )
        if row["client_role"] == "study":
            study_sessions[(str(row["participant_id"]), int(row["day"]))] = row

    cross_day_gaps: dict[str, list[float]] = {"day1_to_day2": [], "day2_to_day3": []}
    for participant_id in active_ids:
        for day, name in ((1, "day1_to_day2"), (2, "day2_to_day3")):
            before = study_sessions.get((participant_id, day))
            after = study_sessions.get((participant_id, day + 1))
            if before and after and before["ended_ms"] is not None:
                cross_day_gaps[name].append(
                    (after["started_ms"] - before["ended_ms"]) / 1000.0
                )

    pair_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM pair_intervals WHERE run_id=?", (run_id,)
        )
    ]
    pair_counts: collections.Counter[int] = collections.Counter()
    pair_durations: dict[int, list[float]] = collections.defaultdict(list)
    qualified_pair_counts: collections.Counter[int] = collections.Counter()
    qualified_pair_durations: dict[int, list[float]] = collections.defaultdict(list)
    pair_attempts_by_participant_day: collections.Counter[tuple[str, int]] = (
        collections.Counter()
    )
    pair_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(pair_intervals)")
    }
    for row in pair_rows:
        day = int(row["day"])
        pair_counts[day] += 1
        pair_attempts_by_participant_day[(str(row["participant_id"]), day)] += 1
        qualified = bool(row.get("qualified")) if "qualified" in pair_columns else False
        qualified_pair_counts[day] += int(qualified)
        if row["ended_ms"] is not None:
            activated_ms = study_sessions.get(
                (str(row["participant_id"]), day), {}
            ).get("activated_ms")
            start_ms = max(
                int(row["started_ms"]), int(activated_ms or row["started_ms"])
            )
            duration = max(0.0, (row["ended_ms"] - start_ms) / 1000.0)
            pair_durations[day].append(duration)
            if qualified:
                qualified_pair_durations[day].append(duration)

    event_type_counts: collections.Counter[str] = collections.Counter()
    coverage: dict[tuple[str, int], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    action_metrics: dict[
        tuple[int, str | None, str | None], collections.Counter[str]
    ] = collections.defaultdict(collections.Counter)
    learning_by_wave_day: dict[tuple[int, int], collections.Counter[str]] = (
        collections.defaultdict(collections.Counter)
    )
    opportunities: collections.Counter[str] = collections.Counter()
    eligible_labels: collections.Counter[tuple[str, int]] = collections.Counter()
    tap_pointers: dict[str, dict[str, Any]] = {}
    tap_results: dict[str, dict[str, Any]] = {}
    quarantine_reasons: collections.Counter[str] = collections.Counter()
    prediction_routes: collections.Counter[tuple[str | None, str | None]] = (
        collections.Counter()
    )
    prediction_skipped = 0
    prediction_probabilities: dict[tuple[int, str | None, str | None], list[float]] = (
        collections.defaultdict(list)
    )
    phase_end_seconds: dict[str, list[float]] = collections.defaultdict(list)
    phase_start_counts: collections.Counter[tuple[str | None, str | None]] = (
        collections.Counter()
    )
    losses: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    shadow_losses: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    active_activation: dict[tuple[str, str], dict[str, Any]] = {}
    tap_activation: dict[str, dict[str, Any]] = {}
    cross_phase_activations: list[dict[str, Any]] = []
    eligible_provenance_errors: list[dict[str, Any]] = []

    event_rows = connection.execute(
        "SELECT * FROM events WHERE run_id=? ORDER BY server_ms, id", (run_id,)
    )
    for event in event_rows:
        event_type = str(event["event_type"])
        event_type_counts[event_type] += 1
        participant_id = str(event["participant_id"])
        session = session_by_id.get(str(event["session_id"]))
        day = int(session["day"]) if session else 0
        if day:
            coverage[(participant_id, day)][event_type] += 1
        payload = json.loads(event["payload"])
        participant = participant_by_id[participant_id]
        wave = int(participant["enrollment_wave"] or 0)
        activation_key = (participant_id, str(event["session_id"] or ""))

        if event_type in {"agent.auto_activate", "tap.manual_activate"}:
            payload.setdefault(
                "activation_id",
                f"legacy-{participant_id}-{event['id']}",
            )
            payload.setdefault("activation_phase", payload.get("phase"))
            payload.setdefault("activation_policy", payload.get("policy"))
            payload.setdefault(
                "activation_source",
                "model" if event_type == "agent.auto_activate" else "manual",
            )
            active_activation[activation_key] = {
                key: payload.get(key)
                for key in (
                    "activation_id",
                    "activation_phase",
                    "activation_policy",
                    "activation_source",
                    "activation_policy_effective",
                )
            }
        elif event_type == "phase.transition_ack" and payload.get(
            "drained_agent_state"
        ) not in {None, "waiting"}:
            active_activation.pop(activation_key, None)
        elif event_type == "tap.interrupt":
            origin = active_activation.get(activation_key, {})
            for key, value in origin.items():
                payload.setdefault(key, value)
            payload.setdefault("tap_phase", payload.get("phase"))
            payload.setdefault("tap_policy", payload.get("policy"))
            if payload.get("activation_phase") is not None:
                payload["phase"] = payload.get("activation_phase")
            if payload.get("activation_policy") is not None:
                payload["policy"] = payload.get("activation_policy")
            tap_id = str(payload.get("tap_id") or "")
            if tap_id:
                tap_activation[tap_id] = dict(origin)
            if (
                payload.get("activation_phase")
                and payload.get("tap_phase")
                and payload.get("activation_phase") != payload.get("tap_phase")
            ):
                cross_phase_activations.append(
                    {
                        "assignment_id": participant.get("assignment_id"),
                        "day": day,
                        "tap_id": tap_id or None,
                        "activation_phase": payload.get("activation_phase"),
                        "tap_phase": payload.get("tap_phase"),
                    }
                )
            active_activation.pop(activation_key, None)
        elif event_type == "tap.result":
            origin = tap_activation.get(str(payload.get("tap_id") or ""), {})
            for key, value in origin.items():
                payload.setdefault(key, value)
            if payload.get("activation_phase") is not None:
                payload["phase"] = payload.get("activation_phase")
            if payload.get("activation_policy") is not None:
                payload["policy"] = payload.get("activation_policy")

        if event_type == "tap.pointer":
            tap_pointers[str(payload.get("tap_id"))] = {
                "participant_id": participant_id,
                "day": day,
                **payload,
            }
        elif event_type == "tap.result":
            tap_results[str(payload.get("tap_id"))] = {
                "participant_id": participant_id,
                "day": day,
                **payload,
            }
            if payload.get("quarantine_reason"):
                quarantine_reasons[str(payload["quarantine_reason"])] += 1
        elif event_type == "model.prediction":
            prediction_skipped += int(bool(payload.get("skipped")))
            prediction_routes[(payload.get("policy"), payload.get("policy_effective"))] += 1
            probability = payload.get("probability")
            if isinstance(probability, (int, float)):
                prediction_probabilities[
                    (day, payload.get("phase"), payload.get("policy"))
                ].append(float(probability))
        elif event_type == "phase.start":
            phase_start_counts[(payload.get("phase"), payload.get("policy_token"))] += 1
        elif event_type == "phase.end":
            value = payload.get("elapsed_seconds", payload.get("completed_seconds"))
            if isinstance(value, (int, float)):
                phase_end_seconds[str(payload.get("phase"))].append(float(value))

        if event_type in {"agent.auto_activate", "tap.manual_activate", "tap.interrupt"}:
            key = (
                day,
                payload.get("activation_phase", payload.get("phase")),
                payload.get("activation_policy", payload.get("policy")),
            )
            metrics = action_metrics[key]
            if event_type.startswith("tap.") and payload.get("training_eligible"):
                if (
                    not payload.get("activation_id")
                    or payload.get("activation_phase") != "learning"
                    or payload.get("activation_policy") != "current"
                ):
                    eligible_provenance_errors.append(
                        {
                            "assignment_id": participant.get("assignment_id"),
                            "day": day,
                            "event_type": event_type,
                            "activation_id": payload.get("activation_id"),
                            "activation_phase": payload.get("activation_phase"),
                            "activation_policy": payload.get("activation_policy"),
                        }
                    )
                label = int(payload["label"])
                eligible_labels[(participant_id, label)] += 1
                if payload.get("loss") is not None:
                    losses[(day, label)].append(float(payload["loss"]))
                if payload.get("shadow_loss") is not None:
                    shadow_losses[(day, label)].append(float(payload["shadow_loss"]))
            if event_type == "agent.auto_activate":
                metrics["auto"] += 1
                if payload.get("phase") == "learning":
                    opportunities[participant_id] += 1
                    learning_by_wave_day[(wave, day)]["auto"] += 1
            elif event_type == "tap.manual_activate":
                metrics["manual"] += 1
                if payload.get("training_eligible"):
                    metrics["eligible_positive"] += 1
                if payload.get("phase") == "learning":
                    opportunities[participant_id] += 1
                    learning_by_wave_day[(wave, day)]["manual"] += 1
                    learning_by_wave_day[(wave, day)]["eligible_positive"] += int(
                        bool(payload.get("training_eligible"))
                    )
            else:
                metrics["interrupt"] += 1
                manual_conflict = (
                    payload.get("quarantine_reason") == "manual_activate_interrupt_conflict"
                )
                metrics[
                    "manual_conflict_interrupt" if manual_conflict else "auto_source_interrupt"
                ] += 1
                if payload.get("training_eligible"):
                    metrics["eligible_negative"] += 1
                if payload.get("phase") == "learning":
                    learning_by_wave_day[(wave, day)]["interrupt"] += 1
                    learning_by_wave_day[(wave, day)]["auto_source_interrupt"] += int(
                        not manual_conflict
                    )
                    learning_by_wave_day[(wave, day)]["eligible_negative"] += int(
                        bool(payload.get("training_eligible"))
                    )

    action_rows: list[dict[str, Any]] = []
    for (day, phase, policy), metrics in sorted(
        action_metrics.items(), key=lambda item: str(item[0])
    ):
        row = {"day": day, "phase": phase, "policy": policy, **dict(metrics)}
        turns = metrics["auto"] + metrics["manual"]
        row["turns"] = turns
        row["manual_share"] = round(metrics["manual"] / turns, 4) if turns else None
        row["auto_interrupt_rate"] = (
            round(metrics["auto_source_interrupt"] / metrics["auto"], 4)
            if metrics["auto"]
            else None
        )
        row["correction_events_per_turn"] = (
            round((metrics["manual"] + metrics["auto_source_interrupt"]) / turns, 4)
            if turns
            else None
        )
        action_rows.append(row)

    learning_rows: list[dict[str, Any]] = []
    for (wave, day), metrics in sorted(learning_by_wave_day.items()):
        turns = metrics["auto"] + metrics["manual"]
        row = {"wave": wave, "day": day, **dict(metrics), "turns": turns}
        row["manual_share"] = round(metrics["manual"] / turns, 4) if turns else None
        row["auto_interrupt_rate"] = (
            round(metrics["auto_source_interrupt"] / metrics["auto"], 4)
            if metrics["auto"]
            else None
        )
        row["correction_events_per_turn"] = (
            round((metrics["manual"] + metrics["auto_source_interrupt"]) / turns, 4)
            if turns
            else None
        )
        learning_rows.append(row)

    expected_coverage = {
        "session.client_started": 1,
        "pair.qualified_start": 1,
        "model.ready": 1,
        "topic.selected": 1,
        "phase.start": 8,
        "phase.end": 8,
        "companion.window_start": 3,
        "companion.window_end": 3,
    }
    coverage_anomalies: list[dict[str, Any]] = []
    for participant_id in active_ids:
        assignment_id = participant_by_id[participant_id]["assignment_id"]
        for day in (1, 2, 3):
            counts = coverage[(participant_id, day)]
            for event_type, expected in expected_coverage.items():
                if (
                    event_type in {"phase.start", "phase.end"}
                    and day == 1
                    and participant_id in candidate_probe_participants
                ):
                    expected = 11
                actual = counts[event_type]
                if actual != expected:
                    coverage_anomalies.append(
                        {
                            "assignment_id": assignment_id,
                            "day": day,
                            "event_type": event_type,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
            if (
                budget_semantics == "per_donor_cap"
                and counts["phase.transition_ack"] < 9
            ):
                coverage_anomalies.append(
                    {
                        "assignment_id": assignment_id,
                        "day": day,
                        "event_type": "phase.transition_ack",
                        "expected_minimum": 9,
                        "actual": counts["phase.transition_ack"],
                    }
                )

    quality_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM quality_checks WHERE run_id=?", (run_id,)
        )
    ]
    quality_counts: collections.Counter[tuple[str, int]] = collections.Counter()
    quality_per_gate: dict[
        tuple[str, int, str], list[dict[str, Any]]
    ] = collections.defaultdict(list)
    network_medians: list[float] = []
    network_p95s: list[float] = []
    for row in quality_rows:
        quality_counts[(str(row["kind"]), int(row["passed"]))] += 1
        session = session_by_id.get(str(row["session_id"]))
        day = int(session["day"]) if session else 0
        quality_per_gate[
            (str(row["participant_id"]), day, str(row["kind"]))
        ].append(row)
        if row["kind"] == "network":
            metrics = json.loads(row["metrics"])
            for key, target in (("median_ms", network_medians), ("p95_ms", network_p95s)):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    target.append(float(value))
    quality_anomalies: list[dict[str, Any]] = []
    quality_attempt_counts: collections.Counter[tuple[str, int]] = collections.Counter()
    max_quality_attempts = int(
        protocol_snapshot.get("quality_gate", {}).get(
            "max_attempts_per_kind_per_day", 2
        )
    )
    for (participant_id, day, kind), attempts in quality_per_gate.items():
        attempts.sort(key=lambda row: (int(row["created_ms"]), int(row["id"])))
        passed = bool(attempts[-1]["passed"])
        quality_attempt_counts[(kind, len(attempts))] += 1
        if len(attempts) > max_quality_attempts or not passed:
            quality_anomalies.append(
                {
                    "assignment_id": participant_by_id[participant_id]["assignment_id"],
                    "day": day,
                    "kind": kind,
                    "final_passed": passed,
                    "attempt_count": len(attempts),
                    "maximum_attempts": max_quality_attempts,
                }
            )

    response_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM responses WHERE run_id=?", (run_id,)
        )
    ]
    response_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    answers: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in response_rows:
        response_counts[(str(row["instrument"]), str(row["version"]))] += 1
        answers[str(row["participant_id"])][str(row["instrument"])] = json.loads(
            row["answers"]
        )
    daily_questionnaire_metrics: dict[str, Any] = {}
    daily_keys = [
        "timing_today",
        "like_today",
        "annoyance_today",
        "predictable_today",
        "control_today",
        "effort_today",
        "fatigue_today",
        "speed_today",
    ]
    for key in daily_keys:
        by_day: list[list[float]] = []
        changed = 0
        for day in (1, 2, 3):
            by_day.append(
                [
                    float(value[f"daily_day_{day}"][key])
                    for value in answers.values()
                    if f"daily_day_{day}" in value
                ]
            )
        for value in answers.values():
            observed = [
                value[f"daily_day_{day}"][key]
                for day in (1, 2, 3)
                if f"daily_day_{day}" in value
            ]
            changed += int(len(observed) == 3 and len(set(observed)) > 1)
        daily_questionnaire_metrics[key] = {
            "day_means": [
                round(statistics.mean(values), 4) if values else None
                for values in by_day
            ],
            "participants_with_within_person_change": changed,
        }
    ended_early = sum(
        int(bool(value.get(f"daily_day_{day}", {}).get("ended_early")))
        for value in answers.values()
        for day in (1, 2, 3)
    )

    counter_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(model_counters)")
    }
    counters: dict[str, dict[str, int]] = {}
    for raw_row in connection.execute(
        "SELECT * FROM model_counters WHERE run_id=?", (run_id,)
    ):
        row = dict(raw_row)
        learning = int(row.get("learning_opportunities") or 0)
        counters[str(row["participant_id"])] = {
            "learning_opportunities": learning,
            "activation_opportunities": int(
                row.get("activation_opportunities", learning) or 0
            ),
            "eligible_feedbacks": int(row.get("eligible_feedbacks", 0) or 0),
            "positive_feedbacks": int(row.get("positive_feedbacks", 0) or 0),
            "negative_feedbacks": int(row.get("negative_feedbacks", 0) or 0),
        }
    snapshots_by_participant: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM model_snapshots WHERE run_id=? ORDER BY opportunity_budget",
        (run_id,),
    ):
        snapshot = dict(row)
        snapshot_kind = str(snapshot.get("budget_kind") or "activation_opportunity")
        if snapshot_kind == budget_kind:
            snapshots_by_participant[str(row["participant_id"])].append(snapshot)

    participant_model_rows: list[dict[str, Any]] = []
    native_q_coverage = {budget: 0 for budget in q_levels}
    stored_q_coverage = {budget: 0 for budget in q_levels}
    valid_capped_q_coverage = {budget: 0 for budget in q_levels}
    realized_feedback_by_cap: dict[int, list[float]] = {
        budget: [] for budget in q_levels
    }
    for participant in sorted(
        participants,
        key=lambda row: participant_number(row["assignment_id"]) or 10_000,
    ):
        participant_id = str(participant["id"])
        event_activation_count = int(opportunities[participant_id])
        snapshot_rows = snapshots_by_participant[participant_id]
        budgets = sorted(int(row["opportunity_budget"]) for row in snapshot_rows)
        snapshots_by_budget = {
            int(row["opportunity_budget"]): row for row in snapshot_rows
        }
        hashes = {
            f"{row['encoder_sha256']}:{row['main_sha256']}" for row in snapshot_rows
        }
        positive = int(eligible_labels[(participant_id, 1)])
        negative = int(eligible_labels[(participant_id, 0)])
        event_feedback_count = positive + negative
        native_budget_count = (
            event_feedback_count
            if budget_kind == "eligible_feedback"
            else event_activation_count
        )
        stored_counts = counters.get(
            participant_id,
            {
                "learning_opportunities": 0,
                "activation_opportunities": 0,
                "eligible_feedbacks": 0,
                "positive_feedbacks": 0,
                "negative_feedbacks": 0,
            },
        )
        for budget in native_q_coverage:
            native_q_coverage[budget] += int(native_budget_count >= budget)
            stored_q_coverage[budget] += int(budget in budgets)
            snapshot = snapshots_by_budget.get(budget)
            if snapshot is not None:
                realized = int(snapshot.get("eligible_feedbacks") or 0)
                realized_feedback_by_cap[budget].append(float(realized))
                valid_capped_q_coverage[budget] += int(
                    budget_semantics != "per_donor_cap"
                    or realized == min(native_budget_count, budget)
                )
        participant_model_rows.append(
            {
                "assignment_id": participant["assignment_id"],
                "wave": participant["enrollment_wave"],
                "prior_n": participant["prior_n"],
                "prior_q": participant["prior_q"],
                "budget_kind": budget_kind,
                "event_derived_activation_opportunities": event_activation_count,
                "event_derived_eligible_feedbacks": event_feedback_count,
                "native_budget_count": native_budget_count,
                "stored_counters": stored_counts,
                "counter_event_feedback_delta": stored_counts["eligible_feedbacks"]
                - event_feedback_count,
                "eligible_positive": positive,
                "eligible_negative": negative,
                "stored_snapshot_budgets": budgets,
                "realized_feedback_by_cap": {
                    str(budget): (
                        int(snapshots_by_budget[budget].get("eligible_feedbacks") or 0)
                        if budget in snapshots_by_budget
                        else None
                    )
                    for budget in q_levels
                },
                "distinct_snapshot_models": len(hashes),
            }
        )

    shared_priors = [
        {
            "id": row["id"],
            "n": row["donor_count"],
            "q": row["opportunity_budget"],
            "status": row["status"],
            "algorithm": row["algorithm"],
            "model_hash": f"{row['encoder_sha256']}:{row['main_sha256']}",
        }
        for row in connection.execute(
            "SELECT * FROM shared_priors WHERE run_id=? ORDER BY donor_count, opportunity_budget",
            (run_id,),
        )
    ]
    prior_assignments = [
        {
            "assignment_id": row["assignment_id"],
            "wave": row["enrollment_wave"],
            "prior_n": row["prior_n"],
            "prior_q": row["prior_q"],
            "release_state": row["release_state"],
        }
        for row in sorted(
            all_participants,
            key=lambda value: participant_number(value["assignment_id"]) or 10_000,
        )
        if participant_number(row["assignment_id"]) is not None
    ]

    missing_results = [
        {"tap_id": tap_id, **pointer}
        for tap_id, pointer in tap_pointers.items()
        if tap_id not in tap_results
    ]
    accepted_missing = [row for row in missing_results if row.get("accepted")]
    results_without_pointer = sorted(set(tap_results) - set(tap_pointers))

    prediction_route_rows = [
        {"policy": key[0], "policy_effective": key[1], "count": count}
        for key, count in sorted(prediction_routes.items(), key=lambda item: str(item[0]))
    ]
    fallback_predictions = sum(
        count
        for (policy, effective), count in prediction_routes.items()
        if policy != effective
    )

    pipeline_actions = collections.Counter(
        str(row["action"])
        for row in connection.execute(
            "SELECT action FROM pipeline_actions WHERE run_id=?", (run_id,)
        )
    )

    consent = {"count": None, "versions": []}
    if args.consent_database:
        consent_connection = sqlite3.connect(
            f"file:{args.consent_database.resolve()}?mode=ro", uri=True
        )
        consent_connection.row_factory = sqlite3.Row
        consent["count"] = consent_connection.execute(
            "SELECT COUNT(*) FROM consents WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        consent["versions"] = [
            dict(row)
            for row in consent_connection.execute(
                """SELECT version, participation, raw_audio, future_sharing,
                          COUNT(*) AS count, COUNT(DISTINCT document_hash) AS document_count
                   FROM consents WHERE run_id=?
                   GROUP BY version, participation, raw_audio, future_sharing""",
                (run_id,),
            )
        ]
        consent_connection.close()

    chain = verify_run_event_chains(connection, run_id, active_ids)
    report = {
        "run": dict(run),
        "scope": {
            "roster_participants": len(all_participants),
            "active_participants": len(participants),
            "active_assignment_ids": [
                row["assignment_id"]
                for row in sorted(
                    participants,
                    key=lambda value: participant_number(value["assignment_id"]) or 10_000,
                )
            ],
        },
        "integrity": {
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_errors": len(list(connection.execute("PRAGMA foreign_key_check"))),
            "event_chain": chain,
        },
        "sessions": {
            "total": len(sessions),
            "counts": nested_counter(session_counts),
            "duration_seconds": [
                {"day": day, "role": role, **summary(values)}
                for (day, role), values in sorted(session_durations.items())
            ],
            "cross_day_gap_seconds": {
                name: summary(values) for name, values in cross_day_gaps.items()
            },
        },
        "pair_intervals": {
            "total": len(pair_rows),
            "counts_by_day": dict(sorted(pair_counts.items())),
            "qualified_counts_by_day": dict(sorted(qualified_pair_counts.items())),
            "attempts_per_participant_day": summary(
                pair_attempts_by_participant_day.values()
            ),
            "duration_seconds": {
                str(day): summary(values) for day, values in sorted(pair_durations.items())
            },
            "qualified_interval_duration_seconds": {
                str(day): summary(values)
                for day, values in sorted(qualified_pair_durations.items())
            },
        },
        "events": {
            "total": sum(event_type_counts.values()),
            "counts": dict(event_type_counts.most_common()),
            "coverage_anomalies": coverage_anomalies,
            "phase_start_counts": nested_counter(phase_start_counts),
            "phase_end_seconds": {
                phase: summary(values) for phase, values in phase_end_seconds.items()
            },
        },
        "taps": {
            "pointer_count": len(tap_pointers),
            "result_count": len(tap_results),
            "missing_result_count": len(missing_results),
            "accepted_missing_result_count": len(accepted_missing),
            "missing_results": missing_results,
            "results_without_pointer": results_without_pointer,
            "eligible_positive": sum(
                count for (participant_id, label), count in eligible_labels.items() if label == 1
            ),
            "eligible_negative": sum(
                count for (participant_id, label), count in eligible_labels.items() if label == 0
            ),
            "quarantine_reasons": dict(quarantine_reasons),
            "cross_phase_activation_count": len(cross_phase_activations),
            "cross_phase_activations": cross_phase_activations,
            "eligible_provenance_error_count": len(eligible_provenance_errors),
            "eligible_provenance_errors": eligible_provenance_errors,
            "action_metrics": action_rows,
            "learning_by_wave_day": learning_rows,
            "loss_by_day_label": [
                {"day": day, "label": label, **summary(values)}
                for (day, label), values in sorted(losses.items())
            ],
            "shadow_loss_by_day_label": [
                {"day": day, "label": label, **summary(values)}
                for (day, label), values in sorted(shadow_losses.items())
            ],
        },
        "predictions": {
            "total": sum(prediction_routes.values()),
            "skipped": prediction_skipped,
            "routes": prediction_route_rows,
            "fallback_count": fallback_predictions,
            "probability_by_day_phase_policy": [
                {"day": day, "phase": phase, "policy": policy, **summary(values)}
                for (day, phase, policy), values in sorted(
                    prediction_probabilities.items(), key=lambda item: str(item[0])
                )
            ],
        },
        "quality": {
            "total": len(quality_rows),
            "counts": nested_counter(quality_counts),
            "attempt_counts": nested_counter(quality_attempt_counts),
            "anomalies": quality_anomalies,
            "network_median_ms": summary(network_medians),
            "network_p95_ms": summary(network_p95s),
        },
        "questionnaires": {
            "total": len(response_rows),
            "counts": nested_counter(response_counts),
            "daily_metrics": daily_questionnaire_metrics,
            "ended_early_true": ended_early,
        },
        "models": {
            "budget_kind": budget_kind,
            "budget_semantics": budget_semantics,
            "q_levels": q_levels,
            "counter_schema_columns": sorted(counter_columns),
            "event_derived_activation_opportunities": sum(opportunities.values()),
            "event_derived_eligible_feedbacks": sum(eligible_labels.values()),
            "stored_counter_totals": {
                key: sum(row[key] for row in counters.values())
                for key in (
                    "learning_opportunities",
                    "activation_opportunities",
                    "eligible_feedbacks",
                    "positive_feedbacks",
                    "negative_feedbacks",
                )
            },
            "eligible_labels": sum(eligible_labels.values()),
            "participants_with_zero_eligible_labels": sum(
                row["eligible_positive"] + row["eligible_negative"] == 0
                for row in participant_model_rows
            ),
            "native_q_coverage": native_q_coverage,
            "stored_q_coverage": stored_q_coverage,
            "valid_capped_q_coverage": valid_capped_q_coverage,
            "cap_attainment_rate": {
                str(budget): round(
                    native_q_coverage[budget] / len(participant_model_rows), 4
                )
                if participant_model_rows
                else None
                for budget in q_levels
            },
            "realized_feedback_by_cap": {
                str(budget): summary(values)
                for budget, values in realized_feedback_by_cap.items()
            },
            "participants": participant_model_rows,
            "shared_priors": shared_priors,
            "distinct_shared_prior_models": len(
                {row["model_hash"] for row in shared_priors}
            ),
            "prior_assignments": prior_assignments,
        },
        "pipeline": {"actions": dict(pipeline_actions)},
        "consent": consent,
    }
    connection.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "active_participants": len(participants),
                "events": report["events"]["total"],
                "event_chain_passed": chain["passed"],
                "accepted_taps_without_result": len(accepted_missing),
                "native_q_coverage": native_q_coverage,
                "stored_q_coverage": stored_q_coverage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
