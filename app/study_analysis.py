"""Participant-level behavioral summaries and bootstrap comparisons.

The analysis aggregates eligible timing corrections, opportunity denominators,
and within-session changes before forming participant-level comparisons.
It reads persisted research records and does not alter adaptive model state.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any

from .db import StudyDB
from .operator_resume import hide_block5_event, load_operator_resume


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(present) if present else None


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _participant_day_changes(
    rows: list[dict[str, Any]], group: str, metric: str
) -> dict[str, float]:
    by_participant: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        value = row.get(metric)
        if row.get("group") != group or value is None or not math.isfinite(float(value)):
            continue
        by_participant[str(row["participant_id"])][int(row["day"])] = float(value)
    return {
        participant_id: days[1] - days[3]
        for participant_id, days in by_participant.items()
        if 1 in days and 3 in days
    }


def _bootstrap_change_decomposition(
    personalized: dict[str, float],
    frozen: dict[str, float],
    *,
    replicates: int = 10_000,
    seed: int = 20260720,
) -> dict[str, Any]:
    personalized_values = list(personalized.values())
    frozen_values = list(frozen.values())
    if not personalized_values or not frozen_values:
        return {
            "personalized_participants": len(personalized_values),
            "frozen_participants": len(frozen_values),
            "personalized_day1_minus_day3": None,
            "personalized_bootstrap_95_ci": [None, None],
            "frozen_day1_minus_day3_proxy": None,
            "frozen_bootstrap_95_ci": [None, None],
            "difference_in_differences": None,
            "difference_in_differences_bootstrap_95_ci": [None, None],
        }
    rng = random.Random(seed)
    personalized_bootstrap: list[float] = []
    frozen_bootstrap: list[float] = []
    did_bootstrap: list[float] = []
    for _ in range(max(1, replicates)):
        p_mean = statistics.fmean(
            rng.choices(personalized_values, k=len(personalized_values))
        )
        f_mean = statistics.fmean(rng.choices(frozen_values, k=len(frozen_values)))
        personalized_bootstrap.append(p_mean)
        frozen_bootstrap.append(f_mean)
        did_bootstrap.append(p_mean - f_mean)
    return {
        "personalized_participants": len(personalized_values),
        "frozen_participants": len(frozen_values),
        "personalized_day1_minus_day3": _round(
            statistics.fmean(personalized_values)
        ),
        "personalized_bootstrap_95_ci": [
            _round(_percentile(personalized_bootstrap, 0.025)),
            _round(_percentile(personalized_bootstrap, 0.975)),
        ],
        "frozen_day1_minus_day3_proxy": _round(statistics.fmean(frozen_values)),
        "frozen_bootstrap_95_ci": [
            _round(_percentile(frozen_bootstrap, 0.025)),
            _round(_percentile(frozen_bootstrap, 0.975)),
        ],
        "difference_in_differences": _round(
            statistics.fmean(personalized_values) - statistics.fmean(frozen_values)
        ),
        "difference_in_differences_bootstrap_95_ci": [
            _round(_percentile(did_bootstrap, 0.025)),
            _round(_percentile(did_bootstrap, 0.975)),
        ],
    }


def _new_metric(**metadata: Any) -> dict[str, Any]:
    return {
        **metadata,
        "duration_seconds": 0.0,
        "predictions": 0,
        "auto_activations": 0,
        "manual_activations": 0,
        "interruptions": 0,
        "timing_interruptions": 0,
        "other_interruptions": 0,
        "state_transition_mistouches": 0,
        "_correction_latencies_ms": [],
    }


def _is_timing_interrupt(payload: dict[str, Any]) -> bool:
    evidence = payload.get("timing_attribution_confidence")
    if evidence is not None:
        return evidence == "speech_overlap_observed"
    return payload.get("user_speech_resumed") is True


def _finish_metric(metric: dict[str, Any]) -> dict[str, Any]:
    duration_minutes = float(metric["duration_seconds"]) / 60.0
    predictions = int(metric["predictions"])
    correction_count = int(metric["manual_activations"]) + int(metric["interruptions"])
    correction_latencies = [
        float(value)
        for value in metric.get("_correction_latencies_ms", [])
        if value is not None and math.isfinite(float(value))
    ]
    public_metric = {
        key: value for key, value in metric.items() if not key.startswith("_")
    }
    return {
        **public_metric,
        "duration_minutes": _round(duration_minutes),
        "activation_frequency_per_minute": _round(
            int(metric["auto_activations"]) / duration_minutes
            if duration_minutes > 0
            else None
        ),
        "manual_activation_frequency_per_minute": _round(
            int(metric["manual_activations"]) / duration_minutes
            if duration_minutes > 0
            else None
        ),
        "timing_interruption_frequency_per_minute": _round(
            int(metric["timing_interruptions"]) / duration_minutes
            if duration_minutes > 0
            else None
        ),
        "raw_interruption_frequency_per_minute": _round(
            int(metric["interruptions"]) / duration_minutes
            if duration_minutes > 0
            else None
        ),
        "correction_burden_per_minute": _round(
            correction_count / duration_minutes if duration_minutes > 0 else None
        ),
        "correction_burden_per_100_predictions": _round(
            100.0 * correction_count / predictions if predictions > 0 else None
        ),
        "state_transition_mistouch_frequency_per_minute": _round(
            int(metric["state_transition_mistouches"]) / duration_minutes
            if duration_minutes > 0
            else None
        ),
        "correction_latency_count": len(correction_latencies),
        "correction_latency_median_ms": _round(
            _percentile(correction_latencies, 0.5)
        ),
        "correction_latency_p90_ms": _round(
            _percentile(correction_latencies, 0.9)
        ),
    }


def _aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "activation_frequency_per_minute",
        "manual_activation_frequency_per_minute",
        "timing_interruption_frequency_per_minute",
        "raw_interruption_frequency_per_minute",
        "correction_burden_per_minute",
        "correction_burden_per_100_predictions",
        "state_transition_mistouch_frequency_per_minute",
        "correction_latency_median_ms",
        "correction_latency_p90_ms",
    )
    return {
        "participants": len({row["participant_id"] for row in metrics}),
        "probe_records": len(metrics),
        "records_with_duration": sum(float(row["duration_seconds"]) > 0 for row in metrics),
        "predictions": sum(int(row["predictions"]) for row in metrics),
        "auto_activations": sum(int(row["auto_activations"]) for row in metrics),
        "manual_activations": sum(int(row["manual_activations"]) for row in metrics),
        "interruptions": sum(int(row["interruptions"]) for row in metrics),
        "timing_interruptions": sum(int(row["timing_interruptions"]) for row in metrics),
        "other_interruptions": sum(int(row["other_interruptions"]) for row in metrics),
        "state_transition_mistouches": sum(
            int(row["state_transition_mistouches"]) for row in metrics
        ),
        "correction_latency_count": sum(
            int(row["correction_latency_count"]) for row in metrics
        ),
        **{key: _round(_mean([row.get(key) for row in metrics])) for key in keys},
    }


def _paired_bootstrap(
    low: dict[str, float],
    high: dict[str, float],
    *,
    margin: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    participant_ids = sorted(set(low) & set(high))
    differences = [high[value] - low[value] for value in participant_ids]
    if not differences:
        return {
            "paired_participants": 0,
            "mean_high_minus_low": None,
            "bootstrap_95_ci": [None, None],
            "equivalent_within_margin": False,
        }
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(max(1, replicates)):
        bootstrap.append(
            statistics.fmean(rng.choice(differences) for _ in differences)
        )
    bootstrap.sort()
    lower = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    upper = bootstrap[int(0.975 * (len(bootstrap) - 1))]
    return {
        "paired_participants": len(differences),
        "mean_high_minus_low": _round(statistics.fmean(differences)),
        "bootstrap_95_ci": [_round(lower), _round(upper)],
        "equivalence_margin": margin,
        "equivalent_within_margin": lower >= -margin and upper <= margin,
    }


def analyze_run(db: StudyDB, run_id: str, study: dict[str, Any]) -> dict[str, Any]:
    participants = {row["id"]: row for row in db.participants_in_run(run_id)}
    events = db.events_in_run(run_id)
    assignments = db.prior_probe_assignments(run_id=run_id, day=1)
    by_token = {str(row["probe_token"]): row for row in assignments}

    operator_resume_by_id = {
        participant_id: load_operator_resume(
            getattr(getattr(db, "settings", None), "data_dir", None),
            participant_id,
        )
        for participant_id in participants
    }
    candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for assignment in assignments:
        participant = participants[assignment["participant_id"]]
        key = (assignment["participant_id"], str(assignment["probe_token"]))
        candidate[key] = _new_metric(
            participant_id=assignment["participant_id"],
            neutral_id=participant.get("neutral_id"),
            wave=int(participant.get("enrollment_wave") or 0),
            probe_token=assignment["probe_token"],
            n=int(assignment["donor_count"]),
            q=int(assignment["opportunity_budget"]),
        )

    phase_starts: dict[tuple[str, str], int] = {}
    learning: dict[tuple[str, int], dict[str, Any]] = {}
    probes: dict[tuple[str, int, str], dict[str, Any]] = {}
    blocks: dict[tuple[str, int, int], dict[str, Any]] = {}
    block_phase_starts: dict[tuple[str, int, str], int] = {}
    feedback_by_day: dict[int, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0}
    )
    calibration_updates: list[dict[str, Any]] = []
    decision_eligible_predictions = 0
    decision_positive_predictions = 0

    def study_b_metric(participant_id: str, day: int) -> dict[str, Any] | None:
        participant = participants.get(participant_id)
        if participant is None or participant.get("cohort") != "study_b" or not day:
            return None
        key = (participant_id, day)
        if key not in learning:
            learning[key] = _new_metric(
                participant_id=participant_id,
                neutral_id=participant.get("neutral_id"),
                group=participant.get("group_name"),
                day=day,
            )
        return learning[key]

    def policy_metric(
        participant_id: str, day: int, policy: str
    ) -> dict[str, Any] | None:
        participant = participants.get(participant_id)
        if participant is None or participant.get("cohort") != "study_b" or not day:
            return None
        key = (participant_id, day, policy)
        if key not in probes:
            probes[key] = _new_metric(
                participant_id=participant_id,
                neutral_id=participant.get("neutral_id"),
                group=participant.get("group_name"),
                day=day,
                policy=policy,
            )
        return probes[key]

    def block_metric(
        participant_id: str, day: int, block: int
    ) -> dict[str, Any] | None:
        participant = participants.get(participant_id)
        if participant is None or not day or block < 1:
            return None
        key = (participant_id, day, block)
        if key not in blocks:
            blocks[key] = _new_metric(
                participant_id=participant_id,
                neutral_id=participant.get("neutral_id"),
                group=participant.get("group_name"),
                day=day,
                active_block_index=block,
                analysis_eligible=True,
            )
        return blocks[key]

    for event in events:
        payload = event["payload"]
        participant_id = str(event["participant_id"])
        day = int(event.get("day") or payload.get("day") or 0)
        event_type = str(event["event_type"])
        analysis_eligible = bool(
            payload.get(
                "analysis_eligible",
                event_type not in {"rest.start", "rest.end", "rest.resume"},
            )
        )
        try:
            active_block_index = int(
                payload.get("active_block_index", payload.get("block", 0)) or 0
            )
        except (TypeError, ValueError):
            active_block_index = 0
        if hide_block5_event(
            operator_resume_by_id.get(participant_id),
            block=active_block_index,
            server_ms=int(event.get("server_ms") or 0),
        ):
            continue
        token = str(payload.get("candidate_probe_token") or "")
        if not token and payload.get("phase") == "candidate_probe":
            token = str(payload.get("policy_token") or "")
        candidate_metric = candidate.get((participant_id, token)) if token else None

        if event_type == "model.prediction" and payload.get("decision_eligible") is True:
            decision_eligible_predictions += 1
            decision_positive_predictions += int(payload.get("activate") is True)
        if (
            event_type in {"tap.manual_activate", "tap.interrupt"}
            and payload.get("training_eligible") is True
            and payload.get("training_applied") is True
            and payload.get("label") in {0, 1}
        ):
            label = int(payload["label"])
            feedback_by_day[day]["positive" if label == 1 else "negative"] += 1
            if payload.get("personal_calibration_applied") is True:
                calibration_updates.append(
                    {
                        "participant_id": participant_id,
                        "day": day,
                        "label": label,
                        "bias_before": float(payload["personal_logit_bias_before"]),
                        "bias_after": float(payload["personal_logit_bias_after"]),
                        "network_update_scale": float(
                            payload.get("network_update_scale", 1.0)
                        ),
                        "update_count": int(
                            payload.get("personal_calibration_update_count", 0)
                        ),
                    }
                )

        if event_type == "phase.start" and token in by_token:
            phase_starts[(str(event.get("session_id") or ""), token)] = int(event["server_ms"])
        elif event_type == "phase.end" and token in by_token and candidate_metric:
            start = phase_starts.pop((str(event.get("session_id") or ""), token), None)
            server_seconds = max(0.0, (int(event["server_ms"]) - start) / 1000.0) if start else 0.0
            reported = float(
                payload.get("elapsed_seconds", payload.get("completed_seconds", 0)) or 0
            )
            candidate_metric["duration_seconds"] += max(server_seconds, reported)

        phase = str(payload.get("phase") or payload.get("activation_phase") or "")
        policy = str(payload.get("policy") or payload.get("activation_policy") or "")
        if event_type in {"phase.start", "phase.end"} and phase == "learning":
            metric = study_b_metric(participant_id, day)
            if metric is not None:
                key = (str(event.get("session_id") or ""), "learning")
                if event_type == "phase.start":
                    phase_starts[key] = int(event["server_ms"])
                else:
                    start = phase_starts.pop(key, None)
                    server_seconds = (
                        max(0.0, (int(event["server_ms"]) - start) / 1000.0)
                        if start
                        else 0.0
                    )
                    reported = float(
                        payload.get("elapsed_seconds", payload.get("completed_seconds", 0))
                        or 0
                    )
                    metric["duration_seconds"] += max(server_seconds, reported)
        if event_type in {"phase.start", "phase.end"} and phase == "micro_probe":
            token_letter = str(payload.get("policy_token") or "")
            policy_name = {
                "A": "current",
                "B": "matched_other",
                "C": "vad",
            }.get(token_letter, "")
            metric = policy_metric(participant_id, day, policy_name) if policy_name else None
            if metric is not None:
                key = (str(event.get("session_id") or ""), f"micro:{policy_name}")
                if event_type == "phase.start":
                    phase_starts[key] = int(event["server_ms"])
                else:
                    start = phase_starts.pop(key, None)
                    server_seconds = (
                        max(0.0, (int(event["server_ms"]) - start) / 1000.0)
                        if start
                        else 0.0
                    )
                    reported = float(
                        payload.get("elapsed_seconds", payload.get("completed_seconds", 0))
                        or 0
                    )
                    metric["duration_seconds"] += max(server_seconds, reported)

        current_block_metric = (
            block_metric(participant_id, day, active_block_index)
            if active_block_index and analysis_eligible
            else None
        )
        if current_block_metric is not None and event_type in {
            "phase.start",
            "phase.end",
            "companion.window_start",
            "companion.window_end",
        }:
            segment = (
                "primary"
                if event_type.startswith("phase.")
                else "companion"
            )
            key = (
                str(event.get("session_id") or ""),
                active_block_index,
                segment,
            )
            if event_type.endswith("start"):
                block_phase_starts[key] = int(event["server_ms"])
            else:
                start = block_phase_starts.pop(key, None)
                server_seconds = (
                    max(0.0, (int(event["server_ms"]) - start) / 1000.0)
                    if start
                    else 0.0
                )
                reported = float(
                    payload.get("elapsed_seconds", payload.get("completed_seconds", 0))
                    or 0
                )
                current_block_metric["duration_seconds"] += max(
                    server_seconds, reported
                )

        metrics: list[dict[str, Any]] = []
        if candidate_metric is not None:
            metrics.append(candidate_metric)
        if phase == "learning" and policy in {"", "current"}:
            metric = study_b_metric(participant_id, day)
            if metric is not None:
                metrics.append(metric)
        if phase == "micro_probe" and policy:
            metric = policy_metric(participant_id, day, policy)
            if metric is not None:
                metrics.append(metric)
        for metric in metrics:
            if event_type == "model.prediction" and not payload.get("skipped"):
                metric["predictions"] += 1
            elif event_type == "agent.auto_activate":
                metric["auto_activations"] += 1
            elif event_type == "tap.manual_activate":
                metric["manual_activations"] += 1
            elif event_type == "tap.interrupt":
                metric["interruptions"] += 1
                if payload.get("activation_to_tap_ms") is not None:
                    metric["_correction_latencies_ms"].append(
                        float(payload["activation_to_tap_ms"])
                    )
                if _is_timing_interrupt(payload):
                    metric["timing_interruptions"] += 1
                else:
                    metric["other_interruptions"] += 1
            elif event_type == "tap.state_transition_mistouch":
                metric["state_transition_mistouches"] += 1

        if current_block_metric is not None:
            if event_type == "model.prediction" and not payload.get("skipped"):
                current_block_metric["predictions"] += 1
            elif event_type == "agent.auto_activate":
                current_block_metric["auto_activations"] += 1
            elif event_type == "tap.manual_activate":
                current_block_metric["manual_activations"] += 1
            elif event_type == "tap.interrupt":
                current_block_metric["interruptions"] += 1
                if payload.get("activation_to_tap_ms") is not None:
                    current_block_metric["_correction_latencies_ms"].append(
                        float(payload["activation_to_tap_ms"])
                    )
                if _is_timing_interrupt(payload):
                    current_block_metric["timing_interruptions"] += 1
                else:
                    current_block_metric["other_interruptions"] += 1
            elif event_type == "tap.state_transition_mistouch":
                current_block_metric["state_transition_mistouches"] += 1

    candidate_rows = [_finish_metric(value) for value in candidate.values()]
    learning_rows = [_finish_metric(value) for value in learning.values()]
    policy_rows = [_finish_metric(value) for value in probes.values()]
    block_rows = [_finish_metric(value) for value in blocks.values()]

    cell_rows: list[dict[str, Any]] = []
    cell_values: dict[tuple[int, int], dict[str, float]] = {}
    for n, q in sorted({(row["n"], row["q"]) for row in candidate_rows}):
        rows = [row for row in candidate_rows if row["n"] == n and row["q"] == q]
        cell_rows.append({"n": n, "q": q, **_aggregate(rows)})
        cell_values[(n, q)] = {
            row["participant_id"]: float(row["correction_burden_per_minute"])
            for row in rows
            if row["correction_burden_per_minute"] is not None
        }

    saturation_rule = study["pipeline"]["saturation_decision"]
    low_n, high_n = [int(value) for value in saturation_rule["paired_n_contrast"]]
    margin = float(saturation_rule["equivalence_margin_per_minute"])
    replicates = int(saturation_rule["bootstrap_replicates"])
    saturation_contrasts = []
    for q in saturation_rule["required_q_levels"]:
        contrast = _paired_bootstrap(
            cell_values.get((low_n, int(q)), {}),
            cell_values.get((high_n, int(q)), {}),
            margin=margin,
            replicates=replicates,
            seed=int(q) * 1009 + high_n,
        )
        saturation_contrasts.append({"q": int(q), **contrast})
    saturation_supported = bool(saturation_contrasts) and all(
        row["equivalent_within_margin"] for row in saturation_contrasts
    )

    group_day: list[dict[str, Any]] = []
    for group, day in sorted({(row["group"], row["day"]) for row in learning_rows}):
        rows = [row for row in learning_rows if row["group"] == group and row["day"] == day]
        group_day.append({"group": group, "day": day, **_aggregate(rows)})
    policy_summary: list[dict[str, Any]] = []
    for group, day, policy in sorted(
        {(row["group"], row["day"], row["policy"]) for row in policy_rows}
    ):
        rows = [
            row
            for row in policy_rows
            if row["group"] == group and row["day"] == day and row["policy"] == policy
        ]
        policy_summary.append(
            {"group": group, "day": day, "policy": policy, **_aggregate(rows)}
        )

    selected = [row for row in db.shared_priors(run_id) if row["status"] == "selected"]
    selected_cell = (
        (int(selected[0]["donor_count"]), int(selected[0]["opportunity_budget"]))
        if len(selected) == 1
        else None
    )
    cell_j = {
        (row["n"], row["q"]): row["correction_burden_per_minute"]
        for row in cell_rows
    }
    j0 = cell_j.get((0, 0))
    j_shared = cell_j.get(selected_cell) if selected_cell else None
    shared_gain = j0 - j_shared if j0 is not None and j_shared is not None else None
    personalized_changes = _participant_day_changes(
        learning_rows, "personalized", "correction_burden_per_minute"
    )
    frozen_changes = _participant_day_changes(
        learning_rows, "frozen", "correction_burden_per_minute"
    )
    # Use the same complete-case participant estimand for the point estimate
    # and its bootstrap CI. Separate Day 1/Day 3 group means can silently use
    # different participants under attrition and would not match the bootstrap.
    personalized_observed = _mean(list(personalized_changes.values()))
    user_accommodation = _mean(list(frozen_changes.values()))
    personalized_net = (
        personalized_observed - user_accommodation
        if personalized_observed is not None and user_accommodation is not None
        else None
    )
    total_model_gain = (
        shared_gain + personalized_net
        if shared_gain is not None and personalized_net is not None
        else None
    )
    shared_fraction = (
        shared_gain / total_model_gain
        if total_model_gain is not None and total_model_gain > 0
        else None
    )
    personalized_fraction = (
        personalized_net / total_model_gain
        if total_model_gain is not None and total_model_gain > 0
        else None
    )
    longitudinal_decomposition = _bootstrap_change_decomposition(
        personalized_changes,
        frozen_changes,
    )
    calibration_states = [
        row
        for row in db.table_rows("participant_calibrations")
        if row["run_id"] == run_id
    ]
    event_update_count = {
        participant_id: max(
            (
                row["update_count"]
                for row in calibration_updates
                if row["participant_id"] == participant_id
            ),
            default=0,
        )
        for participant_id in participants
    }
    state_update_count = {
        str(row["participant_id"]): int(row["update_count"])
        for row in calibration_states
    }
    persistence_mismatches = {
        participant_id: {
            "event_update_count": event_update_count.get(participant_id, 0),
            "state_update_count": state_update_count.get(participant_id, 0),
        }
        for participant_id in set(event_update_count) | set(state_update_count)
        if event_update_count.get(participant_id, 0)
        != state_update_count.get(participant_id, 0)
    }

    return {
        "run_id": run_id,
        "data_origin": next(iter(participants.values())).get("data_origin")
        if participants
        else None,
        "analysis_scope": (
            "Model-, task-, and tap-conditioned predictive/behavioral contribution; "
            "not a literal percentage of neural parameters or universal semantics."
        ),
        "primary_metric": (
            "manual activations plus all user interruptions of observable activation "
            "episodes per minute"
        ),
        "negative_label_semantics": "observable_activation_interrupt_v2",
        "six_minute_blocks": {
            "rows": block_rows,
            "summary": _aggregate(block_rows),
            "rest_excluded": True,
            "note": (
                "Each row is one participant's pre-registered six-minute active "
                "block (four primary-agent minutes plus two companion-agent minutes). "
                "The following one-minute rest contributes neither duration nor events."
            ),
        },
        "operating_point_calibration": {
            "eligible_feedback_by_day": [
                {"day": day, **counts}
                for day, counts in sorted(feedback_by_day.items())
            ],
            "personal_updates": len(calibration_updates),
            "personal_state_rows": len(calibration_states),
            "personal_bias_mean": _round(
                _mean(
                    [float(row["personal_logit_bias"]) for row in calibration_states]
                )
            ),
            "personal_bias_min": (
                min(float(row["personal_logit_bias"]) for row in calibration_states)
                if calibration_states
                else None
            ),
            "personal_bias_max": (
                max(float(row["personal_logit_bias"]) for row in calibration_states)
                if calibration_states
                else None
            ),
            "observed_network_update_scales": sorted(
                {row["network_update_scale"] for row in calibration_updates}
            ),
            "decision_eligible_predictions": decision_eligible_predictions,
            "decision_positive_rate": _round(
                decision_positive_predictions / decision_eligible_predictions
                if decision_eligible_predictions
                else None
            ),
            "state_persistence_consistent": not persistence_mismatches,
            "state_persistence_mismatches": persistence_mismatches,
            "label_quota_used": False,
        },
        "candidate_grid": cell_rows,
        "saturation": {
            "low_n": low_n,
            "high_n": high_n,
            "contrasts": saturation_contrasts,
            "supported": saturation_supported,
            "decision_must_remain_blinded": True,
        },
        "study_b": {
            "group_by_day": group_day,
            "policy_probes": policy_summary,
            "selected_cell": list(selected_cell) if selected_cell else None,
        },
        "contribution_decomposition": {
            "shared_gain_m0_minus_selected": _round(shared_gain),
            "personalized_observed_day1_minus_day3": _round(personalized_observed),
            "frozen_day_change_proxy_day1_minus_day3": _round(user_accommodation),
            "frozen_user_accommodation_day1_minus_day3_deprecated": _round(
                user_accommodation
            ),
            "personalized_net_difference_in_differences": _round(personalized_net),
            "participant_level_bootstrap": longitudinal_decomposition,
            "interpretation": {
                "frozen_day_change_proxy": (
                    "Human accommodation plus practice, day/task context, regression, and "
                    "other time-varying effects while network weights are frozen."
                ),
                "difference_in_differences": (
                    "Additional change associated with enabling personalized learning, "
                    "conditional on the randomized design and a parallel-trends assumption."
                ),
                "not_a_literal_parameter_fraction": True,
            },
            "shared_fraction_of_positive_model_gain": _round(shared_fraction),
            "personalized_fraction_of_positive_model_gain": _round(personalized_fraction),
            "estimable": all(
                value is not None
                for value in (shared_gain, personalized_net, total_model_gain)
            ),
        },
        "readiness": {
            "candidate_assignments": len(assignments),
            "candidate_records_with_duration": sum(
                float(row["duration_seconds"]) > 0 for row in candidate_rows
            ),
            "study_b_participant_days": len(learning_rows),
            "formal_inference_allowed": bool(
                participants
                and all(row.get("data_origin") == "formal" for row in participants.values())
            ),
        },
    }
