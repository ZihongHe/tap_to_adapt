"""Participant-level read-only audit for one Study A wave.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Iterable


def finite(values: Iterable[float | int | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def describe(values: Iterable[float | int | None]) -> dict[str, Any]:
    data = finite(values)
    if not data:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: Iterable[float | int | None], *, seed: int, draws: int = 20_000
) -> list[float] | None:
    data = finite(values)
    if not data:
        return None
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(data, k=len(data))) for _ in range(draws)]
    return [float(percentile(means, 0.025)), float(percentile(means, 0.975))]


def paired_test(
    first: dict[str, float | None],
    second: dict[str, float | None],
    *,
    seed: int,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    ids = sorted(
        key
        for key in set(first) & set(second)
        if first[key] is not None
        and second[key] is not None
        and math.isfinite(float(first[key]))
        and math.isfinite(float(second[key]))
    )
    first_values = [float(first[key]) for key in ids]
    second_values = [float(second[key]) for key in ids]
    differences = [right - left for left, right in zip(first_values, second_values)]
    result: dict[str, Any] = {
        "n": len(ids),
        "first": describe(first_values),
        "second": describe(second_values),
        "second_minus_first": {
            **describe(differences),
            "bootstrap_mean_ci95": bootstrap_mean_ci(differences, seed=seed),
        },
        "lower_is_better": lower_is_better,
        "improved": sum(value < 0 if lower_is_better else value > 0 for value in differences),
        "unchanged": sum(value == 0 for value in differences),
        "worsened": sum(value > 0 if lower_is_better else value < 0 for value in differences),
    }
    try:
        from scipy.stats import rankdata, wilcoxon

        nonzero = [value for value in differences if value != 0]
        if nonzero:
            test = wilcoxon(first_values, second_values, zero_method="wilcox")
            ranks = rankdata([abs(value) for value in nonzero])
            positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
            negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
            denominator = positive + negative
            result["wilcoxon"] = {
                "statistic": float(test.statistic),
                "p": float(test.pvalue),
                "rank_biserial_second_minus_first": (
                    float((positive - negative) / denominator) if denominator else None
                ),
            }
        else:
            result["wilcoxon"] = {"statistic": 0.0, "p": 1.0, "rank_biserial_second_minus_first": 0.0}
    except (ImportError, ValueError) as exc:
        result["wilcoxon"] = {"available": False, "reason": str(exc)}
    return result


def assignment_number(value: str | None) -> int:
    match = re.fullmatch(r"A(\d+)", value or "")
    return int(match.group(1)) if match else 10_000


def parsed(value: str | None) -> dict[str, Any]:
    try:
        candidate = json.loads(value or "{}")
        return candidate if isinstance(candidate, dict) else {}
    except json.JSONDecodeError:
        return {}


def action_values(counter: collections.Counter[str]) -> dict[str, Any]:
    auto = int(counter["auto"])
    manual = int(counter["manual"])
    interruption = int(counter["interrupt"])
    turns = auto + manual
    proxy_true_positive = max(0, auto - interruption)
    proxy_precision = proxy_true_positive / auto if auto else None
    proxy_recall_denominator = proxy_true_positive + manual
    proxy_recall = (
        proxy_true_positive / proxy_recall_denominator
        if proxy_recall_denominator
        else None
    )
    proxy_f1 = (
        2 * proxy_precision * proxy_recall / (proxy_precision + proxy_recall)
        if proxy_precision is not None
        and proxy_recall is not None
        and proxy_precision + proxy_recall
        else None
    )
    return {
        "auto": auto,
        "manual": manual,
        "interrupt": interruption,
        "eligible_positive": int(counter["eligible_positive"]),
        "eligible_negative": int(counter["eligible_negative"]),
        "turns": turns,
        "manual_share": manual / turns if turns else None,
        "auto_interrupt_rate": interruption / auto if auto else None,
        "correction_rate": (manual + interruption) / turns if turns else None,
        "revealed_preference_proxy": {
            "assumed_true_positive": proxy_true_positive,
            "assumed_false_positive": interruption,
            "assumed_false_negative": manual,
            "precision": proxy_precision,
            "recall": proxy_recall,
            "f1": proxy_f1,
        },
    }


def add_counter(
    target: collections.Counter[str], source: collections.Counter[str]
) -> collections.Counter[str]:
    target.update(source)
    return target


def metric_map(
    rows: dict[tuple[str, int], dict[str, Any]], day: int, metric: str
) -> dict[str, float | None]:
    return {
        participant_id: values.get(metric)
        for (participant_id, observed_day), values in rows.items()
        if observed_day == day
    }


def counter_rows(counter: collections.Counter[Any]) -> list[dict[str, Any]]:
    output = []
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        values = key if isinstance(key, tuple) else (key,)
        output.append({"key": list(values), "count": int(count)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Participant-level read-only audit for one Study A wave."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--wave", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    run = connection.execute(
        "SELECT * FROM study_runs WHERE slug=?", (args.run_slug,)
    ).fetchone()
    if run is None:
        raise SystemExit(f"Run not found: {args.run_slug}")
    run_id = str(run["id"])
    protocol = parsed(str(run["protocol_json"] or "{}"))
    q_levels = [
        int(value)
        for value in protocol.get("pipeline", {}).get("q_feedback_levels", [2, 4, 8])
    ]
    participants = [
        dict(row)
        for row in connection.execute(
            """SELECT * FROM participants
               WHERE run_id=? AND cohort='study_a' AND enrollment_wave=?""",
            (run_id, args.wave),
        )
    ]
    participants.sort(key=lambda row: assignment_number(row.get("assignment_id")))
    if not participants:
        raise SystemExit(f"No Study A participants found in wave {args.wave}")
    participant_ids = {str(row["id"]) for row in participants}
    participant_by_id = {str(row["id"]): row for row in participants}
    assignment_by_id = {
        str(row["id"]): str(row.get("assignment_id") or row["id"])
        for row in participants
    }
    candidate_probe_participants = {
        str(row["participant_id"])
        for row in connection.execute(
            "SELECT DISTINCT participant_id FROM prior_probe_assignments WHERE run_id=?",
            (run_id,),
        )
    }

    sessions = [
        dict(row)
        for row in connection.execute("SELECT * FROM sessions WHERE run_id=?", (run_id,))
        if str(row["participant_id"]) in participant_ids
    ]
    session_by_id = {str(row["id"]): row for row in sessions}
    study_session_by_participant_day = {
        (str(row["participant_id"]), int(row["day"])): row
        for row in sessions
        if row["client_role"] == "study"
    }

    actions: dict[
        tuple[str, int, str | None, str | None], collections.Counter[str]
    ] = collections.defaultdict(collections.Counter)
    event_counts: collections.Counter[str] = collections.Counter()
    coverage: dict[tuple[str, int], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    tap_pointers: dict[str, dict[str, Any]] = {}
    tap_results: dict[str, dict[str, Any]] = {}
    quarantine: collections.Counter[str] = collections.Counter()
    eligible_labels: dict[str, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    cross_phase: list[dict[str, Any]] = []
    missing_interrupt_provenance: list[dict[str, Any]] = []
    provenance_errors: list[dict[str, Any]] = []
    topics: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    tts_generated: collections.Counter[Any] = collections.Counter()
    tts_latency: list[float] = []
    tts_failures: list[dict[str, Any]] = []
    turn_errors: list[dict[str, Any]] = []
    playback: collections.Counter[Any] = collections.Counter()
    chat_turns: collections.Counter[Any] = collections.Counter()
    primary_chat_coverage: collections.Counter[tuple[str, int]] = collections.Counter()
    prediction_routes: collections.Counter[tuple[Any, Any]] = collections.Counter()

    event_rows = connection.execute(
        "SELECT * FROM events WHERE run_id=? ORDER BY server_ms, id", (run_id,)
    )
    for raw_event in event_rows:
        event = dict(raw_event)
        participant_id = str(event["participant_id"])
        if participant_id not in participant_ids:
            continue
        session = session_by_id.get(str(event.get("session_id") or ""))
        day = int(session["day"]) if session else 0
        role = str(session["client_role"]) if session else "unknown"
        event_type = str(event["event_type"])
        payload = parsed(str(event["payload"]))
        event_counts[event_type] += 1
        if day:
            coverage[(participant_id, day)][event_type] += 1

        if event_type in {"agent.auto_activate", "tap.manual_activate", "tap.interrupt"}:
            phase = payload.get("activation_phase", payload.get("phase"))
            policy = payload.get("activation_policy", payload.get("policy"))
            counter = actions[(participant_id, day, phase, policy)]
            if event_type == "agent.auto_activate":
                counter["auto"] += 1
            elif event_type == "tap.manual_activate":
                counter["manual"] += 1
                if payload.get("training_eligible"):
                    counter["eligible_positive"] += 1
            elif payload.get("quarantine_reason") != "manual_activate_interrupt_conflict":
                counter["interrupt"] += 1
                if payload.get("training_eligible"):
                    counter["eligible_negative"] += 1
            if event_type.startswith("tap.") and payload.get("training_eligible"):
                label = int(payload.get("label"))
                eligible_labels[participant_id][label] += 1
                if (
                    not payload.get("activation_id")
                    or payload.get("activation_phase") != "learning"
                    or payload.get("activation_policy") != "current"
                ):
                    provenance_errors.append(
                        {
                            "assignment_id": assignment_by_id[participant_id],
                            "day": day,
                            "event_type": event_type,
                            "activation_id": payload.get("activation_id"),
                            "activation_phase": payload.get("activation_phase"),
                            "activation_policy": payload.get("activation_policy"),
                        }
                    )
            if event_type == "tap.interrupt":
                activation_phase = payload.get("activation_phase")
                tap_phase = payload.get("tap_phase", payload.get("phase"))
                activation_policy = payload.get("activation_policy")
                tap_policy = payload.get("tap_policy", payload.get("policy"))
                if activation_phase is None or activation_policy is None:
                    missing_interrupt_provenance.append(
                        {
                            "assignment_id": assignment_by_id[participant_id],
                            "day": day,
                            "tap_phase": tap_phase,
                            "tap_policy": tap_policy,
                            "training_eligible": bool(payload.get("training_eligible")),
                            "quarantine_reason": payload.get("quarantine_reason"),
                        }
                    )
                elif activation_phase != tap_phase or activation_policy != tap_policy:
                    cross_phase.append(
                        {
                            "assignment_id": assignment_by_id[participant_id],
                            "day": day,
                            "activation_phase": activation_phase,
                            "tap_phase": tap_phase,
                            "activation_policy": activation_policy,
                            "tap_policy": tap_policy,
                            "training_eligible": bool(payload.get("training_eligible")),
                        }
                    )

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
                quarantine[str(payload["quarantine_reason"])] += 1
        elif event_type in {"topic.selected", "topic.changed"}:
            topics[(participant_id, day)].append(
                {"event_type": event_type, "server_ms": event["server_ms"], **payload}
            )
        elif event_type == "tts.generated":
            tts_generated[
                (
                    role,
                    payload.get("provider"),
                    payload.get("voice"),
                    payload.get("format"),
                    payload.get("sample_rate_hz"),
                    bool(payload.get("fallback")),
                )
            ] += 1
            if isinstance(payload.get("latency_ms"), (int, float)):
                tts_latency.append(float(payload["latency_ms"]))
        elif event_type == "tts.server_failed":
            tts_failures.append(
                {
                    "assignment_id": assignment_by_id[participant_id],
                    "day": day,
                    "role": role,
                    "message": payload.get("message"),
                    "fallback_used": payload.get("fallback_used"),
                }
            )
        elif event_type == "agent.turn_error":
            turn_errors.append(
                {
                    "assignment_id": assignment_by_id[participant_id],
                    "day": day,
                    "role": role,
                    "message": payload.get("message"),
                }
            )
        elif event_type in {"tts.playback_start", "tts.playback_end", "tts.playback_failed"}:
            playback[(role, event_type)] += 1
        elif event_type == "chat.turn":
            source = str(payload.get("source") or role)
            chat_turns[(day, source, payload.get("mode"), payload.get("model"))] += 1
            if source == "primary":
                primary_chat_coverage[(participant_id, day)] += 1
        elif event_type == "model.prediction":
            prediction_routes[(payload.get("policy"), payload.get("policy_effective"))] += 1

    participant_action_rows: list[dict[str, Any]] = []
    learning_rows: dict[tuple[str, int], dict[str, Any]] = {}
    frozen_rows: dict[tuple[str, int], dict[str, Any]] = {}
    probe_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for (participant_id, day, phase, policy), counter in actions.items():
        values = action_values(counter)
        participant_action_rows.append(
            {
                "assignment_id": assignment_by_id[participant_id],
                "day": day,
                "phase": phase,
                "policy": policy,
                **values,
            }
        )
        if phase == "learning" and policy == "current":
            learning_rows[(participant_id, day)] = values
        elif phase == "frozen" and policy == "current":
            frozen_rows[(participant_id, day)] = values
        elif phase == "micro_probe" and policy in {"base", "current", "vad"}:
            probe_rows[(participant_id, day, str(policy))] = values
    participant_action_rows.sort(
        key=lambda row: (assignment_number(row["assignment_id"]), row["day"], str(row["phase"]), str(row["policy"]))
    )

    learning_metrics = [
        "auto",
        "manual",
        "interrupt",
        "manual_share",
        "auto_interrupt_rate",
        "correction_rate",
    ]
    learning_summary: dict[str, Any] = {}
    learning_tests: dict[str, Any] = {}
    for metric in learning_metrics:
        learning_summary[metric] = {
            str(day): describe(
                values.get(metric)
                for (participant_id, observed_day), values in learning_rows.items()
                if observed_day == day
            )
            for day in (1, 2, 3)
        }
        learning_tests[metric] = paired_test(
            metric_map(learning_rows, 1, metric),
            metric_map(learning_rows, 3, metric),
            seed=args.bootstrap_seed + len(learning_tests),
            lower_is_better=metric != "auto",
        )

    learning_pooled: dict[str, Any] = {}
    frozen_pooled: dict[str, Any] = {}
    for day in (1, 2, 3):
        learning_counter: collections.Counter[str] = collections.Counter()
        frozen_counter: collections.Counter[str] = collections.Counter()
        for participant_id in participant_ids:
            add_counter(
                learning_counter,
                actions[(participant_id, day, "learning", "current")],
            )
            add_counter(
                frozen_counter,
                actions[(participant_id, day, "frozen", "current")],
            )
        learning_pooled[str(day)] = action_values(learning_counter)
        frozen_pooled[str(day)] = action_values(frozen_counter)

    probe_per_participant: dict[tuple[str, str], dict[str, Any]] = {}
    for participant_id in participant_ids:
        for policy in ("base", "current", "vad"):
            combined: collections.Counter[str] = collections.Counter()
            for day in (1, 2, 3):
                add_counter(combined, actions[(participant_id, day, "micro_probe", policy)])
            probe_per_participant[(participant_id, policy)] = action_values(combined)

    probe_summary: dict[str, Any] = {}
    for policy in ("base", "current", "vad"):
        probe_summary[policy] = {
            metric: describe(
                probe_per_participant[(participant_id, policy)].get(metric)
                for participant_id in participant_ids
            )
            for metric in learning_metrics
        }
        probe_summary[policy]["revealed_preference_proxy_f1"] = describe(
            probe_per_participant[(participant_id, policy)]["revealed_preference_proxy"].get("f1")
            for participant_id in participant_ids
        )
    probe_pooled_by_day: dict[str, Any] = {}
    for day in (1, 2, 3):
        probe_pooled_by_day[str(day)] = {}
        for policy in ("base", "current", "vad"):
            combined = collections.Counter()
            for participant_id in participant_ids:
                add_counter(combined, actions[(participant_id, day, "micro_probe", policy)])
            probe_pooled_by_day[str(day)][policy] = action_values(combined)

    probe_tests: dict[str, Any] = {"all_days": {}, "by_day": {}}
    for metric in ("manual_share", "auto_interrupt_rate", "correction_rate"):
        current = {
            participant_id: probe_per_participant[(participant_id, "current")].get(metric)
            for participant_id in participant_ids
        }
        vad = {
            participant_id: probe_per_participant[(participant_id, "vad")].get(metric)
            for participant_id in participant_ids
        }
        base = {
            participant_id: probe_per_participant[(participant_id, "base")].get(metric)
            for participant_id in participant_ids
        }
        probe_tests["all_days"][f"current_minus_vad_{metric}"] = paired_test(
            vad,
            current,
            seed=args.bootstrap_seed + 100 + len(probe_tests["all_days"]),
        )
        probe_tests["all_days"][f"current_minus_base_{metric}"] = paired_test(
            base,
            current,
            seed=args.bootstrap_seed + 200 + len(probe_tests["all_days"]),
        )
    current_proxy_f1 = {
        participant_id: probe_per_participant[(participant_id, "current")]["revealed_preference_proxy"].get("f1")
        for participant_id in participant_ids
    }
    vad_proxy_f1 = {
        participant_id: probe_per_participant[(participant_id, "vad")]["revealed_preference_proxy"].get("f1")
        for participant_id in participant_ids
    }
    probe_tests["all_days"]["current_minus_vad_revealed_preference_proxy_f1"] = paired_test(
        vad_proxy_f1,
        current_proxy_f1,
        seed=args.bootstrap_seed + 299,
        lower_is_better=False,
    )
    for day in (1, 2, 3):
        probe_tests["by_day"][str(day)] = {}
        for metric in ("manual_share", "auto_interrupt_rate", "correction_rate"):
            vad = {
                participant_id: probe_rows.get((participant_id, day, "vad"), {}).get(metric)
                for participant_id in participant_ids
            }
            current = {
                participant_id: probe_rows.get((participant_id, day, "current"), {}).get(metric)
                for participant_id in participant_ids
            }
            probe_tests["by_day"][str(day)][f"current_minus_vad_{metric}"] = paired_test(
                vad,
                current,
                seed=args.bootstrap_seed + day * 10 + len(probe_tests["by_day"][str(day)]),
            )
    try:
        from scipy.stats import friedmanchisquare

        base_values = []
        current_values = []
        vad_values = []
        for participant_id in sorted(participant_ids):
            values = [
                probe_per_participant[(participant_id, policy)].get("correction_rate")
                for policy in ("base", "current", "vad")
            ]
            if all(value is not None for value in values):
                base_values.append(float(values[0]))
                current_values.append(float(values[1]))
                vad_values.append(float(values[2]))
        friedman = friedmanchisquare(base_values, current_values, vad_values)
        probe_tests["friedman_correction_rate"] = {
            "n": len(base_values),
            "statistic": float(friedman.statistic),
            "p": float(friedman.pvalue),
        }
    except (ImportError, ValueError) as exc:
        probe_tests["friedman_correction_rate"] = {"available": False, "reason": str(exc)}

    order_by_day: dict[int, collections.Counter[str]] = {
        day: collections.Counter() for day in (1, 2, 3)
    }
    position_counts: collections.Counter[tuple[str, int]] = collections.Counter()
    for participant in participants:
        schedule = json.loads(str(participant.get("probe_schedule") or "[]"))
        for day, order in enumerate(schedule, start=1):
            order = str(order)
            order_by_day[day][order] += 1
            for position, policy in enumerate(order, start=1):
                position_counts[(policy, position)] += 1

    snapshot_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM model_snapshots WHERE run_id=? AND source='donor_shadow'",
            (run_id,),
        )
        if str(row["participant_id"]) in participant_ids
    ]
    snapshots = {
        (str(row["participant_id"]), int(row["opportunity_budget"])): row
        for row in snapshot_rows
        if str(row.get("budget_kind")) == "eligible_feedback"
    }
    snapshot_summary: dict[str, Any] = {}
    previous_hashes: dict[str, str] = {}
    for budget in q_levels:
        rows = [
            snapshots[(participant_id, budget)]
            for participant_id in participant_ids
            if (participant_id, budget) in snapshots
        ]
        realized = [int(row["eligible_feedbacks"]) for row in rows]
        support = collections.Counter()
        positive_total = 0
        negative_total = 0
        repeated_from_previous = 0
        for row in rows:
            positive = int(row["positive_feedbacks"])
            negative = int(row["negative_feedbacks"])
            positive_total += positive
            negative_total += negative
            if positive and negative:
                support["both_classes"] += 1
            elif positive:
                support["positive_only"] += 1
            elif negative:
                support["negative_only"] += 1
            else:
                support["no_labels"] += 1
            participant_id = str(row["participant_id"])
            model_hash = f"{row['encoder_sha256']}:{row['main_sha256']}"
            if previous_hashes.get(participant_id) == model_hash:
                repeated_from_previous += 1
            previous_hashes[participant_id] = model_hash
        valid = 0
        attained = 0
        for participant_id in participant_ids:
            native = sum(eligible_labels[participant_id].values())
            attained += int(native >= budget)
            row = snapshots.get((participant_id, budget))
            valid += int(row is not None and int(row["eligible_feedbacks"]) == min(native, budget))
        snapshot_summary[str(budget)] = {
            "stored": len(rows),
            "valid_capped": valid,
            "native_attained": attained,
            "realized_feedback": describe(realized),
            "realized_feedback_total": sum(realized),
            "positive_feedback_total": positive_total,
            "negative_feedback_total": negative_total,
            "class_support": dict(support),
            "same_model_as_previous_cap": repeated_from_previous,
        }

    donor_labels = []
    for participant in participants:
        participant_id = str(participant["id"])
        positive = int(eligible_labels[participant_id][1])
        negative = int(eligible_labels[participant_id][0])
        donor_labels.append(
            {
                "assignment_id": assignment_by_id[participant_id],
                "positive": positive,
                "negative": negative,
                "total": positive + negative,
            }
        )

    pair_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM pair_intervals WHERE run_id=?", (run_id,)
        )
        if str(row["participant_id"]) in participant_ids
    ]
    pair_groups: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in pair_rows:
        pair_groups[(str(row["participant_id"]), int(row["day"]))].append(row)
    pair_anomalies = []
    for (participant_id, day), rows in pair_groups.items():
        qualified = [row for row in rows if bool(row.get("qualified"))]
        if len(rows) != 1 or len(qualified) != 1:
            pair_anomalies.append(
                {
                    "assignment_id": assignment_by_id[participant_id],
                    "day": day,
                    "attempts": len(rows),
                    "qualified": len(qualified),
                }
            )

    quality_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM quality_checks WHERE run_id=? ORDER BY created_ms, id", (run_id,)
        )
        if str(row["participant_id"]) in participant_ids
    ]
    quality_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    network_medians: list[float] = []
    network_p95: list[float] = []
    for row in quality_rows:
        session = session_by_id.get(str(row.get("session_id") or ""))
        day = int(session["day"]) if session else 0
        quality_groups[(str(row["participant_id"]), day, str(row["kind"]))].append(row)
        if row["kind"] == "network":
            metrics = parsed(str(row["metrics"]))
            if isinstance(metrics.get("median_ms"), (int, float)):
                network_medians.append(float(metrics["median_ms"]))
            if isinstance(metrics.get("p95_ms"), (int, float)):
                network_p95.append(float(metrics["p95_ms"]))
    retries = []
    failed_final = []
    for (participant_id, day, kind), rows in quality_groups.items():
        if len(rows) > 1:
            retries.append(
                {
                    "assignment_id": assignment_by_id[participant_id],
                    "day": day,
                    "kind": kind,
                    "attempts": len(rows),
                    "passed_sequence": [bool(row["passed"]) for row in rows],
                }
            )
        if not bool(rows[-1]["passed"]):
            failed_final.append(
                {"assignment_id": assignment_by_id[participant_id], "day": day, "kind": kind}
            )

    response_rows = [
        dict(row)
        for row in connection.execute("SELECT * FROM responses WHERE run_id=?", (run_id,))
        if str(row["participant_id"]) in participant_ids
    ]
    response_counts: collections.Counter[str] = collections.Counter(
        str(row["instrument"]) for row in response_rows
    )
    daily_answers: dict[tuple[str, int], dict[str, Any]] = {}
    for row in response_rows:
        match = re.fullmatch(r"daily_day_(\d)", str(row["instrument"]))
        if match:
            daily_answers[(str(row["participant_id"]), int(match.group(1)))] = parsed(
                str(row["answers"])
            )
    questionnaire_keys = (
        "timing_today",
        "effort_today",
        "annoyance_today",
        "predictable_today",
        "speed_today",
        "like_today",
        "fatigue_today",
        "control_today",
    )
    questionnaire_variation = {}
    for key in questionnaire_keys:
        changed = 0
        means = []
        for participant_id in participant_ids:
            values = [daily_answers.get((participant_id, day), {}).get(key) for day in (1, 2, 3)]
            if all(value is not None for value in values) and len(set(values)) > 1:
                changed += 1
        for day in (1, 2, 3):
            means.append(
                describe(
                    daily_answers.get((participant_id, day), {}).get(key)
                    for participant_id in participant_ids
                ).get("mean")
            )
        questionnaire_variation[key] = {
            "day_means": means,
            "participants_with_change": changed,
        }

    session_counts = collections.Counter(
        (int(row["day"]), str(row["client_role"]), str(row["state"])) for row in sessions
    )
    session_duration: dict[tuple[int, str], list[float]] = collections.defaultdict(list)
    for row in sessions:
        if row.get("ended_ms") is not None:
            start = row.get("activated_ms") or row.get("started_ms")
            session_duration[(int(row["day"]), str(row["client_role"]))].append(
                (int(row["ended_ms"]) - int(start)) / 1000
            )
    gaps: dict[str, list[float]] = {"day1_to_day2": [], "day2_to_day3": []}
    for participant_id in participant_ids:
        for day, key in ((1, "day1_to_day2"), (2, "day2_to_day3")):
            before = study_session_by_participant_day.get((participant_id, day))
            after = study_session_by_participant_day.get((participant_id, day + 1))
            if before and after and before.get("ended_ms") is not None:
                gaps[key].append((int(after["started_ms"]) - int(before["ended_ms"])) / 1000)

    coverage_anomalies = []
    expected_exact = {
        "session.client_started": 1,
        "pair.qualified_start": 1,
        "model.ready": 1,
        "topic.selected": 1,
        "phase.start": 8,
        "phase.end": 8,
        "companion.window_start": 3,
        "companion.window_end": 3,
    }
    for participant_id in participant_ids:
        for day in (1, 2, 3):
            counts = coverage[(participant_id, day)]
            for event_type, expected in expected_exact.items():
                if (
                    event_type in {"phase.start", "phase.end"}
                    and day == 1
                    and participant_id in candidate_probe_participants
                ):
                    expected = 11
                if counts[event_type] != expected:
                    coverage_anomalies.append(
                        {
                            "assignment_id": assignment_by_id[participant_id],
                            "day": day,
                            "event_type": event_type,
                            "expected": expected,
                            "actual": counts[event_type],
                        }
                    )
            if counts["phase.transition_ack"] < 9:
                coverage_anomalies.append(
                    {
                        "assignment_id": assignment_by_id[participant_id],
                        "day": day,
                        "event_type": "phase.transition_ack",
                        "expected_minimum": 9,
                        "actual": counts["phase.transition_ack"],
                    }
                )

    missing_results = [
        {"tap_id": tap_id, **row}
        for tap_id, row in tap_pointers.items()
        if tap_id not in tap_results
    ]
    accepted_missing = [row for row in missing_results if row.get("accepted")]
    topic_initial = [
        rows[0]
        for rows in topics.values()
        if rows and rows[0]["event_type"] == "topic.selected"
    ]
    topic_changes = [
        row for rows in topics.values() for row in rows if row["event_type"] == "topic.changed"
    ]
    invalid_topic_changes = [
        row
        for row in topic_changes
        if not row.get("from_topic_id")
        or not row.get("topic_id")
        or row.get("from_topic_id") == row.get("topic_id")
        or row.get("user_initiated") is not True
    ]

    report = {
        "run": {
            "slug": args.run_slug,
            "id": run_id,
            "protocol_version": run["protocol_version"],
            "protocol_mode": run["protocol_mode"],
            "data_origin": run["data_origin"],
            "wave": args.wave,
        },
        "scope": {
            "participants": len(participants),
            "assignment_ids": [row["assignment_id"] for row in participants],
        },
        "integrity": {
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_errors": len(list(connection.execute("PRAGMA foreign_key_check"))),
            "coverage_anomalies": coverage_anomalies,
            "cross_phase_or_policy_interrupts": cross_phase,
            "interrupts_missing_activation_provenance": missing_interrupt_provenance,
            "eligible_provenance_errors": provenance_errors,
        },
        "sessions": {
            "counts": counter_rows(session_counts),
            "duration_seconds": {
                f"day{day}_{role}": describe(values)
                for (day, role), values in sorted(session_duration.items())
            },
            "cross_day_gap_seconds": {key: describe(values) for key, values in gaps.items()},
            "pair_intervals": len(pair_rows),
            "pair_anomalies": pair_anomalies,
        },
        "events": {"total": sum(event_counts.values()), "counts": dict(event_counts.most_common())},
        "taps": {
            "pointer": len(tap_pointers),
            "result": len(tap_results),
            "missing_result": len(missing_results),
            "accepted_missing_result": len(accepted_missing),
            "results_without_pointer": sorted(set(tap_results) - set(tap_pointers)),
            "eligible_positive": sum(labels[1] for labels in eligible_labels.values()),
            "eligible_negative": sum(labels[0] for labels in eligible_labels.values()),
            "quarantine_reasons": dict(quarantine),
        },
        "learning": {
            "pooled_by_day": learning_pooled,
            "equal_participant_summary": learning_summary,
            "paired_day1_day3": learning_tests,
            "participant_day": [
                {
                    "assignment_id": assignment_by_id[participant_id],
                    "day": day,
                    **values,
                }
                for (participant_id, day), values in sorted(
                    learning_rows.items(),
                    key=lambda item: (assignment_number(assignment_by_id[item[0][0]]), item[0][1]),
                )
            ],
        },
        "frozen": {
            "pooled_by_day": frozen_pooled,
            "participant_day": [
                {"assignment_id": assignment_by_id[participant_id], "day": day, **values}
                for (participant_id, day), values in sorted(
                    frozen_rows.items(),
                    key=lambda item: (assignment_number(assignment_by_id[item[0][0]]), item[0][1]),
                )
            ],
        },
        "micro_probe": {
            "revealed_preference_proxy_warning": (
                "Proxy only: an automatic activation is treated as accepted unless interrupted; "
                "manual activation is treated as a false negative. This is not independent ground truth."
            ),
            "order_by_day": {
                str(day): dict(sorted(counter.items())) for day, counter in order_by_day.items()
            },
            "position_counts": counter_rows(position_counts),
            "equal_participant_all_days": probe_summary,
            "pooled_by_day": probe_pooled_by_day,
            "paired_tests": probe_tests,
            "participant_policy_all_days": [
                {
                    "assignment_id": assignment_by_id[participant_id],
                    "policy": policy,
                    **values,
                }
                for (participant_id, policy), values in sorted(
                    probe_per_participant.items(),
                    key=lambda item: (assignment_number(assignment_by_id[item[0][0]]), item[0][1]),
                )
            ],
        },
        "models": {
            "q_levels": q_levels,
            "eligible_labels_by_donor": donor_labels,
            "donors_without_negative": sum(row["negative"] == 0 for row in donor_labels),
            "snapshots": snapshot_summary,
            "prediction_routes": counter_rows(prediction_routes),
            "fallback_predictions": sum(
                count for (requested, effective), count in prediction_routes.items() if requested != effective
            ),
        },
        "quality": {
            "rows": len(quality_rows),
            "participant_day_kind_groups": len(quality_groups),
            "retries": retries,
            "failed_final": failed_final,
            "network_median_ms": describe(network_medians),
            "network_p95_ms": describe(network_p95),
        },
        "topics": {
            "participant_days": len(topics),
            "selected": len(topic_initial),
            "changed": len(topic_changes),
            "invalid_changes": len(invalid_topic_changes),
            "unique_initial_topics": len({row.get("topic_id") for row in topic_initial}),
            "unique_changed_to_topics": len({row.get("topic_id") for row in topic_changes}),
            "changed_to_counts": dict(collections.Counter(str(row.get("topic_id")) for row in topic_changes)),
        },
        "chat_tts": {
            "chat_turns": counter_rows(chat_turns),
            "participant_days_with_primary_chat": len(primary_chat_coverage),
            "primary_chat_count": describe(primary_chat_coverage.values()),
            "tts_generated": counter_rows(tts_generated),
            "tts_latency_ms": describe(tts_latency),
            "playback": counter_rows(playback),
            "tts_server_failures": tts_failures,
            "agent_turn_errors": turn_errors,
        },
        "questionnaires": {
            "counts": dict(response_counts),
            "daily_variation": questionnaire_variation,
        },
        "raw_participant_action_rows": participant_action_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    connection.close()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "participants": len(participants),
                "events": sum(event_counts.values()),
                "accepted_missing_results": len(accepted_missing),
                "coverage_anomalies": len(coverage_anomalies),
                "cross_phase_or_policy_interrupts": len(cross_phase),
                "interrupts_missing_activation_provenance": len(missing_interrupt_provenance),
                "eligible_provenance_errors": len(provenance_errors),
                "fallback_predictions": report["models"]["fallback_predictions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
