"""Analyze a complete 32-profile intent-aware duration run.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    positives = [row for row in records if int(row["label"]) == 1]
    corrected_opportunities = sum(int(row["corrected"]) for row in records)
    correction_taps = sum(int(row.get("correction_count", 0)) for row in records)
    manual_activates = sum(
        int(row.get("manual_activate_count", 0)) for row in records
    )
    manual_interrupts = sum(
        int(row.get("manual_interrupt_count", 0)) for row in records
    )
    primary = [row for row in records if row.get("phase") == "primary"]
    companion = [row for row in records if row.get("phase") == "companion"]
    participant_to_companion = [
        row for row in records if row.get("category") == "participant_to_companion"
    ]
    companion_reply = [
        row for row in records if row.get("category") == "companion_reply"
    ]
    primary_no_reply = [
        row for row in records if row.get("category") == "primary_no_reply"
    ]
    full_utterance_hardness = [
        float(row["full_utterance_shared_error_propensity"])
        for row in records
        if row.get("full_utterance_shared_error_propensity") is not None
    ]
    valid_response_latencies = [
        float(row["first_valid_response_latency_ms"])
        for row in records
        if row.get("first_valid_response_latency_ms") is not None
    ]
    feedback_attributions = [
        str(attribution)
        for row in records
        for attribution in row.get("feedback_attributions", [])
        if attribution is not None
    ]
    acoustic_corrections = feedback_attributions.count(
        "acoustic_personalizer"
    )
    context_corrections = feedback_attributions.count(
        "reply_needed_context"
    )

    def subset_rate(rows: list[dict[str, Any]]) -> float | None:
        return (
            float(
                sum(int(row.get("correction_count", 0)) for row in rows)
                / len(rows)
            )
            if rows
            else None
        )

    return {
        "eligible_opportunities": len(records),
        "corrected_opportunities": corrected_opportunities,
        "corrected_opportunity_rate": (
            corrected_opportunities / len(records) if records else None
        ),
        "corrections": correction_taps,
        "correction_rate": correction_taps / len(records) if records else None,
        "activate_opportunities": len(positives),
        "manual_activates": manual_activates,
        "manual_activate_rate": (
            manual_activates / len(positives)
            if positives
            else None
        ),
        "interrupt_opportunities": len(records),
        "interrupts": manual_interrupts,
        "interrupt_rate": (
            manual_interrupts / len(records)
            if records
            else None
        ),
        "acoustic_personalizer_corrections": acoustic_corrections,
        "acoustic_personalizer_correction_rate": (
            acoustic_corrections / len(records) if records else None
        ),
        "reply_needed_context_corrections": context_corrections,
        "reply_needed_context_correction_rate": (
            context_corrections / len(records) if records else None
        ),
        "wake_rate": (
            sum(
                int(row.get("any_activation", row.get("activated", False)))
                for row in records
            )
            / len(records)
            if records
            else None
        ),
        "primary_correction_rate": subset_rate(primary),
        "companion_correction_rate": subset_rate(companion),
        "primary_no_reply_interrupt_rate": subset_rate(primary_no_reply),
        "participant_to_companion_interrupt_rate": subset_rate(
            participant_to_companion
        ),
        "companion_reply_interrupt_rate": subset_rate(companion_reply),
        "mean_shared_probability": (
            float(
                np.mean(
                    [
                        float(row["shared_probability"])
                        for row in records
                        if row.get("shared_probability") is not None
                    ]
                )
            )
            if any(row.get("shared_probability") is not None for row in records)
            else None
        ),
        "mean_personal_probability": (
            float(
                np.mean(
                    [
                        float(row["personal_probability"])
                        for row in records
                        if row.get("personal_probability") is not None
                    ]
                )
            )
            if any(row.get("personal_probability") is not None for row in records)
            else None
        ),
        "mean_stream_stop_shared_error_propensity": (
            float(
                np.mean(
                    [
                        (
                            1.0 - float(row["shared_probability"])
                            if int(row["label"]) == 1
                            else float(row["shared_probability"])
                        )
                        for row in records
                        if row.get("shared_probability") is not None
                    ]
                )
            )
            if any(row.get("shared_probability") is not None for row in records)
            else None
        ),
        "mean_full_utterance_shared_error_propensity": (
            float(np.mean(full_utterance_hardness))
            if full_utterance_hardness
            else None
        ),
        "mean_text_length": (
            float(np.mean([int(row["text_length"]) for row in records]))
            if records
            else None
        ),
        "valid_response_latency_ms_median": (
            float(np.median(valid_response_latencies))
            if valid_response_latencies
            else None
        ),
        "valid_response_latency_ms_p95": (
            percentile(valid_response_latencies, 0.95)
            if valid_response_latencies
            else None
        ),
    }


def participant_bootstrap(
    records: list[dict[str, Any]],
    value_fn,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_participant[str(row["participant_id"])].append(row)
    ids = sorted(by_participant)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sampled = []
        for _draw in ids:
            sampled.extend(by_participant[rng.choice(ids)])
        values.append(float(value_fn(sampled)))
    return (
        percentile(values, 0.025),
        percentile(values, 0.5),
        percentile(values, 0.975),
    )


def correction_rate(rows: list[dict[str, Any]]) -> float:
    return float(
        sum(int(row.get("correction_count", 0)) for row in rows) / len(rows)
    )


def manual_activate_rate(rows: list[dict[str, Any]]) -> float:
    positives = [row for row in rows if int(row["label"]) == 1]
    return float(
        sum(int(row.get("manual_activate_count", 0)) for row in positives)
        / len(positives)
    )


def interrupt_rate(rows: list[dict[str, Any]]) -> float:
    return float(
        sum(int(row.get("manual_interrupt_count", 0)) for row in rows)
        / len(rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a complete 32-profile intent-aware duration run."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--activation-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-participants-csv", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--equivalence-margin", type=float, default=0.02)
    parser.add_argument("--upper-improvement-margin", type=float, default=0.03)
    parser.add_argument("--consecutive-cycles", type=int, default=3)
    parser.add_argument("--difficulty-deviation-max", type=float, default=0.03)
    parser.add_argument("--tail-manual-activate-rate-max", type=float, default=0.10)
    parser.add_argument("--tail-interrupt-rate-max", type=float, default=0.05)
    parser.add_argument("--tail-correction-rate-max", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    run = json.loads(args.input.read_text(encoding="utf-8"))
    activation_audit = json.loads(
        args.activation_audit.read_text(encoding="utf-8")
    )
    if run.get("duration_evidence_valid") is not True:
        raise SystemExit("Run is not a real-time 32-profile duration fixture")
    if (
        activation_audit.get("schema")
        != "tap-to-adapt-server-eou-activation-audit/v1"
        or activation_audit.get("run_slug") != run.get("run_slug")
        or int(activation_audit.get("participant_count", 0)) != 32
        or int(activation_audit.get("eou_gate_violations", -1)) != 0
        or float(activation_audit.get("minimum_observed_vad_silence_ms", -1))
        < 400
    ):
        raise SystemExit("Server-side EOU activation audit did not pass")
    if run.get("failures") or int(run.get("participants_succeeded", 0)) != 32:
        raise SystemExit("All 32 profiles must complete before duration analysis")
    protocol = dict(run["protocol"])
    if (
        int(protocol["target_voice_count"]) != 32
        or int(protocol["primary_minutes"]) != 4
        or int(protocol["companion_minutes"]) != 2
        or int(protocol["max_minutes"]) != 60
        or int(protocol.get("smoke_cycles", -1)) != 0
        or float(run.get("clock_scale", 0)) != 1.0
        or protocol.get("frozen_pcm_bundle_used") is not True
        or protocol.get("stimulus_difficulty_score_mode")
        != "full_utterance_shared_only_no_truncation"
    ):
        raise SystemExit(
            "Run does not satisfy the locked real-time 32-voice, 60-minute 4+2 protocol"
        )
    agent_manifest = dict(run.get("agent_reply_manifest") or {})
    if (
        int(agent_manifest.get("rows", 0)) != 2560
        or agent_manifest.get("provider_order") != ["z-ai"]
        or agent_manifest.get("allow_fallbacks") is not False
        or not agent_manifest.get("rows_sha256")
    ):
        raise SystemExit("Run lacks the locked frozen GLM-5.2 reply manifest")
    records = [
        row
        for participant in run["results"]
        for row in participant["records"]
    ]
    if any(
        row.get("schema") != "tap-to-adapt-factorized-eou-opportunity/v4"
        or row.get("eou_gate_violation") is not False
        or row.get("opposite_label_conflict") is not False
        or int(row.get("post_speech_silence_frame_ms", 0)) != 200
        or not 1000 <= int(row.get("post_speech_silence_ms_sent", 0)) <= 3000
        or int(row.get("post_speech_silence_frames_sent", -1))
        != int(row.get("post_speech_silence_frames_planned", -2))
        or int(row.get("premature_activation_count", 0)) != 0
        or int(row.get("manual_activate_count", 0))
        + int(row.get("manual_interrupt_count", 0))
        > 1
        or row.get("agent_reply_mode") != "frozen_remote_recorded"
        or row.get("agent_reply_manifest_rows_sha256")
        != agent_manifest["rows_sha256"]
        for row in records
    ):
        raise SystemExit(
            "Run lacks factorized EOU evidence, real 200 ms post-TTS silence, or one-label integrity"
        )
    if len(records) != 32 * 10 * 12:
        raise SystemExit("A valid 60-minute run must contain exactly 3,840 opportunities")
    context_feedback_rows = [
        row
        for row in records
        if "reply_needed_context" in row.get("feedback_attributions", [])
    ]
    if any(
        row.get("acoustic_training_applied") is not False
        for row in context_feedback_rows
    ):
        raise SystemExit("Context feedback leaked into the acoustic personal head")
    valid_latencies = [
        float(row["first_valid_response_latency_ms"])
        for row in records
        if row.get("first_valid_response_latency_ms") is not None
    ]
    if any(value <= 0 or value > 3000 for value in valid_latencies):
        raise SystemExit("A valid response falls outside the 1-3 s observation tail")
    automatic_activation_records = sum(
        bool(row.get("any_activation"))
        and int(row.get("manual_activate_count", 0)) == 0
        for row in records
    )
    if automatic_activation_records != int(
        activation_audit["automatic_activations"]
    ):
        raise SystemExit(
            "Server-side activation audit does not match the duration report"
        )
    immutable_fingerprint_fields = (
        "model_family",
        "revision",
        "release_manifest_sha256",
        "shared_head_artifact_sha256",
        "shared_head_fingerprint",
        "replay_per_class",
        "participant_disjoint_training",
    )
    fingerprints = [
        tuple(
            participant["model_fingerprint"].get(field)
            for field in immutable_fingerprint_fields
        )
        for participant in run["results"]
    ]
    if (
        len(set(fingerprints)) != 1
        or fingerprints[0][-1] is not True
        or int(fingerprints[0][-2]) != 12
    ):
        raise SystemExit("Model release/replay fingerprint changed across profiles")
    if any(
        int(participant.get("pair_reconnect_count", 0)) < 2
        or int(participant.get("timing_reconnect_count", 0)) < 6
        or any(
            audit.get("runtime_instance_id")
            != participant.get("runtime_instance_id")
            or int(audit.get("model_update_count_after", -1))
            < int(audit.get("model_update_count_before", 0))
            for audit in participant.get("timing_reconnect_audits", [])
        )
        for participant in run["results"]
    ):
        raise SystemExit(
            "Public WebSocket lifecycle audit is incomplete or rolled back a personal head"
        )
    if any(
        row.get("full_utterance_shared_error_propensity") is None
        for row in records
    ):
        raise SystemExit("Run lacks independent full-utterance difficulty scores")
    participant_rows = []
    for participant in sorted(
        run["results"], key=lambda row: int(row["participant_index"])
    ):
        selected = list(participant["records"])
        selected_first = [
            row for row in selected if int(row["cycle_index"]) <= 3
        ]
        participant_max_cycle = max(
            int(row["cycle_index"]) for row in selected
        )
        selected_last = [
            row
            for row in selected
            if int(row["cycle_index"]) >= participant_max_cycle - 2
        ]
        first_rate = correction_rate(selected_first)
        last_rate = correction_rate(selected_last)
        participant_rows.append(
            {
                "participant_index": int(participant["participant_index"]),
                "profile_id": participant["profile"]["profile_id"],
                "voice_id": int(participant["voice_id"]),
                "companion_voice": participant["companion_voice"],
                **summarize(selected),
                "first_18m_correction_rate": first_rate,
                "last_18m_correction_rate": last_rate,
                "relative_correction_reduction_first_to_last": (
                    1.0 - last_rate / first_rate if first_rate > 0 else None
                ),
                "wall_elapsed_seconds": float(participant["wall_elapsed_seconds"]),
            }
        )
    cycles = sorted({int(row["cycle_index"]) for row in records})
    expected = 32 * 12
    cycle_rows = []
    for cycle in cycles:
        selected = [row for row in records if int(row["cycle_index"]) == cycle]
        if len(selected) != expected:
            raise SystemExit(f"Cycle {cycle} has {len(selected)} rather than {expected} events")
        band_counts = {
            band: sum(row["difficulty_band"] == band for row in selected)
            for band in ("low", "medium", "high")
        }
        if set(band_counts.values()) != {32 * 4}:
            raise SystemExit(f"Cycle {cycle} difficulty strata are imbalanced")
        summary = summarize(selected)
        low, median, high = participant_bootstrap(
            selected,
            correction_rate,
            iterations=args.bootstrap,
            seed=20260726 + cycle,
        )
        cycle_rows.append(
            {
                "cycle_index": cycle,
                "start_minute": (cycle - 1) * 6,
                "end_minute": cycle * 6,
                **summary,
                "correction_rate_bootstrap_low": low,
                "correction_rate_bootstrap_median": median,
                "correction_rate_bootstrap_high": high,
                "difficulty_low_n": band_counts["low"],
                "difficulty_medium_n": band_counts["medium"],
                "difficulty_high_n": band_counts["high"],
            }
        )
    terminal_cycles = set(cycles[-3:])
    terminal_records = [
        row for row in records if int(row["cycle_index"]) in terminal_cycles
    ]
    terminal_rate = correction_rate(terminal_records)
    for row in cycle_rows:
        cycle = int(row["cycle_index"])

        def improvement(sampled: list[dict[str, Any]]) -> float:
            current = [
                item for item in sampled if int(item["cycle_index"]) == cycle
            ]
            terminal = [
                item
                for item in sampled
                if int(item["cycle_index"]) in terminal_cycles
            ]
            return correction_rate(current) - correction_rate(terminal)

        low, median, high = participant_bootstrap(
            records,
            improvement,
            iterations=args.bootstrap,
            seed=20260800 + cycle,
        )
        row["remaining_improvement_bootstrap_low"] = low
        row["remaining_improvement_bootstrap_median"] = median
        row["remaining_improvement_bootstrap_high"] = high

    shared_hardness = np.asarray(
        [
            float(row["mean_full_utterance_shared_error_propensity"])
            for row in cycle_rows
        ]
    )
    grand_hardness = float(np.mean(shared_hardness))
    max_hardness_deviation = float(np.max(np.abs(shared_hardness - grand_hardness)))
    difficulty_guard_passed = max_hardness_deviation <= args.difficulty_deviation_max

    practical_candidate: int | None = None
    precision_candidate: int | None = None
    successful_fit_entry_minute: int | None = None
    successful_fit_confirmation_minute: int | None = None
    for index, row in enumerate(cycle_rows):
        tail = cycle_rows[index:]
        if len(tail) < args.consecutive_cycles:
            continue
        confirmation = tail[: args.consecutive_cycles]
        stable = all(
            abs(float(item["correction_rate"]) - terminal_rate)
            <= args.equivalence_margin
            for item in confirmation
        ) and all(
            abs(float(item["correction_rate"]) - terminal_rate)
            <= args.equivalence_margin
            for item in tail
        )
        if stable and practical_candidate is None:
            practical_candidate = int(row["start_minute"])
        if (
            stable
            and float(row["remaining_improvement_bootstrap_high"])
            <= args.upper_improvement_margin
            and difficulty_guard_passed
            and precision_candidate is None
        ):
            precision_candidate = int(row["start_minute"])
        tail_adequate = all(
            float(item["manual_activate_rate"])
            <= args.tail_manual_activate_rate_max
            and float(item["interrupt_rate"]) <= args.tail_interrupt_rate_max
            and float(item["correction_rate"])
            <= args.tail_correction_rate_max
            for item in tail
        )
        if (
            stable
            and float(row["remaining_improvement_bootstrap_high"])
            <= args.upper_improvement_margin
            and difficulty_guard_passed
            and tail_adequate
            and successful_fit_entry_minute is None
        ):
            successful_fit_entry_minute = int(row["start_minute"])
            successful_fit_confirmation_minute = int(
                confirmation[-1]["end_minute"]
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cycle_rows[0]))
        writer.writeheader()
        writer.writerows(cycle_rows)
    with args.output_participants_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(participant_rows[0]))
        writer.writeheader()
        writer.writerows(participant_rows)
    first = [row for row in records if int(row["cycle_index"]) <= 3]
    last = terminal_records
    terminal_summary = summarize(last)
    tail_adequacy_passed = bool(
        float(terminal_summary["manual_activate_rate"] or 0)
        <= args.tail_manual_activate_rate_max
        and float(terminal_summary["interrupt_rate"] or 0)
        <= args.tail_interrupt_rate_max
        and float(terminal_summary["correction_rate"] or 0)
        <= args.tail_correction_rate_max
    )
    overall_bootstrap = participant_bootstrap(
        records,
        correction_rate,
        iterations=args.bootstrap,
        seed=20260901,
    )
    first_last_change = {}
    for name, rate_fn, seed in (
        ("all_corrections", correction_rate, 20260911),
        ("manual_activate", manual_activate_rate, 20260912),
        ("manual_interrupt", interrupt_rate, 20260913),
    ):
        first_rate = rate_fn(first)
        last_rate = rate_fn(last)

        def reduction(sampled: list[dict[str, Any]]) -> float:
            sampled_first = [
                row for row in sampled if int(row["cycle_index"]) <= 3
            ]
            sampled_last = [
                row
                for row in sampled
                if int(row["cycle_index"]) in terminal_cycles
            ]
            return rate_fn(sampled_first) - rate_fn(sampled_last)

        low, median, high = participant_bootstrap(
            records,
            reduction,
            iterations=args.bootstrap,
            seed=seed,
        )
        first_last_change[name] = {
            "first_18m_rate": first_rate,
            "last_18m_rate": last_rate,
            "absolute_reduction": first_rate - last_rate,
            "absolute_reduction_participant_bootstrap_ci95": {
                "low": low,
                "median": median,
                "high": high,
            },
            "relative_reduction": (
                1.0 - last_rate / first_rate if first_rate > 0 else None
            ),
        }
    improved_profiles = sum(
        float(row["last_18m_correction_rate"])
        < float(row["first_18m_correction_rate"])
        for row in participant_rows
    )
    unchanged_profiles = sum(
        math.isclose(
            float(row["last_18m_correction_rate"]),
            float(row["first_18m_correction_rate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in participant_rows
    )
    worsened_profiles = len(participant_rows) - improved_profiles - unchanged_profiles
    report = {
        "schema": "tap-to-adapt-factorized-eou-duration-analysis/v4",
        "simulation_only": True,
        "human_effect_evidence": False,
        "run_slug": run["run_slug"],
        "participants": 32,
        "distinct_target_voices": 32,
        "distinct_companion_voices": int(protocol["companion_voice_count"]),
        "llm_provider": run["llm_provider"],
        "llm_model": run["llm_model"],
        "tts_provider": run["tts_provider"],
        "tts_model": run["tts_model"],
        "model": run["model"],
        "engineering_integrity": {
            "opportunities": len(records),
            "eou_gate_violations": 0,
            "server_eou_activation_audit_rows_sha256": activation_audit[
                "rows_sha256"
            ],
            "minimum_observed_vad_silence_ms": float(
                activation_audit["minimum_observed_vad_silence_ms"]
            ),
            "opposite_label_conflicts": 0,
            "context_to_acoustic_training_leaks": 0,
            "frozen_agent_reply_manifest_rows_sha256": agent_manifest[
                "rows_sha256"
            ],
            "pair_reconnects_total": int(run["pair_reconnects_total"]),
            "timing_reconnects_total": int(run["timing_reconnects_total"]),
            "runtime_uuid_changes": 0,
            "personal_head_rollbacks": 0,
            "valid_response_latency_ms_median": (
                float(np.median(valid_latencies)) if valid_latencies else None
            ),
            "valid_response_latency_ms_p95": (
                percentile(valid_latencies, 0.95) if valid_latencies else None
            ),
        },
        "cycle_minutes": 6,
        "maximum_observed_minutes": int(protocol["max_minutes"]),
        "overall": summarize(records),
        "overall_correction_rate_participant_bootstrap_ci95": {
            "low": overall_bootstrap[0],
            "median": overall_bootstrap[1],
            "high": overall_bootstrap[2],
        },
        "participant_correction_rate_mean": float(
            np.mean([row["correction_rate"] for row in participant_rows])
        ),
        "participant_correction_rate_sd": float(
            np.std(
                [row["correction_rate"] for row in participant_rows],
                ddof=1,
            )
        ),
        "first_18_minutes": summarize(first),
        "last_18_minutes": terminal_summary,
        "first_to_last_changes": first_last_change,
        "relative_correction_reduction_first_to_last": (
            1.0 - correction_rate(last) / correction_rate(first)
            if correction_rate(first) > 0
            else None
        ),
        "participant_direction_first_to_last": {
            "improved": improved_profiles,
            "unchanged": unchanged_profiles,
            "worsened": worsened_profiles,
            "comparison_windows": "first 18 minutes versus last 18 minutes",
        },
        "terminal_cycle_rate": terminal_rate,
        "terminal_cycles": sorted(terminal_cycles),
        "descriptive_practical_plateau_minute": practical_candidate,
        "precision_qualified_plateau_minute": precision_candidate,
        "successful_fit_entry_minute": successful_fit_entry_minute,
        "successful_fit_confirmation_minute": successful_fit_confirmation_minute,
        "tail_adequacy": {
            "window_minutes": 18,
            "manual_activate_rate_max": args.tail_manual_activate_rate_max,
            "interrupt_rate_max": args.tail_interrupt_rate_max,
            "correction_rate_max": args.tail_correction_rate_max,
            "passed": tail_adequacy_passed,
        },
        "stimulus_balance": {
            "score_mode": "full_utterance_shared_only_no_truncation",
            "fixed_per_cycle_difficulty_counts": {
                "low": 128,
                "medium": 128,
                "high": 128,
            },
            "mean_full_utterance_shared_error_propensity": grand_hardness,
            "max_cycle_deviation": max_hardness_deviation,
            "allowed_max_cycle_deviation": args.difficulty_deviation_max,
            "guard_passed": difficulty_guard_passed,
        },
        "plateau_rule": {
            "cycle_minutes": 6,
            "absolute_rate_equivalence_margin": args.equivalence_margin,
            "remaining_improvement_upper_ci_margin": args.upper_improvement_margin,
            "consecutive_cycles": args.consecutive_cycles,
            "requires_no_later_equivalence_departure": True,
            "terminal_reference_cycles": 3,
            "bootstrap_unit": "simulated_participant_profile",
            "bootstrap_iterations": args.bootstrap,
            "requires_stimulus_balance_guard": True,
            "requires_absolute_tail_adequacy": True,
        },
        "interpretation": (
            "A precision-qualified minute remains null unless the participant-level "
            "uncertainty and stimulus-balance guards both pass. This is engineering "
            "evidence for selecting a human pre-pilot duration, not a human effect."
        ),
        "cycles": cycle_rows,
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["end_minute"]) for row in cycle_rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(
        x,
        [float(row["correction_rate"]) for row in cycle_rows],
        marker="o",
        label="All eligible corrections",
    )
    axes[0].fill_between(
        x,
        [float(row["correction_rate_bootstrap_low"]) for row in cycle_rows],
        [float(row["correction_rate_bootstrap_high"]) for row in cycle_rows],
        alpha=0.18,
        label="Participant-profile bootstrap 95% CI",
    )
    axes[0].plot(
        x,
        [float(row["manual_activate_rate"]) for row in cycle_rows],
        marker=".",
        label="Manual activate",
    )
    axes[0].plot(
        x,
        [float(row["interrupt_rate"]) for row in cycle_rows],
        marker=".",
        label="Manual interrupt",
    )
    axes[0].axhline(terminal_rate, color="black", linestyle="--", alpha=0.5)
    axes[0].axhline(
        args.tail_correction_rate_max,
        color="#5f7f55",
        linestyle=":",
        alpha=0.7,
        label="Tail total-burden gate",
    )
    if practical_candidate is not None:
        axes[0].axvline(
            practical_candidate,
            linestyle=":",
            color="#9b3d3d",
            label="Retrospective plateau entry",
        )
    if successful_fit_confirmation_minute is not None:
        axes[0].axvline(
            successful_fit_confirmation_minute,
            linestyle="--",
            color="#6f4b8b",
            label="First confirmable duration",
        )
    axes[0].set_ylabel("Rate")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(
        x,
        [
            float(row["mean_full_utterance_shared_error_propensity"])
            for row in cycle_rows
        ],
        marker="o",
        label="Full-utterance immutable shared-head error propensity",
    )
    axes[1].axhline(grand_hardness, color="black", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Stimulus difficulty proxy")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        x,
        [float(row["mean_shared_probability"]) for row in cycle_rows],
        marker="o",
        label="Shared probability",
    )
    axes[2].plot(
        x,
        [float(row["mean_personal_probability"]) for row in cycle_rows],
        marker="o",
        label="Personal probability",
    )
    axes[2].set_ylabel("Mean wake probability")
    axes[2].set_xlabel("Elapsed interaction time (minutes)")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Intent-aware 4+2 duration simulation: 32 fixed voice/profile clusters")
    fig.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=180)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
