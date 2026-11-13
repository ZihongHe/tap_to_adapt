"""Build a voice-calibrated 32-profile schedule from a compressed preflight.

Only full-utterance frozen shared-head probabilities are used.  They are
computed from the exact frozen PCM without streaming truncation, personal
updates, or activation-state effects.  For each fixed voice/profile, all
12 utterances in each within-cycle slot are retained exactly once.  A
deterministic load-balancing assignment distributes hard and easy voice-text
pairs over the 12 cycles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

try:
    from .smooth_simulation_protocol import (
        RUN_SLUG_PATTERN,
        SCHEMA,
        validate_schedule,
    )
except ImportError:
    from smooth_simulation_protocol import RUN_SLUG_PATTERN, SCHEMA, validate_schedule


PROFILES = 32
CYCLES = 12
SLOTS = 12
UTTERANCES_PER_SLOT = 12
MAX_PREDICTED_HARDNESS_DEVIATION = 0.03


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shared_error_propensity(record: dict[str, Any]) -> float:
    probability = float(record["shared_probability"])
    return 1.0 - probability if int(record["label"]) == 1 else probability


def records_from_shared_scores(
    *,
    participant_index: int,
    events: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for event in events:
        role = str(event["speaker_role"])
        text = str(event["text"])
        cache_key = hashlib.sha256(f"{role}\0{text}".encode("utf-8")).hexdigest()
        score = scores.get(cache_key)
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if (
            score is None
            or int(score["participant_index"]) != participant_index
            or score["speaker_role"] != role
            or score["text_sha256"] != text_sha256
        ):
            raise ValueError(
                f"Full-utterance shared score is absent for profile {participant_index}"
            )
        records.append(
            {
                "opportunity_index": int(event["opportunity_index"]),
                "label": int(event["label"]),
                "text_sha256": text_sha256,
                "shared_probability": float(score["shared_probability"]),
            }
        )
    return records


def select_slot_candidates(
    events: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[int, list[tuple[float, dict[str, Any]]]]:
    if len(events) != CYCLES * SLOTS or len(records) != CYCLES * SLOTS:
        raise ValueError("Calibration requires 144 events and records per profile")
    event_by_opportunity = {
        int(row["opportunity_index"]): row for row in events
    }
    grouped: dict[int, list[tuple[float, dict[str, Any]]]] = {
        slot: [] for slot in range(SLOTS)
    }
    for record in records:
        opportunity = int(record["opportunity_index"])
        event = event_by_opportunity[opportunity]
        if str(record["text_sha256"]) != hashlib.sha256(
            str(event["text"]).encode("utf-8")
        ).hexdigest():
            raise ValueError("Preflight record and turn text hashes do not match")
        slot = (opportunity - 1) % SLOTS
        grouped[slot].append((shared_error_propensity(record), event))
    selected: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for slot, candidates in grouped.items():
        if len(candidates) != CYCLES:
            raise ValueError(f"Slot {slot} does not contain twelve candidates")
        selected[slot] = sorted(
            candidates, key=lambda item: (item[0], str(item[1]["text"]))
        )
    return selected


def balance_slot_candidates(
    selected: dict[int, list[tuple[float, dict[str, Any]]]],
) -> tuple[dict[tuple[int, int], tuple[float, dict[str, Any]]], list[float]]:
    """Assign every slot candidate once while flattening each profile's cycles."""
    totals = [0.0] * CYCLES
    assignments: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    slot_order = sorted(
        range(SLOTS),
        key=lambda slot: (
            -statistics.pstdev(value for value, _ in selected[slot]),
            slot,
        ),
    )
    for slot in slot_order:
        candidates = sorted(
            selected[slot],
            key=lambda item: (-item[0], str(item[1]["text"])),
        )
        cycles = sorted(range(CYCLES), key=lambda cycle: (totals[cycle], cycle))
        for cycle, candidate in zip(cycles, candidates, strict=True):
            assignments[(cycle, slot)] = candidate
            totals[cycle] += candidate[0]
    return assignments, [value / SLOTS for value in totals]


def build_profile_schedule(
    *,
    participant_index: int,
    selected: dict[int, list[tuple[float, dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[float]]:
    assignments, hardness = balance_slot_candidates(selected)
    events: list[dict[str, Any]] = []
    for cycle in range(CYCLES):
        for slot in range(SLOTS):
            value, source = assignments[(cycle, slot)]
            row = dict(source)
            row.update(
                {
                    "opportunity_index": len(events) + 1,
                    "cycle_index": cycle + 1,
                    "elapsed_seconds": cycle * 360 + 15 + slot * 30,
                    "phase_minute_start": cycle * 6
                    + (0 if row["phase"] == "primary" else 4),
                    "voice_calibrated": True,
                    "calibration_shared_error_propensity": value,
                }
            )
            events.append(row)
    validate_schedule(events, max_minutes=72)
    return events, hardness


def cycle_candidates(
    events: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[tuple[float, list[dict[str, Any]]]]:
    """Score whole six-minute cycles so coherent dialogue is never split."""
    if len(events) != CYCLES * SLOTS or len(records) != CYCLES * SLOTS:
        raise ValueError("Calibration requires 144 events and records per profile")
    records_by_opportunity = {
        int(row["opportunity_index"]): row for row in records
    }
    candidates = []
    for cycle in range(1, CYCLES + 1):
        rows = [
            dict(row) for row in events if int(row["cycle_index"]) == cycle
        ]
        if len(rows) != SLOTS:
            raise ValueError(f"Cycle {cycle} is incomplete")
        values = []
        for row in rows:
            record = records_by_opportunity[int(row["opportunity_index"])]
            expected_hash = hashlib.sha256(
                str(row["text"]).encode("utf-8")
            ).hexdigest()
            if str(record["text_sha256"]) != expected_hash:
                raise ValueError("Preflight record and turn text hashes do not match")
            value = shared_error_propensity(record)
            row["calibration_shared_error_propensity"] = value
            values.append(value)
        candidates.append((statistics.mean(values), rows))
    return candidates


def balance_whole_cycles(
    profiles: dict[int, list[tuple[float, list[dict[str, Any]]]]]
) -> tuple[
    dict[int, dict[int, tuple[float, list[dict[str, Any]]]]],
    list[float],
]:
    """Load-balance immutable whole cycles across final time positions."""
    totals = [0.0] * CYCLES
    assignments: dict[
        int, dict[int, tuple[float, list[dict[str, Any]]]]
    ] = {}
    profile_order = sorted(
        profiles,
        key=lambda participant_index: (
            -statistics.pstdev(
                value for value, _rows in profiles[participant_index]
            ),
            participant_index,
        ),
    )
    for participant_index in profile_order:
        candidates = sorted(
            profiles[participant_index],
            key=lambda item: (
                -item[0],
                str(item[1][0].get("topic_id") or ""),
            ),
        )
        target_cycles = sorted(range(CYCLES), key=lambda cycle: (totals[cycle], cycle))
        assignments[participant_index] = {}
        for target_cycle, candidate in zip(
            target_cycles, candidates, strict=True
        ):
            assignments[participant_index][target_cycle] = candidate
            totals[target_cycle] += candidate[0]
    return assignments, [value / len(profiles) for value in totals]


def materialize_whole_cycle_schedule(
    assignment: dict[int, tuple[float, list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for target_cycle in range(CYCLES):
        value, source_rows = assignment[target_cycle]
        for slot, source in enumerate(source_rows):
            row = dict(source)
            row.update(
                {
                    "opportunity_index": len(events) + 1,
                    "cycle_index": target_cycle + 1,
                    "elapsed_seconds": target_cycle * 360 + 15 + slot * 30,
                    "phase_minute_start": target_cycle * 6
                    + (0 if row["phase"] == "primary" else 4),
                    "voice_calibrated": True,
                    "calibration_cycle_shared_error_propensity": value,
                    "whole_cycle_reordered": True,
                }
            )
            events.append(row)
    validate_schedule(events, max_minutes=72)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-run", type=Path, required=True)
    parser.add_argument("--preflight-turns", type=Path, required=True)
    parser.add_argument("--shared-score-map", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    preflight = json.loads(args.preflight_run.read_text(encoding="utf-8"))
    turns = json.loads(args.preflight_turns.read_text(encoding="utf-8"))
    shared_scores = json.loads(args.shared_score_map.read_text(encoding="utf-8"))
    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = list(csv.DictReader(handle))
    if (
        preflight.get("simulation_only") is not True
        or preflight.get("duration_evidence_valid") is not False
        or preflight.get("cohort") != "duration32"
        or int(preflight.get("participants_succeeded", 0)) != PROFILES
        or preflight.get("failures")
        or int((preflight.get("protocol") or {}).get("smoke_cycles", 0)) != 1
        or int(
            (preflight.get("protocol") or {}).get(
                "opportunities_per_profile", 0
            )
        )
        != SLOTS
        or any(
            int(row.get("opportunities", 0)) != SLOTS
            for row in preflight.get("results", [])
        )
    ):
        raise SystemExit(
            "Input must be a complete one-cycle compressed duration32 preflight"
        )
    if (
        turns.get("schema") != SCHEMA
        or turns.get("cohort") != "duration32"
        or int(turns.get("max_minutes", 0)) != 72
    ):
        raise SystemExit("Preflight turns are not the locked 72-minute protocol")
    if (
        shared_scores.get("schema")
        != "tap-to-adapt-full-utterance-shared-score-map/v1"
        or shared_scores.get("simulation_only") is not True
        or shared_scores.get("duration_evidence") is not False
        or shared_scores.get("scoring_mode")
        != "full_utterance_shared_only_no_truncation"
    ):
        raise SystemExit("Shared score map is not a full-utterance audit artifact")
    if len(roster) != PROFILES:
        raise SystemExit("Final roster must contain 32 profiles")
    run_slugs = {row["run_slug"] for row in roster}
    if len(run_slugs) != 1 or RUN_SLUG_PATTERN.fullmatch(next(iter(run_slugs))) is None:
        raise SystemExit("Final roster is outside the smooth simulation namespace")

    preflight_results = {
        int(row["participant_index"]): row for row in preflight["results"]
    }
    preflight_turns = {
        int(row["participant_index"]): row for row in turns["participants"]
    }
    final_rows = {int(row["participant_index"]): row for row in roster}
    scores_by_participant: dict[int, dict[str, dict[str, Any]]] = {
        participant_index: {} for participant_index in range(1, PROFILES + 1)
    }
    for row in shared_scores["rows"]:
        participant_index = int(row["participant_index"])
        cache_key = str(row["cache_key"])
        if (
            participant_index not in scores_by_participant
            or cache_key in scores_by_participant[participant_index]
        ):
            raise SystemExit("Shared score map has invalid or duplicate rows")
        scores_by_participant[participant_index][cache_key] = row
    if (
        set(preflight_results)
        != set(range(1, PROFILES + 1))
        or set(preflight_turns) != set(preflight_results)
        or set(final_rows) != set(preflight_results)
    ):
        raise SystemExit("Profile indices are incomplete or inconsistent")

    profile_candidates: dict[
        int, list[tuple[float, list[dict[str, Any]]]]
    ] = {}
    retained_text_counts: list[int] = []
    for participant_index in range(1, PROFILES + 1):
        result = preflight_results[participant_index]
        source = preflight_turns[participant_index]
        final = final_rows[participant_index]
        if (
            int(final["voice_id"]) != int(result["voice_id"])
        ):
            raise SystemExit("Final and preflight voices must match by profile index")
        calibration_records = records_from_shared_scores(
            participant_index=participant_index,
            events=source["events"],
            scores=scores_by_participant[participant_index],
        )
        profile_candidates[participant_index] = cycle_candidates(
            source["events"], calibration_records
        )
        retained_text_counts.append(
            len({str(event["text"]) for event in source["events"]})
        )

    assignments, predicted_hardness = balance_whole_cycles(
        profile_candidates
    )
    participants = []
    for participant_index in range(1, PROFILES + 1):
        source = preflight_turns[participant_index]
        final = final_rows[participant_index]
        events = materialize_whole_cycle_schedule(
            assignments[participant_index]
        )
        participants.append(
            {
                "participant_id": final["participant_id"],
                "participant_index": participant_index,
                "profile": source["profile"],
                "events": events,
            }
        )

    grand = float(statistics.mean(predicted_hardness))
    max_deviation = max(abs(value - grand) for value in predicted_hardness)
    if max_deviation > MAX_PREDICTED_HARDNESS_DEVIATION:
        raise SystemExit(
            f"Voice-calibrated schedule still drifts by {max_deviation:.6f}"
        )

    payload = dict(turns)
    payload.update(
        {
            "run_slug": next(iter(run_slugs)),
            "participants": participants,
            "difficulty_control": (
                "Every coherent six-minute dialogue cycle remains intact and "
                "preserves its 4+2 intent, role and difficulty-band pattern. All "
                "12 preflight cycles per fixed participant voice are retained "
                "exactly once, then whole cycles are deterministically load-balanced "
                "over time using only full-utterance probabilities from the immutable "
                "shared head and exact frozen PCM. No utterance is moved across its "
                "topic cycle; streaming truncation, personal predictions, corrections "
                "and training outcomes are not used."
            ),
            "voice_calibration": {
                "schema": "tap-to-adapt-voice-calibrated-stimuli/v1",
                "preflight_run_slug": preflight["run_slug"],
                "preflight_run_sha256": sha256(args.preflight_run),
                "preflight_turns_sha256": sha256(args.preflight_turns),
                "shared_score_map_sha256": sha256(args.shared_score_map),
                "pcm_bundle_aggregate_manifest_sha256": shared_scores[
                    "bundle_aggregate_manifest_sha256"
                ],
                "release_manifest_sha256": shared_scores[
                    "release_manifest_sha256"
                ],
                "shared_head_sha256": shared_scores["shared_head_sha256"],
                "full_utterance_shared_probability_only": True,
                "streaming_truncation_used": False,
                "personal_outcomes_used": False,
                "retained_whole_cycles_per_profile": CYCLES,
                "cycle_dialogue_integrity_preserved": True,
                "repeated_opportunities_per_profile": 0,
                "unique_retained_texts_per_profile_min": min(retained_text_counts),
                "unique_retained_texts_per_profile_max": max(retained_text_counts),
                "predicted_cycle_shared_error_propensity": predicted_hardness,
                "predicted_grand_shared_error_propensity": grand,
                "predicted_max_cycle_deviation": max_deviation,
                "required_max_cycle_deviation": (
                    MAX_PREDICTED_HARDNESS_DEVIATION
                ),
            },
        }
    )
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "run_slug": payload["run_slug"],
                "participants": len(participants),
                "events_per_participant": len(participants[0]["events"]),
                "predicted_cycle_shared_error_propensity": predicted_hardness,
                "predicted_max_cycle_deviation": max_deviation,
                "unique_retained_texts_per_profile_min": min(retained_text_counts),
                "unique_retained_texts_per_profile_max": max(retained_text_counts),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
