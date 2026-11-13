"""Audit the frozen GLM intent-dialogue fixture before any account is used."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED = (True, True, False, True, True, False, True, True)
COMPANION_ROLES = ("participant", "companion", "participant", "companion")


def text_length(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema") != "tap-to-adapt-glm-intent-dialogues/v1":
        raise SystemExit("Unexpected fixture schema")
    participants = list(payload.get("participants") or [])
    if len(participants) != 40:
        raise SystemExit("Fixture must contain exactly 8 development and 32 duration profiles")
    expected_ids = {
        *(f"DV{index:02d}" for index in range(1, 9)),
        *(f"PR{index:02d}" for index in range(1, 33)),
    }
    if {str(row["profile_id"]) for row in participants} != expected_ids:
        raise SystemExit("Profile IDs are incomplete or duplicated")

    total_primary = 0
    total_companion = 0
    total_expected = 0
    all_texts: list[str] = []
    per_profile: list[dict[str, Any]] = []
    duplicate_profiles: dict[str, list[str]] = {}
    for participant in participants:
        cycles = list(participant.get("cycles") or [])
        if len(cycles) != 12:
            raise SystemExit(f"{participant['profile_id']} does not contain 12 cycles")
        seen: set[str] = set()
        occurrence_signatures: dict[str, set[tuple[str, str, bool]]] = {}
        cycle_means: list[float] = []
        for cycle_index, cycle in enumerate(cycles, start=1):
            if int(cycle["cycle_index"]) != cycle_index:
                raise SystemExit("Cycle indices are not sequential")
            primary = list(cycle.get("primary") or [])
            companion = list(cycle.get("companion") or [])
            if len(primary) != 8 or len(companion) != 4:
                raise SystemExit("Every cycle must contain 8 primary and 4 companion turns")
            if tuple(bool(row["response_expected"]) for row in primary) != EXPECTED:
                raise SystemExit("Primary response-intent pattern changed")
            if tuple(str(row["speaker_role"]) for row in companion) != COMPANION_ROLES:
                raise SystemExit("Companion speaker-role pattern changed")
            rows = [*primary, *companion]
            for row in rows:
                occurrence_signatures.setdefault(str(row["text"]).strip(), set()).add(
                    (
                        "primary" if row in primary else "companion",
                        str(row.get("speaker_role") or "participant"),
                        bool(row.get("response_expected", False)),
                    )
                )
            if {
                band: sum(str(row["difficulty_band"]) == band for row in rows)
                for band in ("low", "medium", "high")
            } != {"low": 4, "medium": 4, "high": 4}:
                raise SystemExit("A cycle is not 4/4/4 difficulty-balanced")
            texts = [str(row["text"]).strip() for row in rows]
            empty = next((text for text in texts if not text), None)
            if empty is not None:
                raise SystemExit(
                    f"{participant['profile_id']} cycle {cycle_index} contains "
                    "an empty utterance"
                )
            duplicates = sorted({text for text in texts if text in seen})
            if duplicates:
                duplicate_profiles.setdefault(
                    str(participant["profile_id"]), []
                ).extend(duplicates)
            seen.update(texts)
            all_texts.extend(texts)
            cycle_means.append(sum(map(text_length, texts)) / 12.0)
            total_primary += len(primary)
            total_companion += len(companion)
            total_expected += sum(bool(row["response_expected"]) for row in primary)
        third = len(cycle_means) // 3
        early = sum(cycle_means[:third]) / third
        late = sum(cycle_means[-third:]) / third
        max_adjacent = max(
            abs(right - left)
            for left, right in zip(cycle_means, cycle_means[1:])
        )
        if abs(late - early) > 4.0 or max_adjacent > 12.0:
            raise SystemExit(f"{participant['profile_id']} fails lexical smoothness")
        inconsistent = {
            text: signatures
            for text, signatures in occurrence_signatures.items()
            if len(signatures) > 1
        }
        if inconsistent:
            raise SystemExit(
                f"{participant['profile_id']} repeats text across incompatible "
                "phase/role/intent labels"
            )
        repeated_occurrences = sum(
            len(texts) for profile_id, texts in duplicate_profiles.items()
            if profile_id == str(participant["profile_id"])
        )
        if repeated_occurrences > 6:
            raise SystemExit(
                f"{participant['profile_id']} exceeds the pre-audio repetition cap"
            )
        per_profile.append(
            {
                "profile_id": participant["profile_id"],
                "generation_seed_offset": int(
                    participant.get("generation_seed_offset", 0)
                ),
                "mean_text_length": sum(cycle_means) / len(cycle_means),
                "absolute_early_late_mean_length_difference": abs(late - early),
                "maximum_adjacent_cycle_mean_length_change": max_adjacent,
            }
        )
    providers = sorted({str(row.get("provider") or "") for row in participants})
    if providers != ["Z.AI"]:
        raise SystemExit(f"Unexpected providers: {providers}")
    result = {
        "schema": "tap-to-adapt-glm-intent-dialogue-audit/v1",
        "passed": True,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "profiles": len(participants),
        "cycles_per_profile": 12,
        "minutes_per_profile": 72,
        "opportunities_per_profile": 144,
        "total_opportunities": len(participants) * 144,
        "primary_turns": total_primary,
        "companion_turns": total_companion,
        "response_expected_turns": total_expected,
        "providers": providers,
        "model": payload["generator"]["model_id"],
        "provider_routing": payload["generator"]["provider_routing"],
        "reasoning_enabled": payload["generator"]["reasoning_enabled"],
        "unique_utterances_global": len(set(all_texts)),
        "utterances_global": len(all_texts),
        "duplicate_profiles": {
            profile_id: {
                "repeated_occurrences": len(texts),
                "unique_repeated_texts": len(set(texts)),
                "text_sha256": sorted(
                    {
                        hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for text in texts
                    }
                ),
            }
            for profile_id, texts in sorted(duplicate_profiles.items())
        },
        "repetition_policy": {
            "within_cycle_repetition": "prohibited",
            "cross_cycle_habitual_repetition": (
                "allowed only with identical phase, speaker role and reply "
                "intent; capped at 6 repeated occurrences per 144 opportunities"
            ),
            "maximum_observed_repeated_occurrences_per_profile": max(
                (
                    len(texts)
                    for texts in duplicate_profiles.values()
                ),
                default=0,
            ),
        },
        "maximum_profile_early_late_length_difference": max(
            row["absolute_early_late_mean_length_difference"]
            for row in per_profile
        ),
        "maximum_adjacent_cycle_length_change": max(
            row["maximum_adjacent_cycle_mean_length_change"]
            for row in per_profile
        ),
        "profiles_by_seed_offset": {
            str(offset): sum(
                int(row["generation_seed_offset"]) == offset
                for row in per_profile
            )
            for offset in sorted(
                {int(row["generation_seed_offset"]) for row in per_profile}
            )
        },
        "per_profile": per_profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
