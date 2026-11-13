"""Command-line utility: analyze public selection confirmation.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[a-f0-9]{64}", args.release_manifest_sha256) is None:
        raise SystemExit("--release-manifest-sha256 must be a lowercase SHA-256")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    participants = payload["results"]
    if len(participants) != 4 or any(len(row["records"]) != 36 for row in participants):
        raise SystemExit("Selection confirmation requires 4 complete 18-minute profiles")
    records = [record for row in participants for record in row["records"]]
    cycles = []
    for cycle in (1, 2, 3):
        selected = [row for row in records if int(row["cycle_index"]) == cycle]
        positives = [row for row in selected if int(row["label"]) == 1]
        negatives = [row for row in selected if int(row["label"]) == 0]
        cycles.append(
            {
                "cycle_index": cycle,
                "start_minute": (cycle - 1) * 6,
                "end_minute": cycle * 6,
                "opportunities": len(selected),
                "correction_rate": sum(bool(row["corrected"]) for row in selected)
                / len(selected),
                "wake_rate": sum(bool(row["activated"]) for row in selected)
                / len(selected),
                "manual_activate_rate": sum(
                    bool(row["corrected"]) for row in positives
                )
                / len(positives),
                "interrupt_rate": sum(bool(row["corrected"]) for row in negatives)
                / len(negatives),
            }
        )
    rebounds = [
        cycles[index]["correction_rate"] - cycles[index - 1]["correction_rate"]
        for index in range(1, len(cycles))
    ]
    result = {
        "schema": "tap-to-adapt-public-selection-confirmation/v1",
        "simulation_only": True,
        "human_effect_evidence": False,
        "run_slug": payload["run_slug"],
        "profiles": len(participants),
        "minutes_per_profile": 18,
        "clock_scale": payload["clock_scale"],
        "candidate_id": args.candidate_id,
        "release_manifest_sha256": args.release_manifest_sha256,
        "passed": (
            all(0.15 <= row["wake_rate"] <= 0.75 for row in cycles)
            and max(rebounds) <= 0.25
        ),
        "maximum_correction_rebound": max(rebounds),
        "cycles": cycles,
    }
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cycles[0]))
        writer.writeheader()
        writer.writerows(cycles)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
