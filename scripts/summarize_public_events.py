"""Aggregate a consent-filtered event export without publishing individual rows.

Input must already contain only authorized participants from the analysis cohort.
Output contains one row per condition and active block, with at least ten people
per row. It contains no participant identifiers, timestamps, or acoustic values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
from pathlib import Path
import statistics


GROUPS = {
    ("share_fit", "share_fit"): (1, "shared_initialization_construction"),
    ("contrast", "cold_learn"): (2, "cold_start_adaptation"),
    ("contrast", "shared_learn"): (2, "shared_initialization_adaptation"),
    ("contrast", "shared_frozen"): (2, "shared_initialization_frozen"),
}


def summarize(source: Path, destination: Path) -> int:
    """Read private events and write cohort-level counts and participant means."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["analysis_eligible"] != "1" or row["block"] not in {"1", "2", "3", "4", "5"}:
                continue
            stage, condition = GROUPS[row["cohort"], row["condition"]]
            person = grouped[stage, condition, int(row["block"])][row["participant_id"]]
            kind = row["event_type"]
            if kind == "model.prediction":
                person["decision_opportunities"] += int(row["decision_eligible"])
            if kind in {"tap.manual_activate", "tap.interrupt"}:
                person["raw_taps"] += 1
                person["eligible_corrections"] += int(row["behavior_correction"])
                if row["behavior_correction"] == "1":
                    person["eligible_activations" if kind == "tap.manual_activate" else "eligible_interruptions"] += 1
    output = []
    count_fields = ("decision_opportunities", "raw_taps", "eligible_corrections", "eligible_activations", "eligible_interruptions")
    for (stage, condition, block), people in sorted(grouped.items()):
        if len(people) < 10:
            raise ValueError("A group has fewer than ten contributors; no output written.")
        if any(p["decision_opportunities"] <= 0 for p in people.values()):
            raise ValueError("Each contributor requires positive decision exposure.")
        result = {"stage": stage, "condition": condition, "block": block, "n_participants": len(people), "active_minutes_per_participant": 6}
        for field in count_fields:
            result[field] = sum(p[field] for p in people.values())
        rates = [100 * p["eligible_corrections"] / p["decision_opportunities"] for p in people.values()]
        result["mean_participant_correction_rate_percent"] = round(statistics.mean(rates), 6)
        result["sd_participant_correction_rate_percent"] = round(statistics.stdev(rates), 6)
        result["mean_corrections_per_participant"] = round(result["eligible_corrections"] / len(people), 6)
        result["mean_corrections_per_active_minute"] = round(result["eligible_corrections"] / len(people) / 6, 6)
        output.append(result)
    if not output:
        raise ValueError("No active-block groups found.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return len(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Wrote {summarize(args.events, args.output)} group/block summaries.")
