"""Estimate the correction-rate plateau from a public duration simulation.

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


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    total = len(records)
    corrections = sum(int(row["corrected"]) for row in records)
    positives = [row for row in records if int(row["label"]) == 1]
    negatives = [row for row in records if int(row["label"]) == 0]
    manual = sum(
        row.get("correction_type") == "manual_activate" for row in positives
    )
    interrupts = sum(row.get("correction_type") == "interrupt" for row in negatives)
    return {
        "eligible_opportunities": total,
        "corrections": corrections,
        "correction_rate": corrections / total if total else None,
        "activate_opportunities": len(positives),
        "manual_activates": manual,
        "manual_activate_rate": manual / len(positives) if positives else None,
        "interrupt_opportunities": len(negatives),
        "interrupts": interrupts,
        "interrupt_rate": interrupts / len(negatives) if negatives else None,
        "wake_rate": (
            sum(int(row["activated"]) for row in records) / total if total else None
        ),
        "mean_shared_probability": (
            sum(float(row["shared_probability"]) for row in records)
            / sum(row["shared_probability"] is not None for row in records)
            if any(row["shared_probability"] is not None for row in records)
            else None
        ),
        "mean_personal_probability": (
            sum(float(row["personal_probability"]) for row in records)
            / sum(row["personal_probability"] is not None for row in records)
            if any(row["personal_probability"] is not None for row in records)
            else None
        ),
    }


def rate(records: list[dict[str, Any]]) -> float:
    return (
        sum(int(row["corrected"]) for row in records) / len(records)
        if records
        else math.nan
    )


def bootstrap_remaining_improvement(
    records: list[dict[str, Any]],
    candidate_bin: int,
    terminal_bins: set[int],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_participant[str(row["participant_id"])].append(row)
    participants = sorted(by_participant)
    rng = random.Random(seed + candidate_bin)
    differences: list[float] = []
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for draw in range(len(participants)):
            source = rng.choice(participants)
            for row in by_participant[source]:
                sampled.append({**row, "_bootstrap_id": draw})
        candidate = [
            row for row in sampled if int(row["bin_5m"]) == candidate_bin
        ]
        terminal = [
            row for row in sampled if int(row["bin_5m"]) in terminal_bins
        ]
        if candidate and terminal:
            differences.append(rate(candidate) - rate(terminal))
    return (
        percentile(differences, 0.025),
        percentile(differences, 0.5),
        percentile(differences, 0.975),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the correction-rate plateau from a public duration simulation."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--equivalence-margin", type=float, default=0.02)
    parser.add_argument("--upper-improvement-margin", type=float, default=0.03)
    parser.add_argument("--consecutive-bins", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    run = json.loads(args.input.read_text(encoding="utf-8"))
    if run.get("duration_evidence_valid") is not True:
        raise SystemExit("Refusing to estimate duration from a time-compressed smoke run")
    if run.get("failures") or int(run.get("participants_succeeded", 0)) < 4:
        raise SystemExit("Duration run is incomplete")
    records = [
        row
        for participant in run["results"]
        for row in participant["records"]
    ]
    if not records:
        raise SystemExit("Duration run contains no opportunities")

    bins = sorted({int(row["bin_5m"]) for row in records})
    maximum_bin = max(bins)
    terminal_bins = set(range(max(1, maximum_bin - 2), maximum_bin + 1))
    rows: list[dict[str, Any]] = []
    for bin_number in bins:
        selected = [row for row in records if int(row["bin_5m"]) == bin_number]
        summary = summarize(selected)
        low, median, high = bootstrap_remaining_improvement(
            records,
            bin_number,
            terminal_bins,
            iterations=args.bootstrap,
            seed=20260723,
        )
        rows.append(
            {
                "bin_5m": bin_number,
                "start_minute": (bin_number - 1) * 5,
                "end_minute": bin_number * 5,
                **summary,
                "remaining_improvement_bootstrap_low": low,
                "remaining_improvement_bootstrap_median": median,
                "remaining_improvement_bootstrap_high": high,
            }
        )

    terminal_records = [
        row for row in records if int(row["bin_5m"]) in terminal_bins
    ]
    terminal_rate = rate(terminal_records)
    empirical_candidate: int | None = None
    confirmatory_candidate: int | None = None
    for index, row in enumerate(rows):
        window = rows[index : index + args.consecutive_bins + 1]
        if len(window) < args.consecutive_bins + 1:
            continue
        stable = all(
            abs(float(item["correction_rate"]) - terminal_rate)
            <= args.equivalence_margin
            for item in window
        )
        if stable and empirical_candidate is None:
            empirical_candidate = int(row["start_minute"])
        if (
            stable
            and float(row["remaining_improvement_bootstrap_high"])
            <= args.upper_improvement_margin
            and confirmatory_candidate is None
        ):
            confirmatory_candidate = int(row["start_minute"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first_three = [
        row for row in records if float(row["elapsed_seconds"]) < 15 * 60
    ]
    last_three = [
        row
        for row in records
        if float(row["elapsed_seconds"])
        >= max(float(item["elapsed_seconds"]) for item in records) - 15 * 60
    ]
    report = {
        "schema": "tap-to-adapt-duration-analysis/v1",
        "simulation_only": True,
        "human_effect_evidence": False,
        "run_slug": run["run_slug"],
        "participants": run["participants_succeeded"],
        "llm_provider": run["llm_provider"],
        "llm_model": run["llm_model"],
        "tts_provider": run["tts_provider"],
        "tts_model": run["tts_model"],
        "model": run["model"],
        "bin_minutes": 5,
        "maximum_observed_minutes": run["max_minutes"],
        "opportunity_seconds": run["opportunity_seconds"],
        "overall": summarize(records),
        "first_15_minutes": summarize(first_three),
        "last_15_minutes": summarize(last_three),
        "relative_correction_reduction_first_to_last": (
            1.0 - rate(last_three) / rate(first_three)
            if rate(first_three) > 0
            else None
        ),
        "terminal_bin_rate": terminal_rate,
        "terminal_bins": sorted(terminal_bins),
        "empirical_practical_plateau_minute": empirical_candidate,
        "confirmatory_plateau_minute": confirmatory_candidate,
        "plateau_rule": {
            "absolute_rate_equivalence_margin": args.equivalence_margin,
            "bootstrap_remaining_improvement_upper_margin": (
                args.upper_improvement_margin
            ),
            "consecutive_following_bins": args.consecutive_bins,
            "bootstrap_unit": "synthetic_participant",
            "bootstrap_iterations": args.bootstrap,
        },
        "interpretation": (
            "If confirmatory_plateau_minute is null, the run did not establish a "
            "precision-qualified plateau within the observed duration. The empirical "
            "candidate is descriptive only."
        ),
        "bins": rows,
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["end_minute"]) for row in rows]
    correction = [float(row["correction_rate"]) for row in rows]
    manual = [float(row["manual_activate_rate"]) for row in rows]
    interrupt = [float(row["interrupt_rate"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, correction, marker="o", label="All corrections")
    axes[0].plot(x, manual, marker="o", label="Manual activate")
    axes[0].plot(x, interrupt, marker="o", label="Manual interrupt")
    axes[0].axhline(terminal_rate, color="black", linestyle="--", alpha=0.55)
    if empirical_candidate is not None:
        axes[0].axvline(
            empirical_candidate,
            color="#b44",
            linestyle=":",
            label="Descriptive plateau candidate",
        )
    axes[0].set_ylabel("Correction rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2)
    axes[1].plot(
        x,
        [float(row["mean_shared_probability"]) for row in rows],
        marker="o",
        label="Shared head",
    )
    axes[1].plot(
        x,
        [float(row["mean_personal_probability"]) for row in rows],
        marker="o",
        label="Personal head",
    )
    axes[1].set_xlabel("Elapsed interaction time (minutes)")
    axes[1].set_ylabel("Mean wake probability")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.suptitle("M0 + Personalised duration simulation (5-minute bins)")
    fig.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=180)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
