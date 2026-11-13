"""Rebind a frozen smooth-turn schedule to fresh simulation-only accounts without changing text, intent, timing, or profile assignment.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from .smooth_simulation_protocol import RUN_SLUG_PATTERN
except ImportError:
    from smooth_simulation_protocol import RUN_SLUG_PATTERN


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind a frozen smooth-turn schedule to fresh simulation-only accounts "
            "without changing text, intent, timing, or profile assignment."
        )
    )
    parser.add_argument("--template-turns", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    template_bytes = args.template_turns.read_bytes()
    template = json.loads(template_bytes.decode("utf-8"))
    cohort = str(template.get("cohort", ""))
    expected = {"selection8": 8, "duration32": 32}.get(cohort)
    if (
        template.get("schema") != "tap-to-adapt-intent-aware-4plus2-turns/v3"
        or expected is None
        or not bool(template.get("simulation_only"))
    ):
        raise SystemExit("Template is not a supported smooth simulation turn schedule")
    participants = template.get("participants")
    if not isinstance(participants, list) or len(participants) != expected:
        raise SystemExit(
            f"{cohort} template must contain exactly {expected} profiles"
        )

    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster_rows = list(csv.DictReader(handle))
    if len(roster_rows) != expected:
        raise SystemExit(f"Roster must contain exactly {expected} participants")
    if {row["cohort"] for row in roster_rows} != {cohort}:
        raise SystemExit("Roster cohort does not match the frozen template")
    run_slugs = {row["run_slug"] for row in roster_rows}
    if len(run_slugs) != 1:
        raise SystemExit("Roster must contain one run slug")
    run_slug = next(iter(run_slugs))
    if RUN_SLUG_PATTERN.fullmatch(run_slug) is None:
        raise SystemExit("Roster run slug is outside simulation-intent-v3")
    roster_by_index = {
        int(row["participant_index"]): row for row in roster_rows
    }
    if set(roster_by_index) != set(range(1, expected + 1)):
        raise SystemExit(
            f"Roster participant indices must be exactly 1..{expected}"
        )
    if len({row["participant_id"] for row in roster_rows}) != expected:
        raise SystemExit("Roster participant IDs must be unique")

    output: dict[str, Any] = deepcopy(template)
    output["run_slug"] = run_slug
    output["account_rebinding_only"] = True
    output["template_turns_sha256"] = hashlib.sha256(template_bytes).hexdigest()
    for participant in output["participants"]:
        index = int(participant["participant_index"])
        participant["participant_id"] = roster_by_index[index]["participant_id"]

    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "simulation_only": True,
                "run_slug": run_slug,
                "cohort": cohort,
                "participants": expected,
                "account_rebinding_only": True,
                "template_turns_sha256": output["template_turns_sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
