"""Build a read-only server-event audit of every automatic activation in one isolated simulation run.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only server-event audit of every automatic activation "
            "in one isolated simulation run."
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    connection = sqlite3.connect(
        f"file:{args.database.resolve()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    run = connection.execute(
        "SELECT id, slug, data_origin FROM study_runs WHERE slug=?",
        (args.run_slug,),
    ).fetchone()
    if run is None or run["data_origin"] != "simulation":
        raise SystemExit("Run is missing or is not simulation-only")
    participant_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM participants WHERE run_id=?",
            (run["id"],),
        ).fetchone()[0]
    )
    raw_rows = connection.execute(
        """
        SELECT e.participant_id, e.seq, e.server_ms, e.payload
        FROM events AS e
        JOIN participants AS p ON p.id=e.participant_id
        WHERE p.run_id=? AND e.event_type='agent.auto_activate'
        ORDER BY e.participant_id, e.seq
        """,
        (run["id"],),
    ).fetchall()
    rows = []
    for raw in raw_rows:
        payload = json.loads(raw["payload"])
        rows.append(
            {
                "participant_id": raw["participant_id"],
                "seq": int(raw["seq"]),
                "server_ms": int(raw["server_ms"]),
                "activation_id": payload.get("activation_id"),
                "activate": payload.get("activate"),
                "activation_source": payload.get("activation_source"),
                "decision_eligible": payload.get("decision_eligible"),
                "silence_ms": float(payload.get("silence_ms", -1)),
                "minimum_end_of_turn_silence_ms": float(
                    payload.get("minimum_end_of_turn_silence_ms", -1)
                ),
            }
        )
    violations = [
        row
        for row in rows
        if row["activate"] is not True
        or row["activation_source"] != "model"
        or row["decision_eligible"] is not True
        or row["silence_ms"] < row["minimum_end_of_turn_silence_ms"]
        or row["minimum_end_of_turn_silence_ms"] < 400
    ]
    rows_text = "\n".join(canonical(row) for row in rows)
    report = {
        "schema": "tap-to-adapt-server-eou-activation-audit/v1",
        "simulation_only": True,
        "run_slug": args.run_slug,
        "participant_count": participant_count,
        "automatic_activations": len(rows),
        "eou_gate_violations": len(violations),
        "minimum_observed_vad_silence_ms": (
            min(row["silence_ms"] for row in rows) if rows else None
        ),
        "maximum_observed_vad_silence_ms": (
            max(row["silence_ms"] for row in rows) if rows else None
        ),
        "minimum_locked_gate_ms": (
            min(row["minimum_end_of_turn_silence_ms"] for row in rows)
            if rows
            else None
        ),
        "rows_sha256": hashlib.sha256(rows_text.encode("utf-8")).hexdigest(),
        "violations": violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
