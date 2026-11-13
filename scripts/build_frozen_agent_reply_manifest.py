"""Freeze already-recorded remote LLM replies against an immutable smooth-simulation schedule.

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
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = "tap-to-adapt-frozen-agent-replies/v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_digest(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def required_source(event: dict[str, Any]) -> str | None:
    if event["phase"] == "primary" and bool(event["response_expected"]):
        return "primary"
    if event["category"] == "participant_to_companion":
        return "companion"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze already-recorded remote LLM replies against an immutable "
            "smooth-simulation schedule."
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-run-slug", required=True)
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    turns_bytes = args.turns.read_bytes()
    turns = json.loads(turns_bytes.decode("utf-8"))
    if (
        turns.get("schema") != "tap-to-adapt-intent-aware-4plus2-turns/v3"
        or turns.get("cohort") != "duration32"
        or not bool(turns.get("simulation_only"))
        or len(turns.get("participants", [])) != 32
    ):
        raise SystemExit("Turns must be one immutable duration32 simulation schedule")
    with args.source_roster.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        source_roster = list(csv.DictReader(handle))
    if len(source_roster) != 32:
        raise SystemExit("Source roster must contain exactly 32 participants")
    source_by_participant = {
        row["participant_id"]: int(row["participant_index"])
        for row in source_roster
    }
    if set(source_by_participant.values()) != set(range(1, 33)):
        raise SystemExit("Source roster participant indices must be exactly 1..32")

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    source_run = connection.execute(
        "SELECT * FROM study_runs WHERE slug=?", (args.source_run_slug,)
    ).fetchone()
    if source_run is None or source_run["data_origin"] != "simulation":
        raise SystemExit("Source run is absent or not simulation-origin")
    rows = connection.execute(
        """
        SELECT e.participant_id, e.seq, e.server_ms, e.event_hash, e.payload
        FROM events e
        JOIN study_runs r ON r.id=e.run_id
        WHERE r.slug=? AND e.event_type='chat.turn'
        ORDER BY e.participant_id, e.seq
        """,
        (args.source_run_slug,),
    ).fetchall()
    by_index: dict[int, list[sqlite3.Row]] = {
        index: [] for index in range(1, 33)
    }
    for row in rows:
        participant_id = str(row["participant_id"])
        if participant_id not in source_by_participant:
            raise SystemExit("Chat event participant is absent from source roster")
        by_index[source_by_participant[participant_id]].append(row)

    participants = {
        int(item["participant_index"]): item for item in turns["participants"]
    }
    output_rows: list[dict[str, Any]] = []
    for participant_index in range(1, 33):
        required = [
            (event, source)
            for event in participants[participant_index]["events"]
            if (source := required_source(event)) is not None
        ]
        recorded = by_index[participant_index]
        if len(recorded) < len(required):
            raise SystemExit(
                f"Profile {participant_index} has {len(recorded)} recorded replies "
                f"but requires {len(required)}"
            )
        for (event, source), row in zip(required, recorded, strict=False):
            payload = json.loads(str(row["payload"]))
            transcript = str(event["text"])
            reply = str(payload.get("reply") or "").strip()
            if (
                payload.get("source") != source
                or payload.get("transcript") != transcript
                or payload.get("mode") != "remote"
                or payload.get("model") != "z-ai/glm-5.2"
                or not reply
            ):
                raise SystemExit(
                    "Recorded GLM-5.2 reply provenance mismatch for "
                    f"profile={participant_index}, "
                    f"opportunity={event['opportunity_index']}"
                )
            output_rows.append(
                {
                    "participant_index": participant_index,
                    "opportunity_index": int(event["opportunity_index"]),
                    "source": source,
                    "transcript_sha256": sha256_text(transcript),
                    "reply": reply,
                    "reply_sha256": sha256_text(reply),
                    "model": "z-ai/glm-5.2",
                    "mode": "remote_recorded",
                    "source_event_seq": int(row["seq"]),
                    "source_event_hash": str(row["event_hash"]),
                    "source_server_ms": int(row["server_ms"]),
                }
            )
    expected_rows = sum(
        required_source(event) is not None
        for participant in turns["participants"]
        for event in participant["events"]
    )
    if len(output_rows) != expected_rows:
        raise SystemExit("Frozen reply manifest row count is incomplete")
    output = {
        "schema": SCHEMA,
        "simulation_only": True,
        "source_run_slug": args.source_run_slug,
        "source_data_origin": "simulation",
        "source_mode": "remote",
        "llm_provider": "OpenRouter",
        "llm_model": "z-ai/glm-5.2",
        "provider_order": ["z-ai"],
        "allow_fallbacks": False,
        "turns_sha256": sha256_bytes(turns_bytes),
        "source_roster_sha256": sha256_bytes(args.source_roster.read_bytes()),
        "participants": 32,
        "rows": output_rows,
        "rows_sha256": canonical_digest(output_rows),
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "participants": 32,
                "rows": len(output_rows),
                "source_run_slug": args.source_run_slug,
                "llm_model": output["llm_model"],
                "rows_sha256": output["rows_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
