"""Verify the append-only event hash chains.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path


def verify(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    checked = 0
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    participants = connection.execute("SELECT id FROM participants ORDER BY id").fetchall()
    for participant in participants:
        previous = "0" * 64
        expected_seq = 1
        rows = connection.execute(
            "SELECT * FROM events WHERE participant_id=? ORDER BY seq", (participant["id"],)
        ).fetchall()
        for row in rows:
            material = "|".join(
                [
                    previous,
                    row["participant_id"],
                    str(row["session_id"] or ""),
                    str(row["seq"]),
                    row["event_type"],
                    str(row["server_ms"]),
                    row["payload"],
                ]
            )
            digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if row["seq"] != expected_seq:
                errors.append(f"{participant['id']}: expected seq {expected_seq}, got {row['seq']}")
            if row["prev_hash"] != previous:
                errors.append(f"{participant['id']} seq {row['seq']}: previous hash mismatch")
            if row["event_hash"] != digest:
                errors.append(f"{participant['id']} seq {row['seq']}: event hash mismatch")
            previous = row["event_hash"]
            expected_seq += 1
            checked += 1
    connection.close()
    return checked, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the append-only event hash chains.")
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    checked, errors = verify(args.database)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Verified {checked} events with no hash-chain errors.")


if __name__ == "__main__":
    main()
