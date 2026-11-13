"""Exhaustively validate a fresh simulation roster without creating sessions. The command refuses formal or pilot runs.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PARTICIPANT_DATA_TABLES = (
    "sessions",
    "pair_intervals",
    "events",
    "quality_checks",
    "responses",
    "audio_files",
    "model_snapshots",
    "prior_probe_assignments",
    "model_counters",
)
EXPECTED_WAVES = {1: 12, 2: 8, 3: 8, 4: 8, 5: 32, 6: 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float = 20.0,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return int(response.status), json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body[:500]}
        return int(exc.code), parsed


def page_request(url: str, timeout: float = 20.0) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(4096).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(4096).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively validate a fresh simulation roster without creating sessions. "
            "The command refuses formal or pilot runs."
        )
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--roster-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        help="API base URL; repeat to validate internal and public routes",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    db = StudyDB(settings)
    run = db.study_run(args.run_slug)
    if run is None:
        raise SystemExit(f"Unknown run slug: {args.run_slug}")
    if run["data_origin"] != "simulation" or run["protocol_mode"] != "compressed":
        raise SystemExit("Preflight accepts only simulation/compressed runs")

    roster_dir = args.roster_dir.resolve()
    manifest_path = roster_dir / "roster_manifest.json"
    assignment_path = roster_dir / "simulation_assignments.csv"
    runner_path = roster_dir / "synthetic_runner_roster.csv"
    blinded_path = roster_dir / "blinded_simulation_roster.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = load_csv(assignment_path)
    runner_rows = load_csv(runner_path)
    blinded_rows = load_csv(blinded_path)
    participants = db.participants_in_run(run["id"])
    participants_by_id = {row["id"]: row for row in participants}

    failures: list[dict[str, Any]] = []

    def fail(check: str, detail: Any) -> None:
        failures.append({"check": check, "detail": detail})

    codes = [row["participant_code"].strip().upper() for row in assignments]
    assignment_ids = [row["assignment_id"] for row in assignments]
    neutral_ids = [row["neutral_id"] for row in assignments]
    participant_ids = [row["participant_id"] for row in assignments]
    if len(assignments) != 72:
        fail("assignment_count", len(assignments))
    for name, values in (
        ("participant_code", codes),
        ("assignment_id", assignment_ids),
        ("neutral_id", neutral_ids),
        ("participant_id", participant_ids),
    ):
        if len(set(values)) != 72:
            fail(f"unique_{name}", {"count": len(values), "unique": len(set(values))})

    if len(participants) != 72:
        fail("database_participant_count", len(participants))
    if manifest.get("run_id") != run["id"] or manifest.get("run_slug") != args.run_slug:
        fail(
            "manifest_run_identity",
            {"manifest_run_id": manifest.get("run_id"), "database_run_id": run["id"]},
        )
    if (
        manifest.get("protocol_sha256") != run.get("protocol_sha256")
        or not run.get("protocol_json")
    ):
        fail(
            "protocol_snapshot",
            {
                "manifest": manifest.get("protocol_sha256"),
                "database": run.get("protocol_sha256"),
                "snapshot_embedded": bool(run.get("protocol_json")),
            },
        )
    if manifest.get("analysis_eligible") is not False:
        fail("manifest_analysis_eligible", manifest.get("analysis_eligible"))

    manifest_hashes: dict[str, dict[str, Any]] = {}
    for filename, expected_digest in manifest.get("files", {}).items():
        path = roster_dir / filename
        actual_digest = sha256_file(path) if path.is_file() else None
        manifest_hashes[filename] = {
            "expected": expected_digest,
            "actual": actual_digest,
            "matches": actual_digest == expected_digest,
        }
        if actual_digest != expected_digest:
            fail("manifest_file_hash", {"file": filename, "actual": actual_digest})

    runner_map = {
        row["neutral_id"]: row["participant_code"].strip().upper() for row in runner_rows
    }
    blinded_map = {
        row["neutral_id"]: row["participant_code"].strip().upper() for row in blinded_rows
    }
    if len(runner_map) != 72 or len(blinded_map) != 72:
        fail("cross_roster_counts", {"runner": len(runner_map), "blinded": len(blinded_map)})

    expected_wave_counts = Counter()
    expected_cohort_counts = Counter()
    expected_group_counts = Counter()
    for row in assignments:
        code = row["participant_code"].strip().upper()
        participant = db.participant_by_code(code)
        if participant is None:
            fail("database_code_lookup", row["assignment_id"])
            continue
        wave = int(row["enrollment_wave"])
        expected_wave_counts[wave] += 1
        expected_cohort_counts[row["cohort"]] += 1
        expected_group_counts[row["internal_group"]] += 1
        expected_release = "released" if wave == 1 else "held"
        expected = {
            "id": row["participant_id"],
            "run_id": run["id"],
            "run_slug": args.run_slug,
            "assignment_id": row["assignment_id"],
            "neutral_id": row["neutral_id"],
            "cohort": row["cohort"],
            "group_name": row["internal_group"],
            "enrollment_wave": wave,
            "release_state": expected_release,
            "preview_mode": 1,
            "enabled": 1,
        }
        mismatches = {
            key: {"expected": value, "actual": participant.get(key)}
            for key, value in expected.items()
            if participant.get(key) != value
        }
        if mismatches:
            fail("database_assignment_mapping", {"assignment_id": row["assignment_id"], "mismatches": mismatches})
        if participants_by_id.get(row["participant_id"], {}).get("assignment_id") != row["assignment_id"]:
            fail("database_participant_id_mapping", row["assignment_id"])
        if runner_map.get(row["neutral_id"]) != code:
            fail("runner_roster_mapping", row["neutral_id"])
        if blinded_map.get(row["neutral_id"]) != code:
            fail("blinded_roster_mapping", row["neutral_id"])

    if dict(sorted(expected_wave_counts.items())) != EXPECTED_WAVES:
        fail("wave_distribution", dict(sorted(expected_wave_counts.items())))
    if expected_cohort_counts != Counter({"study_a": 36, "study_b": 36}):
        fail("cohort_distribution", dict(expected_cohort_counts))
    if expected_group_counts != Counter(
        {
            "seed_donor": 12,
            "evaluator_donor": 16,
            "holdout_evaluator": 8,
            "personalized": 18,
            "frozen": 18,
        }
    ):
        fail("group_distribution", dict(expected_group_counts))

    def participant_table_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        with db.connect() as conn:
            for table in PARTICIPANT_DATA_TABLES:
                counts[table] = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (run["id"],)
                    ).fetchone()[0]
                )
        with db.connect_consent() as conn:
            counts["consents"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM consents WHERE run_id=?", (run["id"],)
                ).fetchone()[0]
            )
        return counts

    before_counts = participant_table_counts()
    for table, count in before_counts.items():
        if count != 0:
            fail("fresh_run_before_login", {"table": table, "count": count})

    page_checks: list[dict[str, Any]] = []
    normalized_urls = sorted({url.rstrip("/") for url in args.base_url})
    for base_url in normalized_urls:
        for role, suffix in (("study", "/"), ("companion", "/companion")):
            status, body = page_request(base_url + suffix)
            passed = status == 200 and "<html" in body.lower()
            page_checks.append(
                {"base_url": base_url, "role": role, "status": status, "passed": passed}
            )
            if not passed:
                fail("page_route", page_checks[-1])

    def probe(base_url: str, row: dict[str, str], role: str) -> dict[str, Any]:
        code = row["participant_code"].strip().upper()
        wave = int(row["enrollment_wave"])
        expected_status = 200 if wave == 1 else 423
        result: dict[str, Any] = {
            "base_url": base_url,
            "assignment_id": row["assignment_id"],
            "role": role,
            "wave": wave,
            "expected_status": expected_status,
        }
        try:
            status, body = json_request(
                base_url + "/api/login", payload={"code": code, "role": role}
            )
            result["status"] = status
            if status != expected_status:
                result["error"] = {"unexpected_status": status, "body": body}
                return result
            if expected_status == 423:
                detail = body.get("detail", {})
                if detail.get("code") != "study_pipeline_locked":
                    result["error"] = {"unexpected_lock": detail}
                    return result
                result["lock_code"] = detail["code"]
                result["passed"] = True
                return result
            if (
                body.get("role") != role
                or body.get("preview_mode") is not True
                or body.get("pilot_only") is not True
                or not body.get("token")
            ):
                result["error"] = {"invalid_login_payload": body}
                return result
            bootstrap_status, bootstrap = json_request(
                base_url + "/api/bootstrap", token=str(body["token"])
            )
            if (
                bootstrap_status != 200
                or bootstrap.get("role") != role
                or bootstrap.get("participant", {}).get("neutral_id") != row["neutral_id"]
                or bootstrap.get("progress", {}).get("sessions") != []
            ):
                result["error"] = {
                    "invalid_bootstrap": {"status": bootstrap_status, "body": bootstrap}
                }
                return result
            result["bootstrap_status"] = bootstrap_status
            result["passed"] = True
            return result
        except Exception as exc:  # Network and decoding failures must be reported per account.
            result["error"] = {"exception": f"{type(exc).__name__}: {exc}"}
            return result

    futures = []
    probes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 32))) as executor:
        for base_url in normalized_urls:
            for row in assignments:
                for role in ("study", "companion"):
                    futures.append(executor.submit(probe, base_url, row, role))
        for future in as_completed(futures):
            result = future.result()
            probes.append(result)
            if not result.get("passed"):
                fail("login_probe", result)

    after_counts = participant_table_counts()
    if after_counts != before_counts:
        fail("login_created_participant_data", {"before": before_counts, "after": after_counts})

    probe_statuses = Counter(int(result.get("status", 0)) for result in probes)
    report = {
        "passed": not failures,
        "run": {
            "slug": args.run_slug,
            "id": run["id"],
            "data_origin": run["data_origin"],
            "protocol_mode": run["protocol_mode"],
            "analysis_eligible": False,
        },
        "roster": {
            "participants": len(assignments),
            "unique_codes": len(set(codes)),
            "unique_assignment_ids": len(set(assignment_ids)),
            "unique_neutral_ids": len(set(neutral_ids)),
            "wave_counts": dict(sorted(expected_wave_counts.items())),
            "cohort_counts": dict(expected_cohort_counts),
            "group_counts": dict(expected_group_counts),
            "manifest_hashes": manifest_hashes,
        },
        "routes": page_checks,
        "login_preflight": {
            "base_urls": normalized_urls,
            "total_probes": len(probes),
            "passed_probes": sum(bool(result.get("passed")) for result in probes),
            "status_counts": dict(sorted(probe_statuses.items())),
            "successful_login_and_bootstrap": sum(
                result.get("status") == 200 and result.get("passed") for result in probes
            ),
            "expected_pipeline_locks": sum(
                result.get("status") == 423 and result.get("passed") for result in probes
            ),
        },
        "non_consuming_check": {
            "before": before_counts,
            "after": after_counts,
            "unchanged": before_counts == after_counts,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
