"""Build an eight-profile public-route TTS fixture for model/trainer selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .run_duration_simulation_public import request_pcm
    from .simulate_feasibility import ensure_consent, login, request_json
    from .smooth_simulation_protocol import SCHEMA, validate_schedule
except ImportError:
    from run_duration_simulation_public import request_pcm
    from simulate_feasibility import ensure_consent, login, request_json
    from smooth_simulation_protocol import SCHEMA, validate_schedule


def write_pcm(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm)


def build_participant(
    *,
    base_url: str,
    roster: dict[str, str],
    turns: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    events = list(turns["events"])
    validate_schedule(events, max_minutes=int(turns["_max_minutes"]))
    study_auth = login(base_url, roster["participant_code"], "study")
    companion_auth = login(base_url, roster["participant_code"], "companion")
    study_token = str(study_auth["token"])
    companion_token = str(companion_auth["token"])
    ensure_consent(base_url, study_token)
    study_session = request_json(
        base_url,
        "/api/sessions/start",
        method="POST",
        token=study_token,
        body={"day": 1},
    )
    companion_session = request_json(
        base_url,
        "/api/sessions/start",
        method="POST",
        token=companion_token,
        body={"day": 1},
    )
    cache: dict[tuple[str, str], tuple[str, bytes]] = {}
    rows = []
    try:
        for event in events:
            role = str(event["speaker_role"])
            text = str(event["text"])
            key = (role, text)
            if key not in cache:
                if role == "participant":
                    pcm = request_pcm(
                        base_url,
                        study_token,
                        session_id=str(study_session["id"]),
                        text=text,
                        speaker_id=int(roster["voice_id"]),
                        speed=float(roster["speed"]),
                        source="participant",
                    )
                elif role == "companion":
                    pcm = request_pcm(
                        base_url,
                        companion_token,
                        session_id=str(companion_session["id"]),
                        text=text,
                        speaker_id=None,
                        speed=1.0,
                        source="companion",
                    )
                else:
                    raise RuntimeError(f"Unknown speaker role: {role}")
                digest = hashlib.sha256(
                    f"{roster['participant_index']}|{role}|{text}".encode("utf-8")
                ).hexdigest()[:16]
                filename = f"p{int(roster['participant_index']):02d}-{role}-{digest}.wav"
                write_pcm(output_dir / "audio" / filename, pcm)
                cache[key] = (filename, pcm)
            filename, pcm = cache[key]
            rows.append(
                {
                    **event,
                    "file": filename,
                    "participant_id": roster["participant_id"],
                    "participant_index": int(roster["participant_index"]),
                    "profile_id": turns["profile"]["profile_id"],
                    "role": "evaluator",
                    "voice_id": int(roster["voice_id"]),
                    "companion_voice": roster["companion_agent_voice_name"],
                    "speed": float(roster["speed"]),
                    "pcm_samples": len(pcm) // 2,
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                }
            )
    finally:
        try:
            request_json(
                base_url,
                f"/api/sessions/{study_session['id']}/abort",
                method="POST",
                token=study_token,
            )
        except Exception:
            pass
        try:
            request_json(
                base_url,
                f"/api/sessions/{companion_session['id']}/abort",
                method="POST",
                token=companion_token,
            )
        except Exception:
            pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirm-simulation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_simulation:
        raise SystemExit("Pass --confirm-simulation")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = list(csv.DictReader(handle))
    turns = json.loads(args.turns.read_text(encoding="utf-8"))
    if turns.get("schema") != SCHEMA or turns.get("cohort") != "selection8":
        raise SystemExit("Fixture construction requires a selection8 intent-aware schedule")
    if len(roster) != 8 or len({row["voice_id"] for row in roster}) != 8:
        raise SystemExit("Model/trainer selection requires eight distinct target voices")
    by_id = {str(item["participant_id"]): item for item in turns["participants"]}
    if {row["participant_id"] for row in roster} != set(by_id):
        raise SystemExit("Roster and turns participant IDs do not match")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audio").mkdir()
    rows: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as pool:
            futures = {}
            for row in roster:
                item = dict(by_id[row["participant_id"]])
                item["_max_minutes"] = turns["max_minutes"]
                futures[
                    pool.submit(
                        build_participant,
                        base_url=args.base_url.rstrip("/"),
                        roster=row,
                        turns=item,
                        output_dir=args.output_dir,
                    )
                ] = row
            for future in as_completed(futures):
                rows.extend(future.result())
    except BaseException:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise
    rows.sort(
        key=lambda row: (int(row["participant_index"]), int(row["opportunity_index"]))
    )
    for index, row in enumerate(rows, start=1):
        row["global_event_index"] = index
        row["local_event"] = int(row["opportunity_index"])
        row["cycle"] = int(row["cycle_index"])
        row["timestamp_seconds"] = int(row["elapsed_seconds"])
    manifest = {
        "schema": "tap-to-adapt-intent-aware-selection-fixture/v3",
        "simulation_only": True,
        "run_slug": turns["run_slug"],
        "protocol": {
            "total_minutes": turns["max_minutes"],
            "cycles": turns["max_minutes"] // 6,
            "primary_minutes_per_cycle": 4,
            "companion_minutes_per_cycle": 2,
            "opportunities_per_cycle": 12,
            "clip_seconds": 15,
            "silent_gap_representation": "left-zero-padding after >=15 s gap",
        },
        "participants": 8,
        "donors": 0,
        "evaluators": 8,
        "target_voice_ids": sorted({int(row["voice_id"]) for row in rows}),
        "companion_voice": roster[0]["companion_agent_voice_name"],
        "content_source": {
            "kind": "locked_llm_generated_pool",
            "provider": turns["llm_provider"],
            "model": turns["llm_model"],
            "sha256": turns["source_pool_sha256"],
        },
        "voice_scope_note": (
            "Eight fixed AISHELL3 speaker identities are engineering profiles, "
            "not independent human participants."
        ),
        "events": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "simulation_only": True,
                "participants": 8,
                "events": len(rows),
                "unique_audio_files": len(list((args.output_dir / "audio").glob("*.wav"))),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
