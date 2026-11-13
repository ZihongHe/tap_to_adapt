"""Freeze silent multi-voice PCM through the public experiment route.

Participant and agent speech use mutually exclusive, role-locked local AISHELL3
speakers. No waveform is sent to an audio output device. The per-participant files are resumable and can be
replayed by the real-time duration runner without another TTS request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .run_smooth_duration_simulation_public import (
        _load_frozen_schedule,
        _synthesize_schedule,
    )
    from .simulate_feasibility import ensure_consent, login, request_json
    from .smooth_simulation_protocol import SCHEMA, validate_schedule
except ImportError:
    from run_smooth_duration_simulation_public import (
        _load_frozen_schedule,
        _synthesize_schedule,
    )
    from simulate_feasibility import ensure_consent, login, request_json
    from smooth_simulation_protocol import SCHEMA, validate_schedule


def build_participant(
    *,
    base_url: str,
    roster: dict[str, str],
    participant_turns: dict[str, Any],
    bundle_root: Path,
    max_minutes: int,
) -> dict[str, Any]:
    events = list(participant_turns["events"])
    validate_schedule(events, max_minutes=max_minutes)
    participant_dir = (
        bundle_root / f"participant-{int(roster['participant_index']):02d}"
    )
    if participant_dir.is_dir():
        _, audit = _load_frozen_schedule(
            bundle_root=bundle_root,
            roster=roster,
            events=events,
        )
        return {
            "participant_index": int(roster["participant_index"]),
            "profile_id": participant_turns["profile"]["profile_id"],
            "cached": True,
            **audit,
        }

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
    try:
        _, audit = _synthesize_schedule(
            base_url=base_url,
            study_token=study_token,
            study_session_id=str(study_session["id"]),
            companion_token=companion_token,
            companion_session_id=str(companion_session["id"]),
            roster=roster,
            events=events,
            bundle_root=bundle_root,
        )
        if audit is None:
            raise RuntimeError("PCM bundle writer did not return an audit")
        return {
            "participant_index": int(roster["participant_index"]),
            "profile_id": participant_turns["profile"]["profile_id"],
            "cached": False,
            **audit,
        }
    finally:
        for token, session in (
            (study_token, study_session),
            (companion_token, companion_session),
        ):
            try:
                request_json(
                    base_url,
                    f"/api/sessions/{session['id']}/abort",
                    method="POST",
                    token=token,
                )
            except Exception:
                pass


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
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be in [1, 8]")
    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = list(csv.DictReader(handle))
    turns = json.loads(args.turns.read_text(encoding="utf-8"))
    if turns.get("schema") != SCHEMA:
        raise SystemExit("Turns are not the intent-aware simulation schema")
    if turns.get("cohort") not in {"selection8", "duration32"}:
        raise SystemExit("Turns cohort must be selection8 or duration32")
    expected = 8 if turns["cohort"] == "selection8" else 32
    if len(roster) != expected or len({row["voice_id"] for row in roster}) != expected:
        raise SystemExit("Roster size or distinct local voice count is invalid")
    by_id = {str(row["participant_id"]): row for row in turns["participants"]}
    if {row["participant_id"] for row in roster} != set(by_id):
        raise SystemExit("Roster and turns participant IDs do not match")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                build_participant,
                base_url=args.base_url.rstrip("/"),
                roster=row,
                participant_turns=by_id[row["participant_id"]],
                bundle_root=args.output_dir,
                max_minutes=int(turns["max_minutes"]),
            ): int(row["participant_index"])
            for row in roster
        }
        for future in as_completed(futures):
            audit = future.result()
            audits.append(audit)
            print(
                json.dumps(
                    {
                        "participant_index": audit["participant_index"],
                        "profile_id": audit["profile_id"],
                        "cached": audit["cached"],
                        "entries": audit["entries"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    audits.sort(key=lambda row: int(row["participant_index"]))
    root_manifest = {
        "schema": "tap-to-adapt-frozen-pcm-bundle/v1",
        "simulation_only": True,
        "silent_playback": True,
        "run_slug": turns["run_slug"],
        "cohort": turns["cohort"],
        "participants": expected,
        "local_participant_voices": expected,
        "experimental_agent_voice": roster[0]["experimental_agent_voice_name"],
        "companion_agent_voice": roster[0]["companion_agent_voice_name"],
        "source_turns_sha256": hashlib.sha256(args.turns.read_bytes()).hexdigest(),
        "profiles": audits,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
