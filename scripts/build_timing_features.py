#!/usr/bin/env python3
"""Export coarse timing features from an explicitly configured local dataset.

Source databases are read-only; derived tables and linkage material are written
under the output directory. Coarsening and pseudonyms do not establish anonymity:
review consent, disclosure risk, and every output before any redistribution.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import hmac
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TTA_FEATURE_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = ROOT / "data"
ORCH = DATA / "orchestration_state.json"
STUDY = DATA / "study.sqlite3"
CONSENT = DATA / "consent.sqlite3"
OUT = ROOT / "dataset"
FFMPEG = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg") or "ffmpeg"
OPENSSL = os.environ.get("OPENSSL_BINARY") or shutil.which("openssl") or "openssl"
EBML_MAGIC = bytes.fromhex("1a45dfa3")
ARMS = ("cold_learn", "shared_learn", "shared_frozen")
# Exclusions belong to a private analysis configuration, never to public source.
DROP_HINTS = {value.strip() for value in os.environ.get("TTA_EXCLUDED_CODE_HINTS", "").split(",") if value.strip()}
SR = 16000
WIN_MS = 250
WIN = SR * WIN_MS // 1000
FRAME_FIELDS = (
    "participant_id",
    "session_id",
    "split",
    "cohort",
    "condition",
    "rel_ms",
    "block",
    "analysis_eligible",
    "log_rms",
    "zcr",
    "centroid_hz",
    "f0_hz",
    "voicing",
    "mfcc_c0",
    "mfcc_c1",
    "mfcc_c2",
    "speaking",
    "decision_eligible",
    "model_p",
    "auto_activate",
    "manual_activate",
    "interrupt",
    "behavior_correction",
)
EVENT_FIELDS = (
    "participant_id",
    "session_id",
    "split",
    "cohort",
    "condition",
    "rel_ms",
    "block",
    "event_type",
    "decision_eligible",
    "model_p",
    "auto_activate",
    "manual_activate",
    "interrupt",
    "behavior_correction",
    "analysis_eligible",
)


def hmac_id(pepper: bytes, kind: str, value: str) -> str:
    digest = hmac.new(pepper, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"{kind[:1]}{digest[:16]}"


def latest_consents(conn: sqlite3.Connection, pids: list[str]) -> dict[str, dict]:
    if not pids:
        return {}
    rows = conn.execute(
        "SELECT * FROM consents WHERE participant_id IN ({}) ORDER BY signed_ms DESC".format(
            ",".join("?" * len(pids))
        ),
        pids,
    ).fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        pid = str(row["participant_id"])
        if pid not in out:
            out[pid] = {k: row[k] for k in row.keys()}
    return out


def starts_with_ebml(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == EBML_MAGIC


def segment_files(files: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    for item in files:
        path = DATA / str(item["relative_path"])
        if not path.is_file():
            continue
        if not current or starts_with_ebml(path):
            if current:
                segments.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        segments.append(current)
    return segments


def decode_segment(files: list[dict]) -> np.ndarray:
    blob = b"".join((DATA / str(item["relative_path"])).read_bytes() for item in files)
    if not blob:
        return np.zeros(0, np.float32)
    timeout = max(120, 4 * len(files))
    try:
        proc = subprocess.run(
            [
                FFMPEG,
                "-nostdin",
                "-hide_banner",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=blob,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return np.zeros(0, np.float32)
    del blob
    if not proc.stdout:
        return np.zeros(0, np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def frame_feats(x: np.ndarray) -> dict[str, float]:
    empty = {
        "log_rms": -8.0,
        "zcr": 0.0,
        "centroid_hz": 0.0,
        "f0_hz": 0.0,
        "voicing": 0.0,
        "mfcc_c0": 0.0,
        "mfcc_c1": 0.0,
        "mfcc_c2": 0.0,
        "speaking": 0,
    }
    if x.size < 32 or not np.any(np.isfinite(x)):
        return empty
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(x * x)))
    log_rms = math.log10(rms + 1e-8)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    win = x * np.hanning(x.size)
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    denom = float(spec.sum()) + 1e-12
    centroid = float((spec * freqs).sum() / denom)
    f0 = 0.0
    voicing = 0.0
    min_lag = int(SR / 400)
    max_lag = int(SR / 80)
    if x.size > max_lag + 1:
        xc = x - float(x.mean())
        ac = np.correlate(xc, xc, mode="full")[x.size - 1 :]
        if ac[0] > 1e-12:
            seg = ac[min_lag : max_lag + 1]
            peak = int(np.argmax(seg))
            lag = peak + min_lag
            voicing = float(max(0.0, min(1.0, seg[peak] / ac[0])))
            if voicing >= 0.35:
                f0 = float(SR / lag)
    n_fft = 1 << int(math.ceil(math.log2(max(x.size, 512))))
    mag = np.abs(np.fft.rfft(win, n=n_fft)) ** 2
    n_mels = 20
    mel_f = np.linspace(0, 2595 * math.log10(1 + (SR / 2) / 700), n_mels + 2)
    hz = 700 * (10 ** (mel_f / 2595) - 1)
    bins = np.floor((n_fft / 2 + 1) * hz / (SR / 2)).astype(int)
    bins = np.clip(bins, 0, mag.size - 1)
    fb = np.zeros((n_mels, mag.size), np.float32)
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        if center > left:
            fb[m, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if right > center:
            fb[m, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    mel = np.log(fb @ mag + 1e-10)
    n = n_mels
    dct = np.zeros(3, np.float32)
    for k in range(3):
        dct[k] = np.sqrt(2.0 / n) * np.sum(
            mel * np.cos(math.pi * k * (np.arange(n) + 0.5) / n)
        )
    speaking = 1 if log_rms > -2.6 and zcr < 0.35 else 0
    return {
        "log_rms": log_rms,
        "zcr": zcr,
        "centroid_hz": centroid,
        "f0_hz": f0,
        "voicing": voicing,
        "mfcc_c0": float(dct[0]),
        "mfcc_c1": float(dct[1]),
        "mfcc_c2": float(dct[2]),
        "speaking": speaking,
    }


def gather_people() -> list[dict]:
    orch = json.loads(ORCH.read_text(encoding="utf-8"))
    share = orch.get("share_fit") or {}
    contrast = orch.get("contrast") or {}
    donors = [str(pid) for pid in (share.get("completed_donor_ids") or [])]
    arm_of = {
        str(item["participant_id"]): item["arm"]
        for item in contrast.get("assignments") or []
        if isinstance(item, dict) and item.get("arm") in ARMS
    }
    e2e = [
        str(pid)
        for pid in (contrast.get("completed_participant_ids") or [])
        if str(pid) in arm_of
    ]
    study = sqlite3.connect(f"file:{STUDY}?mode=ro", uri=True)
    study.row_factory = sqlite3.Row
    all_ids = list(dict.fromkeys(donors + e2e))
    hints = {
        str(row["id"]): str(row["code_hint"] or "")
        for row in study.execute(
            "SELECT id, code_hint FROM participants WHERE id IN ({})".format(
                ",".join("?" * len(all_ids))
            ),
            all_ids,
        )
    }
    study.close()
    consent = sqlite3.connect(f"file:{CONSENT}?mode=ro", uri=True)
    consent.row_factory = sqlite3.Row
    cons = latest_consents(consent, all_ids)
    consent.close()
    people = []
    for pid in donors:
        c = cons.get(pid) or {}
        if not int(c.get("participation") or 0) or not int(c.get("public_derived") or 0):
            continue
        people.append({"pid": pid, "hint": hints.get(pid, ""), "cohort": "share_fit", "arm": ""})
    for pid in e2e:
        hint = hints.get(pid, "")
        if hint in DROP_HINTS:
            continue
        c = cons.get(pid) or {}
        if not int(c.get("participation") or 0) or not int(c.get("public_derived") or 0):
            continue
        people.append(
            {"pid": pid, "hint": hint, "cohort": "contrast", "arm": arm_of[pid]}
        )
    return people


def session_audio(study: sqlite3.Connection, pid: str) -> dict[str, list[dict]]:
    rows = study.execute(
        "SELECT id, session_id, relative_path, created_ms FROM audio_files "
        "WHERE participant_id=? AND kind='study_stream' ORDER BY created_ms",
        (pid,),
    ).fetchall()
    by: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by[str(row["session_id"] or "")].append(dict(row))
    return by


def session_start(study: sqlite3.Connection, sid: str) -> int:
    row = study.execute("SELECT started_ms FROM sessions WHERE id=?", (sid,)).fetchone()
    return int(row["started_ms"] or 0) if row else 0


def load_events(study: sqlite3.Connection, pid: str) -> list[dict]:
    rows = study.execute(
        "SELECT session_id, event_type, server_ms, payload FROM events "
        "WHERE participant_id=? AND event_type IN "
        "('model.prediction','tap.manual_activate','tap.interrupt','agent.auto_activate') "
        "ORDER BY server_ms",
        (pid,),
    ).fetchall()
    out = []
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        if payload.get("cancelled") is True:
            continue
        try:
            block = int(payload.get("active_block_index", payload.get("block", 0)) or 0)
        except (TypeError, ValueError):
            block = 0
        out.append(
            {
                "sid": str(row["session_id"] or ""),
                "et": str(row["event_type"]),
                "ms": int(row["server_ms"] or 0),
                "eligible": payload.get("decision_eligible") is True
                and not payload.get("skipped"),
                "p": payload.get("probability"),
                "analysis": payload.get("analysis_eligible") is not False,
                "block": block,
                "corr": payload.get("training_eligible") is True
                or payload.get("quarantine_reason") == "activation_not_learning_current",
            }
        )
    return out


def window_index(events: list[dict], sid: str, origin: int, n_win: int) -> list[list[dict]]:
    buckets: list[list[dict]] = [[] for _ in range(n_win)]
    if origin <= 0 or n_win <= 0:
        return buckets
    for ev in events:
        if ev["sid"] != sid:
            continue
        i = (int(ev["ms"]) - origin) // WIN_MS
        if 0 <= i < n_win:
            buckets[i].append(ev)
    return buckets


def labels_from(window_events: list[dict]) -> dict:
    eligible = 0
    analysis = 0
    auto = act = intr = corr = 0
    p_val = ""
    block = 0
    for ev in window_events:
        if ev["analysis"]:
            analysis = 1
        if ev["block"] in {1, 2, 3, 4, 5}:
            block = ev["block"]
        if ev["et"] == "model.prediction":
            if ev["eligible"]:
                eligible = 1
            if ev["p"] is not None:
                try:
                    p_val = f"{float(ev['p']):.6f}"
                except (TypeError, ValueError):
                    pass
        elif ev["et"] == "agent.auto_activate":
            auto = 1
        elif ev["et"] == "tap.manual_activate":
            act = 1
            if ev["corr"]:
                corr = 1
        elif ev["et"] == "tap.interrupt":
            intr = 1
            if ev["corr"]:
                corr = 1
    return {
        "analysis_eligible": analysis,
        "block": block,
        "decision_eligible": eligible,
        "model_p": p_val,
        "auto_activate": auto,
        "manual_activate": act,
        "interrupt": intr,
        "behavior_correction": corr,
    }


def split_of(export_id: str) -> str:
    bucket = int(export_id[-1], 16) % 10
    if bucket <= 6:
        return "train"
    if bucket <= 8:
        return "val"
    return "test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_card(path: Path, payload: dict) -> None:
    text = f"""# Tap-to-Adapt public derived timing features

Version: 2026-09-03
Location: ./dataset/release/

## Contents
- frames.csv.gz: 250 ms coarse timing frames aligned to interaction labels
- events.csv.gz: activate / interrupt / auto-activate / eligible prediction events
- participants.csv: one row per contributor, HMAC export IDs only
- stats.json: aggregate counts used by the study document
- this card and MANIFEST.json (SHA-256)

## Privacy transform
Waveforms, Mel spectrograms, full MFCC, HuBERT, ASR text, absolute timestamps,
IP, file paths, and study UUIDs are not stored. Each 250 ms frame keeps only
log-energy, zero-crossing rate, spectral centroid, autocorrelation F0 (80-400 Hz),
voicing, and MFCC c0-c2. These features may retain speaker or linguistic
information; anonymity and resistance to reconstruction have not been established.
IDs use HMAC-SHA256 with an owner-only export pepper outside the bundle. The shareable archive is AES-256-CBC (PBKDF2).

## Contributors
- n_participants: {payload.get("n_participants")}
- share_fit: {payload.get("n_share_fit")}
- contrast: {payload.get("n_contrast")}
- contrast_by_arm: {json.dumps(payload.get("contrast_by_arm"), ensure_ascii=False)}

## Scale
- n_frames: {payload.get("n_frames")}
- audio_hours: {payload.get("audio_hours")}
- n_audio_files_read: {payload.get("n_audio_files_read")}
- n_events: {payload.get("n_events")}

## Label totals (frame-aligned)
- speaking: {payload.get("n_speaking_frames")}
- decision_eligible: {payload.get("n_decision_eligible_frames")}
- manual_activate: {payload.get("n_manual_activate_frames")}
- interrupt: {payload.get("n_interrupt_frames")}
- auto_activate: {payload.get("n_auto_activate_frames")}
- behavior_correction: {payload.get("n_behavior_correction_frames")}

## License / access
Owner-held derived release. Do not copy .aes_key, .export_pepper, or .id_map.json.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "release").mkdir(exist_ok=True)
    (OUT / "tools").mkdir(exist_ok=True)
    pepper_path = OUT / ".export_pepper"
    key_path = OUT / ".aes_key"
    if pepper_path.exists():
        pepper = pepper_path.read_bytes().strip()
    else:
        pepper_path.write_bytes(os.urandom(32).hex().encode() + b"\n")
        os.chmod(pepper_path, 0o600)
        pepper = pepper_path.read_bytes().strip()
    pepper_b = bytes.fromhex(pepper.decode().strip())
    if not key_path.exists():
        key_path.write_text(os.urandom(32).hex() + "\n", encoding="ascii")
        os.chmod(key_path, 0o600)
    people = gather_people()
    study = sqlite3.connect(f"file:{STUDY}?mode=ro", uri=True)
    study.row_factory = sqlite3.Row
    raw_csv = OUT / "release" / "frames.csv.gz"
    ev_csv = OUT / "release" / "events.csv.gz"
    part_csv = OUT / "release" / "participants.csv"
    id_map = []
    stats = Counter()
    arm_people = Counter()
    split_people = Counter()
    event_type_counts = Counter()
    with gzip.open(raw_csv, "wt", encoding="utf-8", newline="") as fh_frames, gzip.open(
        ev_csv, "wt", encoding="utf-8", newline=""
    ) as fh_events:
        frame_w = csv.DictWriter(fh_frames, fieldnames=FRAME_FIELDS)
        event_w = csv.DictWriter(fh_events, fieldnames=EVENT_FIELDS)
        frame_w.writeheader()
        event_w.writeheader()
        part_rows = []
        for idx, person in enumerate(people, start=1):
            pid = person["pid"]
            export_pid = hmac_id(pepper_b, "participant", pid)
            split = split_of(export_pid)
            events = load_events(study, pid)
            audio_by = session_audio(study, pid)
            n_frames = 0
            n_eligible = n_act = n_intr = n_corr = n_speak = n_auto = 0
            n_files = 0
            n_fail = 0
            n_seconds = 0.0
            n_events = 0
            starts: dict[str, int] = {}
            for sid, files in audio_by.items():
                export_sid = hmac_id(pepper_b, "session", sid)
                started_ms = session_start(study, sid)
                starts[sid] = started_ms
                for segment in segment_files(files):
                    wav = decode_segment(segment)
                    if wav.size == 0:
                        n_fail += len(segment)
                        continue
                    n_files += len(segment)
                    n_seconds += wav.size / SR
                    origin = int(segment[0]["created_ms"] or started_ms or 0)
                    n_win = wav.size // WIN
                    indexed = window_index(events, sid, origin, n_win)
                    for i in range(n_win):
                        sl = wav[i * WIN : (i + 1) * WIN]
                        feat = frame_feats(sl)
                        t0 = origin + i * WIN_MS
                        lab = labels_from(indexed[i])
                        rel = t0 - started_ms if started_ms else i * WIN_MS
                        frame_w.writerow(
                            {
                                "participant_id": export_pid,
                                "session_id": export_sid,
                                "split": split,
                                "cohort": person["cohort"],
                                "condition": person["arm"] or "share_fit",
                                "rel_ms": max(int(rel), 0),
                                "block": lab["block"],
                                "analysis_eligible": lab["analysis_eligible"],
                                "log_rms": f"{feat['log_rms']:.5f}",
                                "zcr": f"{feat['zcr']:.5f}",
                                "centroid_hz": f"{feat['centroid_hz']:.2f}",
                                "f0_hz": f"{feat['f0_hz']:.2f}",
                                "voicing": f"{feat['voicing']:.4f}",
                                "mfcc_c0": f"{feat['mfcc_c0']:.5f}",
                                "mfcc_c1": f"{feat['mfcc_c1']:.5f}",
                                "mfcc_c2": f"{feat['mfcc_c2']:.5f}",
                                "speaking": feat["speaking"],
                                "decision_eligible": lab["decision_eligible"],
                                "model_p": lab["model_p"],
                                "auto_activate": lab["auto_activate"],
                                "manual_activate": lab["manual_activate"],
                                "interrupt": lab["interrupt"],
                                "behavior_correction": lab["behavior_correction"],
                            }
                        )
                        n_frames += 1
                        n_eligible += lab["decision_eligible"]
                        n_act += lab["manual_activate"]
                        n_intr += lab["interrupt"]
                        n_auto += lab["auto_activate"]
                        n_corr += lab["behavior_correction"]
                        n_speak += feat["speaking"]
                    del wav
            for ev in events:
                sid = ev["sid"]
                if sid not in starts:
                    starts[sid] = session_start(study, sid)
                started_ms = starts[sid]
                export_sid = hmac_id(pepper_b, "session", sid)
                p_val = ""
                if ev["p"] is not None:
                    try:
                        p_val = f"{float(ev['p']):.6f}"
                    except (TypeError, ValueError):
                        p_val = ""
                event_w.writerow(
                    {
                        "participant_id": export_pid,
                        "session_id": export_sid,
                        "split": split,
                        "cohort": person["cohort"],
                        "condition": person["arm"] or "share_fit",
                        "rel_ms": max(int(ev["ms"] - started_ms), 0) if started_ms else 0,
                        "block": ev["block"] if ev["block"] in {1, 2, 3, 4, 5} else 0,
                        "event_type": ev["et"],
                        "decision_eligible": int(ev["eligible"]),
                        "model_p": p_val,
                        "auto_activate": int(ev["et"] == "agent.auto_activate"),
                        "manual_activate": int(ev["et"] == "tap.manual_activate"),
                        "interrupt": int(ev["et"] == "tap.interrupt"),
                        "behavior_correction": int(
                            ev["et"] in {"tap.manual_activate", "tap.interrupt"} and ev["corr"]
                        ),
                        "analysis_eligible": int(ev["analysis"]),
                    }
                )
                n_events += 1
                event_type_counts[ev["et"]] += 1
            stats["people"] += 1
            stats["frames"] += n_frames
            stats["files"] += n_files
            stats["fail"] += n_fail
            stats["eligible"] += n_eligible
            stats["act"] += n_act
            stats["intr"] += n_intr
            stats["auto"] += n_auto
            stats["corr"] += n_corr
            stats["speak"] += n_speak
            stats["seconds"] += int(n_seconds)
            stats["events"] += n_events
            if person["cohort"] == "contrast" and person["arm"]:
                arm_people[person["arm"]] += 1
            split_people[split] += 1
            part_rows.append(
                {
                    "participant_id": export_pid,
                    "split": split,
                    "cohort": person["cohort"],
                    "condition": person["arm"] or "share_fit",
                    "n_frames": n_frames,
                    "n_audio_files": n_files,
                    "n_audio_fail": n_fail,
                    "audio_seconds": int(n_seconds),
                    "n_speaking": n_speak,
                    "n_decision_eligible": n_eligible,
                    "n_manual_activate": n_act,
                    "n_interrupt": n_intr,
                    "n_auto_activate": n_auto,
                    "n_behavior_correction": n_corr,
                    "n_events": n_events,
                }
            )
            id_map.append(
                {
                    "hint": person["hint"],
                    "cohort": person["cohort"],
                    "condition": person["arm"] or "share_fit",
                    "export_id": export_pid,
                }
            )
            print(
                f"[{idx}/{len(people)}] {person['cohort']} {person['hint']} "
                f"frames={n_frames} files={n_files} fail={n_fail} "
                f"sec={int(n_seconds)} events={n_events}",
                flush=True,
            )
    study.close()
    with part_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(part_rows[0].keys()))
        writer.writeheader()
        writer.writerows(part_rows)
    map_path = OUT / ".id_map.json"
    map_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(map_path, 0o600)
    payload = {
        "window_ms": WIN_MS,
        "sample_rate": SR,
        "n_participants": stats["people"],
        "n_share_fit": sum(1 for p in people if p["cohort"] == "share_fit"),
        "n_contrast": sum(1 for p in people if p["cohort"] == "contrast"),
        "contrast_by_arm": {arm: int(arm_people[arm]) for arm in ARMS},
        "split_people": dict(split_people),
        "n_frames": stats["frames"],
        "n_audio_files_read": stats["files"],
        "n_audio_decode_fail": stats["fail"],
        "audio_hours": round(stats["seconds"] / 3600, 3),
        "n_speaking_frames": stats["speak"],
        "n_decision_eligible_frames": stats["eligible"],
        "n_manual_activate_frames": stats["act"],
        "n_interrupt_frames": stats["intr"],
        "n_auto_activate_frames": stats["auto"],
        "n_behavior_correction_frames": stats["corr"],
        "n_events": stats["events"],
        "event_type_counts": dict(event_type_counts),
        "elapsed_sec": int(time.time() - started),
        "ffmpeg": FFMPEG,
        "isolated_from": [
            "data/study.sqlite3",
            "data/consent.sqlite3",
            "data/orchestration_state.json",
            "data/media",
            "current-formal",
        ],
    }
    stats_path = OUT / "release" / "stats.json"
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    card_path = OUT / "release" / "DATASET_CARD.md"
    write_dataset_card(card_path, payload)
    public_names = (
        "frames.csv.gz",
        "events.csv.gz",
        "participants.csv",
        "stats.json",
        "DATASET_CARD.md",
    )
    manifest = {
        name: {"sha256": sha256_file(OUT / "release" / name), "bytes": (OUT / "release" / name).stat().st_size}
        for name in public_names
    }
    man_path = OUT / "release" / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = OUT / "release" / "README.txt"
    readme.write_text(
        """Tap-to-Adapt public timing-feature release (server-side)
======================================================

This directory is isolated from study runtime:
  data/study.sqlite3, data/orchestration_state.json, data/media, and
  current-formal/ are never written by the builder.

Content protection
------------------
Original waveforms, Mel spectrograms, full MFCCs, HuBERT embeddings,
ASR text, absolute timestamps, IP, and participant UUIDs are not stored
here. Each 250 ms frame keeps only coarse timing cues (log-energy, ZCR,
spectral centroid, F0, voicing, MFCC c0-c2). These cannot reconstruct
intelligible conversation.

At-rest encryption
------------------
tap_to_adapt_timing_features.tar.aes256 is AES-256-CBC (PBKDF2).
Unlock with the owner-only key file ../.aes_key:

  openssl enc -d -aes-256-cbc -pbkdf2 -in tap_to_adapt_timing_features.tar.aes256 \\
    -out tap_to_adapt_timing_features.tar -pass file:../.aes_key
  tar -xf tap_to_adapt_timing_features.tar

Do not copy .aes_key, .export_pepper, or .id_map.json into any public bundle.
""",
        encoding="utf-8",
    )
    archive = OUT / "release" / "tap_to_adapt_timing_features.tar"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w") as tar:
        for name in public_names + ("MANIFEST.json", "README.txt"):
            tar.add(OUT / "release" / name, arcname=name)
    enc = Path(str(archive) + ".aes256")
    if enc.exists():
        enc.unlink()
    subprocess.run(
        [
            OPENSSL,
            "enc",
            "-aes-256-cbc",
            "-pbkdf2",
            "-salt",
            "-in",
            str(archive),
            "-out",
            str(enc),
            "-pass",
            f"file:{key_path}",
        ],
        check=True,
    )
    archive.unlink()
    for path in (enc, raw_csv, ev_csv, part_csv, stats_path, card_path, man_path, readme):
        os.chmod(path, 0o640)
    os.chmod(key_path, 0o600)
    os.chmod(pepper_path, 0o600)
    print("STATS", json.dumps(payload), flush=True)
    print("ENC", enc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
