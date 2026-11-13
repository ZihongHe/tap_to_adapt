"""Regression tests for smooth simulation protocol.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

from app.config import Settings
from app.timing import ParticipantRuntime
from scripts.smooth_simulation_protocol import (
    build_schedule,
    load_llm_bank,
    validate_schedule,
)
from scripts.build_voice_calibrated_smooth_turns import (
    build_profile_schedule,
    records_from_shared_scores,
    select_slot_candidates,
)
from scripts import run_smooth_duration_simulation_public as public_runner


def source_pools() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "model_selection_outputs"
        / "llm_tts_text_pools.json"
    )


def test_pair_activation_retries_only_presence_propagation(monkeypatch) -> None:
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise public_runner.SimulationError(
                "POST activate failed: HTTP 409 同时在线，companion ready"
            )
        return {"ok": True}

    monkeypatch.setattr(public_runner, "request_json", request)
    monkeypatch.setattr(public_runner.time, "sleep", lambda _seconds: None)

    result = public_runner.activate_after_pair_ready(
        "https://example.test", "token", "session"
    )

    assert result == {"ok": True}
    assert len(calls) == 2


def test_pair_activation_does_not_hide_other_conflicts(monkeypatch) -> None:
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        raise public_runner.SimulationError(
            "POST activate failed: HTTP 409 session already closed"
        )

    monkeypatch.setattr(public_runner, "request_json", request)

    try:
        public_runner.activate_after_pair_ready(
            "https://example.test", "token", "session"
        )
    except public_runner.SimulationError as error:
        assert "already closed" in str(error)
    else:
        raise AssertionError("Non-presence conflicts must fail closed")
    assert len(calls) == 1


def test_blocking_agent_call_keeps_timing_socket_serviced(monkeypatch) -> None:
    serviced = []
    socket = object()

    def service(actual_socket, seconds):
        assert actual_socket is socket
        serviced.append(seconds)
        time.sleep(0.005)

    def callback():
        time.sleep(0.025)
        return "complete"

    monkeypatch.setattr(
        public_runner, "wait_while_servicing_timing_socket", service
    )

    result = public_runner.call_while_servicing_timing_socket(socket, callback)

    assert result == "complete"
    assert len(serviced) >= 2


def test_pair_reconnect_helper_retries_and_starts_heartbeat(monkeypatch) -> None:
    calls = []
    heartbeat_instances = []
    study_socket = object()
    companion_socket = object()

    def pair(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("transient pair transport failure")
        return study_socket, companion_socket

    class Heartbeat:
        def __init__(self, *sockets):
            self.sockets = sockets
            self.started = False
            heartbeat_instances.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(public_runner, "pair_websockets", pair)
    monkeypatch.setattr(public_runner, "PairHeartbeat", Heartbeat)
    monkeypatch.setattr(public_runner.time, "sleep", lambda _seconds: None)

    actual_study, actual_companion, heartbeat = (
        public_runner.connect_pair_with_heartbeat(
            "https://example.test",
            "study-token",
            "study-session",
            "companion-token",
            "companion-session",
        )
    )

    assert actual_study is study_socket
    assert actual_companion is companion_socket
    assert heartbeat is heartbeat_instances[0]
    assert heartbeat.sockets == (study_socket, companion_socket)
    assert heartbeat.started is True
    assert len(calls) == 2


def test_smooth_schedule_is_exactly_balanced_and_deterministic() -> None:
    bank = load_llm_bank(source_pools())
    profile, first = build_schedule(
        participant_id="p-fixed",
        participant_index=1,
        cohort="duration32",
        max_minutes=72,
        bank=bank,
    )
    _, second = build_schedule(
        participant_id="p-fixed",
        participant_index=1,
        cohort="duration32",
        max_minutes=72,
        bank=bank,
    )
    assert profile.profile_id == "PR01"
    assert first == second
    assert len(first) == 12 * 12
    validate_schedule(first, max_minutes=72)
    for cycle in range(1, 13):
        rows = [row for row in first if row["cycle_index"] == cycle]
        assert sum(row["phase"] == "primary" for row in rows) == 8
        assert sum(row["phase"] == "companion" for row in rows) == 4
        assert sum(row["label"] for row in rows) == 6
        assert {
            band: sum(row["difficulty_band"] == band for row in rows)
            for band in ("low", "medium", "high")
        } == {"low": 4, "medium": 4, "high": 4}


def test_frozen_glm_dialogue_preserves_intent_and_topic_cycle(
    tmp_path: Path,
) -> None:
    primary_bands = [
        "medium",
        "high",
        "low",
        "low",
        "high",
        "low",
        "medium",
        "high",
    ]
    companion_bands = ["low", "medium", "high", "medium"]
    payload = {
        "schema": "tap-to-adapt-glm-intent-dialogues/v1",
        "generator": {"provider": "OpenRouter", "model_id": "z-ai/glm-5.2"},
        "participants": [
            {
                "profile_id": "DV01",
                "cycles": [
                    {
                        "cycle_index": 1,
                        "topic_id": "ordinary_life_01",
                        "topic_prompt": "你平时一天的作息是怎样的？",
                        "topic_status": "ended_at_break",
                        "primary": [
                            {
                                "text": f"主对话{index}",
                                "response_expected": expected,
                                "difficulty_band": band,
                            }
                            for index, (expected, band) in enumerate(
                                zip(
                                    [
                                        True,
                                        True,
                                        False,
                                        True,
                                        True,
                                        False,
                                        True,
                                        True,
                                    ],
                                    primary_bands,
                                    strict=True,
                                ),
                                1,
                            )
                        ],
                        "companion": [
                            {
                                "speaker_role": role,
                                "text": f"陪伴对话{index}",
                                "difficulty_band": band,
                            }
                            for index, (role, band) in enumerate(
                                zip(
                                    [
                                        "participant",
                                        "companion",
                                        "participant",
                                        "companion",
                                    ],
                                    companion_bands,
                                    strict=True,
                                ),
                                1,
                            )
                        ],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "glm-dialogues.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    bank = load_llm_bank(path)
    profile, events = build_schedule(
        participant_id="p-dev",
        participant_index=1,
        cohort="selection8",
        max_minutes=6,
        bank=bank,
    )
    assert profile.profile_id == "DV01"
    assert len(events) == 12
    assert [row["response_expected"] for row in events[:8]] == [
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert all(
        row["interaction_intent"] == "talk_to_companion"
        for row in events[8:]
    )
    assert {row["topic_id"] for row in events} == {"ordinary_life_01"}
    assert all(1.0 <= row["reply_deadline_seconds"] <= 3.0 for row in events)


def test_duration_cohort_has_identical_stimulus_multiset_in_every_cycle() -> None:
    bank = load_llm_bank(source_pools())
    schedules = [
        build_schedule(
            participant_id=f"p-{participant_index:02d}",
            participant_index=participant_index,
            cohort="duration32",
            max_minutes=72,
            bank=bank,
        )[1]
        for participant_index in range(1, 33)
    ]
    reference = None
    for cycle in range(1, 13):
        multiset = sorted(
            (
                row["category"],
                row["difficulty_band"],
                row["text"],
                int(row["label"]),
                row["phase"],
            )
            for schedule in schedules
            for row in schedule
            if int(row["cycle_index"]) == cycle
        )
        if reference is None:
            reference = multiset
        else:
            assert multiset == reference


def test_voice_calibration_retains_each_stimulus_once_per_slot() -> None:
    bank = load_llm_bank(source_pools())
    _, events = build_schedule(
        participant_id="p-calibration",
        participant_index=1,
        cohort="duration32",
        max_minutes=72,
        bank=bank,
    )
    records = []
    for event in events:
        cycle = int(event["cycle_index"])
        hardness = 0.05 * cycle
        probability = (
            1.0 - hardness if int(event["label"]) == 1 else hardness
        )
        records.append(
            {
                "opportunity_index": event["opportunity_index"],
                "label": event["label"],
                "shared_probability": probability,
                "text_sha256": hashlib.sha256(
                    event["text"].encode("utf-8")
                ).hexdigest(),
            }
    )
    selected = select_slot_candidates(events, records)
    assert set(selected) == set(range(12))
    assert all(len(rows) == 12 for rows in selected.values())
    calibrated, hardness = build_profile_schedule(
        participant_index=1,
        selected=selected,
    )
    assert len(calibrated) == 144
    assert len(hardness) == 12
    assert all(row["voice_calibrated"] is True for row in calibrated)
    assert len({row["opportunity_index"] for row in calibrated}) == 144
    for slot in range(12):
        assert len(
            {
                row["text_pool_index"]
                for row in calibrated
                if (int(row["opportunity_index"]) - 1) % 12 == slot
            }
        ) == len(
            {
                row["text_pool_index"]
                for row in events
                if (int(row["opportunity_index"]) - 1) % 12 == slot
            }
        )
    validate_schedule(calibrated, max_minutes=72)


def test_empirical_difficulty_map_can_reassign_without_empty_strata(
    tmp_path: Path,
) -> None:
    lexical = load_llm_bank(source_pools())
    rows = []
    for category, bands in lexical.items():
        for band, texts in bands.items():
            for text in texts[:2]:
                rows.append(
                    {
                        "category": category,
                        "text": text,
                        "empirical_difficulty_band": band,
                    }
                )
    path = tmp_path / "difficulty.json"
    path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    empirical = load_llm_bank(source_pools(), path)
    assert set(empirical) == set(lexical)
    assert all(
        empirical[category][band]
        for category in empirical
        for band in ("low", "medium", "high")
    )


def test_simulated_gap_clears_audio_and_requires_waiting() -> None:
    settings = Settings.from_env()
    runtime = ParticipantRuntime(
        participant_id="p",
        group_name="personalized",
        cohort="pilot",
        settings=settings,
    )
    import numpy as np

    runtime.ring.append(np.ones(8000, dtype=np.float32))
    result = runtime.reset_audio_after_simulated_gap()
    assert result["discarded_samples"] == 8000
    assert runtime.ring.samples == 0
    runtime.agent_state = "thinking"
    try:
        runtime.reset_audio_after_simulated_gap()
    except ValueError as error:
        assert "waiting" in str(error)
    else:
        raise AssertionError("active-state simulated gap must fail closed")


def test_frozen_pcm_bundle_replays_identical_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def fake_request_pcm(
        base_url: str,
        token: str,
        *,
        session_id: str,
        text: str,
        speaker_id: int | None,
        speed: float,
        source: str = "participant",
    ) -> bytes:
        calls.append((text, speaker_id, source))
        return f"{source}:{speaker_id}:{speed}:{text}".encode("utf-8")

    monkeypatch.setattr(public_runner, "request_pcm", fake_request_pcm)
    roster = {
        "participant_index": "1",
        "voice_id": "3",
        "speed": "1.0",
        "companion_agent_voice_name": "aishell3-sid92",
    }
    events = [
        {"speaker_role": "participant", "text": "hello"},
        {"speaker_role": "participant", "text": "hello"},
        {"speaker_role": "companion", "text": "reply"},
    ]
    bundle = tmp_path / "pcm-bundle"
    synthesized, written = public_runner._synthesize_schedule(
        base_url="https://example.test",
        study_token="study-token",
        study_session_id="study-session",
        companion_token="companion-token",
        companion_session_id="companion-session",
        roster=roster,
        events=events,
        bundle_root=bundle,
    )
    loaded, replayed = public_runner._load_frozen_schedule(
        bundle_root=bundle,
        roster=roster,
        events=events,
    )
    assert calls == [
        ("hello", 3, "participant"),
        ("reply", None, "companion"),
    ]
    assert loaded == synthesized
    assert written is not None and written["entries"] == 2
    assert replayed["mode"] == "read"
    assert (
        replayed["participant_manifest_sha256"]
        == written["participant_manifest_sha256"]
    )


def test_full_utterance_score_map_uses_role_and_text_identity() -> None:
    events = [
        {
            "opportunity_index": 1,
            "speaker_role": "target",
            "text": "same text",
            "label": 1,
        },
        {
            "opportunity_index": 2,
            "speaker_role": "companion",
            "text": "same text",
            "label": 0,
        },
    ]
    text_sha256 = hashlib.sha256("same text".encode("utf-8")).hexdigest()
    scores = {}
    for role, probability in (("target", 0.8), ("companion", 0.2)):
        cache_key = hashlib.sha256(
            f"{role}\0same text".encode("utf-8")
        ).hexdigest()
        scores[cache_key] = {
            "participant_index": 1,
            "speaker_role": role,
            "text_sha256": text_sha256,
            "shared_probability": probability,
        }
    records = records_from_shared_scores(
        participant_index=1,
        events=events,
        scores=scores,
    )
    assert [row["shared_probability"] for row in records] == [0.8, 0.2]
