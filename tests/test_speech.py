"""Regression tests for speech.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from typing import Any

import numpy as np

from app.config import settings
from app.speech import SpeechService, SpeechSynthesisError


class FakeXunfeiSocket:
    def __init__(self, messages: list[dict[str, Any]]):
        self.messages = iter(messages)
        self.sent: list[str] = []
        self.closed = False

    def send(self, value: str) -> None:
        self.sent.append(value)

    def recv(self) -> str:
        return json.dumps(next(self.messages))

    def close(self) -> None:
        self.closed = True


def xunfei_settings():
    return replace(
        settings,
        tts_mode="xunfei",
        xunfei_app_id="dummy-app-id",
        xunfei_api_key="dummy-api-key",
        xunfei_api_secret="dummy-api-secret",
        xunfei_voice="xiaoyan",
    )


def test_xunfei_adapter_sends_utf8_lame_request_and_concatenates_chunks(
    monkeypatch,
) -> None:
    socket = FakeXunfeiSocket(
        [
            {
                "code": 0,
                "data": {
                    "status": 1,
                    "audio": base64.b64encode(b"first").decode("ascii"),
                },
            },
            {
                "code": 0,
                "data": {
                    "status": 2,
                    "audio": base64.b64encode(b"second").decode("ascii"),
                },
            },
        ]
    )
    captured: dict[str, Any] = {}

    def connect(url: str, **options: Any) -> FakeXunfeiSocket:
        captured.update(url=url, options=options)
        return socket

    monkeypatch.setattr("app.speech.websocket.create_connection", connect)
    service = SpeechService(xunfei_settings())

    audio = service.synthesize("你好，语音实验。")

    assert audio == b"firstsecond"
    assert socket.closed is True
    assert len(socket.sent) == 1
    request = json.loads(socket.sent[0])
    assert request["common"]["app_id"] == "dummy-app-id"
    assert request["business"] == {
        "aue": "lame",
        "sfl": 1,
        "auf": "audio/L16;rate=16000",
        "vcn": "xiaoyan",
        "speed": 50,
        "volume": 50,
        "pitch": 50,
        "bgs": 0,
        "tte": "UTF8",
    }
    assert base64.b64decode(request["data"]["text"]).decode("utf-8") == "你好，语音实验。"
    assert request["data"]["status"] == 2
    assert captured["options"] == {"timeout": 15}
    assert "dummy-api-secret" not in captured["url"]
    assert "dummy-api-key" not in captured["url"]


def test_xunfei_adapter_closes_socket_and_does_not_echo_credentials_on_error(
    monkeypatch,
) -> None:
    socket = FakeXunfeiSocket(
        [{"code": 11200, "message": "synthetic provider rejection", "data": {}}]
    )
    monkeypatch.setattr(
        "app.speech.websocket.create_connection", lambda *_args, **_kwargs: socket
    )
    service = SpeechService(xunfei_settings())

    try:
        service._synthesize_xunfei("测试".encode("utf-8"))
    except SpeechSynthesisError as error:
        detail = str(error)
    else:
        raise AssertionError("The provider error should have failed the request")

    assert socket.closed is True
    assert "11200" in detail
    assert "dummy-api-key" not in detail
    assert "dummy-api-secret" not in detail


def test_agent_roles_use_distinct_locked_xunfei_voices() -> None:
    service = SpeechService(
        replace(
            xunfei_settings(),
            xunfei_primary_voice="xiaoyan",
            xunfei_companion_voice="xiaoyu",
        )
    )
    assert service.voice_for("study") == "xiaoyan"
    assert service.voice_for("companion") == "xiaoyu"


def test_agent_roles_use_distinct_locked_local_voices() -> None:
    service = SpeechService(
        replace(
            settings,
            tts_mode="sherpa_onnx",
            sherpa_tts_primary_speaker_id=2,
            sherpa_tts_companion_speaker_id=7,
        )
    )
    assert service.provider == "sherpa_onnx_vits_aishell3"
    assert service.voice_for("study") == "aishell3-sid2"
    assert service.voice_for("companion") == "aishell3-sid7"
    assert service.voice_for("study") != service.voice_for("companion")


def test_agent_voices_are_stably_counterbalanced_by_participant() -> None:
    service = SpeechService(
        replace(
            settings,
            tts_mode="sherpa_onnx",
            sherpa_tts_primary_speaker_id=25,
            sherpa_tts_companion_speaker_id=92,
            code_pepper="voice-counterbalance-test-pepper",
        )
    )
    participant_ids = [f"participant-{index}" for index in range(200)]
    assignments = {
        participant_id: service.voice_assignment_for(participant_id)
        for participant_id in participant_ids
    }
    assert set(assignments.values()) == {"AB", "BA"}
    assert assignments == {
        participant_id: service.voice_assignment_for(participant_id)
        for participant_id in participant_ids
    }
    for participant_id, assignment in assignments.items():
        study_voice = service.voice_for("study", participant_id=participant_id)
        companion_voice = service.voice_for("companion", participant_id=participant_id)
        assert study_voice != companion_voice
        if assignment == "AB":
            assert study_voice == "aishell3-sid25"
            assert companion_voice == "aishell3-sid92"
        else:
            assert study_voice == "aishell3-sid92"
            assert companion_voice == "aishell3-sid25"


def test_local_voice_parser_rejects_non_locked_voice_names() -> None:
    service = SpeechService(
        replace(
            settings,
            tts_mode="sherpa_onnx",
            sherpa_tts_primary_speaker_id=2,
            sherpa_tts_companion_speaker_id=7,
        )
    )
    assert service._speaker_id_from_voice("aishell3-sid7") == 7
    try:
        service._speaker_id_from_voice("xiaoyan")
    except SpeechSynthesisError:
        pass
    else:
        raise AssertionError("Non-local voice names must be rejected")


def test_short_local_pcm_is_right_zero_padded_without_repeating_speech() -> None:
    original = np.asarray([100, -200, 300], dtype="<i2")
    result = np.frombuffer(SpeechService._pad_streaming_pcm(original), dtype="<i2")
    assert result.size == SpeechService.MINIMUM_STREAMING_PCM_SAMPLES
    assert np.array_equal(result[:3], original)
    assert np.count_nonzero(result[3:]) == 0
