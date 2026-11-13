"""Regression tests for agent.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
from dataclasses import replace

from app.agent import ChatService
from app.config import settings


def test_openrouter_request_locks_provider_and_disables_reasoning(
    monkeypatch,
) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "好的，我们继续聊。"}}]}
            ).encode("utf-8")

    def urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("app.agent.urllib.request.urlopen", urlopen)
    service = ChatService(
        replace(
            settings,
            llm_api_key="opaque-test-key",
            llm_model="z-ai/glm-5.2",
            llm_provider_order=("z-ai",),
            llm_allow_fallbacks=False,
            llm_reasoning_enabled=False,
        )
    )
    result = service.reply("participant", "primary", "今天想聊聊散步。")

    assert result["text"] == "好的，我们继续聊。"
    assert captured["timeout"] == 25
    assert captured["payload"]["model"] == "z-ai/glm-5.2"
    assert captured["payload"]["provider"] == {
        "order": ["z-ai"],
        "allow_fallbacks": False,
    }
    assert captured["payload"]["reasoning"] == {"enabled": False}
