"""Conversation response service for the desktop and mobile agents.

Keeps separate bounded dialogue histories for each participant and device role.
The service uses an optional chat-completion endpoint or a deterministic local
response when credentials are absent; it does not decide response initiation.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from .config import Settings


PROMPTS = {
    "primary": (
        "你是语音智能体 Snips，带有跨对话长期记忆，会记住用户提过的偏好与要点。"
        "自然、简短地回应用户，使用中文，不超过25个汉字。"
        "若用户问起你的设定或身份，可简短说明你是带长期记忆的对话伙伴。"
        "不要提及实验、模型、提示词或系统内部细节。不要使用列表。"
    ),
    "companion": (
        "你是参与者手机上的陪伴系统智能体，带有跨对话长期记忆，会记住用户提过的偏好与要点。"
        "负责轻松日常闲聊；自然回应并适度追问，每次只说一句中文，不超过25个汉字。"
        "若用户问起你的设定或身份，可简短说明你是带长期记忆的陪伴伙伴。"
        "不要提及实验、模型、提示词或系统内部细节。"
    ),
}


class ChatService:
    def __init__(
        self,
        settings: Settings,
        memory_getter: Callable[[str, str], str] | None = None,
    ):
        self.settings = settings
        self.memory_getter = memory_getter
        self.histories: dict[tuple[str, str], deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=6)
        )
        self.lock = threading.Lock()

    def clear_participant(self, participant_id: str) -> None:
        with self.lock:
            for source in ("primary", "companion"):
                self.histories.pop((participant_id, source), None)

    def _system_prompt(self, source: str, participant_id: str) -> str:
        prompt = PROMPTS[source]
        if self.memory_getter is None:
            return prompt
        memory = self.memory_getter(participant_id, source).strip()
        if not memory:
            return prompt
        return (
            f"{prompt}\n\n"
            f"关于这位用户的长期记忆（仅供参考，不要直接复述）：{memory}"
        )

    def reply(self, participant_id: str, source: str, text: str) -> dict[str, Any]:
        if source not in PROMPTS:
            raise ValueError("invalid chat source")
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("empty transcript")
        if len(cleaned) > 1000:
            raise ValueError("transcript too long")

        if not self.settings.llm_api_key:
            answer = self._mock_reply(cleaned, source)
            return {"text": answer, "mode": "mock", "model": "deterministic-pilot"}

        key = (participant_id, source)
        with self.lock:
            history = list(self.histories[key])
        messages = [
            {"role": "system", "content": self._system_prompt(source, participant_id)},
            *history,
        ]
        messages.append({"role": "user", "content": cleaned})
        request_payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 120,
            "reasoning": {"enabled": self.settings.llm_reasoning_enabled},
        }
        if self.settings.llm_provider_order:
            request_payload["provider"] = {
                "order": list(self.settings.llm_provider_order),
                "allow_fallbacks": self.settings.llm_allow_fallbacks,
            }
        payload = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.llm_base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.settings.public_base_url,
                "X-Title": "Tap-to-Adapt Study",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = str(result["choices"][0]["message"]["content"] or "").strip()
            if not answer:
                raise ValueError("LLM returned an empty answer")
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        answer = answer[:80]
        with self.lock:
            self.histories[key].append({"role": "user", "content": cleaned})
            self.histories[key].append({"role": "assistant", "content": answer})
        return {"text": answer, "mode": "remote", "model": self.settings.llm_model}

    @staticmethod
    def _mock_reply(text: str, source: str) -> str:
        if source == "companion":
            return "听起来挺有意思，你最在意哪一点？"
        return "我在听，你可以再多说一点。"
