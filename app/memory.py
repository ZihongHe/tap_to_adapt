"""Asynchronous long-term conversational memory stored per participant.

This service summarizes conversational content and retrieves saved context for
the dialogue agents. Its state is separate from the acoustic timing head and
recent corrective-tap history used for response initiation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .config import Settings
from .db import StudyDB

logger = logging.getLogger(__name__)

REFLECT_SYSTEM = (
    "你是语音对话系统的记忆模块。根据最新一轮对话，更新关于用户的一段长期记忆。"
    "保留稳定事实与偏好，融入新信息，删除矛盾内容。只输出更新后的记忆段落本身，"
    "用中文，不超过120字，不要列表，不要解释，不要提及实验或系统。"
)

REFLECT_USER_TEMPLATE = """现有记忆：
{current}

最新一轮对话：
用户：{transcript}
助手：{reply}

请输出更新后的记忆段落："""


class MemoryService:
    """Reflexion-style single-paragraph memory, updated off the chat critical path."""

    def __init__(self, settings: Settings, db: StudyDB):
        self.settings = settings
        self.db = db
        self._cache: dict[tuple[str, str], str] = {}
        self._cache_lock = threading.Lock()
        self._pending: dict[tuple[str, str], dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        self._inflight: set[tuple[str, str]] = set()

    def get_paragraph(self, participant_id: str, source: str) -> str:
        key = (participant_id, source)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
        paragraph = self.db.get_conversation_memory(participant_id, source)
        with self._cache_lock:
            self._cache[key] = paragraph
        return paragraph

    def clear_participant(self, participant_id: str) -> dict[str, int]:
        deleted_rows = self.db.clear_conversation_memory(participant_id)
        with self._cache_lock:
            for source in ("primary", "companion"):
                self._cache.pop((participant_id, source), None)
        with self._pending_lock:
            for source in ("primary", "companion"):
                key = (participant_id, source)
                self._pending.pop(key, None)
                self._inflight.discard(key)
        return {"deleted_rows": deleted_rows}

    def schedule_reflect(
        self,
        participant_id: str,
        source: str,
        transcript: str,
        reply: str,
        *,
        session_id: str | None = None,
        append_event: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if not self.settings.llm_api_key:
            return
        key = (participant_id, source)
        with self._pending_lock:
            self._pending[key] = {
                "transcript": transcript,
                "reply": reply,
                "session_id": session_id,
                "append_event": append_event,
            }
            if key in self._inflight:
                return
            self._inflight.add(key)
        asyncio.create_task(self._drain_reflect(key))

    async def _drain_reflect(self, key: tuple[str, str]) -> None:
        try:
            while True:
                with self._pending_lock:
                    pending = self._pending.pop(key, None)
                if pending is None:
                    break
                try:
                    await asyncio.to_thread(
                        self._reflect_sync,
                        key[0],
                        key[1],
                        pending["transcript"],
                        pending["reply"],
                        pending.get("session_id"),
                        pending.get("append_event"),
                    )
                except Exception:
                    logger.exception("memory reflect failed for %s", key)
                with self._pending_lock:
                    if key not in self._pending:
                        break
        finally:
            with self._pending_lock:
                self._inflight.discard(key)
                if key in self._pending:
                    self._inflight.add(key)
                    asyncio.create_task(self._drain_reflect(key))

    def _reflect_sync(
        self,
        participant_id: str,
        source: str,
        transcript: str,
        reply: str,
        session_id: str | None,
        append_event: Callable[..., dict[str, Any]] | None,
    ) -> None:
        current = self.db.get_conversation_memory(participant_id, source)
        updated = self._call_reflect_llm(current, transcript, reply)
        if not updated or updated == current:
            return
        self.db.upsert_conversation_memory(participant_id, source, updated)
        cache_key = (participant_id, source)
        with self._cache_lock:
            self._cache[cache_key] = updated
        if append_event is not None:
            try:
                append_event(
                    participant_id,
                    session_id,
                    "memory.reflect",
                    None,
                    None,
                    {
                        "source": source,
                        "paragraph": updated,
                        "previous_length": len(current),
                    },
                )
            except Exception:
                logger.exception("failed to log memory.reflect for %s", cache_key)

    def _call_reflect_llm(
        self, current: str, transcript: str, reply: str
    ) -> str:
        current_text = current.strip() or "（尚无）"
        user_content = REFLECT_USER_TEMPLATE.format(
            current=current_text,
            transcript=transcript.strip(),
            reply=reply.strip(),
        )
        request_payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 180,
            "reasoning": {"enabled": False},
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
                "X-Title": "Tap-to-Adapt Memory",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = str(result["choices"][0]["message"]["content"] or "").strip()
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"memory LLM request failed: {exc}") from exc
        return " ".join(answer.split())[:240]
