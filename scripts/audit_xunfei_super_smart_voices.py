#!/usr/bin/env python3
"""Audit Xunfei Super Smart TTS voice authorization without playing audio.

Credentials are read only from environment variables:
  XUNFEI_APP_ID, XUNFEI_API_KEY, XUNFEI_API_SECRET

The probe keeps synthesized bytes in memory and prints one credential-free JSON
record per voice. It never writes or plays audio.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import ssl
import sys
import time
from dataclasses import asdict, dataclass
from email.utils import formatdate
from typing import Any
from urllib.parse import urlencode, urlparse

import websocket


ENDPOINT = "wss://cbm01.cn-huabei-1.xf-yun.com/v1/private/mcd9m97e6"


@dataclass(frozen=True)
class ProbeResult:
    voice: str
    ok: bool
    audio_bytes: int
    elapsed_ms: int
    code: int | None = None
    error: str | None = None


def _auth_url(endpoint: str, api_key: str, api_secret: str) -> str:
    parsed = urlparse(endpoint)
    date = formatdate(timeval=None, localtime=False, usegmt=True)
    signature_origin = (
        f"host: {parsed.netloc}\n"
        f"date: {date}\n"
        f"GET {parsed.path} HTTP/1.1"
    )
    digest = hmac.new(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature = base64.b64encode(digest).decode("ascii")
    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(
        authorization_origin.encode("utf-8")
    ).decode("ascii")
    return endpoint + "?" + urlencode(
        {
            "host": parsed.netloc,
            "date": date,
            "authorization": authorization,
        }
    )


def _safe_error(message: Any) -> str:
    text = str(message or "unknown error").replace("\r", " ").replace("\n", " ")
    return text[:8192]


def synthesize_in_memory(
    *,
    app_id: str,
    api_key: str,
    api_secret: str,
    voice: str,
    text: str,
    timeout_s: float,
) -> ProbeResult:
    websocket.setdefaulttimeout(timeout_s)
    started = time.perf_counter()
    audio_parts: list[bytes] = []
    result_code: int | None = None
    result_error: str | None = None
    finished = False

    def elapsed_ms() -> int:
        return round((time.perf_counter() - started) * 1000)

    def on_open(ws: websocket.WebSocketApp) -> None:
        payload = {
            "header": {"app_id": app_id, "status": 2},
            "parameter": {
                "tts": {
                    "vcn": voice,
                    "volume": 50,
                    "rhy": 0,
                    "speed": 50,
                    "pitch": 50,
                    "bgs": 0,
                    "reg": 0,
                    "rdn": 0,
                    "audio": {
                        "encoding": "lame",
                        "sample_rate": 16000,
                        "channels": 1,
                        "bit_depth": 16,
                        "frame_size": 0,
                    },
                }
            },
            "payload": {
                "text": {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "plain",
                    "status": 2,
                    "seq": 0,
                    "text": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                }
            },
        }
        ws.send(json.dumps(payload, ensure_ascii=False))

    def on_message(ws: websocket.WebSocketApp, raw: str) -> None:
        nonlocal result_code, result_error, finished
        try:
            message = json.loads(raw)
            header = message.get("header") or {}
            result_code = int(header.get("code", 0))
            if result_code != 0:
                result_error = _safe_error(
                    header.get("message")
                    or message.get("message")
                    or f"service code {result_code}"
                )
                finished = True
                ws.close()
                return
            audio = ((message.get("payload") or {}).get("audio") or {})
            encoded = audio.get("audio")
            if encoded:
                audio_parts.append(base64.b64decode(encoded))
            if int(audio.get("status", 0)) == 2:
                finished = True
                ws.close()
        except Exception as exc:  # pragma: no cover - defensive network path
            result_error = _safe_error(exc)
            finished = True
            ws.close()

    def on_error(_ws: websocket.WebSocketApp, error: Any) -> None:
        nonlocal result_error, finished
        result_error = _safe_error(error)
        finished = True

    ws = websocket.WebSocketApp(
        _auth_url(ENDPOINT, api_key, api_secret),
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )
    ws.run_forever(
        sslopt={"cert_reqs": ssl.CERT_REQUIRED},
    )
    if not finished and result_error is None:
        result_error = f"timeout after {timeout_s:g}s"
    audio_bytes = sum(len(part) for part in audio_parts)
    ok = result_code == 0 and audio_bytes > 0 and result_error is None
    return ProbeResult(
        voice=voice,
        ok=ok,
        audio_bytes=audio_bytes,
        elapsed_ms=elapsed_ms(),
        code=result_code,
        error=None if ok else result_error or "no audio returned",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices", nargs="+", required=True)
    parser.add_argument("--text", default="你好，这是静默音色授权测试。")
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app_id = os.environ.get("XUNFEI_APP_ID", "")
    api_key = os.environ.get("XUNFEI_API_KEY", "")
    api_secret = os.environ.get("XUNFEI_API_SECRET", "")
    if not all((app_id, api_key, api_secret)):
        print(
            json.dumps(
                {"ok": False, "error": "missing Xunfei credential environment"},
                ensure_ascii=False,
            )
        )
        return 2

    failures = 0
    for voice in args.voices:
        result = synthesize_in_memory(
            app_id=app_id,
            api_key=api_key,
            api_secret=api_secret,
            voice=voice,
            text=args.text,
            timeout_s=args.timeout,
        )
        print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
        failures += int(not result.ok)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
