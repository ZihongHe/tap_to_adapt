"""Generate a locked, private LLM turn pool for duration simulation.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


RUN_SLUG_PATTERN = re.compile(
    r"^simulation-duration-v1-[0-9]{13}-[a-f0-9]{8}$"
)
PERSONAS = (
    "喜欢旅行、城市散步和摄影，表达简洁",
    "喜欢做饭、咖啡和居家生活，说话较慢",
    "喜欢运动、健康和户外活动，语气直接",
    "喜欢电影、音乐和播客，表达有情绪",
    "喜欢学习、读书和工作效率，逻辑清晰",
    "喜欢宠物、植物和周末活动，语气轻松",
    "喜欢科技产品和游戏，偶尔使用口语",
    "喜欢家庭、朋友和本地生活，语气温和",
)


def _extract_json_array(text: str) -> list[str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        values: list[Any] = [json.loads(stripped)]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        values = []
        cursor = 0
        while cursor < len(stripped):
            start = stripped.find("[", cursor)
            if start < 0:
                break
            try:
                value, consumed = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            values.append(value)
            cursor = start + consumed
        if not values:
            raise ValueError("LLM response did not contain a JSON array") from None
    # Some OpenAI-compatible models append further one-item arrays despite the
    # JSON-only instruction.  Merge every complete top-level array.
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(value)
    if not flattened:
        raise ValueError("LLM response is not a JSON array")
    result: list[str] = []
    for item in flattened:
        normalized = " ".join(str(item).split())
        if 4 <= len(normalized) <= 80:
            result.append(normalized)
    return result


def _completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    public_base_url: str,
    prompt: str,
) -> list[str]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你为语音交互研究生成自然、可朗读的普通话用户话语。"
                        "只输出严格 JSON 字符串数组，不要解释。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 1800,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": public_base_url,
            "X-Title": "Tap-to-Adapt Duration Simulation",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = str(raw["choices"][0]["message"]["content"])
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ):
        # Provider errors can include signed URLs or request metadata.
        raise RuntimeError("LLM turn generation failed; provider details suppressed") from None
    return _extract_json_array(content)


def _generate_pool(
    *,
    base_url: str,
    api_key: str,
    model: str,
    public_base_url: str,
    persona: str,
    count: int,
    background: bool,
) -> list[str]:
    role = (
        "这些话来自房间中另一位说话者，不是在召唤语音助手"
        if background
        else "这些话来自目标用户，内容应当像自然地向语音助手或同伴继续聊天"
    )
    unique: list[str] = []
    for attempt in range(3):
        remaining = count - len(unique)
        if remaining <= 0:
            break
        exclusions = "；".join(unique[-32:])
        prompt = (
            f"生成 {remaining + 4} 句互不重复的普通话短句。人格：{persona}。{role}。"
            "每句 8–35 个汉字，涵盖陈述、追问、犹豫和话题转换；"
            "不要出现“实验、模型、标签、激活、打断、点击”等研究词汇；"
            f"这是第 {attempt + 1} 批。"
            + (f"不要重复这些已有句子：{exclusions}。" if exclusions else "")
            + "不要编号，只输出 JSON 数组。"
        )
        values = _completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            public_base_url=public_base_url,
            prompt=prompt,
        )
        unique = list(dict.fromkeys([*unique, *values]))
    if len(unique) < count:
        raise RuntimeError(
            f"LLM returned only {len(unique)} valid unique turns after retries; "
            f"expected {count}"
        )
    return unique[:count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a locked, private LLM turn pool for duration simulation."
    )
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-turns", type=int, default=24)
    parser.add_argument("--background-turns", type=int, default=16)
    parser.add_argument("--acknowledge-provider-cost", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_provider_cost:
        raise SystemExit("Pass --acknowledge-provider-cost")
    if not 8 <= args.positive_turns <= 64 or not 8 <= args.background_turns <= 64:
        raise SystemExit("Turn pool sizes must each be in [8, 64]")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("LLM_MODEL", "")
    public_base_url = os.environ.get(
        "PUBLIC_BASE_URL", "http://127.0.0.1:8765"
    )
    if not api_key or not model:
        raise SystemExit("LLM_API_KEY and LLM_MODEL must be configured on the server")

    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Roster is empty")
    run_slugs = {row["run_slug"] for row in rows}
    if len(run_slugs) != 1 or RUN_SLUG_PATTERN.fullmatch(next(iter(run_slugs))) is None:
        raise SystemExit("Roster is outside the isolated duration-simulation namespace")

    participants: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        persona = PERSONAS[index % len(PERSONAS)]
        target = _generate_pool(
            base_url=base_url,
            api_key=api_key,
            model=model,
            public_base_url=public_base_url,
            persona=persona,
            count=args.positive_turns,
            background=False,
        )
        background = _generate_pool(
            base_url=base_url,
            api_key=api_key,
            model=model,
            public_base_url=public_base_url,
            persona=PERSONAS[(index + 3) % len(PERSONAS)],
            count=args.background_turns,
            background=True,
        )
        participants.append(
            {
                "participant_id": row["participant_id"],
                "participant_index": int(row["participant_index"]),
                "persona": persona,
                "target_user_turns": target,
                "background_speaker_turns": background,
            }
        )

    output = {
        "schema": "tap-to-adapt-duration-llm-turns/v1",
        "simulation_only": True,
        "run_slug": next(iter(run_slugs)),
        "llm_provider": "OpenRouter",
        "llm_model": model,
        "generation_temperature": 0.8,
        "participants": participants,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "simulation_only": True,
                "run_slug": output["run_slug"],
                "llm_provider": output["llm_provider"],
                "llm_model": model,
                "participants": len(participants),
                "target_turns_each": args.positive_turns,
                "background_turns_each": args.background_turns,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
