"""Generate frozen, intent-labelled simulation dialogues with GLM-5.2.

Credentials are read only from ``OPENROUTER_API_KEY``.  The output contains no
credential and is suitable for freezing before model comparison.  Each
six-minute cycle has eight coherent primary turns followed by four companion
turns.  The primary reply-intent pattern and lexical-length bands are fixed so
that time is not confounded with a late block of harder utterances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .smooth_simulation_protocol import (
        DEVELOPMENT_PROFILE_LIBRARY,
        DIFFICULTY_BANDS,
        PRIMARY_PROFILE_LIBRARY,
    )
except ImportError:
    from smooth_simulation_protocol import (
        DEVELOPMENT_PROFILE_LIBRARY,
        DIFFICULTY_BANDS,
        PRIMARY_PROFILE_LIBRARY,
    )


MODEL = "z-ai/glm-5.2"
PROVIDER = {"order": ["z-ai"], "allow_fallbacks": False}
EXPECTED_PATTERN = (True, True, False, True, True, False, True, True)
PRIMARY_BANDS = (
    "medium",
    "high",
    "low",
    "low",
    "high",
    "low",
    "medium",
    "high",
)
COMPANION_BANDS = ("low", "medium", "high", "medium")
BAND_LIMITS = {
    "low": (1, 18),
    "medium": (4, 32),
    "high": (10, 55),
}


def _clean_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # GLM occasionally emits an ASCII quote inside otherwise valid JSON
        # string content without escaping it. Preserve utterance semantics
        # while escaping only quotes that cannot close a JSON string.
        repaired: list[str] = []
        in_string = False
        escaped = False
        for index, character in enumerate(cleaned):
            if not in_string:
                repaired.append(character)
                if character == '"':
                    in_string = True
                continue
            if escaped:
                repaired.append(character)
                escaped = False
                continue
            if character == "\\":
                repaired.append(character)
                escaped = True
                continue
            if character != '"':
                repaired.append(character)
                continue
            next_index = index + 1
            while (
                next_index < len(cleaned)
                and cleaned[next_index] in {" ", "\t", "\r", "\n"}
            ):
                next_index += 1
            next_character = (
                cleaned[next_index] if next_index < len(cleaned) else ""
            )
            if next_character in {",", ":", "}", "]", ""}:
                repaired.append(character)
                in_string = False
            else:
                repaired.append('\\"')
        return json.loads("".join(repaired))


def _han_length(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))


def _topic_plan(
    topics: list[dict[str, Any]], *, profile_index: int, cycles: int
) -> list[dict[str, str]]:
    flat = [
        {
            "topic_id": str(prompt["id"]),
            "category_id": str(category["id"]),
            "prompt": str(prompt["text"]),
        }
        for category in topics
        for prompt in category["prompts"]
    ]
    rng = random.Random(20260727 + 997 * profile_index)
    rng.shuffle(flat)
    return [flat[index % len(flat)] for index in range(cycles)]


def _prompt(
    *,
    profile_id: str,
    description: str,
    topic_plan: list[dict[str, str]],
) -> list[dict[str, str]]:
    plan = "\n".join(
        f"{index + 1}. [{row['topic_id']}] {row['prompt']}"
        for index, row in enumerate(topic_plan)
    )
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的中文语音交互实验刺激设计者。只输出合法 JSON，"
                "不输出 Markdown。话语必须自然口语化，不提实验、模型或提示词。"
                "JSON 字符串内容不得使用 ASCII 双引号；需要引用时使用中文引号。"
            ),
        },
        {
            "role": "user",
            "content": f"""
为模拟参与者 {profile_id}（{description}）生成 {len(topic_plan)} 个六分钟周期。
每个周期先与实验智能体连续多轮对话四分钟，再与陪伴智能体对话两分钟。
主题按下列顺序使用；四分钟内保持同一主题连续，陪伴段构成自然短对话。

{plan}

每周期必须输出：
1. primary：恰好 8 条参与者话语，response_expected 固定为
   [true,true,false,true,true,false,true,true]。
2. response_expected=false 的话语必须明确不期待回应，例如自然收束、
   自我确认或“好吧/知道了”一类，不得含问题或继续请求。
3. primary 的 difficulty_band 固定为
   ["medium","high","low","low","high","low","medium","high"]。
4. companion：恰好 4 条，speaker_role 固定交替为
   ["participant","companion","participant","companion"]，
   difficulty_band 固定为 ["low","medium","high","medium"]。
5. low 约 2–10 个汉字/字母数字，medium 约 8–20，high 约 16–34；
   不要让后面周期系统性变难。不同周期措辞不得重复。
6. topic_status 只能是 "ended_at_break" 或 "ended_naturally"。

严格使用：
{{
  "profile_id": "{profile_id}",
  "cycles": [
    {{
      "cycle_index": 1,
      "topic_id": "...",
      "topic_prompt": "...",
      "topic_status": "ended_at_break",
      "primary": [
        {{"text":"...","response_expected":true,"difficulty_band":"medium"}}
      ],
      "companion": [
        {{"speaker_role":"participant","text":"...","difficulty_band":"low"}}
      ]
    }}
  ]
}}
""".strip(),
        },
    ]


def _request(
    *, api_key: str, messages: list[dict[str, str]], seed: int
) -> tuple[dict[str, Any], str]:
    payload = json.dumps(
        {
            "model": MODEL,
            "provider": PROVIDER,
            "reasoning": {"enabled": False},
            "messages": messages,
            "temperature": 0.65,
            "seed": seed,
            "max_tokens": 7000,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8765",
            "X-Title": "Tap-to-Adapt simulation fixture",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    choice = result["choices"][0]["message"]
    content = str(choice.get("content") or "")
    if not content:
        raise ValueError("GLM-5.2 returned empty content")
    provider = str(result.get("provider") or "")
    return _clean_json(content), provider


def _validate(
    payload: dict[str, Any],
    *,
    profile_id: str,
    topic_plan: list[dict[str, str]],
) -> dict[str, Any]:
    if payload.get("profile_id") != profile_id:
        raise ValueError("profile_id mismatch")
    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or len(cycles) != len(topic_plan):
        raise ValueError("cycle count mismatch")
    seen: set[str] = set()
    cycle_mean_lengths: list[float] = []
    for index, (cycle, topic) in enumerate(zip(cycles, topic_plan, strict=True), 1):
        if int(cycle.get("cycle_index", -1)) != index:
            raise ValueError(f"cycle index mismatch at {index}")
        if cycle.get("topic_id") != topic["topic_id"]:
            raise ValueError(f"topic mismatch at cycle {index}")
        cycle["topic_prompt"] = topic["prompt"]
        primary = cycle.get("primary")
        companion = cycle.get("companion")
        if not isinstance(primary, list) or len(primary) != 8:
            raise ValueError(f"primary turn count mismatch at cycle {index}")
        if not isinstance(companion, list) or len(companion) != 4:
            raise ValueError(f"companion turn count mismatch at cycle {index}")
        if tuple(bool(row.get("response_expected")) for row in primary) != EXPECTED_PATTERN:
            raise ValueError(f"reply-intent pattern mismatch at cycle {index}")
        if tuple(str(row.get("speaker_role")) for row in companion) != (
            "participant",
            "companion",
            "participant",
            "companion",
        ):
            raise ValueError(f"companion speaker pattern mismatch at cycle {index}")
        cycle_lengths: list[int] = []
        all_rows = [*primary, *companion]
        for row in all_rows:
            text = " ".join(str(row.get("text") or "").split())
            if not text or text in seen:
                raise ValueError(f"empty or repeated utterance at cycle {index}")
            seen.add(text)
            row["text"] = text
            length = _han_length(text)
            cycle_lengths.append(length)
        # Treat GLM's requested bands as generation guidance, then derive the
        # frozen bands from observed length ranks.  This guarantees four
        # low/medium/high items per cycle without mislabelling a long sentence
        # as "low" merely because the generator missed a length instruction.
        ranked = sorted(
            range(len(all_rows)),
            key=lambda row_index: (cycle_lengths[row_index], row_index),
        )
        for rank, row_index in enumerate(ranked):
            all_rows[row_index]["difficulty_band"] = DIFFICULTY_BANDS[
                min(2, rank // 4)
            ]
        cycle_mean_lengths.append(sum(cycle_lengths) / len(cycle_lengths))
    if len(cycle_mean_lengths) >= 6:
        third = max(1, len(cycle_mean_lengths) // 3)
        early = sum(cycle_mean_lengths[:third]) / third
        late = sum(cycle_mean_lengths[-third:]) / third
        if late - early > 3.0:
            raise ValueError(
                "dialogue length drifts materially upward in later cycles"
            )
        if max(cycle_mean_lengths) - min(cycle_mean_lengths) > 10.0:
            raise ValueError("cycle-level dialogue length is not smooth")
    return payload


def _generate_profile(
    *,
    api_key: str,
    profile: Any,
    profile_index: int,
    topics: list[dict[str, Any]],
    cycles: int,
    cycles_per_request: int,
    seed_offset: int,
) -> dict[str, Any]:
    plan = _topic_plan(topics, profile_index=profile_index, cycles=cycles)
    merged_cycles: list[dict[str, Any]] = []
    providers: set[str] = set()
    for chunk_start in range(0, cycles, cycles_per_request):
        chunk = plan[chunk_start : chunk_start + cycles_per_request]
        last_error: BaseException | None = None
        for attempt in range(1, 7):
            try:
                payload, provider = _request(
                    api_key=api_key,
                    messages=_prompt(
                        profile_id=profile.profile_id,
                        description=profile.description,
                        topic_plan=chunk,
                    ),
                    seed=20260727
                    + 1009 * profile_index
                    + 37 * chunk_start
                    + attempt
                    + seed_offset,
                )
                validated = _validate(
                    payload,
                    profile_id=profile.profile_id,
                    topic_plan=chunk,
                )
                for local_index, cycle in enumerate(
                    validated["cycles"], start=1
                ):
                    cycle["cycle_index"] = chunk_start + local_index
                merged_cycles.extend(validated["cycles"])
                providers.add(provider)
                break
            except (
                urllib.error.URLError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt < 6:
                    time.sleep(attempt * 2)
        else:
            raise RuntimeError(
                "GLM fixture generation failed for "
                f"{profile.profile_id} cycles {chunk_start + 1}-"
                f"{chunk_start + len(chunk)}: {last_error}"
            )
    cycle_means = [
        sum(
            _han_length(str(row["text"]))
            for row in [*cycle["primary"], *cycle["companion"]]
        )
        / 12.0
        for cycle in merged_cycles
    ]
    rng = random.Random(20260727 + 7919 * profile_index)

    def order_metrics(order: list[int]) -> tuple[float, float, float]:
        ordered = [cycle_means[index] for index in order]
        third = max(1, len(ordered) // 3)
        early = sum(ordered[:third]) / third
        late = sum(ordered[-third:]) / third
        maximum_adjacent = max(
            (
                abs(right - left)
                for left, right in zip(ordered, ordered[1:])
            ),
            default=0.0,
        )
        return maximum_adjacent, abs(late - early), max(ordered) - min(ordered)

    best_order = list(range(len(merged_cycles)))
    best_metrics = order_metrics(best_order)
    # Topic order is not an outcome.  Search deterministic permutations and
    # freeze the one that best prevents a late difficulty block or an abrupt
    # adjacent-cycle change.
    for _ in range(20_000):
        candidate = list(range(len(merged_cycles)))
        rng.shuffle(candidate)
        metrics = order_metrics(candidate)
        score = (max(metrics[0] - 8.0, 0.0), metrics[1], metrics[0])
        best_score = (
            max(best_metrics[0] - 8.0, 0.0),
            best_metrics[1],
            best_metrics[0],
        )
        if score < best_score:
            best_order, best_metrics = candidate, metrics
    merged_cycles = [merged_cycles[index] for index in best_order]
    for cycle_index, cycle in enumerate(merged_cycles, start=1):
        cycle["cycle_index"] = cycle_index
    if best_metrics[0] > 12.0 or best_metrics[1] > 4.0:
        raise RuntimeError(
            f"{profile.profile_id} failed lexical smoothness after ordering: "
            f"max_adjacent={best_metrics[0]:.2f}, early_late={best_metrics[1]:.2f}"
        )
    return {
        "profile_id": profile.profile_id,
        "cycles": merged_cycles,
        "provider": ",".join(sorted(providers)),
        "generation_seed_offset": seed_offset,
        "lexical_smoothness": {
            "maximum_adjacent_cycle_mean_length_change": round(
                best_metrics[0], 3
            ),
            "absolute_early_late_mean_length_difference": round(
                best_metrics[1], 3
            ),
            "cycle_mean_length_range": round(best_metrics[2], 3),
            "ordering_seed": 20260727 + 7919 * profile_index,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-minutes", type=int, default=72)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cycles-per-request", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--profile-limit",
        type=int,
        help="Engineering smoke only; omit when freezing the complete fixture.",
    )
    args = parser.parse_args()
    if args.max_minutes < 6 or args.max_minutes % 6:
        raise SystemExit("--max-minutes must be a positive multiple of six")
    if not 1 <= args.cycles_per_request <= 6:
        raise SystemExit("--cycles-per-request must be in [1, 6]")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    topic_payload = json.loads(args.topics.read_text(encoding="utf-8"))
    topics = list(topic_payload["categories"])
    profiles = [
        *DEVELOPMENT_PROFILE_LIBRARY,
        *PRIMARY_PROFILE_LIBRARY,
    ]
    if args.profile_limit is not None:
        if not 1 <= args.profile_limit <= len(profiles):
            raise SystemExit("--profile-limit is outside the available profiles")
        profiles = profiles[: args.profile_limit]
    generated: list[dict[str, Any]] = []
    shard_dir = args.output.with_name(f"{args.output.name}.profiles")
    shard_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[int, Any]] = []
    for index, profile in enumerate(profiles, 1):
        shard = shard_dir / f"{profile.profile_id}.json"
        if shard.is_file():
            cached = json.loads(shard.read_text(encoding="utf-8"))
            if (
                cached.get("profile_id") != profile.profile_id
                or len(cached.get("cycles") or []) != args.max_minutes // 6
            ):
                raise SystemExit(f"Invalid cached profile shard: {shard}")
            generated.append(cached)
            print(f"{profile.profile_id}: cached", flush=True)
        else:
            pending.append((index, profile))
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        futures = {
            pool.submit(
                _generate_profile,
                api_key=api_key,
                profile=profile,
                profile_index=index,
                topics=topics,
                cycles=args.max_minutes // 6,
                cycles_per_request=args.cycles_per_request,
                seed_offset=args.seed_offset,
            ): profile.profile_id
            for index, profile in pending
        }
        for future in as_completed(futures):
            profile_id = futures[future]
            try:
                result = future.result()
                shard = shard_dir / f"{profile_id}.json"
                temporary = shard.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, shard)
                generated.append(result)
                print(f"{profile_id}: generated", flush=True)
            except BaseException as exc:
                failures.append(
                    f"{profile_id}: {type(exc).__name__}: {exc}"
                )
                print(f"{profile_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
    if failures:
        raise SystemExit(
            "Profile generation incomplete; rerun to resume missing shards:\n"
            + "\n".join(failures)
        )
    generated.sort(key=lambda row: str(row["profile_id"]))
    result = {
        "schema": "tap-to-adapt-glm-intent-dialogues/v1",
        "generator": {
            "provider": "OpenRouter",
            "model_id": MODEL,
            "provider_routing": PROVIDER,
            "reasoning_enabled": False,
            "generation_seed_offset_for_uncached_profiles": args.seed_offset,
            "generated_at_unix_ms": int(time.time() * 1000),
        },
        "source_topics": {
            "path": str(args.topics),
            "sha256": hashlib.sha256(args.topics.read_bytes()).hexdigest(),
            "version": topic_payload.get("version"),
        },
        "max_minutes": args.max_minutes,
        "participants": generated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profiles": len(generated),
                "model": MODEL,
                "provider": sorted(
                    {str(row.get("provider") or "") for row in generated}
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
