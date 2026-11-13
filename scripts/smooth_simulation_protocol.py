"""Locked stimulus scheduling for the intent-aware 4+2 minute simulation.

Each six-minute cycle contains four minutes with the experimental agent and
two minutes with the companion agent.  A correction is determined by the
simulated user's conversational intent, not by a bare acoustic label:

* a response-requesting primary turn may activate only after speech ends;
* a no-response primary turn must not activate;
* no speech during companion dialogue may activate the experimental agent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


SCHEMA = "tap-to-adapt-intent-aware-4plus2-turns/v3"
RUN_SLUG_PATTERN = re.compile(r"^simulation-intent-v3-[0-9]{13}-[a-f0-9]{8}$")
DIFFICULTY_BANDS = ("low", "medium", "high")
SLOTS_PER_CYCLE = (
    ("primary", "primary_reply_expected", True, "participant"),
    ("primary", "primary_reply_expected", True, "participant"),
    ("primary", "primary_no_reply", False, "participant"),
    ("primary", "primary_reply_expected", True, "participant"),
    ("primary", "primary_reply_expected", True, "participant"),
    ("primary", "primary_no_reply", False, "participant"),
    ("primary", "primary_reply_expected", True, "participant"),
    ("primary", "primary_reply_expected", True, "participant"),
    ("companion", "participant_to_companion", False, "participant"),
    ("companion", "companion_reply", False, "companion"),
    ("companion", "participant_to_companion", False, "participant"),
    ("companion", "companion_reply", False, "companion"),
)


@dataclass(frozen=True)
class ParticipantProfile:
    profile_id: str
    description: str
    correction_delay_ms: int
    topic_offset: int


PRIMARY_PROFILE_LIBRARY = (
    ParticipantProfile("PR01", "城市散步与摄影；表达简洁", 260, 0),
    ParticipantProfile("PR02", "做饭与咖啡；语速偏慢", 420, 1),
    ParticipantProfile("PR03", "运动与户外；表达直接", 310, 2),
    ParticipantProfile("PR04", "电影与音乐；语气活跃", 360, 3),
    ParticipantProfile("PR05", "学习与阅读；逻辑清晰", 280, 4),
    ParticipantProfile("PR06", "宠物与植物；语气轻松", 470, 5),
    ParticipantProfile("PR07", "科技与游戏；口语化", 330, 6),
    ParticipantProfile("PR08", "家庭与本地生活；语气温和", 520, 7),
    ParticipantProfile("PR09", "旅行与交通；表达有条理", 300, 0),
    ParticipantProfile("PR10", "健康与作息；语速平稳", 440, 1),
    ParticipantProfile("PR11", "设计与手工；细节较多", 350, 2),
    ParticipantProfile("PR12", "新闻与播客；常用追问", 390, 3),
    ParticipantProfile("PR13", "办公与协作；表达精炼", 270, 4),
    ParticipantProfile("PR14", "购物与理财；语气谨慎", 480, 5),
    ParticipantProfile("PR15", "历史与文化；句式完整", 320, 6),
    ParticipantProfile("PR16", "社交与周末活动；自然随意", 410, 7),
    ParticipantProfile("PR17", "烹饪与地方美食；描述具体", 340, 8),
    ParticipantProfile("PR18", "校园与终身学习；喜欢追问", 460, 9),
    ParticipantProfile("PR19", "通勤与城市服务；表达务实", 300, 0),
    ParticipantProfile("PR20", "艺术与展览；语气舒缓", 430, 1),
    ParticipantProfile("PR21", "亲友与节日；叙述自然", 370, 2),
    ParticipantProfile("PR22", "数码产品与创作；节奏较快", 290, 3),
    ParticipantProfile("PR23", "园艺与居住环境；语气温和", 510, 4),
    ParticipantProfile("PR24", "职业发展与团队合作；有条理", 320, 5),
    ParticipantProfile("PR25", "公共议题与社区生活；措辞谨慎", 480, 6),
    ParticipantProfile("PR26", "运动赛事与训练；表达活跃", 280, 7),
    ParticipantProfile("PR27", "旅行见闻与语言；细节丰富", 390, 8),
    ParticipantProfile("PR28", "健康饮食与睡眠；语速平稳", 450, 9),
    ParticipantProfile("PR29", "电影叙事与表演；情绪明显", 310, 0),
    ParticipantProfile("PR30", "消费选择与预算；善于比较", 420, 1),
    ParticipantProfile("PR31", "历史故事与博物馆；句子较长", 350, 2),
    ParticipantProfile("PR32", "周末计划与社交活动；自然随意", 400, 3),
)

DEVELOPMENT_PROFILE_LIBRARY = (
    ParticipantProfile("DV01", "开发集：饮食与日常计划", 290, 1),
    ParticipantProfile("DV02", "开发集：学习与效率工具", 450, 3),
    ParticipantProfile("DV03", "开发集：运动与休闲", 340, 5),
    ParticipantProfile("DV04", "开发集：影视与旅行", 500, 7),
    ParticipantProfile("DV05", "开发集：家庭与节日安排", 310, 9),
    ParticipantProfile("DV06", "开发集：数码与创作工具", 470, 0),
    ParticipantProfile("DV07", "开发集：工作与团队沟通", 360, 2),
    ParticipantProfile("DV08", "开发集：文化与城市体验", 520, 4),
)


def _han_length(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))


def _split_tertiles(values: list[str]) -> dict[str, list[str]]:
    unique = list(dict.fromkeys(" ".join(str(item).split()) for item in values))
    if len(unique) < 9:
        raise ValueError("Each source pool needs at least nine unique utterances")
    ordered = sorted(unique, key=lambda text: (_han_length(text), text))
    result: dict[str, list[str]] = {}
    for index, band in enumerate(DIFFICULTY_BANDS):
        start = round(index * len(ordered) / 3)
        end = round((index + 1) * len(ordered) / 3)
        result[band] = ordered[start:end]
    if any(not rows for rows in result.values()):
        raise ValueError("Lexical difficulty stratification produced an empty band")
    return result


def load_llm_bank(
    path: Path, difficulty_map_path: Path | None = None
) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") == "tap-to-adapt-glm-intent-dialogues/v1":
        if difficulty_map_path is not None:
            raise ValueError(
                "Frozen GLM intent dialogues already contain locked difficulty bands"
            )
        return {"__intent_dialogues__": payload}  # type: ignore[return-value]
    pools = payload.get("text_pools") or payload
    complete = list(pools["complete"])
    incomplete = list(pools["incomplete"])
    background = list(pools["background"])
    lexical = {
        "primary_reply_expected": _split_tertiles(complete),
        "primary_no_reply": _split_tertiles(incomplete),
        "participant_to_companion": _split_tertiles(background),
        # Rotate the same LLM-generated daily-conversation bank independently.
        "companion_reply": _split_tertiles(list(reversed(background))),
    }
    if difficulty_map_path is None:
        return lexical
    payload = json.loads(difficulty_map_path.read_text(encoding="utf-8"))
    empirical = {
        (str(row["category"]), str(row["text"])): str(
            row["empirical_difficulty_band"]
        )
        for row in payload["rows"]
    }
    result: dict[str, dict[str, list[str]]] = {}
    for category, bands in lexical.items():
        all_texts = [text for band in DIFFICULTY_BANDS for text in bands[band]]
        reassigned = {band: [] for band in DIFFICULTY_BANDS}
        lexical_band = {
            text: band for band in DIFFICULTY_BANDS for text in bands[band]
        }
        for text in all_texts:
            band = empirical.get((category, text), lexical_band[text])
            reassigned[band].append(text)
        if any(not reassigned[band] for band in DIFFICULTY_BANDS):
            raise ValueError(
                f"Empirical difficulty map emptied a {category} stratum"
            )
        result[category] = reassigned
    return result


def profile_for(participant_index: int, cohort: str) -> ParticipantProfile:
    if cohort == "duration32":
        if not 1 <= participant_index <= 32:
            raise ValueError("duration32 requires participant_index in [1, 32]")
        return PRIMARY_PROFILE_LIBRARY[participant_index - 1]
    if cohort == "selection8":
        if not 1 <= participant_index <= 8:
            raise ValueError("selection8 requires participant_index in [1, 8]")
        return DEVELOPMENT_PROFILE_LIBRARY[participant_index - 1]
    raise ValueError(f"Unknown smooth-simulation cohort: {cohort}")


def build_schedule(
    *,
    participant_id: str,
    participant_index: int,
    cohort: str,
    max_minutes: int,
    bank: dict[str, dict[str, list[str]]],
) -> tuple[ParticipantProfile, list[dict[str, Any]]]:
    if max_minutes < 6 or max_minutes % 6:
        raise ValueError("max_minutes must be a positive multiple of six")
    profile = profile_for(participant_index, cohort)
    if "__intent_dialogues__" in bank:
        payload = bank["__intent_dialogues__"]  # type: ignore[assignment]
        by_profile = {
            str(row["profile_id"]): row for row in payload["participants"]
        }
        source = by_profile.get(profile.profile_id)
        if source is None:
            raise ValueError(
                f"Frozen GLM dialogue is absent for {profile.profile_id}"
            )
        source_cycles = list(source["cycles"])
        required_cycles = max_minutes // 6
        if len(source_cycles) < required_cycles:
            raise ValueError(
                f"{profile.profile_id} has fewer than {required_cycles} cycles"
            )
        events: list[dict[str, Any]] = []
        for cycle_index, cycle in enumerate(
            source_cycles[:required_cycles], start=1
        ):
            rows = [
                *[
                    {
                        **row,
                        "phase": "primary",
                        "speaker_role": "participant",
                        "category": (
                            "primary_reply_expected"
                            if bool(row["response_expected"])
                            else "primary_no_reply"
                        ),
                    }
                    for row in cycle["primary"]
                ],
                *[
                    {
                        **row,
                        "phase": "companion",
                        "response_expected": False,
                        "category": (
                            "participant_to_companion"
                            if row["speaker_role"] == "participant"
                            else "companion_reply"
                        ),
                    }
                    for row in cycle["companion"]
                ],
            ]
            for slot, row in enumerate(rows):
                response_expected = bool(row["response_expected"])
                deadline_seed = hashlib.sha256(
                    f"{participant_id}:{cycle_index}:{slot}:reply-deadline".encode(
                        "utf-8"
                    )
                ).digest()
                reply_deadline_seconds = 1.0 + (
                    int.from_bytes(deadline_seed[:2], "big") / 65535.0
                ) * 2.0
                text = str(row["text"])
                events.append(
                    {
                        "opportunity_index": len(events) + 1,
                        "cycle_index": cycle_index,
                        "elapsed_seconds": (cycle_index - 1) * 360
                        + 15
                        + slot * 30,
                        "phase": row["phase"],
                        "phase_minute_start": (cycle_index - 1) * 6
                        + (0 if row["phase"] == "primary" else 4),
                        "category": row["category"],
                        "speaker_role": row["speaker_role"],
                        "response_expected": response_expected,
                        "label": int(response_expected),
                        "interaction_intent": (
                            "request_experimental_agent_reply"
                            if response_expected
                            else (
                                "talk_to_companion"
                                if row["phase"] == "companion"
                                else "no_experimental_agent_reply"
                            )
                        ),
                        "reply_deadline_seconds": round(
                            reply_deadline_seconds, 3
                        ),
                        "difficulty_band": row["difficulty_band"],
                        "text_pool_index": slot,
                        "text": text,
                        "text_length": _han_length(text),
                        "topic_id": cycle["topic_id"],
                        "topic_prompt": cycle["topic_prompt"],
                        "topic_status": cycle["topic_status"],
                        "conversation_turn_index": slot + 1,
                    }
                )
        validate_schedule(events, max_minutes=max_minutes)
        return profile, events

    events: list[dict[str, Any]] = []
    for cycle in range(max_minutes // 6):
        for slot, (
            phase,
            category,
            response_expected,
            speaker_role,
        ) in enumerate(SLOTS_PER_CYCLE):
            # Twelve slots give exactly four low/medium/high events per cycle.
            # Keep the category-by-difficulty multiset identical across time.
            # Participants rotate through the same cohort-sized text set as a
            # Latin square, so every cycle contains the same aggregate texts
            # without forcing one person to repeat the same utterance.
            band = DIFFICULTY_BANDS[slot % 3]
            pool = bank[category][band]
            cohort_size = 32 if cohort == "duration32" else 8
            latin_pool = [
                pool[min(len(pool) - 1, int(index * len(pool) / cohort_size))]
                for index in range(cohort_size)
            ]
            chosen_latin = (
                participant_index - 1 + cycle + slot
            ) % cohort_size
            chosen_text = latin_pool[chosen_latin]
            chosen = pool.index(chosen_text)
            elapsed_seconds = cycle * 360 + 15 + slot * 30
            deadline_seed = hashlib.sha256(
                f"{participant_id}:{cycle}:{slot}:reply-deadline".encode("utf-8")
            ).digest()
            reply_deadline_seconds = 1.0 + (
                int.from_bytes(deadline_seed[:2], "big") / 65535.0
            ) * 2.0
            events.append(
                {
                    "opportunity_index": len(events) + 1,
                    "cycle_index": cycle + 1,
                    "elapsed_seconds": elapsed_seconds,
                    "phase": phase,
                    "phase_minute_start": cycle * 6 + (0 if phase == "primary" else 4),
                    "category": category,
                    "speaker_role": speaker_role,
                    "response_expected": response_expected,
                    "label": int(response_expected),
                    "interaction_intent": (
                        "request_experimental_agent_reply"
                        if response_expected
                        else (
                            "talk_to_companion"
                            if phase == "companion"
                            else "no_experimental_agent_reply"
                        )
                    ),
                    "reply_deadline_seconds": round(reply_deadline_seconds, 3),
                    "difficulty_band": band,
                    "text_pool_index": chosen,
                    "text": pool[chosen],
                    "text_length": _han_length(pool[chosen]),
                }
            )
    validate_schedule(events, max_minutes=max_minutes)
    return profile, events


def validate_schedule(events: list[dict[str, Any]], *, max_minutes: int) -> None:
    expected_cycles = max_minutes // 6
    if len(events) != expected_cycles * 12:
        raise ValueError("Smooth schedule does not contain twelve opportunities per cycle")
    for cycle in range(1, expected_cycles + 1):
        rows = [row for row in events if int(row["cycle_index"]) == cycle]
        if len(rows) != 12:
            raise ValueError(f"Cycle {cycle} does not contain twelve opportunities")
        if sum(row["phase"] == "primary" for row in rows) != 8:
            raise ValueError(f"Cycle {cycle} does not contain four primary minutes")
        if sum(row["phase"] == "companion" for row in rows) != 4:
            raise ValueError(f"Cycle {cycle} does not contain two companion minutes")
        if sum(int(row["response_expected"]) for row in rows) != 6:
            raise ValueError(f"Cycle {cycle} is not label-balanced")
        counts = {
            band: sum(row["difficulty_band"] == band for row in rows)
            for band in DIFFICULTY_BANDS
        }
        if set(counts.values()) != {4}:
            raise ValueError(f"Cycle {cycle} is not difficulty-balanced: {counts}")


def build_turn_payload(
    *,
    roster_rows: list[dict[str, str]],
    source_pools: Path,
    difficulty_map: Path | None,
    cohort: str,
    max_minutes: int,
) -> dict[str, Any]:
    if not roster_rows:
        raise ValueError("Roster is empty")
    run_slugs = {row["run_slug"] for row in roster_rows}
    if len(run_slugs) != 1 or RUN_SLUG_PATTERN.fullmatch(next(iter(run_slugs))) is None:
        raise ValueError("Roster is outside the smooth-simulation namespace")
    bank = load_llm_bank(source_pools, difficulty_map)
    frozen_intent_dialogues = "__intent_dialogues__" in bank
    participants = []
    for row in roster_rows:
        profile, events = build_schedule(
            participant_id=row["participant_id"],
            participant_index=int(row["participant_index"]),
            cohort=cohort,
            max_minutes=max_minutes,
            bank=bank,
        )
        participants.append(
            {
                "participant_id": row["participant_id"],
                "participant_index": int(row["participant_index"]),
                "profile": asdict(profile),
                "events": events,
            }
        )
    return {
        "schema": SCHEMA,
        "simulation_only": True,
        "run_slug": next(iter(run_slugs)),
        "cohort": cohort,
        "max_minutes": max_minutes,
        "cycle_minutes": 6,
        "primary_minutes_per_cycle": 4,
        "companion_minutes_per_cycle": 2,
        "opportunity_seconds": 30,
        "llm_provider": payload_provider(source_pools),
        "llm_model": payload_model(source_pools),
        "source_pool_sha256": hashlib.sha256(source_pools.read_bytes()).hexdigest(),
        "difficulty_map_sha256": (
            hashlib.sha256(difficulty_map.read_bytes()).hexdigest()
            if difficulty_map is not None
            else None
        ),
        "intent_policy": {
            "premature_activation": (
                "Any experimental-agent activation before participant TTS ends "
                "is interrupted, including response-requesting turns."
            ),
            "no_reply_activation": (
                "Any activation for a primary no-reply turn or anywhere in the "
                "two-minute companion phase is interrupted."
            ),
            "missed_reply": (
                "After a response-requesting primary utterance ends, manual "
                "activation occurs only if no response starts within the fixed "
                "per-opportunity 1–3 second deadline."
            ),
        },
        "difficulty_control": (
            (
                "Frozen GLM-5.2 dialogues contain exactly four low, four medium "
                "and four high lexical-length bands per six-minute cycle.  The "
                "reply-intent and speaker-role pattern is identical in every "
                "cycle; topics rotate from the source document without a late "
                "difficulty block."
            )
            if frozen_intent_dialogues
            else (
                "Every six-minute cycle contains four low, four medium and four "
                "high difficulty items; category/label/speaker-role counts are "
                "fixed. A cohort-level Latin-square rotation keeps the aggregate "
                "category-by-difficulty-by-text multiset identical across cycles. "
                + (
                    "Difficulty strata use the independent development run."
                    if difficulty_map is not None
                    else "Difficulty strata use lexical-complexity tertiles."
                )
            )
        ),
        "participants": participants,
    }


def payload_provider(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    generator = payload.get("generator") or {}
    return str(generator.get("provider") or "local-HuggingFace")


def payload_model(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    generator = payload.get("generator") or {}
    return str(generator.get("model_id") or "unknown")
