"""Regression tests for single session flow.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
CONTENT = ROOT / "content"


def test_participant_pages_expose_one_complete_session_only() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    companion_page = (STATIC / "companion.html").read_text(encoding="utf-8")

    for stale in ("第 1 天", "第 2 天", "第 3 天", "自由交流</strong><span>累计约 25 分钟"):
        assert stale not in study_page
        assert stale not in companion_page

    assert 'id="day-selector"' not in study_page
    assert 'id="phone-day-selector"' not in companion_page
    assert "每周期前 4 分钟" in study_page
    assert "随后 2 分钟" in study_page
    assert "周期后休息 1 分钟" in study_page
    assert "话题本会一直可用" in study_page


def test_baseline_language_question_lists_english_explicitly() -> None:
    questionnaire = json.loads(
        (CONTENT / "questionnaires.zh-CN.json").read_text(encoding="utf-8")
    )
    fields = {
        field["id"]: field
        for field in questionnaire["instruments"]["baseline"]["fields"]
    }
    language = fields["primary_spoken_language"]
    options = {option["value"]: option["label"] for option in language["options"]}

    assert questionnaire["version"] == "questionnaires-2026-08-03-v5"
    assert options["english"] == "英语"
    assert options["other"] == "其他语言"


def test_baseline_age_question_uses_matching_five_year_intervals() -> None:
    zh = json.loads(
        (CONTENT / "questionnaires.zh-CN.json").read_text(encoding="utf-8")
    )
    en = json.loads(
        (CONTENT / "questionnaires.en.json").read_text(encoding="utf-8")
    )

    def age_options(catalog: dict) -> list[dict]:
        fields = catalog["instruments"]["baseline"]["fields"]
        return next(field for field in fields if field["id"] == "age_range")["options"]

    zh_options = age_options(zh)
    en_options = age_options(en)
    expected_values = [
        "under_18",
        "18_22",
        "23_27",
        "28_32",
        "33_37",
        "38_42",
        "43_47",
        "48_52",
        "53_57",
        "58_plus",
        "prefer_not",
    ]

    assert zh["version"] == en["version"] == "questionnaires-2026-08-03-v5"
    assert [option["value"] for option in zh_options] == expected_values
    assert [option["value"] for option in en_options] == expected_values
    assert [option["label"] for option in zh_options[1:-2]] == [
        "18–22 岁",
        "23–27 岁",
        "28–32 岁",
        "33–37 岁",
        "38–42 岁",
        "43–47 岁",
        "48–52 岁",
        "53–57 岁",
    ]


def test_locked_single_session_schedule_is_55_minutes() -> None:
    protocol = json.loads(
        (CONTENT / "study_protocol.json").read_text(encoding="utf-8")
    )
    design = protocol["session_design"]
    schedule = protocol["modes"]["strict"]["block_schedule"]

    assert design["total_sessions"] == 1
    assert design["participant_facing_day_selector"] is False
    assert schedule["active_blocks"] == 8
    assert schedule["primary_seconds"] == 240
    assert schedule["companion_seconds"] == 120
    assert schedule["rest_seconds"] == 60
    assert schedule["rest_after_final_block"] is False

    timed_seconds = (
        schedule["active_blocks"]
        * (schedule["primary_seconds"] + schedule["companion_seconds"])
        + (schedule["active_blocks"] - 1) * schedule["rest_seconds"]
    )
    assert timed_seconds == 3300
    assert timed_seconds == design["current_timed_flow_seconds"]
    assert timed_seconds <= design["maximum_timed_flow_seconds"]


def test_cold_start_requires_fresh_end_of_turn_and_reply_needed_context() -> None:
    protocol = json.loads(
        (CONTENT / "study_protocol.json").read_text(encoding="utf-8")
    )
    initialization = protocol["initialization"]
    factorization = initialization["decision_factorization"]

    assert initialization["activation_threshold"] >= 0.75
    assert factorization["minimum_end_of_turn_silence_ms"] >= 800
    assert factorization["explicit_turn_end_required"] is True
    assert factorization["context_required"] is True
    assert 0.5 < factorization["reply_needed_threshold"] < 1.0
    assert "one accepted context permits at most one" in factorization["context_lifecycle"]


def test_obsolete_multiday_snapshots_are_blocked_client_side() -> None:
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    companion_script = (STATIC / "companion.js").read_text(encoding="utf-8")

    assert "function protocolUsesSingleSession()" in study_script
    assert "该编号绑定的是已停用的旧版多日流程" in study_script
    assert "该编号绑定的是已停用的旧版多日流程" in companion_script
