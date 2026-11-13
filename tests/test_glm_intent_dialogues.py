"""Regression tests for glm intent dialogues.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from scripts.generate_glm_intent_dialogues import _clean_json


def test_clean_json_repairs_only_non_structural_inner_quotes() -> None:
    payload = '{"text":"我觉得"好吧"挺自然","ok":true}'
    assert _clean_json(payload) == {
        "text": '我觉得"好吧"挺自然',
        "ok": True,
    }


def test_clean_json_keeps_valid_json_unchanged() -> None:
    payload = '{"text":"自然口语","items":[1,2],"ok":true}'
    assert _clean_json(payload) == {
        "text": "自然口语",
        "items": [1, 2],
        "ok": True,
    }
