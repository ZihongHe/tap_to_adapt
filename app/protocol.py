"""Versioned protocol loading and prerequisite validation.

Protocol selects consent, questionnaire, and session definitions and validates
completion prerequisites. Open-text checks assess response length without
using response content as supervision for acoustic timing adaptation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import Settings

_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
OPEN_TEXT_MIN_CHARS = 20
OPEN_TEXT_MIN_WORDS = 12


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def open_text_meets_minimum(
    text: str,
    *,
    min_chars: int = OPEN_TEXT_MIN_CHARS,
    min_words: int = OPEN_TEXT_MIN_WORDS,
) -> bool:
    """Accept either enough CJK characters or enough English words."""
    value = str(text or "").strip()
    if not value:
        return False
    if len(_CJK_CHAR_RE.findall(value)) >= min_chars:
        return True
    return len(_ENGLISH_WORD_RE.findall(value)) >= min_words


class Protocol:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.consents = {
            "zh-CN": _load_json(settings.content_dir / "consent.zh-CN.json"),
            "en": _load_json(settings.content_dir / "consent.en.json"),
        }
        self.topic_catalogs = {
            "zh-CN": _load_json(settings.content_dir / "topics.zh-CN.json"),
            "en": _load_json(settings.content_dir / "topics.en.json"),
        }
        self.questionnaire_catalogs = {
            "zh-CN": _load_json(settings.content_dir / "questionnaires.zh-CN.json"),
            "en": _load_json(settings.content_dir / "questionnaires.en.json"),
        }
        self.compensation_notices = {
            "zh-CN": _load_json(settings.content_dir / "compensation.zh-CN.json"),
            "en": _load_json(settings.content_dir / "compensation.en.json"),
        }
        self.consent = self.consents["zh-CN"]
        self.topics = self.topic_catalogs["zh-CN"]
        self.questionnaires = self.questionnaire_catalogs["zh-CN"]
        self.study = _load_json(settings.content_dir / "study_protocol.json")
        self.background = _load_json(Path(settings.background_manifest))
        self.consent_hashes = {
            locale: hashlib.sha256(
                json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for locale, document in self.consents.items()
        }
        self.consent_hash = self.consent_hashes["zh-CN"]
        self.compensation_hashes = {
            locale: hashlib.sha256(
                json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for locale, document in self.compensation_notices.items()
        }
        self._validate_localized_materials()

    @staticmethod
    def normalize_locale(value: str | None) -> str:
        normalized = str(value or "").strip().lower().replace("_", "-")
        return "en" if normalized == "en" or normalized.startswith("en-") else "zh-CN"

    def _validate_localized_materials(self) -> None:
        consent_version = str(self.consent["version"])
        questionnaire_version = str(self.questionnaires["version"])
        topics_version = str(self.topics["version"])
        consent_choice_ids = [str(item["id"]) for item in self.consent["choices"]]
        questionnaire_ids = set(self.questionnaires["instruments"])
        topic_ids = [
            str(prompt["id"])
            for category in self.topics["categories"]
            for prompt in category["prompts"]
        ]
        for locale, document in self.consents.items():
            if str(document["version"]) != consent_version:
                raise ValueError(f"Consent version mismatch for locale {locale}")
            if [str(item["id"]) for item in document["choices"]] != consent_choice_ids:
                raise ValueError(f"Consent choice mismatch for locale {locale}")
        for locale, catalog in self.questionnaire_catalogs.items():
            if str(catalog["version"]) != questionnaire_version:
                raise ValueError(f"Questionnaire version mismatch for locale {locale}")
            if set(catalog["instruments"]) != questionnaire_ids:
                raise ValueError(f"Questionnaire instrument mismatch for locale {locale}")
        for locale, catalog in self.topic_catalogs.items():
            localized_ids = [
                str(prompt["id"])
                for category in catalog["categories"]
                for prompt in category["prompts"]
            ]
            if str(catalog["version"]) != topics_version or localized_ids != topic_ids:
                raise ValueError(f"Topic catalog mismatch for locale {locale}")
        compensation_version = str(self.compensation_notices["zh-CN"]["version"])
        for locale, document in self.compensation_notices.items():
            if str(document["version"]) != compensation_version:
                raise ValueError(
                    f"Compensation notice version mismatch for locale {locale}"
                )

    @property
    def consent_version(self) -> str:
        return str(self.consent["version"])

    def public_consent(self, locale: str | None = None) -> dict[str, Any]:
        normalized = self.normalize_locale(locale)
        return {
            **self.consents[normalized],
            "sha256": self.consent_hashes[normalized],
            "locale": normalized,
        }

    def consent_document(self, version: str, document_hash: str) -> dict[str, Any] | None:
        if str(version) != self.consent_version:
            return None
        for locale, expected_hash in self.consent_hashes.items():
            if str(document_hash) == expected_hash:
                return self.consents[locale]
        return None

    @property
    def compensation_version(self) -> str:
        return str(self.compensation_notices["zh-CN"]["version"])

    def public_compensation_notice(
        self, locale: str | None = None
    ) -> dict[str, Any]:
        normalized = self.normalize_locale(locale)
        return {
            **self.compensation_notices[normalized],
            "sha256": self.compensation_hashes[normalized],
            "locale": normalized,
        }

    def valid_compensation_notice(
        self, version: str, document_hash: str
    ) -> bool:
        return self.compensation_notice_document(version, document_hash) is not None

    def compensation_notice_document(
        self, version: str, document_hash: str
    ) -> dict[str, Any] | None:
        if str(version) != self.compensation_version:
            return None
        for locale, expected_hash in self.compensation_hashes.items():
            if str(document_hash) == expected_hash:
                return self.compensation_notices[locale]
        return None

    def valid_consent_signature(self, version: str, document_hash: str) -> bool:
        return self.consent_document(version, document_hash) is not None

    def topics_for(self, locale: str | None = None) -> dict[str, Any]:
        return self.topic_catalogs[self.normalize_locale(locale)]

    def questionnaires_for(self, locale: str | None = None) -> dict[str, Any]:
        return self.questionnaire_catalogs[self.normalize_locale(locale)]

    @property
    def study_version(self) -> str:
        return str(self.study["version"])

    def study_for(self, participant: dict[str, Any]) -> dict[str, Any]:
        protocol_text = str(participant.get("run_protocol_json") or "")
        expected_sha256 = str(participant.get("run_protocol_sha256") or "")
        if not protocol_text or not expected_sha256:
            raise ValueError("Participant run is missing its protocol snapshot")
        if hashlib.sha256(protocol_text.encode("utf-8")).hexdigest() != expected_sha256:
            raise ValueError("Participant run protocol snapshot integrity check failed")
        try:
            study = json.loads(protocol_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Participant run protocol snapshot is invalid JSON") from exc
        if str(participant.get("protocol_version") or "") != str(
            study.get("version") or ""
        ):
            raise ValueError("Participant protocol version does not match its snapshot")
        return study

    def rules_for(self, participant: dict[str, Any]) -> dict[str, Any]:
        study = self.study_for(participant)
        mode = str(participant.get("protocol_mode", ""))
        rules = study["modes"].get(mode)
        if rules is None:
            raise ValueError("Participant protocol mode is invalid")
        return rules

    def validate_response(
        self,
        instrument: str,
        version: str,
        answers: dict[str, Any],
        session_day: int,
        final_day: int = 3,
    ) -> dict[str, Any]:
        if version != self.questionnaires["version"]:
            raise ValueError("问卷版本已更新，请刷新页面后重新填写。")
        if instrument == "baseline":
            if session_day != 1:
                raise ValueError("开始前信息只能在第 1 天提交。")
            schema_name = "baseline"
        elif instrument.startswith("daily_day_"):
            try:
                declared_day = int(instrument.rsplit("_", 1)[1])
            except ValueError as exc:
                raise ValueError("每日问卷名称无效。") from exc
            if declared_day != session_day:
                raise ValueError("每日问卷与当前实验日不一致。")
            schema_name = "daily"
        elif instrument == "final":
            if session_day != final_day:
                raise ValueError("总体问卷只能在完整实验的最后一次会话提交。")
            schema_name = "final"
        else:
            raise ValueError("问卷名称不在当前协议中。")

        schema = self.questionnaires["instruments"][schema_name]
        scale = self.questionnaires["scale"]
        item_ids = {str(item["id"]) for item in schema.get("items", [])}
        open_items = {str(item["id"]): item for item in schema.get("open_items", [])}
        followups = {str(item["id"]): item for item in schema.get("followups", [])}
        metadata_keys = {"ended_early"} if schema_name == "daily" else set()
        fields = {str(item["id"]): item for item in schema.get("fields", [])}
        allowed = item_ids | set(open_items) | set(followups) | set(fields) | metadata_keys
        unknown = sorted(set(answers) - allowed)
        if unknown:
            raise ValueError(f"问卷包含未定义字段：{', '.join(unknown)}")

        normalized: dict[str, Any] = {}
        missing_items: list[str] = []
        for item_id in sorted(item_ids):
            value = answers.get(item_id)
            if isinstance(value, bool) or not isinstance(value, int):
                missing_items.append(item_id)
                continue
            if not int(scale["min"]) <= value <= int(scale["max"]):
                raise ValueError(f"题目 {item_id} 的评分超出量表范围。")
            normalized[item_id] = value
        if missing_items:
            raise ValueError(f"以下必答评分尚未完成：{', '.join(missing_items)}")

        failed_checks: list[str] = []
        for field_id, field in fields.items():
            value = answers.get(field_id)
            required = bool(field.get("required"))
            if value is None or value == "":
                if required:
                    raise ValueError(f"以下必答选择尚未完成：{field_id}")
                continue
            if field.get("type") != "choice" or not isinstance(value, str):
                raise ValueError(f"字段 {field_id} 的格式无效。")
            options = {str(option["value"]) for option in field.get("options", [])}
            if value not in options:
                raise ValueError(f"字段 {field_id} 的选项无效。")
            if value in {str(item) for item in field.get("reject_values", [])}:
                raise ValueError("当前选择不符合本研究的参与条件，请不要继续并联系研究团队。")
            if field.get("correct") is not None and value != str(field["correct"]):
                failed_checks.append(field_id)
            normalized[field_id] = value
        if failed_checks:
            raise ValueError("操作理解题未通过，请重新阅读说明并按真实理解作答。")

        for item_id, item in open_items.items():
            value = answers.get(item_id, "")
            if not isinstance(value, str):
                raise ValueError(f"开放题 {item_id} 必须是文本。")
            value = value.strip()
            if len(value) > 4000:
                raise ValueError(f"开放题 {item_id} 超过 4000 字符。")
            if item.get("required") and not value:
                raise ValueError(f"开放题 {item_id} 为必答题。")
            if value and schema_name == "final" and not open_text_meets_minimum(value):
                raise ValueError(
                    f"开放题 {item_id} 至少填写 {OPEN_TEXT_MIN_CHARS} 个汉字，"
                    f"或至少 {OPEN_TEXT_MIN_WORDS} 个英文单词。"
                )
            if value:
                normalized[item_id] = value

        for item_id, followup in followups.items():
            condition = followup["when"]
            score = normalized.get(str(condition["item"]))
            applies = bool(
                score is not None
                and (
                    ("lte" in condition and score <= int(condition["lte"]))
                    or ("gte" in condition and score >= int(condition["gte"]))
                )
            )
            value = answers.get(item_id, "")
            if not isinstance(value, str):
                raise ValueError(f"追问题 {item_id} 必须是文本。")
            value = value.strip()
            if len(value) > 4000:
                raise ValueError(f"追问题 {item_id} 超过 4000 字符。")
            if applies and not value:
                raise ValueError(f"请完成当前评分触发的追问题：{item_id}")
            if value and schema_name == "final" and not open_text_meets_minimum(value):
                raise ValueError(
                    f"追问题 {item_id} 至少填写 {OPEN_TEXT_MIN_CHARS} 个汉字，"
                    f"或至少 {OPEN_TEXT_MIN_WORDS} 个英文单词。"
                )
            if value:
                normalized[item_id] = value

        if "ended_early" in answers:
            if not isinstance(answers["ended_early"], bool):
                raise ValueError("ended_early 必须为布尔值。")
            normalized["ended_early"] = answers["ended_early"]
        return normalized
