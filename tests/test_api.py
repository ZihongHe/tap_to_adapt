"""Regression tests for api.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
import zipfile
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest


TEST_DATA = Path(tempfile.mkdtemp(prefix="tta-study-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    {
        "STUDY_DATA_DIR": str(TEST_DATA),
        "STUDY_TOKEN_SECRET": "test-token-secret",
        "STUDY_CODE_PEPPER": "test-code-pepper",
        "STUDY_IP_PEPPER": "test-ip-pepper",
        "STUDY_ADMIN_TOKEN": "test-admin-token",
        "CONSENT_EVIDENCE_SIGNING_KEY": (
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        ),
        "COMPENSATION_VAULT_KEY": (
            "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
        ),
        "FORMAL_STUDY_ENABLED": "0",
        "AGENT_MODE": "mock",
        "PUBLIC_BASE_URL": "http://testserver",
        "STUDY_INSTANCE_ID": "test",
        "PUBLIC_ENROLLMENT_ENABLED": "1",
    }
)

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import (
    PairConnection,
    PairHub,
    TRANSPORT_PROBE_PCM_BYTES,
    TimingHub,
    _ensure_server_consent_archive,
    _is_model_resource_error,
    _participant_readiness,
    app,
    compensation_vault,
    db,
    protocol,
    settings,
    study_state,
    timing_service,
)
from app.consent_evidence import verify_evidence_bundle
from app.matched import file_sha256, matched_assignment
from app.study_analysis import (
    _bootstrap_change_decomposition,
    _finish_metric,
    _is_timing_interrupt,
    _new_metric,
    _participant_day_changes,
)
from app.study_state import ProtocolViolation
from app.timing import DTCNModel, ModelUnavailable, ParticipantRuntime, TimingService


client = TestClient(app)
ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}
DESKTOP_DEVICE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
    ),
    "Sec-CH-UA-Mobile": "?0",
    "X-TTA-Device-Class": "desktop",
}
PHONE_DEVICE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
        "AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"
    ),
    "Sec-CH-UA-Mobile": "?1",
    "X-TTA-Device-Class": "phone",
}


def test_companion_code_path_redirects_to_auto_login_fragment() -> None:
    response = client.get(
        "/companion/TTA-ABCD-EFGH",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "http://testserver/companion#TTA-ABCD-EFGH"


def test_public_enrollment_creates_isolated_test_participant() -> None:
    response = client.post(
        "/api/enroll",
        headers=DESKTOP_DEVICE_HEADERS,
        json={
            "device_id": "browser-device-public-enrollment-001",
            "contact_email": "reviewer@example.invalid",
        },
    )
    assert response.status_code == 200
    value = response.json()
    assert value["instance_id"] == "test"
    assert value["reward_cny"] == 40
    assert value["help_wechat"] == "study-contact"
    participant = db.participant_by_code(value["code"])
    assert participant is not None
    assert participant["data_origin"] == "pilot"
    assert participant["analysis_excluded"] == 1
    model_dir = timing_service._personal_model_dir(participant)
    assert settings.data_dir in model_dir.parents
    assert model_dir.parts[-4:-1] == (
        "personal_debug",
        "public_enrollment",
        participant["id"],
    )
    assert model_dir.name == "models"
    bootstrap = client.get(
        "/api/bootstrap",
        headers={"Authorization": f"Bearer {value['token']}"},
    )
    assert bootstrap.status_code == 200
    assert bootstrap.json()["instance_id"] == "test"


def test_public_enrollment_rejects_phone_as_study_device() -> None:
    response = client.post(
        "/api/enroll",
        headers=PHONE_DEVICE_HEADERS,
        json={"device_id": "phone-device-invalid-primary-001"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "desktop_device_required"
    assert response.json()["detail"]["message"] == "无法开始，需要在电脑端打开实验系统"


def test_public_formal_enrollment_opens_when_readiness_is_advisory() -> None:
    overrides = {
        "instance_id": "formal",
        "formal_study_enabled": True,
        "formal_readiness_enforced": False,
        "public_enrollment_cohort": "study_a",
        "public_enrollment_group": "personalized",
        "public_enrollment_run_slug": "test-public-formal-advisory",
    }
    previous = {name: getattr(settings, name) for name in overrides}
    for name, value in overrides.items():
        object.__setattr__(settings, name, value)
    try:
        response = client.post(
            "/api/enroll",
            headers=DESKTOP_DEVICE_HEADERS,
            json={
                "device_id": "desktop-public-formal-advisory-001",
                "contact_email": "reviewer@example.invalid",
            },
        )
    finally:
        for name, value in previous.items():
            object.__setattr__(settings, name, value)

    assert response.status_code == 200
    value = response.json()
    assert value["instance_id"] == "formal"
    participant = db.participant_by_code(value["code"])
    assert participant is not None
    assert participant["data_origin"] == "formal"
    assert participant["analysis_excluded"] == 0


def test_login_and_session_start_enforce_physical_device_roles() -> None:
    code = create_participant(run_slug="test-device-role-gate")

    wrong_study = client.post(
        "/api/login",
        headers=PHONE_DEVICE_HEADERS,
        json={"code": code, "role": "study"},
    )
    assert wrong_study.status_code == 403
    assert wrong_study.json()["detail"]["code"] == "desktop_device_required"

    wrong_companion = client.post(
        "/api/login",
        headers=DESKTOP_DEVICE_HEADERS,
        json={"code": code, "role": "companion"},
    )
    assert wrong_companion.status_code == 403
    assert wrong_companion.json()["detail"]["code"] == "phone_device_required"

    study_login = client.post(
        "/api/login",
        headers=DESKTOP_DEVICE_HEADERS,
        json={"code": code, "role": "study"},
    )
    assert study_login.status_code == 200
    study_start = client.post(
        "/api/sessions/start",
        headers={
            **PHONE_DEVICE_HEADERS,
            "Authorization": f"Bearer {study_login.json()['token']}",
        },
        json={"day": 1},
    )
    assert study_start.status_code == 403
    assert study_start.json()["detail"]["code"] == "desktop_device_required"

    companion_login = client.post(
        "/api/login",
        headers=PHONE_DEVICE_HEADERS,
        json={"code": code, "role": "companion"},
    )
    assert companion_login.status_code == 200
    companion_start = client.post(
        "/api/sessions/start",
        headers={
            **DESKTOP_DEVICE_HEADERS,
            "Authorization": f"Bearer {companion_login.json()['token']}",
        },
        json={"day": 1},
    )
    assert companion_start.status_code == 403
    assert companion_start.json()["detail"]["code"] == "phone_device_required"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/admin/stage-accounts", {"stage": "engineering_smoke", "count": 1}),
        (
            "/api/admin/participants",
            {"count": 1, "cohort": "pilot", "group_name": "personalized"},
        ),
    ],
)
def test_admin_cannot_create_participant_accounts(path: str, payload: dict[str, Any]) -> None:
    response = client.post(path, headers=ADMIN_HEADERS, json=payload)
    assert response.status_code == 405


def test_english_companion_code_path_preserves_language() -> None:
    response = client.get(
        "/companion/TTA-ABCD-EFGH?lang=en",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == (
        "http://testserver/companion?lang=en#TTA-ABCD-EFGH"
    )


def test_english_materials_preserve_canonical_ids_and_consent_audit() -> None:
    code = create_participant()
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}

    bootstrap = client.get("/api/bootstrap?lang=en", headers=headers)
    assert bootstrap.status_code == 200
    consent = bootstrap.json()["consent"]
    assert consent["locale"] == "en"
    assert consent["title"].startswith("Tap-to-Adapt Voice Interaction Study")
    assert consent["sha256"] == protocol.consent_hashes["en"]
    assert [item["id"] for item in consent["choices"]] == [
        item["id"] for item in protocol.consent["choices"]
    ]

    signed = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert signed.status_code == 200
    assert client.get("/api/bootstrap", headers=headers).json()["consent_valid"] is True
    participant = db.participant_by_code(code)
    assert participant is not None
    stored = db.latest_consent(participant["id"])
    assert stored is not None
    assert json.loads(stored["document_json"])["title"] == consent["title"]

    topics = client.get("/api/topics?lang=en", headers=headers)
    assert topics.status_code == 200
    assert topics.json()["categories"][0]["label"] == "Ordinary Life"
    assert sum(len(item["prompts"]) for item in topics.json()["categories"]) == 150

    questionnaire = client.get("/api/questionnaires/baseline?lang=en", headers=headers)
    assert questionnaire.status_code == 200
    assert questionnaire.json()["title"] == "Pre-study information and procedure check"
    language = next(
        field
        for field in questionnaire.json()["fields"]
        if field["id"] == "primary_spoken_language"
    )
    assert {option["value"] for option in language["options"]} == {
        "mandarin",
        "cantonese",
        "english",
        "other_chinese",
        "other",
    }


def test_cuda_oom_is_classified_as_a_model_resource_error() -> None:
    assert _is_model_resource_error(
        RuntimeError("CUDA error: out of memory")
    )
    assert not _is_model_resource_error(RuntimeError("ordinary validation failure"))


def test_distilhubert_cpu_uses_a_non_backlogging_prediction_interval() -> None:
    runtime = ParticipantRuntime(
        participant_id="cpu-cadence",
        group_name="pilot",
        cohort="pilot",
        settings=replace(
            settings,
            agent_mode="distilhubert",
            distilhubert_device="cpu",
        ),
    )

    assert runtime.prediction_interval_ms == 400.0
    runtime.last_prediction_ms = 1_000.0
    skipped = runtime.process_pcm(np.zeros(1600, dtype="<i2").tobytes(), 1_300.0)
    assert skipped["skipped"] is True


def create_participant(
    cohort: str = "pilot",
    group: str = "pilot",
    preview_mode: bool = False,
    data_origin: str | None = None,
    protocol_mode: str | None = None,
    run_slug: str | None = None,
) -> str:
    # Tests provision fixtures directly. Production HTTP administrators are
    # deliberately unable to create participant accounts.
    return db.create_participants(
        1,
        cohort,
        group,
        preview_mode=preview_mode,
        data_origin=data_origin,
        protocol_mode=protocol_mode,
        run_slug=run_slug,
    )[0]


def login(code: str, role: str = "study") -> dict:
    response = client.post("/api/login", json={"code": code, "role": role})
    assert response.status_code == 200
    return response.json()


def consent_and_start(code: str, day: int = 1) -> tuple[dict, dict]:
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    bootstrap = client.get("/api/bootstrap", headers=headers).json()
    consent = bootstrap["consent"]
    response = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert response.status_code == 200
    save_compensation_identity(headers)
    session = client.post("/api/sessions/start", headers=headers, json={"day": day})
    assert session.status_code == 200
    if day == 1:
        baseline = client.post(
            "/api/responses",
            headers=headers,
            json={
                "session_id": session.json()["id"],
                "instrument": "baseline",
                "version": protocol.questionnaires["version"],
                "answers": baseline_answers(),
            },
        )
        assert baseline.status_code == 200
    return headers, session.json()


def save_compensation_identity(headers: dict[str, str]) -> dict[str, Any]:
    bootstrap = client.get("/api/bootstrap", headers=headers)
    assert bootstrap.status_code == 200
    bootstrap_payload = bootstrap.json()
    notice = bootstrap_payload["compensation_registration"]
    personal_debug = bool(
        bootstrap_payload.get("participant", {}).get("analysis_excluded")
    )
    signature_png = b"\x89PNG\r\n\x1a\n" + b"test-signature-pixels" * 24
    response = client.post(
        "/api/compensation-identity",
        headers=headers,
        json={
            "notice_version": notice["version"],
            "notice_hash": notice["sha256"],
            "document_type": "passport_or_other",
            "full_name": (
                "个人调试（非真实身份）"
                if personal_debug
                else "Integration Test Participant"
            ),
            "document_number": (
                f"DEBUG-{bootstrap_payload['participant']['code_hint']}"
                if personal_debug
                else f"TEST-{bootstrap_payload['participant']['code_hint']}"
            ),
            "signature_png_base64": base64.b64encode(signature_png).decode("ascii"),
            "separate_consent": True,
            "affirmation": notice["separate_affirmation"],
            "contact_email": "integration@example.test",
            "contact_phone": "",
            "contact_wechat": "",
        },
    )
    assert response.status_code == 200
    return response.json()


def test_compensation_notice_discloses_finance_basis_and_separate_authorisation() -> None:
    zh = protocol.public_compensation_notice("zh-CN")
    en = protocol.public_compensation_notice("en")
    zh_consent = protocol.public_consent("zh-CN")
    en_consent = protocol.public_consent("en")

    assert zh["version"] == "draft-compensation-2026-08-24-v8"
    assert "至少一项" in zh["purpose"]
    assert "邮箱、电话号码或微信号" in zh["contact_help"]
    assert any(
        "研究团队中经授权的报酬经办人员" in item and "学校财务部门" in item
        for item in zh["protections"]
    )
    assert any("保存期限：按照学校财务凭证保存规定" in item for item in zh["protections"])
    assert zh["separate_affirmation"] == (
        "我理解并同意研究团队为发放和报销实验报酬，收集并向学校财务部门提交"
        "我的姓名、身份证号码和电子签名。上述信息不会用于研究分析、模型训练或数据集发布。"
    )
    assert "at least one of email, telephone number, or WeChat ID" in en["purpose"]
    assert "University's finance department" in en["separate_affirmation"]
    assert "不影响40元基础报酬" in zh["choice_prompt"]
    assert "不保存上述身份信息" in zh["decline_affirmation"]
    assert "Study-Consent-TapToAdapt-v1.1-20260824" == zh_consent["version"] == en_consent["version"]
    zh_section = next(
        section
        for section in zh_consent["sections"]
        if section["heading"] == "学校统一报销信息与电子签名选择"
    )
    en_section = next(
        section
        for section in en_consent["sections"]
        if section["heading"] == "University reimbursement details and electronic-signature choice"
    )
    assert "学校财务统一报销要求" in " ".join(zh_section["paragraphs"])
    assert "学校财务部门" in " ".join(zh_section["paragraphs"])
    assert "学校财务凭证保存规定" in " ".join(zh_section["paragraphs"])
    assert "University's finance department" in " ".join(en_section["paragraphs"])


def baseline_answers() -> dict:
    return {
        "voice_technology_familiarity": 4,
        "privacy_concern": 4,
        "speech_agent_confidence": 4,
        "age_range": "18_22",
        "gender": "prefer_not",
        "primary_spoken_language": "mandarin",
        "voice_agent_frequency": "weekly",
        "hearing_or_speech_condition": "no",
        "tap_rule_check": "felt_need",
        "failure_rule_check": "pause_contact",
        "device_rule_check": "same_code_session",
    }


def test_compensation_identity_is_required_encrypted_and_auditable() -> None:
    code = create_participant(run_slug=f"test-compensation-{time.time_ns()}")
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    bootstrap = client.get("/api/bootstrap?lang=en", headers=headers).json()
    consent = bootstrap["consent"]
    signed = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert signed.status_code == 200

    blocked = client.post(
        "/api/sessions/start", headers=headers, json={"day": 1}
    )
    assert blocked.status_code == 403
    assert blocked.json()["detail"]["code"] == "compensation_choice_required"

    notice = client.get("/api/bootstrap?lang=en", headers=headers).json()[
        "compensation_registration"
    ]
    full_name = "Encrypted Example Participant"
    document_number = f"TEST-{time.time_ns()}"
    signature_png = b"\x89PNG\r\n\x1a\n" + b"private-signature-pixels" * 24
    missing_contact = client.post(
        "/api/compensation-identity",
        headers=headers,
        json={
            "notice_version": notice["version"],
            "notice_hash": notice["sha256"],
            "document_type": "passport_or_other",
            "full_name": full_name,
            "document_number": document_number,
            "signature_png_base64": base64.b64encode(signature_png).decode("ascii"),
            "separate_consent": True,
            "affirmation": notice["separate_affirmation"],
        },
    )
    assert missing_contact.status_code == 422
    assert "至少填写" in missing_contact.json()["detail"]
    saved = client.post(
        "/api/compensation-identity",
        headers=headers,
        json={
            "notice_version": notice["version"],
            "notice_hash": notice["sha256"],
            "document_type": "passport_or_other",
            "full_name": full_name,
            "document_number": document_number,
            "signature_png_base64": base64.b64encode(signature_png).decode("ascii"),
            "separate_consent": True,
            "affirmation": notice["separate_affirmation"],
            "contact_email": "encrypted-contact@example.test",
            "contact_phone": "",
            "contact_wechat": "",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["identity_plaintext_returned"] is False
    assert client.get("/api/bootstrap", headers=headers).json()[
        "compensation_identity_complete"
    ] is True

    vault_bytes = compensation_vault.path.read_bytes()
    assert full_name.encode("utf-8") not in vault_bytes
    assert document_number.encode("utf-8") not in vault_bytes
    assert b"encrypted-contact@example.test" not in vault_bytes
    assert signature_png not in vault_bytes
    record_id = saved.json()["record_id"]
    decrypted = compensation_vault.decrypt_for_offline_authorised_processing(
        record_id
    )
    assert decrypted["full_name"] == full_name
    assert decrypted["document_number"] == document_number
    assert decrypted["signature_png"] == signature_png
    assert decrypted["contact_email"] == "encrypted-contact@example.test"
    assert decrypted["contact_phone"] == ""
    assert decrypted["contact_wechat"] == ""
    assert json.loads(decrypted["notice_json"])["title"] == notice["title"]
    assert decrypted["affirmation"] == notice["separate_affirmation"]


def test_personal_debug_compensation_registration_accepts_the_normal_form() -> None:
    run_slug = f"test-debug-identity-{time.time_ns()}"
    code = db.create_participants(
        1,
        "pilot",
        "personalized",
        data_origin="pilot",
        protocol_mode="strict",
        run_slug=run_slug,
        analysis_excluded=True,
    )
    code = code[0]
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    bootstrap = client.get("/api/bootstrap", headers=headers).json()
    consent = bootstrap["consent"]
    signed = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert signed.status_code == 200
    notice = client.get("/api/bootstrap", headers=headers).json()[
        "compensation_registration"
    ]
    signature_png = b"\x89PNG\r\n\x1a\n" + b"debug-signature-pixels" * 24
    saved = client.post(
        "/api/compensation-identity",
        headers=headers,
        json={
            "notice_version": notice["version"],
            "notice_hash": notice["sha256"],
            "document_type": "passport_or_other",
            "full_name": "Personal Debug Test",
            "document_number": "DEBUG-NORMAL-FORM-123",
            "signature_png_base64": base64.b64encode(signature_png).decode("ascii"),
            "separate_consent": True,
            "affirmation": notice["separate_affirmation"],
            "contact_email": "",
            "contact_phone": "",
            "contact_wechat": "debug-wechat",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["decision"] == "provide"
    participant = db.participant_by_code(code)
    assert participant["analysis_excluded"] == 1
    assert compensation_vault.status(participant["id"])["decision"] == "provide"


def test_declining_identity_stores_contact_but_no_identity_and_allows_study() -> None:
    code = create_participant(run_slug=f"test-decline-identity-{time.time_ns()}")
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    bootstrap = client.get("/api/bootstrap", headers=headers).json()
    consent = bootstrap["consent"]
    signed = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert signed.status_code == 200
    notice = client.get("/api/bootstrap", headers=headers).json()[
        "compensation_registration"
    ]
    identity_count_before = compensation_vault.count()
    declined = client.post(
        "/api/compensation-decline",
        headers=headers,
        json={
            "notice_version": notice["version"],
            "notice_hash": notice["sha256"],
            "decision": "decline",
            "affirmation": notice["decline_affirmation"],
            "contact_email": "",
            "contact_phone": "+86 138-0000-0000",
            "contact_wechat": "",
        },
    )
    assert declined.status_code == 200
    assert declined.json()["decision"] == "decline"
    assert declined.json()["identity_plaintext_returned"] is False
    assert compensation_vault.count() == identity_count_before

    refreshed = client.get("/api/bootstrap", headers=headers).json()
    assert refreshed["compensation_registration_complete"] is True
    assert refreshed["compensation_registration_decision"] == "decline"
    assert refreshed["compensation_identity_complete"] is False
    participant = db.participant_by_code(code)
    registration = compensation_vault.status(participant["id"])
    assert registration["decision"] == "decline"
    assert registration["identity_record_id"] is None
    contact = compensation_vault.decrypt_contact_for_offline_authorised_processing(
        participant["id"]
    )
    assert contact["contact_phone"] == "+86 138-0000-0000"

    session = client.post("/api/sessions/start", headers=headers, json={"day": 1})
    assert session.status_code == 200


def test_existing_identity_choice_can_add_the_new_required_contact() -> None:
    code = create_participant(run_slug=f"test-contact-migration-{time.time_ns()}")
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    bootstrap = client.get("/api/bootstrap", headers=headers).json()
    consent = bootstrap["consent"]
    signed = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert signed.status_code == 200
    participant = db.participant_by_code(code)
    notice = client.get("/api/bootstrap", headers=headers).json()[
        "compensation_registration"
    ]
    compensation_vault.save_decline(
        run_id=str(participant["run_id"]),
        participant_id=str(participant["id"]),
        notice_version=notice["version"],
        notice_hash=notice["sha256"],
        notice_json=json.dumps(notice, ensure_ascii=False),
        affirmation=notice["decline_affirmation"],
        ip_hash="legacy-test",
        user_agent="legacy-test",
    )
    before = client.get("/api/bootstrap", headers=headers).json()
    assert before["compensation_registration_decision"] == "decline"
    assert before["compensation_registration_complete"] is False
    assert before["compensation_contact_complete"] is False

    saved = client.post(
        "/api/compensation-contact",
        headers=headers,
        json={
            "notice_version": notice["version"],
            "notice_hash": notice["sha256"],
            "contact_email": "",
            "contact_phone": "",
            "contact_wechat": "legacy-contact",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["contact_plaintext_returned"] is False
    after = client.get("/api/bootstrap", headers=headers).json()
    assert after["compensation_registration_complete"] is True
    assert after["compensation_contact_complete"] is True
    decrypted = compensation_vault.decrypt_contact_for_offline_authorised_processing(
        participant["id"]
    )
    assert decrypted["contact_wechat"] == "legacy-contact"


def save_required_quality(
    headers: dict, session_id: str, *, device_id: str | None = "test-browser-device"
) -> None:
    metrics = {
        "network": {
            "median_ms": 40,
            "p95_ms": 80,
            "jitter_ms": 12,
            "failures": 0,
            "samples": [35, 37, 38, 39, 40, 40, 42, 44, 47, 52, 60, 80],
        },
        "microphone": {
            "snr_db": 20,
            "quiet_rms_p90": 0.02,
            "speech_rms_p90": 0.2,
            "clipping_ratio": 0,
            "asr_probe_passed": True,
        },
        "reaction_latency": {
            "median_ms": 260,
            "samples_ms": [220, 240, 250, 270, 290, 310],
        },
        "browser": {
            "secure_context": True,
            "media_devices": True,
            "audio_worklet": True,
            "websocket": True,
            "speech_recognition": True,
            "media_recorder": True,
            "speech_synthesis": True,
            "playback_api": "speechSynthesis",
            "phrase_id": "zh-blue-kite-v1",
            "test_language": "zh-CN",
            "verification_method": "typed_keyword_recall",
            "playback_verified": True,
            "typed_answer_stored": False,
            "answer_chars": 12,
            "matched_keyword_count": 4,
            "required_keyword_count": 4,
            **({"device_id": device_id} if device_id else {}),
        },
    }
    for kind in ("network", "microphone", "reaction_latency", "browser"):
        response = client.post(
            "/api/quality",
            headers=headers,
            json={
                "session_id": session_id,
                "kind": kind,
                "passed": True,
                "metrics": {
                    **metrics[kind],
                    "attempt_id": f"{session_id}-{kind}-1",
                    "attempt_number": 1,
                },
            },
        )
        assert response.status_code == 200


def daily_answers() -> dict:
    return {
        "timing_today": 4,
        "effort_today": 4,
        "annoyance_today": 4,
        "predictable_today": 4,
        "speed_today": 4,
        "like_today": 4,
        "fatigue_today": 4,
        "control_today": 4,
    }


def final_answers() -> dict:
    instrument = protocol.questionnaires["instruments"]["final"]
    answers = {str(item["id"]): 4 for item in instrument.get("items", [])}
    answers.update(
        {
            str(item["id"]): "integration test answer"
            for item in instrument.get("open_items", [])
            if item.get("required")
        }
    )
    return answers


def compressed_protocol_events(probe_order: str) -> list[dict]:
    events = [
        {
            "event_type": "topic.selected",
            "payload": {
                "topic_id": "free_topic",
                "category_id": "free",
                "reused_topic_id": False,
            },
        },
        {"event_type": "phase.start", "payload": {"phase": "frozen", "policy_token": "A"}},
        {
            "event_type": "phase.end",
            "payload": {"phase": "frozen", "elapsed_seconds": 0, "opportunities": 0},
        },
    ]
    for window in range(1, 4):
        events.extend(
            [
                {
                    "event_type": "phase.start",
                    "payload": {"phase": "learning", "policy_token": "A"},
                },
                {"event_type": "phase.end", "payload": {"phase": "learning", "completed_seconds": 0}},
                {"event_type": "companion.window_start", "payload": {"window": window}},
                {
                    "event_type": "companion.window_end",
                    "payload": {"window": window, "completed_seconds": 0},
                },
            ]
        )
    events.extend(
        [
            {"event_type": "phase.start", "payload": {"phase": "learning", "policy_token": "A"}},
            {"event_type": "phase.end", "payload": {"phase": "learning", "completed_seconds": 0}},
        ]
    )
    for letter in probe_order:
        events.extend(
            [
                {
                    "event_type": "phase.start",
                    "payload": {"phase": "micro_probe", "policy_token": letter},
                },
                {
                    "event_type": "phase.end",
                    "payload": {
                        "phase": "micro_probe",
                        "policy_token": letter,
                        "completed_seconds": 0,
                    },
                },
            ]
        )
    return events


def test_pilot_flow_and_hash_chained_events() -> None:
    code = create_participant()
    headers, session = consent_and_start(code)

    events = [
        {
            "event_type": "test.phase_marker",
            "client_ms": 1000,
            "monotonic_ms": 10,
            "payload": {"phase": "frozen"},
        },
        {
            "event_type": "tap.pointer",
            "client_ms": 1200,
            "monotonic_ms": 210,
            "payload": {"tap_id": "test-tap", "accepted": False},
        },
    ]
    response = client.post(
        "/api/events/batch",
        headers=headers,
        json={"session_id": session["id"], "events": events},
    )
    assert response.status_code == 200
    assert response.json()["saved"] == 2

    participant = db.participant_by_code(code)
    rows = db.table_rows("events")
    assert [row["seq"] for row in rows if row["participant_id"] == participant["id"]] == [1, 2]
    participant_rows = [row for row in rows if row["participant_id"] == participant["id"]]
    assert participant_rows[0]["prev_hash"] == "0" * 64
    assert participant_rows[1]["prev_hash"] == participant_rows[0]["event_hash"]

    quality = client.post(
        "/api/quality",
        headers=headers,
        json={
            "session_id": session["id"],
            "kind": "network",
            "passed": True,
            "metrics": {
                "attempt_id": "network-hash-test-1",
                "attempt_number": 1,
                "median_ms": 20,
                "p95_ms": 25,
                "jitter_ms": 3,
                "failures": 0,
                "samples": [20] * 12,
            },
        },
    )
    assert quality.status_code == 200

    response = client.post(
        f"/api/audio/{session['id']}/study_stream",
        headers={**headers, "Content-Type": "audio/webm"},
        content=b"not-retained",
    )
    assert response.status_code == 403


def test_entry_html_is_not_cached() -> None:
    for path in ("/", "/companion", "/admin", "/transport-probe"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"


def _probe_id(label: str) -> str:
    return f"probe-{label}-{time.time_ns()}"


def _study_storage_counts() -> dict[str, int]:
    with db.connect() as connection:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("participants", "sessions", "events", "pair_intervals")
        }


def test_transport_probe_acks_timing_and_pair_without_study_side_effects() -> None:
    before = _study_storage_counts()
    probe_id = _probe_id("happy")

    with client.websocket_connect("/ws/transport-probe") as socket:
        socket.send_json(
            {
                "type": "hello",
                "probe_id": probe_id,
                "user_index": 0,
                "channel": "timing",
            }
        )
        hello = socket.receive_json()
        assert hello["type"] == "hello_ack"
        assert hello["protocol_version"] == "transport-probe-v1"
        assert hello["pcm_frame_bytes"] == TRANSPORT_PROBE_PCM_BYTES
        client_sent_ms = time.time() * 1000
        socket.send_bytes(
            struct.pack("!Id", 7, client_sent_ms)
            + bytes(TRANSPORT_PROBE_PCM_BYTES)
        )
        ack = socket.receive_json()
        assert ack["type"] == "probe_ack"
        assert ack["channel"] == "timing"
        assert ack["seq"] == 7
        assert ack["client_sent_ms"] == pytest.approx(client_sent_ms)
        assert ack["server_sent_ms"] >= ack["server_received_ms"]

    with client.websocket_connect("/ws/transport-probe") as socket:
        socket.send_json(
            {
                "type": "hello",
                "probe_id": probe_id,
                "user_index": 0,
                "channel": "study_pair",
            }
        )
        assert socket.receive_json()["type"] == "hello_ack"
        client_sent_ms = time.time() * 1000
        socket.send_json(
            {"type": "heartbeat", "seq": 0, "client_sent_ms": client_sent_ms}
        )
        ack = socket.receive_json()
        assert ack["type"] == "probe_ack"
        assert ack["channel"] == "study_pair"
        assert ack["seq"] == 0
        assert ack["client_sent_ms"] == pytest.approx(client_sent_ms)

    assert _study_storage_counts() == before


def test_transport_probe_rejects_wrong_frame_size_and_repeated_sequence() -> None:
    with client.websocket_connect("/ws/transport-probe") as socket:
        socket.send_json(
            {
                "type": "hello",
                "probe_id": _probe_id("size"),
                "user_index": 0,
                "channel": "timing",
            }
        )
        assert socket.receive_json()["type"] == "hello_ack"
        socket.send_bytes(bytes(TRANSPORT_PROBE_PCM_BYTES))
        error = socket.receive_json()
        assert error["type"] == "probe_error"
        assert error["code"] == "invalid_frame_size"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 1009

    with client.websocket_connect("/ws/transport-probe") as socket:
        socket.send_json(
            {
                "type": "hello",
                "probe_id": _probe_id("sequence"),
                "user_index": 1,
                "channel": "companion_pair",
            }
        )
        assert socket.receive_json()["type"] == "hello_ack"
        heartbeat = {"type": "heartbeat", "seq": 4, "client_sent_ms": time.time() * 1000}
        socket.send_json(heartbeat)
        assert socket.receive_json()["type"] == "probe_ack"
        socket.send_json(heartbeat)
        error = socket.receive_json()
        assert error["code"] == "non_monotonic_sequence"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 4400


def test_transport_probe_requires_strict_hello_and_bounds_connection_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with client.websocket_connect("/ws/transport-probe") as socket:
        socket.send_json(
            {
                "type": "hello",
                "probe_id": _probe_id("hello"),
                "user_index": 0,
                "channel": "study_pair",
                "unexpected": True,
            }
        )
        error = socket.receive_json()
        assert error["code"] == "invalid_hello"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 4400

    monkeypatch.setattr("app.main.TRANSPORT_PROBE_MAX_CONNECTION_SECONDS", 0.03)
    with client.websocket_connect("/ws/transport-probe") as socket:
        socket.send_json(
            {
                "type": "hello",
                "probe_id": _probe_id("duration"),
                "user_index": 0,
                "channel": "study_pair",
            }
        )
        assert socket.receive_json()["type"] == "hello_ack"
        error = socket.receive_json()
        assert error["code"] == "connection_duration_exceeded"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 4408


def test_transport_probe_enforces_pair_rate_and_per_ip_connection_limits() -> None:
    with client.websocket_connect("/ws/transport-probe") as socket:
        socket.send_json(
            {
                "type": "hello",
                "probe_id": _probe_id("rate"),
                "user_index": 0,
                "channel": "study_pair",
            }
        )
        assert socket.receive_json()["type"] == "hello_ack"
        sent_ms = time.time() * 1000
        for sequence in range(5):
            socket.send_json(
                {"type": "heartbeat", "seq": sequence, "client_sent_ms": sent_ms}
            )
        for sequence in range(4):
            ack = socket.receive_json()
            assert ack["type"] == "probe_ack"
            assert ack["seq"] == sequence
        error = socket.receive_json()
        assert error["code"] == "rate_limit_exceeded"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 4429

    probe_id = _probe_id("connections")
    channels = ("timing", "study_pair", "companion_pair")
    with ExitStack() as stack:
        for user_index in (0, 1):
            for channel in channels:
                socket = stack.enter_context(
                    client.websocket_connect("/ws/transport-probe")
                )
                socket.send_json(
                    {
                        "type": "hello",
                        "probe_id": probe_id,
                        "user_index": user_index,
                        "channel": channel,
                    }
                )
                assert socket.receive_json()["type"] == "hello_ack"
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/ws/transport-probe"):
                pass
        assert closed.value.code == 4429


def test_bootstrap_contains_open_session_and_quality_recovery_state() -> None:
    code = create_participant()
    headers, session = consent_and_start(code)
    save_required_quality(headers, session["id"])

    bootstrap = client.get("/api/bootstrap", headers=headers)
    assert bootstrap.status_code == 200
    progress = bootstrap.json()["progress"]
    study_session = next(
        row for row in progress["sessions"] if row["client_role"] == "study"
    )
    assert study_session["id"] == session["id"]
    assert study_session["activated_ms"] is None
    recovered_quality = {
        row["kind"]: row
        for row in progress["quality"]
        if row["session_id"] == session["id"]
    }
    assert set(recovered_quality) == {
        "network",
        "microphone",
        "reaction_latency",
        "browser",
    }
    assert all(row["day"] == 1 and row["passed"] == 1 for row in recovered_quality.values())


def test_model_preflight_failure_does_not_activate_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = create_participant(run_slug="test-model-preflight-before-activation")
    headers, session = consent_and_start(code)
    save_required_quality(headers, session["id"])
    participant = db.participant_by_code(code)
    assert participant is not None

    def unavailable(*_args: Any, **_kwargs: Any) -> ParticipantRuntime:
        raise ModelUnavailable("synthetic model preflight failure")

    monkeypatch.setattr(timing_service, "runtime", unavailable)
    response = client.post(
        f"/api/sessions/{session['id']}/activate",
        headers=headers,
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "timing_model_unavailable"
    persisted = db.session(session["id"], participant["id"])
    assert persisted is not None
    assert persisted["state"] == "started"
    assert persisted["activated_ms"] is None


def test_same_code_same_day_pair_is_a_server_side_activation_gate() -> None:
    code = create_participant()
    study_headers, study_session = consent_and_start(code, day=1)
    companion_login = login(code, "companion")
    companion_headers = {"Authorization": f"Bearer {companion_login['token']}"}
    companion_session_response = client.post(
        "/api/sessions/start", headers=companion_headers, json={"day": 1}
    )
    assert companion_session_response.status_code == 200
    companion_session = companion_session_response.json()
    save_required_quality(study_headers, study_session["id"])

    before_pair = client.post(
        f"/api/sessions/{study_session['id']}/activate", headers=study_headers
    )
    assert before_pair.status_code == 409

    study_url = (
        f"/ws/pair?token={urllib.parse.quote(study_headers['Authorization'][7:])}"
        f"&session_id={study_session['id']}"
    )
    companion_url = (
        f"/ws/pair?token={urllib.parse.quote(companion_login['token'])}"
        f"&session_id={companion_session['id']}"
    )
    with client.websocket_connect(study_url) as study_ws:
        assert study_ws.receive_json()["same_day_online"] is False
        with client.websocket_connect(companion_url) as companion_ws:
            companion_ws.send_json(
                {"type": "background_state", "action": "ready", "ready": True}
            )
            time.sleep(1.0)
            ready = None
            for _ in range(8):
                message = study_ws.receive_json()
                if message.get("type") == "pair_status" and message.get("companion_ready"):
                    ready = message
                    break
            assert ready is not None
            assert ready["same_day_online"] is True
            assert ready["day"] == 1

            activated = client.post(
                f"/api/sessions/{study_session['id']}/activate", headers=study_headers
            )
            assert activated.status_code == 200
            assert activated.json()["pair_qualified"] is True
            assert db.session(study_session["id"], db.participant_by_code(code)["id"])[
                "state"
            ] == "active"

    intervals = db.pair_intervals(db.participant_by_code(code)["id"], day=1)
    assert intervals
    assert sum(int(row["qualified"]) for row in intervals) == 1
    qualified = next(row for row in intervals if row["qualified"])
    assert qualified["qualified_ms"] is not None
    assert intervals[-1]["ended_ms"] is not None


def test_activated_pair_reconnect_is_requalified_without_reactivation() -> None:
    code = create_participant()
    study_headers, study_session = consent_and_start(code, day=1)
    companion_login = login(code, "companion")
    companion_headers = {"Authorization": f"Bearer {companion_login['token']}"}
    companion_session = client.post(
        "/api/sessions/start", headers=companion_headers, json={"day": 1}
    ).json()
    save_required_quality(study_headers, study_session["id"])
    participant = db.participant_by_code(code)
    study_url = (
        f"/ws/pair?token={urllib.parse.quote(study_headers['Authorization'][7:])}"
        f"&session_id={study_session['id']}"
    )
    companion_url = (
        f"/ws/pair?token={urllib.parse.quote(companion_login['token'])}"
        f"&session_id={companion_session['id']}"
    )

    with client.websocket_connect(study_url) as study_ws:
        study_ws.receive_json()
        with client.websocket_connect(companion_url) as companion_ws:
            companion_ws.send_json(
                {"type": "background_state", "action": "ready", "ready": True}
            )
            time.sleep(1.0)
            for _ in range(8):
                if study_ws.receive_json().get("companion_ready"):
                    break
            activated = client.post(
                f"/api/sessions/{study_session['id']}/activate", headers=study_headers
            )
            assert activated.status_code == 200

    with client.websocket_connect(study_url) as study_ws:
        study_ws.receive_json()
        with client.websocket_connect(companion_url) as companion_ws:
            companion_ws.send_json(
                {"type": "background_state", "action": "ready", "ready": True}
            )
            time.sleep(1.0)
            ready = None
            for _ in range(8):
                message = study_ws.receive_json()
                if message.get("type") == "pair_status" and message.get("companion_ready"):
                    ready = message
                    break
            assert ready is not None
            intervals = db.pair_intervals(participant["id"], day=1)
            assert sum(int(row["qualified"]) for row in intervals) == 2
            assert len(
                {
                    (row["study_session_id"], row["companion_session_id"])
                    for row in intervals
                    if row["qualified"]
                }
            ) == 1


def test_timing_attribution_ignores_client_wall_clock_skew() -> None:
    code = create_participant(run_slug="test-server-monotonic-timing")
    study_headers, study_session = consent_and_start(code, day=1)
    companion_login = login(code, "companion")
    companion_headers = {"Authorization": f"Bearer {companion_login['token']}"}
    companion_session = client.post(
        "/api/sessions/start", headers=companion_headers, json={"day": 1}
    ).json()
    save_required_quality(study_headers, study_session["id"])
    participant = db.participant_by_code(code)

    study_url = (
        f"/ws/pair?token={urllib.parse.quote(study_headers['Authorization'][7:])}"
        f"&session_id={study_session['id']}"
    )
    companion_url = (
        f"/ws/pair?token={urllib.parse.quote(companion_login['token'])}"
        f"&session_id={companion_session['id']}"
    )
    timing_url = (
        f"/ws/timing?token={urllib.parse.quote(study_headers['Authorization'][7:])}"
        f"&session_id={study_session['id']}"
    )
    with client.websocket_connect(study_url) as study_pair:
        study_pair.receive_json()
        with client.websocket_connect(companion_url) as companion_pair:
            companion_pair.send_json(
                {"type": "background_state", "action": "ready", "ready": True}
            )
            time.sleep(1.0)
            for _ in range(8):
                if study_pair.receive_json().get("companion_ready"):
                    break
            activated = client.post(
                f"/api/sessions/{study_session['id']}/activate", headers=study_headers
            )
            assert activated.status_code == 200

            with client.websocket_connect(timing_url) as timing_socket:
                ready = timing_socket.receive_json()
                assert (
                    ready["latency_clock"]
                    == "server_monotonic_activation_commit_v2"
                )
                assert (
                    ready["negative_label_semantics"]
                    == "observable_activation_interrupt_v2"
                )
                assert ready["negative_label_latency_cutoff_ms"] is None
                assert ready["negative_label_requires_pre_tap_speech"] is False
                timing_socket.send_json({"type": "tap", "tap_id": ""})
                invalid_tap = timing_socket.receive_json()
                assert invalid_tap["type"] == "timing_error"
                assert invalid_tap["code"] == "invalid_tap_id"
                timing_socket.send_json(
                    {
                        "type": "tap",
                        "tap_id": "partial-conditional-tap",
                        "intended_action": "manual_activate",
                    }
                )
                incomplete_tap = timing_socket.receive_json()
                assert incomplete_tap["type"] == "timing_error"
                assert incomplete_tap["code"] == "incomplete_conditional_tap"
                assert set(incomplete_tap["missing_fields"]) == {
                    "expected_state",
                    "expected_activation_id",
                    "expected_state_version",
                }
                timing_socket.send_json(
                    {
                        "type": "phase",
                        "phase": "learning",
                        "policy_token": "A",
                        "drain_active": True,
                    }
                )
                assert timing_socket.receive_json()["type"] == "phase_ack"
                runtime = timing_service.runtime(
                    participant, 1, session_id=study_session["id"]
                )

                class FakeModel:
                    def predict(self, _audio):
                        return 1.0, 8.0

                    def train(self, _audio, _label, weight=1.0):
                        return float(weight)

                    def save(self, target):
                        return target / "encoder.pth", target / "main.pth"

                runtime.model = FakeModel()  # type: ignore[assignment]
                runtime.last_voice_ms = -10_000.0
                timing_socket.send_json(
                    {
                        "type": "decision_context",
                        "conversation_target": "experimental_agent",
                        "reply_needed_probability": 1.0,
                        "source": "unit_test",
                    }
                )
                assert timing_socket.receive_json()["type"] == "decision_context_ack"
                timing_socket.send_bytes(b"\x00\x00" * 3200)
                prediction = timing_socket.receive_json()
                assert prediction["activate"] is True
                assert prediction["activation_source"] == "model"
                assert prediction["activation_compute_ms"] >= 0

                # A reply-needed classifier can finish after the model has
                # already activated.  That stale result must be acknowledged
                # and discarded, not promoted to a global timing-model error.
                timing_socket.send_json(
                    {
                        "type": "decision_context",
                        "transition_id": "late-context-after-activation",
                        "conversation_target": "experimental_agent",
                        "reply_needed_probability": 1.0,
                        "source": "test-late-classifier",
                    }
                )
                stale_context = timing_socket.receive_json()
                assert stale_context["type"] == "decision_context_ack"
                assert stale_context["transition_id"] == "late-context-after-activation"
                assert stale_context["accepted"] is False
                assert stale_context["stale"] is True
                assert stale_context["rejection_reason"] == "agent_not_waiting"
                assert stale_context["server_agent_state"] == "thinking"

                timing_socket.send_bytes(b"\xff\x1f" * 3200)
                timing_socket.receive_json()
                timing_socket.send_json(
                    {
                        "type": "tap",
                        "tap_id": "clock-skew-tap",
                        "pointer_agent_state": "thinking",
                        "client_ms": 9_999_999_999_999.0,
                        "monotonic_ms": 9_999_999_999_999.0,
                        "visible": True,
                    }
                )
                result = timing_socket.receive_json()
                assert result["type"] == "tap_result"
                assert result["training_eligible"] is True
                assert result["label"] == 0
                assert result["label_semantics"] == "observable_activation_interrupt_v2"
                assert result["label_confidence"] == "high"
                assert result["interrupt_agent_state"] == "thinking"
                assert result["activation_to_tap_ms"] < 2000
                assert (
                    result["latency_clock"]
                    == "server_monotonic_activation_commit_v2"
                )
                assert result["client_clock_offset_ms"] > 1_000_000


def test_topic_selection_is_idempotent_and_explicit_changes_are_auditable() -> None:
    code = create_participant(run_slug="test-v61-topic-state")
    headers, session = consent_and_start(code)

    selected = client.post(
        "/api/events",
        headers=headers,
        json={
            "session_id": session["id"],
            "event_type": "topic.selected",
            "payload": {
                "topic_id": "ordinary_life_01",
                "category_id": "ordinary_life",
                "reused_topic_id": False,
            },
        },
    )
    assert selected.status_code == 200
    duplicate = client.post(
        "/api/events",
        headers=headers,
        json={
            "session_id": session["id"],
            "event_type": "topic.selected",
            "payload": {
                "topic_id": "ordinary_life_01",
                "category_id": "ordinary_life",
                "reused_topic_id": False,
            },
        },
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "topic_already_selected"

    changed = client.post(
        "/api/events",
        headers=headers,
        json={
            "session_id": session["id"],
            "event_type": "topic.changed",
            "payload": {
                "from_topic_id": "ordinary_life_01",
                "topic_id": "school_life_01",
                "category_id": "school_life",
                "reused_topic_id": False,
                "user_initiated": True,
            },
        },
    )
    assert changed.status_code == 200
    history = db.selected_topics(db.participant_by_code(code)["id"])
    assert [row["event_type"] for row in history] == [
        "topic.selected",
        "topic.changed",
    ]
    assert history[-1]["topic_id"] == "school_life_01"


def test_compressed_run_requires_protocol_and_then_completes_server_side() -> None:
    code = create_participant(run_slug="test-compressed-completion")
    headers, study_session = consent_and_start(code, day=1)
    companion = login(code, "companion")
    companion_headers = {"Authorization": f"Bearer {companion['token']}"}
    companion_session = client.post(
        "/api/sessions/start", headers=companion_headers, json={"day": 1}
    ).json()
    save_required_quality(headers, study_session["id"])

    blocked = client.post(f"/api/sessions/{study_session['id']}/end", headers=headers)
    assert blocked.status_code == 409
    assert "phase_sequence_or_duration" in blocked.json()["detail"]["checks"]["failures"]

    participant = db.participant_by_code(code)
    activated = db.activate_paired_sessions(
        participant["id"], study_session["id"], companion_session["id"]
    )
    interval_id = db.open_pair_interval(
        participant["id"], 1, study_session["id"], companion_session["id"]
    )
    db.qualify_pair_interval(
        interval_id,
        participant["id"],
        study_session["id"],
        companion_session["id"],
        int(activated["activated_ms"]),
    )
    events = compressed_protocol_events(json.loads(participant["probe_schedule"])[0])
    saved = client.post(
        "/api/events/batch",
        headers=headers,
        json={"session_id": study_session["id"], "events": events},
    )
    assert saved.status_code == 200
    db.append_event(
        participant["id"],
        study_session["id"],
        "chat.turn",
        None,
        None,
        {"source": "primary", "transcript": "test", "reply": "test"},
    )
    db.append_event(
        participant["id"],
        study_session["id"],
        "tts.generated",
        None,
        None,
        {"provider": "iflytek", "fallback": False},
    )
    db.append_event(
        participant["id"],
        study_session["id"],
        "tap.pointer",
        None,
        None,
        {"tap_id": "late-complete-tap", "accepted": True},
    )
    db.append_event(
        participant["id"],
        study_session["id"],
        "tap.manual_activate",
        None,
        None,
        {
            "tap_id": "late-complete-tap",
            "action": "manual_activate",
            "phase": "complete",
            "tap_phase": "complete",
            "training_eligible": False,
        },
    )
    response = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": study_session["id"],
            "instrument": "daily_day_1",
            "version": protocol.questionnaires["version"],
            "answers": daily_answers(),
        },
    )
    assert response.status_code == 200

    completed = client.post(f"/api/sessions/{study_session['id']}/end", headers=headers)
    assert completed.status_code == 200
    assert completed.json()["protocol"]["tap_receipts"]["after_complete_ignored"] == 1
    assert completed.json()["protocol"]["complete"] is True
    assert db.session(study_session["id"], participant["id"])["state"] == "completed"

    audited = client.get(
        f"/api/sessions/{study_session['id']}/protocol-status", headers=headers
    )
    assert audited.status_code == 200
    assert audited.json()["complete"] is True
    assert audited.json()["session"]["state"] == "completed"
    assert audited.json()["pair"]["paired_at_completion"] is True

    day_two = client.post("/api/sessions/start", headers=headers, json={"day": 2})
    assert day_two.status_code == 409
    assert day_two.json()["detail"]["code"] == "session_out_of_range"


def test_same_session_pair_cannot_be_activated_twice() -> None:
    code = create_participant(run_slug="test-single-activation")
    headers, study_session = consent_and_start(code, day=1)
    companion = login(code, "companion")
    companion_headers = {"Authorization": f"Bearer {companion['token']}"}
    companion_session = client.post(
        "/api/sessions/start", headers=companion_headers, json={"day": 1}
    ).json()
    save_required_quality(headers, study_session["id"])
    participant = db.participant_by_code(code)

    db.activate_paired_sessions(
        participant["id"], study_session["id"], companion_session["id"]
    )

    with pytest.raises(ProtocolViolation) as violation:
        study_state.assert_can_activate(participant, study_session["id"])
    assert violation.value.code == "session_already_active"

    with pytest.raises(ValueError, match="already been activated"):
        db.activate_paired_sessions(
            participant["id"], study_session["id"], companion_session["id"]
        )


def test_single_session_boundary_is_server_enforced() -> None:
    compressed_code = create_participant(run_slug="test-day-order")
    compressed_headers, _ = consent_and_start(compressed_code, day=1)
    skipped = client.post("/api/sessions/start", headers=compressed_headers, json={"day": 2})
    assert skipped.status_code == 409
    assert skipped.json()["detail"]["code"] == "session_out_of_range"

    strict_code = create_participant(
        protocol_mode="strict", run_slug="test-strict-pilot"
    )
    strict_headers, strict_day_one = consent_and_start(strict_code, day=1)
    participant = db.participant_by_code(strict_code)
    db.end_session(strict_day_one["id"], participant["id"])
    too_soon = client.post("/api/sessions/start", headers=strict_headers, json={"day": 2})
    assert too_soon.status_code == 409
    assert too_soon.json()["detail"]["code"] == "session_out_of_range"


def test_questionnaire_schema_rejects_unknown_missing_and_wrong_day_answers() -> None:
    code = create_participant(run_slug="test-questionnaire-schema")
    headers, session = consent_and_start(code, day=1)
    invalid = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": session["id"],
            "instrument": "daily_day_1",
            "version": protocol.questionnaires["version"],
            "answers": {"timing_satisfaction": 4},
        },
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "questionnaire_invalid"

    wrong_day = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": session["id"],
            "instrument": "daily_day_2",
            "version": protocol.questionnaires["version"],
            "answers": daily_answers(),
        },
    )
    assert wrong_day.status_code == 422

    valid = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": session["id"],
            "instrument": "daily_day_1",
            "version": protocol.questionnaires["version"],
            "answers": daily_answers(),
        },
    )
    assert valid.status_code == 200


def test_questionnaire_persists_random_order_and_selected_option_text_for_admin() -> None:
    code = create_participant(run_slug="test-questionnaire-answer-labels")
    headers, session = consent_and_start(code, day=1)
    public = client.get(
        "/api/questionnaires/baseline?lang=en", headers=headers
    ).json()
    canonical_order = [
        *[f"item:{item['id']}" for item in public.get("items", [])],
        *[f"field:{field['id']}" for field in public.get("fields", [])],
        *[f"open:{item['id']}" for item in public.get("open_items", [])],
    ]
    presented_order = list(reversed(canonical_order))
    saved = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": session["id"],
            "instrument": "baseline",
            "version": protocol.questionnaires["version"],
            "answers": baseline_answers(),
            "locale": "en",
            "question_order": presented_order,
        },
    )
    assert saved.status_code == 200

    participant = db.participant_by_code(code)
    assert participant is not None
    row = next(
        value
        for value in db.participant_table_rows("responses", participant["id"])
        if value["instrument"] == "baseline"
    )
    assert row["locale"] == "en"
    assert json.loads(row["question_order"]) == presented_order
    snapshots = json.loads(row["answer_labels"])
    age_field = next(
        field for field in public["fields"] if field["id"] == "age_range"
    )
    expected_label = next(
        option["label"]
        for option in age_field["options"]
        if option["value"] == "18_22"
    )
    age_snapshot = next(
        value for value in snapshots if value["question_id"] == "age_range"
    )
    assert age_snapshot["question_text"] == age_field["text"]
    assert age_snapshot["answer_value"] == "18_22"
    assert age_snapshot["answer_text"] == expected_label
    assert age_snapshot["position"] == presented_order.index("field:age_range") + 1

    detail = client.get(
        f"/api/admin/participants/{participant['id']}/records",
        headers=ADMIN_HEADERS,
    )
    assert detail.status_code == 200
    baseline_record = next(
        value
        for value in detail.json()["responses"]
        if value["instrument"] == "baseline"
    )
    assert baseline_record["question_order"] == presented_order
    assert any(
        value["question_id"] == "age_range"
        and value["answer_text"] == expected_label
        for value in baseline_record["selections"]
    )

    invalid_order = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": session["id"],
            "instrument": "baseline",
            "version": protocol.questionnaires["version"],
            "answers": baseline_answers(),
            "locale": "en",
            "question_order": presented_order[:-1],
        },
    )
    assert invalid_order.status_code == 422
    assert invalid_order.json()["detail"]["code"] == "questionnaire_presentation_invalid"


def test_questionnaire_without_study_session_returns_actionable_error() -> None:
    code = create_participant(run_slug="test-questionnaire-session-required")
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    response = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": None,
            "instrument": "baseline",
            "version": protocol.questionnaires["version"],
            "answers": baseline_answers(),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "study_session_required"
    assert "电脑端实验会话" in response.json()["detail"]["message"]


def test_baseline_comprehension_is_server_validated_and_answers_are_not_disclosed() -> None:
    code = create_participant(run_slug="test-baseline-comprehension")
    headers, session = consent_and_start(code, day=1)
    public = client.get("/api/questionnaires/baseline", headers=headers)
    assert public.status_code == 200
    assert all("correct" not in field for field in public.json()["fields"])

    wrong = baseline_answers()
    wrong["tap_rule_check"] = "fixed_rate"
    rejected = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": session["id"],
            "instrument": "baseline",
            "version": protocol.questionnaires["version"],
            "answers": wrong,
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "questionnaire_invalid"


def test_formal_activation_requires_a_pseudonymous_study_device_id() -> None:
    code = create_participant(run_slug="test-formal-device-id")
    headers, session = consent_and_start(code, day=1)
    save_required_quality(headers, session["id"], device_id=None)
    participant = {**db.participant_by_code(code), "data_origin": "formal"}

    with pytest.raises(ProtocolViolation) as violation:
        study_state.assert_can_activate(participant, session["id"])
    assert violation.value.code == "device_identity_missing"

    saved = client.post(
        "/api/quality",
        headers=headers,
        json={
            "session_id": session["id"],
            "kind": "browser",
            "passed": True,
            "metrics": {
                "attempt_id": "formal-browser-2",
                "attempt_number": 2,
                "device_id": "browser-fixture-1",
                "secure_context": True,
                "media_devices": True,
                "audio_worklet": True,
                "websocket": True,
                "speech_recognition": True,
                "media_recorder": True,
                "speech_synthesis": True,
                "playback_api": "speechSynthesis",
                "phrase_id": "zh-blue-kite-v1",
                "test_language": "zh-CN",
                "verification_method": "typed_keyword_recall",
                "playback_verified": True,
                "typed_answer_stored": False,
                "answer_chars": 12,
                "matched_keyword_count": 4,
                "required_keyword_count": 4,
            },
        },
    )
    assert saved.status_code == 200
    study_state.assert_can_activate(participant, session["id"])


def test_run_origin_is_immutable_and_propagates_to_child_records() -> None:
    code = create_participant(
        cohort="study_a",
        group="donor",
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug="test-simulation-origin",
    )
    headers, session = consent_and_start(code)
    participant = db.participant_by_code(code)
    assert participant["data_origin"] == "simulation"
    assert participant["run_slug"] == "test-simulation-origin"
    assert participant["run_protocol_sha256"] == db.protocol_sha256
    run = db.study_run("test-simulation-origin")
    assert run["protocol_sha256"] == db.protocol_sha256
    assert run["protocol_json"] == db.protocol_text
    db.append_event(participant["id"], session["id"], "test.origin", None, None, {})
    event = [row for row in db.table_rows("events") if row["participant_id"] == participant["id"]][-1]
    assert event["run_id"] == participant["run_id"]

    with db.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE participants SET run_id=? WHERE id=?",
                ("different-run", participant["id"]),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE study_runs SET protocol_sha256=? WHERE id=?",
                ("0" * 64, participant["run_id"]),
            )


def test_assignment_events_and_consent_are_database_immutable() -> None:
    code = create_participant(
        cohort="study_a",
        group="donor",
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug="test-immutable-records",
    )
    headers, session = consent_and_start(code)
    participant = db.participant_by_code(code)
    db.configure_participant(
        participant["id"],
        assignment_id="QA01",
        neutral_id="Q001",
        enrollment_wave=1,
        release_state="released",
    )
    event = db.append_event(participant["id"], session["id"], "test.immutable", None, None, {})

    with db.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE participants SET group_name='frozen' WHERE id=?",
                (participant["id"],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE events SET payload='{}' WHERE participant_id=? AND seq=?",
                (participant["id"], event["seq"]),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM events WHERE participant_id=?", (participant["id"],))

    consent = db.latest_consent(participant["id"])
    with db.connect_consent() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE consents SET raw_audio=1 WHERE id=?", (consent["id"],))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM consents WHERE id=?", (consent["id"],))


def test_formal_participant_is_hard_locked_before_consent() -> None:
    code = db.create_participants(
        1,
        "study_b",
        "personalized",
        data_origin="formal",
        protocol_mode="strict",
        run_slug="test-formal-lock",
    )[0]
    response = client.post("/api/login", json={"code": code, "role": "study"})
    assert response.status_code == 423
    detail = response.json()["detail"]
    assert detail["code"] == "formal_study_locked"
    assert "participant_released" in detail["missing"]
    assert "正式研究" in detail["message"]
    participant = db.participant_by_code(code)
    assert db.latest_consent(participant["id"]) is None


def test_formal_participant_can_enter_when_readiness_is_advisory() -> None:
    code = db.create_participants(
        1,
        "study_a",
        "personalized",
        data_origin="formal",
        protocol_mode="strict",
        run_slug="test-formal-advisory-readiness",
    )[0]
    previous = settings.formal_readiness_enforced
    object.__setattr__(settings, "formal_readiness_enforced", False)
    try:
        response = client.post(
            "/api/login",
            headers=DESKTOP_DEVICE_HEADERS,
            json={"code": code, "role": "study"},
        )
    finally:
        object.__setattr__(settings, "formal_readiness_enforced", previous)

    assert response.status_code == 200
    assert response.json()["formal_ready"] is False


def test_admin_cannot_create_unassigned_formal_accounts() -> None:
    response = client.post(
        "/api/admin/participants",
        headers=ADMIN_HEADERS,
        json={
            "count": 1,
            "cohort": "study_a",
            "group_name": "donor",
            "preview_mode": False,
            "data_origin": "formal",
            "protocol_mode": "strict",
            "run_slug": "forbidden-formal-shortcut",
        },
    )
    assert response.status_code == 405
    assert db.study_run("forbidden-formal-shortcut") is None


def test_admin_stage_templates_and_simple_account_initialization() -> None:
    unauthorized = client.get("/api/admin/stages")
    assert unauthorized.status_code == 401

    templates = client.get("/api/admin/stages", headers=ADMIN_HEADERS)
    assert templates.status_code == 200
    by_id = {row["id"]: row for row in templates.json()["stages"]}
    assert "personal_debug" not in by_id
    assert "author_walkthrough" not in by_id
    assert "dataset_pilot" not in by_id
    assert by_id["duration_prepilot"]["recommended_count"] == 4
    assert by_id["duration_prepilot"]["label"] == "试验时长选型试验"
    assert by_id["duration_prepilot"]["protocol_mode"] == "strict"
    assert by_id["study_a_formal"]["initializable"] is False

    run_slug = "test-admin-duration-prepilot"
    created = client.post(
        "/api/admin/stage-accounts",
        headers=ADMIN_HEADERS,
        json={
            "stage": "duration_prepilot",
            "count": 4,
            "run_slug": run_slug,
            "label": "four-person convergence check",
            "personal_debug": False,
        },
    )
    assert created.status_code == 405
    assert "public self-enrollment" in created.json()["detail"]
    return
    assert created.headers["cache-control"] == "no-store"
    payload = created.json()
    assert payload["codes_shown_once"] is True
    assert payload["run_slug"] == run_slug
    assert payload["assigned_count"] == 4
    assert payload["personal_debug"] is False
    assert payload["analysis_excluded"] is False
    assert payload["archive"] is None
    assert len(payload["accounts"]) == 4
    assert len({row["participant_id"] for row in payload["accounts"]}) == 4
    assert len({row["access_code"] for row in payload["accounts"]}) == 4
    for account in payload["accounts"]:
        participant = db.participant_by_code(account["access_code"])
        assert participant is not None
        assert participant["id"] == account["participant_id"]
        assert participant["run_slug"] == run_slug
        assert participant["cohort"] == "pilot"
        assert participant["group_name"] == "personalized"
        assert participant["data_origin"] == "pilot"
        assert participant["protocol_mode"] == "strict"
        assert participant["analysis_excluded"] == 0

    batches = [
        row
        for row in db.table_rows("account_batches")
        if row["id"] == payload["batch_id"]
    ]
    assert len(batches) == 1
    assert batches[0]["requested_count"] == 4
    assert batches[0]["is_pre_experiment"] == 0
    assert batches[0]["analysis_excluded"] == 0
    assert json.loads(batches[0]["participant_ids"]) == [
        row["participant_id"] for row in payload["accounts"]
    ]
    formal = client.post(
        "/api/admin/stage-accounts",
        headers=ADMIN_HEADERS,
        json={"stage": "study_a_formal", "count": 1},
    )
    assert formal.status_code == 409
    assert formal.json()["detail"]["code"] == "formal_stage_managed"

    ordinary = client.post(
        "/api/admin/stage-accounts",
        headers=ADMIN_HEADERS,
        json={
            "stage": "engineering_smoke",
            "count": 2,
            "run_slug": "test-admin-non-pre-experiment",
            "personal_debug": False,
        },
    )
    assert ordinary.status_code == 200
    assert ordinary.json()["assigned_count"] == 2
    assert ordinary.json()["personal_debug"] is False
    assert ordinary.json()["archive"] is None
    refused_archive = client.post(
        f"/api/admin/account-batches/{ordinary.json()['batch_id']}/archive",
        headers=ADMIN_HEADERS,
    )
    assert refused_archive.status_code == 409


@pytest.mark.skip(reason="superseded by the fully separate /tap-test data root")
def test_personal_debug_is_permanently_isolated() -> None:
    created = client.post(
        "/api/admin/stage-accounts",
        headers=ADMIN_HEADERS,
        json={
            "stage": "duration_prepilot",
            "count": 1,
            "run_slug": "test-personal-debug-isolation",
            "label": "author checks duration-selection experiment",
            "personal_debug": True,
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["personal_debug"] is True
    assert payload["analysis_excluded"] is True
    assert payload["storage_namespace"] == "personal_debug"
    account = payload["accounts"][0]
    assert account["analysis_excluded"] is True

    participant = db.participant_by_code(account["access_code"])
    assert participant is not None
    assert participant["analysis_excluded"] == 1
    assert participant["group_name"] == "personalized"
    batch = db.account_batch(payload["batch_id"])
    assert batch is not None
    assert batch["analysis_excluded"] == 1
    assert batch["storage_namespace"] == "personal_debug"
    archive_path = Path(payload["archive"]["archive_path"])
    assert archive_path.parts[-2:] == ("personal_debug", payload["batch_id"])
    assert (archive_path / "models").is_dir()
    manifest = json.loads(
        (archive_path / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["archive_kind"] == "personal_debug_isolated"
    assert manifest["analysis_excluded"] is True
    assert manifest["forbidden_uses"] == [
        "formal_analysis",
        "shared_prior",
        "dataset_release",
    ]

    timing = TimingService.__new__(TimingService)
    timing.settings = settings
    timing.db = db
    model_dir = timing._personal_model_dir(participant)
    assert archive_path in model_dir.parents
    assert model_dir.parts[-2:] == ("models", participant["id"])

    with pytest.raises(
        ValueError, match="analysis-excluded participants cannot contribute"
    ):
        db.create_shared_prior(
            {
                "run_id": participant["run_id"],
                "donor_count": 1,
                "opportunity_budget": 1,
                "algorithm": "must-not-run",
                "donor_ids": [participant["id"]],
                "encoder_path": "unused",
                "main_path": "unused",
                "encoder_sha256": "unused",
                "main_sha256": "unused",
            }
        )

    too_many = client.post(
        "/api/admin/stage-accounts",
        headers=ADMIN_HEADERS,
        json={
            "stage": "duration_prepilot",
            "count": 2,
            "personal_debug": True,
        },
    )
    assert too_many.status_code == 400


def test_admin_data_overview_and_participant_records_hide_content() -> None:
    run_slug = "test-admin-data-overview"
    code = create_participant(run_slug=run_slug)
    participant = db.participant_by_code(code)
    assert participant is not None
    participant_id = participant["id"]
    db.record_account_batch(
        run_id=participant["run_id"],
        stage="legacy_test_fixture",
        label="overview privacy test",
        cohort=participant["cohort"],
        group_name=participant["group_name"],
        data_origin=participant["data_origin"],
        protocol_mode=participant["protocol_mode"],
        participant_ids=[participant_id],
        is_pre_experiment=False,
    )
    db.append_event(
        participant_id,
        None,
        "admin.privacy_probe",
        None,
        None,
        {
            "transcript": "this raw transcript must not appear in the dashboard",
            "action": "inspect",
            "analysis_eligible": False,
        },
    )

    unauthorized = client.get(
        f"/api/admin/data-overview?run_slug={run_slug}"
    )
    assert unauthorized.status_code == 401
    overview_response = client.get(
        f"/api/admin/data-overview?run_slug={run_slug}",
        headers=ADMIN_HEADERS,
    )
    assert overview_response.status_code == 200
    assert overview_response.headers["cache-control"] == "no-store"
    overview = overview_response.json()
    assert overview["counts"]["runs"] == 1
    assert overview["counts"]["participants"] == 1
    assert overview["counts"]["account_batches"] == 1
    assert any(row["label"] == "实验数据库" for row in overview["paths"])
    assert any(
        row["label"] == "历史内部归档根目录"
        for row in overview["paths"]
    )
    assert any(
        row["label"] == "个人调试隔离根目录"
        for row in overview["paths"]
    )
    assert overview["account_batches"][0]["personal_debug"] is False
    assert overview["account_batches"][0]["archive_exists"] is False
    privacy_event = next(
        row
        for row in overview["recent_events"]
        if row["event_type"] == "admin.privacy_probe"
    )
    assert privacy_event["contains_text"] is True
    serialized_overview = json.dumps(overview, ensure_ascii=False)
    assert "this raw transcript must not appear" not in serialized_overview

    detail_response = client.get(
        f"/api/admin/participants/{participant_id}/records",
        headers=ADMIN_HEADERS,
    )
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["participant"]["id"] == participant_id
    assert any(
        row["event_type"] == "admin.privacy_probe"
        and row["contains_text"] is True
        for row in detail["recent_events"]
    )
    serialized_detail = json.dumps(detail, ensure_ascii=False)
    assert "this raw transcript must not appear" not in serialized_detail


@pytest.mark.skip(reason="superseded by the fully separate /tap-test data root")
def test_personal_debug_audio_and_records_share_one_isolated_archive() -> None:
    run_slug = "test-personal-debug-unified-archive"
    created = client.post(
        "/api/admin/stage-accounts",
        headers=ADMIN_HEADERS,
        json={
            "stage": "duration_prepilot",
            "count": 1,
            "run_slug": run_slug,
            "label": "unified archive test",
            "personal_debug": True,
        },
    )
    assert created.status_code == 200
    payload = created.json()
    code = payload["accounts"][0]["access_code"]
    participant_id = payload["accounts"][0]["participant_id"]
    archive_path = Path(payload["archive"]["archive_path"])

    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    bootstrap = client.get("/api/bootstrap", headers=headers).json()
    consent = bootstrap["consent"]
    saved_consent = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": True,
            "future_sharing": False,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert saved_consent.status_code == 200
    save_compensation_identity(headers)
    session_response = client.post(
        "/api/sessions/start", headers=headers, json={"day": 1}
    )
    assert session_response.status_code == 200
    session = session_response.json()
    audio_response = client.post(
        f"/api/audio/{session['id']}/study_stream",
        headers={**headers, "Content-Type": "audio/webm"},
        content=b"authorized-pre-experiment-audio",
    )
    assert audio_response.status_code == 200
    audio_row = next(
        row
        for row in db.table_rows("audio_files")
        if row["id"] == audio_response.json()["audio_id"]
    )
    relative_audio = Path(audio_row["relative_path"])
    assert relative_audio.parts[:2] == (
        "personal_debug",
        payload["batch_id"],
    )
    absolute_audio = settings.data_dir / relative_audio
    assert absolute_audio.is_file()
    assert archive_path in absolute_audio.parents

    db.append_event(
        participant_id,
        session["id"],
        "archive.test",
        None,
        None,
        {"transcript": "restricted internal archive content"},
    )
    unauthorized = client.post(
        f"/api/admin/account-batches/{payload['batch_id']}/archive"
    )
    assert unauthorized.status_code == 401
    archived = client.post(
        f"/api/admin/account-batches/{payload['batch_id']}/archive",
        headers=ADMIN_HEADERS,
    )
    assert archived.status_code == 200
    archive_result = archived.json()
    assert archive_result["assigned_count"] == 1
    assert archive_result["record_counts"]["events"] == 1
    assert archive_result["audio_file_count"] == 1
    records_dir = archive_path / "records"
    assert (records_dir / "events.csv").is_file()
    assert "restricted internal archive content" in (
        records_dir / "events.csv"
    ).read_text(encoding="utf-8-sig")
    assert "code_hash" not in (
        records_dir / "participants.csv"
    ).read_text(encoding="utf-8-sig").splitlines()[0]
    consent_status = (
        records_dir / "consent_status.csv"
    ).read_text(encoding="utf-8-sig")
    assert "consent_id" in consent_status.splitlines()[0]
    assert "future_sharing" in consent_status.splitlines()[0]
    assert "public_audio" in consent_status.splitlines()[0]
    assert "ai_training" in consent_status.splitlines()[0]
    assert "commercial_research" in consent_status.splitlines()[0]
    assert "server_archive_saved" in consent_status.splitlines()[0]
    consent_archive_dir = (
        archive_path / "consents" / participant_id
    )
    assert any(consent_archive_dir.rglob("consent-record.json"))
    manifest = json.loads(
        (archive_path / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["access_codes_stored"] is False
    assert manifest["audio_file_count"] == 1


def test_study_a_preview_uses_formal_flow_but_is_analysis_ineligible() -> None:
    code = create_participant("study_a", "donor", preview_mode=True)
    login_result = login(code)
    assert "cohort" not in login_result
    assert "group_name" not in login_result
    assert "run_slug" not in login_result
    assert login_result["preview_mode"] is True
    assert login_result["pilot_only"] is True
    assert login_result["formal_ready"] is False

    headers, session = consent_and_start(code)
    assert session["day"] == 1
    participant = db.participant_by_code(code)
    assert participant["preview_mode"] == 1
    assert db.latest_consent(participant["id"])["participation"] == 1


def test_matched_manifest_is_required_only_for_study_b() -> None:
    study_a_checks, _ = _participant_readiness({"id": "study-a", "cohort": "study_a"})
    study_b_checks, _ = _participant_readiness({"id": "study-b", "cohort": "study_b"})
    assert not any(name.startswith("matched_day") for name in study_a_checks)
    assert any(name.startswith("matched_day") for name in study_b_checks)


def test_pipeline_managed_simulation_is_locked_until_prior_is_ready() -> None:
    code = create_participant(
        cohort="study_b",
        group="personalized",
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug="test-simulation-pipeline-lock",
    )
    participant = db.participant_by_code(code)
    db.configure_participant(
        participant["id"],
        assignment_id="BP01",
        neutral_id="SIM-LOCK-01",
        enrollment_wave=6,
        release_state="released",
    )
    response = client.post("/api/login", json={"code": code, "role": "study"})
    assert response.status_code == 423
    assert response.json()["detail"]["code"] == "study_pipeline_locked"
    assert "prior_assigned" in response.json()["detail"]["missing"]


def test_environment_flag_cannot_approve_a_draft_consent_document() -> None:
    draft_settings = replace(settings, approved_consent_version=protocol.consent_version)
    checks = draft_settings.formal_readiness(protocol.consent_version, False)
    assert checks["consent_approved"] is False


def test_consent_discloses_dataset_scope_and_named_contact() -> None:
    text = json.dumps(protocol.consent, ensure_ascii=False)
    assert "筛查后的转写" in text
    assert "去标识化衍生数据" in text
    assert "不会在本同意下以可识别形式公开发布" in text
    assert "研究负责人：研究团队" in text
    assert "study@example.invalid" in text
    contact = next(
        section for section in protocol.consent["sections"] if section["heading"] == "联系信息"
    )
    assert contact["paragraphs"] == [
        "研究负责人：研究团队（Research team）",
        "机构：研究机构／the research institution",
        "微信：study-contact（微信名：Research team）",
        "邮箱：study@example.invalid",
    ]


def test_stale_consent_version_is_rejected_without_saving() -> None:
    code = create_participant()
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    consent = client.get("/api/bootstrap", headers=headers).json()["consent"]
    response = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": "outdated-consent-version",
            "document_hash": consent["sha256"],
            "participation": True,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert response.status_code == 409
    assert "知情同意书已更新" in response.json()["detail"]
    participant = db.participant_by_code(code)
    assert db.latest_consent(participant["id"]) is None


def test_signed_consent_copy_contains_snapshot_choices_and_audit_fields() -> None:
    code = create_participant()
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    before = client.get("/api/consent-copy", headers=headers)
    assert before.status_code == 404
    consent = client.get("/api/bootstrap", headers=headers).json()["consent"]
    signed = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "public_derived": True,
            "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
        },
    )
    assert signed.status_code == 200
    stored = db.latest_consent(db.participant_by_code(code)["id"])
    assert json.loads(stored["document_json"])["version"] == consent["version"]

    response = client.get("/api/consent-copy", headers=headers)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "attachment;" in response.headers["content-disposition"]
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["content-disposition"].endswith('.html"')
    assert "参与者留存副本" in response.text
    assert "文件 SHA-256" in response.text
    assert "证据包 SHA-256" in response.text
    assert "Ed25519 公钥指纹" in response.text
    assert "同意证据链序号" in response.text
    assert consent["sha256"] in response.text
    assert "研究负责人：研究团队" in response.text
    assert "[已同意]" in response.text
    assert "我单独同意公开发布" in response.text
    assert "[未同意]" in response.text
    assert "我单独同意研究团队保存原始语音" in response.text
    assert "研究团队归档副本与参与者留存副本对应同一记录 ID" in response.text

    participant_id = db.participant_by_code(code)["id"]
    research_copy = client.get(
        f"/api/admin/participants/{participant_id}/consent-copy",
        headers=ADMIN_HEADERS,
    )
    assert research_copy.status_code == 200
    assert "研究团队归档副本" in research_copy.text
    assert consent["sha256"] in research_copy.text
    assert (
        research_copy.headers["content-disposition"]
        .split("filename=", 1)[1]
        .startswith('"tap-to-adapt-consent-research-archive-')
    )


def test_consent_evidence_is_publicly_verifiable_and_chained() -> None:
    key = client.get("/api/consent-evidence-key")
    assert key.status_code == 200
    key_payload = key.json()
    assert key_payload["instance_id"] == "test"
    assert key_payload["algorithm"] == "Ed25519"
    assert key_payload["identity_certification"] is False

    records = []
    for suffix in ("a", "b"):
        code = create_participant(run_slug=f"signed-consent-chain-{suffix}")
        login_result = login(code)
        headers = {"Authorization": f"Bearer {login_result['token']}"}
        consent = client.get("/api/bootstrap", headers=headers).json()["consent"]
        signed = client.post(
            "/api/consent",
            headers=headers,
            json={
                "version": consent["version"],
                "document_hash": consent["sha256"],
                "participation": True,
                "typed_name": "Review Participant",
            "affirmation": consent["required_affirmation"],
                "client_nonce": ("a1" if suffix == "a" else "b2") * 16,
            },
        )
        assert signed.status_code == 200
        evidence = client.get("/api/consent-evidence", headers=headers)
        assert evidence.status_code == 200
        assert evidence.headers["cache-control"] == "no-store"
        bundle = evidence.json()
        verification = verify_evidence_bundle(
            bundle,
            expected_key_id=key_payload["key_id"],
            expected_instance_id="test",
        )
        assert verification["valid"] is True
        assert bundle["payload"]["document"]["sha256"] == consent["sha256"]
        assert bundle["payload"]["received"]["client_nonce"] in {
            "a1" * 16,
            "b2" * 16,
        }
        records.append((signed.json(), bundle))

    first, second = records
    assert second[0]["ledger_sequence"] == first[0]["ledger_sequence"] + 1
    assert (
        second[1]["payload"]["previous_record_sha256"]
        == first[0]["evidence_sha256"]
    )

    tampered = json.loads(json.dumps(second[1]))
    tampered["payload"]["affirmation"] = "tampered"
    with pytest.raises(ValueError, match="payload hash"):
        verify_evidence_bundle(tampered)


def test_every_consent_choice_is_archived_on_server_before_session_start() -> None:
    code = create_participant(run_slug="server-consent-archive")
    login_result = login(code)
    headers = {"Authorization": f"Bearer {login_result['token']}"}
    consent_document = client.get("/api/bootstrap", headers=headers).json()["consent"]
    selected = {
        "participation": True,
        "raw_audio": True,
        "internal_reuse": True,
        "future_sharing": False,
        "controlled_sharing": True,
        "public_derived": False,
        "public_audio": True,
        "ai_training": False,
        "commercial_research": True,
    }
    signed = client.post(
        "/api/consent",
        headers=headers,
        json={
            "version": consent_document["version"],
            "document_hash": consent_document["sha256"],
            **selected,
            "typed_name": "Review Participant",
            "affirmation": consent_document["required_affirmation"],
        },
    )
    assert signed.status_code == 200
    signed_payload = signed.json()
    assert signed_payload["server_archive_saved"] is True
    assert len(signed_payload["server_archive_record_sha256"]) == 64

    participant = db.participant_by_code(code)
    stored = db.consent(signed_payload["consent_id"])
    assert stored is not None
    archive_dir = (
        settings.data_dir
        / "consent_archives"
        / participant["run_id"]
        / participant["id"]
        / stored["id"]
    )
    manifest_path = archive_dir / "MANIFEST.json"
    record_path = archive_dir / "consent-record.json"
    document_path = archive_dir / "signed-document.json"
    research_copy_path = archive_dir / "research-team-copy.html"
    assert all(
        path.is_file()
        for path in (
            manifest_path,
            record_path,
            document_path,
            research_copy_path,
        )
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert {
        field: bool(value["selected"])
        for field, value in record["choices"].items()
    } == selected
    assert record["storage"] == "linux_server"
    assert record["consent_id"] == stored["id"]
    assert record["document_sha256"] == consent_document["sha256"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_sha256"] == file_sha256(record_path)
    assert {row["path"] for row in manifest["files"]} == {
        "consent-record.json",
        "consent-evidence.json",
        "signed-document.json",
        "research-team-copy.html",
    }

    bootstrap = client.get("/api/bootstrap", headers=headers)
    assert bootstrap.status_code == 200
    assert bootstrap.json()["consent_archive_saved"] is True
    detail = client.get(
        f"/api/admin/participants/{participant['id']}/records",
        headers=ADMIN_HEADERS,
    )
    assert detail.status_code == 200
    detail_consent = detail.json()["consent"]
    assert detail_consent["server_archive_saved"] is True
    assert detail_consent["server_archive_path"] == str(archive_dir.resolve())
    for field, expected in selected.items():
        if field == "participation":
            assert detail_consent["signed"] is expected
        else:
            assert detail_consent[field] is expected

    research_copy_path.chmod(0o600)
    research_copy_path.write_text("tampered", encoding="utf-8")
    blocked = client.post("/api/sessions/start", headers=headers, json={"day": 1})
    assert blocked.status_code == 503
    assert "知情同意归档" in blocked.json()["detail"]


def test_legacy_consent_choices_are_archived_without_inventing_missing_document() -> None:
    legacy_id = f"legacy-{time.time_ns()}"
    participant_id = f"missing-participant-{time.time_ns()}"
    selected = {
        "participation": True,
        "raw_audio": False,
        "internal_reuse": True,
        "future_sharing": False,
        "controlled_sharing": True,
        "public_derived": False,
        "public_audio": False,
        "ai_training": True,
        "commercial_research": False,
    }
    consent = {
        "id": legacy_id,
        "run_id": None,
        "participant_id": participant_id,
        "version": "legacy-consent-without-document-snapshot",
        "document_hash": "a" * 64,
        "document_json": "",
        **selected,
        "affirmation": "legacy affirmation retained from database",
        "signed_ms": int(time.time() * 1000),
        "ip_hash": "legacy-ip-hash",
        "user_agent": "legacy-agent",
    }
    participant = {
        "id": participant_id,
        "code_hint": None,
        "run_id": None,
        "participant_record_available": False,
    }
    archived = _ensure_server_consent_archive(consent, participant)
    assert archived["verified"] is True
    archive_dir = Path(archived["archive_path"])
    record = json.loads(
        (archive_dir / "consent-record.json").read_text(encoding="utf-8")
    )
    assert record["document_snapshot_available"] is False
    assert record["participant_record_available"] is False
    assert {
        field: bool(value["selected"])
        for field, value in record["choices"].items()
    } == selected
    assert not (archive_dir / "signed-document.json").exists()
    assert not (archive_dir / "research-team-copy.html").exists()
    manifest = json.loads(
        (archive_dir / "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["document_snapshot_available"] is False
    assert {row["path"] for row in manifest["files"]} == {
        "consent-record.json"
    }


def test_setup_workspace_keeps_navigation_and_content_as_two_grid_children() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "static" / "index.html"
    ).read_text(encoding="utf-8")
    assert '<div class="workspace">\n          <nav class="step-nav"' in source
    assert '<div class="consent-copy-strip"' in source


def test_pair_hub_prunes_a_socket_that_closes_during_relay() -> None:
    class ClosedSocket:
        async def send_json(self, _message: dict) -> None:
            raise RuntimeError("Unexpected ASGI message 'websocket.send'")

    hub = PairHub()
    socket = ClosedSocket()
    hub.connections["participant"]["companion"] = PairConnection(
        websocket=socket,  # type: ignore[arg-type]
        session_id="session",
        day=1,
        last_seen_ms=0,
    )
    asyncio.run(hub.relay("participant", "study", {"type": "heartbeat"}))
    assert "participant" not in hub.connections


def test_timing_hub_serializes_replacement_and_rejects_old_generation() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.closed: tuple[int, str] | None = None

        async def close(self, *, code: int, reason: str) -> None:
            self.closed = (code, reason)

    async def exercise() -> None:
        hub = TimingHub()
        old_socket = FakeSocket()
        new_socket = FakeSocket()
        old_generation = await hub.connect(
            "timing-session", old_socket  # type: ignore[arg-type]
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def long_old_operation() -> str:
            entered.set()
            await release.wait()
            return "finished"

        old_operation = asyncio.create_task(
            hub.execute(
                "timing-session",
                old_socket,  # type: ignore[arg-type]
                old_generation,
                long_old_operation,
            )
        )
        await entered.wait()
        replacement = asyncio.create_task(
            hub.connect("timing-session", new_socket)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        assert not replacement.done()

        release.set()
        assert await old_operation == (True, "finished")
        new_generation = await replacement
        assert new_generation == old_generation + 1
        assert old_socket.closed is not None
        assert old_socket.closed[0] == 4001

        async with hub.operation(
            "timing-session",
            old_socket,  # type: ignore[arg-type]
            old_generation,
        ) as old_is_current:
            assert old_is_current is False
        await hub.disconnect(
            "timing-session",
            old_socket,  # type: ignore[arg-type]
            old_generation,
        )
        async with hub.operation(
            "timing-session",
            new_socket,  # type: ignore[arg-type]
            new_generation,
        ) as new_is_current:
            assert new_is_current is True

    asyncio.run(exercise())


def test_mock_timing_stream_activates_after_speech_then_silence() -> None:
    code = create_participant()
    participant = db.participant_by_code(code)
    runtime = timing_service.runtime(participant)
    runtime.set_phase("learning", "current")
    runtime.set_decision_context(
        conversation_target="experimental_agent",
        reply_needed_probability=1.0,
        source="unit_test",
        text_sha256=None,
    )
    speech = (b"\xff\x1f" * 3200)
    silence = (b"\x00\x00" * 3200)
    runtime.process_pcm(speech, 1000)
    result = runtime.process_pcm(silence, 2300)
    assert result["activate"] is True
    assert result["policy_effective"] == "pilot_vad_fallback"


def test_tap_after_complete_never_starts_or_trains_an_agent_turn() -> None:
    runtime = ParticipantRuntime(
        participant_id="runtime-complete-tap",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        model=None,
    )
    assert runtime.set_phase("complete", "current")["committed"] is True

    result = runtime.tap(1000, pointer_agent_state="waiting", page_visible=True)

    assert result["action"] == "after_complete_ignored"
    assert result["training_eligible"] is False
    assert result["training_applied"] is False
    assert result["label"] is None
    assert result["quarantine_reason"] == "interaction_complete"
    assert runtime.agent_state == "waiting"
    assert runtime.activation_id is None


def test_conditional_tap_is_atomic_across_waiting_and_activation_races() -> None:
    class FakeModel:
        def predict(self, _audio):
            return 1.0, 8.0

        def train(self, _audio, _label, weight=1.0):
            return float(weight)

        def save(self, target):
            return target / "encoder.pth", target / "main.pth"

    manual_runtime = ParticipantRuntime(
        participant_id="runtime-conditional-manual",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        model=FakeModel(),  # type: ignore[arg-type]
    )
    incomplete = manual_runtime.tap(
        500,
        intended_action="manual_activate",
    )
    assert incomplete["action"] == "conditional_tap_cancelled"
    assert incomplete["cancel_reason"] == "conditional_fields_required"
    assert manual_runtime.last_tap_ms == 0
    manual = manual_runtime.tap(
        1000,
        pointer_agent_state="waiting",
        page_visible=True,
        intended_action="manual_activate",
        expected_state="waiting",
        expected_activation_id="",
        expected_state_version=0,
    )
    assert manual["action"] == "manual_activate"
    assert manual["server_agent_state"] == "thinking"
    assert manual["server_state_version"] == 1
    assert manual["server_activation_id"]
    manual_interrupt = manual_runtime.tap(
        1100,
        pointer_agent_state="thinking",
        page_visible=True,
        intended_action="interrupt",
        expected_state="thinking",
        expected_activation_id=str(manual["server_activation_id"]),
        expected_state_version=int(manual["server_state_version"]),
    )
    assert manual_interrupt["action"] == "interrupt"
    assert manual_interrupt["action"] != "tap_burst_ignored"
    assert manual_interrupt["quarantine_reason"] == "manual_activate_interrupt_conflict"
    assert manual_interrupt["training_eligible"] is False
    assert manual_interrupt["server_agent_state"] == "waiting"

    unbound_runtime = ParticipantRuntime(
        participant_id="runtime-conditional-unbound",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        model=FakeModel(),  # type: ignore[arg-type]
    )
    unbound_runtime.set_state("thinking")
    unbound_interrupt = unbound_runtime.tap(
        1000,
        intended_action="interrupt",
        expected_state="thinking",
        expected_activation_id="",
        expected_state_version=1,
    )
    assert unbound_interrupt["action"] == "conditional_tap_cancelled"
    assert unbound_interrupt["cancel_reason"] == "interrupt_requires_activation"
    assert unbound_runtime.agent_state == "thinking"
    assert unbound_runtime.last_tap_ms == 0

    runtime = ParticipantRuntime(
        participant_id="runtime-conditional-race",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        model=FakeModel(),  # type: ignore[arg-type]
    )
    runtime.set_phase("learning", "current")
    runtime.process_pcm(b"\xff\x1f" * 3200, 1000)
    activated = runtime.process_pcm(b"\x00\x00" * 3200, 2600)
    assert activated["activate"] is True
    activation_id = str(activated["activation_id"])
    activation_version = int(activated["activation_state_version"])

    stale_manual = runtime.tap(
        2800,
        intended_action="manual_activate",
        expected_state="waiting",
        expected_activation_id="",
        expected_state_version=0,
    )
    assert stale_manual["action"] == "conditional_tap_cancelled"
    assert stale_manual["cancel_reason"] == "expected_state_mismatch"
    assert stale_manual["training_eligible"] is False
    assert runtime.agent_state == "thinking"
    assert runtime.activation_id == activation_id
    assert runtime.last_tap_ms == 0

    mismatched_interrupt = runtime.tap(
        2900,
        intended_action="interrupt",
        expected_state="thinking",
        expected_activation_id="different-activation",
        expected_state_version=activation_version,
    )
    assert mismatched_interrupt["action"] == "conditional_tap_cancelled"
    assert mismatched_interrupt["cancel_reason"] == "expected_activation_mismatch"
    assert runtime.agent_state == "thinking"
    assert runtime.activation_id == activation_id
    assert runtime.last_tap_ms == 0

    exact_interrupt = runtime.tap(
        3000,
        pointer_agent_state="thinking",
        page_visible=True,
        intended_action="interrupt",
        expected_state="thinking",
        expected_activation_id=activation_id,
        expected_state_version=activation_version,
    )
    assert exact_interrupt["action"] == "interrupt"
    assert exact_interrupt["server_agent_state"] == "waiting"
    assert exact_interrupt["server_state_version"] == activation_version + 1
    assert exact_interrupt["server_activation_id"] is None


def test_activation_provenance_controls_labels_and_phase_transition_is_atomic() -> None:
    class FakeModel:
        def predict(self, _audio):
            return 1.0, 8.0

        def train(self, _audio, _label, weight=1.0):
            return float(weight)

        def save(self, target):
            return target / "encoder.pth", target / "main.pth"

    runtime = ParticipantRuntime(
        participant_id="runtime-provenance",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        budget_kind="eligible_feedback",
        budget_semantics="per_donor_cap",
        q_levels=(2, 4, 8),
        model=FakeModel(),  # type: ignore[arg-type]
    )
    assert runtime.set_phase("learning", "current")["committed"] is True
    speech = b"\xff\x1f" * 3200
    silence = b"\x00\x00" * 3200
    runtime.process_pcm(speech, 1000)
    activated = runtime.process_pcm(silence, 2600)
    assert activated["activate"] is True
    assert activated["activation_phase"] == "learning"
    assert activated["activation_policy"] == "current"
    assert activated["activation_id"]

    blocked = runtime.set_phase("micro_probe", "vad")
    assert blocked["committed"] is False
    runtime.process_pcm(speech, 2800)
    result = runtime.tap(3000)
    assert result["training_eligible"] is True
    assert result["phase"] == "learning"
    assert result["tap_phase"] == "learning"
    assert result["activation_id"] == activated["activation_id"]

    runtime.tap(5000)
    drained = runtime.set_phase("micro_probe", "vad", drain_active=True)
    assert drained == {"committed": True, "drained_state": "thinking"}
    assert runtime.agent_state == "waiting"
    assert runtime.activation_id is None
    assert runtime.phase == "micro_probe"


def test_timing_configuration_comes_from_the_immutable_run_snapshot(
    tmp_path: Path,
) -> None:
    code = create_participant(run_slug="immutable-timing-protocol")
    participant = db.participant_by_code(code)
    assert participant is not None
    expected = json.loads(participant["run_protocol_json"])["initialization"]

    decoy_content = tmp_path / "content"
    decoy_content.mkdir()
    decoy = json.loads(participant["run_protocol_json"])
    decoy["initialization"]["temperature"] = 99.0
    (decoy_content / "study_protocol.json").write_text(
        json.dumps(decoy), encoding="utf-8"
    )
    service = TimingService(replace(settings, content_dir=decoy_content), db)

    configuration = service._runtime_configuration(participant)

    assert configuration["temperature"] == expected["temperature"]
    assert configuration["temperature"] != 99.0
    assert configuration["protocol_sha256"] == participant["run_protocol_sha256"]


def test_personal_intercept_is_bounded_and_network_updates_unfreeze_gradually() -> None:
    class FakeModel:
        last_update_applied = True

        def __init__(self):
            self.weights = []

        def predict(self, _audio):
            return 0.5, 0.0

        def can_train(self, _audio):
            return True

        def train(self, _audio, _label, weight=1.0):
            self.weights.append(float(weight))
            return float(weight)

        def save(self, target):
            return target / "encoder.pth", target / "main.pth"

    current = FakeModel()
    shadow = FakeModel()
    persisted = []

    def record(expected: int, label: int, bias: float) -> dict:
        assert expected == len(persisted)
        persisted.append((label, bias))
        return {
            "personal_logit_bias": bias,
            "update_count": len(persisted),
            "positive_updates": sum(item[0] == 1 for item in persisted),
            "negative_updates": sum(item[0] == 0 for item in persisted),
        }

    runtime = ParticipantRuntime(
        participant_id="runtime-personal-calibration",
        group_name="seed_donor",
        cohort="study_a",
        settings=settings,
        budget_kind="eligible_feedback",
        model=current,  # type: ignore[arg-type]
        donor_shadow_model=shadow,  # type: ignore[arg-type]
        network_update_schedule=((2, 0.15), (4, 0.35), (6, 0.65)),
        personal_calibration_enabled=True,
        personal_calibration_learning_rate=10.0,
        personal_calibration_l2=0.0,
        personal_logit_bias_min=-0.5,
        personal_logit_bias_max=0.5,
        calibration_recorder=record,
    )
    audio = np.ones(16_000, dtype=np.float32)

    positive: dict[str, Any] = {"training_eligible": True}
    runtime._apply_training_update(audio, 1, positive)
    negative: dict[str, Any] = {"training_eligible": True}
    runtime._apply_training_update(audio, 0, negative)
    third: dict[str, Any] = {"training_eligible": True}
    runtime._apply_training_update(audio, 1, third)

    assert positive["personal_logit_bias_after"] == 0.5
    assert negative["personal_logit_bias_after"] == -0.5
    assert positive["network_update_scale"] == 0.15
    assert negative["network_update_scale"] == 0.15
    assert third["network_update_scale"] == 0.35
    assert current.weights[0] == pytest.approx(0.15)
    assert shadow.weights[0] == pytest.approx(1.0)
    assert runtime.personal_calibration_update_count == 3


def test_first_two_feedbacks_calibrate_intercept_without_updating_current_network() -> None:
    class FakeModel:
        last_update_applied = True

        def __init__(self):
            self.weights = []
            self.save_count = 0

        def predict(self, _audio):
            return 0.5, 0.0

        def can_train(self, _audio):
            return True

        def train(self, _audio, _label, weight=1.0):
            self.weights.append(float(weight))
            return float(weight)

        def save(self, target):
            self.save_count += 1
            return target / "encoder.pth", target / "main.pth"

    current = FakeModel()
    shadow = FakeModel()
    runtime = ParticipantRuntime(
        participant_id="runtime-intercept-only-warmup",
        group_name="seed_donor",
        cohort="study_a",
        settings=settings,
        budget_kind="eligible_feedback",
        model=current,  # type: ignore[arg-type]
        donor_shadow_model=shadow,  # type: ignore[arg-type]
        network_update_schedule=((2, 0.0), (4, 0.05), (6, 0.2), (8, 0.5)),
        personal_calibration_enabled=True,
        personal_calibration_learning_rate=0.25,
        personal_calibration_l2=0.05,
        personal_logit_bias_min=-1.25,
        personal_logit_bias_max=1.25,
    )
    audio = np.ones(16_000, dtype=np.float32)

    results = []
    for _ in range(3):
        result: dict[str, Any] = {"training_eligible": True}
        runtime._apply_training_update(audio, 1, result)
        results.append(result)

    first, second, third = results
    assert first["network_update_scale"] == 0.0
    assert second["network_update_scale"] == 0.0
    assert first["current_network_update_applied"] is False
    assert second["current_network_update_applied"] is False
    assert first["requested_training_steps"] == 0
    assert second["requested_training_steps"] == 0
    assert first["training_applied"] is True
    assert second["training_applied"] is True
    assert first["personal_calibration_applied"] is True
    assert second["personal_calibration_applied"] is True
    assert current.weights == pytest.approx([third["class_weight"]])
    assert third["network_update_scale"] == 0.05
    assert third["current_network_update_applied"] is True
    assert len(shadow.weights) == 3
    assert shadow.weights[0] == pytest.approx(first["base_class_weight"])
    assert shadow.weights[1] == pytest.approx(second["base_class_weight"])
    assert shadow.weights[2] == pytest.approx(third["base_class_weight"])
    assert runtime.feedback_count == 3
    assert runtime.personal_calibration_update_count == 3


def test_dtcn_network_scale_controls_learning_rate_and_restores_it() -> None:
    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.001}]
            self.token = object()
            self.state = {self.token: {"step": 0}}

    class FakeTrainer:
        def __init__(self):
            self.optimizer = FakeOptimizer()
            self.lr_during = None

        def add_training_sample(self, *_args, **_kwargs):
            return None

        def train_step(self, **_kwargs):
            self.lr_during = self.optimizer.param_groups[0]["lr"]
            self.optimizer.state[self.optimizer.token]["step"] += 1
            return 0.25

    model = object.__new__(DTCNModel)
    model.lock = threading.Lock()
    model.trainer = FakeTrainer()
    model.config = {"num_of_step": 6}
    model.can_train = lambda _audio: True  # type: ignore[method-assign]
    model._snapshot_model_state = lambda: "before"  # type: ignore[method-assign]
    interpolations = []
    model._interpolate_model_state = (  # type: ignore[method-assign]
        lambda snapshot, scale: interpolations.append((snapshot, scale))
    )

    loss = model.train(
        np.ones(16_000, dtype=np.float32),
        label=1,
        weight=0.5,
        update_scale=0.05,
    )

    assert loss == pytest.approx(0.25)
    assert model.last_requested_training_steps == 1
    assert model.last_training_steps == 1
    assert model.last_update_applied is True
    assert model.trainer.lr_during == pytest.approx(0.00005)
    assert model.last_optimizer_learning_rates_before == pytest.approx([0.001])
    assert model.last_optimizer_learning_rates_during == pytest.approx([0.00005])
    assert model.last_optimizer_learning_rates_after == pytest.approx([0.001])
    assert model.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert model.last_parameter_state_interpolated is True
    assert model.last_parameter_interpolation_scale == pytest.approx(0.05)
    assert interpolations == [("before", 0.05)]


def test_personal_intercept_applies_only_to_the_current_policy() -> None:
    runtime = ParticipantRuntime(
        participant_id="runtime-policy-calibration",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        temperature=1.0,
        logit_bias=0.0,
        personal_calibration_enabled=True,
        personal_logit_bias=1.0,
    )

    current = runtime._calibrate(
        0.5,
        0.0,
        policy_effective="current",
        include_personal=True,
    )
    comparison = runtime._calibrate(
        0.5,
        0.0,
        policy_effective="base",
        include_personal=False,
    )

    assert current.probability > comparison.probability
    assert current.personal_logit_bias == 1.0
    assert comparison.personal_logit_bias == 0.0


def test_personal_calibration_state_persists_and_rejects_stale_updates() -> None:
    code = create_participant(run_slug="calibration-persistence")
    participant = db.participant_by_code(code)
    assert participant is not None
    version = "calibration-persistence-v1"
    protocol_sha256 = participant["run_protocol_sha256"]

    initial = db.calibration_state(
        participant["id"],
        calibration_version=version,
        protocol_sha256=protocol_sha256,
        initial_logit_bias=0.2,
    )
    updated = db.update_calibration_state(
        participant["id"],
        calibration_version=version,
        protocol_sha256=protocol_sha256,
        expected_update_count=0,
        personal_logit_bias=0.35,
        label=1,
    )
    reloaded = db.calibration_state(
        participant["id"],
        calibration_version=version,
        protocol_sha256=protocol_sha256,
        initial_logit_bias=-0.9,
    )

    assert initial["update_count"] == 0
    assert updated["positive_updates"] == 1
    assert reloaded["personal_logit_bias"] == pytest.approx(0.35)
    with pytest.raises(RuntimeError, match="Concurrent"):
        db.update_calibration_state(
            participant["id"],
            calibration_version=version,
            protocol_sha256=protocol_sha256,
            expected_update_count=0,
            personal_logit_bias=0.4,
            label=1,
        )


def test_activation_observable_timestamp_excludes_precommit_compute_time() -> None:
    class FakeModel:
        def predict(self, _audio):
            return 1.0, 8.0

        def train(self, _audio, _label, weight=1.0):
            return float(weight)

        def save(self, target):
            return target / "encoder.pth", target / "main.pth"

    runtime = ParticipantRuntime(
        participant_id="runtime-observable-clock",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        budget_kind="eligible_feedback",
        budget_semantics="per_donor_cap",
        model=FakeModel(),  # type: ignore[arg-type]
    )
    runtime.set_phase("learning", "current")
    speech = b"\xff\x1f" * 3200
    silence = b"\x00\x00" * 3200
    runtime.process_pcm(speech, 1000)
    activated = runtime.process_pcm(silence, 2600)
    activation_id = str(activated["activation_id"])

    assert runtime.mark_activation_observable("wrong-id", 5000) is False
    assert runtime.mark_activation_observable(activation_id, 5000) is True
    runtime.process_pcm(speech, 5100)
    result = runtime.tap(5250)

    assert result["training_eligible"] is True
    assert result["label"] == 0
    assert result["activation_to_tap_ms"] == 250


@pytest.mark.parametrize("agent_state", ["thinking", "speaking"])
def test_visible_model_interrupt_is_negative_without_voice_or_latency_cutoff(
    agent_state: str,
) -> None:
    trained: list[tuple[int, float]] = []

    class FakeModel:
        def predict(self, _audio):
            return 1.0, 8.0

        def train(self, _audio, label, weight=1.0):
            trained.append((int(label), float(weight)))
            return float(weight)

        def save(self, target):
            return target / "encoder.pth", target / "main.pth"

    runtime = ParticipantRuntime(
        participant_id=f"runtime-visible-{agent_state}",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        budget_kind="eligible_feedback",
        model=FakeModel(),  # type: ignore[arg-type]
    )
    runtime.set_phase("learning", "current")
    runtime.process_pcm(b"\xff\x1f" * 3200, 1000)
    activated = runtime.process_pcm(b"\x00\x00" * 3200, 2600)
    runtime.mark_activation_observable(str(activated["activation_id"]), 3000)
    runtime.set_state(agent_state)

    result = runtime.tap(
        6500,
        pointer_agent_state=agent_state,
        page_visible=True,
    )

    assert result["activation_to_tap_ms"] == 3500
    assert result["user_speech_resumed"] is False
    assert result["label"] == 0
    assert result["training_eligible"] is True
    assert result["training_applied"] is True
    assert result["quarantined"] is False
    assert result["label_confidence"] == "high"
    assert result["timing_attribution_confidence"] == "no_pre_tap_speech_observed"
    assert result["interrupt_agent_state"] == agent_state
    assert trained and trained[-1][0] == 0


def test_interruption_confidence_is_telemetry_based_but_does_not_gate_training() -> None:
    class FakeModel:
        def predict(self, _audio):
            return 1.0, 8.0

        def train(self, _audio, _label, weight=1.0):
            return float(weight)

        def save(self, target):
            return target / "encoder.pth", target / "main.pth"

    runtime = ParticipantRuntime(
        participant_id="runtime-unverified-visible-state",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        model=FakeModel(),  # type: ignore[arg-type]
    )
    runtime.set_phase("learning", "current")
    runtime.process_pcm(b"\xff\x1f" * 3200, 1000)
    activated = runtime.process_pcm(b"\x00\x00" * 3200, 2600)
    runtime.mark_activation_observable(str(activated["activation_id"]), 3000)

    result = runtime.tap(5000, pointer_agent_state=None, page_visible=None)

    assert result["training_eligible"] is True
    assert result["label_confidence"] == "telemetry_unverified"
    assert result["label_confidence_basis"]["activation_id_bound"] is True


def test_analysis_counts_every_interrupt_in_primary_correction_burden() -> None:
    metric = _new_metric(participant_id="participant")
    metric.update(
        {
            "duration_seconds": 60.0,
            "predictions": 100,
            "manual_activations": 2,
            "interruptions": 3,
            "timing_interruptions": 1,
            "other_interruptions": 2,
        }
    )

    result = _finish_metric(metric)

    assert result["correction_burden_per_minute"] == 5.0
    assert result["correction_burden_per_100_predictions"] == 5.0
    assert _is_timing_interrupt(
        {"timing_attribution_confidence": "speech_overlap_observed"}
    )
    assert not _is_timing_interrupt(
        {"timing_attribution_confidence": "no_pre_tap_speech_observed"}
    )


def test_short_training_audio_cannot_create_an_unlearnable_activation() -> None:
    class RejectingModel:
        train_calls = 0

        def predict(self, _audio):
            return 1.0, 8.0

        def can_train(self, _audio):
            return False

        def train(self, _audio, _label, weight=1.0):
            self.train_calls += 1
            return float(weight)

    model = RejectingModel()
    runtime = ParticipantRuntime(
        participant_id="runtime-short-training-audio",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        budget_kind="eligible_feedback",
        model=model,  # type: ignore[arg-type]
    )
    runtime.set_phase("learning", "current")
    runtime.process_pcm(b"\xff\x1f" * 3200, 1000)
    activated = runtime.process_pcm(b"\x00\x00" * 3200, 2600)
    assert activated["activate"] is False
    assert activated["decision_eligible"] is False
    assert "training_audio_not_ready" in activated["decision_gate_reasons"]
    assert runtime.activation_id is None
    assert runtime.feedback_count == 0
    assert model.train_calls == 0


def test_failed_optimizer_step_is_not_counted_as_eligible_feedback() -> None:
    class NoUpdateModel:
        last_update_applied = False
        last_update_reason = "optimizer_update_not_applied"

        def predict(self, _audio):
            return 1.0, 8.0

        def can_train(self, _audio):
            return True

        def train(self, _audio, _label, weight=1.0):
            return 0.0

    runtime = ParticipantRuntime(
        participant_id="runtime-no-optimizer-update",
        group_name="personalized",
        cohort="study_b",
        settings=settings,
        budget_kind="eligible_feedback",
        model=NoUpdateModel(),  # type: ignore[arg-type]
    )
    runtime.set_phase("learning", "current")
    runtime.process_pcm(b"\xff\x1f" * 3200, 1000)
    activated = runtime.process_pcm(b"\x00\x00" * 3200, 2600)
    runtime.mark_activation_observable(str(activated["activation_id"]), 3000)
    runtime.process_pcm(b"\xff\x1f" * 3200, 3100)
    result = runtime.tap(3250)

    assert result["training_eligible"] is False
    assert result["training_applied"] is False
    assert result["quarantine_reason"] == "optimizer_update_not_applied"
    assert runtime.feedback_count == 0


def test_capped_snapshot_finalization_uses_realized_feedback_count() -> None:
    recorded = []

    class FakeModel:
        pass

    runtime = ParticipantRuntime(
        participant_id="runtime-caps",
        group_name="donor",
        cohort="study_a",
        settings=settings,
        day=3,
        budget_kind="eligible_feedback",
        budget_semantics="per_donor_cap",
        q_levels=(2, 4, 8),
        feedback_count=3,
        positive_feedbacks=2,
        negative_feedbacks=1,
        donor_shadow_model=FakeModel(),  # type: ignore[arg-type]
        snapshot_recorder=lambda source, budget, model, metadata: recorded.append(
            (source, budget, model, metadata)
        ),
    )
    runtime.finalize_capped_snapshots()
    assert [row[1] for row in recorded] == [4, 8]
    for _, cap, _, metadata in recorded:
        assert metadata["cap_budget"] == cap
        assert metadata["realized_feedbacks"] == 3
        assert metadata["cap_attained"] is False
        assert metadata["capped_final"] is True


def test_final_session_end_delegates_cap_semantics_to_participant_runtime() -> None:
    code = create_participant(
        cohort="study_a",
        group="donor",
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug="test-day-three-cap-finalization",
    )
    headers, final_session = consent_and_start(code, day=1)
    participant = db.participant_by_code(code)
    db.configure_participant(
        participant["id"],
        assignment_id="DAY3-CAP-01",
        neutral_id="DAY3-CAP-01",
        enrollment_wave=1,
        release_state="released",
    )
    participant = db.participant_by_code(code)
    companion = db.start_session(participant["id"], 1, "companion")
    save_required_quality(headers, final_session["id"])
    activated = db.activate_paired_sessions(
        participant["id"], final_session["id"], companion["id"]
    )
    interval_id = db.open_pair_interval(
        participant["id"], 1, final_session["id"], companion["id"]
    )
    db.qualify_pair_interval(
        interval_id,
        participant["id"],
        final_session["id"],
        companion["id"],
        int(activated["activated_ms"]),
    )
    probe_order = json.loads(participant["probe_schedule"])[0]
    events = client.post(
        "/api/events/batch",
        headers=headers,
        json={
            "session_id": final_session["id"],
            "events": compressed_protocol_events(probe_order),
        },
    )
    assert events.status_code == 200
    for event_type, payload in (
        ("model.ready", {"model_fingerprint": {}}),
        ("chat.turn", {"source": "primary", "transcript": "test", "reply": "test"}),
        ("tts.generated", {"provider": "iflytek", "fallback": False}),
    ):
        db.append_event(
            participant["id"], final_session["id"], event_type, None, None, payload
        )
    daily = client.post(
        "/api/responses",
        headers=headers,
        json={
            "session_id": final_session["id"],
            "instrument": "daily_day_1",
            "version": protocol.questionnaires["version"],
            "answers": daily_answers(),
        },
    )
    assert daily.status_code == 200
    recorded: list[tuple[str, int, object, dict[str, Any]]] = []

    class FakeModel:
        pass

    runtime = ParticipantRuntime(
        participant_id=participant["id"],
        group_name="donor",
        cohort="study_a",
        settings=settings,
        day=1,
        budget_kind="eligible_feedback",
        budget_semantics="per_donor_cap",
        q_levels=(2, 4, 8),
        donor_shadow_model=FakeModel(),  # type: ignore[arg-type]
        snapshot_recorder=lambda source, budget, model, metadata: recorded.append(
            (source, budget, model, metadata)
        ),
    )
    timing_service.runtimes[(participant["id"], 1)] = runtime
    try:
        response = client.post(f"/api/sessions/{final_session['id']}/end", headers=headers)
    finally:
        timing_service.release(participant["id"])

    assert response.status_code == 200
    assert response.json()["protocol"]["complete"] is True
    assert [row[1] for row in recorded] == [2, 4, 8]
    assert all(row[3]["capped_final"] is True for row in recorded)
    assert db.session(final_session["id"], participant["id"])["state"] == "completed"
    assert (participant["id"], 1) not in timing_service.runtimes


def test_v6_feedback_budget_is_not_an_activation_counter() -> None:
    code = create_participant(run_slug="test-v6-feedback-counter")
    participant = db.participant_by_code(code)
    for _ in range(7):
        db.next_activation_opportunity(participant["id"])
    counts = db.feedback_counts(participant["id"])
    assert counts == {
        "activation_opportunities": 7,
        "eligible_feedbacks": 0,
        "positive_feedbacks": 0,
        "negative_feedbacks": 0,
    }
    db.next_eligible_feedback(participant["id"], 1)
    counts = db.next_eligible_feedback(participant["id"], 0)
    assert counts["activation_opportunities"] == 7
    assert counts["eligible_feedbacks"] == 2
    assert counts["positive_feedbacks"] == 1
    assert counts["negative_feedbacks"] == 1


def test_server_recomputes_network_quality_and_caps_attempts() -> None:
    code = create_participant(run_slug="test-v6-network-gate")
    headers, session = consent_and_start(code)
    bad = client.post(
        "/api/quality",
        headers=headers,
        json={
            "session_id": session["id"],
            "kind": "network",
            "passed": True,
            "metrics": {
                "attempt_id": "bad-network-1",
                "attempt_number": 1,
                "median_ms": 450,
                "p95_ms": 1400,
                "jitter_ms": 260,
                "failures": 0,
                "samples": [450] * 11 + [1400],
            },
        },
    )
    assert bad.status_code == 200
    assert bad.json()["passed"] is False
    assert bad.json()["server_decision"]["hard_failure"] is True

    good = client.post(
        "/api/quality",
        headers=headers,
        json={
            "session_id": session["id"],
            "kind": "network",
            "passed": False,
            "metrics": {
                "attempt_id": "good-network-2",
                "attempt_number": 2,
                "median_ms": 40,
                "p95_ms": 90,
                "jitter_ms": 10,
                "failures": 0,
                "samples": [35, 36, 37, 38, 39, 40, 41, 42, 50, 60, 75, 90],
            },
        },
    )
    assert good.status_code == 200
    assert good.json()["passed"] is True

    exhausted = client.post(
        "/api/quality",
        headers=headers,
        json={
            "session_id": session["id"],
            "kind": "network",
            "passed": True,
            "metrics": {
                "attempt_id": "network-3",
                "attempt_number": 2,
                "median_ms": 40,
                "p95_ms": 90,
                "jitter_ms": 10,
                "failures": 0,
                "samples": [40] * 12,
            },
        },
    )
    assert exhausted.status_code == 409
    assert exhausted.json()["detail"]["code"] == "quality_attempts_exhausted"


def test_client_cannot_forge_server_originated_events() -> None:
    code = create_participant(run_slug="test-v6-server-event-integrity")
    headers, session = consent_and_start(code)
    for event_type in (
        "agent.auto_activate",
        "chat.turn",
        "tts.generated",
        "tap.manual_activate",
    ):
        response = client.post(
            "/api/events",
            headers=headers,
            json={
                "session_id": session["id"],
                "event_type": event_type,
                "payload": {},
            },
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "server_event_required"


def test_non_simulation_account_cannot_bypass_microphone_quality() -> None:
    code = create_participant(run_slug="test-v6-quality-origin")
    headers, session = consent_and_start(code)
    response = client.post(
        "/api/quality",
        headers=headers,
        json={
            "session_id": session["id"],
            "kind": "microphone",
            "passed": True,
            "metrics": {
                "attempt_id": "forged-simulation-mic",
                "attempt_number": 1,
                "simulation": True,
                "snr_db": 30,
                "quiet_rms_p90": 0.01,
                "speech_rms_p90": 0.2,
                "clipping_ratio": 0,
                "asr_probe_passed": True,
            },
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "simulation_quality_forbidden"


def test_simulation_study_a_loads_locked_base_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[tuple[str, str]] = []

    class FakeDTCNModel:
        def __init__(self, _settings, encoder_path: str, main_path: str):
            loaded.append((str(encoder_path), str(main_path)))

    monkeypatch.setattr("app.timing.DTCNModel", FakeDTCNModel)
    code = create_participant(
        cohort="study_a",
        group="donor",
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug="test-simulation-study-a-m0",
    )
    participant = db.participant_by_code(code)
    service = TimingService(replace(settings, agent_mode="dtcn"), db)
    runtime = service.runtime(participant, day=1)

    expected = (settings.base_encoder_checkpoint, settings.base_main_checkpoint)
    assert runtime.model is not None
    assert runtime.matched_model is not None
    assert len(loaded) == 2
    assert loaded[-1] == expected


def test_background_assets_match_manifest() -> None:
    manifest = protocol.background
    assert manifest["approved_for_distribution"] is True
    assert len(manifest["tracks"]) == 3
    for track in manifest["tracks"]:
        target = settings.static_dir / track["url"].removeprefix("/static/")
        assert target.is_file()
        assert target.stat().st_size == track["bytes"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == track["sha256"]


def test_study_tap_zone_layout_is_viewport_bounded() -> None:
    css = (settings.static_dir / "styles.css").read_text(encoding="utf-8")
    app_js = (settings.static_dir / "app.js").read_text(encoding="utf-8")

    assert "@media (min-width: 901px)" in css
    assert "height: calc(100dvh - 65px);" in css
    assert "#pilot-banner:not(.hidden) + main .study-shell" in css
    assert ".study-body {\n    overflow: hidden;" in css
    assert "window.scrollTo(0, 0);" in app_js
    assert "requestAnimationFrame(() => window.scrollTo(0, 0));" in app_js
    for filename in ("index.html", "companion.html", "admin.html"):
        html = (settings.static_dir / filename).read_text(encoding="utf-8")
        assert "styles.css?v=20260803-guarded-voice-v24" in html


def test_admin_export_excludes_consent_and_compensation_databases() -> None:
    response = client.get("/api/admin/export", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert b"consents.csv" not in response.content
    assert b"study_runs.csv" in response.content
    assert b"account_batches.csv" in response.content
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert not any("consent" in name.lower() for name in names)
    assert not any("compensation" in name.lower() for name in names)
    assert not any("identity" in name.lower() for name in names)
    assert manifest["study_protocol_version"] == protocol.study_version
    assert manifest["study_runs"]


def test_matched_other_requires_frozen_per_day_manifest_and_self_test() -> None:
    root = TEST_DATA / "matched-test"
    root.mkdir(exist_ok=True)
    encoder = root / "encoder.pth"
    main_model = root / "main.pth"
    encoder.write_bytes(b"encoder-checkpoint")
    main_model.write_bytes(b"main-checkpoint")
    participant_id = "target-participant"
    donor_id = "donor-participant"
    entry = {
        "donor_participant_id": donor_id,
        "matching_strata": {"day": 2},
        "encoder_checkpoint": str(encoder.resolve()),
        "encoder_sha256": file_sha256(encoder),
        "main_checkpoint": str(main_model.resolve()),
        "main_sha256": file_sha256(main_model),
    }
    manifest = {
        "version": "test-v1",
        "approved": True,
        "assignment_rule": "test rule",
        "matching_variables": ["day"],
        "random_seed": 7,
        "frozen_at_ms": 1,
        "assignments": {participant_id: {"2": entry}},
    }
    manifest_path = root / "manifest.json"
    sentinel_path = root / "sentinel.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sentinel = {
        "passed": True,
        "manifest_sha256": file_sha256(manifest_path),
        "assignments": {
            participant_id: {
                "2": {
                    "donor_participant_id": donor_id,
                    "encoder_sha256": entry["encoder_sha256"],
                    "main_sha256": entry["main_sha256"],
                }
            }
        },
    }
    sentinel_path.write_text(json.dumps(sentinel), encoding="utf-8")
    matched_settings = replace(
        settings,
        matched_model_manifest=str(manifest_path),
        matched_model_ready_sentinel=str(sentinel_path),
    )

    assignment, checks = matched_assignment(matched_settings, participant_id, 2)
    assert assignment is not None
    assert all(checks.values())
    missing_day, missing_checks = matched_assignment(matched_settings, participant_id, 1)
    assert missing_day is None
    assert not missing_checks["participant_day_assignment_present"]
