"""FastAPI entry point for the two-device speech-interaction study.

HTTP endpoints serve participant interfaces, consent, questionnaires, and
operator views. WebSocket endpoints coordinate device pairing and streaming
timing inference. Authentication, state validation, consent evidence, and
participant-scoped storage are enforced before the corresponding operations.
Importing this module initializes the configured local database and services.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import html
import io
import json
import logging
import math
import re
import struct
import threading
import time
import uuid
import zipfile
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import (
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import ChatService
from .asr import ServerASRCancelled, ServerASRUnavailable, WhisperASRService
from .auth import (
    TokenError,
    decode_token,
    ip_hash,
    is_full_participant_code,
    issue_token,
    normalize_participant_code,
)
from .compensation import CompensationVault, CompensationVaultUnavailable
from .config import settings
from .consent_evidence import (
    ASSURANCE_SCOPE,
    EVIDENCE_SCHEMA_VERSION,
    ConsentEvidenceSigner,
    ConsentEvidenceUnavailable,
    canonical_json as canonical_evidence_json,
    sha256_text,
    verify_evidence_bundle,
)
from .db import StudyCapacityExceeded, StudyDB, estimated_timed_progress
from .ethics import (
    formal_readiness_is_ready,
    normalize_ethics_status,
    should_show_document_status_notice,
)
from .integrity import file_sha256
from .matched import matched_assignment
from .pipeline import FormalPipeline, PipelineError
from .pipeline_guard import pipeline_readiness, requires_locked_pipeline
from .protocol import Protocol
from .priors import PriorError, prior_readiness
from .quality import QualityPayloadError, evaluate_quality
from .reply_needed import ReplyNeededModel, ReplyNeededUnavailable
from .speech import SpeechService, SpeechSynthesisError
from .interrupt_checkpoint import (
    infer_interrupt_checkpoint,
    is_code_hint,
    normalize_code_hint,
    public_interrupt_checkpoint,
)
from .operator_resume import hide_block5_event, load_operator_resume, public_operator_resume
from .study_state import ProtocolViolation, StudyState
from .realtime_lease import CapacityExceeded, RealtimeLeaseRegistry
from .orchestration import (
    AutoOrchestrator,
    OrchestrationError,
    ShareAppointmentError,
)
from .three_arm_diff import (
    CONTRAST_ARMS,
    QUESTIONNAIRE_FACETS,
    build_three_arm_difference_board,
    questionnaire_facet,
)
from .timing import ModelUnavailable, TimingService


logger = logging.getLogger(__name__)


settings.ensure_directories()
protocol = Protocol(settings)
db = StudyDB(settings)
orchestrator = AutoOrchestrator(settings, db)
compensation_vault = CompensationVault(
    settings.data_dir, settings.compensation_vault_key
)
consent_evidence_signer = ConsentEvidenceSigner.from_secret(
    settings.instance_id, settings.consent_evidence_signing_key
)
study_state = StudyState(db, protocol)
chat_service = ChatService(settings)
timing_service = TimingService(settings, db)
speech_service = SpeechService(settings)
server_asr_service = WhisperASRService(
    enabled=settings.server_asr_enabled,
    model_name=settings.server_asr_model,
    device=settings.server_asr_device,
    download_root=settings.server_asr_download_root,
    max_seconds=settings.server_asr_max_seconds,
    min_rms=settings.server_asr_min_rms,
    replica_count=settings.server_asr_replicas,
    devices=settings.server_asr_devices,
)
_asr_turn_lock = threading.Lock()
_asr_turn_generation: dict[str, int] = {}


def _begin_asr_turn(session_id: str | None) -> int:
    if not session_id:
        return 0
    with _asr_turn_lock:
        generation = int(_asr_turn_generation.get(session_id, 0)) + 1
        _asr_turn_generation[session_id] = generation
        return generation


def _asr_turn_is_current(session_id: str | None, generation: int) -> bool:
    if not session_id:
        return True
    with _asr_turn_lock:
        return int(_asr_turn_generation.get(session_id, 0)) == generation
try:
    reply_needed_model = (
        ReplyNeededModel(settings.reply_needed_model)
        if settings.reply_needed_model
        else None
    )
except ReplyNeededUnavailable:
    logger.exception("Reply-needed context model could not be loaded")
    reply_needed_model = None

consent_archive_lock = threading.RLock()
public_enrollment_lock = threading.RLock()
public_enrollment_attempts: dict[str, deque[float]] = defaultdict(deque)
final_recovery_lock = threading.RLock()
final_recovery_attempts: dict[str, deque[float]] = defaultdict(deque)
appointment_resume_lock = threading.RLock()
appointment_resume_attempts: dict[str, deque[float]] = defaultdict(deque)
code_hint_entry_lock = threading.RLock()
code_hint_entry_attempts: dict[str, deque[float]] = defaultdict(deque)
realtime_lease_registry = RealtimeLeaseRegistry()


def _participant_holds_activated_study(participant_id: str) -> bool:
    for session in db.sessions(participant_id):
        if (
            session.get("client_role") == "study"
            and session.get("state") == "active"
            and session.get("activated_ms") is not None
            and session.get("ended_ms") is None
        ):
            return True
    return False


def _capacity_status(
    *,
    participant_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    load = db.concurrent_study_load()
    max_concurrent = int(settings.max_concurrent_study_sessions)
    retry_after = int(settings.capacity_retry_after_minutes)
    # Preferential restart: an already-activated, unfinished study session can
    # re-enter even if its recent-event heartbeat briefly went quiet.
    holding = bool(participant_id and _participant_holds_activated_study(participant_id))
    if session_id and not holding:
        holding = any(row["session_id"] == session_id for row in load["sessions"])
    ok = holding or load["active_count"] < max_concurrent
    message_zh = (
        f"当前同时进行实验的人数较多（{load['active_count']}/{max_concurrent}），"
        f"为保证语音与网络稳定，请约 {retry_after} 分钟后再来尝试。"
    )
    message_en = (
        f"Too many participants are running at once "
        f"({load['active_count']}/{max_concurrent}). "
        f"For stable voice/network quality, please try again in about "
        f"{retry_after} minutes."
    )
    return {
        "ok": ok,
        "active_count": load["active_count"],
        "max_concurrent": max_concurrent,
        "retry_after_minutes": retry_after,
        "holding_slot": holding,
        "message": message_zh,
        "message_zh": message_zh,
        "message_en": message_en,
        "server_ms": load["server_ms"],
    }


def _require_study_capacity(
    *,
    participant_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    status = _capacity_status(participant_id=participant_id, session_id=session_id)
    if status["ok"]:
        return status
    raise HTTPException(
        status_code=429,
        detail={
            "code": "study_capacity_full",
            "message": status["message_zh"],
            "message_en": status["message_en"],
            "retry_after_minutes": status["retry_after_minutes"],
            "active_count": status["active_count"],
            "max_concurrent": status["max_concurrent"],
        },
    )


def _pipeline_for_run(run_slug: str) -> FormalPipeline:
    return FormalPipeline.from_run(
        db,
        settings,
        run_slug,
        allow_simulation=True,
    )

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.tts_mode == "sherpa_onnx" and settings.listener_role != "realtime":
        await asyncio.to_thread(speech_service.synthesize, "语音系统已经准备完成。")
    if settings.server_asr_enabled and settings.listener_role != "realtime":
        async def warm_server_asr() -> None:
            try:
                await asyncio.to_thread(server_asr_service.warmup)
            except ServerASRUnavailable:
                logger.exception("Server ASR warmup failed; transcripts stay unavailable until Whisper is ready")

        asyncio.create_task(warm_server_asr())
    yield


app = FastAPI(
    title="Tap-to-Adapt Online Study",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


class LoginRequest(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    role: Literal["study", "companion"]


class PublicEnrollmentRequest(BaseModel):
    device_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{8,128}$")
    contact_email: str = Field(default="", max_length=254)
    contact_phone: str = Field(default="", max_length=40)
    contact_wechat: str = Field(default="", max_length=100)
    intent: Literal["start_now", "future_book"] = "start_now"
    group_session_password: str = Field(default="", max_length=32)


class FinalQuestionnaireRecoveryRequest(BaseModel):
    credential: str = Field(min_length=8, max_length=20)


class CodeHintEntryRequest(BaseModel):
    hint: str = Field(min_length=4, max_length=8)


class AppointmentResumeRequest(BaseModel):
    device_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{8,128}$")
    voucher: str = Field(default="", max_length=8)
    full_name: str = Field(default="", max_length=100)
    contact_email: str = Field(default="", max_length=254)
    contact_phone: str = Field(default="", max_length=40)
    contact_wechat: str = Field(default="", max_length=100)


class ConsentRequest(BaseModel):
    version: str
    document_hash: str
    participation: bool
    raw_audio: bool = False
    future_sharing: bool = False
    internal_reuse: bool = False
    controlled_sharing: bool = False
    public_derived: bool = False
    public_audio: bool = False
    ai_training: bool = False
    commercial_research: bool = False
    typed_name: str = Field(min_length=2, max_length=100)
    affirmation: str = Field(min_length=4, max_length=400)
    client_nonce: str = Field(default="", pattern=r"^[0-9a-f]{0,64}$")


class CompensationIdentityRequest(BaseModel):
    notice_version: str = Field(min_length=4, max_length=100)
    notice_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_type: Literal["resident_identity_card", "passport_or_other"]
    # Optional: server prefers the consent-form typed name from consent.
    full_name: str = Field(default="", max_length=100)
    document_number: str = Field(min_length=5, max_length=40)
    signature_png_base64: str = Field(min_length=40, max_length=700_000)
    separate_consent: bool
    affirmation: str = Field(min_length=10, max_length=600)
    contact_email: str = Field(default="", max_length=254)
    contact_phone: str = Field(default="", max_length=40)
    contact_wechat: str = Field(default="", max_length=100)


class CompensationDeclineRequest(BaseModel):
    notice_version: str = Field(min_length=4, max_length=100)
    notice_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["decline"]
    affirmation: str = Field(min_length=10, max_length=600)
    contact_email: str = Field(default="", max_length=254)
    contact_phone: str = Field(default="", max_length=40)
    contact_wechat: str = Field(default="", max_length=100)


class CompensationContactRequest(BaseModel):
    notice_version: str = Field(min_length=4, max_length=100)
    notice_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contact_email: str = Field(default="", max_length=254)
    contact_phone: str = Field(default="", max_length=40)
    contact_wechat: str = Field(default="", max_length=100)


class SessionRequest(BaseModel):
    day: int = Field(ge=1, le=3)
    group_session_password: str = Field(default="", max_length=32)


class EventRequest(BaseModel):
    event_type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    session_id: str | None = None
    client_ms: float | None = None
    monotonic_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class EventBatchRequest(BaseModel):
    session_id: str | None = None
    events: list[EventRequest] = Field(min_length=1, max_length=100)


class QualityRequest(BaseModel):
    session_id: str | None = None
    kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    passed: bool
    metrics: dict[str, Any]


class ASRTranscriptionRequest(BaseModel):
    session_id: str | None = None
    pcm16_base64: str = Field(min_length=4, max_length=1_000_000)
    sample_rate_hz: Literal[16000] = 16000
    language: Literal["zh", "en"]
    context: Literal["preflight", "primary", "companion"]


class ResponseRequest(BaseModel):
    session_id: str | None = None
    instrument: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    version: str = Field(max_length=100)
    answers: dict[str, Any]
    locale: Literal["zh-CN", "en"] = "zh-CN"
    question_order: list[str] = Field(default_factory=list, max_length=100)


class ChatRequest(BaseModel):
    session_id: str
    source: Literal["primary", "companion"]
    transcript: str = Field(min_length=1, max_length=1000)


class ReplyNeededRequest(BaseModel):
    session_id: str
    transcript: str = Field(min_length=1, max_length=1000)


class SpeechRequest(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=120)


class SimulationSpeechRequest(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=120)
    source: Literal["participant", "companion"] = "participant"
    speaker_id: int | None = Field(default=None, ge=0, le=999)
    speed: float = Field(ge=0.75, le=1.25)


class ParticipantCreateRequest(BaseModel):
    count: int = Field(ge=1, le=250)
    cohort: Literal["pilot", "study_a", "study_b"]
    group_name: str = Field(min_length=1, max_length=40)
    preview_mode: bool = False
    data_origin: Literal["formal", "pilot", "simulation"] | None = None
    protocol_mode: Literal["strict", "compressed"] | None = None
    run_slug: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_.-]{2,79}$")


class StageAccountInitRequest(BaseModel):
    stage: str = Field(pattern=r"^[a-z][a-z0-9_]{2,39}$")
    count: int = Field(ge=1, le=64)
    run_slug: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9_.-]{2,79}$"
    )
    label: str = Field(default="", max_length=120)
    personal_debug: bool = False
    pre_experiment: bool | None = None


class PipelineWaveRequest(BaseModel):
    run_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,79}$")
    wave: int = Field(ge=1, le=4)


class PipelineRunRequest(BaseModel):
    run_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,79}$")


class PipelinePrepareRequest(PipelineRunRequest):
    selected_n: int = Field(ge=12, le=28)
    selected_q: int = Field(ge=1, le=1000)
    decision_record: str = Field(min_length=20, max_length=2000)


class PipelineReserveRequest(PipelineRunRequest):
    reason: str = Field(min_length=10, max_length=1000)


class OrchestrationAutoRequest(BaseModel):
    enabled: bool


class OrchestrationLockRequest(BaseModel):
    donor_count: int | None = Field(default=None, ge=1, le=500)


class OrchestrationBlockedSlotsRequest(BaseModel):
    date_cst: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    scheduled_start_ms: list[int] = Field(default_factory=list, max_length=48)


class ShareAppointmentRequest(BaseModel):
    scheduled_start_ms: int = Field(gt=0)


class ShareAppointmentArrivalRequest(BaseModel):
    voucher: str | None = Field(default=None, pattern=r"^\d{4}$")
    group_session_password: str = Field(default="", max_length=32)


class SessionLockWindow(BaseModel):
    start_ms: int = Field(gt=0)
    end_ms: int = Field(gt=0)


class SessionLockRequest(BaseModel):
    start_ms: int | None = Field(default=None, gt=0)
    end_ms: int | None = Field(default=None, gt=0)
    windows: list[SessionLockWindow] = Field(default_factory=list, max_length=8)


class SessionLockUnlockRequest(BaseModel):
    password: str = Field(min_length=4, max_length=32)


ADMIN_STAGE_TEMPLATES: dict[str, dict[str, Any]] = {
    "engineering_smoke": {
        "label": "工程 smoke",
        "description": "压缩协议；只验证登录、配对、音频、状态机和数据落盘。",
        "recommended_count": 4,
        "maximum_count": 16,
        "cohort": "pilot",
        "group_name": "pilot",
        "data_origin": "pilot",
        "protocol_mode": "compressed",
        "preview_mode": False,
        "personal_debug_default": False,
        "initializable": True,
    },
    # Duration-fitting (4-person T_formal estimation) is retired; T_formal is
    # engineering-locked at 30 minutes. Keep the template visible as disabled.
    "duration_prepilot": {
        "label": "试验时长选型试验（已停用）",
        "description": "已停用：正式时长改为工程锁定 30 分钟（5×6 分钟块），不再用 4 人拟合估计 T_formal。",
        "recommended_count": 4,
        "maximum_count": 8,
        "cohort": "pilot",
        "group_name": "personalized",
        "data_origin": "pilot",
        "protocol_mode": "strict",
        "preview_mode": False,
        "personal_debug_default": False,
        "initializable": False,
    },
    "model_selection_simulation": {
        "label": "模型选型仿真",
        "description": "压缩 simulation/preview 账号；不得计作真人样本。",
        "recommended_count": 8,
        "maximum_count": 32,
        "cohort": "pilot",
        "group_name": "personalized",
        "data_origin": "simulation",
        "protocol_mode": "compressed",
        "preview_mode": True,
        "personal_debug_default": False,
        "initializable": True,
    },
    "study_a_formal": {
        "label": "Study A 正式 donor",
        "description": "必须由锁定 roster provisioner 和 A→B 流水线统一创建/释放。",
        "recommended_count": None,
        "maximum_count": None,
        "initializable": False,
        "personal_debug_default": False,
        "managed_by": "scripts/provision_formal_pipeline_roster.py",
    },
    "study_b_formal": {
        "label": "Study B 正式 C_B/F/P",
        "description": "必须在 S_sat 锁定后由正式流水线随机分配，不允许从简易表单创建。",
        "recommended_count": None,
        "maximum_count": None,
        "initializable": False,
        "personal_debug_default": False,
        "managed_by": "A→B formal pipeline",
    },
}


def _batch_archive_dir(batch: dict[str, Any]) -> Path:
    relative = str(batch.get("archive_relative_path") or "")
    if not relative:
        raise ValueError("batch does not have a pre-experiment archive")
    root = settings.data_dir.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("batch archive path escapes the study data directory")
    return target


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _archive_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    if rows:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_text(path, f"\ufeff{buffer.getvalue()}")


def _initialize_pre_experiment_archive(batch: dict[str, Any]) -> dict[str, Any]:
    archive_dir = _batch_archive_dir(batch)
    (archive_dir / "records").mkdir(parents=True, exist_ok=True)
    (archive_dir / "audio").mkdir(parents=True, exist_ok=True)
    analysis_excluded = bool(batch.get("analysis_excluded"))
    if analysis_excluded:
        (archive_dir / "models").mkdir(parents=True, exist_ok=True)
    participant_ids = json.loads(str(batch["participant_ids"]))
    manifest = {
        "schema_version": "tap-to-adapt-internal-pre-experiment-archive-v1",
        "archive_kind": (
            "personal_debug_isolated"
            if analysis_excluded
            else "internal_pre_experiment"
        ),
        "batch_id": batch["id"],
        "run_id": batch["run_id"],
        "run_slug": batch["run_slug"],
        "stage": batch["stage"],
        "label": batch["label"],
        "assigned_count": batch["requested_count"],
        "participant_ids": participant_ids,
        "analysis_excluded": analysis_excluded,
        "storage_namespace": batch.get("storage_namespace"),
        "forbidden_uses": (
            ["formal_analysis", "shared_prior", "dataset_release"]
            if analysis_excluded
            else []
        ),
        "access_codes_stored": False,
        "created_ms": batch["created_ms"],
        "refreshed_ms": None,
        "records": [],
        "notice": (
            "Permanently isolated personal-debug archive. Records may be retained for audit "
            "and debugging, but must never enter formal analysis, shared-prior training, or "
            "dataset release."
            if analysis_excluded
            else
            "Restricted internal archive. It may contain raw event payloads and questionnaire "
            "answers after refresh; it is not a public dataset release."
        ),
    }
    _atomic_text(
        archive_dir / "ARCHIVE_MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    _atomic_text(
        archive_dir / "README.txt",
        (
            (
                "Tap-to-Adapt 个人调试隔离归档\n\n"
                if analysis_excluded
                else "Tap-to-Adapt 预实验批次归档\n\n"
            )
            + "records/ 保存从不可变研究数据库生成的批次快照；audio/ 保存该批次后续授权留存的原始音频。\n"
            + (
                "models/ 保存该调试账号独有的 personal head；不得复制到正式模型目录。\n"
                if analysis_excluded
                else ""
            )
            + "这里不保存完整登录码，也不保存姓名、邮箱或招募名单。\n"
            + (
                "本批次永久排除于正式统计、shared prior 训练和任何数据集发布。\n"
                if analysis_excluded
                else
                "本目录仅供受限内部研究使用，不等于经过同意筛选和去标识的数据集发布。\n"
            )
        ),
    )
    return {
        "archive_path": str(archive_dir),
        "manifest_path": str((archive_dir / "ARCHIVE_MANIFEST.json").resolve()),
        "assigned_count": int(batch["requested_count"]),
    }


def _refresh_pre_experiment_archive(batch_id: str) -> dict[str, Any]:
    batch = db.account_batch(batch_id)
    if batch is None:
        raise ValueError("account batch not found")
    if not bool(batch["is_pre_experiment"]):
        raise ValueError("only pre-experiment batches have a unified archive")
    archive_dir = _batch_archive_dir(batch)
    records_dir = archive_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    participant_ids = set(json.loads(str(batch["participant_ids"])))
    run_id = str(batch["run_id"])

    participant_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "code_hash"
        }
        for row in db.table_rows("participants")
        if str(row["id"]) in participant_ids
    ]
    record_sets: dict[str, list[dict[str, Any]]] = {
        "study_run": [
            row for row in db.table_rows("study_runs") if str(row["id"]) == run_id
        ],
        "participants": participant_rows,
    }
    participant_tables = (
        "sessions",
        "pair_intervals",
        "events",
        "quality_checks",
        "responses",
        "final_questionnaire_recoveries",
        "audio_files",
        "model_snapshots",
        "model_counters",
        "participant_calibrations",
    )
    for table in participant_tables:
        record_sets[table] = [
            row
            for row in db.table_rows(table)
            if str(row.get("participant_id") or "") in participant_ids
        ]
    record_sets["account_batch"] = [
        {
            key: value
            for key, value in batch.items()
            if key not in {"participant_ids"}
        }
    ]
    consent_rows = []
    for participant_id in sorted(participant_ids):
        consent = db.latest_consent(participant_id)
        participant = db.participant(participant_id)
        archive = (
            _ensure_server_consent_archive(consent, participant)
            if consent is not None and participant is not None
            else None
        )
        consent_rows.append(
            {
                "participant_id": participant_id,
                "consent_id": consent["id"] if consent else None,
                "signed": bool(consent and consent["participation"]),
                "version": consent["version"] if consent else None,
                "document_hash": consent["document_hash"] if consent else None,
                "signed_ms": consent["signed_ms"] if consent else None,
                "raw_audio": bool(consent and consent["raw_audio"]),
                "internal_reuse": bool(
                    consent and consent.get("internal_reuse")
                ),
                "future_sharing": bool(
                    consent and consent.get("future_sharing")
                ),
                "controlled_sharing": bool(
                    consent and consent.get("controlled_sharing")
                ),
                "public_derived": bool(
                    consent and consent.get("public_derived")
                ),
                "public_audio": bool(
                    consent and consent.get("public_audio")
                ),
                "ai_training": bool(
                    consent and consent.get("ai_training")
                ),
                "commercial_research": bool(
                    consent and consent.get("commercial_research")
                ),
                "server_archive_saved": bool(archive and archive["verified"]),
                "server_archive_path": (
                    archive["archive_path"] if archive else None
                ),
                "server_archive_record_sha256": (
                    archive["record_sha256"] if archive else None
                ),
            }
        )
    record_sets["consent_status"] = consent_rows

    record_paths = []
    for name, rows in record_sets.items():
        target = records_dir / f"{name}.csv"
        _archive_csv(target, rows)
        record_paths.append(target)
    files = [
        {
            "path": str(path.relative_to(archive_dir)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(record_paths)
    ]
    audio_rows = record_sets["audio_files"]
    analysis_excluded = bool(batch.get("analysis_excluded"))
    manifest = {
        "schema_version": "tap-to-adapt-internal-pre-experiment-archive-v1",
        "archive_kind": (
            "personal_debug_isolated"
            if analysis_excluded
            else "internal_pre_experiment"
        ),
        "batch_id": batch["id"],
        "run_id": batch["run_id"],
        "run_slug": batch["run_slug"],
        "stage": batch["stage"],
        "label": batch["label"],
        "assigned_count": batch["requested_count"],
        "participant_ids": sorted(participant_ids),
        "analysis_excluded": analysis_excluded,
        "storage_namespace": batch.get("storage_namespace"),
        "forbidden_uses": (
            ["formal_analysis", "shared_prior", "dataset_release"]
            if analysis_excluded
            else []
        ),
        "access_codes_stored": False,
        "created_ms": batch["created_ms"],
        "refreshed_ms": int(time.time() * 1000),
        "record_counts": {
            name: len(rows) for name, rows in record_sets.items()
        },
        "audio_file_count": len(audio_rows),
        "records": files,
        "notice": (
            "Permanently isolated personal-debug archive. Raw records may be retained for "
            "audit, but formal analysis, shared-prior training, and dataset release are forbidden."
            if analysis_excluded
            else
            "Restricted internal archive. Raw event payloads and questionnaire answers may "
            "be present. Public or controlled release must use export_research_dataset.py."
        ),
    }
    manifest_path = archive_dir / "ARCHIVE_MANIFEST.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    return {
        "batch_id": batch_id,
        "archive_path": str(archive_dir),
        "manifest_path": str(manifest_path.resolve()),
        "assigned_count": int(batch["requested_count"]),
        "record_counts": manifest["record_counts"],
        "audio_file_count": len(audio_rows),
        "refreshed_ms": manifest["refreshed_ms"],
    }


def _request_ip(request: Request) -> str:
    direct = request.client.host if request.client else "unknown"
    if direct in {"127.0.0.1", "::1"}:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    return direct


_PHONE_USER_AGENT = re.compile(
    r"iphone|ipod|windows phone|iemobile|opera mini|android.+mobile|mobile safari",
    re.IGNORECASE,
)
_TABLET_USER_AGENT = re.compile(
    r"ipad|tablet|kindle|silk|playbook|android(?!.*mobile)",
    re.IGNORECASE,
)
_DESKTOP_USER_AGENT = re.compile(
    r"windows nt|macintosh|x11|cros|linux (?:x86_64|i[3-6]86|aarch64)",
    re.IGNORECASE,
)


def _request_device_class(request: Request) -> str:
    """Classify the physical browser role using browser and HTTP signals.

    The web clients send their capability-based result in a same-origin header.
    User-Agent signals prevent an obviously wrong desktop/phone role from being
    accepted merely by changing that reported value.  ``testclient`` is a
    deliberate exemption for in-process protocol tests, not a public browser.
    """

    user_agent = request.headers.get("user-agent", "").strip()
    if user_agent.lower().startswith("testclient"):
        return "testclient"

    reported = request.headers.get("x-tta-device-class", "").strip().lower()
    if reported not in {"desktop", "phone", "tablet", "unknown"}:
        reported = "unknown"

    wechat_mobile = bool(
        re.search(r"micromessenger", user_agent, re.IGNORECASE)
        and not re.search(r"windowswechat|macwechat", user_agent, re.IGNORECASE)
        and (
            re.search(r"mobile|iphone|ipod|android", user_agent, re.IGNORECASE)
            or request.headers.get("sec-ch-ua-mobile", "").strip() == "?1"
            or reported == "phone"
        )
        and not re.search(r"ipad", user_agent, re.IGNORECASE)
    )

    if wechat_mobile:
        detected = "phone"
    elif _TABLET_USER_AGENT.search(user_agent):
        detected = "tablet"
    elif _PHONE_USER_AGENT.search(user_agent):
        detected = "phone"
    elif request.headers.get("sec-ch-ua-mobile", "").strip() == "?1":
        detected = "phone"
    elif _DESKTOP_USER_AGENT.search(user_agent):
        detected = "desktop"
    elif request.headers.get("sec-ch-ua-mobile", "").strip() == "?0":
        detected = "desktop"
    else:
        detected = "unknown"

    # iPadOS can identify itself as Macintosh.  The client-side touch-capability
    # check can identify that case more accurately, so preserve a tablet report.
    if reported == "tablet":
        return "tablet"
    # Prefer WeChat-on-phone signals over the broad android-without-mobile tablet rule.
    if wechat_mobile or (reported == "phone" and "micromessenger" in user_agent.lower()):
        return "phone"
    if detected != "unknown":
        return detected
    return reported


def _require_device_role(request: Request, role: Literal["study", "companion"]) -> None:
    device_class = _request_device_class(request)
    if device_class == "testclient":
        return
    if role == "study" and device_class != "desktop":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "desktop_device_required",
                "message": "无法开始，需要在电脑端打开实验系统",
                "detected_device": device_class,
            },
        )
    if role == "companion" and device_class != "phone":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "phone_device_required",
                "message": "无法开始，需要在手机端打开陪伴系统",
                "detected_device": device_class,
            },
        )


def _authorization_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization[7:].strip()


def current_auth(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        claims = decode_token(_authorization_token(authorization), settings)
    except (TokenError, HTTPException) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session") from exc
    participant = db.participant(str(claims["sub"]))
    if participant is None:
        raise HTTPException(status_code=401, detail="Participant is disabled")
    return {"claims": claims, "participant": participant}


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    import hmac

    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(status_code=401, detail="Invalid admin token")


def _group_session_password(request: Request, payload: Any = None) -> str | None:
    header = (request.headers.get("X-Group-Session-Password") or "").strip()
    if header:
        return header
    if payload is not None:
        value = str(getattr(payload, "group_session_password", "") or "").strip()
        if value:
            return value
    return None


def _validate_session(auth: dict[str, Any], session_id: str | None) -> None:
    if session_id and db.session(session_id, auth["participant"]["id"]) is None:
        raise HTTPException(status_code=404, detail="Session not found")


def _formal_ready() -> tuple[dict[str, bool], bool]:
    checks = settings.formal_readiness(
        protocol.consent_version,
        consent_document=protocol.consent,
        compensation_document=protocol.compensation_notices["zh-CN"],
    )
    checks["compensation_vault_configured"] = compensation_vault.ready
    checks["consent_evidence_signer_ready"] = consent_evidence_signer.ready
    return checks, formal_readiness_is_ready(checks)


def _participant_readiness(participant: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    checks, _ = _formal_ready()
    try:
        participant_study = protocol.study_for(participant)
        checks["protocol_snapshot_match"] = True
    except ValueError:
        participant_study = protocol.study
        checks["protocol_snapshot_match"] = False
    if requires_locked_pipeline(participant):
        checks.update(prior_readiness(db, participant))
        probe_assignments = db.prior_probe_assignments(
            participant_id=participant["id"], day=1
        )
        if probe_assignments:
            checks["candidate_probe_assignments_ready"] = all(
                Path(row["encoder_path"]).is_file()
                and Path(row["main_path"]).is_file()
                and file_sha256(Path(row["encoder_path"])) == row["encoder_sha256"]
                and file_sha256(Path(row["main_path"])) == row["main_sha256"]
                for row in probe_assignments
            )
    if participant["cohort"] == "study_b":
        total_sessions = int(
            participant_study.get("session_design", {}).get("total_sessions", 3)
        )
        for day in range(1, total_sessions + 1):
            _, matched_checks = matched_assignment(
                settings, participant["id"], day, participant.get("run_id")
            )
            checks.update(
                {f"matched_day{day}_{name}": passed for name, passed in matched_checks.items()}
            )
    return checks, formal_readiness_is_ready(checks)


def _formal_lock_detail(checks: dict[str, bool]) -> dict[str, Any]:
    return {
        "code": "formal_study_locked",
        "message": "该编号属于正式研究，当前尚未开放。请联系研究团队，不要改用他人的编号。",
        "missing": [name for name, passed in checks.items() if not passed],
    }


def _pipeline_lock_detail(checks: dict[str, bool]) -> dict[str, Any]:
    return {
        "code": "study_pipeline_locked",
        "message": "该测试编号尚未完成前序模型准备，请联系研究团队。",
        "missing": [name for name, passed in pipeline_readiness(checks).items() if not passed],
    }


def _require_formal_ready(participant: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    checks, is_ready = _participant_readiness(participant)
    data_origin = participant.get("data_origin")
    if data_origin is None:
        data_origin = "pilot" if participant["cohort"] == "pilot" else "formal"
    if (
        data_origin == "formal"
        and settings.formal_readiness_enforced
        and not is_ready
    ):
        raise HTTPException(status_code=423, detail=_formal_lock_detail(checks))
    if data_origin != "formal" and requires_locked_pipeline(participant):
        pipeline_checks = pipeline_readiness(checks)
        if not pipeline_checks or not all(pipeline_checks.values()):
            raise HTTPException(status_code=423, detail=_pipeline_lock_detail(checks))
    return checks, is_ready


def _protocol_http_error(exc: ProtocolViolation) -> HTTPException:
    return HTTPException(status_code=409, detail=exc.detail())


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    if request.url.path.startswith("/api/") or request.url.path in {
        "/",
        "/companion",
        "/admin",
        "/transport-probe",
    } or request.url.path.startswith("/companion/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def study_page() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/companion")
def companion_page() -> FileResponse:
    return FileResponse(settings.static_dir / "companion.html")


@app.get("/companion/{participant_code}")
def companion_deep_link(participant_code: str, lang: str | None = None) -> RedirectResponse:
    # Redirect to the normal /companion page with ?code= in the query string.
    # Serving HTML under /companion/{code} with a <base> tag broke deferred script
    # startup in WeChat (users stuck on “正在准备连接…” with Continue disabled).
    # Query parameters survive WeChat redirects; hash fragments often do not.
    code = quote(participant_code.strip(), safe="")
    public_base = settings.public_base_url.rstrip("/")
    params: list[str] = []
    if protocol.normalize_locale(lang) == "en":
        params.append("lang=en")
    params.append(f"code={code}")
    return RedirectResponse(
        url=f"{public_base}/companion?{'&'.join(params)}",
        status_code=status.HTTP_302_FOUND,
    )


@app.get("/transport-probe")
def transport_probe_page() -> FileResponse:
    return FileResponse(settings.static_dir / "transport-probe.html")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(settings.static_dir / "admin.html")


@app.get("/health")
def health() -> dict[str, Any]:
    lease_snapshot = realtime_lease_registry.snapshot()
    capacity = _capacity_status()
    return {
        "ok": True,
        "time_ms": int(time.time() * 1000),
        "mode": settings.agent_mode,
        "instance_id": settings.instance_id,
        "storage_namespace": settings.storage_namespace,
        "public_enrollment": settings.public_enrollment_enabled,
        "listener_role": settings.listener_role,
        "preload_timing_model": settings.loads_timing_model_on_demand,
        "realtime_leases_active": lease_snapshot["active_count"],
        "capacity_ok": capacity["ok"],
        "capacity_active": capacity["active_count"],
        "capacity_max": capacity["max_concurrent"],
    }


@app.get("/api/capacity")
def public_capacity() -> dict[str, Any]:
    return {"ok": True, "capacity": _capacity_status()}


@app.get("/ready")
def ready() -> JSONResponse:
    checks, is_ready = _formal_ready()
    return JSONResponse(
        {"formal_ready": is_ready, "checks": checks, "consent_version": protocol.consent_version},
        status_code=200 if is_ready else 503,
    )


@app.get("/api/ping")
def ping() -> dict[str, int]:
    return {"server_ms": int(time.time() * 1000)}


@app.get("/api/session-lock")
def public_session_lock() -> dict[str, Any]:
    return {"ok": True, **orchestrator.peek_session_lock()}


@app.post("/api/session-lock/unlock")
def public_session_lock_unlock(payload: SessionLockUnlockRequest) -> dict[str, Any]:
    if not orchestrator.session_lock.verify(payload.password):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "session_lock_invalid_password",
                "message": "召集密码不正确，或当前不在锁定时段。",
                "session_lock": orchestrator.peek_session_lock(),
            },
        )
    return {"ok": True, **orchestrator.peek_session_lock()}


@app.post("/api/enroll")
def public_enroll(payload: PublicEnrollmentRequest, request: Request) -> dict[str, Any]:
    """Create one internal participant ID from the public intake page.

    The access code is returned only to this browser so it can pair the phone
    companion. Administrators cannot call an account-creation endpoint.
    """
    if not settings.public_enrollment_enabled:
        raise HTTPException(
            status_code=403,
            detail="实验暂时结束不接收新的受试者，如果您有兴趣请联系study-contact",
        )

    _require_device_role(request, "study")

    contact_email, contact_phone, contact_wechat = (
        _validated_compensation_contacts(payload)
    )
    try:
        compensation_vault.require_ready()
    except CompensationVaultUnavailable as exc:
        logger.exception("Encrypted enrollment contact vault is unavailable")
        raise HTTPException(
            status_code=503,
            detail="加密联系方式库尚未就绪，暂时无法报名；请联系研究负责人",
        ) from exc

    if settings.instance_id == "formal" and settings.formal_readiness_enforced:
        checks, ready_for_formal = _formal_ready()
        if not ready_for_formal:
            raise HTTPException(status_code=423, detail=_formal_lock_detail(checks))

    if str(payload.intent or "start_now") != "future_book":
        password = _group_session_password(request, payload)
        if orchestrator.session_lock.is_active() and not orchestrator.session_lock.verify(
            password
        ):
            raise HTTPException(
                status_code=423,
                detail={
                    "code": "session_lock_active",
                    "message": (
                        "该时段已锁定集体实验。请在页面底部输入召集密码后再开始，"
                        "或改选「预约未来时间」。"
                    ),
                    "session_lock": orchestrator.peek_session_lock(),
                },
            )

    client_key = hashlib.sha256(
        f"{_request_ip(request)}:{payload.device_id}".encode("utf-8")
    ).hexdigest()
    is_formal = settings.instance_id == "formal"
    with public_enrollment_lock:
        now = time.time()
        attempts = public_enrollment_attempts[client_key]
        while attempts and attempts[0] < now - 1800:
            attempts.popleft()
        if len(attempts) >= 3:
            raise HTTPException(
                status_code=429,
                detail="This browser has created too many study records. Contact WeChat study-contact for help.",
            )
        try:
            assignment = orchestrator.assign_enrollment()
        except OrchestrationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        cohort = str(assignment["cohort"])
        group_name = str(assignment["group_name"])
        if is_formal and cohort not in {"study_a", "study_b"}:
            orchestrator.abandon_enrollment(assignment)
            raise HTTPException(
                status_code=503,
                detail="Public formal enrollment supports Study A/B arms only",
            )
        data_origin = "formal" if is_formal else "pilot"
        try:
            run_slug = db.resolve_enrollment_run_slug(
                settings.public_enrollment_run_slug,
                data_origin=data_origin,
                protocol_mode="strict",
            )
            codes = db.create_participants(
                1,
                cohort,
                group_name,
                preview_mode=False,
                data_origin=data_origin,
                protocol_mode="strict",
                run_slug=run_slug,
                analysis_excluded=not is_formal,
            )
            code = codes[0]
            participant = db.participant_by_code(code)
            if participant is None:
                raise RuntimeError("Enrollment record could not be loaded")
            prior_id = assignment.get("prior_id")
            if prior_id:
                db.assign_prior(participant["id"], str(prior_id), actor="orchestration")
                participant = db.participant(participant["id"]) or participant
            contact_notice = protocol.public_compensation_notice("zh-CN")
            compensation_vault.save_contact(
                run_id=str(participant["run_id"]),
                participant_id=str(participant["id"]),
                email=contact_email,
                phone=contact_phone,
                wechat=contact_wechat,
                notice_version=str(contact_notice["version"]),
                notice_hash=str(contact_notice["sha256"]),
                ip_hash=ip_hash(_request_ip(request), settings),
                user_agent=request.headers.get("user-agent", ""),
            )
            orchestrator.commit_enrollment(
                participant_id=str(participant["id"]),
                assignment=assignment,
            )
            share_appointment = orchestrator.register_share_enrollment(
                str(participant["id"])
            )
        except ValueError as exc:
            orchestrator.abandon_enrollment(assignment)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception:
            orchestrator.abandon_enrollment(assignment)
            raise
        # Only successful record creation consumes an attempt. In particular,
        # transient database errors remain retryable.
        attempts.append(now)
    token = issue_token(participant["id"], "study", settings)
    return {
        "token": token,
        "code": code,
        "code_hint": participant["code_hint"],
        "instance_id": settings.instance_id,
        "reward_cny": 40,
        "help_wechat": "study-contact",
        "orchestration": {
            "mode": assignment.get("mode"),
            "phase": assignment.get("phase"),
            "arm": assignment.get("arm"),
        },
        "share_fit_appointment": share_appointment,
    }


_CHECKPOINT_EVENT_TYPES = (
    "session.interrupted",
    "session.interaction_presence_expired",
    "session.interrupt_continue_chosen",
    "session.interrupt_restart_chosen",
    "phase.start",
    "phase.end",
)


def _participant_interrupt_checkpoint(participant_id: str) -> dict[str, Any] | None:
    completed = bool(db.end_to_end_completion_records([participant_id]))
    return infer_interrupt_checkpoint(
        completed=completed,
        sessions=db.sessions(participant_id),
        events=db.typed_events_for_participant(
            participant_id, list(_CHECKPOINT_EVENT_TYPES)
        ),
    )


def _rate_limit_code_hint_entry(request: Request) -> None:
    now = time.monotonic()
    request_key = ip_hash(_request_ip(request), settings)
    with code_hint_entry_lock:
        attempts = code_hint_entry_attempts[request_key]
        while attempts and attempts[0] < now - 30 * 60:
            attempts.popleft()
        if len(attempts) >= 12:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "code_hint_entry_rate_limited",
                    "message": "尝试次数过多，请稍后再试或联系微信 study-contact。",
                },
            )
        attempts.append(now)


def _code_hint_not_found() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={
            "code": "code_hint_entry_not_found",
            "message": "未找到匹配的编号，请核对四位登录码尾号或联系微信 study-contact。",
        },
    )


@app.post("/api/public/code-hint-entry")
def public_code_hint_entry(
    payload: CodeHintEntryRequest,
    request: Request,
) -> dict[str, Any]:
    """Issue a study token from a 4-character code hint.

    Matches exactly one unfinished interrupted number. The study session may
    still be open, or aborted after logout; the interrupt cursor is kept.
    """
    _require_device_role(request, "study")
    _rate_limit_code_hint_entry(request)
    hint = normalize_code_hint(payload.hint)
    if not is_code_hint(hint):
        raise _code_hint_not_found()
    matches = []
    for participant in db.participants_by_code_hint(hint):
        checkpoint = _participant_interrupt_checkpoint(str(participant["id"]))
        if checkpoint:
            matches.append((participant, checkpoint))
    if len(matches) != 1:
        raise _code_hint_not_found()
    participant, _checkpoint = matches[0]
    token = issue_token(str(participant["id"]), "study", settings)
    full_code = db.participant_login_code(str(participant["id"]))
    return {
        "ok": True,
        "token": token,
        "role": "study",
        "code_hint": participant["code_hint"],
        "code": full_code,
    }


@app.post("/api/login")
def login(payload: LoginRequest, request: Request) -> dict[str, Any]:
    _require_device_role(request, payload.role)
    code = normalize_participant_code(payload.code)
    if not is_full_participant_code(code):
        raise HTTPException(
            status_code=400,
            detail=(
                "请使用完整编号（TTA-XXXX-XXXX）打开专属陪伴链接，"
                "不要只使用四位尾号。"
            ),
        )
    participant = db.participant_by_code(code)
    if participant is None:
        raise HTTPException(
            status_code=401,
            detail=(
                "编号不存在或已停用。请使用研究团队单独发放的完整编号；"
                "不要使用 TTA-XXXX-XXXX、TTA-AUTO-0001 等示例或自行创建编号"
            ),
        )
    checks, is_ready = _require_formal_ready(participant)
    preview_mode = bool(participant.get("preview_mode"))
    pilot_only = participant["data_origin"] != "formal"
    token = issue_token(participant["id"], payload.role, settings)
    return {
        "token": token,
        "role": payload.role,
        "code_hint": participant["code_hint"],
        "neutral_id": participant.get("neutral_id"),
        "preview_mode": preview_mode,
        "formal_ready": is_ready,
        "pilot_only": pilot_only,
        "readiness": checks if pilot_only else None,
    }


@app.post("/api/final-questionnaire/recover")
def recover_final_questionnaire(
    payload: FinalQuestionnaireRecoveryRequest,
    request: Request,
) -> dict[str, Any]:
    """Restore only the participant bound to a final-questionnaire credential."""
    _require_device_role(request, "study")
    now = time.monotonic()
    request_key = ip_hash(_request_ip(request), settings)
    with final_recovery_lock:
        attempts = final_recovery_attempts[request_key]
        while attempts and attempts[0] < now - 30 * 60:
            attempts.popleft()
        if len(attempts) >= 12:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "final_questionnaire_recovery_rate_limited",
                    "message": "尝试次数过多，请稍后再试或联系微信 study-contact。",
                },
            )
        attempts.append(now)
    recovery = db.final_questionnaire_recovery_by_credential(payload.credential)
    if recovery is None:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "final_questionnaire_credential_invalid",
                "message": "最终问卷凭据无效，请核对后重试或联系微信 study-contact。",
            },
        )
    participant = db.participant(str(recovery["participant_id"]))
    if participant is None:
        raise HTTPException(status_code=401, detail="Participant is disabled")
    token = issue_token(participant["id"], "study", settings)
    return {
        "token": token,
        "role": "study",
        "code_hint": participant["code_hint"],
        "session_id": recovery["session_id"],
        "credential": recovery["credential"],
        "final_submitted": recovery.get("final_submitted_ms") is not None,
    }


@app.get("/api/bootstrap")
def bootstrap(
    request: Request,
    lang: str = "zh-CN",
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    participant = auth["participant"]
    participant_study = protocol.study_for(participant)
    consent = db.latest_consent(participant["id"])
    consent_valid = bool(
        consent
        and protocol.valid_consent_signature(
            str(consent["version"]), str(consent["document_hash"])
        )
        and consent["participation"]
    )
    compensation_registration = compensation_vault.status(participant["id"])
    completion_record = db.end_to_end_completion_records(
        [str(participant["id"])]
    ).get(str(participant["id"]))
    interrupt_checkpoint = _participant_interrupt_checkpoint(
        str(participant["id"])
    )
    if interrupt_checkpoint:
        completion_record = None
    final_recovery = db.final_questionnaire_recovery(str(participant["id"]))
    share_chain_committed: bool | None = None
    if (
        final_recovery is not None
        and auth["claims"]["role"] == "study"
        and orchestrator.is_share_fit_member(str(participant["id"]))
    ):
        share_chain_committed = orchestrator.share_chain_committed(
            str(participant["id"])
        )
        if not share_chain_committed:
            # Idempotent recovery for a closeout interrupted between the E2E
            # receipt and the cumulative prior commit.  Missing learned state
            # still fails closed inside finalize_personal_checkpoint; only a
            # provably all-zero network-update participant may be rebuilt from
            # its immutable starting head.
            try:
                model_receipt = timing_service.finalize_personal_checkpoint(
                    participant,
                    int(db.session(
                        str(final_recovery["session_id"]), participant["id"]
                    )["day"]),
                    session_id=str(final_recovery["session_id"]),
                )
                model_ready_ms = int(time.time() * 1000)
                db.update_final_questionnaire_recovery(
                    str(participant["id"]), model_finalized_ms=model_ready_ms
                )
                orchestrator.commit_share_completion(
                    str(participant["id"]), model_receipt
                )
                db.update_final_questionnaire_recovery(
                    str(participant["id"]),
                    chain_committed_ms=int(time.time() * 1000),
                    clear_error=True,
                )
                db.enter_final_questionnaire(
                    str(participant["id"]), str(final_recovery["session_id"])
                )
                timing_service.release(str(participant["id"]))
                realtime_lease_registry.release(str(final_recovery["session_id"]))
                final_recovery = db.final_questionnaire_recovery(
                    str(participant["id"])
                )
                share_chain_committed = True
            except (ModelUnavailable, OrchestrationError, TypeError, ValueError) as exc:
                logger.exception(
                    "Final-questionnaire Share participant still awaits safe chain commit"
                )
                db.update_final_questionnaire_recovery(
                    str(participant["id"]), last_error=str(exc)
                )
    consent_archive = None
    if (
        consent_valid
        and consent is not None
        and completion_record is None
        and not (
            final_recovery is not None
            and final_recovery.get("chain_committed_ms") is not None
        )
    ):
        try:
            consent_archive = _ensure_server_consent_archive(consent, participant)
        except (OSError, TypeError, ValueError) as exc:
            logger.exception("Stored consent does not have a valid server archive")
            raise HTTPException(
                status_code=503,
                detail=(
                    "服务器无法验证知情同意归档，实验不能继续。"
                    "请保留当前编号并联系研究负责人。"
                ),
            ) from exc
    checks, is_ready = _participant_readiness(participant)
    candidate_probes = [
        {
            "probe_token": row["probe_token"],
            "order_index": int(row["order_index"]),
        }
        for row in db.prior_probe_assignments(
            participant_id=participant["id"], day=1
        )
    ]
    rules = protocol.rules_for(participant)
    group_name = str(participant.get("group_name") or "").strip().lower()
    personalisation_phase = (
        "frozen"
        if group_name in {"frozen", "f", "holdout", "holdout_evaluator"}
        else "learning"
    )
    return {
        "instance_id": settings.instance_id,
        "reward_cny": 40,
        "help_wechat": "study-contact",
        "participant": {
            "code_hint": participant["code_hint"],
            "neutral_id": participant.get("neutral_id"),
            "probe_schedule": json.loads(participant["probe_schedule"]),
            "candidate_probes": candidate_probes,
            "preview_mode": bool(participant.get("preview_mode")),
            "analysis_excluded": bool(participant.get("analysis_excluded")),
        },
        "role": auth["claims"]["role"],
        "voice_counterbalance": speech_service.voice_counterbalance(participant["id"]),
        "consent": protocol.public_consent(lang),
        "consent_valid": consent_valid,
        "compensation_registration": protocol.public_compensation_notice(lang),
        "compensation_registration_complete": bool(
            compensation_registration
            and compensation_registration.get("decision") in {"provide", "decline"}
            and compensation_registration.get("contact_complete")
        ),
        "compensation_registration_decision": (
            str(compensation_registration.get("decision"))
            if compensation_registration
            and compensation_registration.get("decision") in {"provide", "decline"}
            else None
        ),
        "compensation_identity_complete": bool(
            compensation_registration
            and compensation_registration.get("decision") == "provide"
        ),
        "compensation_contact_complete": bool(
            compensation_registration
            and compensation_registration.get("contact_complete")
        ),
        "end_to_end_complete": completion_record is not None,
        "share_chain_committed": share_chain_committed,
        "share_chain_commit_pending": bool(
            final_recovery is not None and share_chain_committed is False
        ),
        "final_questionnaire_recovery": (
            {
                "session_id": final_recovery["session_id"],
                "credential": final_recovery["credential"],
                "entered_ms": final_recovery["entered_ms"],
                "model_finalized_ms": final_recovery.get("model_finalized_ms"),
                "chain_committed_ms": final_recovery.get("chain_committed_ms"),
                "final_submitted_ms": final_recovery.get("final_submitted_ms"),
                "last_error": final_recovery.get("last_error"),
                "pending": not bool(final_recovery.get("final_submitted_ms")),
                "model_ready": bool(
                    final_recovery.get("chain_committed_ms")
                    and (
                        share_chain_committed is not False
                        or not orchestrator.is_share_fit_member(
                            str(participant["id"])
                        )
                    )
                ),
            }
            if final_recovery is not None
            else None
        ),
        "completion_receipt": (
            {
                "study_session_id": completion_record["study_session_id"],
                "final_day": completion_record["final_day"],
                "completed_ms": completion_record["completed_ms"],
                "final_response_ms": completion_record["final_response_ms"],
            }
            if completion_record is not None
            else None
        ),
        "consent_archive_saved": bool(
            consent_archive and consent_archive.get("verified")
        ),
        "consent_choices": {
            "raw_audio": bool(consent and consent["raw_audio"]),
            "future_sharing": bool(consent and consent["future_sharing"]),
            "internal_reuse": bool(consent and consent.get("internal_reuse")),
            "controlled_sharing": bool(consent and consent.get("controlled_sharing")),
            "public_derived": bool(consent and consent.get("public_derived")),
            "public_audio": bool(consent and consent.get("public_audio")),
            "ai_training": bool(consent and consent.get("ai_training")),
            "commercial_research": bool(consent and consent.get("commercial_research")),
        },
        # Same participant already typed this on the consent page; reimbursement
        # identity reuse it so the form does not ask for the name again.
        "consent_typed_name": (
            str(consent.get("typed_name") or "").strip()
            if consent_valid and consent is not None
            else ""
        ),
        "progress": db.participant_progress(participant["id"]),
        "formal_ready": is_ready,
        "pilot_only": participant["data_origin"] != "formal",
        "readiness": (
            checks
            if participant["data_origin"] != "formal"
            else None
        ),
        "capacity": _capacity_status(participant_id=str(participant["id"])),
        "share_fit_appointment": orchestrator.share_appointment_status(
            str(participant["id"]),
            group_session_password=_group_session_password(request),
        ),
        "questionnaire_version": protocol.questionnaires["version"],
        "background": protocol.background,
        "client_protocol": {
            "version": participant_study["version"],
            "session_design": participant_study.get("session_design", {}),
            "quality_gate": participant_study.get("quality_gate", {}),
            "topic_gate": participant_study.get("topic_gate", {}),
            "tts_gate": participant_study.get("tts_gate", {}),
            "block_schedule": rules.get("block_schedule", {}),
            "personalisation_phase": personalisation_phase,
        },
        "topic_history": db.selected_topics(participant["id"]),
        "operator_resume": public_operator_resume(
            load_operator_resume(settings.data_dir, str(participant["id"]))
        ),
        "interrupt_checkpoint": public_interrupt_checkpoint(
            interrupt_checkpoint
        ),
    }


@app.get("/api/share-fit/appointment")
def share_fit_appointment_status(
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    return {
        "ok": True,
        "appointment": orchestrator.share_appointment_status(
            str(auth["participant"]["id"]),
            group_session_password=_group_session_password(request),
        ),
    }


@app.post("/api/share-fit/appointment")
def book_share_fit_appointment(
    payload: ShareAppointmentRequest,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    try:
        appointment = orchestrator.book_share_appointment(
            str(auth["participant"]["id"]), payload.scheduled_start_ms
        )
    except ShareAppointmentError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": str(exc),
                "appointment": exc.status,
            },
        ) from exc
    return {"ok": True, "appointment": appointment}


@app.post("/api/share-fit/appointment/arrive")
def claim_share_fit_appointment_arrival(
    payload: ShareAppointmentArrivalRequest,
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    try:
        appointment = orchestrator.claim_share_appointment_arrival(
            str(auth["participant"]["id"]),
            payload.voucher,
            group_session_password=_group_session_password(request, payload),
        )
    except ShareAppointmentError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": str(exc),
                "appointment": exc.status,
            },
        ) from exc
    return {"ok": True, "appointment": appointment}


def _rate_limit_appointment_resume(request: Request) -> None:
    now = time.monotonic()
    request_key = ip_hash(_request_ip(request), settings)
    with appointment_resume_lock:
        attempts = appointment_resume_attempts[request_key]
        while attempts and attempts[0] < now - 30 * 60:
            attempts.popleft()
        if len(attempts) >= 12:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "appointment_resume_rate_limited",
                    "message": "尝试次数过多，请稍后再试或联系微信 study-contact。",
                },
            )
        attempts.append(now)


def _resume_not_found() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={
            "code": "appointment_resume_not_found",
            "message": "未找到匹配的预约。请核对四位凭证、联系方式或姓名，或联系微信 study-contact。",
        },
    )


def _issue_appointment_resume(
    participant_id: str, *, resume_method: str
) -> dict[str, Any]:
    participant = db.participant(participant_id)
    if participant is None:
        raise _resume_not_found()
    appointment = orchestrator.acknowledge_share_appointment_resume(participant_id)
    token = issue_token(participant["id"], "study", settings)
    full_code = db.participant_login_code(str(participant["id"]))
    return {
        "ok": True,
        "resumed": True,
        "resume_method": resume_method,
        "token": token,
        "role": "study",
        "code": full_code,
        "code_hint": participant["code_hint"],
        "instance_id": settings.instance_id,
        "share_fit_appointment": appointment,
    }


@app.post("/api/share-fit/appointment/resume")
def resume_share_fit_appointment(
    payload: AppointmentResumeRequest,
    request: Request,
) -> dict[str, Any]:
    """Restore an already-booked participant from the public landing page."""
    _require_device_role(request, "study")
    _rate_limit_appointment_resume(request)
    voucher = re.sub(r"\D", "", payload.voucher or "")
    full_name = re.sub(r"\s+", " ", payload.full_name.strip())
    contact_email, contact_phone, contact_wechat = _validated_compensation_contacts(
        payload, allow_empty=True
    )
    if voucher and len(voucher) != 4:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "appointment_voucher_invalid",
                "message": "预约凭证必须是四位数字。",
            },
        )
    if not voucher and not full_name and not any(
        (contact_email, contact_phone, contact_wechat)
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "appointment_resume_input_required",
                "message": "请输入四位预约凭证，或填写已登记的联系方式/姓名。",
            },
        )

    if voucher:
        found = orchestrator.find_appointment_by_voucher(voucher)
        if found is None or not found.get("participant_id"):
            raise _resume_not_found()
        if not found.get("live"):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "appointment_resume_expired",
                    "message": "该预约凭证对应的位置已失效，请重新报名或联系微信 study-contact。",
                },
            )
        return _issue_appointment_resume(
            str(found["participant_id"]), resume_method="voucher"
        )

    contact_ids: list[str] = []
    name_ids: list[str] = []
    if any((contact_email, contact_phone, contact_wechat)):
        try:
            compensation_vault.require_ready()
            contact_ids = compensation_vault.match_participant_ids_by_contact(
                email=contact_email,
                phone=contact_phone,
                wechat=contact_wechat,
            )
        except CompensationVaultUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="加密联系方式库尚未就绪，暂时无法核对预约；请联系研究负责人",
            ) from exc
    if full_name:
        name_ids = db.find_participant_ids_by_consent_name(full_name)
        try:
            compensation_vault.require_ready()
            name_ids = list(
                dict.fromkeys(
                    name_ids
                    + compensation_vault.match_participant_ids_by_full_name(full_name)
                )
            )
        except CompensationVaultUnavailable:
            pass

    if contact_ids and name_ids:
        candidates = [item for item in contact_ids if item in set(name_ids)]
    elif contact_ids:
        candidates = contact_ids
    else:
        candidates = name_ids
    candidates = [
        participant_id
        for participant_id in candidates
        if db.participant(participant_id) is not None
    ]
    if not candidates:
        raise _resume_not_found()

    live_ids = orchestrator.live_appointment_participant_ids()
    live_candidates = [
        participant_id for participant_id in candidates if participant_id in live_ids
    ]
    if len(live_candidates) > 1:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "appointment_resume_ambiguous",
                "message": "匹配到多个预约，请补充四位凭证或另一项更具体的联系方式。",
            },
        )
    if len(live_candidates) == 1:
        return _issue_appointment_resume(
            live_candidates[0],
            resume_method="name" if not contact_ids else "contact",
        )
    if full_name and not contact_ids:
        raise _resume_not_found()
    if len(candidates) > 1:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "appointment_resume_ambiguous",
                "message": "匹配到多个报名记录，请补充四位凭证或另一项更具体的联系方式。",
            },
        )
    return _issue_appointment_resume(candidates[0], resume_method="contact")


def _consent_evidence_payload(
    *,
    consent_id: str,
    consent_document: dict[str, Any],
    request_payload: ConsentRequest,
    request: Request,
    auth: dict[str, Any],
    signed_ms: int,
    client_nonce: str,
    ledger_sequence: int,
    previous_record_sha256: str,
    auth_context_sha256: str,
) -> dict[str, Any]:
    participant = auth["participant"]
    choice_labels = {
        str(choice["id"]): str(choice["label"])
        for choice in consent_document.get("choices", [])
        if isinstance(choice, dict) and choice.get("id")
    }
    choices = {
        field: {
            "selected": bool(getattr(request_payload, field)),
            "label": choice_labels.get(field, field),
        }
        for field in CONSENT_CHOICE_FIELDS
    }
    user_agent = request.headers.get("user-agent", "")[:500]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "assurance_scope": ASSURANCE_SCOPE,
        "identity_certification": False,
        "trusted_timestamp": False,
        "instance_id": settings.instance_id,
        "consent_id": consent_id,
        "ledger_sequence": ledger_sequence,
        "previous_record_sha256": previous_record_sha256,
        "run_id": str(participant["run_id"]),
        "participant": {
            "code_hint": str(participant.get("code_hint") or ""),
            "pseudonym_sha256": sha256_text(
                f"{settings.instance_id}:{participant['id']}"
            ),
        },
        "document": {
            "version": request_payload.version,
            "sha256": request_payload.document_hash,
            "ethics_status": normalize_ethics_status(consent_document),
        },
        "choices": choices,
        "typed_name": request_payload.typed_name.strip(),
        "affirmation": request_payload.affirmation.strip(),
        "received": {
            "server_unix_ms": signed_ms,
            "server_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(signed_ms / 1000)
            ),
            "client_nonce": client_nonce,
            "auth_context_sha256": auth_context_sha256,
            "ip_hmac_sha256": ip_hash(_request_ip(request), settings),
            "user_agent_sha256": sha256_text(user_agent),
        },
    }


def _legacy_save_consent(
    payload: ConsentRequest, request: Request, auth: dict[str, Any] = Depends(current_auth)
) -> dict[str, Any]:
    if auth["claims"]["role"] != "study":
        raise HTTPException(status_code=403, detail="Consent must be completed on the study device")
    _require_formal_ready(auth["participant"])
    consent_document = protocol.consent_document(
        payload.version, payload.document_hash
    )
    if consent_document is None:
        raise HTTPException(
            status_code=409,
            detail="知情同意书已更新，请重新载入并完整阅读最新版后再签署",
        )
    if not payload.participation:
        raise HTTPException(status_code=400, detail="Participation consent is required")
    typed_name = " ".join(payload.typed_name.split())
    if len(typed_name) < 2:
        raise HTTPException(
            status_code=400,
            detail="Typed participant name is required under this consent form electronic consent",
        )
    if payload.public_audio and not payload.raw_audio:
        raise HTTPException(
            status_code=400,
            detail="公开语音授权要求先单独同意保存原始音频",
        )
    if payload.affirmation.strip() != consent_document["required_affirmation"]:
        raise HTTPException(status_code=400, detail="Affirmation does not match the displayed statement")
    consent_id = db.save_consent(
        {
            "participant_id": auth["participant"]["id"],
            "run_id": auth["participant"]["run_id"],
            "version": payload.version,
            "document_hash": payload.document_hash,
            "document_json": json.dumps(
                consent_document, ensure_ascii=False, sort_keys=True
            ),
            "participation": payload.participation,
            "raw_audio": payload.raw_audio,
            "future_sharing": payload.future_sharing,
            "internal_reuse": payload.internal_reuse,
            "controlled_sharing": payload.controlled_sharing,
            "public_derived": payload.public_derived,
            "public_audio": payload.public_audio,
            "ai_training": payload.ai_training,
            "commercial_research": payload.commercial_research,
            "typed_name": typed_name,
            "affirmation": payload.affirmation.strip(),
            "signed_ms": int(time.time() * 1000),
            "ip_hash": ip_hash(_request_ip(request), settings),
            "user_agent": request.headers.get("user-agent", "")[:500],
        }
    )
    stored = db.consent(consent_id)
    if stored is None:
        raise HTTPException(
            status_code=500,
            detail="Consent was not readable after the server saved it",
        )
    try:
        archive = _ensure_server_consent_archive(stored, auth["participant"])
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("Server consent archive could not be written")
        raise HTTPException(
            status_code=503,
            detail=(
                "服务器尚未完成知情同意归档，实验不能开始。"
                "请保留当前编号并联系研究负责人。"
            ),
        ) from exc
    return {
        "consent_id": consent_id,
        "server_archive_saved": True,
        "server_archive_record_sha256": archive["record_sha256"],
    }


@app.post("/api/consent")
def save_consent(
    payload: ConsentRequest, request: Request, auth: dict[str, Any] = Depends(current_auth)
) -> dict[str, Any]:
    if auth["claims"]["role"] != "study":
        raise HTTPException(status_code=403, detail="Consent must be completed on the study device")
    _require_formal_ready(auth["participant"])
    consent_document = protocol.consent_document(payload.version, payload.document_hash)
    if consent_document is None:
        raise HTTPException(
            status_code=409,
            detail="知情同意书已更新，请重新载入并完整阅读最新版本后再签署",
        )
    if not payload.participation:
        raise HTTPException(status_code=400, detail="Participation consent is required")
    typed_name = " ".join(payload.typed_name.split())
    if len(typed_name) < 2:
        raise HTTPException(
            status_code=400,
            detail="Typed participant name is required under this consent form electronic consent",
        )
    if payload.public_audio and not payload.raw_audio:
        raise HTTPException(
            status_code=400,
            detail="Public-audio permission requires raw-audio retention permission",
        )
    if payload.affirmation.strip() != consent_document["required_affirmation"]:
        raise HTTPException(
            status_code=400,
            detail="Affirmation does not match the displayed statement",
        )
    try:
        consent_evidence_signer.require_ready()
    except ConsentEvidenceUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Consent evidence signing is not configured",
        ) from exc

    participant = auth["participant"]
    consent_id = str(uuid.uuid4())
    signed_ms = int(time.time() * 1000)
    client_nonce = payload.client_nonce or uuid.uuid4().hex
    user_agent = request.headers.get("user-agent", "")[:500]
    auth_context_sha256 = sha256_text(
        canonical_evidence_json(
            {
                "participant_pseudonym_sha256": sha256_text(
                    f"{settings.instance_id}:{participant['id']}"
                ),
                "role": auth["claims"].get("role"),
                "issued_at": auth["claims"].get("iat"),
                "expires_at": auth["claims"].get("exp"),
                "client_nonce": client_nonce,
                "document_sha256": payload.document_hash,
            }
        )
    )
    # Persist the normalized typed name on the request object used by evidence.
    payload.typed_name = typed_name
    with consent_archive_lock:
        tip = db.consent_chain_tip()
        ledger_sequence = int(tip["ledger_seq"]) + 1
        previous_record_sha256 = str(tip["record_sha256"] or "")
        evidence_payload = _consent_evidence_payload(
            consent_id=consent_id,
            consent_document=consent_document,
            request_payload=payload,
            request=request,
            auth=auth,
            signed_ms=signed_ms,
            client_nonce=client_nonce,
            ledger_sequence=ledger_sequence,
            previous_record_sha256=previous_record_sha256,
            auth_context_sha256=auth_context_sha256,
        )
        evidence_bundle = consent_evidence_signer.sign(evidence_payload)
        evidence_json = canonical_evidence_json(evidence_bundle)
        evidence_sha256 = sha256_text(evidence_json)
        db.save_consent(
            {
                "id": consent_id,
                "participant_id": participant["id"],
                "run_id": participant["run_id"],
                "version": payload.version,
                "document_hash": payload.document_hash,
                "document_json": json.dumps(
                    consent_document, ensure_ascii=False, sort_keys=True
                ),
                "participation": payload.participation,
                "raw_audio": payload.raw_audio,
                "future_sharing": payload.future_sharing,
                "internal_reuse": payload.internal_reuse,
                "controlled_sharing": payload.controlled_sharing,
                "public_derived": payload.public_derived,
                "public_audio": payload.public_audio,
                "ai_training": payload.ai_training,
                "commercial_research": payload.commercial_research,
                "typed_name": typed_name,
                "affirmation": payload.affirmation.strip(),
                "signed_ms": signed_ms,
                "ip_hash": ip_hash(_request_ip(request), settings),
                "user_agent": user_agent,
                "client_nonce": client_nonce,
                "auth_context_sha256": auth_context_sha256,
                "evidence_json": evidence_json,
                "evidence_sha256": evidence_sha256,
                "evidence_key_id": consent_evidence_signer.key_id,
                "ledger_seq": ledger_sequence,
                "previous_record_sha256": previous_record_sha256,
                "record_sha256": evidence_sha256,
                "server_instance": settings.instance_id,
            }
        )
        stored = db.consent(consent_id)
        if stored is None:
            raise HTTPException(
                status_code=500,
                detail="Consent was not readable after the server saved it",
            )
        try:
            archive = _ensure_server_consent_archive_unlocked(stored, participant)
        except (OSError, TypeError, ValueError) as exc:
            logger.exception("Server consent archive could not be written")
            raise HTTPException(
                status_code=503,
                detail=(
                    "The server could not complete the immutable consent archive. "
                    "Keep the current participant code and contact the research team."
                ),
            ) from exc
    return {
        "consent_id": consent_id,
        "server_archive_saved": True,
        "server_archive_record_sha256": archive["record_sha256"],
        "evidence_sha256": evidence_sha256,
        "evidence_key_id": consent_evidence_signer.key_id,
        "ledger_sequence": ledger_sequence,
    }


def _valid_resident_identity_card(value: str) -> bool:
    normalized = value.strip().upper()
    if not re.fullmatch(r"\d{17}[\dX]", normalized):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    expected = checks[
        sum(int(digit) * weight for digit, weight in zip(normalized[:17], weights))
        % 11
    ]
    return normalized[-1] == expected


def _validated_compensation_contacts(
    payload: (
        PublicEnrollmentRequest
        | CompensationIdentityRequest
        | CompensationDeclineRequest
        | CompensationContactRequest
        | AppointmentResumeRequest
    ),
    *,
    allow_empty: bool = False,
) -> tuple[str, str, str]:
    email = payload.contact_email.strip()
    phone = re.sub(r"\s+", " ", payload.contact_phone.strip())
    wechat = payload.contact_wechat.strip()
    if not allow_empty and not any((email, phone, wechat)):
        raise HTTPException(
            status_code=422,
            detail="请至少填写邮箱、电话号码或微信号中的一项联系方式",
        )
    if email and not re.fullmatch(r"[^@\s]{1,64}@[^@\s]{1,189}", email):
        raise HTTPException(status_code=422, detail="邮箱格式无效")
    if phone:
        if not re.fullmatch(r"[+0-9()\-\s]{5,40}", phone) or len(
            re.sub(r"\D", "", phone)
        ) < 5:
            raise HTTPException(status_code=422, detail="电话号码格式无效")
    if any(ord(character) < 32 for character in email + phone + wechat):
        raise HTTPException(status_code=422, detail="联系方式包含无效字符")
    return email, phone, wechat


@app.post("/api/compensation-contact")
def save_compensation_contact(
    payload: CompensationContactRequest,
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403,
            detail="Compensation registration is available only on the study device",
        )
    participant = auth["participant"]
    consent = db.latest_consent(participant["id"])
    if (
        consent is None
        or not consent["participation"]
        or not protocol.valid_consent_signature(
            str(consent["version"]), str(consent["document_hash"])
        )
    ):
        raise HTTPException(
            status_code=403, detail="Please complete the current informed consent first"
        )
    notice = protocol.compensation_notice_document(
        payload.notice_version, payload.notice_hash
    )
    if notice is None:
        raise HTTPException(
            status_code=409,
            detail="补偿联系信息说明已更新，请重新载入并阅读最新版后再登记",
        )
    registration = compensation_vault.status(str(participant["id"]))
    if registration is None:
        raise HTTPException(
            status_code=409,
            detail="请先选择是否提供学校统一报销所需身份资料",
        )
    contact_email, contact_phone, contact_wechat = (
        _validated_compensation_contacts(payload)
    )
    try:
        record_id, created = compensation_vault.save_contact(
            run_id=str(participant["run_id"]),
            participant_id=str(participant["id"]),
            email=contact_email,
            phone=contact_phone,
            wechat=contact_wechat,
            notice_version=payload.notice_version,
            notice_hash=payload.notice_hash,
            ip_hash=ip_hash(_request_ip(request), settings),
            user_agent=request.headers.get("user-agent", ""),
        )
    except CompensationVaultUnavailable as exc:
        logger.exception("Encrypted compensation contact vault is unavailable")
        raise HTTPException(
            status_code=503,
            detail="加密补偿联系库尚未就绪，实验不能开始；请联系研究负责人",
        ) from exc
    return {
        "saved": True,
        "created": created,
        "record_id": record_id,
        "contact_plaintext_returned": False,
    }


@app.post("/api/compensation-identity")
def save_compensation_identity(
    payload: CompensationIdentityRequest,
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403,
            detail="Compensation registration is available only on the study device",
        )
    participant = auth["participant"]
    consent = db.latest_consent(participant["id"])
    if (
        consent is None
        or not consent["participation"]
        or not protocol.valid_consent_signature(
            str(consent["version"]), str(consent["document_hash"])
        )
    ):
        raise HTTPException(
            status_code=403, detail="Please complete the current informed consent first"
        )
    notice = protocol.compensation_notice_document(
        payload.notice_version, payload.notice_hash
    )
    if notice is None:
        raise HTTPException(
            status_code=409,
            detail="补偿身份信息说明已更新，请重新载入并阅读最新版后再登记",
        )
    if (
        not payload.separate_consent
        or payload.affirmation.strip() != notice["separate_affirmation"]
    ):
        raise HTTPException(
            status_code=400,
            detail="必须单独确认补偿身份信息处理说明",
        )
    consent_row = db.latest_consent(participant["id"])
    consent_typed_name = re.sub(
        r"\s+", " ", str((consent_row or {}).get("typed_name") or "").strip()
    )
    # Prefer the consent-form typed name from the previous page; fall back to payload
    # only for legacy clients / tests that still send full_name explicitly.
    full_name = consent_typed_name or re.sub(r"\s+", " ", payload.full_name.strip())
    if len(full_name) < 2:
        raise HTTPException(
            status_code=422,
            detail="缺少知情同意页姓名，请返回上一页重新签署后再登记报销信息",
        )
    document_number = re.sub(r"\s+", "", payload.document_number.strip()).upper()
    if any(ord(character) < 32 for character in full_name + document_number):
        raise HTTPException(status_code=422, detail="身份信息包含无效字符")
    if payload.document_type == "resident_identity_card":
        if not _valid_resident_identity_card(document_number):
            raise HTTPException(status_code=422, detail="居民身份证号校验未通过")
    elif not re.fullmatch(r"[A-Z0-9._()/-]{5,40}", document_number):
        raise HTTPException(status_code=422, detail="证件号格式无效")
    existing_registration = compensation_vault.status(str(participant["id"]))
    contact_already_saved = bool(
        existing_registration and existing_registration.get("contact_complete")
    )
    contact_email, contact_phone, contact_wechat = _validated_compensation_contacts(
        payload,
        allow_empty=contact_already_saved,
    )

    try:
        signature_png = base64.b64decode(
            payload.signature_png_base64.encode("ascii"), validate=True
        )
    except (ValueError, UnicodeEncodeError) as exc:
        raise HTTPException(status_code=422, detail="手写确认图样编码无效") from exc
    if (
        len(signature_png) < 300
        or len(signature_png) > 500_000
        or not signature_png.startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise HTTPException(status_code=422, detail="手写确认图样必须是有效的 PNG")
    try:
        if any((contact_email, contact_phone, contact_wechat)):
            compensation_vault.save_contact(
                run_id=str(participant["run_id"]),
                participant_id=str(participant["id"]),
                email=contact_email,
                phone=contact_phone,
                wechat=contact_wechat,
                notice_version=payload.notice_version,
                notice_hash=payload.notice_hash,
                ip_hash=ip_hash(_request_ip(request), settings),
                user_agent=request.headers.get("user-agent", ""),
            )
        record_id, created = compensation_vault.save(
            run_id=str(participant["run_id"]),
            participant_id=str(participant["id"]),
            document_type=payload.document_type,
            full_name=full_name,
            document_number=document_number,
            signature_png=signature_png,
            notice_version=payload.notice_version,
            notice_hash=payload.notice_hash,
            notice_json=json.dumps(
                notice, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            affirmation=payload.affirmation.strip(),
            separate_consent=payload.separate_consent,
            ip_hash=ip_hash(_request_ip(request), settings),
            user_agent=request.headers.get("user-agent", ""),
        )
    except CompensationVaultUnavailable as exc:
        logger.exception("Encrypted compensation vault is unavailable")
        raise HTTPException(
            status_code=503,
            detail="加密补偿身份库尚未就绪，实验不能开始；请联系研究负责人",
        ) from exc
    return {
        "saved": True,
        "created": created,
        "record_id": record_id,
        "decision": "provide",
        "identity_plaintext_returned": False,
    }


@app.post("/api/compensation-decline")
def decline_compensation_identity(
    payload: CompensationDeclineRequest,
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403,
            detail="Compensation registration is available only on the study device",
        )
    participant = auth["participant"]
    consent = db.latest_consent(participant["id"])
    if (
        consent is None
        or not consent["participation"]
        or not protocol.valid_consent_signature(
            str(consent["version"]), str(consent["document_hash"])
        )
    ):
        raise HTTPException(
            status_code=403, detail="Please complete the current informed consent first"
        )
    notice = protocol.compensation_notice_document(
        payload.notice_version, payload.notice_hash
    )
    if notice is None:
        raise HTTPException(
            status_code=409,
            detail="补偿报销说明已更新，请重新载入并阅读最新版后再选择",
        )
    if payload.affirmation.strip() != notice["decline_affirmation"]:
        raise HTTPException(
            status_code=400,
            detail="必须明确选择不提供补偿身份信息后才能继续",
        )
    existing_registration = compensation_vault.status(str(participant["id"]))
    contact_already_saved = bool(
        existing_registration and existing_registration.get("contact_complete")
    )
    contact_email, contact_phone, contact_wechat = _validated_compensation_contacts(
        payload,
        allow_empty=contact_already_saved,
    )
    try:
        if any((contact_email, contact_phone, contact_wechat)):
            compensation_vault.save_contact(
                run_id=str(participant["run_id"]),
                participant_id=str(participant["id"]),
                email=contact_email,
                phone=contact_phone,
                wechat=contact_wechat,
                notice_version=payload.notice_version,
                notice_hash=payload.notice_hash,
                ip_hash=ip_hash(_request_ip(request), settings),
                user_agent=request.headers.get("user-agent", ""),
            )
        record_id, created = compensation_vault.save_decline(
            run_id=str(participant["run_id"]),
            participant_id=str(participant["id"]),
            notice_version=payload.notice_version,
            notice_hash=payload.notice_hash,
            notice_json=json.dumps(
                notice, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            affirmation=payload.affirmation.strip(),
            ip_hash=ip_hash(_request_ip(request), settings),
            user_agent=request.headers.get("user-agent", ""),
        )
    except CompensationVaultUnavailable as exc:
        logger.exception("Compensation decision vault is unavailable")
        raise HTTPException(
            status_code=503,
            detail="补偿选择记录库尚未就绪，实验不能开始；请联系研究负责人",
        ) from exc
    return {
        "saved": True,
        "created": created,
        "record_id": record_id,
        "decision": "decline",
        "identity_plaintext_returned": False,
    }


def _signed_consent_copy(
    consent: dict[str, Any],
    participant: dict[str, Any],
    *,
    copy_label: str,
) -> str:
    document_json = str(consent.get("document_json") or "")
    if not document_json:
        document = protocol.consent_document(
            str(consent["version"]), str(consent["document_hash"])
        )
        if document is None:
            raise HTTPException(
                status_code=409,
                detail="The exact signed consent document snapshot is unavailable",
            )
    else:
        try:
            document = json.loads(document_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail="Stored consent document snapshot is invalid"
            ) from exc
        canonical = json.dumps(document, ensure_ascii=False, sort_keys=True)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != consent["document_hash"]:
            raise HTTPException(
                status_code=500, detail="Stored consent document snapshot failed integrity check"
            )

    english_document = (
        str(document.get("title") or "") == str(protocol.consents["en"]["title"])
    )
    if english_document:
        copy_label = {
            "参与者留存副本": "Participant copy",
            "研究团队归档副本": "Research-team archive copy",
        }.get(copy_label, copy_label)
    words = {
        "lang": "en" if english_document else "zh-CN",
        "status": "Status notice:" if english_document else "状态提示：",
        "accepted": "Agreed" if english_document else "已同意",
        "declined": "Not agreed" if english_document else "未同意",
        "study_name": "Study:" if english_document else "研究名称：",
        "audit_label": (
            "Electronic-signature verification"
            if english_document
            else "电子签署核验信息"
        ),
        "code_hint": "Participant code ending" if english_document else "参与者编号尾号",
        "signed_at": "Electronic signing time (UTC)" if english_document else "电子签署时间（UTC）",
        "version": "Document version" if english_document else "文件版本",
        "record_id": "Consent record ID" if english_document else "同意记录 ID",
        "hash": "Document SHA-256" if english_document else "文件 SHA-256",
        "choices": "Permission choices" if english_document else "授权选择",
        "typed_name": "Participant name (typed)" if english_document else "参与者姓名（打字）",
        "researcher": "Researcher name" if english_document else "研究人员姓名",
        "affirmation": "Electronic affirmation" if english_document else "电子确认",
        "verification": (
            f"This {copy_label} was generated by the Tap-to-Adapt system from the immutable electronic-signature record in the server's separate consent archive. "
            "The research-team archive copy and participant copy share the same record ID, document version, SHA-256, signing time, and permission choices, allowing the two copies to be verified as identical."
            if english_document
            else (
                f"本{copy_label}由 Tap-to-Adapt 系统根据服务器独立同意库中的不可修改电子签署记录生成。"
                "研究团队归档副本与参与者留存副本对应同一记录 ID、文件版本、SHA-256、签署时间和授权选择；"
                "可据此核验两份内容的一致性。"
            )
        ),
    }
    signed_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(consent["signed_ms"]) / 1000)
    )
    escaped = lambda value: html.escape(str(value), quote=True)
    evidence_audit = ""
    if consent.get("evidence_json"):
        _verified_consent_evidence(consent)
        sequence_label = "Consent ledger sequence" if english_document else "同意证据链序号"
        key_label = (
            "Ed25519 public-key fingerprint" if english_document else "Ed25519 公钥指纹"
        )
        evidence_hash_label = (
            "Evidence bundle SHA-256" if english_document else "证据包 SHA-256"
        )
        evidence_audit = (
            f'<div><strong>{escaped(sequence_label)}</strong>'
            f'{escaped(consent.get("ledger_seq"))}</div>'
            f'<div><strong>{escaped(key_label)}</strong>'
            f'{escaped(consent.get("evidence_key_id"))}</div>'
            f'<div style="grid-column: 1 / -1"><strong>{escaped(evidence_hash_label)}</strong>'
            f'{escaped(consent.get("evidence_sha256"))}</div>'
        )
    sections: list[str] = []
    if document is not None and should_show_document_status_notice(document):
        sections.append(
            f'<aside class="notice"><strong>{escaped(words["status"])}</strong>'
            f"{escaped(document['status_notice'])}</aside>"
        )
    for section in document["sections"]:
        paragraphs = "".join(
            f"<p>{escaped(paragraph)}</p>" for paragraph in section["paragraphs"]
        )
        sections.append(
            f"<section><h2>{escaped(section['heading'])}</h2>{paragraphs}</section>"
        )
    choices: list[str] = []
    for choice in document["choices"]:
        selected = bool(consent.get(str(choice["id"])))
        status_text = words["accepted"] if selected else words["declined"]
        status_class = "accepted" if selected else "declined"
        choices.append(
            '<li class="choice">'
            f'<span class="choice-status {status_class}">[{status_text}]</span>'
            f"<span>{escaped(choice['label'])}</span>"
            "</li>"
        )
    return f"""<!doctype html>
<html lang="{words["lang"]}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped(document["study_name"])} - {escaped(copy_label)}</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: #172027;
      background: #f3f3ef;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      font-size: 14px;
      line-height: 1.72;
    }}
    .document {{
      width: min(900px, calc(100% - 32px));
      margin: 28px auto;
      padding: 54px 62px;
      background: #fff;
      border: 1px solid #d9ddd9;
      box-shadow: 0 10px 32px rgba(26, 42, 40, .08);
    }}
    .copy-label {{
      display: inline-block;
      padding: 5px 10px;
      color: #175b52;
      background: #e6f2ef;
      border: 1px solid #b8d8d1;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .06em;
    }}
    h1 {{ margin: 16px 0 4px; font-size: 28px; line-height: 1.35; }}
    .study-name {{ margin: 0 0 24px; color: #68747a; }}
    .audit {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin: 24px 0 30px;
      border-top: 1px solid #d9ddd9;
      border-left: 1px solid #d9ddd9;
    }}
    .audit div {{
      min-width: 0;
      padding: 10px 12px;
      border-right: 1px solid #d9ddd9;
      border-bottom: 1px solid #d9ddd9;
      overflow-wrap: anywhere;
    }}
    .audit strong {{ display: block; color: #68747a; font-size: 11px; }}
    h2 {{ margin: 26px 0 8px; font-size: 18px; }}
    p {{ margin: 0 0 10px; }}
    .notice {{ padding: 12px 14px; background: #fff4d6; border: 1px solid #e7cf93; }}
    .choices {{ display: grid; gap: 9px; margin: 0; padding: 0; list-style: none; }}
    .choice {{ display: grid; grid-template-columns: 76px 1fr; gap: 10px; }}
    .choice-status {{ font-weight: 800; }}
    .accepted {{ color: #175b52; }}
    .declined {{ color: #7a3e34; }}
    .affirmation {{
      margin-top: 10px;
      padding: 14px 16px;
      background: #f5f7f6;
      border-left: 4px solid #2f756b;
      font-weight: 700;
    }}
    .verification {{
      margin-top: 30px;
      padding-top: 16px;
      color: #56646a;
      border-top: 1px solid #d9ddd9;
      font-size: 12px;
    }}
    @media (max-width: 640px) {{
      .document {{ width: 100%; margin: 0; padding: 28px 20px; border: 0; box-shadow: none; }}
      .audit {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      @page {{ size: A4; margin: 16mm; }}
      body {{ background: #fff; }}
      .document {{ width: auto; margin: 0; padding: 0; border: 0; box-shadow: none; }}
      section, .choice, .audit {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <article class="document">
    <span class="copy-label">{escaped(copy_label)}</span>
    <h1>{escaped(document["title"])}</h1>
    <p class="study-name">{escaped(words["study_name"])} {escaped(document["study_name"])}</p>
    <div class="audit" aria-label="{escaped(words["audit_label"])}">
      <div><strong>{escaped(words["code_hint"])}</strong>{escaped(participant["code_hint"])}</div>
      <div><strong>{escaped(words["signed_at"])}</strong>{escaped(signed_at)}</div>
      <div><strong>{escaped(words["version"])}</strong>{escaped(consent["version"])}</div>
      <div><strong>{escaped(words["record_id"])}</strong>{escaped(consent["id"])}</div>
      <div style="grid-column: 1 / -1"><strong>{escaped(words["hash"])}</strong>{escaped(consent["document_hash"])}</div>
      <div><strong>{escaped(words["typed_name"])}</strong>{escaped(consent.get("typed_name") or "")}</div>
      <div><strong>{escaped(words["researcher"])}</strong>{escaped(
        (document.get("researcher_name_en") if english_document else document.get("researcher_name"))
        or document.get("researcher_name")
        or "Research team / 研究团队"
      )}</div>
      {evidence_audit}
    </div>
    {"".join(sections)}
    <section>
      <h2>{escaped(words["choices"])}</h2>
      <ul class="choices">{"".join(choices)}</ul>
    </section>
    <section>
      <h2>{escaped(words["affirmation"])}</h2>
      <p class="affirmation">{escaped(consent["affirmation"])}</p>
    </section>
    <p class="verification">
      {escaped(words["verification"])}
    </p>
  </article>
</body>
</html>"""


CONSENT_CHOICE_FIELDS = (
    "participation",
    "raw_audio",
    "internal_reuse",
    "future_sharing",
    "controlled_sharing",
    "public_derived",
    "public_audio",
    "ai_training",
    "commercial_research",
)


def _server_consent_archive_dir(
    consent: dict[str, Any],
    participant: dict[str, Any],
) -> Path:
    batch = db.account_batch_for_participant(str(participant["id"]))
    if batch is not None:
        base = _batch_archive_dir(batch) / "consents"
    else:
        run_id = str(consent.get("run_id") or "legacy-unlinked")
        base = settings.data_dir / "consent_archives" / run_id
    target = (
        base
        / str(participant["id"])
        / str(consent["id"])
    ).resolve()
    data_root = settings.data_dir.resolve()
    if not target.is_relative_to(data_root):
        raise ValueError("consent archive path escapes the study data directory")
    return target


def _verified_consent_evidence(consent: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(consent.get("evidence_json") or "")
    if not raw:
        return None
    try:
        bundle = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored consent evidence is invalid JSON") from exc
    result = verify_evidence_bundle(
        bundle,
        expected_key_id=str(consent.get("evidence_key_id") or "") or None,
        expected_instance_id=str(consent.get("server_instance") or "") or None,
    )
    canonical = canonical_evidence_json(bundle)
    if sha256_text(canonical) != str(consent.get("evidence_sha256") or ""):
        raise ValueError("stored consent evidence hash does not match the database")
    payload = bundle["payload"]
    expected = {
        "consent_id": str(consent["id"]),
        "ledger_sequence": int(consent["ledger_seq"]),
        "previous_record_sha256": str(consent.get("previous_record_sha256") or ""),
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise ValueError(f"stored consent evidence {field} does not match")
    if str(payload.get("document", {}).get("sha256") or "") != str(
        consent["document_hash"]
    ):
        raise ValueError("stored consent evidence document hash does not match")
    if str(consent.get("record_sha256") or "") != str(
        consent.get("evidence_sha256") or ""
    ):
        raise ValueError("stored consent chain record hash does not match evidence")
    return {"bundle": bundle, "verification": result, "canonical": canonical}


def _verify_server_consent_archive(
    archive_dir: Path,
    consent: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = archive_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored consent archive manifest is invalid") from exc
    if (
        str(manifest.get("consent_id") or "") != str(consent["id"])
        or str(manifest.get("document_sha256") or "")
        != str(consent["document_hash"])
    ):
        raise ValueError("stored consent archive identity does not match the database")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("stored consent archive manifest has no files")
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("stored consent archive contains an invalid file record")
        relative = Path(str(record.get("path") or ""))
        candidate = (archive_dir / relative).resolve()
        if not candidate.is_relative_to(archive_dir.resolve()):
            raise ValueError("stored consent archive file escapes its archive directory")
        if (
            not candidate.is_file()
            or file_sha256(candidate) != str(record.get("sha256") or "")
        ):
            raise ValueError(
                f"stored consent archive failed integrity verification: {relative}"
            )
    evidence = _verified_consent_evidence(consent)
    if evidence is not None:
        evidence_path = archive_dir / "consent-evidence.json"
        if not evidence_path.is_file():
            raise ValueError("signed consent archive has no evidence bundle")
        try:
            archived_bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("archived consent evidence is invalid") from exc
        if canonical_evidence_json(archived_bundle) != evidence["canonical"]:
            raise ValueError("archived consent evidence differs from the consent ledger")
    return manifest


def _ensure_server_consent_archive_unlocked(
    consent: dict[str, Any],
    participant: dict[str, Any],
) -> dict[str, Any]:
    archive_dir = _server_consent_archive_dir(consent, participant)
    existing = _verify_server_consent_archive(archive_dir, consent)
    if existing is not None:
        return {
            "archive_path": str(archive_dir),
            "manifest_path": str((archive_dir / "MANIFEST.json").resolve()),
            "record_sha256": existing["record_sha256"],
            "verified": True,
        }

    document_json = str(consent.get("document_json") or "")
    document: dict[str, Any] | None = None
    if not document_json:
        document = protocol.consent_document(
            str(consent["version"]), str(consent["document_hash"])
        )
    else:
        try:
            document = json.loads(document_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("signed consent document snapshot is invalid") from exc
    if document is not None:
        canonical_document = json.dumps(document, ensure_ascii=False, sort_keys=True)
        if (
            hashlib.sha256(canonical_document.encode("utf-8")).hexdigest()
            != str(consent["document_hash"])
        ):
            raise ValueError("signed consent document snapshot failed integrity verification")
    evidence = _verified_consent_evidence(consent)

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.parent.parent.chmod(0o700)
    archive_dir.parent.chmod(0o700)
    archive_dir.chmod(0o700)
    choice_labels = {
        str(choice["id"]): str(choice["label"])
        for choice in (document or {}).get("choices", [])
        if isinstance(choice, dict) and choice.get("id")
    }
    choices = {
        field: {
            "selected": bool(consent.get(field)),
            "label": choice_labels.get(field, field),
        }
        for field in CONSENT_CHOICE_FIELDS
    }
    signed_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(consent["signed_ms"]) / 1000)
    )
    record = {
        "schema_version": "tap-to-adapt-server-consent-archive-v2",
        "consent_id": consent["id"],
        "run_id": consent.get("run_id"),
        "participant_id": participant["id"],
        "participant_code_hint": participant.get("code_hint"),
        "participant_record_available": bool(
            participant.get("participant_record_available", True)
        ),
        "document_version": consent["version"],
        "document_sha256": consent["document_hash"],
        "document_ethics_status": (
            normalize_ethics_status(document) if document is not None else None
        ),
        "document_snapshot_available": document is not None,
        "signed_ms": int(consent["signed_ms"]),
        "signed_utc": signed_at,
        "choices": choices,
        "affirmation": consent["affirmation"],
        "ip_hash": consent["ip_hash"],
        "user_agent": consent["user_agent"],
        "storage": "linux_server",
        "evidence": (
            {
                "algorithm": "Ed25519",
                "sha256": consent["evidence_sha256"],
                "key_id": consent["evidence_key_id"],
                "ledger_sequence": int(consent["ledger_seq"]),
                "previous_record_sha256": consent["previous_record_sha256"],
                "assurance_scope": ASSURANCE_SCOPE,
                "identity_certification": False,
            }
            if evidence is not None
            else {"legacy_unsigned_record": True}
        ),
    }
    file_contents = {
        "consent-record.json": json.dumps(record, ensure_ascii=False, indent=2),
    }
    if evidence is not None:
        file_contents["consent-evidence.json"] = json.dumps(
            evidence["bundle"], ensure_ascii=False, indent=2
        )
    if document is not None:
        file_contents["signed-document.json"] = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        )
        file_contents["research-team-copy.html"] = _signed_consent_copy(
            consent,
            participant,
            copy_label="研究团队归档副本",
        )
    file_records = []
    for filename, content in file_contents.items():
        target = archive_dir / filename
        _atomic_text(target, content)
        target.chmod(0o400)
        file_records.append(
            {
                "path": filename,
                "bytes": target.stat().st_size,
                "sha256": file_sha256(target),
            }
        )
    record_path = archive_dir / "consent-record.json"
    manifest = {
        "schema_version": "tap-to-adapt-server-consent-archive-manifest-v2",
        "consent_id": consent["id"],
        "run_id": consent.get("run_id"),
        "participant_id": participant["id"],
        "document_version": consent["version"],
        "document_sha256": consent["document_hash"],
        "document_snapshot_available": document is not None,
        "participant_record_available": bool(
            participant.get("participant_record_available", True)
        ),
        "record_sha256": file_sha256(record_path),
        "signed_ms": int(consent["signed_ms"]),
        "evidence_sha256": str(consent.get("evidence_sha256") or ""),
        "evidence_key_id": str(consent.get("evidence_key_id") or ""),
        "ledger_sequence": consent.get("ledger_seq"),
        "files": file_records,
    }
    manifest_path = archive_dir / "MANIFEST.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    manifest_path.chmod(0o400)
    archive_dir.chmod(0o500)
    verified = _verify_server_consent_archive(archive_dir, consent)
    if verified is None:
        raise ValueError("server consent archive was not created")
    return {
        "archive_path": str(archive_dir),
        "manifest_path": str(manifest_path.resolve()),
        "record_sha256": verified["record_sha256"],
        "verified": True,
    }


def _ensure_server_consent_archive(
    consent: dict[str, Any],
    participant: dict[str, Any],
) -> dict[str, Any]:
    with consent_archive_lock:
        return _ensure_server_consent_archive_unlocked(consent, participant)


def _consent_copy_response(
    consent: dict[str, Any],
    participant: dict[str, Any],
    *,
    copy_label: str,
    filename_prefix: str,
) -> Response:
    body = _signed_consent_copy(
        consent,
        participant,
        copy_label=copy_label,
    )
    filename = f"{filename_prefix}-{participant['code_hint']}.html"
    return Response(
        content=body,
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/consent-copy")
def consent_copy(auth: dict[str, Any] = Depends(current_auth)) -> Response:
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403, detail="Consent copy is available on the study device"
        )
    consent = db.latest_consent(auth["participant"]["id"])
    if consent is None or not consent["participation"]:
        raise HTTPException(status_code=404, detail="No signed consent record exists")
    return _consent_copy_response(
        consent,
        auth["participant"],
        copy_label="参与者留存副本",
        filename_prefix="tap-to-adapt-consent-participant",
    )


@app.get("/api/consent-evidence-key")
def consent_evidence_key() -> dict[str, Any]:
    try:
        return consent_evidence_signer.descriptor()
    except ConsentEvidenceUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Consent evidence signing is not configured"
        ) from exc


def _consent_evidence_response(
    consent: dict[str, Any], participant: dict[str, Any]
) -> Response:
    evidence = _verified_consent_evidence(consent)
    if evidence is None:
        raise HTTPException(
            status_code=404,
            detail="This legacy consent record predates signed evidence bundles",
        )
    filename = f"tap-to-adapt-consent-evidence-{participant['code_hint']}.json"
    return Response(
        content=json.dumps(evidence["bundle"], ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/consent-evidence")
def consent_evidence(auth: dict[str, Any] = Depends(current_auth)) -> Response:
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403, detail="Consent evidence is available on the study device"
        )
    consent = db.latest_consent(auth["participant"]["id"])
    if consent is None or not consent["participation"]:
        raise HTTPException(status_code=404, detail="No signed consent record exists")
    return _consent_evidence_response(consent, auth["participant"])


@app.get("/api/topics")
def topics(
    lang: str = "zh-CN",
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    return protocol.topics_for(lang)


@app.get("/api/questionnaires/{instrument}")
def questionnaire(
    instrument: str,
    lang: str = "zh-CN",
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    questionnaires = protocol.questionnaires_for(lang)
    value = questionnaires["instruments"].get(instrument)
    if value is None:
        raise HTTPException(status_code=404, detail="Questionnaire not found")
    public_value = json.loads(json.dumps(value, ensure_ascii=False))
    for field in public_value.get("fields", []):
        field.pop("correct", None)
    return {
        "version": questionnaires["version"],
        "scale": questionnaires["scale"],
        **public_value,
    }


@app.post("/api/sessions/start")
def start_session(
    payload: SessionRequest,
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    _require_device_role(request, auth["claims"]["role"])
    participant = auth["participant"]
    consent = db.latest_consent(participant["id"])
    if not consent or not consent["participation"]:
        raise HTTPException(status_code=403, detail="Please complete consent first")
    try:
        _ensure_server_consent_archive(consent, participant)
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("Session start blocked by invalid consent archive")
        raise HTTPException(
            status_code=503,
            detail="服务器无法验证知情同意归档，实验不能开始。",
        ) from exc
    compensation_registration = compensation_vault.status(participant["id"])
    if (
        not compensation_registration
        or not compensation_registration.get("contact_complete")
        or compensation_registration.get("decision") not in {"provide", "decline"}
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "compensation_choice_required",
                "message": (
                    "请先留下至少一种联系方式，并选择是否提供学校统一报销所需的身份信息"
                ),
            },
        )
    _require_formal_ready(participant)
    allow_aborted_restart = bool(
        auth["claims"]["role"] == "study"
        and orchestrator.can_restart_aborted_session(str(participant["id"]))
    )
    interrupt_checkpoint = _participant_interrupt_checkpoint(
        str(participant["id"])
    )
    allow_interrupt_reopen = bool(
        interrupt_checkpoint
        and any(
            int(row.get("day") or 0) == payload.day
            and str(row.get("client_role") or "") == "study"
            and str(row.get("state") or "") == "aborted"
            for row in db.sessions(str(participant["id"]))
        )
    )
    try:
        study_state.assert_can_start(
            participant,
            payload.day,
            auth["claims"]["role"],
            allow_aborted_restart=allow_aborted_restart,
            allow_interrupt_reopen=allow_interrupt_reopen,
        )
    except ProtocolViolation as exc:
        raise _protocol_http_error(exc) from exc
    if auth["claims"]["role"] == "study":
        try:
            orchestrator.claim_share_session_start(
                str(participant["id"]),
                group_session_password=_group_session_password(request, payload),
            )
        except ShareAppointmentError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                    "appointment": exc.status,
                },
            ) from exc
    restarted_session = None
    historical_aborted_session = None
    if allow_interrupt_reopen:
        reopened = db.reopen_interrupted_aborted_day(
            str(participant["id"]), payload.day
        )
        if reopened is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "interrupt_reopen_failed",
                    "message": "中断会话未能恢复，请刷新后重试或联系研究团队。",
                },
            )
        db.append_event(
            str(participant["id"]),
            str(reopened["id"]),
            "session.interrupt_reopened",
            None,
            None,
            {
                "server_authoritative": True,
                "reason": "code_hint_or_login_after_abort",
                "keep_timing": True,
                "cursor": interrupt_checkpoint.get("cursor"),
                "stage_key": interrupt_checkpoint.get("stage_key"),
            },
        )
    elif allow_aborted_restart:
        historical_aborted_session = next(
            (
                row
                for row in db.sessions(str(participant["id"]))
                if int(row.get("day") or 0) == payload.day
                and str(row.get("client_role") or "") == "study"
                and str(row.get("state") or "") == "aborted"
            ),
            None,
        )
        restarted_session = db.restart_aborted_session_day(
            str(participant["id"]), payload.day
        )
        if restarted_session is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "aborted_session_restart_failed",
                    "message": "历史实验会话未能安全恢复，请刷新后重试或联系研究团队。",
                },
            )
    session = db.start_session(participant["id"], payload.day, auth["claims"]["role"])
    if restarted_session is not None:
        db.append_event(
            str(participant["id"]),
            str(session["id"]),
            "session.restart_required",
            None,
            None,
            {
                "server_authoritative": True,
                "reason": "verified_appointment_after_historical_abort",
                "appointment_credential_verified": True,
                "historical_session_started_ms": int(
                    (historical_aborted_session or {}).get("started_ms") or 0
                ),
                "historical_session_aborted_ms": int(
                    (historical_aborted_session or {}).get("ended_ms") or 0
                ),
            },
        )
    if auth["claims"]["role"] == "study":
        return {
            **session,
            "share_fit_appointment": orchestrator.share_appointment_status(
                str(participant["id"]),
                group_session_password=_group_session_password(request, payload),
            ),
        }
    return session


@app.get("/api/sessions/{session_id}/protocol-status")
def protocol_status(session_id: str, auth: dict[str, Any] = Depends(current_auth)) -> dict[str, Any]:
    _validate_session(auth, session_id)
    return study_state.completion_report(auth["participant"], session_id)


@app.post("/api/sessions/{session_id}/enter-final-questionnaire")
def enter_final_questionnaire(
    session_id: str,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    """Commit the timed-interaction model, release capacity, then expose the form."""
    _validate_session(auth, session_id)
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403,
            detail="Only the study device can enter the final questionnaire",
        )
    participant = auth["participant"]
    session = db.session(session_id, participant["id"])
    assert session is not None
    timing_service.flush_pending(participant["id"])
    report = study_state.completion_report(
        participant,
        session_id,
        require_final_response=False,
    )
    if not report["complete"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "interaction_protocol_incomplete",
                "message": "服务器尚未确认完整的计时交互，暂不能进入最终问卷。请保持页面打开并重试。",
                "checks": report,
            },
        )

    recovery = db.ensure_final_questionnaire_recovery(
        str(participant["id"]), session_id
    )
    share_member = orchestrator.is_share_fit_member(str(participant["id"]))
    chain_committed = bool(
        share_member
        and orchestrator.share_chain_committed(str(participant["id"]))
    )
    chain_commit: dict[str, Any] | None = None
    try:
        model_finalization: dict[str, Any] | None = None
        if share_member and not chain_committed:
            model_finalization = timing_service.finalize_personal_checkpoint(
                participant,
                int(session["day"]),
                session_id=session_id,
            )
            db.update_final_questionnaire_recovery(
                str(participant["id"]),
                model_finalized_ms=int(time.time() * 1000),
            )

        participant_study = protocol.study_for(participant)
        final_day = int(
            participant_study.get("session_design", {}).get("total_sessions", 3)
        )
        if int(session["day"]) == final_day and requires_locked_pipeline(participant):
            timing_service.finalize_capped_snapshots(
                participant["id"], final_day
            )

        if share_member and not chain_committed:
            assert model_finalization is not None
            chain_commit = orchestrator.commit_share_completion(
                str(participant["id"]), model_finalization
            )
            chain_committed = True
        elif share_member:
            chain_commit = {
                "committed": True,
                "idempotent": True,
                "committed_n": None,
            }

        ready_ms = int(time.time() * 1000)
        recovery = db.update_final_questionnaire_recovery(
            str(participant["id"]),
            model_finalized_ms=(ready_ms if share_member else None),
            chain_committed_ms=ready_ms,
            clear_error=True,
        )
        db.enter_final_questionnaire(str(participant["id"]), session_id)
    except ModelUnavailable as exc:
        db.update_final_questionnaire_recovery(
            str(participant["id"]), last_error=str(exc)
        )
        logger.exception("Share model finalization failed before final questionnaire")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "share_model_finalization_failed",
                "message": "模型尚未安全保存，实验位仍为你保留。请保持页面打开并重试。",
                "error": str(exc),
            },
        ) from exc
    except (OrchestrationError, ValueError) as exc:
        db.update_final_questionnaire_recovery(
            str(participant["id"]), last_error=str(exc)
        )
        logger.exception("Share chain commit failed before final questionnaire")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "share_chain_commit_failed",
                "message": "累计模型尚未安全提交，实验位仍为你保留。请保持页面打开并重试。",
                "error": str(exc),
            },
        ) from exc

    timing_service.release(str(participant["id"]))
    realtime_lease_registry.release(session_id)
    return {
        "entered": True,
        "session_id": session_id,
        "credential": recovery["credential"],
        "entered_ms": recovery["entered_ms"],
        "model_ready": (not share_member) or chain_committed,
        "share_chain_commit": chain_commit,
        "protocol": report,
    }


@app.post("/api/sessions/{session_id}/end")
def end_session(session_id: str, auth: dict[str, Any] = Depends(current_auth)) -> dict[str, Any]:
    _validate_session(auth, session_id)
    participant = auth["participant"]
    session = db.session(session_id, participant["id"])
    assert session is not None
    timing_service.flush_pending(participant["id"])
    report = study_state.completion_report(participant, session_id)
    if not report["complete"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "protocol_incomplete",
                "message": "服务端尚未确认今天的完整实验流程，请按页面提示完成后再提交。",
                "checks": report,
            },
        )
    share_member = bool(
        auth["claims"]["role"] == "study"
        and orchestrator.is_share_fit_member(str(participant["id"]))
    )
    recovery = db.ensure_final_questionnaire_recovery(
        str(participant["id"]), session_id
    )
    db.update_final_questionnaire_recovery(
        str(participant["id"]),
        final_submitted_ms=int(time.time() * 1000),
    )
    chain_already_committed = bool(
        share_member
        and orchestrator.share_chain_committed(str(participant["id"]))
    )
    model_finalization: dict[str, Any] | None = None
    if share_member and not chain_already_committed:
        try:
            model_finalization = timing_service.finalize_personal_checkpoint(
                participant,
                int(session["day"]),
                session_id=session_id,
            )
        except ModelUnavailable as exc:
            logger.exception(
                "Share personal-head finalization failed before protocol closeout"
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "share_model_finalization_failed",
                    "message": (
                        "模型最终权重尚未安全保存，当前位置会继续保留。"
                        "请保持页面打开并重试完成确认。"
                    ),
                    "error": str(exc),
                },
            ) from exc
    participant_study = protocol.study_for(participant)
    final_day = int(
        participant_study.get("session_design", {}).get("total_sessions", 3)
    )
    if int(session["day"]) == final_day and requires_locked_pipeline(participant):
        try:
            timing_service.finalize_capped_snapshots(
                participant["id"], final_day
            )
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    db.enter_final_questionnaire(str(participant["id"]), session_id)
    db.end_session(session_id, participant["id"])
    final_report = study_state.completion_report(participant, session_id)
    completion_record = db.end_to_end_completion_records(
        [str(participant["id"])]
    ).get(str(participant["id"]))
    if completion_record is None:
        logger.error(
            "Completed session did not produce end-to-end completion receipt: "
            "participant=%s session=%s day=%s",
            participant["id"],
            session_id,
            session["day"],
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "end_to_end_completion_not_recorded",
                "message": (
                    "服务器尚未确认端到端完成记录；完成页不会显示。"
                    "请保持本页打开并重试完成确认。"
                ),
            },
        )
    chain_commit: dict[str, Any] | None = None
    if share_member and not chain_already_committed:
        assert model_finalization is not None
        try:
            db.update_final_questionnaire_recovery(
                str(participant["id"]),
                model_finalized_ms=int(time.time() * 1000),
            )
            chain_commit = orchestrator.commit_share_completion(
                str(participant["id"]), model_finalization
            )
            db.update_final_questionnaire_recovery(
                str(participant["id"]),
                chain_committed_ms=int(time.time() * 1000),
                clear_error=True,
            )
        except OrchestrationError as exc:
            db.update_final_questionnaire_recovery(
                str(participant["id"]), last_error=str(exc)
            )
            logger.exception(
                "Share cumulative-prior commit failed after protocol completion"
            )
            # Do not release the timing runtime, realtime lease or serial
            # holder.  A retry is idempotent and must finish this exact donor
            # before any successor can claim M(n).
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "share_chain_commit_failed",
                    "message": (
                        "实验数据已提交，但累计模型尚未安全提交。"
                        "实验位置仍为你保留，请重试完成确认。"
                    ),
                    "error": str(exc),
                },
            ) from exc
    elif share_member:
        chain_commit = {
            "committed": True,
            "idempotent": True,
            "committed_n": None,
        }
    try:
        orchestrator.on_session_completed(str(participant["id"]))
    except Exception:
        logger.exception("Auto-orchestration completion refresh failed")
    timing_service.release(participant["id"])
    realtime_lease_registry.release(session_id)
    return {
        "ended": True,
        "completed": True,
        "end_to_end_complete": True,
        "completion_receipt": {
            "study_session_id": completion_record["study_session_id"],
            "final_day": completion_record["final_day"],
            "completed_ms": completion_record["completed_ms"],
            "final_response_ms": completion_record["final_response_ms"],
        },
        "share_chain_commit": chain_commit,
        "protocol": final_report,
    }


@app.post("/api/participants/me/exit")
def exit_participant_number(
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    """End an unfinished number and immediately release all live capacity.

    This is the server-authoritative counterpart of the participant-facing
    “退出编号” control.  Uploaded research records remain immutable, while
    open study/companion sessions, pair intervals, model runtimes, realtime
    leases, Share appointments and the serial holder are released now rather
    than waiting for an inactivity timeout.
    """

    participant_id = str(auth["participant"]["id"])
    sessions = db.sessions(participant_id)
    open_sessions = [
        row
        for row in sessions
        if str(row.get("state") or "") in {"started", "active"}
        and row.get("ended_ms") is None
    ]
    event_session = next(
        (
            row
            for row in open_sessions
            if str(row.get("client_role") or "") == "study"
        ),
        open_sessions[0] if open_sessions else None,
    )
    db.append_event(
        participant_id,
        str(event_session["id"]) if event_session else None,
        "participant.logout",
        None,
        None,
        {
            "role": str(auth["claims"].get("role") or ""),
            "open_session_count": len(open_sessions),
            "immediate_release_requested": True,
        },
    )

    aborted_days: set[int] = set()
    aborted_session_ids: list[str] = []
    for session in open_sessions:
        day = int(session["day"])
        if day in aborted_days:
            continue
        db.abort_session(str(session["id"]), participant_id)
        aborted_days.add(day)
        aborted_session_ids.extend(
            str(row["id"])
            for row in open_sessions
            if int(row["day"]) == day
        )

    release = orchestrator.on_participant_exit(participant_id)
    timing_service.release(participant_id)
    chat_service.clear_participant(participant_id)
    for session in sessions:
        realtime_lease_registry.release(str(session["id"]))
    recovery = db.final_questionnaire_recovery(participant_id)
    return {
        "exited": True,
        "immediate_release_confirmed": True,
        "aborted_session_ids": list(dict.fromkeys(aborted_session_ids)),
        "share_release": release,
        "final_questionnaire_recoverable": bool(
            recovery and recovery.get("final_submitted_ms") is None
        ),
    }


@app.get("/api/participants/me/portable-bundle")
def download_portable_bundle(
    auth: dict[str, Any] = Depends(current_auth),
) -> StreamingResponse:
    """Download only the authenticated participant's fully completed state."""
    if str(auth["claims"].get("role") or "") != "study":
        raise HTTPException(status_code=403, detail="Study-device session required")
    participant = auth["participant"]
    participant_id = str(participant["id"])
    completion = db.end_to_end_completion_records([participant_id]).get(
        participant_id
    )
    if completion is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "portable_bundle_requires_completion",
                "message": "完成全部实验流程并提交最终问卷后才能下载。",
            },
        )
    try:
        payload = _portable_bundle_for_participant(participant, completion)
    except (ModelUnavailable, OSError, TypeError, ValueError) as exc:
        logger.exception("Participant portable bundle could not be generated")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "portable_bundle_unavailable",
                "message": "个性化模型数据包暂时无法生成，请稍后重试。",
            },
        ) from exc
    db.append_event(
        participant_id,
        str(completion["study_session_id"]),
        "participant.portable_bundle_downloaded",
        None,
        None,
        {"completed": True, "bytes": len(payload)},
    )
    filename = f"tap-to-adapt-personal-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, private",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/api/sessions/{session_id}/abort")
def abort_session(session_id: str, auth: dict[str, Any] = Depends(current_auth)) -> dict[str, bool]:
    _validate_session(auth, session_id)
    if auth["claims"]["role"] != "study":
        raise HTTPException(status_code=403, detail="Only the study device can end early")
    db.abort_session(session_id, auth["participant"]["id"])
    orchestrator.on_session_aborted(str(auth["participant"]["id"]))
    timing_service.release(auth["participant"]["id"])
    realtime_lease_registry.release(session_id)
    return {"ended": True, "completed": False}


class ResetInteractionRequest(BaseModel):
    reason: str = "restart_after_interrupt"


@app.post("/api/sessions/{session_id}/reset-interaction-state")
def reset_interaction_state(
    session_id: str,
    payload: ResetInteractionRequest,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    _validate_session(auth, session_id)
    if auth["claims"]["role"] != "study":
        raise HTTPException(status_code=403, detail="Only the study device can reset interaction state")
    participant = auth["participant"]
    chat_service.clear_participant(str(participant["id"]))
    try:
        timing_service.release(str(participant["id"]))
    except Exception:
        logger.exception("timing release during interaction reset failed")
    realtime_lease_registry.release(session_id)
    db.append_event(
        str(participant["id"]),
        session_id,
        "interaction.reset",
        None,
        None,
        {"reason": payload.reason, "capacity_released": True},
    )
    return {
        "ok": True,
        "cleared": True,
        "capacity_released": True,
        "capacity": _capacity_status(participant_id=str(participant["id"])),
    }


@app.post("/api/events")
def save_event(payload: EventRequest, auth: dict[str, Any] = Depends(current_auth)) -> dict[str, Any]:
    _validate_session(auth, payload.session_id)
    if len(json.dumps(payload.payload, ensure_ascii=False)) > 32_000:
        raise HTTPException(status_code=413, detail="Event payload too large")
    try:
        study_state.assert_events_allowed(
            auth["participant"],
            str(payload.session_id),
            [payload.model_dump(exclude={"session_id"})],
        )
    except ProtocolViolation as exc:
        raise _protocol_http_error(exc) from exc
    return db.append_event(
        auth["participant"]["id"],
        payload.session_id,
        payload.event_type,
        payload.client_ms,
        payload.monotonic_ms,
        payload.payload,
    )


@app.post("/api/events/batch")
def save_event_batch(
    payload: EventBatchRequest, auth: dict[str, Any] = Depends(current_auth)
) -> dict[str, Any]:
    _validate_session(auth, payload.session_id)
    events = [event.model_dump(exclude={"session_id"}) for event in payload.events]
    if len(json.dumps(events, ensure_ascii=False)) > 256_000:
        raise HTTPException(status_code=413, detail="Event batch too large")
    try:
        study_state.assert_events_allowed(
            auth["participant"], str(payload.session_id), events
        )
    except ProtocolViolation as exc:
        raise _protocol_http_error(exc) from exc
    saved = db.append_events(auth["participant"]["id"], payload.session_id, events)
    return {"saved": len(saved), "last": saved[-1]}


@app.post("/api/quality")
def save_quality(payload: QualityRequest, auth: dict[str, Any] = Depends(current_auth)) -> dict[str, Any]:
    _validate_session(auth, payload.session_id)
    session_id = str(payload.session_id)
    if bool(payload.metrics.get("simulation")) and auth["participant"]["data_origin"] != "simulation":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "simulation_quality_forbidden",
                "message": "只有 simulation 账号可以使用合成质量检测数据。",
            },
        )
    participant_study = protocol.study_for(auth["participant"])
    metrics = dict(payload.metrics)
    attempt_id = str(metrics.get("attempt_id") or "").strip()
    if not attempt_id or len(attempt_id) > 100:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "quality_payload_invalid",
                "message": "检测记录无效，请刷新页面后重新检测。",
            },
        )
    existing_attempts: dict[str, dict[str, Any]] = {}
    for row in db.quality_history(auth["participant"]["id"], payload.kind):
        if str(row.get("session_id") or "") != session_id:
            continue
        previous_metrics = json.loads(row["metrics"])
        previous_attempt_id = str(previous_metrics.get("attempt_id") or "")
        if previous_attempt_id:
            existing_attempts.setdefault(previous_attempt_id, row)
    if attempt_id in existing_attempts:
        previous = existing_attempts[attempt_id]
        previous_metrics = json.loads(previous["metrics"])
        return {
            "saved": False,
            "passed": bool(previous["passed"]),
            "attempt_id": attempt_id,
            "idempotent": True,
            "server_decision": previous_metrics.get("server_decision", {}),
        }
    expected_number = len(existing_attempts) + 1
    metrics["client_attempt_number"] = int(metrics.get("attempt_number") or expected_number)
    metrics["attempt_number"] = expected_number
    metrics["attempt_id"] = attempt_id
    metrics["client_claimed_passed"] = bool(payload.passed)
    try:
        passed, metrics = evaluate_quality(payload.kind, metrics, participant_study)
    except QualityPayloadError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "quality_payload_invalid", "message": str(exc)},
        ) from exc
    db.save_quality(
        auth["participant"]["id"], payload.session_id, payload.kind, passed, metrics
    )
    return {
        "saved": True,
        "passed": passed,
        "attempt_id": attempt_id,
        "attempt_number": expected_number,
        "server_decision": metrics.get("server_decision", {}),
    }


@app.post("/api/asr/transcribe")
async def transcribe_short_pcm(
    payload: ASRTranscriptionRequest,
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403,
            detail="Server ASR is provided through the study computer",
        )
    if payload.context != "preflight" and not payload.session_id:
        raise HTTPException(status_code=422, detail="Runtime ASR requires a session")
    _validate_session(auth, payload.session_id)
    if not settings.server_asr_enabled:
        raise HTTPException(status_code=503, detail="Server ASR is disabled")
    try:
        pcm16 = base64.b64decode(payload.pcm16_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="PCM payload is not valid base64") from exc
    if not pcm16 or len(pcm16) % 2:
        raise HTTPException(status_code=422, detail="PCM payload is empty or incomplete")
    duration_seconds = (len(pcm16) // 2) / 16_000.0
    if duration_seconds < 0.2 or duration_seconds > settings.server_asr_max_seconds + 0.05:
        raise HTTPException(status_code=422, detail="PCM duration is outside the accepted range")
    generation = _begin_asr_turn(payload.session_id)
    cancel = threading.Event()

    async def _watch_disconnect() -> None:
        try:
            while not cancel.is_set():
                if await request.is_disconnected():
                    cancel.set()
                    return
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return

    def cancel_check() -> bool:
        return cancel.is_set() or not _asr_turn_is_current(payload.session_id, generation)

    watcher = asyncio.create_task(_watch_disconnect())
    try:
        result = await asyncio.to_thread(
            server_asr_service.transcribe_pcm16,
            pcm16,
            sample_rate_hz=payload.sample_rate_hz,
            language=payload.language,
            cancel_check=cancel_check,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ServerASRCancelled:
        raise HTTPException(status_code=409, detail="ASR cancelled") from None
    except ServerASRUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        cancel.set()
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
    return {
        "text": result.text,
        "source": "server_whisper",
        "model": result.model,
        "language": result.language,
        "duration_seconds": result.duration_seconds,
        "latency_ms": result.latency_ms,
        "signal_rms": result.signal_rms,
        "audio_persisted": False,
    }


def _questionnaire_schema_name(instrument: str) -> str:
    if instrument == "baseline":
        return "baseline"
    if instrument.startswith("daily_day_"):
        return "daily"
    if instrument == "final":
        return "final"
    raise ValueError("Questionnaire instrument is not recognized")


def _questionnaire_answer_snapshot(
    instrument: str,
    answers: dict[str, Any],
    locale: str,
    submitted_order: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a localized, immutable snapshot of closed questionnaire answers."""
    catalog = protocol.questionnaires_for(locale)
    schema = catalog["instruments"][_questionnaire_schema_name(instrument)]
    scale = catalog["scale"]
    entries: dict[str, tuple[str, dict[str, Any]]] = {}
    for item in schema.get("items", []):
        entries[f"item:{item['id']}"] = ("likert", item)
    for field in schema.get("fields", []):
        entries[f"field:{field['id']}"] = ("choice", field)
    for item in schema.get("open_items", []):
        entries[f"open:{item['id']}"] = ("open", item)

    canonical_order = list(entries)
    order = [str(value) for value in submitted_order] or canonical_order
    if len(order) != len(set(order)) or set(order) != set(canonical_order):
        raise ValueError("Questionnaire presentation order is invalid")

    result: list[dict[str, Any]] = []
    for position, key in enumerate(order, start=1):
        kind, item = entries[key]
        item_id = str(item["id"])
        if kind == "open" or item_id not in answers:
            continue
        answer_value = answers[item_id]
        if kind == "choice":
            option = next(
                (
                    candidate
                    for candidate in item.get("options", [])
                    if str(candidate["value"]) == str(answer_value)
                ),
                None,
            )
            if option is None:
                raise ValueError(f"Questionnaire option is invalid: {item_id}")
            answer_text = str(option["label"])
            scale_context = None
        else:
            answer_text = str(answer_value)
            scale_context = {
                "min": int(scale["min"]),
                "min_label": str(scale["min_label"]),
                "max": int(scale["max"]),
                "max_label": str(scale["max_label"]),
            }
        result.append(
            {
                "position": position,
                "question_id": item_id,
                "question_text": str(item["text"]),
                "answer_value": answer_value,
                "answer_text": answer_text,
                "answer_kind": kind,
                "scale": scale_context,
            }
        )
    return result, order


@app.post("/api/responses")
def save_response(payload: ResponseRequest, auth: dict[str, Any] = Depends(current_auth)) -> dict[str, bool]:
    _validate_session(auth, payload.session_id)
    session = db.session(str(payload.session_id), auth["participant"]["id"])
    if session is None or session["client_role"] != "study":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "study_session_required",
                "message": "问卷提交前必须先创建电脑端实验会话。请刷新页面，并按页面流程重新进入。",
            },
        )
    if len(json.dumps(payload.answers, ensure_ascii=False)) > 128_000:
        raise HTTPException(status_code=413, detail="Response payload too large")
    participant_study = protocol.study_for(auth["participant"])
    final_day = int(
        participant_study.get("session_design", {}).get("total_sessions", 3)
    )
    try:
        answers = protocol.validate_response(
            payload.instrument,
            payload.version,
            payload.answers,
            int(session["day"]),
            final_day,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "questionnaire_invalid", "message": str(exc)},
        ) from exc
    try:
        answer_labels, question_order = _questionnaire_answer_snapshot(
            payload.instrument,
            answers,
            payload.locale,
            payload.question_order,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "questionnaire_presentation_invalid", "message": str(exc)},
        ) from exc
    # Multi-day legacy runs still require the same-day daily instrument before
    # final. Single-session formal flow (total_sessions == 1) goes straight to
    # the final questionnaire and must not block on a daily response.
    if (
        payload.instrument == "final"
        and final_day > 1
        and not db.response_exists(
            auth["participant"]["id"], payload.session_id, f"daily_day_{final_day}"
        )
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "daily_required", "message": "请先提交本次实验使用感受问卷。"},
        )
    db.save_response(
        auth["participant"]["id"],
        payload.session_id,
        payload.instrument,
        payload.version,
        answers,
        answer_labels,
        question_order,
        payload.locale,
    )
    if payload.instrument == "final":
        recovery = db.final_questionnaire_recovery(
            str(auth["participant"]["id"])
        )
        if recovery is not None:
            db.update_final_questionnaire_recovery(
                str(auth["participant"]["id"]),
                final_submitted_ms=int(time.time() * 1000),
                clear_error=True,
            )
    return {"saved": True}


@app.post("/api/chat")
async def chat(payload: ChatRequest, auth: dict[str, Any] = Depends(current_auth)) -> dict[str, Any]:
    _validate_session(auth, payload.session_id)
    expected = "companion" if payload.source == "companion" else "study"
    if auth["claims"]["role"] != expected:
        raise HTTPException(status_code=403, detail="Chat source does not match this device")
    if payload.source == "primary":
        try:
            study_state.assert_primary_chat_allowed(auth["participant"], payload.session_id)
        except ProtocolViolation as exc:
            raise _protocol_http_error(exc) from exc
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            chat_service.reply, auth["participant"]["id"], payload.source, payload.transcript
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    db.append_event(
        auth["participant"]["id"],
        payload.session_id,
        "chat.turn",
        None,
        None,
        {
            "source": payload.source,
            "transcript": payload.transcript,
            "reply": result["text"],
            "mode": result["mode"],
            "model": result["model"],
            "latency_ms": result["latency_ms"],
        },
    )
    return result


@app.post("/api/reply-needed")
async def reply_needed(
    payload: ReplyNeededRequest,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    _validate_session(auth, payload.session_id)
    if auth["claims"]["role"] != "study":
        raise HTTPException(
            status_code=403,
            detail="Reply-needed classification is limited to the study device",
        )
    if reply_needed_model is None:
        raise HTTPException(
            status_code=503,
            detail="Reply-needed context model is unavailable",
        )
    try:
        result = await asyncio.to_thread(
            reply_needed_model.predict, payload.transcript
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.append_event(
        auth["participant"]["id"],
        payload.session_id,
        "model.reply_needed",
        None,
        None,
        {
            "text_sha256": result["text_sha256"],
            "transcript_chars": len(payload.transcript.strip()),
            "probability": result["probability"],
            "prediction": result["prediction"],
            "threshold": result["threshold"],
            "model_version": result["version"],
            "artifact_sha256": result["artifact_sha256"],
            "recognized_features": result["recognized_features"],
        },
    )
    return result


@app.post("/api/tts")
async def synthesize_speech(
    payload: SpeechRequest, auth: dict[str, Any] = Depends(current_auth)
) -> Response:
    _validate_session(auth, payload.session_id)
    participant = auth["participant"]
    client_role = str(auth["claims"]["role"])
    assignment = speech_service.voice_assignment_for(participant["id"])
    voice = speech_service.voice_for(client_role, participant_id=participant["id"])
    started = time.perf_counter()
    try:
        audio = await asyncio.to_thread(
            speech_service.synthesize, payload.text, voice=voice
        )
    except SpeechSynthesisError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    db.append_event(
        auth["participant"]["id"],
        payload.session_id,
        "tts.generated",
        None,
        None,
        {
            "provider": speech_service.provider,
            "voice": voice,
            "voice_assignment": assignment,
            "speed": speech_service.speed,
            "sample_rate_hz": speech_service.sample_rate_hz,
            "format": speech_service.output_format,
            "fallback": False,
            "text_length": len(payload.text),
            "bytes": len(audio),
            "latency_ms": latency_ms,
        },
    )
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "X-TTS-Latency-Ms": str(latency_ms),
            "X-TTS-Provider": speech_service.provider,
            "X-TTS-Voice": voice,
            "X-TTS-Sample-Rate": str(speech_service.sample_rate_hz),
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/simulation/tts-pcm")
async def synthesize_simulation_pcm(
    payload: SimulationSpeechRequest,
    auth: dict[str, Any] = Depends(current_auth),
) -> Response:
    _validate_session(auth, payload.session_id)
    participant = auth["participant"]
    if (
        participant.get("data_origin") != "simulation"
        or not bool(participant.get("preview_mode"))
    ):
        raise HTTPException(
            status_code=403,
            detail="Multi-voice PCM injection is restricted to simulation preview accounts",
        )
    started = time.perf_counter()
    try:
        if payload.source == "participant":
            if payload.speaker_id is None:
                raise SpeechSynthesisError("Participant simulation TTS needs a speaker ID")
            audio = await asyncio.to_thread(
                speech_service.synthesize_pcm,
                payload.text,
                speaker_id=payload.speaker_id,
                speed=payload.speed,
            )
            voice = f"aishell3-sid{payload.speaker_id}"
            provider = "sherpa_onnx_vits_aishell3"
        else:
            if auth["claims"]["role"] != "companion":
                raise SpeechSynthesisError(
                    "Companion simulation TTS requires the companion session"
                )
            assignment = speech_service.voice_assignment_for(participant["id"])
            voice = speech_service.voice_for(
                "companion", participant_id=participant["id"]
            )
            audio = await asyncio.to_thread(
                speech_service.synthesize_agent_pcm,
                payload.text,
                client_role="companion",
                participant_id=participant["id"],
            )
            provider = speech_service.provider
    except SpeechSynthesisError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    db.append_event(
        participant["id"],
        payload.session_id,
        "simulation.tts_pcm_generated",
        None,
        None,
        {
            "provider": provider,
            "voice": voice,
            "voice_assignment": (
                assignment if payload.source == "companion" else "participant_voice"
            ),
            "speed": payload.speed,
            "sample_rate_hz": speech_service.sample_rate_hz,
            "format": "pcm_s16le",
            "minimum_streaming_samples": speech_service.MINIMUM_STREAMING_PCM_SAMPLES,
            "short_utterance_policy": "right_zero_pad_only_no_stretch_no_repeat",
            "text_length": len(payload.text),
            "bytes": len(audio),
            "latency_ms": latency_ms,
            "simulation": True,
        },
    )
    return Response(
        content=audio,
        media_type="application/octet-stream",
        headers={
            "X-TTS-Latency-Ms": str(latency_ms),
            "X-TTS-Provider": provider,
            "X-TTS-Voice": voice,
            "X-TTS-Sample-Rate": str(speech_service.sample_rate_hz),
            "X-Audio-Format": "pcm_s16le",
            "Cache-Control": "no-store",
        },
    )


CONTENT_TYPE_EXTENSIONS = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",
}


@app.post("/api/audio/{session_id}/{kind}")
async def upload_audio(
    session_id: str,
    kind: str,
    request: Request,
    auth: dict[str, Any] = Depends(current_auth),
) -> dict[str, Any]:
    _validate_session(auth, session_id)
    consent = db.latest_consent(auth["participant"]["id"])
    if not consent or not consent["raw_audio"]:
        raise HTTPException(status_code=403, detail="Raw-audio retention was not authorized")
    if not kind.replace("_", "").replace("-", "").isalnum() or len(kind) > 40:
        raise HTTPException(status_code=400, detail="Invalid audio kind")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    extension = CONTENT_TYPE_EXTENSIONS.get(content_type)
    if extension is None:
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    body = await request.body()
    if not body or len(body) > settings.max_audio_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio upload is empty or too large")
    pre_experiment_batch = db.account_batch_for_participant(
        auth["participant"]["id"]
    )
    if pre_experiment_batch is not None:
        participant_dir = (
            _batch_archive_dir(pre_experiment_batch)
            / "audio"
            / auth["participant"]["id"]
            / session_id
        )
    else:
        participant_dir = settings.media_dir / auth["participant"]["id"] / session_id
    participant_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time() * 1000)}-{uuid.uuid4().hex}{extension}"
    target = participant_dir / filename
    target.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    relative = str(target.relative_to(settings.data_dir))
    audio_id = db.save_audio_record(
        auth["participant"]["id"], session_id, kind, relative, content_type, len(body), digest
    )
    return {"audio_id": audio_id, "bytes": len(body), "sha256": digest}


PAIR_HEARTBEAT_TIMEOUT_MS = 30_000
# Keep a just-qualified pair alive across brief WeChat/WebView blips so the
# timing socket does not emit pair_required and eject an in-progress session.
# Qualified grace, desktop disconnect grace, and deferred interval close must
# stay aligned so a 30s companion reopen is seamless.
PAIR_QUALIFIED_GRACE_MS = 30_000
PAIR_INTERVAL_CLOSE_GRACE_MS = 35_000
_WEBSOCKET_DISCONNECT_ERRORS = (
    "WebSocket is not connected",
    "Cannot call \"receive\" once a disconnect message has been received",
    "Unexpected ASGI message 'websocket.send'",
)


def _is_websocket_disconnect_error(error: RuntimeError) -> bool:
    return any(marker in str(error) for marker in _WEBSOCKET_DISCONNECT_ERRORS)


def _is_model_resource_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message and (
        "cuda" in message
        or "accelerator" in error.__class__.__name__.lower()
        or "memoryallocation" in message
    )


TRANSPORT_PROBE_PROTOCOL_VERSION = "transport-probe-v1"
TRANSPORT_PROBE_HELLO_TIMEOUT_SECONDS = 5.0
TRANSPORT_PROBE_IDLE_TIMEOUT_SECONDS = 10.0
TRANSPORT_PROBE_MAX_CONNECTION_SECONDS = 75.0
TRANSPORT_PROBE_MAX_CONNECTIONS_PER_IP = 6
TRANSPORT_PROBE_TIMING_MESSAGES_PER_SECOND = 15
TRANSPORT_PROBE_PAIR_MESSAGES_PER_SECOND = 4
TRANSPORT_PROBE_HELLO_MAX_BYTES = 512
TRANSPORT_PROBE_PAIR_MAX_BYTES = 256
TRANSPORT_PROBE_PCM_BYTES = 3200
TRANSPORT_PROBE_BINARY_HEADER = struct.Struct("!Id")
TRANSPORT_PROBE_BINARY_BYTES = (
    TRANSPORT_PROBE_BINARY_HEADER.size + TRANSPORT_PROBE_PCM_BYTES
)
TRANSPORT_PROBE_CHANNELS = frozenset(
    {"timing", "study_pair", "companion_pair"}
)
_TRANSPORT_PROBE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,63}")


@dataclass(frozen=True)
class TransportProbeIdentity:
    probe_id: str
    user_index: int
    channel: str


class TransportProbeRegistry:
    """Bound unauthenticated probe resources without touching study storage."""

    def __init__(self) -> None:
        self.connections_per_ip: dict[str, int] = {}
        self.identities: set[TransportProbeIdentity] = set()
        self.lock = asyncio.Lock()

    async def acquire_ip(self, client_ip: str) -> bool:
        async with self.lock:
            current = self.connections_per_ip.get(client_ip, 0)
            if current >= TRANSPORT_PROBE_MAX_CONNECTIONS_PER_IP:
                return False
            self.connections_per_ip[client_ip] = current + 1
            return True

    async def bind_identity(self, identity: TransportProbeIdentity) -> bool:
        async with self.lock:
            if identity in self.identities:
                return False
            self.identities.add(identity)
            return True

    async def release(
        self,
        client_ip: str,
        identity: TransportProbeIdentity | None,
    ) -> None:
        async with self.lock:
            current = self.connections_per_ip.get(client_ip, 0)
            if current <= 1:
                self.connections_per_ip.pop(client_ip, None)
            else:
                self.connections_per_ip[client_ip] = current - 1
            if identity is not None:
                self.identities.discard(identity)


transport_probe_registry = TransportProbeRegistry()


class TransportProbeViolation(ValueError):
    def __init__(self, error_code: str, close_code: int, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.close_code = close_code
        self.message = message


def _websocket_ip(websocket: WebSocket) -> str:
    direct = websocket.client.host if websocket.client else "unknown"
    if direct in {"127.0.0.1", "::1"}:
        forwarded = websocket.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    return direct


def _transport_probe_wall_ms() -> float:
    return time.time_ns() / 1_000_000


def _transport_probe_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransportProbeViolation(
            "invalid_message", 4400, f"{field_name} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 1_000_000_000_000_000:
        raise TransportProbeViolation(
            "invalid_message", 4400, f"{field_name} must be a finite timestamp"
        )
    return result


def _transport_probe_sequence(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 0xFFFFFFFF
    ):
        raise TransportProbeViolation(
            "invalid_message", 4400, "seq must be an unsigned 32-bit integer"
        )
    return value


def _transport_probe_hello(message: dict[str, Any]) -> TransportProbeIdentity:
    required = {"type", "probe_id", "user_index", "channel"}
    if set(message) != required or message.get("type") != "hello":
        raise TransportProbeViolation(
            "invalid_hello", 4400, "The first frame must be a valid probe hello"
        )
    probe_id = message.get("probe_id")
    if not isinstance(probe_id, str) or _TRANSPORT_PROBE_ID_PATTERN.fullmatch(probe_id) is None:
        raise TransportProbeViolation(
            "invalid_hello", 4400, "probe_id has an invalid format"
        )
    user_index = message.get("user_index")
    if isinstance(user_index, bool) or not isinstance(user_index, int) or user_index not in {0, 1}:
        raise TransportProbeViolation(
            "invalid_hello", 4400, "user_index must be 0 or 1"
        )
    channel = message.get("channel")
    if not isinstance(channel, str) or channel not in TRANSPORT_PROBE_CHANNELS:
        raise TransportProbeViolation(
            "invalid_hello", 4400, "channel is not a supported probe role"
        )
    return TransportProbeIdentity(probe_id, user_index, channel)


def _transport_probe_rate_allowed(
    arrivals: deque[float], now: float, maximum: int
) -> bool:
    cutoff = now - 1.0
    while arrivals and arrivals[0] <= cutoff:
        arrivals.popleft()
    if len(arrivals) >= maximum:
        return False
    arrivals.append(now)
    return True


async def _close_transport_probe(
    websocket: WebSocket,
    violation: TransportProbeViolation,
) -> None:
    try:
        await websocket.send_json(
            {
                "type": "probe_error",
                "code": violation.error_code,
                "message": violation.message,
            }
        )
        await websocket.close(code=violation.close_code, reason=violation.error_code)
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        if not _is_websocket_disconnect_error(exc):
            raise


@dataclass
class PairConnection:
    websocket: WebSocket
    session_id: str
    day: int
    last_seen_ms: int
    ready: bool = False


class PairHub:
    def __init__(self):
        self.connections: dict[str, dict[str, PairConnection]] = defaultdict(dict)
        self.active_intervals: dict[str, tuple[str, str, str]] = {}
        self.qualified_intervals: set[str] = set()
        self.last_qualified_ok_ms: dict[str, int] = {}
        # Once a companion session announces ready, keep it sticky across brief
        # reconnects so desktop setup does not flicker ready↔not-ready.
        self.companion_ready_sessions: dict[str, str] = {}
        self._deferred_close_tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()

    def _fresh(self, connection: PairConnection, now_ms: int) -> bool:
        return now_ms - connection.last_seen_ms <= PAIR_HEARTBEAT_TIMEOUT_MS

    def _cancel_deferred_close_locked(self, participant_id: str) -> None:
        task = self._deferred_close_tasks.pop(participant_id, None)
        if task is not None and not task.done():
            task.cancel()

    def _arm_deferred_interval_close_locked(self, participant_id: str) -> None:
        existing = self._deferred_close_tasks.get(participant_id)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._deferred_close_tasks[participant_id] = loop.create_task(
            self._deferred_close_interval(participant_id)
        )

    async def _deferred_close_interval(self, participant_id: str) -> None:
        try:
            await asyncio.sleep(PAIR_INTERVAL_CLOSE_GRACE_MS / 1000)
        except asyncio.CancelledError:
            return
        should_broadcast = False
        async with self.lock:
            if self._deferred_close_tasks.get(participant_id) is not asyncio.current_task():
                return
            status = self._status_locked(participant_id)
            if status.get("same_day_online"):
                self._deferred_close_tasks.pop(participant_id, None)
                return
            active = self.active_intervals.get(participant_id)
            if active is None:
                self._deferred_close_tasks.pop(participant_id, None)
                return
            db.close_pair_interval(active[0], "heartbeat_timeout_or_disconnect")
            self.qualified_intervals.discard(active[0])
            self.active_intervals.pop(participant_id, None)
            self.last_qualified_ok_ms.pop(participant_id, None)
            self._deferred_close_tasks.pop(participant_id, None)
            should_broadcast = True
        if should_broadcast:
            await self.broadcast_status(participant_id)

    def _status_locked(self, participant_id: str) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        connections = self.connections.get(participant_id, {})
        fresh = {
            role: connection
            for role, connection in connections.items()
            if self._fresh(connection, now_ms)
        }
        study = fresh.get("study")
        companion = fresh.get("companion")
        same_day_online = bool(study and companion and study.day == companion.day)
        sticky_ready = bool(
            companion is not None
            and self.companion_ready_sessions.get(participant_id) == companion.session_id
        )
        companion_ready = bool(
            same_day_online
            and companion is not None
            and (companion.ready or sticky_ready)
        )
        if companion_ready and companion is not None and not companion.ready:
            companion.ready = True
        return {
            "type": "pair_status",
            "roles": sorted(fresh),
            "same_day_online": same_day_online,
            "day": study.day if same_day_online and study is not None else None,
            "companion_ready": companion_ready,
            "study_session_id": study.session_id if same_day_online and study else None,
            "companion_session_id": companion.session_id if same_day_online and companion else None,
            "server_ms": now_ms,
        }

    @staticmethod
    def _status_signature(status: dict[str, Any]) -> tuple[Any, ...]:
        return (
            tuple(status.get("roles") or ()),
            bool(status.get("same_day_online")),
            status.get("day"),
            bool(status.get("companion_ready")),
            status.get("study_session_id"),
            status.get("companion_session_id"),
        )

    def _persist_role_presence(
        self,
        participant_id: str,
        role: str,
        last_seen_ms: int,
        *,
        companion_ready: bool | None = None,
    ) -> None:
        active = self.active_intervals.get(participant_id)
        if active is None:
            return
        db.update_pair_role_presence(
            active[0],
            role,
            last_seen_ms,
            companion_ready=companion_ready,
        )

    def _persist_interval_presence_locked(self, participant_id: str) -> None:
        active = self.active_intervals.get(participant_id)
        if active is None:
            return
        now_ms = int(time.time() * 1000)
        connections = self.connections.get(participant_id, {})
        study = connections.get("study")
        companion = connections.get("companion")
        if study is not None and self._fresh(study, now_ms):
            self._persist_role_presence(participant_id, "study", study.last_seen_ms)
        if companion is not None and self._fresh(companion, now_ms):
            # Never persist ready=False from presence sync — reconnect storms were
            # wiping companion_ready in SQLite and making /ws/timing reject with 403.
            self._persist_role_presence(
                participant_id,
                "companion",
                companion.last_seen_ms,
                companion_ready=True if companion.ready else None,
            )

    def _db_companion_ready_locked(
        self, participant_id: str, companion_session_id: str
    ) -> bool:
        """Read sticky companion_ready from the open interval (ignore heartbeat age)."""
        active = self.active_intervals.get(participant_id)
        study_session_id = str(active[1]) if active is not None else None
        try:
            return db.pair_interval_companion_ready(
                participant_id,
                companion_session_id,
                study_session_id=study_session_id,
            )
        except Exception:  # noqa: BLE001 - presence restore must stay best-effort
            return False

    def _activation_snapshot(
        self, participant_id: str, study_session_id: str, *, require_qualified: bool
    ) -> dict[str, Any] | None:
        interval = db.activation_ready_interval(
            participant_id,
            study_session_id,
            timeout_ms=PAIR_HEARTBEAT_TIMEOUT_MS,
            require_qualified=require_qualified,
        )
        if interval is None:
            return None
        return {
            "type": "pair_status",
            "same_day_online": True,
            "companion_ready": True,
            "day": int(interval["day"]),
            "study_session_id": study_session_id,
            "companion_session_id": str(interval["companion_session_id"]),
            "pair_interval_id": str(interval["id"]),
            "server_ms": int(time.time() * 1000),
        }

    def _sync_interval_locked(self, participant_id: str) -> None:
        status = self._status_locked(participant_id)
        active = self.active_intervals.get(participant_id)
        desired = None
        if status["same_day_online"]:
            desired = (
                str(status["study_session_id"]),
                str(status["companion_session_id"]),
                int(status["day"]),
            )
            self._cancel_deferred_close_locked(participant_id)
        if active is not None and (desired is None or active[1:] != desired[:2]):
            # Brief companion blips must not tear down a qualified interval; that
            # made /ws/timing emit pair_required and kick participants to setup.
            if desired is None and active[0] in self.qualified_intervals:
                self._arm_deferred_interval_close_locked(participant_id)
                self._persist_interval_presence_locked(participant_id)
                return
            reason = "heartbeat_timeout_or_disconnect" if desired is None else "session_replaced"
            self._cancel_deferred_close_locked(participant_id)
            db.close_pair_interval(active[0], reason)
            self.qualified_intervals.discard(active[0])
            self.active_intervals.pop(participant_id, None)
            self.last_qualified_ok_ms.pop(participant_id, None)
            active = None
        if desired is not None and active is None:
            study_session_id, companion_session_id, day = desired
            interval_id = db.open_pair_interval(
                participant_id, day, study_session_id, companion_session_id
            )
            self.active_intervals[participant_id] = (
                interval_id,
                study_session_id,
                companion_session_id,
            )
            active = self.active_intervals[participant_id]
        if (
            desired is not None
            and active is not None
            and status["companion_ready"]
            and active[0] not in self.qualified_intervals
        ):
            study_session_id, companion_session_id, _day = desired
            resumed = db.qualify_reconnected_pair_interval(
                active[0],
                participant_id,
                study_session_id,
                companion_session_id,
            )
            if resumed is not None:
                self.qualified_intervals.add(active[0])
                db.append_event(
                    participant_id,
                    study_session_id,
                    "pair.qualified_resume",
                    None,
                    None,
                    {
                        "day": int(resumed["day"]),
                        "study_session_id": study_session_id,
                        "companion_session_id": companion_session_id,
                        "pair_interval_id": active[0],
                        "heartbeat_timeout_ms": PAIR_HEARTBEAT_TIMEOUT_MS,
                    },
                )
        self._persist_interval_presence_locked(participant_id)

    async def connect(
        self,
        participant_id: str,
        role: str,
        session: dict[str, Any],
        websocket: WebSocket,
    ) -> bool:
        """Attach a pair socket. Newer connections replace older ones for the same role."""
        await websocket.accept()
        async with self.lock:
            old = self.connections[participant_id].get(role)
            # Keep companion readiness across WeChat/WebView reconnects for the
            # same session. Also restore from DB so a brief ready=False socket
            # cannot clear an already-enabled phone.
            session_id = str(session["id"])
            preserve_ready = bool(
                role == "companion"
                and (
                    (
                        old is not None
                        and old.ready
                        and str(old.session_id) == session_id
                    )
                    or self.companion_ready_sessions.get(participant_id) == session_id
                    or self._db_companion_ready_locked(participant_id, session_id)
                )
            )
            self.connections[participant_id][role] = PairConnection(
                websocket=websocket,
                session_id=session_id,
                day=int(session["day"]),
                last_seen_ms=int(time.time() * 1000),
                ready=preserve_ready,
            )
            if role == "companion" and preserve_ready:
                self.companion_ready_sessions[participant_id] = session_id
                try:
                    db.upsert_companion_presence(
                        participant_id,
                        session_id,
                        int(session["day"]),
                        ready=True,
                        last_seen_ms=int(time.time() * 1000),
                    )
                except Exception:  # noqa: BLE001 - presence is a start-gate fallback
                    pass
            self._sync_interval_locked(participant_id)
            connection = self.connections[participant_id][role]
            self._persist_role_presence(
                participant_id,
                role,
                connection.last_seen_ms,
                # Only promote ready on connect; never demote here.
                companion_ready=True if role == "companion" and connection.ready else None,
            )
        if old is not None and old.websocket is not websocket:
            try:
                await old.websocket.close(
                    code=4001, reason="A newer connection replaced this device"
                )
            except RuntimeError as exc:
                if not _is_websocket_disconnect_error(exc):
                    raise
        await self.broadcast_status(participant_id)
        return True

    async def _remove_socket(self, participant_id: str, websocket: WebSocket) -> bool:
        removed = False
        async with self.lock:
            connections = self.connections.get(participant_id, {})
            for role, current in list(connections.items()):
                if current.websocket is websocket:
                    connections.pop(role, None)
                    removed = True
            if not connections:
                self.connections.pop(participant_id, None)
            if removed:
                self._sync_interval_locked(participant_id)
        return removed

    async def disconnect(self, participant_id: str, role: str, websocket: WebSocket) -> None:
        del role
        if await self._remove_socket(participant_id, websocket):
            await self.broadcast_status(participant_id)

    async def broadcast_status(self, participant_id: str) -> None:
        async with self.lock:
            self._sync_interval_locked(participant_id)
            message = self._status_locked(participant_id)
            sockets = [
                connection.websocket
                for connection in self.connections.get(participant_id, {}).values()
            ]
        failed = []
        for socket in sockets:
            try:
                await socket.send_json(message)
            except (RuntimeError, WebSocketDisconnect):
                failed.append(socket)
        removed = False
        for socket in failed:
            removed = await self._remove_socket(participant_id, socket) or removed
        if removed:
            await self.broadcast_status(participant_id)

    async def touch(
        self, participant_id: str, role: str, websocket: WebSocket, message: dict[str, Any]
    ) -> None:
        should_broadcast = False
        async with self.lock:
            connection = self.connections.get(participant_id, {}).get(role)
            if connection is None or connection.websocket is not websocket:
                return
            before = self._status_signature(self._status_locked(participant_id))
            connection.last_seen_ms = int(time.time() * 1000)
            companion_ready = None
            if (
                role == "companion"
                and message.get("type") == "background_state"
                and message.get("action") == "ready"
            ):
                # Only promote readiness. Ignoring ready:false stops reconnect /
                # duplicate-tab races from flipping desktop setup UI every second.
                if bool(message.get("ready")):
                    connection.ready = True
                    self.companion_ready_sessions[participant_id] = connection.session_id
                    companion_ready = True
                    try:
                        db.upsert_companion_presence(
                            participant_id,
                            connection.session_id,
                            connection.day,
                            ready=True,
                            last_seen_ms=connection.last_seen_ms,
                        )
                    except Exception:  # noqa: BLE001 - presence is a start-gate fallback
                        pass
            self._sync_interval_locked(participant_id)
            self._persist_role_presence(
                participant_id,
                role,
                connection.last_seen_ms,
                companion_ready=companion_ready,
            )
            after = self._status_signature(self._status_locked(participant_id))
            # Heartbeats used to broadcast on every touch and made the desktop
            # setup steps flicker between “连接手机” and “开始实验”.
            should_broadcast = before != after
        if should_broadcast:
            await self.broadcast_status(participant_id)

    async def activation_pair(
        self, participant_id: str, study_session_id: str
    ) -> dict[str, Any] | None:
        return self._activation_snapshot(
            participant_id,
            study_session_id,
            require_qualified=True,
        )

    async def live_ready_pair(
        self, participant_id: str, study_session_id: str
    ) -> dict[str, Any] | None:
        """Require a currently live ready phone for this study session.

        Prefer in-process pair sockets. When those are empty — the production
        split puts `/ws/pair` on the realtime listener while `/activate` is
        served by the HTTP listener — fall back to a fresh open pair_interval
        in SQLite (both heartbeats inside PAIR_HEARTBEAT_TIMEOUT_MS and
        companion_ready). That DB row is written by the realtime PairHub and is
        the cross-process live signal.
        """
        async with self.lock:
            status = self._status_locked(participant_id)
            companion = self.connections.get(participant_id, {}).get("companion")
            live_ok = bool(
                status.get("same_day_online")
                and str(status.get("study_session_id")) == str(study_session_id)
                and status.get("companion_ready")
                and companion is not None
                and companion.ready
            )
            if live_ok:
                return {
                    "day": status.get("day"),
                    "study_session_id": status.get("study_session_id"),
                    "companion_session_id": status.get("companion_session_id"),
                }
        snap = self._activation_snapshot(
            participant_id,
            study_session_id,
            require_qualified=False,
        )
        if snap is not None:
            return {
                "day": snap.get("day"),
                "study_session_id": study_session_id,
                "companion_session_id": snap["companion_session_id"],
            }
        presence = db.live_companion_presence(
            participant_id, timeout_ms=PAIR_HEARTBEAT_TIMEOUT_MS
        )
        if presence is None:
            return None
        study = db.session(study_session_id, participant_id)
        if (
            study is None
            or study.get("client_role") != "study"
            or int(study["day"]) != int(presence["day"])
        ):
            return None
        return {
            "day": int(presence["day"]),
            "study_session_id": study_session_id,
            "companion_session_id": str(presence["companion_session_id"]),
        }

    async def activate_pair(
        self, participant_id: str, study_session_id: str
    ) -> dict[str, Any] | None:
        # DB heartbeats alone used to let desktop start after the phone left;
        # always require a live companion WebSocket that has announced ready.
        live = await self.live_ready_pair(participant_id, study_session_id)
        if live is None:
            return None
        status = self._activation_snapshot(
            participant_id,
            study_session_id,
            require_qualified=False,
        )
        if status is None:
            db.bind_live_companion_for_activation(
                participant_id,
                study_session_id,
                str(live["companion_session_id"]),
                int(live["day"]),
            )
            status = self._activation_snapshot(
                participant_id,
                study_session_id,
                require_qualified=False,
            )
        if status is None:
            return None
        interval_id = str(status["pair_interval_id"])
        try:
            activated = db.activate_paired_sessions(
                participant_id,
                study_session_id,
                str(status["companion_session_id"]),
                max_concurrent=int(settings.max_concurrent_study_sessions),
            )
            db.qualify_pair_interval(
                interval_id,
                participant_id,
                study_session_id,
                str(status["companion_session_id"]),
                int(activated["activated_ms"]),
            )
        except StudyCapacityExceeded:
            raise
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        async with self.lock:
            self.qualified_intervals.add(interval_id)
        return {**activated, "pair_interval_id": interval_id}

    async def is_qualified(self, participant_id: str, study_session_id: str) -> bool:
        now_ms = int(time.time() * 1000)
        if await self.activation_pair(participant_id, study_session_id) is not None:
            self.last_qualified_ok_ms[participant_id] = now_ms
            return True
        # Heal a wiped companion_ready bit when the live companion is already ready.
        async with self.lock:
            status = self._status_locked(participant_id)
            companion = self.connections.get(participant_id, {}).get("companion")
            if (
                status.get("same_day_online")
                and str(status.get("study_session_id")) == str(study_session_id)
                and companion is not None
                and companion.ready
            ):
                self._persist_role_presence(
                    participant_id,
                    "companion",
                    companion.last_seen_ms,
                    companion_ready=True,
                )
        if await self.activation_pair(participant_id, study_session_id) is not None:
            self.last_qualified_ok_ms[participant_id] = now_ms
            return True
        last_ok = self.last_qualified_ok_ms.get(participant_id)
        return last_ok is not None and now_ms - last_ok <= PAIR_QUALIFIED_GRACE_MS

    async def relay(self, participant_id: str, source_role: str, message: dict[str, Any]) -> None:
        target_role = "companion" if source_role == "study" else "study"
        async with self.lock:
            target = self.connections.get(participant_id, {}).get(target_role)
        if target is not None:
            try:
                await target.websocket.send_json({**message, "from": source_role})
            except (RuntimeError, WebSocketDisconnect):
                if await self._remove_socket(participant_id, target.websocket):
                    await self.broadcast_status(participant_id)


pair_hub = PairHub()


@dataclass
class TimingConnection:
    websocket: WebSocket
    generation: int


@dataclass
class TimingSessionOwnership:
    generation: int = 0
    connection: TimingConnection | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TimingHub:
    """Linearize timing mutations and keep one socket owner per study session."""

    def __init__(self) -> None:
        self.sessions: dict[str, TimingSessionOwnership] = {}
        self.lock = asyncio.Lock()

    async def _ownership(self, session_id: str) -> TimingSessionOwnership:
        async with self.lock:
            return self.sessions.setdefault(session_id, TimingSessionOwnership())

    async def connect(self, session_id: str, websocket: WebSocket) -> int:
        ownership = await self._ownership(session_id)
        async with ownership.operation_lock:
            async with self.lock:
                old = ownership.connection
                ownership.generation += 1
                generation = ownership.generation
                ownership.connection = TimingConnection(
                    websocket=websocket,
                    generation=generation,
                )
        if old is not None and old.websocket is not websocket:
            try:
                await old.websocket.close(
                    code=4001,
                    reason="A newer timing connection replaced this session",
                )
            except RuntimeError as exc:
                if not _is_websocket_disconnect_error(exc):
                    raise
        return generation

    @asynccontextmanager
    async def operation(
        self,
        session_id: str,
        websocket: WebSocket,
        generation: int,
    ) -> AsyncIterator[bool]:
        ownership = await self._ownership(session_id)
        async with ownership.operation_lock:
            async with self.lock:
                current = ownership.connection
                is_current = bool(
                    current is not None
                    and current.websocket is websocket
                    and current.generation == generation
                )
            yield is_current

    async def is_current(
        self,
        session_id: str,
        websocket: WebSocket,
        generation: int,
    ) -> bool:
        ownership = await self._ownership(session_id)
        async with self.lock:
            current = ownership.connection
            return bool(
                current is not None
                and current.websocket is websocket
                and current.generation == generation
            )

    async def execute(
        self,
        session_id: str,
        websocket: WebSocket,
        generation: int,
        operation: Callable[[], Awaitable[Any]],
    ) -> tuple[bool, Any]:
        """Run one runtime mutation while connection replacement is excluded."""
        ownership = await self._ownership(session_id)
        async with ownership.operation_lock:
            async with self.lock:
                current = ownership.connection
                is_current = bool(
                    current is not None
                    and current.websocket is websocket
                    and current.generation == generation
                )
            if not is_current:
                return False, None
            return True, await operation()

    async def disconnect(
        self,
        session_id: str,
        websocket: WebSocket,
        generation: int,
    ) -> None:
        ownership = await self._ownership(session_id)
        async with ownership.operation_lock:
            async with self.lock:
                current = ownership.connection
                if (
                    current is not None
                    and current.websocket is websocket
                    and current.generation == generation
                ):
                    ownership.connection = None


timing_hub = TimingHub()


async def _send_timing_error(
    websocket: WebSocket,
    *,
    code: str,
    message: str,
    command: str | None = None,
    tap_id: str | None = None,
    **details: Any,
) -> None:
    await websocket.send_json(
        {
            "type": "timing_error",
            "code": code,
            "message": message,
            "command": command,
            "tap_id": tap_id,
            "server_ms": int(time.time() * 1000),
            **details,
        }
    )


@app.post("/api/sessions/{session_id}/companion-ready")
async def mark_companion_ready(
    session_id: str, auth: dict[str, Any] = Depends(current_auth)
) -> dict[str, Any]:
    participant = auth["participant"]
    session = db.session(session_id, participant["id"])
    if auth["claims"]["role"] != "companion" or session is None or session["client_role"] != "companion":
        raise HTTPException(status_code=403, detail="Only the companion phone can announce ready")
    presence = db.upsert_companion_presence(
        str(participant["id"]),
        str(session["id"]),
        int(session["day"]),
        ready=True,
    )
    return {
        "companion_ready": True,
        "companion_session_id": str(presence["companion_session_id"]),
        "day": int(presence["day"]),
        "last_seen_ms": int(presence["last_seen_ms"]),
    }


@app.get("/api/sessions/{session_id}/pair-live")
async def study_pair_live(
    session_id: str, auth: dict[str, Any] = Depends(current_auth)
) -> dict[str, Any]:
    participant = auth["participant"]
    session = db.session(session_id, participant["id"])
    if auth["claims"]["role"] != "study" or session is None or session["client_role"] != "study":
        raise HTTPException(status_code=403, detail="Only the study device can read pair liveness")
    live = await pair_hub.live_ready_pair(str(participant["id"]), str(session_id))
    return {
        "companion_ready": live is not None,
        "same_day_online": live is not None,
        "day": (live or {}).get("day") or session.get("day"),
        "study_session_id": session_id,
        "companion_session_id": (live or {}).get("companion_session_id"),
    }


@app.post("/api/sessions/{session_id}/activate")
async def activate_session_pair(
    session_id: str, auth: dict[str, Any] = Depends(current_auth)
) -> dict[str, Any]:
    participant = auth["participant"]
    session = db.session(session_id, participant["id"])
    if auth["claims"]["role"] != "study" or session is None or session["client_role"] != "study":
        raise HTTPException(status_code=403, detail="Only the study device can activate a session")
    _require_study_capacity(
        participant_id=str(participant["id"]),
        session_id=str(session_id),
    )
    pair_missing_detail = (
        "无法启动：同编号电脑端与手机端须同时在线，且手机端已完成检测并启用声音。"
        "仅输入同一编号不够——请用电脑页上的专属陪伴链接打开手机端，"
        "完成检测后点击启用声音，确认手机显示已连接/就绪后再开始。"
    )
    # Always require a LIVE phone companion first. This also covers already-active
    # study sessions, which used to skip the pair check and re-enter the timed flow.
    live = await pair_hub.live_ready_pair(participant["id"], session_id)
    if live is None:
        raise HTTPException(status_code=409, detail=pair_missing_detail)
    already_active = session["state"] == "active" or session.get("activated_ms") is not None
    if already_active:
        try:
            orchestrator.require_share_activation(str(participant["id"]))
        except ShareAppointmentError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                    "appointment": exc.status,
                },
            ) from exc
        return {
            "study_session_id": session_id,
            "companion_session_id": live["companion_session_id"],
            "day": live.get("day") or session.get("day"),
            "activated_ms": session.get("activated_ms"),
            "pair_qualified": True,
            "live_pair_reconfirmed": True,
            "realtime_routing": {
                "ws_path_prefix": "/ws",
                "listener_role": settings.listener_role,
            },
            "capacity": _capacity_status(
                participant_id=str(participant["id"]),
                session_id=str(session_id),
            ),
        }
    try:
        study_state.assert_can_activate(participant, session_id)
    except ProtocolViolation as exc:
        raise _protocol_http_error(exc) from exc
    # Construct the participant runtime before committing either paired
    # session to the active state.  A model/configuration error must remain a
    # recoverable start failure; otherwise a rejected timing WebSocket leaves
    # the participant with an irreversibly activated but never-started run.
    if settings.loads_timing_model_on_demand:
        try:
            await asyncio.to_thread(
                timing_service.runtime,
                participant,
                int(session["day"]),
                session_id=str(session["id"]),
            )
        except ModelUnavailable as exc:
            logger.warning(
                "Timing runtime preflight failed: session=%s reason=%s",
                session_id,
                exc,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "timing_model_unavailable",
                    "message": "计时模型暂时不可用，实验尚未开始；请稍后重试或联系研究团队。",
                },
            ) from exc
    try:
        orchestrator.require_share_activation(str(participant["id"]))
    except ShareAppointmentError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": str(exc),
                "appointment": exc.status,
            },
        ) from exc
    try:
        activated = await pair_hub.activate_pair(participant["id"], session_id)
    except StudyCapacityExceeded as exc:
        capacity = _capacity_status()
        raise HTTPException(
            status_code=429,
            detail={
                "code": "study_capacity_full",
                "message": capacity["message_zh"],
                "message_en": capacity["message_en"],
                "retry_after_minutes": capacity["retry_after_minutes"],
                "active_count": capacity["active_count"],
                "max_concurrent": capacity["max_concurrent"],
                "capacity_error": str(exc),
            },
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if message == "This day already has a different qualified device pair":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "qualified_device_pair_conflict",
                    "message": "同一天已经绑定了另一组电脑与手机会话，请刷新页面；若仍无法开始，请联系研究人员。",
                },
            ) from exc
        raise HTTPException(status_code=409, detail=message) from exc
    if activated is None:
        raise HTTPException(status_code=409, detail=pair_missing_detail)
    db.append_event(
        participant["id"],
        session_id,
        "pair.qualified_start",
        None,
        None,
        {
            "day": activated["day"],
            "study_session_id": session_id,
            "companion_session_id": activated["companion_session_id"],
            "pair_interval_id": activated["pair_interval_id"],
            "heartbeat_timeout_ms": PAIR_HEARTBEAT_TIMEOUT_MS,
        },
    )
    try:
        lease = realtime_lease_registry.acquire(
            session_id=session_id,
            participant_id=participant["id"],
            day=int(session["day"]),
            max_concurrent=int(settings.max_concurrent_study_sessions),
        )
    except CapacityExceeded as exc:
        capacity = _capacity_status()
        raise HTTPException(
            status_code=429,
            detail={
                "code": "study_capacity_full",
                "message": capacity["message_zh"],
                "message_en": capacity["message_en"],
                "retry_after_minutes": capacity["retry_after_minutes"],
                "active_count": capacity["active_count"],
                "max_concurrent": capacity["max_concurrent"],
                "lease_error": str(exc),
            },
        ) from exc
    return {
        **activated,
        "pair_qualified": True,
        "realtime_lease": lease.to_dict(),
        "realtime_routing": {
            "ws_path_prefix": "/ws",
            "listener_role": settings.listener_role,
        },
        "capacity": _capacity_status(
            participant_id=str(participant["id"]),
            session_id=str(session_id),
        ),
    }


def _websocket_auth(token: str) -> tuple[dict[str, Any], dict[str, Any]]:
    claims = decode_token(token, settings)
    participant = db.participant(str(claims["sub"]))
    if participant is None:
        raise TokenError("Participant disabled")
    return claims, participant


@app.websocket("/ws/transport-probe")
async def transport_probe_websocket(websocket: WebSocket) -> None:
    """Measure the public WebSocket path without creating study records."""
    client_ip = _websocket_ip(websocket)
    identity: TransportProbeIdentity | None = None
    acquired = await transport_probe_registry.acquire_ip(client_ip)
    if not acquired:
        await websocket.close(code=4429, reason="probe_ip_connection_limit")
        return

    try:
        await websocket.accept()
        started = time.monotonic()
        deadline = started + TRANSPORT_PROBE_MAX_CONNECTION_SECONDS
        try:
            first = await asyncio.wait_for(
                websocket.receive(),
                timeout=TRANSPORT_PROBE_HELLO_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise TransportProbeViolation(
                "hello_timeout", 4408, "Probe hello was not received in time"
            ) from exc
        if first.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect(int(first.get("code", 1000)))
        first_text = first.get("text")
        if not isinstance(first_text, str):
            raise TransportProbeViolation(
                "invalid_hello", 4400, "Probe hello must be a JSON text frame"
            )
        if len(first_text.encode("utf-8")) > TRANSPORT_PROBE_HELLO_MAX_BYTES:
            raise TransportProbeViolation(
                "message_too_large", 1009, "Probe hello exceeds its size limit"
            )
        try:
            hello = json.loads(first_text)
        except json.JSONDecodeError as exc:
            raise TransportProbeViolation(
                "invalid_hello", 4400, "Probe hello is not valid JSON"
            ) from exc
        if not isinstance(hello, dict):
            raise TransportProbeViolation(
                "invalid_hello", 4400, "Probe hello must be a JSON object"
            )
        candidate_identity = _transport_probe_hello(hello)
        if not await transport_probe_registry.bind_identity(candidate_identity):
            raise TransportProbeViolation(
                "duplicate_channel", 4409, "This probe channel is already connected"
            )
        identity = candidate_identity

        hello_received_ms = _transport_probe_wall_ms()
        await websocket.send_json(
            {
                "type": "hello_ack",
                "protocol_version": TRANSPORT_PROBE_PROTOCOL_VERSION,
                "probe_id": identity.probe_id,
                "user_index": identity.user_index,
                "channel": identity.channel,
                "server_received_ms": hello_received_ms,
                "server_sent_ms": max(hello_received_ms, _transport_probe_wall_ms()),
                "max_connection_ms": int(
                    TRANSPORT_PROBE_MAX_CONNECTION_SECONDS * 1000
                ),
                "pcm_frame_bytes": TRANSPORT_PROBE_PCM_BYTES,
            }
        )

        arrivals: deque[float] = deque()
        last_sequence = -1
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportProbeViolation(
                    "connection_duration_exceeded",
                    4408,
                    "Probe connection reached its duration limit",
                )
            receive_timeout = min(remaining, TRANSPORT_PROBE_IDLE_TIMEOUT_SECONDS)
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=receive_timeout
                )
            except asyncio.TimeoutError as exc:
                error_code = (
                    "connection_duration_exceeded"
                    if deadline - time.monotonic() <= 0
                    else "idle_timeout"
                )
                error_message = (
                    "Probe connection reached its duration limit"
                    if error_code == "connection_duration_exceeded"
                    else "Probe channel became idle"
                )
                raise TransportProbeViolation(
                    error_code, 4408, error_message
                ) from exc
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(int(message.get("code", 1000)))

            server_received_ms = _transport_probe_wall_ms()
            arrival = time.monotonic()
            maximum_rate = (
                TRANSPORT_PROBE_TIMING_MESSAGES_PER_SECOND
                if identity.channel == "timing"
                else TRANSPORT_PROBE_PAIR_MESSAGES_PER_SECOND
            )
            if not _transport_probe_rate_allowed(arrivals, arrival, maximum_rate):
                raise TransportProbeViolation(
                    "rate_limit_exceeded",
                    4429,
                    "Probe channel exceeded its message-rate limit",
                )

            if identity.channel == "timing":
                binary = message.get("bytes")
                if not isinstance(binary, bytes):
                    raise TransportProbeViolation(
                        "unsupported_frame", 1003, "Timing probe requires binary frames"
                    )
                if len(binary) != TRANSPORT_PROBE_BINARY_BYTES:
                    raise TransportProbeViolation(
                        "invalid_frame_size",
                        1009,
                        f"Timing frame must contain a 12-byte header and {TRANSPORT_PROBE_PCM_BYTES} payload bytes",
                    )
                raw_sequence, raw_client_sent_ms = TRANSPORT_PROBE_BINARY_HEADER.unpack_from(
                    binary
                )
                sequence = _transport_probe_sequence(raw_sequence)
                client_sent_ms = _transport_probe_number(
                    raw_client_sent_ms, "client_sent_ms"
                )
            else:
                text = message.get("text")
                if not isinstance(text, str):
                    raise TransportProbeViolation(
                        "unsupported_frame", 1003, "Pair probe requires JSON text frames"
                    )
                if len(text.encode("utf-8")) > TRANSPORT_PROBE_PAIR_MAX_BYTES:
                    raise TransportProbeViolation(
                        "message_too_large", 1009, "Pair heartbeat exceeds its size limit"
                    )
                try:
                    heartbeat = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise TransportProbeViolation(
                        "invalid_message", 4400, "Pair heartbeat is not valid JSON"
                    ) from exc
                if (
                    not isinstance(heartbeat, dict)
                    or set(heartbeat) != {"type", "seq", "client_sent_ms"}
                    or heartbeat.get("type") != "heartbeat"
                ):
                    raise TransportProbeViolation(
                        "invalid_message", 4400, "Pair channel requires a valid heartbeat"
                    )
                sequence = _transport_probe_sequence(heartbeat.get("seq"))
                client_sent_ms = _transport_probe_number(
                    heartbeat.get("client_sent_ms"), "client_sent_ms"
                )

            if sequence <= last_sequence:
                raise TransportProbeViolation(
                    "non_monotonic_sequence",
                    4400,
                    "Probe sequence numbers must be strictly increasing",
                )
            last_sequence = sequence
            await websocket.send_json(
                {
                    "type": "probe_ack",
                    "channel": identity.channel,
                    "seq": sequence,
                    "client_sent_ms": client_sent_ms,
                    "server_received_ms": server_received_ms,
                    "server_sent_ms": max(
                        server_received_ms, _transport_probe_wall_ms()
                    ),
                }
            )
    except WebSocketDisconnect:
        pass
    except TransportProbeViolation as exc:
        await _close_transport_probe(websocket, exc)
    except RuntimeError as exc:
        if not _is_websocket_disconnect_error(exc):
            raise
    finally:
        await transport_probe_registry.release(client_ip, identity)


@app.websocket("/ws/pair")
async def pair_websocket(websocket: WebSocket, token: str, session_id: str):
    try:
        claims, participant = _websocket_auth(token)
        session = db.session(session_id, participant["id"])
        if session is None or session["client_role"] != claims["role"]:
            raise TokenError("Pair socket does not match its device session")
    except TokenError:
        await websocket.close(code=4401)
        return
    role = str(claims["role"])
    participant_id = participant["id"]
    connected = await pair_hub.connect(participant_id, role, session, websocket)
    if not connected:
        return
    try:
        while True:
            message = await websocket.receive_json()
            kind = str(message.get("type", ""))
            if kind not in {
                "heartbeat",
                "companion_state",
                "background_state",
                "notice",
                "phone_ptt",
                "desktop_asr",
            }:
                continue
            message["server_ms"] = int(time.time() * 1000)
            await pair_hub.touch(participant_id, role, websocket, message)
            await pair_hub.relay(participant_id, role, message)
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        if not _is_websocket_disconnect_error(exc):
            raise
    finally:
        await pair_hub.disconnect(participant_id, role, websocket)


def _micro_probe_policy(policy_token: str, *, cohort: str, agent_mode: str) -> str:
    """Resolve a locked micro-probe token without mislabelling model provenance."""

    if policy_token == "A":
        return "current"
    if policy_token == "B":
        # The DistilHuBERT engineering prototype has no participant-matched
        # release tooling.  B is its immutable shared/base head.
        return "base" if cohort == "study_a" or agent_mode == "distilhubert" else "matched_other"
    if policy_token == "C":
        return "vad"
    return ""


@app.websocket("/ws/timing")
async def timing_websocket(websocket: WebSocket, token: str, session_id: str):
    try:
        claims, participant = _websocket_auth(token)
        session = db.session(session_id, participant["id"])
        if claims["role"] != "study":
            raise TokenError("Timing socket requires the study device role")
        if session is None:
            raise TokenError("Timing socket study session was not found")
        if session["state"] != "active":
            raise TokenError("Timing socket requires an activated study session")
        if not await pair_hub.is_qualified(participant["id"], session_id):
            raise TokenError(
                "Timing socket requires a live qualified phone companion "
                "(same-day online + companion_ready)"
            )
        runtime = await asyncio.to_thread(
            timing_service.runtime,
            participant,
            int(session["day"]),
            session_id=str(session["id"]),
        )
    except (TokenError, ModelUnavailable) as exc:
        logger.warning(
            "Timing WebSocket rejected before accept: session=%s reason=%s",
            session_id,
            exc,
        )
        # Accept first so browsers receive a JSON error instead of opaque 1006.
        reason = str(exc)
        error_code = (
            "pair_not_ready"
            if ("companion" in reason.lower() or "qualified" in reason.lower())
            else "timing_rejected"
        )
        try:
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "timing_error",
                    "code": error_code,
                    "message": reason,
                }
            )
            await websocket.close(code=4401)
        except Exception:  # noqa: BLE001 - rejection path must stay best-effort
            try:
                await websocket.close(code=4401)
            except Exception:  # noqa: BLE001
                pass
        return
    await websocket.accept()
    generation = await timing_hub.connect(session_id, websocket)
    try:
        await websocket.send_json(
            {
                "type": "timing_ready",
                "mode": settings.agent_mode,
                "timing_generation": generation,
                **runtime.ready_payload(),
            }
        )
        while True:
            message = await websocket.receive()
            if not await timing_hub.is_current(session_id, websocket, generation):
                try:
                    await websocket.close(
                        code=4001,
                        reason="A newer timing connection replaced this session",
                    )
                except RuntimeError as exc:
                    if not _is_websocket_disconnect_error(exc):
                        raise
                break
            if not await pair_hub.is_qualified(participant["id"], session_id):
                await websocket.send_json(
                    {"type": "pair_required", "message": "电脑端与手机端不再同时在线，数据已暂停"}
                )
                continue
            if message.get("bytes") is not None:
                pcm_received_wall_ms = time.time() * 1000
                prediction_started_runtime_ms = time.monotonic() * 1000
                async def process_pcm_for_owner() -> dict[str, Any]:
                    result = await asyncio.to_thread(
                        runtime.process_pcm,
                        message["bytes"],
                        prediction_started_runtime_ms,
                    )
                    result["latency_clock"] = "server_monotonic_activation_commit_v2"
                    if result.get("activate"):
                        activation_commit_runtime_ms = time.monotonic() * 1000
                        activation_id = str(result.get("activation_id") or "")
                        if not runtime.mark_activation_observable(
                            activation_id, activation_commit_runtime_ms
                        ):
                            raise RuntimeError("activation changed before prediction commit")
                        result["activation_compute_ms"] = max(
                            0.0,
                            activation_commit_runtime_ms - prediction_started_runtime_ms,
                        )
                        db.append_event(
                            participant["id"],
                            session_id,
                            "agent.auto_activate",
                            None,
                            activation_commit_runtime_ms,
                            result,
                        )
                    await websocket.send_json(
                        {
                            "type": "prediction",
                            "client_ms": pcm_received_wall_ms,
                            "server_ms": int(time.time() * 1000),
                            **result,
                        }
                    )
                    return result

                owned, result = await timing_hub.execute(
                    session_id,
                    websocket,
                    generation,
                    process_pcm_for_owner,
                )
                if not owned:
                    break
                continue
            raw = message.get("text")
            if not raw:
                continue
            payload = json.loads(raw)
            command = payload.get("type")
            if command == "phase":
                phase = str(payload.get("phase"))
                policy_token = str(payload.get("policy_token"))
                transition_id = str(payload.get("transition_id") or "")
                if phase == "candidate_probe":
                    policy = "candidate_prior"
                elif phase in {"frozen", "learning", "rest", "complete"}:
                    policy = "current"
                else:
                    policy = _micro_probe_policy(
                        policy_token,
                        cohort=str(participant["cohort"]),
                        agent_mode=settings.agent_mode,
                    )
                async def set_phase_for_owner() -> dict[str, Any]:
                    if phase == "candidate_probe":
                        timing_service.select_candidate_prior(
                            runtime,
                            participant,
                            int(session["day"]),
                            policy_token,
                        )
                    transition = runtime.set_phase(
                        phase,
                        policy,
                        drain_active=payload.get("drain_active") is True,
                        active_block_index=(
                            int(payload["active_block_index"])
                            if payload.get("active_block_index") is not None
                            else None
                        ),
                        analysis_eligible=payload.get("analysis_eligible") is not False,
                    )
                    if transition["committed"]:
                        db.append_event(
                            participant["id"],
                            session_id,
                            "phase.server_committed",
                            payload.get("client_ms"),
                            payload.get("monotonic_ms"),
                            {
                                "phase": runtime.phase,
                                "policy_token": policy_token,
                                "policy": runtime.policy,
                                "active_block_index": runtime.active_block_index,
                                "analysis_eligible": runtime.analysis_eligible,
                                "transition_id": transition_id or None,
                            },
                        )
                        await websocket.send_json(
                            {
                                "type": "phase_ack",
                                "transition_id": transition_id or None,
                                "phase": runtime.phase,
                                "policy_token": policy_token,
                                "policy": runtime.policy,
                                "drained_agent_state": transition["drained_state"],
                                "server_agent_state": runtime.agent_state,
                                "server_state_version": runtime.state_version,
                                "server_activation_id": runtime.activation_id,
                                "active_block_index": runtime.active_block_index,
                                "analysis_eligible": runtime.analysis_eligible,
                            }
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "phase_blocked",
                                "transition_id": transition_id or None,
                                "requested_phase": phase,
                                "agent_state": runtime.agent_state,
                                "message": "An active response must be drained before changing phase",
                            }
                        )
                    return transition

                owned, transition = await timing_hub.execute(
                    session_id,
                    websocket,
                    generation,
                    set_phase_for_owner,
                )
                if not owned:
                    break
            elif command == "state":
                async def set_state_for_owner() -> None:
                    runtime.set_state(str(payload.get("state")))
                    await websocket.send_json(
                        {
                            "type": "state_ack",
                            "transition_id": payload.get("transition_id"),
                            "state": runtime.agent_state,
                            "server_agent_state": runtime.agent_state,
                            "server_state_version": runtime.state_version,
                            "server_activation_id": runtime.activation_id,
                        }
                    )

                owned, _ = await timing_hub.execute(
                    session_id,
                    websocket,
                    generation,
                    set_state_for_owner,
                )
                if not owned:
                    break
            elif command == "asr_final":
                # Align with PCM/tap: decision clocks use server monotonic ms.
                asr_runtime_ms = time.monotonic() * 1000

                async def note_asr_final_for_owner() -> dict[str, Any]:
                    result = runtime.note_asr_final(asr_runtime_ms)
                    db.append_event(
                        participant["id"],
                        session_id,
                        "asr.final_timing",
                        payload.get("client_ms"),
                        payload.get("monotonic_ms"),
                        {
                            "transition_id": payload.get("transition_id"),
                            "text_chars": len(str(payload.get("text") or "")),
                            **result,
                        },
                    )
                    await websocket.send_json(
                        {
                            "type": "asr_final_ack",
                            "transition_id": payload.get("transition_id"),
                            **result,
                        }
                    )
                    return result

                owned, _ = await timing_hub.execute(
                    session_id,
                    websocket,
                    generation,
                    note_asr_final_for_owner,
                )
                if not owned:
                    break
            elif command == "decision_context":
                conversation_target = str(
                    payload.get("conversation_target") or ""
                )
                try:
                    reply_needed_probability = float(
                        payload["reply_needed_probability"]
                    )
                except (KeyError, TypeError, ValueError):
                    reply_needed_probability = -1.0
                context_source = str(payload.get("source") or "")
                raw_text_sha256 = payload.get("text_sha256")
                text_sha256 = (
                    str(raw_text_sha256).lower()
                    if raw_text_sha256 is not None
                    else None
                )

                async def set_context_for_owner() -> dict[str, Any]:
                    result = runtime.set_decision_context(
                        conversation_target=conversation_target,
                        reply_needed_probability=reply_needed_probability,
                        source=context_source,
                        text_sha256=text_sha256,
                    )
                    db.append_event(
                        participant["id"],
                        session_id,
                        "model.decision_context",
                        payload.get("client_ms"),
                        payload.get("monotonic_ms"),
                        {
                            "transition_id": payload.get("transition_id"),
                            **result,
                        },
                    )
                    await websocket.send_json(
                        {
                            "type": "decision_context_ack",
                            "transition_id": payload.get("transition_id"),
                            **result,
                            **runtime._server_state_snapshot(),
                        }
                    )
                    return result

                try:
                    owned, _ = await timing_hub.execute(
                        session_id,
                        websocket,
                        generation,
                        set_context_for_owner,
                    )
                except ValueError as exc:
                    if str(exc) == "decision context requires the waiting state":
                        snapshot = {
                            **runtime._decision_context_snapshot(),
                            **runtime._server_state_snapshot(),
                        }
                        db.append_event(
                            participant["id"],
                            session_id,
                            "model.decision_context_stale",
                            payload.get("client_ms"),
                            payload.get("monotonic_ms"),
                            {
                                "transition_id": payload.get("transition_id"),
                                "rejection_reason": "agent_not_waiting",
                                "requested_conversation_target": conversation_target,
                                "requested_source": context_source,
                                **snapshot,
                            },
                        )
                        await websocket.send_json(
                            {
                                "type": "decision_context_ack",
                                "transition_id": payload.get("transition_id"),
                                "accepted": False,
                                "stale": True,
                                "rejection_reason": "agent_not_waiting",
                                **snapshot,
                            }
                        )
                        continue
                    await _send_timing_error(
                        websocket,
                        code="invalid_decision_context",
                        message=str(exc),
                        command="decision_context",
                        transition_id=payload.get("transition_id"),
                    )
                    continue
                if not owned:
                    break
            elif command == "simulation_gap":
                if (
                    participant.get("data_origin") != "simulation"
                    or not bool(participant.get("preview_mode"))
                ):
                    await _send_timing_error(
                        websocket,
                        code="simulation_gap_forbidden",
                        message="Silent-gap fast-forward is restricted to preview simulation accounts",
                        command="simulation_gap",
                    )
                    continue

                async def reset_gap_for_owner() -> dict[str, Any]:
                    result = runtime.reset_audio_after_simulated_gap()
                    db.append_event(
                        participant["id"],
                        session_id,
                        "simulation.audio_gap_fast_forwarded",
                        payload.get("client_ms"),
                        payload.get("monotonic_ms"),
                        {
                            "transition_id": payload.get("transition_id"),
                            "gap_seconds": float(payload.get("gap_seconds") or 0.0),
                            **result,
                        },
                    )
                    await websocket.send_json(
                        {
                            "type": "simulation_gap_ack",
                            "transition_id": payload.get("transition_id"),
                            **result,
                        }
                    )
                    return result

                try:
                    owned, _ = await timing_hub.execute(
                        session_id,
                        websocket,
                        generation,
                        reset_gap_for_owner,
                    )
                except ValueError as exc:
                    await _send_timing_error(
                        websocket,
                        code="simulation_gap_state_conflict",
                        message=str(exc),
                        command="simulation_gap",
                    )
                    continue
                if not owned:
                    break
            elif command == "tap":
                raw_tap_id = payload.get("tap_id")
                tap_id = raw_tap_id.strip() if isinstance(raw_tap_id, str) else ""
                if not tap_id or len(tap_id) > 100:
                    await _send_timing_error(
                        websocket,
                        code="invalid_tap_id",
                        message="tap_id must be a non-empty string of at most 100 characters",
                        command="tap",
                        tap_id=tap_id or None,
                    )
                    continue

                conditional_fields = {
                    "intended_action",
                    "expected_state",
                    "expected_activation_id",
                    "expected_state_version",
                }
                present_conditional_fields = conditional_fields.intersection(payload)
                if present_conditional_fields and present_conditional_fields != conditional_fields:
                    await _send_timing_error(
                        websocket,
                        code="incomplete_conditional_tap",
                        message="Conditional taps require the complete expectation tuple",
                        command="tap",
                        tap_id=tap_id,
                        missing_fields=sorted(
                            conditional_fields - present_conditional_fields
                        ),
                    )
                    continue

                conditional = bool(present_conditional_fields)
                intended_action = (
                    str(payload.get("intended_action")) if conditional else None
                )
                if conditional and intended_action not in {"manual_activate", "interrupt"}:
                    await _send_timing_error(
                        websocket,
                        code="invalid_intended_action",
                        message="intended_action must be manual_activate or interrupt",
                        command="tap",
                        tap_id=tap_id,
                    )
                    continue
                expected_state = (
                    str(payload.get("expected_state")) if conditional else None
                )
                if conditional and expected_state not in {"waiting", "thinking", "speaking"}:
                    await _send_timing_error(
                        websocket,
                        code="invalid_expected_state",
                        message="expected_state must be waiting, thinking, or speaking",
                        command="tap",
                        tap_id=tap_id,
                    )
                    continue
                raw_expected_activation_id = payload.get("expected_activation_id")
                if conditional and (
                    not isinstance(raw_expected_activation_id, str)
                    or len(raw_expected_activation_id) > 100
                ):
                    await _send_timing_error(
                        websocket,
                        code="invalid_expected_activation_id",
                        message="expected_activation_id must be a string of at most 100 characters",
                        command="tap",
                        tap_id=tap_id,
                    )
                    continue
                expected_activation_id = (
                    raw_expected_activation_id if conditional else None
                )
                raw_expected_state_version = payload.get("expected_state_version")
                if conditional and (
                    isinstance(raw_expected_state_version, bool)
                    or not isinstance(raw_expected_state_version, int)
                    or raw_expected_state_version < 0
                ):
                    await _send_timing_error(
                        websocket,
                        code="invalid_expected_state_version",
                        message="expected_state_version must be a non-negative integer",
                        command="tap",
                        tap_id=tap_id,
                    )
                    continue
                expected_state_version = (
                    raw_expected_state_version if conditional else None
                )
                feedback_attribution = str(
                    payload.get("feedback_attribution")
                    or "acoustic_personalizer"
                )
                if feedback_attribution not in {
                    "acoustic_personalizer",
                    "reply_needed_context",
                }:
                    await _send_timing_error(
                        websocket,
                        code="invalid_feedback_attribution",
                        message=(
                            "feedback_attribution must be "
                            "acoustic_personalizer or reply_needed_context"
                        ),
                        command="tap",
                        tap_id=tap_id,
                    )
                    continue
                if intended_action == "interrupt" and not expected_activation_id:
                    await _send_timing_error(
                        websocket,
                        code="interrupt_activation_required",
                        message="Conditional interrupts must identify their activation",
                        command="tap",
                        tap_id=tap_id,
                    )
                    continue
                tap_received_wall_ms = time.time() * 1000
                tap_received_runtime_ms = time.monotonic() * 1000
                try:
                    client_clock_offset_ms = (
                        float(payload["client_ms"]) - tap_received_wall_ms
                    )
                except (KeyError, TypeError, ValueError):
                    client_clock_offset_ms = None

                async def tap_for_owner() -> dict[str, Any]:
                    result = await asyncio.to_thread(
                        runtime.tap,
                        tap_received_runtime_ms,
                        pointer_agent_state=(
                            str(payload.get("pointer_agent_state"))
                            if payload.get("pointer_agent_state") is not None
                            else None
                        ),
                        page_visible=(
                            payload.get("visible")
                            if isinstance(payload.get("visible"), bool)
                            else None
                        ),
                        intended_action=intended_action,
                        expected_state=expected_state,
                        expected_activation_id=expected_activation_id,
                        expected_state_version=expected_state_version,
                        feedback_attribution=feedback_attribution,
                    )
                    result.setdefault(
                        "pointer_agent_state", payload.get("pointer_agent_state")
                    )
                    result.setdefault(
                        "page_visible",
                        payload.get("visible")
                        if isinstance(payload.get("visible"), bool)
                        else None,
                    )
                    result.update(
                        {
                            "tap_id": tap_id,
                            "latency_clock": "server_monotonic_activation_commit_v2",
                            "client_clock_offset_ms": client_clock_offset_ms,
                        }
                    )
                    db.append_event(
                        participant["id"],
                        session_id,
                        f"tap.{result['action']}",
                        payload.get("client_ms"),
                        payload.get("monotonic_ms"),
                        result,
                    )
                    await websocket.send_json({"type": "tap_result", **result})
                    return result

                owned, result = await timing_hub.execute(
                    session_id,
                    websocket,
                    generation,
                    tap_for_owner,
                )
                if not owned:
                    break
    except WebSocketDisconnect:
        pass
    except (ValueError, json.JSONDecodeError):
        logger.exception(
            "Timing WebSocket rejected malformed input or failed while processing session %s",
            session_id,
        )
        try:
            await websocket.send_json(
                {"type": "timing_error", "code": "timing_processing_failed"}
            )
            await websocket.close(code=1007, reason="Timing input or processing failed")
        except (RuntimeError, WebSocketDisconnect):
            pass
    except ModelUnavailable:
        try:
            await websocket.send_json(
                {"type": "timing_error", "code": "model_unavailable"}
            )
            await websocket.close(code=1011, reason="Timing model unavailable")
        except (RuntimeError, WebSocketDisconnect):
            pass
    except RuntimeError as exc:
        if _is_websocket_disconnect_error(exc):
            pass
        elif _is_model_resource_error(exc):
            logger.exception(
                "Timing model compute resources were exhausted for session %s",
                session_id,
            )
            try:
                await websocket.send_json(
                    {
                        "type": "timing_error",
                        "code": "model_resource_exhausted",
                    }
                )
                await websocket.close(
                    code=1013,
                    reason="Timing model compute resources exhausted",
                )
            except (RuntimeError, WebSocketDisconnect):
                pass
        else:
            raise
    finally:
        await timing_hub.disconnect(session_id, websocket, generation)


def _admin_path_record(name: str, path: Path, *, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    exists = resolved.exists()
    return {
        "label": name,
        "kind": kind,
        "path": str(resolved),
        "exists": exists,
        "bytes": resolved.stat().st_size if exists and resolved.is_file() else None,
    }


_ADMIN_EVENT_TEXT_KEYS = ("text", "transcript", "reply", "message", "answer")


def _admin_event_payload(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("payload") or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _admin_event_summary(
    row: dict[str, Any], *, include_text: bool = False
) -> dict[str, Any]:
    payload = _admin_event_payload(row)
    text_fields = {
        key: str(payload[key])
        for key in _ADMIN_EVENT_TEXT_KEYS
        if isinstance(payload.get(key), str) and bool(payload[key].strip())
    }
    summary = {
        "seq": row.get("seq"),
        "session_id": row.get("session_id"),
        "event_type": row.get("event_type"),
        "server_ms": row.get("server_ms"),
        "active_block_index": payload.get("active_block_index", payload.get("block")),
        "phase": payload.get("phase", payload.get("activation_phase")),
        "action": payload.get("action"),
        "analysis_eligible": payload.get("analysis_eligible"),
        "training_applied": payload.get("training_applied"),
        "contains_text": bool(text_fields),
    }
    if include_text:
        summary["text_fields"] = text_fields
    return summary


def _admin_conversation_turn(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("event_type")) != "chat.turn":
        return None
    payload = _admin_event_payload(row)
    transcript = str(payload.get("transcript") or "").strip()
    reply = str(payload.get("reply") or "").strip()
    if not transcript and not reply:
        return None
    return {
        "seq": row.get("seq"),
        "session_id": row.get("session_id"),
        "server_ms": row.get("server_ms"),
        "source": payload.get("source"),
        "transcript": transcript,
        "reply": reply,
        "mode": payload.get("mode"),
        "model": payload.get("model"),
        "latency_ms": payload.get("latency_ms"),
    }


_ADMIN_CONVERSATION_EXPORT_FIELDS = (
    "run_id",
    "participant_id",
    "session_id",
    "event_seq",
    "utterance_order",
    "timestamp_ms",
    "timestamp_iso_utc",
    "timestamp_iso_asia_shanghai",
    "timestamp_basis",
    "role",
    "role_code",
    "system_source",
    "text",
    "model",
    "mode",
    "reply_latency_ms",
)


_PORTABLE_BUNDLE_SCHEMA = "tap-to-adapt-portable/v1"
_PORTABLE_HISTORY_SCHEMA = "tap-to-adapt-conversation-history/v1"
_PORTABLE_MEMORY_SCHEMA = "tap-to-adapt-memory/v1"
_PORTABLE_EVENT_SCHEMA = "tap-to-adapt-social-events/v1"
_PORTABLE_EVENT_PREFIXES = (
    "agent.",
    "tap.",
    "timing.",
    "companion.",
    "rest.",
    "schedule.",
)
_PORTABLE_EVENT_PAYLOAD_KEYS = frozenset(
    {
        "action",
        "active_block_index",
        "activation_phase",
        "activation_to_tap_ms",
        "agent_state",
        "analysis_eligible",
        "block",
        "block_index",
        "completed_seconds",
        "feedback_attribution",
        "intended_action",
        "label",
        "latency_ms",
        "mode",
        "model",
        "page_visible",
        "phase",
        "policy",
        "quarantine_reason",
        "quarantined",
        "reason",
        "reply_needed",
        "reply_needed_probability",
        "source",
        "tap_phase",
        "tap_policy",
        "target",
        "training_applied",
        "training_eligible",
        "user_speech_resumed",
    }
)


def _portable_contained_artifact(root: Path, relative: str) -> Path:
    candidate = Path(str(relative or ""))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("Portable model artifact path is not contained")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    resolved.relative_to(resolved_root)
    if not resolved.is_file():
        raise ValueError("Portable model artifact is missing")
    return resolved


def _portable_event_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = _admin_event_payload(row)
    result: dict[str, Any] = {}
    for key in _PORTABLE_EVENT_PAYLOAD_KEYS:
        value = payload.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            if value is not None:
                result[key] = value
    return result


def _portable_checkpoint_files(
    participant: dict[str, Any], completion: dict[str, Any]
) -> dict[str, bytes]:
    session_id = str(completion["study_session_id"])
    session = db.session(session_id, str(participant["id"]))
    if session is None:
        raise ValueError("Completed study session is unavailable")
    receipt = timing_service.finalize_personal_checkpoint(
        participant,
        int(session["day"]),
        session_id=session_id,
    )
    personal_pointer = Path(str(receipt["pointer_path"])).resolve()
    personal_head = Path(str(receipt["personal_head_path"])).resolve()
    replay = Path(str(receipt["replay_path"])).resolve()
    personal_root = personal_pointer.parent.resolve()
    for path in (personal_pointer, personal_head, replay):
        path.relative_to(personal_root)
        if not path.is_file():
            raise ValueError("Final personal checkpoint is incomplete")

    release_manifest = Path(settings.distilhubert_release_manifest).resolve()
    release = json.loads(release_manifest.read_text(encoding="utf-8"))
    if not isinstance(release, dict):
        raise ValueError("Response-model release manifest is invalid")
    release_root = release_manifest.parent.resolve()
    release_artifacts = [
        _portable_contained_artifact(
            release_root, str((release.get("shared_head") or {}).get("path") or "")
        )
    ]
    replay_seed = release.get("replay_seed")
    if isinstance(replay_seed, dict):
        release_artifacts.append(
            _portable_contained_artifact(
                release_root, str(replay_seed.get("path") or "")
            )
        )

    files = {
        "checkpoint/personal/distilhubert-personal.json": personal_pointer.read_bytes(),
        f"checkpoint/personal/{personal_head.name}": personal_head.read_bytes(),
        f"checkpoint/personal/{replay.name}": replay.read_bytes(),
        "checkpoint/release/release.json": release_manifest.read_bytes(),
    }
    for artifact in release_artifacts:
        relative = artifact.relative_to(release_root).as_posix()
        files[f"checkpoint/release/{relative}"] = artifact.read_bytes()
    return files


def _portable_bundle_for_participant(
    participant: dict[str, Any], completion: dict[str, Any]
) -> bytes:
    participant_id = str(participant["id"])
    event_rows = db.participant_event_rows(participant_id)
    conversation_rows = _admin_export_conversation_rows(event_rows)
    conversation = {
        "schema": _PORTABLE_HISTORY_SCHEMA,
        "turns": [
            {
                "timestamp_ms": int(row["timestamp_ms"]),
                "role": str(row["role_code"]),
                "source": str(row.get("system_source") or "primary"),
                "text": str(row["text"]),
            }
            for row in conversation_rows
        ],
    }
    memory_items = {
        "computer_experiment_agent": db.get_conversation_memory(
            participant_id, "primary"
        ),
        "phone_companion_agent": db.get_conversation_memory(
            participant_id, "companion"
        ),
    }
    memory = {
        "schema": _PORTABLE_MEMORY_SCHEMA,
        "items": {key: value for key, value in memory_items.items() if value},
    }
    social_events = {
        "schema": _PORTABLE_EVENT_SCHEMA,
        "events": [
            {
                "timestamp_ms": int(row.get("server_ms") or 0),
                "event_type": str(row.get("event_type") or ""),
                "payload": _portable_event_payload(row),
            }
            for row in event_rows
            if str(row.get("event_type") or "").startswith(
                _PORTABLE_EVENT_PREFIXES
            )
            and str(row.get("event_type") or "") != "chat.turn"
        ],
    }
    files = _portable_checkpoint_files(participant, completion)
    files.update(
        {
            "conversation_history.json": (
                json.dumps(conversation, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
            "memory.json": (
                json.dumps(memory, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
            "social_events.json": (
                json.dumps(
                    social_events, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("utf-8"),
        }
    )
    manifest = {
        "schema": _PORTABLE_BUNDLE_SCHEMA,
        "created_ms": int(time.time() * 1000),
        "source": "tap-to-adapt-study",
        "completed": True,
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(files.items())
        },
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    return buffer.getvalue()


def _admin_conversation_timestamp(timestamp_ms: int, zone: timezone | ZoneInfo) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=zone).isoformat(
        timespec="milliseconds"
    )


def _admin_export_conversation_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten chat.turn pairs into timestamped, role-labelled utterances."""
    rows: list[dict[str, Any]] = []
    shanghai = ZoneInfo("Asia/Shanghai")
    agent_roles = {
        "primary": ("电脑实验系统agent", "computer_experiment_agent"),
        "companion": ("手机陪伴系统agent", "phone_companion_agent"),
    }
    for event in sorted(events, key=lambda row: int(row.get("id") or 0)):
        turn = _admin_conversation_turn(event)
        if turn is None:
            continue
        source = str(turn.get("source") or "")
        agent_role, agent_role_code = agent_roles.get(
            source,
            ("电脑实验系统agent", "computer_experiment_agent"),
        )
        committed_ms = int(turn.get("server_ms") or 0)
        try:
            reply_latency_ms = max(0.0, float(turn.get("latency_ms") or 0.0))
        except (TypeError, ValueError):
            reply_latency_ms = 0.0
        user_ms = max(0, round(committed_ms - reply_latency_ms))
        common = {
            "run_id": event.get("run_id"),
            "participant_id": event.get("participant_id"),
            "session_id": turn.get("session_id"),
            "event_seq": turn.get("seq"),
            "system_source": source,
            "model": turn.get("model"),
            "mode": turn.get("mode"),
            "reply_latency_ms": turn.get("latency_ms"),
        }
        utterances = (
            (
                1,
                user_ms,
                "chat_request_received_estimate",
                "使用者",
                "user",
                turn.get("transcript"),
            ),
            (
                2,
                committed_ms,
                "chat_turn_server_committed",
                agent_role,
                agent_role_code,
                turn.get("reply"),
            ),
        )
        for order, timestamp_ms, basis, role, role_code, text in utterances:
            if not str(text or "").strip():
                continue
            rows.append(
                {
                    **common,
                    "utterance_order": order,
                    "timestamp_ms": timestamp_ms,
                    "timestamp_iso_utc": _admin_conversation_timestamp(
                        timestamp_ms, timezone.utc
                    ),
                    "timestamp_iso_asia_shanghai": _admin_conversation_timestamp(
                        timestamp_ms, shanghai
                    ),
                    "timestamp_basis": basis,
                    "role": role,
                    "role_code": role_code,
                    "text": str(text).strip(),
                }
            )
    return rows


def _admin_open_response_answers(row: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        answers = json.loads(str(row.get("answers") or "{}"))
        locale = str(row.get("locale") or "zh-CN")
        catalog = protocol.questionnaires_for(locale)
        schema = catalog["instruments"][_questionnaire_schema_name(str(row["instrument"]))]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(answers, dict):
        return []
    result: list[dict[str, Any]] = []
    for kind, items in (
        ("open", schema.get("open_items", [])),
        ("followup", schema.get("followups", [])),
    ):
        for item in items:
            item_id = str(item["id"])
            value = answers.get(item_id)
            if not isinstance(value, str) or not value.strip():
                continue
            result.append(
                {
                    "question_id": item_id,
                    "question_text": str(item["text"]),
                    "answer_text": value,
                    "answer_kind": kind,
                }
            )
    return result


@app.get("/api/admin/realtime-leases", dependencies=[Depends(require_admin)])
def admin_realtime_leases() -> dict[str, Any]:
    return {
        "ok": True,
        "listener_role": settings.listener_role,
        **realtime_lease_registry.snapshot(),
    }


def _format_elapsed_label(elapsed_ms: int | None) -> str:
    if elapsed_ms is None:
        return "—"
    total_seconds = max(0, int(elapsed_ms) // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{seconds}秒"
    return f"{minutes}分{seconds}秒"


def _live_subject_board_status(
    *,
    study: dict[str, Any] | None,
    companion: dict[str, Any] | None,
    appointment: dict[str, Any] | None,
    holder: dict[str, Any] | None,
) -> str:
    if study and study.get("activated_ms"):
        return "running"
    if study:
        return "setup"
    if companion:
        return "companion_only"
    appointment_status = str((appointment or {}).get("status") or "")
    if appointment_status == "claimed" or holder:
        return "claimed"
    if appointment_status == "booked":
        return "booked"
    if appointment_status == "active":
        return "claimed"
    return "setup"


ADMIN_EXPERIMENT_STATUS_LABELS = {
    "in_progress": "实验中",
    "completed": "已完成",
    "exited": "中途退出",
    "setup": "准备中",
    "booked": "已预约",
    "not_started": "未开始",
}


def _admin_experiment_progress_status(
    participant_id: str,
    *,
    sessions: list[dict[str, Any]] | None = None,
    completed_ids: set[str] | None = None,
    exited_ids: set[str] | None = None,
    timed_out_ids: set[str] | None = None,
    live_running_ids: set[str] | None = None,
    live_setup_ids: set[str] | None = None,
    appointment_status: str | None = None,
    now_ms: int | None = None,
    active_window_ms: int = 90 * 60 * 1000,
) -> dict[str, Any]:
    """Read-only experiment-progress label for the admin roster.

    Uses already-loaded session rows and peek lists. Does not write sessions,
    orchestration, models, or participant state.
    """
    pid = str(participant_id)
    clock_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    study_sessions = [
        row
        for row in (sessions or [])
        if str(row.get("client_role") or "") == "study"
    ]
    latest = max(
        study_sessions,
        key=lambda row: int(row.get("started_ms") or 0),
        default=None,
    )
    latest_state = str((latest or {}).get("state") or "")
    latest_activated = (latest or {}).get("activated_ms")
    latest_ended = (latest or {}).get("ended_ms")
    session_completed = any(
        str(row.get("state") or "") in {"completed", "ended"}
        and row.get("activated_ms")
        for row in study_sessions
    )
    if pid in (completed_ids or set()) or session_completed:
        key = "completed"
    elif pid in (live_running_ids or set()):
        key = "in_progress"
    elif latest_state == "aborted" or pid in (exited_ids or set()) or pid in (
        timed_out_ids or set()
    ):
        key = "exited"
    elif (
        latest
        and latest_activated
        and latest_ended is None
        and latest_state in {"started", "active"}
        and clock_ms - int(latest_activated) > max(60_000, int(active_window_ms))
    ):
        key = "exited"
    elif pid in (live_setup_ids or set()) or (
        latest
        and latest_ended is None
        and latest_state in {"started", "active"}
        and not latest_activated
    ):
        key = "setup"
    elif str(appointment_status or "") == "claimed":
        key = "setup"
    elif str(appointment_status or "") == "booked":
        key = "booked"
    elif latest_activated and latest_ended is not None and latest_state not in {
        "completed",
        "ended",
    }:
        key = "exited"
    else:
        key = "not_started"
    return {
        "experiment_status": key,
        "experiment_status_label": ADMIN_EXPERIMENT_STATUS_LABELS[key],
        "readonly": True,
    }


def _admin_live_subjects_payload() -> dict[str, Any]:
    """Assemble the live-subject board from SELECT/file/lease peeks only."""
    now_ms = int(time.time() * 1000)
    live = db.live_experiment_sessions()
    schedule = orchestrator.peek_live_share_schedule()
    leases = realtime_lease_registry.peek()
    hidden = {
        str(pid) for pid in (schedule.get("hidden_from_admin_ids") or []) if pid
    }
    holder = schedule.get("current_holder")
    holder_pid = str((holder or {}).get("participant_id") or "")
    appointments_by_pid: dict[str, dict[str, Any]] = {}
    for row in schedule.get("appointments") or []:
        pid = str(row.get("participant_id") or "")
        if pid:
            appointments_by_pid[pid] = row
    leases_by_pid: dict[str, list[dict[str, Any]]] = {}
    for lease in leases.get("active") or []:
        pid = str(lease.get("participant_id") or "")
        if pid:
            leases_by_pid.setdefault(pid, []).append(lease)

    by_pid: dict[str, dict[str, Any]] = {}
    for row in live.get("participants") or []:
        pid = str(row.get("participant_id") or "")
        if pid:
            by_pid[pid] = dict(row)
    arrived_appointment_ids = [
        pid
        for pid, row in appointments_by_pid.items()
        if str((row or {}).get("status") or "") in {"claimed", "active"}
    ]
    extra_ids = [
        pid
        for pid in dict.fromkeys(
            [*arrived_appointment_ids, holder_pid, *leases_by_pid]
        )
        if pid and pid not in by_pid and pid not in hidden
    ]
    briefs = db.participants_brief(extra_ids)
    for pid in extra_ids:
        brief = briefs.get(pid) or {}
        by_pid[pid] = {
            "participant_id": pid,
            "code_hint": str(brief.get("code_hint") or ""),
            "group_name": str(brief.get("group_name") or ""),
            "run_slug": str(brief.get("run_slug") or ""),
            "study": None,
            "companion": None,
            "pair_live": False,
        }

    cards = _admin_subject_cards([pid for pid in by_pid if pid not in hidden])
    assignment_snapshot = orchestrator.admin_assignment_snapshot()
    status_labels = {
        "running": "正式进行中",
        "setup": "准备中（未正式计时）",
        "companion_only": "仅手机端在线",
        "claimed": "已到场占位",
        "booked": "已预约未到场",
    }
    status_rank = {
        "running": 0,
        "setup": 1,
        "companion_only": 2,
        "claimed": 3,
        "booked": 4,
    }
    subjects: list[dict[str, Any]] = []
    for pid, row in by_pid.items():
        if pid in hidden:
            continue
        study = row.get("study")
        companion = row.get("companion")
        appointment = appointments_by_pid.get(pid)
        row_holder = holder if holder_pid == pid else None
        board_status = _live_subject_board_status(
            study=study,
            companion=companion,
            appointment=appointment,
            holder=row_holder,
        )
        if board_status not in {
            "running",
            "setup",
            "companion_only",
            "claimed",
        }:
            continue
        official_start_ms = None
        if study and study.get("activated_ms"):
            official_start_ms = int(study["activated_ms"])
        progress = estimated_timed_progress(official_start_ms, now_ms)
        subject_leases = leases_by_pid.get(pid) or []
        experiment_arm = _admin_experiment_arm(
            {
                "id": pid,
                "group_name": row.get("group_name"),
                "prior_id": row.get("prior_id"),
            },
            assignment_snapshot,
        )
        subjects.append(
            {
                "participant_id": pid,
                "code_hint": row.get("code_hint") or "",
                "group_name": row.get("group_name") or "",
                "experiment_arm": experiment_arm,
                "experiment_arm_label": _ADMIN_CORE_ARM_LABELS.get(
                    experiment_arm, experiment_arm
                ),
                "run_slug": row.get("run_slug") or "",
                "display_name": (cards.get(pid) or {}).get("display_name") or "",
                "contact": (cards.get(pid) or {}).get("contact")
                or _admin_empty_contact(),
                "board_status": board_status,
                "board_status_label": status_labels.get(board_status, board_status),
                "official_start_ms": official_start_ms,
                "elapsed_ms": progress.get("elapsed_ms"),
                "elapsed_label": _format_elapsed_label(progress.get("elapsed_ms")),
                "progress_phase": progress.get("phase"),
                "progress_block": progress.get("block"),
                "progress_label": progress.get("label"),
                "study_session": study,
                "companion_session": companion,
                "computer_online": bool(study),
                "phone_online": bool(companion),
                "pair_live": bool(row.get("pair_live")),
                "realtime_connected": bool(subject_leases),
                "appointment": appointment,
                "holder_mode": (row_holder or {}).get("mode") if row_holder else None,
            }
        )
    subjects.sort(
        key=lambda row: (
            status_rank.get(str(row["board_status"]), 9),
            int(row["official_start_ms"] or 0)
            or int((row.get("appointment") or {}).get("scheduled_start_ms") or 0)
            or int(((row.get("study_session") or {}).get("started_ms")) or 0),
        )
    )
    counts = {
        "running": sum(row["board_status"] == "running" for row in subjects),
        "setup": sum(row["board_status"] == "setup" for row in subjects),
        "claimed": sum(row["board_status"] == "claimed" for row in subjects),
        "booked": sum(row["board_status"] == "booked" for row in subjects),
        "realtime_connected": sum(row["realtime_connected"] for row in subjects),
        "total": len(subjects),
    }
    return {
        "ok": True,
        "readonly": True,
        "server_ms": now_ms,
        "counts": counts,
        "subjects": subjects,
    }


@app.get("/api/admin/live-subjects", dependencies=[Depends(require_admin)])
def admin_live_subjects() -> dict[str, Any]:
    return _admin_live_subjects_payload()


@app.get("/api/admin/share-fit-trend-board", dependencies=[Depends(require_admin)])
def admin_share_fit_trend_board() -> dict[str, Any]:
    """Read-only share-fit per-donor trends. Does not refresh or lock fit."""
    final_items = {
        str(item["id"]): item
        for item in protocol.questionnaires["instruments"]["final"].get("items", [])
        if isinstance(item, dict) and item.get("id")
    }
    return orchestrator.peek_share_fit_trend_board(final_items=final_items)


@app.get("/api/admin/stages", dependencies=[Depends(require_admin)])
def admin_stage_templates() -> dict[str, Any]:
    return {
        "stages": [
            {"id": stage, **template}
            for stage, template in ADMIN_STAGE_TEMPLATES.items()
        ],
        "formal_rule": (
            "正式 Study A/B 账号不能从简易表单创建；必须通过锁定 roster、"
            "随机分配和 A→B 流水线。"
        ),
    }


@app.post("/api/admin/stage-accounts", dependencies=[Depends(require_admin)])
def admin_initialize_stage_accounts(payload: StageAccountInitRequest) -> Response:
    raise HTTPException(
        status_code=405,
        detail="Participant accounts are created only by public self-enrollment",
    )
    template = ADMIN_STAGE_TEMPLATES.get(payload.stage)
    if template is None:
        raise HTTPException(status_code=400, detail="Unknown experiment stage")
    if not template.get("initializable"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "formal_stage_managed",
                "message": template["description"],
                "managed_by": template.get("managed_by"),
            },
        )
    maximum = int(template["maximum_count"])
    if payload.count > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{template['label']} 单批最多创建 {maximum} 个账号",
        )
    personal_debug = bool(payload.personal_debug)
    if personal_debug and payload.count != 1:
        raise HTTPException(
            status_code=400,
            detail="仅用于个人调试的批次必须且只能创建 1 个账号",
        )
    slug_prefix = (
        f"debug-{payload.stage.replace('_', '-')}"
        if personal_debug
        else payload.stage.replace("_", "-")
    )
    run_slug = payload.run_slug or f"{slug_prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
    storage_namespace = "personal_debug" if personal_debug else "pre_experiments"
    try:
        codes = db.create_participants(
            payload.count,
            str(template["cohort"]),
            str(template["group_name"]),
            preview_mode=bool(template["preview_mode"]),
            data_origin=str(template["data_origin"]),
            protocol_mode=str(template["protocol_mode"]),
            run_slug=run_slug,
            analysis_excluded=personal_debug,
        )
        participant_rows = [db.participant_by_code(code) for code in codes]
        if any(row is None for row in participant_rows):
            raise ValueError("newly created account could not be resolved")
        resolved_rows = [row for row in participant_rows if row is not None]
        batch_id = db.record_account_batch(
            run_id=str(resolved_rows[0]["run_id"]),
            stage=payload.stage,
            label=payload.label.strip() or str(template["label"]),
            cohort=str(template["cohort"]),
            group_name=str(template["group_name"]),
            data_origin=str(template["data_origin"]),
            protocol_mode=str(template["protocol_mode"]),
            participant_ids=[str(row["id"]) for row in resolved_rows],
            is_pre_experiment=personal_debug,
            analysis_excluded=personal_debug,
            storage_namespace=storage_namespace,
        )
        batch = db.account_batch(batch_id)
        if batch is None:
            raise ValueError("account batch could not be resolved")
        archive = (
            _initialize_pre_experiment_archive(batch)
            if personal_debug
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    accounts = [
        {
            "participant_id": row["id"],
            "access_code": code,
            "code_hint": row["code_hint"],
            "run_slug": row["run_slug"],
            "cohort": row["cohort"],
            "group_name": row["group_name"],
            "data_origin": row["data_origin"],
            "protocol_mode": row["protocol_mode"],
            "analysis_excluded": bool(row.get("analysis_excluded")),
        }
        for code, row in zip(codes, resolved_rows)
    ]
    return JSONResponse(
        {
            "batch_id": batch_id,
            "stage": payload.stage,
            "stage_label": template["label"],
            "run_slug": run_slug,
            "assigned_count": len(accounts),
            "personal_debug": personal_debug,
            "pre_experiment": personal_debug,
            "analysis_excluded": personal_debug,
            "storage_namespace": storage_namespace,
            "archive": archive,
            "accounts": accounts,
            "codes_shown_once": True,
            "warning": (
                "请立即下载或复制 access_code。数据库只保存散列，之后无法恢复完整登录码。"
            ),
        },
        headers={"Cache-Control": "no-store"},
    )


_ADMIN_CORE_EVENT_TYPES = (
    "model.prediction",
    "agent.auto_activate",
    "tap.manual_activate",
    "tap.interrupt",
    "tap.state_transition_mistouch",
)
_ADMIN_CORE_ARM_LABELS = {
    "share_fit": "Share 拟合 donor（链式 Personalised）",
    "cold_learn": "① 冷启动在线学（C_B）",
    "shared_learn": "② Share 启动在线学（P）",
    "shared_frozen": "③ Share 启动冻结（F）",
    "unassigned": "未编入当前阶段",
}


def _admin_empty_contact() -> dict[str, Any]:
    return {
        "complete": False,
        "channels": [],
        "email": "",
        "phone": "",
        "wechat": "",
    }


def _admin_subject_cards(
    participant_ids: list[str] | tuple[str, ...] | set[str],
) -> dict[str, dict[str, Any]]:
    ids = list(dict.fromkeys(str(value) for value in participant_ids if value))
    consent_names = db.admin_consent_names(ids)
    try:
        identity_names = compensation_vault.admin_identity_names(ids)
        contacts = compensation_vault.admin_contact_directory(ids)
    except CompensationVaultUnavailable:
        identity_names = {}
        contacts = {}
    cards: dict[str, dict[str, Any]] = {}
    for participant_id in ids:
        contact = {
            **_admin_empty_contact(),
            **(contacts.get(participant_id) or {}),
        }
        display_name = (
            consent_names.get(participant_id)
            or identity_names.get(participant_id)
            or ""
        )
        cards[participant_id] = {
            "display_name": display_name,
            "contact": contact,
        }
    return cards


def _admin_empty_baseline_demographics() -> dict[str, Any]:
    return {
        "identity": None,
        "identity_label": "",
        "spoken_accent": None,
        "spoken_accent_label": "",
        "age_range": None,
        "age_range_label": "",
        "gender": None,
        "gender_label": "",
    }


def _admin_baseline_demographics(row: dict[str, Any] | None) -> dict[str, Any]:
    """Surface baseline sample descriptors for the admin roster and detail."""
    result = _admin_empty_baseline_demographics()
    if row is None:
        return result
    try:
        answers = json.loads(row.get("answers") or "{}")
    except (TypeError, json.JSONDecodeError):
        answers = {}
    if not isinstance(answers, dict):
        answers = {}
    try:
        labels = json.loads(row.get("answer_labels") or "[]")
    except (TypeError, json.JSONDecodeError):
        labels = []
    by_id = {
        str(item.get("question_id")): item
        for item in labels
        if isinstance(item, dict) and item.get("question_id")
    }
    field_labels = {
        str(field["id"]): {
            str(option["value"]): str(option["label"])
            for option in field.get("options", [])
        }
        for field in protocol.questionnaires["instruments"]["baseline"].get("fields", [])
    }
    for key in ("identity", "spoken_accent", "age_range", "gender"):
        value = answers.get(key)
        result[key] = value
        snapshot = by_id.get(key) or {}
        label = str(snapshot.get("answer_text") or "").strip()
        if not label and value is not None:
            label = field_labels.get(key, {}).get(str(value), str(value))
        result[f"{key}_label"] = label
    return result


def _admin_empty_window_taps() -> dict[str, Any]:
    return {
        "windows": [
            {"window": index, "taps": 0, "interrupts": 0}
            for index in range(1, 6)
        ],
        "tap_total": 0,
        "interrupt_total": 0,
    }


def _admin_experiment_arm(
    participant: dict[str, Any], assignment_snapshot: dict[str, Any]
) -> str:
    """Expose the orchestration arm instead of the model-update mode."""
    participant_id = str(participant["id"])
    share_ids = {
        str(value)
        for value in assignment_snapshot.get("share_fit_participant_ids") or []
    }
    share_ids.update(
        str(value)
        for value in assignment_snapshot.get("share_fit_completed_ids") or []
    )
    if participant_id in share_ids:
        return "share_fit"
    contrast_arms = {
        str(row.get("participant_id") or ""): str(row.get("arm") or "")
        for row in assignment_snapshot.get("contrast_assignments") or []
        if isinstance(row, dict) and row.get("participant_id")
    }
    if participant_id in contrast_arms:
        return contrast_arms[participant_id]
    if str(participant.get("group_name") or "").lower() == "frozen":
        return "shared_frozen"
    # After contrast is reopened back to share-fit, do not infer ①/② from
    # a leftover prior_id. Keep those rows visible as unassigned.
    if str(assignment_snapshot.get("phase") or "") == "share_fit":
        return "unassigned"
    if participant.get("prior_id"):
        return "shared_learn"
    return "cold_learn"


def _admin_core_percentile(values: list[float], probability: float) -> float | None:
    present = sorted(value for value in values if math.isfinite(value))
    if not present:
        return None
    position = (len(present) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(present[lower], 3)
    fraction = position - lower
    return round(
        present[lower] * (1.0 - fraction) + present[upper] * fraction,
        3,
    )


def _admin_core_metric() -> dict[str, Any]:
    return {
        "opportunities": 0,
        "auto_activations": 0,
        "eligible_manual_activations": 0,
        "eligible_interruptions": 0,
        "raw_manual_activations": 0,
        "raw_interruptions": 0,
        "behavior_corrections": 0,
        "state_transition_mistouches": 0,
        "latencies_ms": [],
        "blocks": set(),
    }


def _admin_core_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    opportunities = sum(int(row["opportunities"]) for row in metrics)
    eligible_activations = sum(
        int(row["eligible_manual_activations"]) for row in metrics
    )
    eligible_interruptions = sum(
        int(row["eligible_interruptions"]) for row in metrics
    )
    eligible_corrections = eligible_activations + eligible_interruptions
    participant_rates = [
        (
            int(row["eligible_manual_activations"])
            + int(row["eligible_interruptions"])
        )
        / int(row["opportunities"])
        for row in metrics
        if int(row["opportunities"]) > 0
    ]
    latencies = [
        float(value)
        for row in metrics
        for value in row.get("latencies_ms", [])
        if value is not None and math.isfinite(float(value))
    ]
    return {
        "participants": len(metrics),
        "participants_with_opportunities": len(participant_rates),
        "opportunities": opportunities,
        "auto_activations": sum(int(row["auto_activations"]) for row in metrics),
        "eligible_manual_activations": eligible_activations,
        "eligible_interruptions": eligible_interruptions,
        "eligible_corrections": eligible_corrections,
        "eligible_correction_rate": round(
            eligible_corrections / opportunities, 6
        )
        if opportunities
        else None,
        "participant_mean_correction_rate": round(
            sum(participant_rates) / len(participant_rates), 6
        )
        if participant_rates
        else None,
        "raw_manual_activations": sum(
            int(row["raw_manual_activations"]) for row in metrics
        ),
        "raw_interruptions": sum(int(row["raw_interruptions"]) for row in metrics),
        "raw_taps": sum(
            int(row["raw_manual_activations"]) + int(row["raw_interruptions"])
            for row in metrics
        ),
        "state_transition_mistouches": sum(
            int(row["state_transition_mistouches"]) for row in metrics
        ),
        "interrupt_latency_count": len(latencies),
        "interrupt_latency_median_ms": _admin_core_percentile(latencies, 0.5),
        "interrupt_latency_p90_ms": _admin_core_percentile(latencies, 0.9),
    }


@app.get("/api/admin/research-core", dependencies=[Depends(require_admin)])
def admin_research_core(run_slug: str | None = None) -> dict[str, Any]:
    """CHI-facing, privacy-preserving aggregates for the current user study."""
    runs = db.table_rows("study_runs")
    available_by_slug = {str(row["slug"]): row for row in runs}
    effective_slug = run_slug
    if effective_slug is None and settings.public_enrollment_run_slug in available_by_slug:
        effective_slug = settings.public_enrollment_run_slug
    if effective_slug is not None and effective_slug not in available_by_slug:
        raise HTTPException(status_code=404, detail="Study run not found")
    selected_runs = (
        [available_by_slug[effective_slug]] if effective_slug is not None else runs
    )
    run_ids = {str(row["id"]) for row in selected_runs}
    hidden_ids = orchestrator.admin_hidden_participant_ids()
    roster = [
        row
        for row in db.admin_roster()
        if str(row["run_id"]) in run_ids and str(row["id"]) not in hidden_ids
    ]
    # Admin statistics are scoped by the physically isolated instance database
    # (/tap versus /tap-test), not by legacy formal/pilot exclusion metadata.
    instance_participants = list(roster)
    participant_by_id = {
        str(row["id"]): row for row in instance_participants
    }

    assignment_snapshot = orchestrator.admin_assignment_snapshot()
    participant_arm = {
        participant_id: _admin_experiment_arm(row, assignment_snapshot)
        for participant_id, row in participant_by_id.items()
    }
    participant_metrics = {
        participant_id: _admin_core_metric() for participant_id in participant_by_id
    }
    block_metrics: dict[tuple[str, int], dict[str, Any]] = {}
    sessions: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    if run_ids:
        ordered_ids = sorted(run_ids)
        run_placeholders = ",".join("?" for _ in ordered_ids)
        event_placeholders = ",".join("?" for _ in _ADMIN_CORE_EVENT_TYPES)
        with db.connect() as conn:
            sessions = [
                dict(row)
                for row in conn.execute(
                    f"SELECT * FROM sessions WHERE run_id IN ({run_placeholders})",
                    ordered_ids,
                ).fetchall()
            ]
            responses = [
                dict(row)
                for row in conn.execute(
                    f"SELECT * FROM responses WHERE run_id IN ({run_placeholders})",
                    ordered_ids,
                ).fetchall()
            ]
            event_rows = [
                dict(row)
                for row in conn.execute(
                    f"""SELECT participant_id, event_type, server_ms, payload
                        FROM events
                        WHERE run_id IN ({run_placeholders})
                          AND event_type IN ({event_placeholders})
                        ORDER BY id""",
                    (*ordered_ids, *_ADMIN_CORE_EVENT_TYPES),
                ).fetchall()
            ]

    completion_records = db.end_to_end_completion_records(
        list(participant_by_id)
    )
    completed_ids = set(completion_records)
    final_recovery_records = db.admin_final_questionnaire_recoveries(run_ids)
    all_final_responses = {
        str(row["participant_id"]): row
        for row in responses
        if str(row.get("instrument")) == "final"
        and str(row["participant_id"]) in participant_by_id
    }
    # Questionnaire outcomes and every behavioral aggregate below use the same
    # end-to-end completed sample.  A submitted questionnaire alone is progress,
    # not permission to enter the CHI outcome tables.
    final_responses = {
        participant_id: row
        for participant_id, row in all_final_responses.items()
        if participant_id in completed_ids
    }

    operator_resume_by_id = {
        participant_id: load_operator_resume(settings.data_dir, participant_id)
        for participant_id in participant_by_id
    }
    for event in event_rows:
        participant_id = str(event["participant_id"])
        metric = participant_metrics.get(participant_id)
        if metric is None:
            continue
        try:
            payload = json.loads(str(event.get("payload") or "{}"))
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if payload.get("analysis_eligible") is False:
            continue
        try:
            block = int(payload.get("active_block_index", payload.get("block", 0)) or 0)
        except (TypeError, ValueError):
            block = 0
        if block not in {1, 2, 3, 4, 5}:
            continue
        if hide_block5_event(
            operator_resume_by_id.get(participant_id),
            block=block,
            server_ms=int(event.get("server_ms") or 0),
        ):
            continue
        metric["blocks"].add(block)
        block_metric = block_metrics.setdefault(
            (participant_id, block), _admin_core_metric()
        )
        block_metric["blocks"].add(block)
        event_type = str(event["event_type"])
        targets = (metric, block_metric)
        if (
            event_type == "model.prediction"
            and payload.get("decision_eligible") is True
            and not payload.get("skipped")
        ):
            for target in targets:
                target["opportunities"] += 1
        elif event_type == "agent.auto_activate":
            for target in targets:
                target["auto_activations"] += 1
        elif event_type in {"tap.manual_activate", "tap.interrupt"}:
            if payload.get("cancelled") is True:
                continue
            raw_key = (
                "raw_manual_activations"
                if event_type == "tap.manual_activate"
                else "raw_interruptions"
            )
            eligible_key = (
                "eligible_manual_activations"
                if event_type == "tap.manual_activate"
                else "eligible_interruptions"
            )
            for target in targets:
                target[raw_key] += 1
                if payload.get("training_eligible") is True:
                    target[eligible_key] += 1
                    if (
                        event_type == "tap.interrupt"
                        and payload.get("activation_to_tap_ms") is not None
                    ):
                        try:
                            target["latencies_ms"].append(
                                float(payload["activation_to_tap_ms"])
                            )
                        except (TypeError, ValueError):
                            pass
                if payload.get("training_eligible") is True or payload.get(
                    "quarantine_reason"
                ) == "activation_not_learning_current":
                    target["behavior_corrections"] += 1
        elif event_type == "tap.state_transition_mistouch":
            for target in targets:
                target["state_transition_mistouches"] += 1

    consent_signed = {
        participant_id
        for participant_id in participant_by_id
        if (consent := db.latest_consent(participant_id)) is not None
        and bool(consent.get("participation"))
    }

    arms = list(_ADMIN_CORE_ARM_LABELS)
    by_arm = []
    sample_progress = []
    for arm in arms:
        ids = [pid for pid, value in participant_arm.items() if value == arm]
        completed_arm_ids = [pid for pid in ids if pid in completed_ids]
        arm_metrics = [participant_metrics[pid] for pid in completed_arm_ids]
        by_arm.append(
            {
                "arm": arm,
                "label": _ADMIN_CORE_ARM_LABELS[arm],
                **_admin_core_summary(arm_metrics),
            }
        )
        sample_progress.append(
            {
                "arm": arm,
                "label": _ADMIN_CORE_ARM_LABELS[arm],
                "enrolled": len(ids),
                "completed": sum(pid in completed_ids for pid in ids),
                "five_blocks": sum(
                    len(participant_metrics[pid]["blocks"]) == 5 for pid in ids
                ),
                "final_questionnaire": sum(
                    pid in all_final_responses for pid in ids
                ),
            }
        )

    by_block = []
    for arm in arms:
        for block in range(1, 6):
            metrics = [
                metric
                for (participant_id, metric_block), metric in block_metrics.items()
                if metric_block == block and participant_arm.get(participant_id) == arm
                and participant_id in completed_ids
            ]
            by_block.append(
                {
                    "arm": arm,
                    "label": _ADMIN_CORE_ARM_LABELS[arm],
                    "block": block,
                    **_admin_core_summary(metrics),
                }
            )

    final_items = {
        str(item["id"]): item
        for item in protocol.questionnaires["instruments"]["final"].get("items", [])
    }
    dimension_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for participant_id, response in final_responses.items():
        try:
            answers = json.loads(str(response.get("answers") or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        arm = participant_arm[participant_id]
        per_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        for item_id, item in final_items.items():
            value = answers.get(item_id)
            if not isinstance(value, (int, float)):
                continue
            score = float(value)
            if item.get("reverse"):
                score = 8.0 - score
            facet = questionnaire_facet(item_id)
            dimension = str(item.get("dimension") or item_id)
            per_cell[(facet, dimension)].append(score)
        for (facet, dimension), values in per_cell.items():
            dimension_values[(arm, facet, dimension)].append(
                sum(values) / len(values)
            )
    questionnaire_dimensions = [
        {
            "arm": arm,
            "label": _ADMIN_CORE_ARM_LABELS[arm],
            "facet": facet,
            "dimension": dimension,
            "participants": len(values),
            "mean_1_to_7": round(sum(values) / len(values), 3),
        }
        for (arm, facet, dimension), values in sorted(dimension_values.items())
        if values
    ]
    block_groups = {
        block: {arm: [] for arm in CONTRAST_ARMS} for block in range(1, 6)
    }
    for (participant_id, block), metric in block_metrics.items():
        arm = participant_arm.get(participant_id)
        if (
            arm not in CONTRAST_ARMS
            or participant_id not in completed_ids
            or block not in block_groups
        ):
            continue
        opportunities = int(metric["opportunities"])
        if opportunities <= 0:
            continue
        eligible = (
            int(metric["eligible_manual_activations"])
            + int(metric["eligible_interruptions"])
        )
        behavior = int(metric.get("behavior_corrections") or 0)
        raw_taps = (
            int(metric["raw_manual_activations"])
            + int(metric["raw_interruptions"])
        )
        block_groups[block][arm].append(
            {
                "rate": behavior / opportunities,
                "training_rate": eligible / opportunities,
                "opportunities": opportunities,
                "behavior_corrections": behavior,
                "eligible_corrections": eligible,
                "raw_taps": raw_taps,
                "mistouches": int(metric["state_transition_mistouches"]),
            }
        )
    questionnaire_groups: dict[str, dict[str, dict[str, list[float]]]] = {
        facet: {} for facet in QUESTIONNAIRE_FACETS
    }
    for (arm, facet, dimension), values in dimension_values.items():
        if arm not in CONTRAST_ARMS:
            continue
        questionnaire_groups.setdefault(facet, {})
        questionnaire_groups[facet].setdefault(
            dimension, {item: [] for item in CONTRAST_ARMS}
        )
        questionnaire_groups[facet][dimension][arm] = list(values)
    three_arm_differences = build_three_arm_difference_board(
        block_groups=block_groups,
        questionnaire_groups=questionnaire_groups,
        arm_labels=_ADMIN_CORE_ARM_LABELS,
    )
    final_questionnaire_recoveries = [
        {
            "code_hint": row.get("code_hint"),
            "credential": (
                f"{str(row['credential'])[:4]}-{str(row['credential'])[4:]}"
            ),
            "participant_id": row["participant_id"],
            "session_id": row["session_id"],
            "entered_ms": row["entered_ms"],
            "model_finalized_ms": row.get("model_finalized_ms"),
            "chain_committed_ms": row.get("chain_committed_ms"),
            "final_submitted_ms": row.get("final_submitted_ms"),
            "final_questionnaire_submitted": bool(row.get("final_submitted_ms")),
            "end_to_end_complete": str(row["participant_id"]) in completed_ids,
            "last_error": row.get("last_error"),
        }
        for row in final_recovery_records
    ]

    return {
        "instance_id": settings.instance_id,
        "storage_namespace": str(settings.data_dir.resolve()),
        "generated_ms": int(time.time() * 1000),
        "filter": {
            "run_slug": effective_slug,
            "run_count": len(selected_runs),
            "public_run_slug": settings.public_enrollment_run_slug,
        },
        "definitions": {
            "instance_scoped": True,
            "primary_outcome": (
                "eligible manual correction rate = (training-eligible manual activate + "
                "training-eligible manual interrupt) / decision-eligible opportunities"
            ),
            "participant_is_unit": True,
            "end_to_end_completed_only": True,
            "end_to_end_completion_rule": (
                "final-day timed real-time experiment complete (strict: 5×6 min "
                "blocks + rests, 34 minutes). Final questionnaire is recorded "
                "when present but does not grant Share-chain or e2e credit"
            ),
            "active_blocks": 5,
            "active_minutes_per_block": 6,
            "rest_excluded": True,
            "mistouch_policy": (
                "tap.state_transition_mistouch is recorded and labeled for analysis, "
                "but is never eligible for model training or the primary-outcome numerator"
            ),
            "questionnaire_scale": "1–7；反向题已反向计分，越高表示体验越好",
        },
        "data_completeness": {
            "participants": len(instance_participants),
            "consent_signed": len(consent_signed),
            "completed": sum(pid in completed_ids for pid in participant_by_id),
            "five_blocks": sum(
                len(metric["blocks"]) == 5 for metric in participant_metrics.values()
            ),
            "final_questionnaire": len(all_final_responses),
            "entered_final_questionnaire": len(final_recovery_records),
            "pending_final_questionnaire": sum(
                not bool(row.get("final_submitted_ms"))
                for row in final_recovery_records
            ),
            "included_in_outcomes": len(completed_ids),
        },
        "sample_progress": sample_progress,
        "overall": _admin_core_summary(
            [participant_metrics[pid] for pid in completed_ids]
        ),
        "by_arm": by_arm,
        "by_block": by_block,
        "questionnaire_dimensions": questionnaire_dimensions,
        "three_arm_differences": three_arm_differences,
        "interrupt_continues": [
            {
                "code_hint": row.get("code_hint"),
                "interrupt_count": row.get("interrupt_count"),
                "continue_count": row.get("continue_count"),
                "restart_count": row.get("restart_count"),
                "continue_timestamps_ms": row.get("continue_timestamps_ms") or [],
                "last_ms": row.get("last_ms"),
            }
            for row in db.interrupt_continue_summaries()
            if str(row.get("participant_id") or "") in participant_by_id
        ],
        "final_questionnaire_recoveries": final_questionnaire_recoveries,
    }


@app.get("/api/admin/three-arm-difference-board", dependencies=[Depends(require_admin)])
def admin_three_arm_difference_board(run_slug: str | None = None) -> dict[str, Any]:
    """Read-only ①②③ block and questionnaire difference board."""
    core = admin_research_core(run_slug)
    board = dict(core.get("three_arm_differences") or {})
    board["ok"] = True
    board["readonly"] = True
    board["instance_id"] = core.get("instance_id")
    board["filter"] = core.get("filter")
    return board


@app.get("/api/admin/data-overview", dependencies=[Depends(require_admin)])
def admin_data_overview(run_slug: str | None = None) -> dict[str, Any]:
    runs = db.table_rows("study_runs")
    run_ids = {
        str(row["id"])
        for row in runs
        if run_slug is None or str(row["slug"]) == run_slug
    }
    if run_slug is not None and not run_ids:
        raise HTTPException(status_code=404, detail="Study run not found")
    hidden_ids = orchestrator.admin_hidden_participant_ids()
    participants = [
        row
        for row in db.admin_roster()
        if str(row["run_id"]) in run_ids and str(row["id"]) not in hidden_ids
    ]
    participant_display_limit = 300
    displayed_participants = sorted(
        participants, key=lambda row: int(row["created_ms"]), reverse=True
    )[:participant_display_limit]
    participant_ids = {str(row["id"]) for row in displayed_participants}
    assignment_snapshot = orchestrator.admin_assignment_snapshot()
    subject_cards = _admin_subject_cards(participant_ids)
    window_taps = db.admin_window_tap_counts(
        run_ids=run_ids,
        participant_ids=participant_ids,
    )
    baseline_by_participant: dict[str, dict[str, Any]] = {}
    if participant_ids:
        ordered_participant_ids = sorted(participant_ids)
        placeholders = ",".join("?" for _ in ordered_participant_ids)
        with db.connect() as conn:
            for row in conn.execute(
                f"""SELECT participant_id, answers, answer_labels
                    FROM responses
                    WHERE instrument='baseline'
                      AND participant_id IN ({placeholders})""",
                ordered_participant_ids,
            ).fetchall():
                baseline_by_participant[str(row["participant_id"])] = dict(row)
    sessions = [
        row for row in db.table_rows("sessions") if str(row["run_id"]) in run_ids
    ]
    events, event_count_by_run = db.recent_events_and_run_counts(run_ids, limit=200)
    audio_files = [
        row for row in db.table_rows("audio_files") if str(row["run_id"]) in run_ids
    ]
    snapshots = [
        row for row in db.table_rows("model_snapshots") if str(row["run_id"]) in run_ids
    ]
    priors = [
        row for row in db.table_rows("shared_priors") if str(row["run_id"]) in run_ids
    ]
    batches = [
        row for row in db.table_rows("account_batches") if str(row["run_id"]) in run_ids
    ]
    consent_flags: dict[str, dict[str, Any]] = {}
    for participant_id in participant_ids:
        consent = db.latest_consent(participant_id)
        participant = next(
            (
                row
                for row in displayed_participants
                if str(row["id"]) == participant_id
            ),
            None,
        )
        archive_dir = (
            _server_consent_archive_dir(consent, participant)
            if consent is not None and participant is not None
            else None
        )
        consent_flags[participant_id] = {
            "consent_id": consent["id"] if consent else None,
            "signed": bool(consent and consent["participation"]),
            "raw_audio": bool(consent and consent["raw_audio"]),
            "internal_reuse": bool(consent and consent.get("internal_reuse")),
            "future_sharing": bool(consent and consent.get("future_sharing")),
            "public_derived": bool(consent and consent.get("public_derived")),
            "controlled_sharing": bool(
                consent and consent.get("controlled_sharing")
            ),
            "public_audio": bool(consent and consent.get("public_audio")),
            "ai_training": bool(consent and consent.get("ai_training")),
            "commercial_research": bool(
                consent and consent.get("commercial_research")
            ),
            "server_archive_path": str(archive_dir) if archive_dir else None,
            "server_archive_saved": bool(
                archive_dir and (archive_dir / "MANIFEST.json").is_file()
            ),
        }

    event_counts: dict[str, int] = defaultdict(int)
    for row in events:
        event_counts[str(row["event_type"])] += 1
    session_by_id = {str(row["id"]): row for row in sessions}
    audio_assets = []
    for row in sorted(audio_files, key=lambda item: int(item["created_ms"]), reverse=True):
        absolute = (settings.data_dir / str(row["relative_path"])).resolve()
        audio_assets.append(
            {
                "audio_id": row["id"],
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "day": (
                    session_by_id.get(str(row["session_id"] or ""), {}).get("day")
                    if row.get("session_id")
                    else None
                ),
                "kind": row["kind"],
                "content_type": row["content_type"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "relative_path": row["relative_path"],
                "absolute_path": str(absolute),
                "exists": absolute.is_file(),
            }
        )

    now_ms = int(time.time() * 1000)
    schedule = orchestrator.peek_live_share_schedule()
    live = db.live_experiment_sessions()
    live_running_ids = {
        str(row["participant_id"])
        for row in live.get("participants") or []
        if (row.get("study") or {}).get("activated_ms")
    }
    live_setup_ids = {
        str(row["participant_id"])
        for row in live.get("participants") or []
        if row.get("study") and not (row.get("study") or {}).get("activated_ms")
    }
    completed_ids = {
        str(pid)
        for pid in (
            list(schedule.get("completed_donor_ids") or [])
            + list(schedule.get("contrast_completed_ids") or [])
        )
        if pid
    }
    exited_ids = {
        str(pid)
        for pid in (schedule.get("voluntarily_exited_participant_ids") or [])
        if pid
    }
    timed_out_ids = {
        str(pid)
        for pid in (schedule.get("setup_timed_out_participant_ids") or [])
        if pid
    }
    appointments_by_pid = {
        str(row["participant_id"]): row
        for row in (schedule.get("appointments") or [])
        if row.get("participant_id")
    }
    sessions_by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        sessions_by_participant[str(row["participant_id"])].append(row)

    dataset_root = settings.data_dir / "dataset_exports"
    dataset_releases = []
    if dataset_root.is_dir():
        for manifest_path in sorted(
            dataset_root.rglob("MANIFEST.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:100]:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, json.JSONDecodeError):
                manifest = {}
            dataset_releases.append(
                {
                    "path": str(manifest_path.parent.resolve()),
                    "manifest_path": str(manifest_path.resolve()),
                    "run_slug": manifest.get("run_slug"),
                    "schema_version": manifest.get("schema_version"),
                    "files": len(manifest.get("files") or []),
                    "modified_ms": int(manifest_path.stat().st_mtime * 1000),
                }
            )
    if run_slug is not None:
        dataset_releases = [
            row for row in dataset_releases if row["run_slug"] == run_slug
        ]

    run_summaries = []
    for run in runs:
        run_id = str(run["id"])
        if run_id not in run_ids:
            continue
        run_participants = [
            row for row in participants if str(row["run_id"]) == run_id
        ]
        run_sessions = [row for row in sessions if str(row["run_id"]) == run_id]
        run_event_count = event_count_by_run.get(run_id, 0)
        run_audio = [row for row in audio_files if str(row["run_id"]) == run_id]
        run_snapshots = [row for row in snapshots if str(row["run_id"]) == run_id]
        run_priors = [row for row in priors if str(row["run_id"]) == run_id]
        run_batches = [row for row in batches if str(row["run_id"]) == run_id]
        if run_slug is None and not any(
            (
                run_participants,
                run_sessions,
                run_event_count,
                run_audio,
                run_snapshots,
                run_priors,
                run_batches,
            )
        ):
            continue
        run_summaries.append(
            {
                "run_id": run_id,
                "run_slug": run["slug"],
                "origin": run["data_origin"],
                "protocol": run["protocol_version"],
                "mode": run["protocol_mode"],
                "participants": len(run_participants),
                "sessions": len(run_sessions),
                "completed_sessions": sum(
                    str(row["state"]) in {"completed", "ended"} for row in run_sessions
                ),
                "events": run_event_count,
                "audio_files": len(run_audio),
                "audio_bytes": sum(int(row["bytes"]) for row in run_audio),
                "created_ms": run["created_ms"],
            }
        )

    return {
        "filter": {"run_slug": run_slug},
        "display": {
            "participant_limit": participant_display_limit,
            "participants_truncated": len(participants) > len(displayed_participants),
        },
        "counts": {
            "runs": len(run_summaries),
            "participants": len(participants),
            "sessions": len(sessions),
            "events": sum(event_count_by_run.values()),
            "audio_files": len(audio_files),
            "audio_bytes": sum(int(row["bytes"]) for row in audio_files),
            "model_snapshots": len(snapshots),
            "shared_priors": len(priors),
            "account_batches": len(batches),
            "dataset_releases": len(dataset_releases),
            "compensation_identity_records": compensation_vault.count(),
            "compensation_registration_records": (
                compensation_vault.registration_count()
            ),
            "compensation_contact_records": compensation_vault.contact_count(),
            "compensation_decline_records": (
                compensation_vault.registration_count("decline")
            ),
        },
        "paths": [
            _admin_path_record("实验数据库", db.study_path, kind="sqlite"),
            _admin_path_record("同意书数据库（独立库）", db.consent_path, kind="sqlite"),
            _admin_path_record(
                "补偿身份数据库（独立加密库；通用导出不含明文）",
                compensation_vault.path,
                kind="encrypted_sqlite",
            ),
            _admin_path_record(
                "知情同意服务器归档根目录",
                settings.data_dir / "consent_archives",
                kind="directory",
            ),
            _admin_path_record("原始音频根目录", settings.media_dir, kind="directory"),
            _admin_path_record(
                "历史内部归档根目录",
                settings.data_dir / "pre_experiments",
                kind="directory",
            ),
            _admin_path_record(
                "个人调试隔离根目录",
                settings.data_dir / "personal_debug",
                kind="directory",
            ),
            _admin_path_record("数据集导出根目录", dataset_root, kind="directory"),
            _admin_path_record(
                "数据集导出脚本",
                settings.root / "scripts" / "export_research_dataset.py",
                kind="script",
            ),
        ],
        "runs": run_summaries,
        "participants": [
            {
                **row,
                "experiment_arm": _admin_experiment_arm(
                    row, assignment_snapshot
                ),
                **_admin_experiment_progress_status(
                    str(row["id"]),
                    sessions=sessions_by_participant.get(str(row["id"]), []),
                    completed_ids=completed_ids,
                    exited_ids=exited_ids,
                    timed_out_ids=timed_out_ids,
                    live_running_ids=live_running_ids,
                    live_setup_ids=live_setup_ids,
                    appointment_status=(
                        appointments_by_pid.get(str(row["id"])) or {}
                    ).get("status"),
                    now_ms=now_ms,
                ),
                "display_name": (
                    subject_cards.get(str(row["id"])) or {}
                ).get("display_name")
                or "",
                "contact": (
                    (subject_cards.get(str(row["id"])) or {}).get("contact")
                    or _admin_empty_contact()
                ),
                "window_taps": window_taps.get(
                    str(row["id"]), _admin_empty_window_taps()
                ),
                "baseline_demographics": _admin_baseline_demographics(
                    baseline_by_participant.get(str(row["id"]))
                ),
                "consent": consent_flags.get(str(row["id"]), {}),
            }
            for row in displayed_participants
        ],
        "account_batches": [
            {
                **row,
                "pre_experiment": bool(row.get("is_pre_experiment")),
                "analysis_excluded": bool(row.get("analysis_excluded")),
                "personal_debug": bool(row.get("analysis_excluded")),
                "has_archive": bool(row.get("archive_relative_path")),
                "archive_absolute_path": (
                    str((settings.data_dir / str(row["archive_relative_path"])).resolve())
                    if row.get("archive_relative_path")
                    else None
                ),
                "archive_exists": (
                    (settings.data_dir / str(row["archive_relative_path"])).resolve().is_dir()
                    if row.get("archive_relative_path")
                    else False
                ),
                "archive_manifest_path": (
                    str(
                        (
                            settings.data_dir
                            / str(row["archive_relative_path"])
                            / "ARCHIVE_MANIFEST.json"
                        ).resolve()
                    )
                    if row.get("archive_relative_path")
                    else None
                ),
            }
            for row in sorted(
                batches, key=lambda row: int(row["created_ms"]), reverse=True
            )
        ],
        "event_type_counts": [
            {"event_type": key, "count": value, "scope": "latest_200_events"}
            for key, value in sorted(
                event_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "recent_events": [
            _admin_event_summary(row)
            for row in events
        ],
        "audio_assets": audio_assets[:200],
        "model_assets": [
            {
                "kind": "personal_snapshot",
                "id": row["id"],
                "participant_id": row["participant_id"],
                "encoder_path": row["encoder_path"],
                "main_path": row["main_path"],
                "created_ms": row["created_ms"],
            }
            for row in snapshots
        ]
        + [
            {
                "kind": "shared_prior",
                "id": row["id"],
                "participant_id": None,
                "encoder_path": row["encoder_path"],
                "main_path": row["main_path"],
                "created_ms": row["created_ms"],
            }
            for row in priors
        ],
        "dataset_releases": dataset_releases,
    }


@app.post(
    "/api/admin/account-batches/{batch_id}/archive",
    dependencies=[Depends(require_admin)],
)
def admin_refresh_pre_experiment_archive(batch_id: str) -> dict[str, Any]:
    try:
        return _refresh_pre_experiment_archive(batch_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if message == "account batch not found" else 409
        raise HTTPException(status_code=status_code, detail=message) from exc


@app.get(
    "/api/admin/participants/{participant_id}/records",
    dependencies=[Depends(require_admin)],
)
def admin_participant_records(participant_id: str) -> dict[str, Any]:
    if participant_id in orchestrator.admin_hidden_participant_ids():
        raise HTTPException(status_code=404, detail="Participant not found")
    roster = next(
        (row for row in db.admin_roster() if str(row["id"]) == participant_id),
        None,
    )
    if roster is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    sessions = db.participant_table_rows("sessions", participant_id)
    session_ids = {str(row["id"]) for row in sessions}
    events = db.participant_table_rows("events", participant_id)
    quality = db.participant_table_rows("quality_checks", participant_id)
    responses = db.participant_table_rows("responses", participant_id)
    final_questionnaire_recovery = db.final_questionnaire_recovery(participant_id)
    audio = db.participant_table_rows("audio_files", participant_id)
    event_counts: dict[str, int] = defaultdict(int)
    for row in events:
        event_counts[str(row["event_type"])] += 1
    consent = db.latest_consent(participant_id)
    consent_archive = (
        _ensure_server_consent_archive(consent, roster)
        if consent is not None
        else None
    )
    assignment_snapshot = orchestrator.admin_assignment_snapshot()
    subject = _admin_subject_cards([participant_id]).get(participant_id) or {
        "display_name": "",
        "contact": _admin_empty_contact(),
    }
    window_taps = db.admin_window_tap_counts(
        participant_ids=[participant_id]
    ).get(participant_id, _admin_empty_window_taps())
    return {
        "participant": {
            **roster,
            "experiment_arm": _admin_experiment_arm(
                roster, assignment_snapshot
            ),
            "display_name": subject["display_name"],
        },
        "contact": subject["contact"],
        "window_taps": window_taps,
        "baseline_demographics": _admin_baseline_demographics(
            next(
                (
                    row
                    for row in responses
                    if str(row.get("instrument")) == "baseline"
                ),
                None,
            )
        ),
        "consent": {
            "consent_id": consent["id"] if consent else None,
            "signed": bool(consent and consent["participation"]),
            "version": consent["version"] if consent else None,
            "document_hash": consent["document_hash"] if consent else None,
            "signed_ms": consent["signed_ms"] if consent else None,
            "evidence_sha256": consent.get("evidence_sha256") if consent else None,
            "evidence_key_id": consent.get("evidence_key_id") if consent else None,
            "ledger_sequence": consent.get("ledger_seq") if consent else None,
            "server_instance": consent.get("server_instance") if consent else None,
            "third_party_verifiable": bool(
                consent and consent.get("evidence_json") and consent.get("evidence_key_id")
            ),
            "raw_audio": bool(consent and consent["raw_audio"]),
            "internal_reuse": bool(consent and consent.get("internal_reuse")),
            "future_sharing": bool(consent and consent["future_sharing"]),
            "controlled_sharing": bool(
                consent and consent.get("controlled_sharing")
            ),
            "public_derived": bool(consent and consent.get("public_derived")),
            "public_audio": bool(consent and consent.get("public_audio")),
            "ai_training": bool(consent and consent.get("ai_training")),
            "commercial_research": bool(
                consent and consent.get("commercial_research")
            ),
            "server_archive_saved": bool(
                consent_archive and consent_archive["verified"]
            ),
            "server_archive_path": (
                consent_archive["archive_path"] if consent_archive else None
            ),
            "server_archive_manifest_path": (
                consent_archive["manifest_path"] if consent_archive else None
            ),
            "server_archive_record_sha256": (
                consent_archive["record_sha256"] if consent_archive else None
            ),
        },
        "sessions": sessions,
        "event_type_counts": [
            {"event_type": key, "count": value}
            for key, value in sorted(
                event_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "recent_events": [
            _admin_event_summary(row, include_text=True)
            for row in sorted(
                events, key=lambda item: int(item["id"]), reverse=True
            )[:250]
        ],
        "conversation_turns": [
            turn
            for row in sorted(events, key=lambda item: int(item["id"]))
            if (turn := _admin_conversation_turn(row)) is not None
        ],
        "quality_checks": [
            {
                "session_id": row["session_id"],
                "kind": row["kind"],
                "passed": bool(row["passed"]),
                "created_ms": row["created_ms"],
            }
            for row in quality
        ],
        "responses": [
            {
                "session_id": row["session_id"],
                "instrument": row["instrument"],
                "version": row["version"],
                "locale": row.get("locale") or "zh-CN",
                "created_ms": row["created_ms"],
                "question_order": json.loads(row.get("question_order") or "[]"),
                "selections": json.loads(row.get("answer_labels") or "[]"),
                "open_answers": _admin_open_response_answers(row),
            }
            for row in responses
        ],
        "final_questionnaire_recovery": final_questionnaire_recovery,
        "audio_files": [
            {
                **row,
                "absolute_path": str(
                    (settings.data_dir / str(row["relative_path"])).resolve()
                ),
                "exists": (
                    settings.data_dir / str(row["relative_path"])
                ).resolve().is_file(),
            }
            for row in audio
        ],
        "session_ids": sorted(session_ids),
    }


@app.get(
    "/api/admin/participants/{participant_id}/consent-copy",
    dependencies=[Depends(require_admin)],
)
def admin_participant_consent_copy(participant_id: str) -> Response:
    participant = db.participant(participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    consent = db.latest_consent(participant_id)
    if consent is None or not consent["participation"]:
        raise HTTPException(status_code=404, detail="No signed consent record exists")
    return _consent_copy_response(
        consent,
        participant,
        copy_label="研究团队归档副本",
        filename_prefix="tap-to-adapt-consent-research-archive",
    )


@app.get(
    "/api/admin/participants/{participant_id}/consent-evidence",
    dependencies=[Depends(require_admin)],
)
def admin_participant_consent_evidence(participant_id: str) -> Response:
    participant = db.participant(participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    consent = db.latest_consent(participant_id)
    if consent is None or not consent["participation"]:
        raise HTTPException(status_code=404, detail="No signed consent record exists")
    return _consent_evidence_response(consent, participant)


@app.get("/api/admin/roster", dependencies=[Depends(require_admin)])
def admin_roster() -> list[dict[str, Any]]:
    hidden_ids = orchestrator.admin_hidden_participant_ids()
    rows = [
        row for row in db.admin_roster() if str(row["id"]) not in hidden_ids
    ]
    for row in rows:
        checks, ready = _participant_readiness(row)
        row["formal_ready"] = ready
        row["failed_readiness"] = [name for name, passed in checks.items() if not passed]
    return rows


@app.get("/api/admin/readiness", dependencies=[Depends(require_admin)])
def admin_readiness() -> dict[str, Any]:
    checks, is_ready = _formal_ready()
    return {
        "formal_ready": is_ready,
        "checks": checks,
        "consent": protocol.public_consent(),
        "ethics": {
            "consent_status": normalize_ethics_status(protocol.consent),
            "compensation_status": normalize_ethics_status(
                protocol.compensation_notices["zh-CN"]
            ),
        },
    }


@app.post("/api/admin/participants", dependencies=[Depends(require_admin)])
def admin_create_participants(payload: ParticipantCreateRequest) -> Response:
    raise HTTPException(
        status_code=405,
        detail="Participant accounts are created only by public self-enrollment",
    )
    inferred_origin = payload.data_origin or (
        "simulation"
        if payload.preview_mode
        else "pilot"
        if payload.cohort == "pilot"
        else "formal"
    )
    if inferred_origin == "formal":
        raise HTTPException(
            status_code=409,
            detail=(
                "Formal accounts must be created as one locked 36A+36B roster with "
                "scripts/provision_formal_pipeline_roster.py"
            ),
        )
    try:
        codes = db.create_participants(
            payload.count,
            payload.cohort,
            payload.group_name,
            preview_mode=payload.preview_mode,
            data_origin=payload.data_origin,
            protocol_mode=payload.protocol_mode,
            run_slug=payload.run_slug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"codes": codes}, headers={"Cache-Control": "no-store"})


@app.get("/api/admin/pipeline/{run_slug}", dependencies=[Depends(require_admin)])
def admin_pipeline_status(run_slug: str) -> dict[str, Any]:
    try:
        return _pipeline_for_run(run_slug).status(run_slug)
    except PipelineError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(
    "/api/admin/pipeline/{run_slug}/analysis",
    dependencies=[Depends(require_admin)],
)
def admin_pipeline_analysis(run_slug: str) -> dict[str, Any]:
    try:
        return _pipeline_for_run(run_slug).analysis(run_slug)
    except PipelineError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/admin/pipeline/release-study-a", dependencies=[Depends(require_admin)])
def admin_release_study_a(payload: PipelineWaveRequest) -> dict[str, Any]:
    raise HTTPException(status_code=405, detail="The administrator console is data-only")
    try:
        return _pipeline_for_run(payload.run_slug).release_study_a_wave(
            payload.run_slug, payload.wave
        )
    except (PipelineError, PriorError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/admin/pipeline/prepare-study-b", dependencies=[Depends(require_admin)])
def admin_prepare_study_b(payload: PipelinePrepareRequest) -> dict[str, Any]:
    raise HTTPException(status_code=405, detail="The administrator console is data-only")
    try:
        return _pipeline_for_run(payload.run_slug).prepare_study_b(
            payload.run_slug,
            payload.selected_n,
            payload.selected_q,
            payload.decision_record,
        )
    except (PipelineError, PriorError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/admin/pipeline/release-reserve", dependencies=[Depends(require_admin)])
def admin_release_study_b_reserve(payload: PipelineReserveRequest) -> dict[str, Any]:
    raise HTTPException(status_code=405, detail="The administrator console is data-only")
    try:
        return _pipeline_for_run(payload.run_slug).release_study_b_reserve(
            payload.run_slug, payload.reason
        )
    except (PipelineError, PriorError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/admin/orchestration", dependencies=[Depends(require_admin)])
def admin_orchestration_status() -> dict[str, Any]:
    return orchestrator.status()


@app.post("/api/admin/orchestration/auto", dependencies=[Depends(require_admin)])
def admin_orchestration_auto(payload: OrchestrationAutoRequest) -> dict[str, Any]:
    return orchestrator.set_auto_enabled(payload.enabled, actor="admin")


@app.post("/api/admin/orchestration/pause", dependencies=[Depends(require_admin)])
def admin_orchestration_pause() -> dict[str, Any]:
    try:
        return orchestrator.pause(actor="admin")
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/admin/orchestration/resume", dependencies=[Depends(require_admin)])
def admin_orchestration_resume() -> dict[str, Any]:
    try:
        return orchestrator.resume(actor="admin")
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/admin/orchestration/early-stop", dependencies=[Depends(require_admin)])
def admin_orchestration_early_stop() -> dict[str, Any]:
    return orchestrator.early_stop(actor="admin")


@app.post("/api/admin/orchestration/refresh-share-fit", dependencies=[Depends(require_admin)])
def admin_orchestration_refresh_share_fit() -> dict[str, Any]:
    try:
        return orchestrator.refresh_share_fit()
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/admin/orchestration/lock-share-fit", dependencies=[Depends(require_admin)])
def admin_orchestration_lock_share_fit(
    payload: OrchestrationLockRequest,
) -> dict[str, Any]:
    try:
        return orchestrator.lock_share_fit(
            actor="admin", donor_count=payload.donor_count
        )
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/admin/orchestration/blocked-slots", dependencies=[Depends(require_admin)])
def admin_orchestration_blocked_slots(
    payload: OrchestrationBlockedSlotsRequest,
) -> dict[str, Any]:
    try:
        return orchestrator.replace_operator_blocked_slots(
            payload.date_cst,
            payload.scheduled_start_ms,
            actor="admin",
        )
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/admin/session-lock", dependencies=[Depends(require_admin)])
def admin_session_lock_get() -> dict[str, Any]:
    return {"ok": True, **orchestrator.peek_session_lock(include_password=True)}


@app.post("/api/admin/session-lock", dependencies=[Depends(require_admin)])
def admin_session_lock_set(payload: SessionLockRequest) -> dict[str, Any]:
    try:
        windows = [
            (int(row.start_ms), int(row.end_ms)) for row in payload.windows
        ]
        return {"ok": True, **orchestrator.set_session_lock(
            payload.start_ms,
            payload.end_ms,
            windows=windows or None,
            actor="admin",
        )}
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/admin/session-lock", dependencies=[Depends(require_admin)])
def admin_session_lock_clear() -> dict[str, Any]:
    return {"ok": True, **orchestrator.clear_session_lock(actor="admin")}


@app.get("/api/admin/orchestration/share-fit-curve", dependencies=[Depends(require_admin)])
def admin_orchestration_share_fit_curve() -> dict[str, Any]:
    curve = orchestrator.share_fit_curve()
    if not curve.get("points"):
        raise HTTPException(
            status_code=409,
            detail="尚无已完成的 share 拟合受试者，无法生成 tap 频次曲线",
        )
    return curve


@app.get("/api/admin/orchestration/share-fit-weights", dependencies=[Depends(require_admin)])
def admin_orchestration_share_fit_weights() -> StreamingResponse:
    """Read already-written share-fit weights and stream a ZIP. Never writes storage."""
    try:
        snapshot = orchestrator.existing_share_fit_weight_snapshot()
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    exported_ms = int(time.time() * 1000)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = (
        f"share-fit-weights-{settings.instance_id}-n{snapshot['donor_count']}-"
        f"{snapshot['prior_id'][:8]}-{stamp}.zip"
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(snapshot["head_name"], snapshot["head_bytes"])
        archive.writestr(snapshot["pointer_name"], snapshot["pointer_bytes"])
        archive.writestr(
            "backup-manifest.json",
            json.dumps(
                {
                    "schema": "tap-to-adapt-share-fit-weight-backup/v1",
                    "mode": "read_only_copy",
                    "source_files_modified": False,
                    "instance_id": settings.instance_id,
                    "exported_ms": exported_ms,
                    "source": snapshot["source"],
                    "prior_id": snapshot["prior_id"],
                    "run_id": snapshot["run_id"],
                    "donor_count": snapshot["donor_count"],
                    "donor_ids": snapshot["donor_ids"],
                    "algorithm": snapshot["algorithm"],
                    "locked": snapshot["locked"],
                    "head_sha256": snapshot["head_sha256"],
                    "pointer_sha256": snapshot["pointer_sha256"],
                    "source_head_path": snapshot["head_path"],
                    "source_pointer_path": snapshot["pointer_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{filename}\"; "
                f"filename*=UTF-8''{quote(filename)}"
            ),
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/admin/export", dependencies=[Depends(require_admin)])
def admin_export() -> StreamingResponse:
    output = io.BytesIO()
    tables = [
        "study_runs",
        "participants",
        "sessions",
        "pair_intervals",
        "events",
        "quality_checks",
        "responses",
        "final_questionnaire_recoveries",
        "audio_files",
        "shared_priors",
        "model_snapshots",
        "pipeline_actions",
        "prior_probe_assignments",
        "model_counters",
        "participant_calibrations",
        "account_batches",
    ]
    hidden_ids = orchestrator.admin_hidden_participant_ids()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        conversation_events: list[dict[str, Any]] = []
        for table in tables:
            rows = db.table_rows(table)
            if hidden_ids and table == "participants":
                rows = [row for row in rows if str(row.get("id") or "") not in hidden_ids]
            elif hidden_ids and table != "study_runs":
                rows = [
                    row
                    for row in rows
                    if str(row.get("participant_id") or "") not in hidden_ids
                ]
            if table == "events":
                conversation_events = [
                    row for row in rows if str(row.get("event_type")) == "chat.turn"
                ]
            csv_buffer = io.StringIO(newline="")
            fieldnames = list(rows[0].keys()) if rows else []
            if fieldnames:
                writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            archive.writestr(f"{table}.csv", csv_buffer.getvalue().encode("utf-8-sig"))
        conversation_rows = _admin_export_conversation_rows(conversation_events)
        conversation_buffer = io.StringIO(newline="")
        conversation_writer = csv.DictWriter(
            conversation_buffer,
            fieldnames=list(_ADMIN_CONVERSATION_EXPORT_FIELDS),
        )
        conversation_writer.writeheader()
        conversation_writer.writerows(conversation_rows)
        archive.writestr(
            "conversation_transcript.csv",
            conversation_buffer.getvalue().encode("utf-8-sig"),
        )
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "exported_ms": int(time.time() * 1000),
                    "instance_id": settings.instance_id,
                    "storage_namespace": str(settings.data_dir.resolve()),
                    "conversation_transcript": {
                        "file": "conversation_transcript.csv",
                        "row_count": len(conversation_rows),
                        "roles": [
                            "使用者",
                            "电脑实验系统agent",
                            "手机陪伴系统agent",
                        ],
                        "timestamps": [
                            "timestamp_ms",
                            "timestamp_iso_utc",
                            "timestamp_iso_asia_shanghai",
                        ],
                    },
                    "consent_version": protocol.consent_version,
                    "consent_hash": protocol.consent_hash,
                    "questionnaire_version": protocol.questionnaires["version"],
                    "topics_version": protocol.topics["version"],
                    "study_protocol_version": protocol.study_version,
                    "study_runs": [
                        {
                            "id": row["id"],
                            "slug": row["slug"],
                            "data_origin": row["data_origin"],
                            "protocol_version": row["protocol_version"],
                            "protocol_sha256": row.get("protocol_sha256"),
                            "protocol_mode": row["protocol_mode"],
                        }
                        for row in db.table_rows("study_runs")
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    output.seek(0)
    filename = time.strftime("tap-to-adapt-user-study-data-%Y%m%d-%H%M%S.zip")
    return StreamingResponse(
        output,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
