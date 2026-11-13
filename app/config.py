"""Environment-backed configuration and runtime-readiness checks.

Settings resolves project resources, per-instance storage, optional model and
speech providers, and independent authentication secrets. Readiness reports
missing dependencies without silently replacing the requested model family.
"""

from __future__ import annotations

import os
import json
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .integrity import file_sha256


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in os.getenv(name, "").split(",")
        if item.strip()
    )


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int = 16) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None or not str(raw).strip() else int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    static_dir: Path
    content_dir: Path
    media_dir: Path
    token_secret: str
    code_pepper: str
    ip_pepper: str
    admin_token: str
    formal_study_enabled: bool
    formal_readiness_enforced: bool
    approved_consent_version: str
    agent_mode: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_provider_order: tuple[str, ...]
    llm_allow_fallbacks: bool
    llm_reasoning_enabled: bool
    asr_mode: str
    server_asr_enabled: bool
    server_asr_model: str
    server_asr_device: str
    server_asr_devices: tuple[str, ...]
    server_asr_download_root: str
    server_asr_max_seconds: float
    server_asr_min_rms: float
    server_asr_replicas: int
    tts_mode: str
    xunfei_app_id: str
    xunfei_api_key: str
    xunfei_api_secret: str
    xunfei_voice: str
    xunfei_primary_voice: str
    xunfei_companion_voice: str
    sherpa_tts_model_dir: str
    sherpa_tts_speaker_id: int
    sherpa_tts_primary_speaker_id: int
    sherpa_tts_companion_speaker_id: int
    sherpa_tts_speed: float
    sherpa_tts_num_threads: int
    ffmpeg_binary: str
    tts_ready_sentinel: str
    base_encoder_checkpoint: str
    base_main_checkpoint: str
    matched_encoder_checkpoint: str
    matched_main_checkpoint: str
    matched_model_manifest: str
    matched_model_ready_sentinel: str
    background_manifest: str
    model_ready_sentinel: str
    model_code_dir: str
    public_base_url: str
    max_audio_upload_bytes: int
    distilhubert_release_manifest: str
    distilhubert_cache_dir: str
    distilhubert_allow_download: bool
    distilhubert_device: str
    distilhubert_encoder_devices: tuple[str, ...]
    distilhubert_encoder_replicas: int
    tap_history_enabled: bool
    tap_history_variant: str
    reply_needed_model: str
    compensation_vault_key: str
    consent_evidence_signing_key: str
    instance_id: str
    public_enrollment_enabled: bool
    public_enrollment_run_slug: str
    public_enrollment_cohort: str
    public_enrollment_group: str
    listener_role: str
    max_concurrent_study_sessions: int
    capacity_retry_after_minutes: int
    share_global_coordination_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        data_dir = Path(os.getenv("STUDY_DATA_DIR", root / "data")).resolve()
        media_dir = Path(os.getenv("STUDY_MEDIA_DIR", data_dir / "media")).resolve()
        instance_id = os.getenv("STUDY_INSTANCE_ID", "test").strip().lower()
        if instance_id not in {"formal", "test"}:
            raise ValueError("STUDY_INSTANCE_ID must be 'formal' or 'test'")
        default_cohort = "study_a" if instance_id == "formal" else "pilot"
        # Share-fit is serialised separately. Once orchestration enters the
        # three-arm phase, both instances admit at most six active experiments.
        default_max_concurrent = "6"
        return cls(
            root=root,
            data_dir=data_dir,
            static_dir=root / "static",
            content_dir=root / "content",
            media_dir=media_dir,
            token_secret=os.getenv("STUDY_TOKEN_SECRET", "dev-only-change-me"),
            code_pepper=os.getenv("STUDY_CODE_PEPPER", "dev-code-pepper"),
            ip_pepper=os.getenv("STUDY_IP_PEPPER", "dev-ip-pepper"),
            admin_token=os.getenv("STUDY_ADMIN_TOKEN", "dev-admin-token"),
            formal_study_enabled=_bool_env("FORMAL_STUDY_ENABLED", False),
            formal_readiness_enforced=_bool_env(
                "FORMAL_READINESS_ENFORCED", True
            ),
            approved_consent_version=os.getenv("APPROVED_CONSENT_VERSION", ""),
            agent_mode=os.getenv("AGENT_MODE", "mock").strip().lower(),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv(
                "LLM_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-4.1-mini"),
            llm_provider_order=_csv_env("LLM_PROVIDER_ORDER"),
            llm_allow_fallbacks=_bool_env("LLM_ALLOW_FALLBACKS", True),
            llm_reasoning_enabled=_bool_env("LLM_REASONING_ENABLED", True),
            asr_mode=os.getenv("ASR_MODE", "mock").strip().lower(),
            server_asr_enabled=_bool_env("SERVER_ASR_ENABLED", False),
            server_asr_model=os.getenv("SERVER_ASR_MODEL", "large-v3").strip(),
            server_asr_device=os.getenv("SERVER_ASR_DEVICE", "cuda").strip().lower(),
            server_asr_devices=_csv_env("SERVER_ASR_DEVICES")
            or (
                ("cuda:0", "cuda:1", "cuda:3")
                if instance_id == "formal"
                else ("cpu",)
            ),
            server_asr_download_root=os.getenv(
                "SERVER_ASR_DOWNLOAD_ROOT", str(Path.home() / ".cache" / "whisper")
            ).strip(),
            server_asr_max_seconds=max(
                1.0, float(os.getenv("SERVER_ASR_MAX_SECONDS", "15"))
            ),
            server_asr_min_rms=max(
                0.0, float(os.getenv("SERVER_ASR_MIN_RMS", "0.001"))
            ),
            server_asr_replicas=_int_env(
                "SERVER_ASR_REPLICAS", 6 if instance_id == "formal" else 1
            ),
            tts_mode=os.getenv("TTS_MODE", "browser").strip().lower(),
            xunfei_app_id=os.getenv("XUNFEI_APP_ID", ""),
            xunfei_api_key=os.getenv("XUNFEI_API_KEY", ""),
            xunfei_api_secret=os.getenv("XUNFEI_API_SECRET", ""),
            xunfei_voice=os.getenv("XUNFEI_VOICE", "xiaoyan"),
            xunfei_primary_voice=os.getenv(
                "XUNFEI_PRIMARY_VOICE",
                os.getenv("XUNFEI_VOICE", "xiaoyan"),
            ),
            xunfei_companion_voice=os.getenv(
                "XUNFEI_COMPANION_VOICE",
                "xiaoyu",
            ),
            sherpa_tts_model_dir=os.getenv(
                "SHERPA_TTS_MODEL_DIR",
                str(data_dir / "tts" / "vits-icefall-zh-aishell3"),
            ),
            sherpa_tts_speaker_id=int(os.getenv("SHERPA_TTS_SPEAKER_ID", "116")),
            sherpa_tts_primary_speaker_id=int(
                os.getenv(
                    "SHERPA_TTS_PRIMARY_SPEAKER_ID",
                    os.getenv("SHERPA_TTS_SPEAKER_ID", "116"),
                )
            ),
            sherpa_tts_companion_speaker_id=int(
                os.getenv("SHERPA_TTS_COMPANION_SPEAKER_ID", "92")
            ),
            sherpa_tts_speed=float(os.getenv("SHERPA_TTS_SPEED", "0.94")),
            sherpa_tts_num_threads=int(os.getenv("SHERPA_TTS_NUM_THREADS", "4")),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
            tts_ready_sentinel=os.getenv(
                "TTS_READY_SENTINEL", str(data_dir / "tts-self-test.json")
            ),
            base_encoder_checkpoint=os.getenv("BASE_ENCODER_CHECKPOINT", ""),
            base_main_checkpoint=os.getenv("BASE_MAIN_CHECKPOINT", ""),
            matched_encoder_checkpoint=os.getenv("MATCHED_ENCODER_CHECKPOINT", ""),
            matched_main_checkpoint=os.getenv("MATCHED_MAIN_CHECKPOINT", ""),
            matched_model_manifest=os.getenv(
                "MATCHED_MODEL_MANIFEST", str(data_dir / "matched-model-manifest.json")
            ),
            matched_model_ready_sentinel=os.getenv(
                "MATCHED_MODEL_READY_SENTINEL",
                str(data_dir / "matched-model-self-test.json"),
            ),
            background_manifest=os.getenv(
                "BACKGROUND_MANIFEST", str(root / "content" / "background_manifest.json")
            ),
            model_ready_sentinel=os.getenv(
                "MODEL_READY_SENTINEL", str(data_dir / "model-self-test.json")
            ),
            model_code_dir=os.getenv("MODEL_CODE_DIR", str(root.parent)),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8765"),
            max_audio_upload_bytes=int(os.getenv("MAX_AUDIO_UPLOAD_BYTES", str(25 * 1024 * 1024))),
            distilhubert_release_manifest=os.getenv(
                "DISTILHUBERT_RELEASE_MANIFEST", ""
            ),
            distilhubert_cache_dir=os.getenv("DISTILHUBERT_CACHE_DIR", ""),
            distilhubert_allow_download=_bool_env(
                "DISTILHUBERT_ALLOW_DOWNLOAD", False
            ),
            distilhubert_device=os.getenv("DISTILHUBERT_DEVICE", "auto").strip().lower(),
            distilhubert_encoder_devices=_csv_env("DISTILHUBERT_ENCODER_DEVICES")
            or (
                (os.getenv("DISTILHUBERT_DEVICE", "cuda:2").strip().lower() or "cuda:2",)
                if instance_id == "formal"
                else ("cpu",)
            ),
            distilhubert_encoder_replicas=_int_env(
                "DISTILHUBERT_ENCODER_REPLICAS",
                max(1, int(os.getenv("MAX_CONCURRENT_STUDY_SESSIONS", default_max_concurrent)))
                if instance_id == "formal"
                else 1,
            ),
            tap_history_enabled=_bool_env("TAP_HISTORY_ENABLED", True),
            tap_history_variant=os.getenv(
                "TAP_HISTORY_VARIANT", "A10_lr_high_0.05"
            ).strip(),
            reply_needed_model=os.getenv("REPLY_NEEDED_MODEL", ""),
            compensation_vault_key=os.getenv("COMPENSATION_VAULT_KEY", ""),
            consent_evidence_signing_key=os.getenv(
                "CONSENT_EVIDENCE_SIGNING_KEY", ""
            ),
            instance_id=instance_id,
            public_enrollment_enabled=_bool_env("PUBLIC_ENROLLMENT_ENABLED", True),
            public_enrollment_run_slug=os.getenv(
                "PUBLIC_ENROLLMENT_RUN_SLUG", f"public-{instance_id}-v1"
            ).strip(),
            public_enrollment_cohort=os.getenv(
                "PUBLIC_ENROLLMENT_COHORT", default_cohort
            ).strip().lower(),
            public_enrollment_group=os.getenv(
                "PUBLIC_ENROLLMENT_GROUP", "personalized"
            ).strip().lower(),
            listener_role=os.getenv("TAP_LISTENER_ROLE", "full").strip().lower(),
            max_concurrent_study_sessions=max(
                1,
                int(
                    os.getenv(
                        "MAX_CONCURRENT_STUDY_SESSIONS",
                        default_max_concurrent,
                    )
                ),
            ),
            capacity_retry_after_minutes=max(
                5,
                int(os.getenv("CAPACITY_RETRY_AFTER_MINUTES", "30")),
            ),
            share_global_coordination_path=os.getenv(
                "SHARE_GLOBAL_COORDINATION_PATH", ""
            ).strip(),
        )

    @property
    def preload_timing_model(self) -> bool:
        return self.listener_role in {"full", "realtime"}

    @property
    def loads_timing_model_on_demand(self) -> bool:
        return self.listener_role in {"full", "realtime"}

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        marker = self.data_dir / ".study-instance.json"
        expected = {"instance_id": self.instance_id}
        if marker.exists():
            actual = json.loads(marker.read_text(encoding="utf-8"))
            if actual.get("instance_id") != self.instance_id:
                raise RuntimeError(
                    f"data directory belongs to instance {actual.get('instance_id')!r}, "
                    f"not {self.instance_id!r}"
                )
        else:
            marker.write_text(
                json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    @property
    def storage_namespace(self) -> str:
        return hashlib.sha256(str(self.data_dir).encode("utf-8")).hexdigest()[:12]

    def formal_readiness(
        self,
        consent_version: str,
        *,
        consent_document: dict,
        compensation_document: dict,
    ) -> dict[str, bool]:
        from .ethics import (
            compensation_ethics_finalized,
            compensation_recruitment_ready,
            consent_ethics_finalized,
            consent_recruitment_ready,
        )

        background_ready = False
        manifest_path = Path(self.background_manifest)
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                tracks = manifest.get("tracks")
                background_ready = bool(manifest.get("approved_for_distribution")) and bool(tracks)
                for track in tracks or []:
                    relative = str(track["url"]).removeprefix("/static/")
                    target = self.static_dir / relative
                    background_ready = background_ready and (
                        target.is_file()
                        and target.stat().st_size == int(track["bytes"])
                        and file_sha256(target) == str(track["sha256"]).lower()
                    )
            except (KeyError, OSError, TypeError, ValueError):
                background_ready = False

        encoder = Path(self.base_encoder_checkpoint) if self.base_encoder_checkpoint else None
        main_model = Path(self.base_main_checkpoint) if self.base_main_checkpoint else None
        base_model_present = bool(
            encoder and main_model and encoder.is_file() and main_model.is_file()
        )
        model_self_test_passed = False
        sentinel_path = Path(self.model_ready_sentinel)
        if base_model_present and sentinel_path.is_file():
            try:
                sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
                model_self_test_passed = (
                    sentinel.get("passed") is True
                    and Path(str(sentinel.get("encoder_path"))).resolve() == encoder.resolve()
                    and Path(str(sentinel.get("main_path"))).resolve() == main_model.resolve()
                    and sentinel.get("encoder_sha256") == file_sha256(encoder)
                    and sentinel.get("main_sha256") == file_sha256(main_model)
                )
            except (OSError, TypeError, ValueError):
                model_self_test_passed = False

        sherpa_dir = Path(self.sherpa_tts_model_dir)
        ffmpeg_ready = bool(
            Path(self.ffmpeg_binary).is_file()
            if Path(self.ffmpeg_binary).is_absolute()
            else shutil.which(self.ffmpeg_binary)
        )
        sherpa_tts_ready = ffmpeg_ready and all(
            (sherpa_dir / filename).is_file()
            for filename in (
                "model.onnx",
                "lexicon.txt",
                "tokens.txt",
                "phone.fst",
                "date.fst",
                "number.fst",
            )
        )
        tts_self_test_passed = False
        tts_sentinel_path = Path(self.tts_ready_sentinel)
        sherpa_model_path = sherpa_dir / "model.onnx"
        if sherpa_tts_ready and tts_sentinel_path.is_file():
            try:
                sentinel = json.loads(tts_sentinel_path.read_text(encoding="utf-8"))
                tts_self_test_passed = (
                    sentinel.get("passed") is True
                    and sentinel.get("provider") == "sherpa_onnx_vits_aishell3"
                    and sentinel.get("voice")
                    == f"aishell3-sid{self.sherpa_tts_primary_speaker_id}"
                    and sentinel.get("model_sha256") == file_sha256(sherpa_model_path)
                    and int(sentinel.get("sample_rate_hz", 0)) == 16000
                    and sentinel.get("format") == "mp3"
                )
            except (OSError, TypeError, ValueError):
                tts_self_test_passed = False

        return {
            "formal_flag": self.formal_study_enabled,
            "token_secret": self.token_secret != "dev-only-change-me",
            "code_pepper": self.code_pepper != "dev-code-pepper",
            "admin_token": self.admin_token != "dev-admin-token",
            "consent_approved": consent_recruitment_ready(
                consent_document,
                consent_version=consent_version,
                approved_consent_version=self.approved_consent_version,
            ),
            "consent_ethics_finalized": consent_ethics_finalized(
                consent_document,
                consent_version=consent_version,
                approved_consent_version=self.approved_consent_version,
            ),
            "compensation_notice_approved": compensation_recruitment_ready(
                compensation_document
            ),
            "compensation_ethics_finalized": compensation_ethics_finalized(
                compensation_document
            ),
            "consent_evidence_signing": bool(self.consent_evidence_signing_key),
            "agent_configured": self.agent_mode == "dtcn",
            "asr_configured": self.asr_mode != "mock",
            "tts_configured": (
                self.tts_mode == "sherpa_onnx"
                and sherpa_tts_ready
                and tts_self_test_passed
            )
            or (
                self.tts_mode == "xunfei"
                and all(
                    [self.xunfei_app_id, self.xunfei_api_key, self.xunfei_api_secret]
                )
            )
            or (
                self.tts_mode == "hybrid"
                and sherpa_tts_ready
                and tts_self_test_passed
            ),
            "llm_configured": bool(self.llm_api_key),
            "base_model_present": base_model_present,
            "model_code_present": all(
                (Path(self.model_code_dir) / name).is_file()
                for name in ("model.py", "learner.py", "utils.py")
            ),
            "model_self_test_passed": model_self_test_passed,
            "background_approved": background_ready,
            "https_url": self.public_base_url.startswith("https://"),
        }


settings = Settings.from_env()
