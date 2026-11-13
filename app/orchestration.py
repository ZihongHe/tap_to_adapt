"""Per-instance public auto-enrollment orchestration.

When auto mode is enabled (default), public enrollment proceeds in two phases:

1. ``share_fit`` — serial personalised donors start from the latest completed
   prefix prior *at serial-claim / start time* (not at booking) and accumulate
   into a DistilHuBERT personal-head FedAvg prior; relative head change
   estimates N*.
2. ``contrast`` — assign each new seat to the open arm with the fewest
   end-to-end completed people (ties follow ① → ② → ③), using the locked
   fit-moment prior, with a per-arm cap.

State is persisted under ``settings.data_dir`` so formal and test instances remain
mechanically identical but data-isolated.
"""

from __future__ import annotations

import copy
import json
import math
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .distilhubert_online import (
    MODEL_FAMILY,
    DistilHuBERTOnlineModel,
    DistilHuBERTRelease,
    DistilHuBERTUnavailable,
    LinearHeadState,
    write_head_artifact,
    _head_fingerprint,
    _validated_head,
    validated_personal_checkpoint,
)
from .integrity import file_sha256
from .session_lock import SessionLockError, SessionLockStore
from .share_global_coordination import ShareGlobalCoordinator


SCHEMA = "tap-to-adapt-auto-orchestration/v1"
ARM_COLD_LEARN = "cold_learn"
ARM_SHARED_LEARN = "shared_learn"
ARM_SHARED_FROZEN = "shared_frozen"
CONTRAST_ARMS = (ARM_COLD_LEARN, ARM_SHARED_LEARN, ARM_SHARED_FROZEN)
ARM_LABELS = {
    ARM_COLD_LEARN: "① 冷启动模型在线学",
    ARM_SHARED_LEARN: "② share 模型启动模型在线学",
    ARM_SHARED_FROZEN: "③ share 模型启动模型冻结",
}


def choose_contrast_arm(
    completed_counts: dict[str, Any] | None,
    assigned_counts: dict[str, Any] | None,
    caps: dict[str, Any] | None = None,
) -> str | None:
    """Assign the next contrast seat to the arm with fewest e2e completions.

    A missed or aborted assignment does not keep a short completed arm from
    being offered next.  When several open arms share that minimum, follow
    ① → ② → ③.  Already-assigned seat counts are ignored so they cannot
    skip an earlier tied arm.  Caps still use completed counts only.
    """
    del assigned_counts
    open_arms = [
        arm
        for arm in CONTRAST_ARMS
        if int((completed_counts or {}).get(arm, 0)) < int((caps or {}).get(arm, 32))
    ]
    if not open_arms:
        return None
    return min(
        open_arms,
        key=lambda arm: (
            int((completed_counts or {}).get(arm, 0)),
            CONTRAST_ARMS.index(arm),
        ),
    )


DISTILHUBERT_PRIOR_ALGORITHM = (
    "distilhubert-personal-head-equal-fedavg-from-session-end-v1"
)
SHARE_FIT_BUDGET = 0
# Operational share-fit saturation knobs (parameter-space FedAvg path).
MIN_DONORS_FOR_AUTO = 8
STABLE_REQUIRED = 3
REL_DELTA_THRESHOLD = 0.02
MAX_DONORS_FOR_AUTO = 50
BEHAVIOR_WINDOW = 3
# Opening-block lock signal.  Share-fit stop uses block-1 eligible correction
# rate, not full-session taps/min.  A 0.2-point drop is treated as still
# improving; a 2-point rise is the rebound safety limit.  Selection uses the
# start of the latest exact-min Q3 run.  Auto-lock waits for one
# non-overlapping W-window plus PLATEAU_PERSIST_AFTER_NONOVERLAP extra Q3
# values at that min.  A single rebound window is diagnostic only; lock
# requires REBOUND_CONFIRM_WINDOWS consecutive rebound states after the
# first min plus one non-overlapping window.  Do not wait past
# CONFIRM_HORIZON_DONORS.
OPENING_BLOCK_INDEX = 1
OPENING_IMPROVE_EPSILON = 0.002
OPENING_EQUIV_EPSILON = 1e-9
PLATEAU_PERSIST_AFTER_NONOVERLAP = 3
CONFIRM_HORIZON_DONORS = 22
TAP_DELTA_TOLERANCE = 0.02
REBOUND_CONFIRM_WINDOWS = 2
# Engineering-locked effective interaction minutes for per-minute rates.
EFFECTIVE_INTERACTION_MINUTES = 30.0
SHARE_FIT_BLOCK_MINUTES = 6.0
SHARE_FIT_TREND_ALPHA = 0.05
SHARE_FIT_QUESTIONNAIRE_DIMENSIONS = (
    ("predictability", "Predictability 可预测性"),
    ("perceived_control", "Perceived control 掌控感"),
    ("correctability", "Correctability 可纠错性"),
    ("legibility", "Legibility 变化可理解性"),
    ("behavioral_adaptation", "Behavioral adaptation 低行为迁就"),
    ("response_accuracy", "Response accuracy 时机符合度"),
    ("cognitive_demand", "Cognitive demand 低认知负担"),
    ("annoyance", "Annoyance 低烦扰"),
    ("habitability", "Habitability 可习得性"),
    ("speed", "Speed 交互效率"),
    ("likeability", "Likeability 好感"),
)
SHARE_APPOINTMENT_HORIZON_MS = 12 * 60 * 60 * 1000
SHARE_APPOINTMENT_SLOT_MS = 30 * 60 * 1000
SHARE_APPOINTMENT_SPACING_MS = 60 * 60 * 1000
SHARE_APPOINTMENT_GRACE_MS = 10 * 60 * 1000
SHARE_PREDECESSOR_RETRY_MS = 15 * 60 * 1000
# After the predecessor slot actually frees, the arrived waiter has this long
# to re-enter the voucher.  Only rows stamped with the policy below use it.
SHARE_ARRIVAL_ENTRY_MS = SHARE_PREDECESSOR_RETRY_MS
SHARE_ARRIVAL_ENTRY_POLICY = "predecessor_deadline_v1"
# A claimed participant who has created the computer session but has not
# activated the timed experiment must finish consent/setup within 30 minutes.
SHARE_SETUP_TIMEOUT_MS = 30 * 60 * 1000
# A walk-in participant receives enough time to read/sign consent and complete
# setup. Once the study session is created, the claimed serial slot is retained
# for the protocol's bounded one-hour flow.
SHARE_WALK_IN_LEASE_MS = 90 * 60 * 1000
SHARE_CLAIM_LEASE_MS = 90 * 60 * 1000
# The participant-facing protocol is bounded to at most 60 minutes.  Keep a
# further 30-minute recovery margin, then fail closed: an unfinished setup/run
# is aborted and can no longer block the serial model chain indefinitely.
SHARE_SESSION_HARD_TIMEOUT_MS = 90 * 60 * 1000
SHARE_CHAIN_COMMIT_VERSION = 3
SHARE_APPOINTMENT_TZ = timezone(timedelta(hours=8))
SESSION_LOCK_START_MESSAGE = (
    "该时段已锁定集体实验。请在首页底部输入召集密码后再开始，"
    "或改选锁定窗外的预约时间。"
)
SESSION_LOCK_BOOK_MESSAGE = "该时间处于集体实验锁定时段，无法预约。"


class OrchestrationError(RuntimeError):
    """Raised for invalid orchestration transitions or enrollment blocks."""


class ShareAppointmentError(OrchestrationError):
    def __init__(self, code: str, message: str, status: dict[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def _default_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "auto_enabled": True,
        "status": "running",
        "phase": "share_fit",
        "share_fit": {
            "enrolled_participant_ids": [],
            "completed_donor_ids": [],
            "protocol_complete_pending_ids": [],
            "auto_lock_pending_questionnaire_ids": [],
            "model_finalizations": {},
            "chain_commit_version": SHARE_CHAIN_COMMIT_VERSION,
            "last_chain_error": None,
            "candidate_n": None,
            "locked_n": None,
            "locked_prior_id": None,
            "lock_reason": None,
            "parameter_candidate_n": None,
            "behavior_candidate_n": None,
            "delta_history": [],
            "behavior_history": [],
            "min_donors_for_auto": MIN_DONORS_FOR_AUTO,
            "stable_required": STABLE_REQUIRED,
            "rel_delta_threshold": REL_DELTA_THRESHOLD,
            "max_donors_for_auto": MAX_DONORS_FOR_AUTO,
            "behavior_window": BEHAVIOR_WINDOW,
            "opening_improve_epsilon": OPENING_IMPROVE_EPSILON,
            "opening_equiv_epsilon": OPENING_EQUIV_EPSILON,
            "plateau_persist_after_nonoverlap": PLATEAU_PERSIST_AFTER_NONOVERLAP,
            "confirm_horizon_donors": CONFIRM_HORIZON_DONORS,
            "rebound_confirm_windows": REBOUND_CONFIRM_WINDOWS,
            "tap_delta_tolerance": TAP_DELTA_TOLERANCE,
            "appointments": [],
            "current_holder": None,
            "walk_in_granted_participant_ids": [],
            "setup_timed_out_participant_ids": [],
            "voluntarily_exited_participant_ids": [],
            "excluded_donor_ids": [],
            "hidden_from_admin_ids": [],
            "operator_blocked_slots": [],
        },
        "contrast": {
            "next_index": 0,
            "caps": {arm: 32 for arm in CONTRAST_ARMS},
            "counts": {arm: 0 for arm in CONTRAST_ARMS},
            "completed_counts": {arm: 0 for arm in CONTRAST_ARMS},
            "completed_participant_ids": [],
            "assignments": [],
        },
        "history": [],
        "updated_ms": int(time.time() * 1000),
    }


class AutoOrchestrator:
    def __init__(self, settings: Any, db: Any) -> None:
        self.settings = settings
        self.db = db
        self.lock = threading.RLock()
        self.path = Path(settings.data_dir) / "orchestration_state.json"
        self.session_lock = SessionLockStore(
            Path(settings.data_dir) / "session_lock.json"
        )
        self.global_coordination = ShareGlobalCoordinator(
            getattr(settings, "share_global_coordination_path", ""),
            str(getattr(settings, "instance_id", "")),
        )

    def _global_owner_key(self, participant_id: str) -> str:
        return self.global_coordination.owner_key(participant_id)

    def _global_snapshot_locked(
        self, participant_id: str, *, now_ms: int
    ) -> dict[str, Any]:
        return self.global_coordination.snapshot(
            self._global_owner_key(participant_id), now_ms=now_ms
        )

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            state = _default_state()
            self._write(state)
            return state
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = _default_state()
            self._write(state)
            return state
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
            state = _default_state()
            self._write(state)
            return state
        share = dict(raw.get("share_fit") or {})
        # Keep saturation knobs pinned to the current protocol constants so
        # redeploys update live orchestration_state.json without a reset.
        state_need_persist = (
            int(share.get("min_donors_for_auto") or 0) != MIN_DONORS_FOR_AUTO
            or int(share.get("stable_required") or 0) != STABLE_REQUIRED
            or abs(float(share.get("rel_delta_threshold") or 0) - REL_DELTA_THRESHOLD)
            > 1e-12
            or int(share.get("max_donors_for_auto") or 0) != MAX_DONORS_FOR_AUTO
            or int(share.get("behavior_window") or 0) != BEHAVIOR_WINDOW
            or abs(
                float(share.get("opening_improve_epsilon") or 0)
                - OPENING_IMPROVE_EPSILON
            )
            > 1e-12
            or abs(float(share.get("tap_delta_tolerance") or 0) - TAP_DELTA_TOLERANCE)
            > 1e-12
            or abs(
                float(share.get("opening_equiv_epsilon") or 0) - OPENING_EQUIV_EPSILON
            )
            > 1e-18
            or int(share.get("plateau_persist_after_nonoverlap") or 0)
            != PLATEAU_PERSIST_AFTER_NONOVERLAP
            or int(share.get("confirm_horizon_donors") or 0) != CONFIRM_HORIZON_DONORS
            or int(share.get("rebound_confirm_windows") or 0) != REBOUND_CONFIRM_WINDOWS
        )
        share["min_donors_for_auto"] = MIN_DONORS_FOR_AUTO
        share["stable_required"] = STABLE_REQUIRED
        share["rel_delta_threshold"] = REL_DELTA_THRESHOLD
        share["max_donors_for_auto"] = MAX_DONORS_FOR_AUTO
        share["behavior_window"] = BEHAVIOR_WINDOW
        share["opening_improve_epsilon"] = OPENING_IMPROVE_EPSILON
        share["opening_equiv_epsilon"] = OPENING_EQUIV_EPSILON
        share["plateau_persist_after_nonoverlap"] = PLATEAU_PERSIST_AFTER_NONOVERLAP
        share["confirm_horizon_donors"] = CONFIRM_HORIZON_DONORS
        share["rebound_confirm_windows"] = REBOUND_CONFIRM_WINDOWS
        share["tap_delta_tolerance"] = TAP_DELTA_TOLERANCE
        share.setdefault("lock_reason", None)
        share.setdefault("parameter_candidate_n", None)
        share.setdefault("behavior_candidate_n", None)
        share.setdefault("behavior_history", [])
        share.setdefault("appointments", [])
        share.setdefault("current_holder", None)
        share.setdefault("protocol_complete_pending_ids", [])
        share.setdefault("auto_lock_pending_questionnaire_ids", [])
        share.setdefault("model_finalizations", {})
        share.setdefault("chain_commit_version", 0)
        share.setdefault("last_chain_error", None)
        share.setdefault("setup_timed_out_participant_ids", [])
        share.setdefault("voluntarily_exited_participant_ids", [])
        share.setdefault("excluded_donor_ids", [])
        share.setdefault("hidden_from_admin_ids", [])
        share.setdefault("operator_blocked_slots", [])
        if "walk_in_granted_participant_ids" not in share:
            # One-time migration for deployments predating durable walk-in
            # tracking. A historical grant or any already-created study
            # session means this participant must not receive another
            # automatic walk-in lease after expiry or abort.
            granted_ids: list[str] = []
            for row in raw.get("history") or []:
                if (
                    isinstance(row, dict)
                    and str(row.get("action") or "") == "share_serial_walk_in"
                    and row.get("participant_id")
                ):
                    granted_ids.append(str(row["participant_id"]))
            holder = share.get("current_holder")
            if (
                isinstance(holder, dict)
                and str(holder.get("mode") or "").startswith("walk_in")
                and holder.get("participant_id")
            ):
                granted_ids.append(str(holder["participant_id"]))
            for participant_id in share.get("enrolled_participant_ids") or []:
                participant_id = str(participant_id)
                if self._share_study_session(participant_id):
                    granted_ids.append(participant_id)
            share["walk_in_granted_participant_ids"] = list(
                dict.fromkeys(granted_ids)
            )
            state_need_persist = True
        raw["share_fit"] = share
        contrast = dict(raw.get("contrast") or {})
        contrast.setdefault("counts", {arm: 0 for arm in CONTRAST_ARMS})
        contrast.setdefault("completed_counts", {arm: 0 for arm in CONTRAST_ARMS})
        contrast.setdefault("completed_participant_ids", [])
        contrast.setdefault("assignments", [])
        raw["contrast"] = contrast
        raw["_knobs_need_persist"] = state_need_persist
        return raw

    def _write(self, state: dict[str, Any]) -> None:
        state = dict(state)
        state.pop("_knobs_need_persist", None)
        state["updated_ms"] = int(time.time() * 1000)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _append_history(self, state: dict[str, Any], action: str, **payload: Any) -> None:
        history = list(state.get("history") or [])
        history.append(
            {
                "ms": int(time.time() * 1000),
                "action": action,
                **payload,
            }
        )
        state["history"] = history[-200:]

    def _participant_in_current_public_run(self, participant_id: str) -> bool:
        participant = self.db.participant(participant_id)
        return bool(
            participant
            and str(participant.get("run_slug") or "")
            == str(self.settings.public_enrollment_run_slug)
        )

    @staticmethod
    def _share_member(state: dict[str, Any], participant_id: str) -> bool:
        share = state["share_fit"]
        if participant_id in set(share.get("excluded_donor_ids") or []):
            return False
        return participant_id in set(share.get("enrolled_participant_ids") or [])

    def _share_study_session(self, participant_id: str) -> dict[str, Any] | None:
        sessions = [
            row
            for row in self.db.sessions(participant_id)
            if str(row.get("client_role") or "") == "study"
        ]
        return max(sessions, key=lambda row: int(row.get("started_ms") or 0), default=None)

    def _share_session_timeout_window_locked(
        self,
        state: dict[str, Any],
        participant_id: str,
        *,
        appointments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Return the server-authoritative timeout window for an open session."""
        session = self._share_study_session(participant_id)
        if (
            not session
            or str(session.get("state") or "") not in {"started", "active"}
            or session.get("ended_ms") is not None
        ):
            return None
        holder = state["share_fit"].get("current_holder")
        holder_activated_ms = 0
        if (
            isinstance(holder, dict)
            and str(holder.get("participant_id") or "") == participant_id
            and str(holder.get("mode") or "") == "active"
        ):
            holder_activated_ms = int(
                holder.get("activated_ms") or holder.get("claimed_ms") or 0
            )
        activated_ms = int(
            session.get("activated_ms") or holder_activated_ms or 0
        )
        anchor_ms = int(activated_ms or session.get("started_ms") or 0)
        timeout_ms = (
            SHARE_SESSION_HARD_TIMEOUT_MS if activated_ms else SHARE_SETUP_TIMEOUT_MS
        )
        # A setup session that is waiting for its booked slot starts its
        # preparation window from the latest server-confirmed booking/claim
        # boundary.  This is the same anchor used by stale-session cleanup.
        if str(session.get("state") or "") == "started":
            pending_slot = next(
                (
                    row
                    for row in reversed(
                        appointments
                        if appointments is not None
                        else list(state["share_fit"].get("appointments") or [])
                    )
                    if str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "")
                    in {"booked", "claimed", "active"}
                ),
                None,
            )
            if pending_slot is not None:
                anchor_ms = max(
                    anchor_ms,
                    int(pending_slot.get("scheduled_start_ms") or 0),
                    int(pending_slot.get("claimed_ms") or 0),
                    int(pending_slot.get("activated_ms") or 0),
                )
        return {
            "session": session,
            "activated_ms": activated_ms,
            "anchor_ms": anchor_ms,
            "timeout_ms": timeout_ms,
            "deadline_ms": anchor_ms + timeout_ms if anchor_ms else 0,
        }

    @staticmethod
    def _session_belongs_to_appointment(
        session: dict[str, Any] | None,
        appointment: dict[str, Any] | None,
    ) -> bool:
        """Return whether a session state can authoritatively close an appointment.

        A participant can book a new slot after an earlier attempt has already
        ended.  That historical terminal state must not immediately release the
        newly claimed slot.  Per the serial protocol, only a session created at
        or after this appointment's booking boundary belongs to this slot.
        """
        if not session or not appointment:
            return False
        appointment_created_ms = int(appointment.get("created_ms") or 0)
        session_started_ms = int(session.get("started_ms") or 0)
        if appointment_created_ms <= 0:
            return True
        return bool(session_started_ms >= appointment_created_ms)

    def peek_session_lock(self, *, include_password: bool = False) -> dict[str, Any]:
        """Read the lock file only. Never touches orchestration_state.json."""
        return self.session_lock.snapshot(include_password=include_password)

    def set_session_lock(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        *,
        windows: list[tuple[int, int] | dict[str, Any]] | None = None,
        actor: str = "admin",
        password: str | None = None,
    ) -> dict[str, Any]:
        """Write only session_lock.json. Holders, donors, and models stay put."""
        try:
            if windows:
                return self.session_lock.set_windows(
                    windows, actor=actor, password=password
                )
            if start_ms is None or end_ms is None:
                raise SessionLockError("请填写锁定起止时间")
            return self.session_lock.set_window(
                start_ms, end_ms, actor=actor
            )
        except SessionLockError as exc:
            raise OrchestrationError(str(exc)) from exc

    def clear_session_lock(self, *, actor: str = "admin") -> dict[str, Any]:
        """Clear the lock file only. Never writes orchestration_state.json."""
        return self.session_lock.clear(actor=actor)

    def _session_lock_allows_new_start(
        self,
        participant_id: str,
        *,
        now_ms: int,
        password: str | None = None,
    ) -> bool:
        snap = self.session_lock.snapshot(now_ms=now_ms, include_password=True)
        if not snap["active"]:
            return True
        if self.session_lock.verify(password, now_ms=now_ms):
            return True
        participant = self.db.participant(participant_id)
        created_ms = int((participant or {}).get("created_ms") or 0)
        lock_created_ms = int(snap.get("created_ms") or 0)
        return bool(
            created_ms > 0
            and lock_created_ms > 0
            and created_ms < lock_created_ms
            and (now_ms - created_ms) <= SHARE_WALK_IN_LEASE_MS
        )

    def _share_walk_in_available_locked(
        self,
        state: dict[str, Any],
        participant_id: str,
        *,
        now_ms: int | None = None,
        group_session_password: str | None = None,
    ) -> bool:
        """Return eligibility only; never allocate or mutate the serial slot."""
        now_ms = int(now_ms or time.time() * 1000)
        if not self._session_lock_allows_new_start(
            participant_id,
            now_ms=now_ms,
            password=group_session_password,
        ):
            return False
        share = state["share_fit"]
        if not self._share_member(state, participant_id):
            return False
        if state.get("phase") != "share_fit":
            return False
        if state["share_fit"].get("protocol_complete_pending_ids"):
            return False
        if share.get("current_holder"):
            return False
        if participant_id in set(share.get("completed_donor_ids") or []):
            return False
        if participant_id in set(
            share.get("voluntarily_exited_participant_ids") or []
        ):
            return False
        if participant_id in set(
            share.get("walk_in_granted_participant_ids") or []
        ):
            return False
        # A study session proves an earlier flow already reached the claim
        # boundary, even if a legacy state file lacks a grant-history row.
        if self._share_study_session(participant_id):
            return False
        if any(
            str(row.get("participant_id") or "") == participant_id
            and str(row.get("status") or "") in {"booked", "claimed", "active"}
            for row in share.get("appointments") or []
        ):
            return False
        # Reserve the full +/- one-hour neighbourhood of every live booking.
        # In particular, a booking whose start has just passed remains live
        # through its ten-minute arrival grace and must block a new walk-in.
        return not any(
            str(row.get("status") or "") == "booked"
            and int(row.get("scheduled_start_ms") or 0) > 0
            and (
                int(row.get("arrived_ms") or 0) > 0
                or now_ms
                <= int(row.get("scheduled_start_ms") or 0)
                + int(row.get("grace_ms") or SHARE_APPOINTMENT_GRACE_MS)
            )
            and abs(now_ms - int(row.get("scheduled_start_ms") or 0))
            < SHARE_APPOINTMENT_SPACING_MS
            for row in share.get("appointments") or []
        )

    @staticmethod
    def _new_appointment_voucher_locked(share: dict[str, Any]) -> str:
        existing = {
            str(row.get("voucher") or "")
            for row in (share.get("appointments") or [])
            if isinstance(row, dict) and row.get("voucher")
        }
        for _ in range(100):
            voucher = f"{secrets.randbelow(10_000):04d}"
            if voucher not in existing:
                return voucher
        for value in range(10_000):
            voucher = f"{value:04d}"
            if voucher not in existing:
                return voucher
        raise OrchestrationError("预约凭证暂时不可用，请联系研究团队。")

    @staticmethod
    def _record_share_walk_in_grant_locked(
        state: dict[str, Any], participant_id: str
    ) -> None:
        granted = list(
            state["share_fit"].get("walk_in_granted_participant_ids") or []
        )
        if participant_id not in granted:
            granted.append(participant_id)
        state["share_fit"]["walk_in_granted_participant_ids"] = granted[-300:]

    def _predecessor_deadline_ms_locked(
        self,
        state: dict[str, Any],
        participant_id: str,
        *,
        now_ms: int,
        global_snapshot: dict[str, Any] | None = None,
    ) -> int:
        """Earliest time the current occupant must release the serial slot."""
        deadlines: list[int] = []
        holder = state["share_fit"].get("current_holder")
        holder_id = (
            str(holder.get("participant_id") or "")
            if isinstance(holder, dict)
            else ""
        )
        if holder_id and holder_id != participant_id:
            window = self._share_session_timeout_window_locked(state, holder_id)
            if window and int(window.get("deadline_ms") or 0) > 0:
                deadlines.append(int(window["deadline_ms"]))
            expires_ms = int(holder.get("expires_ms") or 0)
            if expires_ms > 0:
                deadlines.append(expires_ms)
            if not deadlines:
                claimed_ms = int(
                    holder.get("activated_ms") or holder.get("claimed_ms") or 0
                )
                timeout_ms = (
                    SHARE_SESSION_HARD_TIMEOUT_MS
                    if str(holder.get("mode") or "") == "active"
                    else SHARE_SETUP_TIMEOUT_MS
                )
                if claimed_ms > 0:
                    deadlines.append(claimed_ms + timeout_ms)
        snapshot = (
            global_snapshot
            if global_snapshot is not None
            else self._global_snapshot_locked(participant_id, now_ms=now_ms)
        )
        for row in snapshot.get("external_holders") or []:
            expires_ms = int(row.get("expires_ms") or 0)
            if expires_ms > 0:
                deadlines.append(expires_ms)
        return min(deadlines) if deadlines else 0

    def _refresh_arrival_entry_windows_locked(
        self,
        state: dict[str, Any],
        appointments: list[dict[str, Any]],
        *,
        now_ms: int,
    ) -> bool:
        """Open or expire the post-predecessor voucher window for new arrivals."""
        changed = False
        snapshot: dict[str, Any] | None = None
        for row in appointments:
            if str(row.get("status") or "") != "booked":
                continue
            if str(row.get("arrival_entry_policy") or "") != SHARE_ARRIVAL_ENTRY_POLICY:
                continue
            if not int(row.get("arrived_ms") or 0):
                continue
            participant_id = str(row["participant_id"])
            if snapshot is None:
                snapshot = self._global_snapshot_locked(
                    participant_id, now_ms=now_ms
                )
            holder = state["share_fit"].get("current_holder")
            holder_id = (
                str(holder.get("participant_id") or "")
                if isinstance(holder, dict)
                else ""
            )
            busy = bool(
                (holder_id and holder_id != participant_id)
                or snapshot.get("external_holders")
            )
            if busy:
                continue
            if not int(row.get("entry_open_ms") or 0):
                row["entry_open_ms"] = now_ms
                changed = True
            if now_ms <= int(row["entry_open_ms"]) + SHARE_ARRIVAL_ENTRY_MS:
                continue
            row["status"] = "released"
            row["released_ms"] = now_ms
            row["release_reason"] = "arrival_entry_expired"
            self.global_coordination.release_reservation(
                str(row.get("appointment_id") or "")
            )
            self._append_history(
                state,
                "share_appointment_arrival_entry_expired",
                participant_id=participant_id,
                appointment_id=row.get("appointment_id"),
                entry_open_ms=row.get("entry_open_ms"),
            )
            changed = True
        return changed

    def _cleanup_share_schedule_locked(
        self, state: dict[str, Any], *, now_ms: int | None = None
    ) -> bool:
        now_ms = int(now_ms or time.time() * 1000)
        share = state["share_fit"]
        changed = False
        appointments = [
            dict(row)
            for row in (share.get("appointments") or [])
            if isinstance(row, dict) and row.get("participant_id")
        ]
        for row in appointments:
            if (
                str(row.get("status") or "booked")
                in {"booked", "claimed", "active"}
                and not str(row.get("voucher") or "").isdigit()
            ):
                shadow_share = {**share, "appointments": appointments}
                row["voucher"] = self._new_appointment_voucher_locked(shadow_share)
                changed = True
        enrolled_ids = [
            str(participant_id)
            for participant_id in (share.get("enrolled_participant_ids") or [])
        ]
        completion_records = self.db.end_to_end_completion_records(
            list(
                dict.fromkeys(
                    enrolled_ids
                    + [str(row["participant_id"]) for row in appointments]
                )
            )
        )
        committed_ids = set(share.get("completed_donor_ids") or [])

        # A legacy or abandoned session must not become a permanent mutex.  In
        # particular, sessions created before the appointment scheduler may be
        # active in SQLite without owning current_holder.  Retire every stale
        # open session in the current Share chain before evaluating holders.
        for participant_id in enrolled_ids:
            if participant_id in completion_records:
                continue
            timeout_window = self._share_session_timeout_window_locked(
                state, participant_id, appointments=appointments
            )
            if timeout_window is None:
                continue
            session = timeout_window["session"]
            activated_ms = int(timeout_window["activated_ms"])
            anchor_ms = int(timeout_window["anchor_ms"])
            timeout_ms = int(timeout_window["timeout_ms"])
            if not anchor_ms or now_ms <= anchor_ms + timeout_ms:
                continue
            prior_state = str(session.get("state") or "")
            self.db.abort_session(str(session["id"]), participant_id)
            for row in appointments:
                if (
                    str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "")
                    in {"booked", "claimed", "active"}
                ):
                    row["status"] = "released"
                    row["released_ms"] = now_ms
                    row["release_reason"] = (
                        "session_hard_timeout"
                        if activated_ms
                        else "setup_over_30_minutes"
                    )
            holder = share.get("current_holder")
            if isinstance(holder, dict) and str(
                holder.get("participant_id") or ""
            ) == participant_id:
                share["current_holder"] = None
            self._append_history(
                state,
                "share_stale_session_released",
                participant_id=participant_id,
                session_id=str(session["id"]),
                prior_state=prior_state,
                anchor_ms=anchor_ms,
                timeout_ms=timeout_ms,
            )
            if not activated_ms:
                expired = list(
                    share.get("setup_timed_out_participant_ids") or []
                )
                if participant_id not in expired:
                    expired.append(participant_id)
                share["setup_timed_out_participant_ids"] = expired[-300:]
            changed = True

        for row in appointments:
            participant_id = str(row["participant_id"])
            status = str(row.get("status") or "booked")
            if participant_id in committed_ids and status not in {
                "completed",
                "released",
            }:
                row["status"] = "completed"
                row["completed_ms"] = int(
                    completion_records.get(participant_id, {}).get("completed_ms")
                    or now_ms
                )
                changed = True
                continue
            if status == "booked":
                grace_deadline = int(row.get("scheduled_start_ms") or 0) + (
                    int(row.get("grace_ms") or SHARE_APPOINTMENT_GRACE_MS)
                )
                if (
                    grace_deadline
                    and now_ms > grace_deadline
                    and not int(row.get("arrived_ms") or 0)
                ):
                    row["status"] = "released"
                    row["released_ms"] = now_ms
                    row["release_reason"] = "late_over_10_minutes"
                    changed = True
            elif status in {"claimed", "active"}:
                session = self._share_study_session(participant_id)
                if (
                    session
                    and str(session.get("state") or "") == "aborted"
                    and self._session_belongs_to_appointment(session, row)
                ):
                    row["status"] = "released"
                    row["released_ms"] = now_ms
                    row["release_reason"] = "session_aborted"
                    changed = True
                else:
                    expires_ms = int(row.get("claim_expires_ms") or 0)
                    protocol_complete_pending = bool(
                        participant_id in completion_records
                        and participant_id not in committed_ids
                    )
                    session_open = bool(
                        session
                        and str(session.get("state") or "") in {"started", "active"}
                        and session.get("ended_ms") is None
                    )
                    if (
                        expires_ms
                        and now_ms > expires_ms
                        and not session_open
                        and not protocol_complete_pending
                    ):
                        row["status"] = "released"
                        row["released_ms"] = now_ms
                        row["release_reason"] = "serial_slot_timeout"
                        changed = True

        holder = share.get("current_holder")
        if isinstance(holder, dict) and holder.get("participant_id"):
            holder_id = str(holder["participant_id"])
            holder_appointment = next(
                (
                    row
                    for row in reversed(appointments)
                    if str(row.get("participant_id") or "") == holder_id
                    and str(row.get("status") or "")
                    in {"booked", "claimed", "active", "completed", "released"}
                ),
                None,
            )
            session = self._share_study_session(holder_id)
            holder_committed = holder_id in committed_ids
            holder_protocol_complete_pending = bool(
                holder_id in completion_records and not holder_committed
            )
            holder_expired = (
                int(holder.get("expires_ms") or 0) > 0
                and now_ms > int(holder.get("expires_ms") or 0)
            )
            session_open = bool(
                session
                and str(session.get("state") or "") in {"started", "active"}
                and session.get("ended_ms") is None
            )
            appointment_closed = bool(
                holder_appointment
                and str(holder_appointment.get("status") or "")
                in {"completed", "released"}
            )
            holder_aborted = bool(
                session
                and str(session.get("state") or "") == "aborted"
                and (
                    holder_appointment is None
                    or self._session_belongs_to_appointment(
                        session, holder_appointment
                    )
                )
            )
            legacy_pre_contact_walk_in = bool(
                str(holder.get("mode") or "") == "walk_in" and session is None
            )
            if (
                holder_committed
                or holder_aborted
                or appointment_closed
                or legacy_pre_contact_walk_in
                or (
                    holder_expired
                    and not session_open
                    and not holder_protocol_complete_pending
                )
            ):
                share["current_holder"] = None
                if legacy_pre_contact_walk_in:
                    self._append_history(
                        state,
                        "share_legacy_pre_contact_holder_released",
                        participant_id=holder_id,
                    )
                changed = True

        # Recover an active Share session after a process restart. This is the
        # final server-side guard against two chain participants running at once.
        if not share.get("current_holder"):
            for participant_id in share.get("enrolled_participant_ids") or []:
                session = self._share_study_session(str(participant_id))
                if not session:
                    continue
                activated_ms = int(session.get("activated_ms") or 0)
                if (
                    str(session.get("state") or "") == "active"
                    and session.get("ended_ms") is None
                    and activated_ms
                    and now_ms <= activated_ms + SHARE_CLAIM_LEASE_MS
                ):
                    share["current_holder"] = {
                        "participant_id": str(participant_id),
                        "mode": "recovered_active",
                        "reference_start_ms": activated_ms,
                        "claimed_ms": activated_ms,
                        "expires_ms": activated_ms + SHARE_CLAIM_LEASE_MS,
                    }
                    changed = True
                    break
        changed = (
            self._refresh_arrival_entry_windows_locked(
                state, appointments, now_ms=now_ms
            )
            or changed
        )
        share["appointments"] = appointments[-300:]
        return changed

    def _latest_chain_prior_id_locked(self, state: dict[str, Any]) -> str | None:
        completed = list(state["share_fit"].get("completed_donor_ids") or [])
        if not completed:
            return None
        try:
            prior = self._ensure_prior_for_donors(completed)
        except OrchestrationError as exc:
            state["share_fit"]["last_chain_error"] = str(exc)
            return None
        return str(prior["id"])

    def _share_study_session_is_live(self, participant_id: str) -> bool:
        session = self._share_study_session(participant_id)
        return bool(
            session
            and str(session.get("state") or "")
            in {"started", "active", "ended", "completed", "questionnaire"}
        )

    def _can_rebind_stale_share_start_prior_locked(
        self, state: dict[str, Any], participant_id: str
    ) -> bool:
        """True when a booked/aborted number may move to the current chain tip."""
        share = state["share_fit"]
        if participant_id in set(share.get("completed_donor_ids") or []):
            return False
        if participant_id in set(share.get("voluntarily_exited_participant_ids") or []):
            return False
        if participant_id in set(share.get("setup_timed_out_participant_ids") or []):
            return False
        if self.db.participant_end_to_end_complete(participant_id):
            return False
        return not self._share_study_session_is_live(participant_id)

    def _archive_stale_personal_start_artifacts(
        self, participant: dict[str, Any]
    ) -> list[str]:
        """Keep a previous attempt's personal head from seeding the new start."""
        participant_id = str(participant.get("id") or "")
        if not participant_id:
            return []
        candidates = [
            self.settings.data_dir / "models" / participant_id,
            self.settings.data_dir
            / "personal_debug"
            / "public_enrollment"
            / participant_id
            / "models",
        ]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        archived: list[str] = []
        for source in candidates:
            if not source.is_dir():
                continue
            target = source.with_name(f"{source.name}.pre-rebind-{stamp}")
            suffix = 1
            while target.exists():
                suffix += 1
                target = source.with_name(f"{source.name}.pre-rebind-{stamp}-{suffix}")
            source.rename(target)
            archived.append(str(target))
        return archived

    def _bind_current_chain_start_prior_locked(
        self,
        state: dict[str, Any],
        participant_id: str,
        chain_prior_id: str | None,
    ) -> str | None:
        """Bind the start model at serial claim time to the current prefix.

        Booking and on-time arrival do not freeze M(n).  If the chain advanced
        while this number waited — or after an aborted earlier attempt — the
        start prior follows the latest committed prefix.  A live experiment is
        never moved.
        """
        if not chain_prior_id:
            return None
        participant = self.db.participant(participant_id)
        if participant is None:
            return chain_prior_id
        assigned = participant.get("prior_id")
        if not assigned:
            self.db.assign_prior(
                participant_id,
                chain_prior_id,
                actor="share-serial-orchestration",
            )
            return chain_prior_id
        if str(assigned) == str(chain_prior_id):
            return chain_prior_id
        if not self._can_rebind_stale_share_start_prior_locked(state, participant_id):
            return None
        rebound = self.db.rebind_share_start_prior(
            participant_id,
            chain_prior_id,
            actor="share-serial-orchestration",
            reason="arrival_uses_latest_committed_prefix",
        )
        archived = self._archive_stale_personal_start_artifacts(participant)
        self._append_history(
            state,
            "share_start_prior_rebound",
            participant_id=participant_id,
            from_prior_id=rebound.get("from_prior_id"),
            to_prior_id=rebound.get("to_prior_id"),
            archived_personal_dirs=archived,
        )
        return chain_prior_id

    def _share_appointment_status_locked(
        self,
        state: dict[str, Any],
        participant_id: str,
        *,
        now_ms: int | None = None,
        include_voucher: bool = False,
        group_session_password: str | None = None,
    ) -> dict[str, Any]:
        now_ms = int(now_ms or time.time() * 1000)
        share = state["share_fit"]
        member = self._share_member(state, participant_id)
        setup_timed_out = participant_id in set(
            share.get("setup_timed_out_participant_ids") or []
        )
        voluntarily_exited = participant_id in set(
            share.get("voluntarily_exited_participant_ids") or []
        )
        end_to_end_complete = bool(
            member and self.db.participant_end_to_end_complete(participant_id)
        )
        chain_committed = bool(
            member
            and participant_id
            in set(share.get("completed_donor_ids") or [])
        )
        holder = share.get("current_holder")
        holder_id = (
            str(holder.get("participant_id") or "")
            if isinstance(holder, dict)
            else ""
        )
        global_snapshot = self._global_snapshot_locked(
            participant_id, now_ms=now_ms
        )
        global_external_holders = list(
            global_snapshot.get("external_holders") or []
        )
        global_self_holder = global_snapshot.get("self_holder")
        global_holder_present = bool(
            global_external_holders or global_self_holder
        )
        own = next(
            (
                row
                for row in reversed(share.get("appointments") or [])
                if str(row.get("participant_id") or "") == participant_id
                and str(row.get("status") or "")
                in {"booked", "claimed", "active"}
            ),
            None,
        )
        status = str(own.get("status") or "unbooked") if own else "unbooked"
        if holder_id == participant_id and not own:
            status = str(holder.get("mode") or "walk_in")
        scheduled_start_ms = int(own.get("scheduled_start_ms") or 0) if own else None
        arrived_ms = int(own.get("arrived_ms") or 0) if own else None
        grace_deadline_ms = (
            scheduled_start_ms + SHARE_APPOINTMENT_GRACE_MS
            if scheduled_start_ms
            else None
        )
        due = bool(
            own
            and status == "booked"
            and scheduled_start_ms is not None
            and scheduled_start_ms <= now_ms <= int(grace_deadline_ms or 0)
        )
        credential_verified_ms = (
            int(own.get("credential_verified_ms") or 0) if own else None
        )
        delayed_retry_after_ms = (
            int(own.get("delayed_retry_after_ms") or 0) if own else None
        )
        holder_available = bool(
            (not holder_id or holder_id == participant_id)
            and not global_external_holders
        )
        walk_in_available = bool(
            not own
            and not global_external_holders
            and self._share_walk_in_available_locked(
                state,
                participant_id,
                now_ms=now_ms,
                group_session_password=group_session_password,
            )
        )
        if voluntarily_exited:
            status = "participant_exited"
            walk_in_available = False
        elif setup_timed_out:
            status = "setup_timed_out"
            walk_in_available = False
        elif end_to_end_complete:
            status = "completed"
            walk_in_available = False
        elif walk_in_available:
            status = "walk_in_available"
        can_start = bool(
            member
            and not end_to_end_complete
            and not setup_timed_out
            and not voluntarily_exited
            and holder_available
            and (
                holder_id == participant_id
                or status in {"claimed", "active"}
                or due
                or walk_in_available
            )
        )
        blocked_starts = sorted(
            {
                int(row.get("scheduled_start_ms") or 0)
                for row in (share.get("appointments") or [])
                if str(row.get("status") or "") in {"booked", "claimed", "active"}
                and str(row.get("participant_id") or "") != participant_id
                and int(row.get("scheduled_start_ms") or 0) > 0
            }
        )
        operator_blocked_starts = _operator_blocked_starts(share)
        scheduling_references = list(blocked_starts)
        if isinstance(holder, dict) and holder_id != participant_id:
            holder_reference = int(holder.get("reference_start_ms") or 0)
            if holder_reference:
                scheduling_references.append(holder_reference)
        scheduling_references.extend(
            int(row.get("reference_start_ms") or 0)
            for row in global_external_holders
            if int(row.get("reference_start_ms") or 0) > 0
        )
        scheduling_references.extend(
            int(row.get("scheduled_start_ms") or 0)
            for row in (global_snapshot.get("appointments") or [])
            if str(row.get("owner_key") or "")
            != self._global_owner_key(participant_id)
            and int(row.get("scheduled_start_ms") or 0) > 0
        )
        earliest_start_ms = now_ms
        for reference in sorted(scheduling_references):
            if abs(earliest_start_ms - reference) < SHARE_APPOINTMENT_SPACING_MS:
                earliest_start_ms = reference + SHARE_APPOINTMENT_SPACING_MS
        slot_start_ms = (
            ((now_ms + 60_000 + SHARE_APPOINTMENT_SLOT_MS - 1)
             // SHARE_APPOINTMENT_SLOT_MS)
            * SHARE_APPOINTMENT_SLOT_MS
        )
        horizon_end_ms = now_ms + SHARE_APPOINTMENT_HORIZON_MS
        time_slots: list[dict[str, Any]] = []
        while slot_start_ms <= horizon_end_ms:
            exact_booking = slot_start_ms in blocked_starts
            operator_blocked = slot_start_ms in operator_blocked_starts
            session_lock_blocked = self.session_lock.blocks_slot(
                slot_start_ms, now_ms=now_ms
            )
            buffer_blocked = any(
                abs(slot_start_ms - reference) < SHARE_APPOINTMENT_SPACING_MS
                for reference in scheduling_references
            )
            time_slots.append(
                {
                    "start_ms": slot_start_ms,
                    "available": (
                        not buffer_blocked
                        and not operator_blocked
                        and not session_lock_blocked
                    ),
                    "reason": (
                        "booked" if exact_booking
                        else "operator_blocked" if operator_blocked
                        else "session_lock" if session_lock_blocked
                        else "buffer" if buffer_blocked
                        else "available"
                    ),
                }
            )
            slot_start_ms += SHARE_APPOINTMENT_SLOT_MS
        predecessor_delay = bool(
            own
            and status == "booked"
            and arrived_ms
            and (
                (holder_id and holder_id != participant_id)
                or global_external_holders
            )
        )
        predecessor_deadline_ms = (
            self._predecessor_deadline_ms_locked(
                state,
                participant_id,
                now_ms=now_ms,
                global_snapshot=global_snapshot,
            )
            if predecessor_delay
            else 0
        )
        entry_open_ms = int(own.get("entry_open_ms") or 0) if own else 0
        arrival_reentry_required = bool(
            own
            and status == "booked"
            and arrived_ms
            and not predecessor_delay
            and str(own.get("arrival_entry_policy") or "")
            == SHARE_ARRIVAL_ENTRY_POLICY
            and entry_open_ms > 0
        )
        entry_deadline_ms = (
            entry_open_ms + SHARE_ARRIVAL_ENTRY_MS
            if arrival_reentry_required and entry_open_ms
            else None
        )
        timeout_window = self._share_session_timeout_window_locked(
            state, participant_id
        )
        setup_timer_active = bool(
            timeout_window
            and not int(timeout_window.get("activated_ms") or 0)
            and not setup_timed_out
        )
        setup_started_ms = (
            int(timeout_window.get("anchor_ms") or 0)
            if setup_timer_active and timeout_window
            else None
        )
        setup_deadline_ms = (
            int(timeout_window.get("deadline_ms") or 0)
            if setup_timer_active and timeout_window
            else None
        )
        return {
            "serial_required": bool(
                member
                and state.get("phase") == "share_fit"
                and not end_to_end_complete
                and not setup_timed_out
                and not voluntarily_exited
            ),
            "phase": state.get("phase"),
            "status": status if member else "not_applicable",
            "appointment_required": bool(
                member
                and state.get("phase") == "share_fit"
                and not end_to_end_complete
                and not setup_timed_out
                and not voluntarily_exited
                and not can_start
                and status not in {"booked", "claimed", "active"}
            ),
            "can_book_future": bool(
                member
                and state.get("phase") == "share_fit"
                and not end_to_end_complete
                and not setup_timed_out
                and not voluntarily_exited
                and status not in {"booked", "claimed", "active"}
            ),
            "end_to_end_complete": end_to_end_complete,
            "setup_timed_out": setup_timed_out,
            "setup_timer_active": setup_timer_active,
            "setup_timeout_ms": SHARE_SETUP_TIMEOUT_MS,
            "setup_started_ms": setup_started_ms,
            "setup_deadline_ms": setup_deadline_ms,
            "setup_remaining_ms": (
                max(0, int(setup_deadline_ms or 0) - now_ms)
                if setup_timer_active
                else None
            ),
            "participant_exited": voluntarily_exited,
            "chain_committed": chain_committed,
            "chain_commit_pending": bool(
                self.db.final_questionnaire_recovery(participant_id)
                and not chain_committed
            ),
            "can_start_setup": can_start,
            "walk_in_available": walk_in_available,
            "holder_present": bool(holder_id or global_holder_present),
            "holder_is_self": bool(
                holder_id == participant_id and not global_external_holders
            ),
            "global_coordination_enabled": bool(global_snapshot.get("enabled")),
            "global_holder_present": global_holder_present,
            "global_external_holder_present": bool(global_external_holders),
            "scheduled_start_ms": scheduled_start_ms,
            "voucher": (
                str(own.get("voucher") or "")
                if own and include_voucher
                else None
            ),
            "credential_verified": bool(credential_verified_ms),
            "credential_verified_ms": credential_verified_ms,
            "arrived_ms": arrived_ms,
            "predecessor_delay": predecessor_delay,
            "predecessor_deadline_ms": (
                predecessor_deadline_ms if predecessor_delay else None
            ),
            "retry_after_ms": (
                predecessor_deadline_ms
                if predecessor_delay and predecessor_deadline_ms
                else delayed_retry_after_ms
                if predecessor_delay
                else None
            ),
            "retry_after_minutes": SHARE_ARRIVAL_ENTRY_MS // 60_000,
            "entry_open_ms": entry_open_ms or None,
            "entry_deadline_ms": entry_deadline_ms,
            "arrival_reentry_required": arrival_reentry_required,
            "grace_deadline_ms": grace_deadline_ms,
            "server_ms": now_ms,
            "horizon_end_ms": horizon_end_ms,
            "earliest_start_ms": min(
                earliest_start_ms, now_ms + SHARE_APPOINTMENT_HORIZON_MS
            ),
            "spacing_ms": SHARE_APPOINTMENT_SPACING_MS,
            "slot_ms": SHARE_APPOINTMENT_SLOT_MS,
            "grace_ms": SHARE_APPOINTMENT_GRACE_MS,
            "blocked_start_ms": blocked_starts,
            "session_lock": self.session_lock.public_status(now_ms=now_ms),
            "time_slots": time_slots,
            "latest_chain_prior_id": self._latest_chain_prior_id_locked(state),
            "completed_donors": len(
                share.get("completed_donor_ids") or []
            ),
        }

    def register_share_enrollment(self, participant_id: str) -> dict[str, Any]:
        """Register visibility without allocating the serial experiment slot."""
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            changed = bool(state.pop("_knobs_need_persist", False))
            changed = self._cleanup_share_schedule_locked(state) or changed
            if reconciled or changed:
                self._write(state)
            return self._share_appointment_status_locked(state, participant_id)

    def is_share_fit_member(self, participant_id: str) -> bool:
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            changed = bool(state.pop("_knobs_need_persist", False))
            changed = self._cleanup_share_schedule_locked(state) or changed
            if reconciled or changed:
                self._write(state)
            return self._share_member(state, participant_id)

    def share_chain_committed(self, participant_id: str) -> bool:
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            if state.pop("_knobs_need_persist", False) or reconciled:
                self._write(state)
            return participant_id in set(
                state["share_fit"].get("completed_donor_ids") or []
            )

    def share_appointment_status(
        self,
        participant_id: str,
        group_session_password: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            changed = bool(state.pop("_knobs_need_persist", False))
            changed = self._cleanup_share_schedule_locked(state) or changed
            if reconciled or changed:
                self._write(state)
            return self._share_appointment_status_locked(
                state,
                participant_id,
                group_session_password=group_session_password,
            )

    def find_appointment_by_voucher(self, voucher: str) -> dict[str, Any] | None:
        submitted = str(voucher or "").strip()
        if not submitted.isdigit() or len(submitted) != 4:
            return None
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            changed = self._cleanup_share_schedule_locked(state)
            if reconciled or changed:
                self._write(state)
            for row in reversed(state["share_fit"].get("appointments") or []):
                if not isinstance(row, dict):
                    continue
                if str(row.get("voucher") or "") != submitted:
                    continue
                status = str(row.get("status") or "")
                return {
                    "participant_id": str(row.get("participant_id") or ""),
                    "appointment_id": str(row.get("appointment_id") or ""),
                    "status": status,
                    "live": status in {"booked", "claimed", "active"},
                }
        return None

    def live_appointment_participant_ids(self) -> set[str]:
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            changed = self._cleanup_share_schedule_locked(state)
            if reconciled or changed:
                self._write(state)
            return {
                str(row.get("participant_id") or "")
                for row in (state["share_fit"].get("appointments") or [])
                if isinstance(row, dict)
                and str(row.get("status") or "") in {"booked", "claimed", "active"}
                and row.get("participant_id")
            }

    def acknowledge_share_appointment_resume(
        self, participant_id: str
    ) -> dict[str, Any]:
        """Allow a resumed participant to continue without retyping the voucher."""
        with self.lock:
            state = self._load()
            self._reconcile_end_to_end_locked(state)
            changed = self._cleanup_share_schedule_locked(state)
            now_ms = int(time.time() * 1000)
            own = next(
                (
                    row
                    for row in reversed(state["share_fit"].get("appointments") or [])
                    if str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "") in {"booked", "claimed", "active"}
                ),
                None,
            )
            if own is not None and not int(own.get("credential_verified_ms") or 0):
                own["credential_verified_ms"] = now_ms
                self._append_history(
                    state,
                    "share_appointment_resume_verified",
                    participant_id=participant_id,
                    appointment_id=own.get("appointment_id"),
                )
                changed = True
            if changed:
                self._write(state)
            return self._share_appointment_status_locked(
                state, participant_id, include_voucher=True
            )

    def can_restart_aborted_session(self, participant_id: str) -> bool:
        """Allow a verified new appointment to supersede an older abort only."""
        with self.lock:
            state = self._load()
            self._reconcile_end_to_end_locked(state)
            changed = self._cleanup_share_schedule_locked(state)
            share = state["share_fit"]
            holder = share.get("current_holder")
            holder_is_self = bool(
                isinstance(holder, dict)
                and str(holder.get("participant_id") or "") == participant_id
            )
            appointment = next(
                (
                    row
                    for row in reversed(share.get("appointments") or [])
                    if str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "") in {"claimed", "active"}
                ),
                None,
            )
            session = self._share_study_session(participant_id)
            allowed = bool(
                state.get("phase") == "share_fit"
                and holder_is_self
                and appointment
                and session
                and str(session.get("state") or "") == "aborted"
                and int(session.get("ended_ms") or 0) > 0
                and int(session.get("ended_ms") or 0)
                < int(appointment.get("created_ms") or 0)
                and participant_id
                not in set(share.get("setup_timed_out_participant_ids") or [])
                and participant_id
                not in set(share.get("voluntarily_exited_participant_ids") or [])
                and not self.db.participant_end_to_end_complete(participant_id)
            )
            if changed:
                self._write(state)
            return allowed

    def book_share_appointment(
        self, participant_id: str, scheduled_start_ms: int
    ) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            self._reconcile_end_to_end_locked(state)
            self._cleanup_share_schedule_locked(state)
            now_ms = int(time.time() * 1000)
            if state.get("phase") != "share_fit" or not self._share_member(
                state, participant_id
            ):
                raise ShareAppointmentError(
                    "share_appointment_not_available",
                    "当前不可预约该用户实验。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if participant_id in set(
                state["share_fit"].get("setup_timed_out_participant_ids") or []
            ):
                raise ShareAppointmentError(
                    "share_setup_timed_out",
                    "此前的实验准备已超过 30 分钟，原匿名编号已释放，请重新开始。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if participant_id in set(
                state["share_fit"].get("voluntarily_exited_participant_ids") or []
            ):
                raise ShareAppointmentError(
                    "share_participant_exited",
                    "该编号已主动退出并立即释放实验位置；如需再次参与，请重新生成编号。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if self.db.participant_end_to_end_complete(participant_id):
                raise ShareAppointmentError(
                    "participant_already_complete",
                    "该参与者编号已完整完成实验，无需预约。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            start_ms = int(scheduled_start_ms)
            if start_ms < now_ms or start_ms > now_ms + SHARE_APPOINTMENT_HORIZON_MS:
                raise ShareAppointmentError(
                    "share_appointment_out_of_window",
                    "请选择从现在起未来 12 小时内的实验开始时间。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if start_ms % SHARE_APPOINTMENT_SLOT_MS != 0:
                raise ShareAppointmentError(
                    "share_appointment_invalid_slot",
                    "请选择页面提供的半小时时间块。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if start_ms in _operator_blocked_starts(state["share_fit"]):
                raise ShareAppointmentError(
                    "share_appointment_operator_blocked",
                    "该时间段已被设为不可预约，请选择其他时间。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if self.session_lock.blocks_slot(start_ms, now_ms=now_ms):
                raise ShareAppointmentError(
                    "share_appointment_session_lock",
                    SESSION_LOCK_BOOK_MESSAGE,
                    self._share_appointment_status_locked(state, participant_id),
                )
            holder = state["share_fit"].get("current_holder")
            holder_id = (
                str(holder.get("participant_id") or "")
                if isinstance(holder, dict)
                else ""
            )
            if holder_id == participant_id:
                raise ShareAppointmentError(
                    "share_slot_already_held",
                    "你已经取得当前实验位置，无需预约。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            references: list[int] = []
            if isinstance(holder, dict) and holder_id:
                reference = int(holder.get("reference_start_ms") or 0)
                if reference:
                    references.append(reference)
            for row in state["share_fit"].get("appointments") or []:
                if (
                    str(row.get("participant_id") or "") != participant_id
                    and str(row.get("status") or "")
                    in {"booked", "claimed", "active"}
                    and int(row.get("scheduled_start_ms") or 0) > 0
                ):
                    references.append(int(row["scheduled_start_ms"]))
            if any(
                abs(start_ms - reference) < SHARE_APPOINTMENT_SPACING_MS
                for reference in references
            ):
                raise ShareAppointmentError(
                    "share_appointment_conflict",
                    "该时间与另一实验位置的开始时间相距不足 1 小时，请选择其他时间。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            for row in state["share_fit"].get("appointments") or []:
                if (
                    str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "") == "booked"
                ):
                    row["status"] = "released"
                    row["released_ms"] = now_ms
                    row["release_reason"] = "participant_rescheduled"
            appointment_id = str(uuid.uuid4())
            voucher = self._new_appointment_voucher_locked(state["share_fit"])
            if not self.global_coordination.book_appointment(
                self._global_owner_key(participant_id),
                appointment_id,
                scheduled_start_ms=start_ms,
                grace_ms=SHARE_APPOINTMENT_GRACE_MS,
                spacing_ms=SHARE_APPOINTMENT_SPACING_MS,
                now_ms=now_ms,
            ):
                raise ShareAppointmentError(
                    "share_appointment_conflict",
                    "该时间与另一实验位置的开始时间相距不足 1 小时，请选择其他时间。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            state["share_fit"].setdefault("appointments", []).append(
                {
                    "appointment_id": appointment_id,
                    "participant_id": participant_id,
                    "scheduled_start_ms": start_ms,
                    "grace_ms": SHARE_APPOINTMENT_GRACE_MS,
                    "voucher": voucher,
                    "status": "booked",
                    "created_ms": now_ms,
                }
            )
            self._append_history(
                state,
                "share_appointment_booked",
                participant_id=participant_id,
                appointment_id=appointment_id,
                scheduled_start_ms=start_ms,
                voucher=voucher,
            )
            self._write(state)
            return self._share_appointment_status_locked(
                state, participant_id, include_voucher=True
            )

    def claim_share_session_start(
        self,
        participant_id: str,
        group_session_password: str | None = None,
    ) -> dict[str, Any]:
        """Atomically claim the serial slot before creating a study session."""
        with self.lock:
            state = self._load()
            self._reconcile_end_to_end_locked(state)
            self._cleanup_share_schedule_locked(state)
            if not self._share_member(state, participant_id):
                return self._share_appointment_status_locked(
                    state,
                    participant_id,
                    group_session_password=group_session_password,
                )
            if state.get("phase") != "share_fit":
                raise ShareAppointmentError(
                    "share_fit_closed",
                    "Share 拟合阶段已经结束，该旧报名编号不能再开始实验。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if participant_id in set(
                state["share_fit"].get("setup_timed_out_participant_ids") or []
            ):
                raise ShareAppointmentError(
                    "share_setup_timed_out",
                    "实验准备超过 30 分钟，原匿名编号已释放，请重新开始。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if participant_id in set(
                state["share_fit"].get("voluntarily_exited_participant_ids") or []
            ):
                raise ShareAppointmentError(
                    "share_participant_exited",
                    "该编号已主动退出并立即释放实验位置；如需再次参与，请重新生成编号。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if self.db.participant_end_to_end_complete(participant_id):
                raise ShareAppointmentError(
                    "participant_already_complete",
                    "该参与者编号已完整完成实验，不能再次开始实验。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if state["share_fit"].get("protocol_complete_pending_ids"):
                raise ShareAppointmentError(
                    "share_chain_commit_pending",
                    "上一位参与者的累计模型正在安全提交，请稍后重试。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            now_ms = int(time.time() * 1000)
            status = self._share_appointment_status_locked(
                state,
                participant_id,
                now_ms=now_ms,
                group_session_password=group_session_password,
            )
            holder = state["share_fit"].get("current_holder")
            holder_id = (
                str(holder.get("participant_id") or "")
                if isinstance(holder, dict)
                else ""
            )
            group_ok = self._session_lock_allows_new_start(
                participant_id,
                now_ms=now_ms,
                password=group_session_password,
            )
            global_snapshot = self._global_snapshot_locked(
                participant_id, now_ms=now_ms
            )
            if global_snapshot.get("external_holders"):
                raise ShareAppointmentError(
                    "share_serial_occupied",
                    "上一位参与者尚未完成，请预约未来 12 小时内的实验时间。",
                    status,
                )
            own = next(
                (
                    row
                    for row in reversed(state["share_fit"].get("appointments") or [])
                    if str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "") == "booked"
                ),
                None,
            )
            if holder_id and holder_id != participant_id:
                raise ShareAppointmentError(
                    "share_serial_occupied",
                    "上一位参与者尚未完成，最新模型尚未释放，即使已到预约时间也请继续等待。",
                    status,
                )
            if holder_id == participant_id:
                group_ok = True
            if not holder_id:
                if own is not None and not group_ok:
                    raise ShareAppointmentError(
                        "session_lock_active",
                        SESSION_LOCK_START_MESSAGE,
                        status,
                    )
                if own is None:
                    if self._share_walk_in_available_locked(
                        state,
                        participant_id,
                        now_ms=now_ms,
                        group_session_password=group_session_password,
                    ):
                        own = {
                            "scheduled_start_ms": now_ms,
                            "appointment_id": None,
                            "_walk_in": True,
                        }
                        self._record_share_walk_in_grant_locked(
                            state, participant_id
                        )
                        self._append_history(
                            state,
                            "share_serial_walk_in",
                            participant_id=participant_id,
                            source="atomic_session_claim",
                        )
                    else:
                        if self.session_lock.is_active(now_ms) and not group_ok:
                            raise ShareAppointmentError(
                                "session_lock_active",
                                SESSION_LOCK_START_MESSAGE,
                                status,
                            )
                        raise ShareAppointmentError(
                        "share_appointment_required",
                        "当前需要先预约未来 12 小时内的实验时间。",
                        status,
                    )
                scheduled = int(own.get("scheduled_start_ms") or 0)
                if now_ms < scheduled:
                    raise ShareAppointmentError(
                        "share_appointment_early",
                        "尚未到预约时间，请在预约时间到达后开始。",
                        status,
                    )
                if now_ms > scheduled + SHARE_APPOINTMENT_GRACE_MS:
                    own["status"] = "released"
                    own["released_ms"] = now_ms
                    own["release_reason"] = "late_over_10_minutes"
                    self._write(state)
                    raise ShareAppointmentError(
                        "share_appointment_released",
                        "已迟到超过 10 分钟，原位置已释放，请重新预约。",
                        self._share_appointment_status_locked(state, participant_id),
                    )
                own["status"] = "claimed"
                own["claimed_ms"] = now_ms
                own["claim_expires_ms"] = now_ms + SHARE_CLAIM_LEASE_MS
                state["share_fit"]["current_holder"] = {
                    "participant_id": participant_id,
                    "mode": (
                        "walk_in_claimed" if own.get("_walk_in") else "appointment"
                    ),
                    "appointment_id": own.get("appointment_id"),
                    "reference_start_ms": scheduled,
                    "claimed_ms": now_ms,
                    "expires_ms": now_ms + SHARE_CLAIM_LEASE_MS,
                }
                if not self.global_coordination.claim_holder(
                    self._global_owner_key(participant_id),
                    reference_start_ms=scheduled,
                    expires_ms=now_ms + SHARE_SETUP_TIMEOUT_MS,
                    now_ms=now_ms,
                ):
                    state["share_fit"]["current_holder"] = None
                    own["status"] = "booked" if own.get("appointment_id") else "released"
                    raise ShareAppointmentError(
                        "share_serial_occupied",
                        "上一位参与者尚未完成，请预约未来 12 小时内的实验时间。",
                        self._share_appointment_status_locked(
                            state, participant_id, now_ms=now_ms
                        ),
                    )
            else:
                holder["claimed_ms"] = now_ms
                holder["expires_ms"] = now_ms + SHARE_CLAIM_LEASE_MS
                if not self.global_coordination.claim_holder(
                    self._global_owner_key(participant_id),
                    reference_start_ms=int(
                        holder.get("reference_start_ms") or now_ms
                    ),
                    expires_ms=now_ms + SHARE_SETUP_TIMEOUT_MS,
                    now_ms=now_ms,
                ):
                    raise ShareAppointmentError(
                        "share_serial_occupied",
                        "上一位参与者尚未完成，请预约未来 12 小时内的实验时间。",
                        self._share_appointment_status_locked(
                            state, participant_id, now_ms=now_ms
                        ),
                    )

            completed_n = len(state["share_fit"].get("completed_donor_ids") or [])
            chain_prior_id = self._latest_chain_prior_id_locked(state)
            if completed_n > 0 and not chain_prior_id:
                self.global_coordination.release_owner(
                    self._global_owner_key(participant_id)
                )
                raise ShareAppointmentError(
                    "share_chain_prior_pending",
                    "上一位参与者的累计模型尚未就绪，即使已到预约时间也请继续等待。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            bound_prior_id = self._bind_current_chain_start_prior_locked(
                state, participant_id, chain_prior_id
            )
            if chain_prior_id and bound_prior_id != chain_prior_id:
                self.global_coordination.release_owner(
                    self._global_owner_key(participant_id)
                )
                raise ShareAppointmentError(
                    "share_chain_prior_mismatch",
                    "当前实验仍绑定着旧的起始模型，不能在进行中改到新的 Share 链。",
                    self._share_appointment_status_locked(
                        state, participant_id
                    ),
                )
            self._append_history(
                state,
                "share_serial_claimed",
                participant_id=participant_id,
                completed_prefix_n=completed_n,
                chain_prior_id=chain_prior_id,
            )
            self._write(state)
            return self._share_appointment_status_locked(state, participant_id)

    def claim_share_appointment_arrival(
        self,
        participant_id: str,
        voucher: str | None,
        group_session_password: str | None = None,
    ) -> dict[str, Any]:
        """Record an on-time booked arrival before unlocking participant setup."""
        with self.lock:
            state = self._load()
            self._reconcile_end_to_end_locked(state)
            self._cleanup_share_schedule_locked(state)
            if participant_id in set(
                state["share_fit"].get("voluntarily_exited_participant_ids") or []
            ):
                raise ShareAppointmentError(
                    "share_participant_exited",
                    "该编号已主动退出并立即释放实验位置；如需再次参与，请重新生成编号。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            now_ms = int(time.time() * 1000)
            status = self._share_appointment_status_locked(
                state,
                participant_id,
                now_ms=now_ms,
                group_session_password=group_session_password,
            )
            holder = state["share_fit"].get("current_holder")
            holder_id = (
                str(holder.get("participant_id") or "")
                if isinstance(holder, dict)
                else ""
            )
            if holder_id == participant_id:
                return status
            if not self._session_lock_allows_new_start(
                participant_id,
                now_ms=now_ms,
                password=group_session_password,
            ):
                raise ShareAppointmentError(
                    "session_lock_active",
                    SESSION_LOCK_START_MESSAGE,
                    status,
                )
            own = next(
                (
                    row
                    for row in reversed(state["share_fit"].get("appointments") or [])
                    if str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "") == "booked"
                ),
                None,
            )
            if own is None:
                raise ShareAppointmentError(
                    "share_appointment_released",
                    "预约位置已释放，请重新预约未来 12 小时内的实验时间。",
                    status,
                )
            submitted_voucher = str(voucher or "").strip()
            expected_voucher = str(own.get("voucher") or "")
            credential_already_verified = bool(
                int(own.get("credential_verified_ms") or 0)
            )
            if (
                (submitted_voucher and submitted_voucher != expected_voucher)
                or (not credential_already_verified and not submitted_voucher)
            ):
                raise ShareAppointmentError(
                    "share_appointment_voucher_invalid",
                    "预约凭证不正确。请输入预约成功时获得的四位凭证；如已遗失，请联系微信 study-contact 获取。",
                    status,
                )
            scheduled = int(own.get("scheduled_start_ms") or 0)
            if now_ms < scheduled:
                raise ShareAppointmentError(
                    "share_appointment_early",
                    "尚未到预约时间，请保持预约等待页面打开。",
                    status,
                )
            arrived_ms = int(own.get("arrived_ms") or 0)
            first_arrival = not arrived_ms
            grace_ms = int(own.get("grace_ms") or SHARE_APPOINTMENT_GRACE_MS)
            if now_ms > scheduled + grace_ms and not arrived_ms:
                own["status"] = "released"
                own["released_ms"] = now_ms
                own["release_reason"] = "late_over_10_minutes"
                self._write(state)
                raise ShareAppointmentError(
                    "share_appointment_released",
                    "已迟到超过 10 分钟，原预约位置已作废，请重新预约。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            if first_arrival:
                own["arrived_ms"] = now_ms
                own["credential_verified_ms"] = now_ms
                own["arrival_entry_policy"] = SHARE_ARRIVAL_ENTRY_POLICY
                self._append_history(
                    state,
                    "share_appointment_arrived",
                    participant_id=participant_id,
                    appointment_id=own.get("appointment_id"),
                    scheduled_start_ms=scheduled,
                )
            reservation_key = str(own.get("appointment_id") or "")
            self.global_coordination.mark_appointment_arrived(
                self._global_owner_key(participant_id),
                reservation_key,
                scheduled_start_ms=scheduled,
                grace_ms=grace_ms,
                now_ms=now_ms,
            )
            global_snapshot = self._global_snapshot_locked(
                participant_id, now_ms=now_ms
            )
            predecessor_deadline_ms = self._predecessor_deadline_ms_locked(
                state,
                participant_id,
                now_ms=now_ms,
                global_snapshot=global_snapshot,
            )
            if holder_id or global_snapshot.get("external_holders"):
                own["delayed_retry_after_ms"] = predecessor_deadline_ms
                self._append_history(
                    state,
                    "share_appointment_predecessor_delay",
                    participant_id=participant_id,
                    appointment_id=own.get("appointment_id"),
                    retry_after_ms=own["delayed_retry_after_ms"],
                    predecessor_deadline_ms=predecessor_deadline_ms,
                )
                self._write(state)
                return self._share_appointment_status_locked(state, participant_id)
            uses_entry_policy = (
                str(own.get("arrival_entry_policy") or "")
                == SHARE_ARRIVAL_ENTRY_POLICY
            )
            if uses_entry_policy and not first_arrival:
                entry_open_ms = int(own.get("entry_open_ms") or 0) or now_ms
                own["entry_open_ms"] = entry_open_ms
                if now_ms > entry_open_ms + SHARE_ARRIVAL_ENTRY_MS:
                    own["status"] = "released"
                    own["released_ms"] = now_ms
                    own["release_reason"] = "arrival_entry_expired"
                    self.global_coordination.release_reservation(reservation_key)
                    self._write(state)
                    raise ShareAppointmentError(
                        "share_appointment_released",
                        "实验位置空闲后超过 15 分钟未输入凭证，原预约凭证已失效，请重新预约。",
                        self._share_appointment_status_locked(state, participant_id),
                    )
                if submitted_voucher != expected_voucher:
                    raise ShareAppointmentError(
                        "share_appointment_voucher_invalid",
                        "请重新输入预约成功时获得的四位凭证；空闲后超过 15 分钟未进入则凭证失效。",
                        self._share_appointment_status_locked(state, participant_id),
                    )
            if not self.global_coordination.arrive_and_claim(
                self._global_owner_key(participant_id),
                reservation_key,
                reference_start_ms=scheduled,
                expires_ms=now_ms + SHARE_CLAIM_LEASE_MS,
                now_ms=now_ms,
                grace_ms=grace_ms,
            ):
                own["delayed_retry_after_ms"] = predecessor_deadline_ms
                self._write(state)
                return self._share_appointment_status_locked(
                    state, participant_id
                )
            own["status"] = "claimed"
            own["claimed_ms"] = now_ms
            own["claim_expires_ms"] = now_ms + SHARE_CLAIM_LEASE_MS
            state["share_fit"]["current_holder"] = {
                "participant_id": participant_id,
                "mode": "appointment_arrival",
                "appointment_id": own.get("appointment_id"),
                "reference_start_ms": scheduled,
                "claimed_ms": now_ms,
                "expires_ms": now_ms + SHARE_CLAIM_LEASE_MS,
            }
            self._write(state)
            return self._share_appointment_status_locked(state, participant_id)

    def require_share_activation(self, participant_id: str) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            self._cleanup_share_schedule_locked(state)
            if not self._share_member(state, participant_id):
                return self._share_appointment_status_locked(state, participant_id)
            holder = state["share_fit"].get("current_holder")
            if not isinstance(holder, dict) or str(
                holder.get("participant_id") or ""
            ) != participant_id:
                raise ShareAppointmentError(
                    "share_serial_not_claimed",
                    "尚未取得当前实验位置，请按预约流程进入。",
                    self._share_appointment_status_locked(state, participant_id),
                )
            now_ms = int(time.time() * 1000)
            global_snapshot = self._global_snapshot_locked(
                participant_id, now_ms=now_ms
            )
            if global_snapshot.get("external_holders"):
                raise ShareAppointmentError(
                    "share_serial_occupied",
                    "上一位参与者尚未完成，请预约未来 12 小时内的实验时间。",
                    self._share_appointment_status_locked(
                        state, participant_id, now_ms=now_ms
                    ),
                )
            for other_id in state["share_fit"].get("enrolled_participant_ids") or []:
                if str(other_id) == participant_id:
                    continue
                session = self._share_study_session(str(other_id))
                if session and str(session.get("state") or "") == "active" and not session.get(
                    "ended_ms"
                ):
                    now_ms = int(time.time() * 1000)
                    other_anchor_ms = int(
                        session.get("activated_ms")
                        or session.get("started_ms")
                        or now_ms
                    )
                    # Repair a legacy/inconsistent holder atomically.  The
                    # blocked participant relinquishes the provisional slot,
                    # the real active predecessor becomes the holder, and the
                    # returned status is immediately bookable by the client.
                    for row in reversed(
                        state["share_fit"].get("appointments") or []
                    ):
                        if (
                            str(row.get("participant_id") or "")
                            == participant_id
                            and str(row.get("status") or "")
                            in {"claimed", "active"}
                        ):
                            row["status"] = "released"
                            row["released_ms"] = now_ms
                            row["release_reason"] = "active_predecessor_detected"
                            break
                    state["share_fit"]["current_holder"] = {
                        "participant_id": str(other_id),
                        "mode": "recovered_active_conflict",
                        "reference_start_ms": other_anchor_ms,
                        "claimed_ms": other_anchor_ms,
                        "expires_ms": (
                            other_anchor_ms + SHARE_SESSION_HARD_TIMEOUT_MS
                        ),
                    }
                    self._append_history(
                        state,
                        "share_active_predecessor_recovered",
                        participant_id=str(other_id),
                        blocked_participant_id=participant_id,
                        session_id=str(session["id"]),
                    )
                    self._write(state)
                    raise ShareAppointmentError(
                        "share_serial_occupied",
                        "上一位参与者尚未完成，请预约未来 12 小时内的实验时间。",
                        self._share_appointment_status_locked(state, participant_id),
                    )
            now_ms = int(time.time() * 1000)
            if not self.global_coordination.activate_holder(
                self._global_owner_key(participant_id),
                expires_ms=now_ms + SHARE_SESSION_HARD_TIMEOUT_MS,
                now_ms=now_ms,
            ):
                raise ShareAppointmentError(
                    "share_serial_occupied",
                    "上一位参与者尚未完成，请预约未来 12 小时内的实验时间。",
                    self._share_appointment_status_locked(
                        state, participant_id, now_ms=now_ms
                    ),
                )
            holder["mode"] = "active"
            holder["activated_ms"] = now_ms
            holder["expires_ms"] = now_ms + SHARE_CLAIM_LEASE_MS
            for row in reversed(state["share_fit"].get("appointments") or []):
                if (
                    str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "") == "claimed"
                ):
                    row["status"] = "active"
                    row["activated_ms"] = now_ms
                    row["claim_expires_ms"] = now_ms + SHARE_CLAIM_LEASE_MS
                    break
            self._write(state)
            return self._share_appointment_status_locked(state, participant_id)

    def on_session_aborted(self, participant_id: str) -> None:
        with self.lock:
            state = self._load()
            now_ms = int(time.time() * 1000)
            for row in reversed(state["share_fit"].get("appointments") or []):
                if (
                    str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "") in {"booked", "claimed", "active"}
                ):
                    row["status"] = "released"
                    row["released_ms"] = now_ms
                    row["release_reason"] = "session_aborted"
                    break
            holder = state["share_fit"].get("current_holder")
            if isinstance(holder, dict) and str(
                holder.get("participant_id") or ""
            ) == participant_id:
                state["share_fit"]["current_holder"] = None
            self._append_history(
                state, "share_serial_released", participant_id=participant_id
            )
            self._write(state)
            self.global_coordination.release_owner(
                self._global_owner_key(participant_id)
            )

    def on_participant_exit(self, participant_id: str) -> dict[str, Any]:
        """Immediately release a voluntarily abandoned participant.

        Browser logout is an explicit server transition, not an inactivity
        timeout.  An unfinished Share participant is retired so its stale
        starting prior cannot re-enter the serial model chain later.  A donor
        whose model is already committed (including the final-questionnaire
        stage) keeps its recovery path and is not retired.
        """

        with self.lock:
            state = self._load()
            share = state["share_fit"]
            now_ms = int(time.time() * 1000)
            released_appointments = 0
            for row in share.get("appointments") or []:
                if (
                    str(row.get("participant_id") or "") == participant_id
                    and str(row.get("status") or "")
                    in {"booked", "claimed", "active"}
                ):
                    row["status"] = "released"
                    row["released_ms"] = now_ms
                    row["release_reason"] = "participant_logout"
                    released_appointments += 1
            holder = share.get("current_holder")
            holder_released = bool(
                isinstance(holder, dict)
                and str(holder.get("participant_id") or "") == participant_id
            )
            if holder_released:
                share["current_holder"] = None

            # Apply the explicit participant transition before generic stale
            # schedule cleanup.  The session-abort transaction may already
            # have completed, but the participant-facing action must retain
            # its authoritative participant_logout reason and release receipt.
            self._reconcile_end_to_end_locked(state)
            self._cleanup_share_schedule_locked(state)

            chain_committed = participant_id in set(
                share.get("completed_donor_ids") or []
            )
            end_to_end_complete = self.db.participant_end_to_end_complete(
                participant_id
            )
            retired = bool(
                self._share_member(state, participant_id)
                and not chain_committed
                and not end_to_end_complete
            )
            if retired:
                exited = list(
                    share.get("voluntarily_exited_participant_ids") or []
                )
                if participant_id not in exited:
                    exited.append(participant_id)
                share["voluntarily_exited_participant_ids"] = exited[-300:]

            self._append_history(
                state,
                "participant_logout_release",
                participant_id=participant_id,
                holder_released=holder_released,
                released_appointments=released_appointments,
                retired=retired,
                chain_committed=chain_committed,
            )
            self._write(state)
            self.global_coordination.release_owner(
                self._global_owner_key(participant_id)
            )
            return {
                "released": bool(holder_released or released_appointments),
                "holder_released": holder_released,
                "released_appointments": released_appointments,
                "participant_retired": retired,
                "appointment": self._share_appointment_status_locked(
                    state, participant_id, now_ms=now_ms
                ),
            }

    def _complete_share_slot_locked(
        self, state: dict[str, Any], participant_id: str
    ) -> None:
        now_ms = int(time.time() * 1000)
        for row in reversed(state["share_fit"].get("appointments") or []):
            if (
                str(row.get("participant_id") or "") == participant_id
                and str(row.get("status") or "") in {"booked", "claimed", "active"}
            ):
                row["status"] = "completed"
                row["completed_ms"] = now_ms
                break
        holder = state["share_fit"].get("current_holder")
        if isinstance(holder, dict) and str(
            holder.get("participant_id") or ""
        ) == participant_id:
            state["share_fit"]["current_holder"] = None
        self._append_history(
            state, "share_serial_completed", participant_id=participant_id
        )
        self.global_coordination.release_owner(
            self._global_owner_key(participant_id)
        )

    def _migrate_legacy_share_chain_locked(
        self, state: dict[str, Any]
    ) -> bool:
        """Convert legacy E2E-derived donors into a verified committed prefix.

        Older builds appended a participant before proving that its personal
        checkpoint and cumulative prior existed.  Migration keeps only the
        longest prefix for which both artifacts validate.  The first failure
        and all later E2E records remain protocol-complete-but-pending; no
        checkpoint is synthesized here.
        """

        share = state["share_fit"]
        if int(share.get("chain_commit_version") or 0) >= SHARE_CHAIN_COMMIT_VERSION:
            return False
        legacy = list(dict.fromkeys(share.get("completed_donor_ids") or []))
        records = self.db.end_to_end_completion_records(legacy)
        verified: list[str] = []
        finalizations: dict[str, Any] = {}
        failure: str | None = None
        for participant_id in legacy:
            if (
                participant_id not in records
                or not self._participant_in_current_public_run(participant_id)
            ):
                failure = f"legacy donor {participant_id} is not E2E complete"
                break
            participant = self.db.participant(participant_id)
            if participant is None:
                failure = f"legacy donor {participant_id} is missing"
                break
            try:
                receipt = self._personal_checkpoint_receipt(participant)
                candidate = verified + [participant_id]
                prior = self._ensure_prior_for_donors(candidate)
            except OrchestrationError as exc:
                failure = str(exc)
                break
            finalizations[participant_id] = {
                **receipt,
                "participant_id": participant_id,
                "base_prior_id": participant.get("prior_id"),
                "chain_prior_id": str(prior["id"]),
                "migrated_legacy": True,
            }
            verified.append(participant_id)

        share["completed_donor_ids"] = verified
        share["model_finalizations"] = finalizations
        share["chain_commit_version"] = SHARE_CHAIN_COMMIT_VERSION
        share["last_chain_error"] = failure
        share["candidate_n"] = None
        share["parameter_candidate_n"] = None
        share["behavior_candidate_n"] = None
        share["delta_history"] = []
        share["behavior_history"] = []
        if verified:
            self._refresh_share_fit_locked(state, reconcile=False)
        self._append_history(
            state,
            "share_chain_legacy_migrated",
            legacy_count=len(legacy),
            verified_count=len(verified),
            pending_count=max(0, len(legacy) - len(verified)),
            failure=failure,
        )
        return True

    def _reconcile_end_to_end_locked(
        self, state: dict[str, Any]
    ) -> tuple[bool, bool]:
        """Reconcile timed-experiment receipts and already-committed donors."""
        changed = self._migrate_legacy_share_chain_locked(state)
        share_changed = False
        share = state["share_fit"]
        stored_enrolled = list(
            dict.fromkeys(share.get("enrolled_participant_ids") or [])
        )
        enrolled = [
            pid
            for pid in stored_enrolled
            if self._participant_in_current_public_run(pid)
        ]
        if enrolled != stored_enrolled:
            share["enrolled_participant_ids"] = enrolled
            changed = True
            self._append_history(
                state,
                "reconcile_share_current_run",
                before=len(stored_enrolled),
                after=len(enrolled),
                run_slug=self.settings.public_enrollment_run_slug,
            )
        stored_completed = list(dict.fromkeys(share.get("completed_donor_ids") or []))
        share_candidates = list(dict.fromkeys(stored_completed + enrolled))
        timed_records = self.db.end_to_end_completion_records(share_candidates)
        # Already-committed donors keep their prefix even if an older receipt
        # used a questionnaire-only predicate.  New pending members must have
        # finished the timed experiment; the final questionnaire does not
        # grant a chain slot.
        finalizations = dict(share.get("model_finalizations") or {})
        share_records = dict(timed_records)
        # A committed Share chain is an ordered prefix.  If a committed record
        # ever loses its timed-interaction receipt and has no stored
        # finalization, truncate from that point; never skip over it.
        reconciled_share: list[str] = []
        excluded_donors = set(share.get("excluded_donor_ids") or [])
        for participant_id in stored_completed:
            if participant_id in excluded_donors:
                break
            if not self._participant_in_current_public_run(participant_id):
                break
            if participant_id not in share_records and participant_id not in finalizations:
                break
            reconciled_share.append(participant_id)
        max_n = int(share.get("max_donors_for_auto") or MAX_DONORS_FOR_AUTO)
        reconciled_share = reconciled_share[:max_n]
        if reconciled_share != stored_completed:
            share["completed_donor_ids"] = reconciled_share
            share["model_finalizations"] = {
                participant_id: receipt
                for participant_id, receipt in dict(
                    share.get("model_finalizations") or {}
                ).items()
                if participant_id in set(reconciled_share)
            }
            if state.get("phase") == "share_fit" and not share.get("locked_prior_id"):
                share["candidate_n"] = None
                share["parameter_candidate_n"] = None
                share["behavior_candidate_n"] = None
                share["delta_history"] = []
                share["behavior_history"] = []
            share_changed = True
            changed = True
            self._append_history(
                state,
                "reconcile_share_model_ready",
                before=len(stored_completed),
                after=len(reconciled_share),
            )

        committed = set(reconciled_share)
        pending = sorted(
            (
                participant_id
                for participant_id in enrolled
                if participant_id in share_records
                and participant_id not in committed
                and participant_id not in excluded_donors
            ),
            key=lambda participant_id: (
                int(
                    share_records[participant_id].get("entered_ms")
                    or share_records[participant_id].get("completed_ms")
                    or 0
                ),
                participant_id,
            ),
        )
        if pending != list(share.get("protocol_complete_pending_ids") or []):
            share["protocol_complete_pending_ids"] = pending
            changed = True
            share_changed = True

        contrast = state["contrast"]
        assignments = [
            row
            for row in (contrast.get("assignments") or [])
            if isinstance(row, dict)
            and row.get("participant_id")
            and row.get("arm") in CONTRAST_ARMS
        ]
        assigned_ids = list(
            dict.fromkeys(
                str(row["participant_id"])
                for row in assignments
                if self._participant_in_current_public_run(
                    str(row["participant_id"])
                )
            )
        )
        contrast_records = self.db.end_to_end_completion_records(assigned_ids)
        completed_ids = sorted(
            contrast_records,
            key=lambda pid: (
                int(contrast_records[pid].get("completed_ms") or 0),
                pid,
            ),
        )
        completed_counts = {arm: 0 for arm in CONTRAST_ARMS}
        arm_by_participant = {
            str(row["participant_id"]): str(row["arm"]) for row in assignments
        }
        for participant_id in completed_ids:
            completed_counts[arm_by_participant[participant_id]] += 1
        if completed_ids != list(contrast.get("completed_participant_ids") or []):
            contrast["completed_participant_ids"] = completed_ids
            changed = True
        prior_completed_counts = {
            arm: int((contrast.get("completed_counts") or {}).get(arm, 0))
            for arm in CONTRAST_ARMS
        }
        if completed_counts != prior_completed_counts:
            contrast["completed_counts"] = completed_counts
            changed = True
        return changed, share_changed

    def status(self) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            schedule_changed = self._cleanup_share_schedule_locked(state)
            if state.pop("_knobs_need_persist", False) or reconciled or schedule_changed:
                self._write(state)
            share = state["share_fit"]
            contrast = state["contrast"]
            active_appointments = [
                row
                for row in (share.get("appointments") or [])
                if str(row.get("status") or "") in {"booked", "claimed", "active"}
            ]
            appointment_records = sorted(
                [
                    {
                        "appointment_id": str(row.get("appointment_id") or ""),
                        "participant_id": str(row.get("participant_id") or ""),
                        "scheduled_start_ms": int(
                            row.get("scheduled_start_ms") or 0
                        ),
                        "voucher": str(row.get("voucher") or ""),
                        "status": str(row.get("status") or ""),
                        "created_ms": int(row.get("created_ms") or 0),
                        "arrived_ms": int(row.get("arrived_ms") or 0),
                        "credential_verified_ms": int(
                            row.get("credential_verified_ms") or 0
                        ),
                        "delayed_retry_after_ms": int(
                            row.get("delayed_retry_after_ms") or 0
                        ),
                        "released_ms": int(row.get("released_ms") or 0),
                        "release_reason": str(row.get("release_reason") or ""),
                    }
                    for row in (share.get("appointments") or [])
                    if isinstance(row, dict)
                ],
                key=lambda row: (row["scheduled_start_ms"], row["created_ms"]),
                reverse=True,
            )[:100]
            return {
                "instance_id": self.settings.instance_id,
                "auto_enabled": bool(state["auto_enabled"]),
                "status": state["status"],
                "phase": state["phase"],
                "share_fit": {
                    "enrolled": len(share.get("enrolled_participant_ids") or []),
                    "completed": len(share.get("completed_donor_ids") or []),
                    "completed_donor_ids": list(share.get("completed_donor_ids") or []),
                    "protocol_complete_pending_ids": list(
                        share.get("protocol_complete_pending_ids") or []
                    ),
                    "auto_lock_pending_questionnaire_ids": list(
                        share.get("auto_lock_pending_questionnaire_ids") or []
                    ),
                    "end_to_end_completed_donors": len(
                        self.db.end_to_end_completion_records(
                            list(share.get("completed_donor_ids") or [])
                        )
                    ),
                    "model_finalization_count": len(
                        share.get("model_finalizations") or {}
                    ),
                    "chain_commit_version": int(
                        share.get("chain_commit_version") or 0
                    ),
                    "last_chain_error": share.get("last_chain_error"),
                    "candidate_n": share.get("candidate_n"),
                    "locked_n": share.get("locked_n"),
                    "locked_prior_id": share.get("locked_prior_id"),
                    "lock_reason": share.get("lock_reason"),
                    "parameter_candidate_n": share.get("parameter_candidate_n"),
                    "behavior_candidate_n": share.get("behavior_candidate_n"),
                    "delta_history": list(share.get("delta_history") or []),
                    "behavior_history": list(share.get("behavior_history") or []),
                    "min_donors_for_auto": int(
                        share.get("min_donors_for_auto") or MIN_DONORS_FOR_AUTO
                    ),
                    "stable_required": int(
                        share.get("stable_required") or STABLE_REQUIRED
                    ),
                    "rel_delta_threshold": float(
                        share.get("rel_delta_threshold") or REL_DELTA_THRESHOLD
                    ),
                    "max_donors_for_auto": int(
                        share.get("max_donors_for_auto") or MAX_DONORS_FOR_AUTO
                    ),
                    "behavior_window": int(
                        share.get("behavior_window") or BEHAVIOR_WINDOW
                    ),
                    "opening_improve_epsilon": float(
                        share.get("opening_improve_epsilon")
                        or OPENING_IMPROVE_EPSILON
                    ),
                    "opening_equiv_epsilon": float(
                        share.get("opening_equiv_epsilon") or OPENING_EQUIV_EPSILON
                    ),
                    "plateau_persist_after_nonoverlap": int(
                        share.get("plateau_persist_after_nonoverlap")
                        or PLATEAU_PERSIST_AFTER_NONOVERLAP
                    ),
                    "confirm_horizon_donors": int(
                        share.get("confirm_horizon_donors") or CONFIRM_HORIZON_DONORS
                    ),
                    "rebound_confirm_windows": int(
                        share.get("rebound_confirm_windows")
                        or REBOUND_CONFIRM_WINDOWS
                    ),
                    "tap_delta_tolerance": float(
                        share.get("tap_delta_tolerance") or TAP_DELTA_TOLERANCE
                    ),
                    "serial_schedule": {
                        "holder_present": bool(share.get("current_holder")),
                        "active_appointments": len(active_appointments),
                        "booked": sum(
                            str(row.get("status") or "") == "booked"
                            for row in active_appointments
                        ),
                        "horizon_hours": SHARE_APPOINTMENT_HORIZON_MS // 3_600_000,
                        "slot_minutes": SHARE_APPOINTMENT_SLOT_MS // 60_000,
                        "spacing_minutes": SHARE_APPOINTMENT_SPACING_MS // 60_000,
                        "late_release_minutes": SHARE_APPOINTMENT_GRACE_MS // 60_000,
                        "unfinished_release_minutes": (
                            SHARE_SESSION_HARD_TIMEOUT_MS // 60_000
                        ),
                        "setup_release_minutes": (
                            SHARE_SETUP_TIMEOUT_MS // 60_000
                        ),
                        "appointments": appointment_records,
                    },
                    "curve_download_available": bool(
                        share.get("completed_donor_ids")
                        or share.get("locked_prior_id")
                        or state.get("phase") in {"contrast", "done"}
                    ),
                    "operator_blocked_slots": _operator_blocked_slot_records(share),
                    "fit_remaining_estimate": _share_fit_remaining_estimate(share),
                },
                "contrast": {
                    "next_arm": choose_contrast_arm(
                        contrast.get("completed_counts") or {},
                        contrast.get("counts") or {},
                        contrast.get("caps") or {},
                    ),
                    "next_index": int(contrast.get("next_index") or 0),
                    "caps": dict(contrast.get("caps") or {}),
                    "counts": dict(contrast.get("counts") or {}),
                    "completed_counts": dict(
                        contrast.get("completed_counts") or {}
                    ),
                    "completed_participant_count": len(
                        contrast.get("completed_participant_ids") or []
                    ),
                    "labels": ARM_LABELS,
                    "assignment_count": len(contrast.get("assignments") or []),
                },
                "history": list(state.get("history") or [])[-40:],
                "updated_ms": state.get("updated_ms"),
                "enrollment_open": bool(state["auto_enabled"])
                and state["status"] == "running"
                and state["phase"] in {"share_fit", "contrast"},
            }

    def replace_operator_blocked_slots(
        self,
        date_cst: str,
        scheduled_start_ms: list[int],
        *,
        actor: str = "admin",
    ) -> dict[str, Any]:
        """Replace operator-blocked half-hour starts for one CST calendar day.

        Existing bookings, holders, donors, models, and the global mutex are
        left untouched.  Only later booking attempts on this instance see the
        blocked starts.
        """
        with self.lock:
            try:
                datetime.strptime(str(date_cst), "%Y-%m-%d")
            except ValueError as exc:
                raise OrchestrationError("日期格式无效，请使用 YYYY-MM-DD") from exc
            unique_starts: list[int] = []
            seen: set[int] = set()
            for raw in scheduled_start_ms:
                start_ms = int(raw)
                if start_ms in seen:
                    continue
                if start_ms <= 0 or start_ms % SHARE_APPOINTMENT_SLOT_MS != 0:
                    raise OrchestrationError("不可预约时段必须对齐半小时时间块")
                if _cst_date_from_ms(start_ms) != date_cst:
                    raise OrchestrationError(
                        "所选时间块必须属于该日期（北京时间）"
                    )
                seen.add(start_ms)
                unique_starts.append(start_ms)
            if len(unique_starts) > 48:
                raise OrchestrationError("同一天最多设置 48 个半小时不可预约块")
            state = self._load()
            share = state["share_fit"]
            kept = [
                row
                for row in (share.get("operator_blocked_slots") or [])
                if isinstance(row, dict)
                and int(row.get("scheduled_start_ms") or 0) > 0
                and _cst_date_from_ms(int(row["scheduled_start_ms"])) != date_cst
            ]
            previous = {
                int(row.get("scheduled_start_ms") or 0)
                for row in (share.get("operator_blocked_slots") or [])
                if isinstance(row, dict)
                and _cst_date_from_ms(int(row.get("scheduled_start_ms") or 0))
                == date_cst
            }
            now_ms = int(time.time() * 1000)
            created_by_start = {
                int(row.get("scheduled_start_ms") or 0): int(
                    row.get("created_ms") or now_ms
                )
                for row in (share.get("operator_blocked_slots") or [])
                if isinstance(row, dict)
                and int(row.get("scheduled_start_ms") or 0) in seen
            }
            if previous == seen:
                return self.status()
            share["operator_blocked_slots"] = kept + [
                {
                    "scheduled_start_ms": start_ms,
                    "created_ms": created_by_start.get(start_ms, now_ms),
                }
                for start_ms in sorted(unique_starts)
            ]
            self._append_history(
                state,
                "set_operator_blocked_slots",
                actor=actor,
                date_cst=date_cst,
                scheduled_start_ms=sorted(unique_starts),
            )
            self._write(state)
            return self.status()

    def admin_hidden_participant_ids(self) -> set[str]:
        """Participant IDs archived out of the live admin/data surfaces."""
        with self.lock:
            state = self._load()
            return {
                str(pid)
                for pid in (state["share_fit"].get("hidden_from_admin_ids") or [])
                if pid
            }

    def peek_live_share_schedule(self) -> dict[str, Any]:
        """Read holder/appointments from disk. Never migrates or writes state."""
        state = self._read_state_readonly()
        share = state.get("share_fit") if isinstance(state.get("share_fit"), dict) else {}
        hidden = [
            str(pid)
            for pid in dict.fromkeys(share.get("hidden_from_admin_ids") or [])
            if pid
        ]
        holder_raw = share.get("current_holder")
        holder = None
        if isinstance(holder_raw, dict) and holder_raw.get("participant_id"):
            holder = {
                "participant_id": str(holder_raw.get("participant_id") or ""),
                "mode": holder_raw.get("mode"),
                "claimed_ms": holder_raw.get("claimed_ms"),
                "activated_ms": holder_raw.get("activated_ms"),
                "scheduled_start_ms": holder_raw.get("scheduled_start_ms"),
            }
        appointments: list[dict[str, Any]] = []
        for row in share.get("appointments") or []:
            if not isinstance(row, dict):
                continue
            participant_id = str(row.get("participant_id") or "")
            status = str(row.get("status") or "")
            if not participant_id or status not in {"booked", "claimed", "active"}:
                continue
            appointments.append(
                {
                    "participant_id": participant_id,
                    "status": status,
                    "scheduled_start_ms": row.get("scheduled_start_ms"),
                    "claimed_ms": row.get("claimed_ms"),
                    "activated_ms": row.get("activated_ms"),
                }
            )
        contrast = (
            state.get("contrast") if isinstance(state.get("contrast"), dict) else {}
        )
        return {
            "readonly": True,
            "hidden_from_admin_ids": hidden,
            "current_holder": holder,
            "appointments": appointments,
            "completed_donor_ids": [
                str(pid)
                for pid in dict.fromkeys(share.get("completed_donor_ids") or [])
                if pid
            ],
            "contrast_completed_ids": [
                str(pid)
                for pid in dict.fromkeys(
                    contrast.get("completed_participant_ids") or []
                )
                if pid
            ],
            "voluntarily_exited_participant_ids": [
                str(pid)
                for pid in dict.fromkeys(
                    share.get("voluntarily_exited_participant_ids") or []
                )
                if pid
            ],
            "setup_timed_out_participant_ids": [
                str(pid)
                for pid in dict.fromkeys(
                    share.get("setup_timed_out_participant_ids") or []
                )
                if pid
            ],
        }

    def peek_share_fit_trend_board(
        self,
        *,
        final_items: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read-only share-fit per-donor trend board. Never writes orch or models."""
        state = self._read_state_readonly()
        share = state.get("share_fit") if isinstance(state.get("share_fit"), dict) else {}
        donor_ids = [
            str(pid)
            for pid in dict.fromkeys(share.get("completed_donor_ids") or [])
            if pid
        ]
        code_by_id: dict[str, str] = {}
        events_by_id: dict[str, list[tuple[str, dict[str, Any]]]] = {
            pid: [] for pid in donor_ids
        }
        answers_by_id: dict[str, dict[str, Any]] = {}
        if donor_ids:
            placeholders = ",".join("?" * len(donor_ids))
            with self.db.connect() as conn:
                for row in conn.execute(
                    f"SELECT id, code_hint FROM participants WHERE id IN ({placeholders})",
                    donor_ids,
                ).fetchall():
                    code_by_id[str(row["id"])] = str(row["code_hint"] or "")
                for row in conn.execute(
                    f"""SELECT participant_id, event_type, payload
                        FROM events
                        WHERE participant_id IN ({placeholders})
                          AND event_type IN (
                              'model.prediction',
                              'tap.manual_activate',
                              'tap.interrupt',
                              'tap.state_transition_mistouch'
                          )
                        ORDER BY id""",
                    donor_ids,
                ).fetchall():
                    participant_id = str(row["participant_id"])
                    if participant_id not in events_by_id:
                        continue
                    try:
                        payload = json.loads(str(row["payload"] or "{}"))
                    except (TypeError, json.JSONDecodeError):
                        payload = {}
                    if not isinstance(payload, dict):
                        payload = {}
                    events_by_id[participant_id].append(
                        (str(row["event_type"]), payload)
                    )
                for row in conn.execute(
                    f"""SELECT participant_id, answers
                        FROM responses
                        WHERE instrument='final'
                          AND participant_id IN ({placeholders})""",
                    donor_ids,
                ).fetchall():
                    try:
                        answers = json.loads(str(row["answers"] or "{}"))
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(answers, dict):
                        answers_by_id[str(row["participant_id"])] = answers

        donors: list[dict[str, Any]] = []
        opening_stats: list[dict[str, Any]] = []
        for index, participant_id in enumerate(donor_ids, start=1):
            blocks: dict[int, dict[str, Any]] = {}
            for event_type, payload in events_by_id.get(participant_id, []):
                _accumulate_share_fit_block_event(blocks, event_type, payload)
            block_metrics = _finalize_block_metrics(blocks)
            session_taps = sum(int(row["taps"]) for row in block_metrics.values())
            session_opportunities = sum(
                int(row["opportunities"]) for row in block_metrics.values()
            )
            session_corrections = sum(
                int(row["corrections"]) for row in block_metrics.values()
            )
            opening = block_metrics.get(OPENING_BLOCK_INDEX) or _empty_block_metric()
            opening_q = opening.get("q")
            opening_stats.append({"opening_correction_burden": opening_q})
            dimension_scores: dict[str, float | None] = {}
            answers = answers_by_id.get(participant_id) or {}
            items = final_items or {}
            per_dimension: dict[str, list[float]] = {}
            for item_id, item in items.items():
                if not isinstance(item, dict):
                    continue
                raw = answers.get(item_id)
                if not isinstance(raw, (int, float)):
                    continue
                score = float(raw)
                if item.get("reverse"):
                    score = 8.0 - score
                dimension = str(item.get("dimension") or item_id)
                per_dimension.setdefault(dimension, []).append(score)
            for key, _label in SHARE_FIT_QUESTIONNAIRE_DIMENSIONS:
                values = per_dimension.get(key) or []
                dimension_scores[key] = (
                    round(sum(values) / len(values), 4) if values else None
                )
            donors.append(
                {
                    "n": index,
                    "participant_id": participant_id,
                    "code_hint": code_by_id.get(participant_id) or "",
                    "blocks": {
                        str(block): {
                            "taps": metric["taps"],
                            "taps_per_minute": metric["taps_per_minute"],
                            "q": metric["q"],
                            "opportunities": metric["opportunities"],
                            "corrections": metric["corrections"],
                        }
                        for block, metric in block_metrics.items()
                    },
                    "session_taps_per_minute": (
                        round(session_taps / EFFECTIVE_INTERACTION_MINUTES, 6)
                        if session_taps or session_opportunities
                        else None
                    ),
                    "session_q": (
                        round(session_corrections / session_opportunities, 6)
                        if session_opportunities > 0
                        else None
                    ),
                    "opening_q": opening_q,
                    "questionnaire": dimension_scores,
                }
            )

        donor_ns = [int(row["n"]) for row in donors]
        codes = [str(row["code_hint"] or row["n"]) for row in donors]
        block_trends: list[dict[str, Any]] = []
        for block in range(1, 6):
            taps = [
                (row["blocks"].get(str(block)) or {}).get("taps_per_minute")
                for row in donors
            ]
            qs = [(row["blocks"].get(str(block)) or {}).get("q") for row in donors]
            tap_trend = _share_fit_series_trend(
                donor_ns, taps, codes, lower_is_better=True
            )
            q_trend = _share_fit_series_trend(
                donor_ns, qs, codes, lower_is_better=True
            )
            block_trends.append(
                {
                    "block": block,
                    "taps_per_minute": tap_trend,
                    "q": q_trend,
                }
            )
        session_trends = {
            "taps_per_minute": _share_fit_series_trend(
                donor_ns,
                [row["session_taps_per_minute"] for row in donors],
                codes,
                lower_is_better=True,
            ),
            "q": _share_fit_series_trend(
                donor_ns,
                [row["session_q"] for row in donors],
                codes,
                lower_is_better=True,
            ),
        }
        questionnaire_trends = []
        for key, label in SHARE_FIT_QUESTIONNAIRE_DIMENSIONS:
            values = [row["questionnaire"].get(key) for row in donors]
            trend = _share_fit_series_trend(
                donor_ns, values, codes, lower_is_better=False
            )
            questionnaire_trends.append(
                {
                    "dimension": key,
                    "label": label,
                    "participants": sum(value is not None for value in values),
                    **trend,
                }
            )

        remaining = _share_fit_remaining_estimate(share, opening_stats=opening_stats)
        return {
            "ok": True,
            "readonly": True,
            "writes_orchestration": False,
            "writes_models": False,
            "instance_id": self.settings.instance_id,
            "generated_ms": int(time.time() * 1000),
            "phase": state.get("phase"),
            "completed_n": len(donor_ids),
            "definitions": {
                "outlier_rule": "IQR 1.5 per series, then Spearman vs donor order n",
                "significance_alpha": SHARE_FIT_TREND_ALPHA,
                "block_minutes": SHARE_FIT_BLOCK_MINUTES,
                "opening_block": OPENING_BLOCK_INDEX,
                "q_definition": (
                    "eligible corrections / decision-eligible opportunities"
                ),
                "lower_is_better": ["taps_per_minute", "q"],
                "higher_is_better": [key for key, _label in SHARE_FIT_QUESTIONNAIRE_DIMENSIONS],
                "remaining_basis": remaining.get("basis"),
            },
            "fit_remaining_estimate": remaining,
            "donors": donors,
            "block_trends": block_trends,
            "session_trends": session_trends,
            "questionnaire_trends": questionnaire_trends,
        }

    def admin_assignment_snapshot(self) -> dict[str, Any]:
        """Return the persisted arm map for the authenticated research dashboard."""
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            if reconciled:
                self._write(state)
            share = state["share_fit"]
            contrast = state["contrast"]
            hidden = {
                str(pid)
                for pid in (share.get("hidden_from_admin_ids") or [])
                if pid
            }
            enrolled = [
                pid
                for pid in (share.get("enrolled_participant_ids") or [])
                if str(pid) not in hidden
            ]
            completed = [
                pid
                for pid in (share.get("completed_donor_ids") or [])
                if str(pid) not in hidden
            ]
            return {
                "phase": state.get("phase"),
                "share_fit_locked": bool(
                    share.get("locked_prior_id") or share.get("locked_n") is not None
                ),
                "share_fit_completed_n": len(completed),
                "share_fit_candidate_n": share.get("candidate_n"),
                "share_fit_participant_ids": list(dict.fromkeys([*enrolled, *completed])),
                "share_fit_completed_ids": completed,
                "contrast_assignments": [
                    {
                        "participant_id": str(row.get("participant_id") or ""),
                        "arm": str(row.get("arm") or ""),
                    }
                    for row in (contrast.get("assignments") or [])
                    if isinstance(row, dict) and row.get("participant_id")
                ],
            }

    def set_auto_enabled(self, enabled: bool, *, actor: str) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            state["auto_enabled"] = bool(enabled)
            self._append_history(
                state, "auto_enabled", enabled=bool(enabled), actor=actor
            )
            self._write(state)
            return self.status()

    def pause(self, *, actor: str) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            if state["status"] == "stopped" and state["phase"] == "done":
                raise OrchestrationError("实验已结束，无法仅暂停；请新建编排或重置")
            state["status"] = "paused"
            self._append_history(state, "pause", actor=actor)
            self._write(state)
            return self.status()

    def resume(self, *, actor: str) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            if state["phase"] == "done":
                raise OrchestrationError("对比阶段已结束，无法继续报名")
            state["status"] = "running"
            self._append_history(state, "resume", actor=actor)
            self._write(state)
            return self.status()

    def early_stop(self, *, actor: str) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            state["status"] = "stopped"
            state["phase"] = "done"
            self._append_history(state, "early_stop", actor=actor)
            self._write(state)
            return self.status()

    def reset(self, *, actor: str) -> dict[str, Any]:
        with self.lock:
            state = _default_state()
            self._append_history(state, "reset", actor=actor)
            self._write(state)
            return self.status()

    def assign_enrollment(self) -> dict[str, Any]:
        """Reserve and return cohort/group/prior for the next public enrollment."""
        with self.lock:
            state = self._load()
            reconciled, share_changed = self._reconcile_end_to_end_locked(state)
            if share_changed and state["phase"] == "share_fit":
                try:
                    self._refresh_share_fit_locked(state, reconcile=False)
                except OrchestrationError as exc:
                    self._append_history(state, "share_fit_error", error=str(exc))
            if reconciled:
                self._write(state)
            if not state["auto_enabled"]:
                return {
                    "mode": "legacy",
                    "cohort": self.settings.public_enrollment_cohort,
                    "group_name": self.settings.public_enrollment_group,
                    "prior_id": None,
                    "arm": None,
                    "phase": None,
                    "reservation_id": None,
                }
            if state["status"] != "running":
                raise OrchestrationError(
                    "实验报名已暂停或结束，请稍后再试或联系管理员"
                )
            phase = state["phase"]
            if phase == "share_fit":
                share = state["share_fit"]
                # The 50-person ceiling is a ceiling on safely committed model
                # donors. Abandoned registrations do not consume a donor slot.
                occupied = len(share.get("completed_donor_ids") or []) + len(
                    share.get("pending_reservations") or []
                )
                max_donors = int(
                    share.get("max_donors_for_auto") or MAX_DONORS_FOR_AUTO
                )
                if occupied >= max_donors:
                    raise OrchestrationError(
                        f"Share 拟合已达到 {max_donors} 名模型 donor 上限；"
                        "如仍有最终问卷待提交，将在其端到端完成后锁定权重并切换阶段"
                    )
                reservation_id = str(uuid.uuid4())
                pending = list(share.get("pending_reservations") or [])
                pending.append(reservation_id)
                share["pending_reservations"] = pending
                self._write(state)
                return {
                    "mode": "auto",
                    "phase": "share_fit",
                    "arm": "share_fit",
                    "cohort": self.settings.public_enrollment_cohort,
                    "group_name": "personalized",
                    "prior_id": None,
                    "reservation_id": reservation_id,
                }
            if phase == "contrast":
                return self._reserve_contrast_assignment(state)
            raise OrchestrationError("当前阶段不再接受新受试者")

    def _reserve_contrast_assignment(self, state: dict[str, Any]) -> dict[str, Any]:
        share = state["share_fit"]
        prior_id = share.get("locked_prior_id")
        if not prior_id:
            raise OrchestrationError("对比阶段尚未锁定 share 拟合先验")
        contrast = state["contrast"]
        counts = dict(contrast.get("counts") or {})
        completed_counts = dict(contrast.get("completed_counts") or {})
        caps = dict(contrast.get("caps") or {})
        chosen = choose_contrast_arm(completed_counts, counts, caps)
        if chosen is not None:
            counts[chosen] = int(counts.get(chosen, 0)) + 1
            contrast["counts"] = counts
            preview = choose_contrast_arm(completed_counts, counts, caps)
            contrast["next_index"] = (
                CONTRAST_ARMS.index(preview) if preview is not None else 0
            )
        if chosen is None:
            state["phase"] = "done"
            state["status"] = "stopped"
            self._append_history(state, "caps_reached")
            self._write(state)
            raise OrchestrationError("三组名额均已满，实验报名已结束")
        reservation_id = str(uuid.uuid4())
        pending = list(contrast.get("pending_reservations") or [])
        pending.append({"reservation_id": reservation_id, "arm": chosen})
        contrast["pending_reservations"] = pending
        state["contrast"] = contrast
        self._write(state)
        if chosen == ARM_COLD_LEARN:
            group_name = "personalized"
            prior = None
        elif chosen == ARM_SHARED_LEARN:
            group_name = "personalized"
            prior = str(prior_id)
        else:
            group_name = "frozen"
            prior = str(prior_id)
        return {
            "mode": "auto",
            "phase": "contrast",
            "arm": chosen,
            "cohort": self.settings.public_enrollment_cohort,
            "group_name": group_name,
            "prior_id": prior,
            "reservation_id": reservation_id,
        }

    def abandon_enrollment(self, assignment: dict[str, Any]) -> None:
        with self.lock:
            if assignment.get("mode") != "auto":
                return
            state = self._load()
            reservation_id = assignment.get("reservation_id")
            phase = assignment.get("phase")
            arm = assignment.get("arm")
            if phase == "share_fit":
                pending = [
                    item
                    for item in (state["share_fit"].get("pending_reservations") or [])
                    if item != reservation_id
                ]
                state["share_fit"]["pending_reservations"] = pending
            elif phase == "contrast" and arm in CONTRAST_ARMS:
                pending = []
                released = False
                for item in state["contrast"].get("pending_reservations") or []:
                    if (
                        not released
                        and isinstance(item, dict)
                        and item.get("reservation_id") == reservation_id
                    ):
                        released = True
                        counts = dict(state["contrast"].get("counts") or {})
                        counts[arm] = max(0, int(counts.get(arm, 0)) - 1)
                        state["contrast"]["counts"] = counts
                        continue
                    pending.append(item)
                state["contrast"]["pending_reservations"] = pending
            self._append_history(state, "abandon_enrollment", arm=arm)
            self._write(state)

    def commit_enrollment(
        self,
        *,
        participant_id: str,
        assignment: dict[str, Any],
    ) -> None:
        with self.lock:
            state = self._load()
            if assignment.get("mode") != "auto":
                return
            phase = assignment.get("phase")
            arm = assignment.get("arm")
            reservation_id = assignment.get("reservation_id")
            if phase == "share_fit":
                pending = [
                    item
                    for item in (state["share_fit"].get("pending_reservations") or [])
                    if item != reservation_id
                ]
                state["share_fit"]["pending_reservations"] = pending
                enrolled = list(state["share_fit"].get("enrolled_participant_ids") or [])
                if participant_id not in enrolled:
                    enrolled.append(participant_id)
                state["share_fit"]["enrolled_participant_ids"] = enrolled
                self._append_history(
                    state,
                    "enroll_share_fit",
                    participant_id=participant_id,
                )
            elif phase == "contrast" and arm in CONTRAST_ARMS:
                pending = [
                    item
                    for item in (state["contrast"].get("pending_reservations") or [])
                    if not (
                        isinstance(item, dict)
                        and item.get("reservation_id") == reservation_id
                    )
                ]
                state["contrast"]["pending_reservations"] = pending
                assignments = list(state["contrast"].get("assignments") or [])
                assignments.append(
                    {
                        "participant_id": participant_id,
                        "arm": arm,
                        "ms": int(time.time() * 1000),
                    }
                )
                state["contrast"]["assignments"] = assignments
                self._append_history(
                    state,
                    "enroll_contrast",
                    participant_id=participant_id,
                    arm=arm,
                )
            self._write(state)

    def commit_share_completion(
        self,
        participant_id: str,
        model_receipt: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically advance the serial chain after all three gates pass.

        Gate 1 is the server-authoritative timed-interaction receipt (strict
        mode: the 34-minute / five-block real-time experiment).  The final
        questionnaire does not grant or withhold this gate.  Gate 2 is a
        fully validated personal checkpoint, and gate 3 is the exact
        cumulative prior for the new donor prefix.
        """

        with self.lock:
            state = self._load()
            self._reconcile_end_to_end_locked(state)
            timed_complete = self.db.participant_end_to_end_complete(
                participant_id
            )
            if (
                not self._participant_in_current_public_run(participant_id)
                or not timed_complete
                or not self._share_member(state, participant_id)
            ):
                raise OrchestrationError(
                    "Share chain commit requires a current-run participant who "
                    "has finished the timed real-time experiment"
                )
            share = state["share_fit"]
            completed = list(share.get("completed_donor_ids") or [])
            if participant_id in completed:
                # Idempotent closeout retry: validate the already committed
                # exact prefix before releasing any stale holder.
                index = completed.index(participant_id) + 1
                prior = self._ensure_prior_for_donors(completed[:index])
                self._complete_share_slot_locked(state, participant_id)
                share["last_chain_error"] = None
                self._write(state)
                return {
                    "committed": True,
                    "idempotent": True,
                    "committed_n": index,
                    "chain_prior_id": str(prior["id"]),
                }

            enrolled_ids = list(
                dict.fromkeys((share.get("enrolled_participant_ids") or []))
            )
            records = self.db.end_to_end_completion_records(enrolled_ids)
            current_ms = int(
                records[participant_id].get("entered_ms")
                or records[participant_id].get("completed_ms")
                or 0
            )
            earlier_pending = sorted(
                (
                    candidate
                    for candidate in records
                    if candidate not in set(completed)
                    and candidate != participant_id
                    and int(
                        records[candidate].get("entered_ms")
                        or records[candidate].get("completed_ms")
                        or 0
                    )
                    <= current_ms
                ),
                key=lambda candidate: (
                    int(
                        records[candidate].get("entered_ms")
                        or records[candidate].get("completed_ms")
                        or 0
                    ),
                    candidate,
                ),
            )
            if earlier_pending:
                raise OrchestrationError(
                    "An earlier model-ready Share participant is still pending "
                    "chain commit"
                )

            participant = self.db.participant(participant_id)
            if participant is None:
                raise OrchestrationError("Share participant is missing")
            actual = self._personal_checkpoint_receipt(participant)
            checked_fields = (
                "pointer_sha256",
                "personal_head_sha256",
                "personal_head_fingerprint",
                "replay_sha256",
                "update_count",
                "optimizer_step_count",
                "release_manifest_sha256",
                "shared_head_sha256",
            )
            if any(
                str(model_receipt.get(field)) != str(actual.get(field))
                for field in checked_fields
            ):
                raise OrchestrationError(
                    "Personal checkpoint changed between finalization and chain commit"
                )
            if str(model_receipt.get("participant_id") or "") != participant_id:
                raise OrchestrationError("Model-finalization receipt participant mismatch")
            if model_receipt.get("base_prior_id") != participant.get("prior_id"):
                raise OrchestrationError("Model-finalization receipt prior mismatch")

            if completed:
                expected_start = self._ensure_prior_for_donors(completed)
                if str(participant.get("prior_id") or "") != str(expected_start["id"]):
                    raise OrchestrationError(
                        "Participant did not start from the immediately preceding "
                        "committed Share prior"
                    )
            elif participant.get("prior_id"):
                raise OrchestrationError(
                    "The first Share donor must be a cold-start participant"
                )

            candidate = copy.deepcopy(state)
            candidate_share = candidate["share_fit"]
            candidate_completed = completed + [participant_id]
            candidate_share["completed_donor_ids"] = candidate_completed
            candidate_share.setdefault("model_finalizations", {})[
                participant_id
            ] = {
                **actual,
                "participant_id": participant_id,
                "base_prior_id": participant.get("prior_id"),
                "eligible_feedbacks": int(
                    model_receipt.get("eligible_feedbacks") or 0
                ),
                "expected_network_updates": int(
                    model_receipt.get("expected_network_updates") or 0
                ),
                "zero_network_update": bool(
                    model_receipt.get("zero_network_update")
                ),
                "finalized_ms": int(
                    model_receipt.get("finalized_ms") or time.time() * 1000
                ),
            }
            try:
                self._refresh_share_fit_locked(candidate, reconcile=False)
                chain_prior = self._ensure_prior_for_donors(candidate_completed)
            except Exception as exc:
                share["last_chain_error"] = str(exc)
                self._append_history(
                    state,
                    "share_chain_commit_failed",
                    participant_id=participant_id,
                    committed_prefix_n=len(completed),
                    error=str(exc),
                )
                self._write(state)
                if isinstance(exc, OrchestrationError):
                    raise
                raise OrchestrationError(str(exc)) from exc

            candidate_share["model_finalizations"][participant_id][
                "chain_prior_id"
            ] = str(chain_prior["id"])
            candidate_share["last_chain_error"] = None
            candidate_share["protocol_complete_pending_ids"] = [
                value
                for value in (
                    candidate_share.get("protocol_complete_pending_ids") or []
                )
                if value != participant_id
            ]
            self._complete_share_slot_locked(candidate, participant_id)
            self._append_history(
                candidate,
                "share_chain_committed",
                participant_id=participant_id,
                committed_n=len(candidate_completed),
                base_prior_id=participant.get("prior_id"),
                chain_prior_id=str(chain_prior["id"]),
                zero_network_update=bool(
                    model_receipt.get("zero_network_update")
                ),
                final_questionnaire_submitted=bool(
                    (self.db.final_questionnaire_recovery(participant_id) or {}).get(
                        "final_submitted_ms"
                    )
                ),
            )
            self._write(candidate)
            return {
                "committed": True,
                "idempotent": False,
                "committed_n": len(candidate_completed),
                "chain_prior_id": str(chain_prior["id"]),
            }

    def on_session_completed(self, participant_id: str) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            if (
                not self._participant_in_current_public_run(participant_id)
                or not self.db.participant_end_to_end_complete(participant_id)
            ):
                self._append_history(
                    state,
                    "completion_rejected_incomplete",
                    participant_id=participant_id,
                )
                self._write(state)
                return self.status()
            if self._share_member(state, participant_id):
                # The model chain was already advanced at final-questionnaire
                # entry.  Submission now only unlocks an automatic phase
                # transition that is deliberately gated on E2E qualification.
                self._append_history(
                    state, "share_final_questionnaire_completed",
                    participant_id=participant_id,
                )
                if state["phase"] == "share_fit" and state["auto_enabled"]:
                    try:
                        self._refresh_share_fit_locked(state, reconcile=False)
                    except OrchestrationError as exc:
                        self._append_history(
                            state, "share_fit_error", error=str(exc)
                        )
                self._write(state)
                return self.status()
            if not state["auto_enabled"]:
                self._write(state)
                return self.status()

            if state["phase"] == "share_fit":
                enrolled = set(
                    state["share_fit"].get("enrolled_participant_ids") or []
                )
                if participant_id not in enrolled:
                    # Still allow a completed, cold personalised donor already
                    # belonging to this public run.
                    participant = self.db.participant(participant_id)
                    if participant is None:
                        return self.status()
                    if str(participant.get("group_name") or "").lower() not in {
                        "personalized",
                        "share_donor",
                    }:
                        return self.status()
                    if participant.get("prior_id"):
                        return self.status()
                completed = list(
                    state["share_fit"].get("completed_donor_ids") or []
                )
                if participant_id not in completed:
                    completed.append(participant_id)
                state["share_fit"]["completed_donor_ids"] = completed
                self._append_history(
                    state, "share_donor_completed", participant_id=participant_id
                )
                try:
                    self._refresh_share_fit_locked(state)
                except OrchestrationError as exc:
                    self._append_history(state, "share_fit_error", error=str(exc))
            elif state["phase"] in {"contrast", "done"}:
                assignment = next(
                    (
                        row
                        for row in reversed(state["contrast"].get("assignments") or [])
                        if isinstance(row, dict)
                        and str(row.get("participant_id") or "") == participant_id
                        and row.get("arm") in CONTRAST_ARMS
                    ),
                    None,
                )
                if assignment is not None:
                    self._reconcile_end_to_end_locked(state)
                    completed_counts = state["contrast"].get("completed_counts") or {}
                    caps = state["contrast"].get("caps") or {}
                    self._append_history(
                        state,
                        "contrast_participant_completed",
                        participant_id=participant_id,
                        arm=assignment["arm"],
                    )
                    if all(
                        int(completed_counts.get(arm, 0))
                        >= int(caps.get(arm, 32))
                        for arm in CONTRAST_ARMS
                    ):
                        state["phase"] = "done"
                        state["status"] = "stopped"
                        self._append_history(state, "completed_caps_reached")
            self._write(state)
            return self.status()

    def refresh_share_fit(self) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            if state["phase"] != "share_fit":
                return self.status()
            self._refresh_share_fit_locked(state)
            self._write(state)
            return self.status()

    def lock_share_fit(self, *, actor: str, donor_count: int | None = None) -> dict[str, Any]:
        with self.lock:
            state = self._load()
            if state["phase"] != "share_fit":
                raise OrchestrationError("当前不在 share 拟合阶段")
            self._reconcile_end_to_end_locked(state)
            completed = list(state["share_fit"].get("completed_donor_ids") or [])
            if not completed:
                raise OrchestrationError("尚无完成会话的 share 拟合受试者")
            n = int(donor_count or len(completed))
            if n < 1 or n > len(completed):
                raise OrchestrationError("拟合人数超出已完成 donor 范围")
            prior = self._ensure_prior_for_donors(completed[:n])
            self.db.select_shared_prior(
                prior["run_id"],
                prior["id"],
                actor=actor,
                reason=f"auto-orchestration lock N*={n}",
            )
            state["share_fit"]["locked_n"] = n
            state["share_fit"]["locked_prior_id"] = prior["id"]
            state["share_fit"]["candidate_n"] = n
            state["share_fit"]["lock_reason"] = "manual"
            state["phase"] = "contrast"
            state["status"] = "running"
            state["contrast"]["next_index"] = 0
            self._append_history(
                state,
                "lock_share_fit",
                actor=actor,
                locked_n=n,
                prior_id=prior["id"],
            )
            self._write(state)
            return self.status()

    def _refresh_share_fit_locked(
        self, state: dict[str, Any], *, reconcile: bool = True
    ) -> None:
        if reconcile:
            self._reconcile_end_to_end_locked(state)
        completed = list(state["share_fit"].get("completed_donor_ids") or [])
        if not completed:
            return
        max_n = int(
            state["share_fit"].get("max_donors_for_auto") or MAX_DONORS_FOR_AUTO
        )
        completed = completed[:max_n]
        deltas: list[dict[str, Any]] = []
        previous_vector: np.ndarray | None = None
        previous_norm = 0.0
        for index in range(1, len(completed) + 1):
            prior = self._ensure_prior_for_donors(completed[:index])
            head_path = Path(prior["encoder_path"])
            release = DistilHuBERTRelease.load(self.settings.distilhubert_release_manifest)
            head = _validated_head(head_path, release.feature_dim)
            fingerprint = _head_fingerprint(head)
            head_vector = np.concatenate(
                (
                    head.weight.astype(np.float64).reshape(-1),
                    np.asarray([head.bias], dtype=np.float64),
                )
            )
            head_norm = float(np.linalg.norm(head_vector))
            if previous_vector is None:
                rel_delta = None
            else:
                delta = float(np.linalg.norm(head_vector - previous_vector))
                rel_delta = delta / max(previous_norm, 1e-6)
            deltas.append(
                {
                    "n": index,
                    "prior_id": prior["id"],
                    "fingerprint": fingerprint,
                    "rel_delta": rel_delta,
                }
            )
            previous_vector = head_vector
            previous_norm = head_norm
        state["share_fit"]["delta_history"] = deltas
        min_n = int(state["share_fit"].get("min_donors_for_auto") or MIN_DONORS_FOR_AUTO)
        stable_required = int(
            state["share_fit"].get("stable_required") or STABLE_REQUIRED
        )
        threshold = float(
            state["share_fit"].get("rel_delta_threshold") or REL_DELTA_THRESHOLD
        )
        parameter_candidate = None
        trailing_stable = 0
        for row in reversed(deltas):
            rel_delta = row.get("rel_delta")
            if rel_delta is None:
                break
            if float(rel_delta) < threshold:
                trailing_stable += 1
                continue
            break
        if len(deltas) >= min_n and trailing_stable >= stable_required:
            parameter_candidate = len(deltas) - trailing_stable + 1

        stats = [self._participant_tap_stats(participant_id) for participant_id in completed]
        behavior = _behavior_fit_summary(
            stats,
            min_n=min_n,
            window=int(state["share_fit"].get("behavior_window") or BEHAVIOR_WINDOW),
            improve_epsilon=float(
                state["share_fit"].get("opening_improve_epsilon")
                or OPENING_IMPROVE_EPSILON
            ),
            rebound_tolerance=float(
                state["share_fit"].get("tap_delta_tolerance")
                or TAP_DELTA_TOLERANCE
            ),
            equiv_epsilon=float(
                state["share_fit"].get("opening_equiv_epsilon")
                or OPENING_EQUIV_EPSILON
            ),
            persist_after_nonoverlap=int(
                state["share_fit"].get("plateau_persist_after_nonoverlap")
                or PLATEAU_PERSIST_AFTER_NONOVERLAP
            ),
            rebound_confirm_windows=int(
                state["share_fit"].get("rebound_confirm_windows")
                or REBOUND_CONFIRM_WINDOWS
            ),
        )
        state["share_fit"]["behavior_history"] = behavior["history"]
        state["share_fit"]["parameter_candidate_n"] = parameter_candidate
        state["share_fit"]["behavior_candidate_n"] = behavior["candidate_n"]

        stop_reason = _share_fit_stop_reason(
            completed_n=len(completed),
            max_n=max_n,
            behavior_state=behavior["current_state"],
            parameter_candidate_n=parameter_candidate,
            persist_ready=bool(behavior.get("persist_ready")),
            rebound_confirmed=bool(behavior.get("rebound_confirmed")),
            confirm_horizon=int(
                state["share_fit"].get("confirm_horizon_donors")
                or CONFIRM_HORIZON_DONORS
            ),
        )

        selected_n = _select_share_fit_n(behavior["candidate_n"], parameter_candidate)
        if stop_reason is not None and selected_n is None:
            selected_n = len(completed)
        state["share_fit"]["candidate_n"] = selected_n
        e2e_records = self.db.end_to_end_completion_records(completed)
        pending_questionnaires = [
            participant_id
            for participant_id in completed
            if participant_id not in e2e_records
        ]
        state["share_fit"]["auto_lock_pending_questionnaire_ids"] = (
            pending_questionnaires
        )
        if (
            selected_n is not None
            and not state["share_fit"].get("locked_prior_id")
            and stop_reason
            and not pending_questionnaires
        ):
            selected_n = max(1, min(int(selected_n), len(completed)))
            prior = self._ensure_prior_for_donors(completed[:selected_n])
            self.db.select_shared_prior(
                prior["run_id"],
                prior["id"],
                actor="orchestration",
                reason=f"auto share-fit {stop_reason} N*={selected_n}",
            )
            state["share_fit"]["locked_n"] = selected_n
            state["share_fit"]["locked_prior_id"] = prior["id"]
            state["share_fit"]["lock_reason"] = stop_reason
            state["phase"] = "contrast"
            state["status"] = "running"
            state["contrast"]["next_index"] = 0
            self._append_history(
                state,
                "auto_lock_share_fit",
                locked_n=selected_n,
                prior_id=prior["id"],
                reason=stop_reason,
                completed_donors=len(completed),
                tap_delta=behavior["current_tap_delta"],
            )

    def _personal_model_root(self, participant: dict[str, Any]) -> Path:
        participant_id = str(participant["id"])
        if bool(participant.get("analysis_excluded")) and self.settings.instance_id == "test":
            return (
                self.settings.data_dir
                / "personal_debug"
                / "public_enrollment"
                / participant_id
                / "models"
            )
        return self.settings.data_dir / "models" / participant_id

    def _personal_checkpoint_receipt(
        self, participant: dict[str, Any]
    ) -> dict[str, Any]:
        participant_id = str(participant["id"])
        try:
            return validated_personal_checkpoint(
                self.settings.distilhubert_release_manifest,
                self._personal_model_root(participant),
            )
        except (OSError, ValueError, DistilHuBERTUnavailable) as exc:
            raise OrchestrationError(
                f"完成者 {participant_id} 缺少有效的 DistilHuBERT personal "
                f"checkpoint: {exc}"
            ) from exc

    def _personal_head_path(self, participant: dict[str, Any]) -> Path:
        return Path(
            self._personal_checkpoint_receipt(participant)[
                "personal_head_path"
            ]
        )

    @staticmethod
    def _prior_donor_ids(prior: dict[str, Any]) -> list[str]:
        raw = prior.get("donor_ids")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OrchestrationError("Shared prior donor 列表不是合法 JSON") from exc
        if not isinstance(raw, list):
            raise OrchestrationError("Shared prior donor 列表无效")
        return [str(value) for value in raw]

    def _validate_prior_record(
        self,
        prior: dict[str, Any],
        donor_ids: list[str],
        release: DistilHuBERTRelease,
    ) -> dict[str, Any]:
        if self._prior_donor_ids(prior) != donor_ids:
            raise OrchestrationError("Shared prior 不属于精确的已提交 donor 前缀")
        if (
            int(prior.get("donor_count") or 0) != len(donor_ids)
            or int(
                prior.get("opportunity_budget")
                if prior.get("opportunity_budget") is not None
                else -1
            )
            != SHARE_FIT_BUDGET
            or str(prior.get("algorithm") or "") != DISTILHUBERT_PRIOR_ALGORITHM
        ):
            raise OrchestrationError("Shared prior 数据库元数据不一致")
        head_path = Path(str(prior.get("encoder_path") or "")).resolve()
        pointer = Path(str(prior.get("main_path") or "")).resolve()
        if (
            not head_path.is_file()
            or file_sha256(head_path) != str(prior.get("encoder_sha256") or "")
            or not pointer.is_file()
            or file_sha256(pointer) != str(prior.get("main_sha256") or "")
        ):
            raise OrchestrationError("Shared prior 权重或指针缺失/已变化")
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestrationError("Shared prior 指针无效") from exc
        pointer_head = (
            pointer.parent / str(payload.get("personal_head_path") or "")
        ).resolve()
        if (
            pointer_head != head_path
            or str(payload.get("model_family") or "") != MODEL_FAMILY
            or str(payload.get("revision") or "") != release.revision
            or str(payload.get("release_manifest_sha256") or "")
            != release.manifest_sha256
            or str(payload.get("shared_head_sha256") or "")
            != release.shared_head_sha256
            or int(payload.get("donor_count") or 0) != len(donor_ids)
            or str(payload.get("algorithm") or "")
            != DISTILHUBERT_PRIOR_ALGORITHM
            or str(payload.get("personal_head_sha256") or "")
            != file_sha256(head_path)
        ):
            raise OrchestrationError("Shared prior 指针元数据不一致")
        head = _validated_head(head_path, release.feature_dim)
        if str(payload.get("personal_head_fingerprint") or "") != _head_fingerprint(
            head
        ):
            raise OrchestrationError("Shared prior 数值指纹已变化")
        return prior

    def _ensure_prior_for_donors(self, donor_ids: list[str]) -> dict[str, Any]:
        if not donor_ids:
            raise OrchestrationError("donor 列表为空")
        run_slug = self.settings.public_enrollment_run_slug
        run = self.db.study_run(run_slug)
        if run is None:
            # Ensure run exists by reading any donor's run.
            participant = self.db.participant(donor_ids[0])
            if participant is None:
                raise OrchestrationError("无法定位 public enrollment run")
            run_id = str(participant["run_id"])
        else:
            run_id = str(run["id"])
        existing = [
            row
            for row in self.db.shared_priors(run_id)
            if int(row["donor_count"]) == len(donor_ids)
            and int(row["opportunity_budget"]) == SHARE_FIT_BUDGET
            and str(row.get("algorithm") or "") == DISTILHUBERT_PRIOR_ALGORITHM
            and self._prior_donor_ids(row) == donor_ids
        ]
        if self.settings.agent_mode != "distilhubert":
            raise OrchestrationError("自动 share 拟合目前仅支持 DistilHuBERT")
        release = DistilHuBERTRelease.load(self.settings.distilhubert_release_manifest)
        if existing:
            return self._validate_prior_record(existing[0], donor_ids, release)
        heads: list[LinearHeadState] = []
        for donor_id in donor_ids:
            participant = self.db.participant(donor_id)
            if participant is None:
                raise OrchestrationError(f"未知 donor {donor_id}")
            if str(participant["run_id"]) != run_id:
                raise OrchestrationError("donor 不属于同一 run")
            path = self._personal_head_path(participant)
            heads.append(_validated_head(path, release.feature_dim))
        weight = np.mean([head.weight for head in heads], axis=0).astype(np.float32)
        bias = float(np.mean([head.bias for head in heads], dtype=np.float64))
        # Keep the release feature scaler; donors must not change it.
        averaged = LinearHeadState(
            feature_mean=heads[0].feature_mean.copy(),
            feature_scale=heads[0].feature_scale.copy(),
            weight=weight,
            bias=bias,
        )
        for head in heads[1:]:
            if not np.array_equal(head.feature_mean, averaged.feature_mean) or not np.array_equal(
                head.feature_scale, averaged.feature_scale
            ):
                raise OrchestrationError("donor personal head 与 release scaler 不一致")
        prior_id = str(uuid.uuid4())
        target = self.settings.data_dir / "priors" / run_id / prior_id
        target.mkdir(parents=True, exist_ok=False)
        head_path = target / "personal_head.npz"
        write_head_artifact(head_path, averaged)
        pointer = target / "distilhubert-prior.json"
        pointer_payload = {
            "schema": "tap-to-adapt-distilhubert-shared-prior/v1",
            "model_family": MODEL_FAMILY,
            "revision": release.revision,
            "release_manifest_sha256": release.manifest_sha256,
            "shared_head_sha256": release.shared_head_sha256,
            "personal_head_path": head_path.name,
            "personal_head_sha256": file_sha256(head_path),
            "personal_head_fingerprint": _head_fingerprint(averaged),
            "donor_count": len(donor_ids),
            "algorithm": DISTILHUBERT_PRIOR_ALGORITHM,
        }
        pointer.write_text(
            json.dumps(pointer_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.db.create_shared_prior(
            {
                "id": prior_id,
                "run_id": run_id,
                "donor_count": len(donor_ids),
                "opportunity_budget": SHARE_FIT_BUDGET,
                "status": "candidate",
                "algorithm": DISTILHUBERT_PRIOR_ALGORITHM,
                "donor_ids": donor_ids,
                "encoder_path": str(head_path.resolve()),
                "main_path": str(pointer.resolve()),
                "encoder_sha256": file_sha256(head_path),
                "main_sha256": file_sha256(pointer),
                "manifest": {
                    "model_family": "distilhubert",
                    "budget_kind": "session_end_personal_head",
                    "donors": donor_ids,
                    "fingerprint": pointer_payload["personal_head_fingerprint"],
                },
            },
            allow_analysis_excluded=self.settings.instance_id == "test",
        )

    def _participant_tap_stats(self, participant_id: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT event_type, payload FROM events
                   WHERE participant_id=?
                      AND event_type IN (
                          'model.prediction',
                          'tap.manual_activate',
                         'tap.interrupt',
                         'tap.state_transition_mistouch'
                     )""",
                (participant_id,),
            ).fetchall()
        activate = 0
        interrupt = 0
        mistouch = 0
        eligible_opportunities = 0
        eligible_corrections = 0
        opening_opportunities = 0
        opening_corrections = 0
        for row in rows:
            event_type = str(row["event_type"])
            if event_type == "tap.state_transition_mistouch":
                mistouch += 1
                continue
            try:
                payload = json.loads(row["payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            if payload.get("analysis_eligible") is False:
                continue
            try:
                block = int(
                    payload.get("active_block_index", payload.get("block", 0)) or 0
                )
            except (TypeError, ValueError):
                block = 0
            in_opening = block == OPENING_BLOCK_INDEX
            if event_type == "model.prediction":
                if payload.get("decision_eligible") is True and not payload.get("skipped"):
                    eligible_opportunities += 1
                    if in_opening:
                        opening_opportunities += 1
                continue
            if payload.get("cancelled") is True:
                continue
            if event_type == "tap.manual_activate":
                activate += 1
            elif event_type == "tap.interrupt":
                interrupt += 1
            if payload.get("training_eligible") is True:
                eligible_corrections += 1
                if in_opening:
                    opening_corrections += 1
        tap_count = activate + interrupt
        minutes = EFFECTIVE_INTERACTION_MINUTES
        correction_burden = (
            eligible_corrections / eligible_opportunities
            if eligible_opportunities > 0
            else None
        )
        opening_correction_burden = (
            opening_corrections / opening_opportunities
            if opening_opportunities > 0
            else None
        )
        return {
            "manual_activate_count": activate,
            "interrupt_count": interrupt,
            "mistouch_count": mistouch,
            "tap_count": tap_count,
            "eligible_opportunities": eligible_opportunities,
            "eligible_correction_count": eligible_corrections,
            "correction_burden": correction_burden,
            "opening_eligible_opportunities": opening_opportunities,
            "opening_eligible_correction_count": opening_corrections,
            "opening_correction_burden": opening_correction_burden,
            "effective_minutes": minutes,
            "taps_per_minute": tap_count / minutes if minutes > 0 else None,
        }

    def share_fit_curve(self) -> dict[str, Any]:
        """Tap-frequency curve as share-fit donors accumulate (ordered by completion)."""
        with self.lock:
            state = self._load()
            reconciled, _ = self._reconcile_end_to_end_locked(state)
            if reconciled:
                self._write(state)
            completed = list(state["share_fit"].get("completed_donor_ids") or [])
            delta_by_n = {
                int(row["n"]): row
                for row in (state["share_fit"].get("delta_history") or [])
                if isinstance(row, dict) and row.get("n") is not None
            }
            behavior_by_n = {
                int(row["n"]): row
                for row in (state["share_fit"].get("behavior_history") or [])
                if isinstance(row, dict) and row.get("n") is not None
            }
            points: list[dict[str, Any]] = []
            tap_rates: list[float] = []
            for index, participant_id in enumerate(completed, start=1):
                stats = self._participant_tap_stats(participant_id)
                rate = float(stats["taps_per_minute"] or 0.0)
                tap_rates.append(rate)
                cumulative_mean = sum(tap_rates) / len(tap_rates)
                delta = delta_by_n.get(index) or {}
                behavior = behavior_by_n.get(index) or {}
                points.append(
                    {
                        "n": index,
                        "participant_id": participant_id,
                        "tap_count": stats["tap_count"],
                        "manual_activate_count": stats["manual_activate_count"],
                        "interrupt_count": stats["interrupt_count"],
                        "mistouch_count": stats["mistouch_count"],
                        "eligible_opportunities": stats["eligible_opportunities"],
                        "eligible_correction_count": stats["eligible_correction_count"],
                        "correction_burden": stats["correction_burden"],
                        "opening_eligible_opportunities": stats[
                            "opening_eligible_opportunities"
                        ],
                        "opening_eligible_correction_count": stats[
                            "opening_eligible_correction_count"
                        ],
                        "opening_correction_burden": stats["opening_correction_burden"],
                        "taps_per_minute": round(rate, 6),
                        "cumulative_mean_taps_per_minute": round(cumulative_mean, 6),
                        "rel_delta": delta.get("rel_delta"),
                        "rolling_3_correction_burden": behavior.get("rolling_burden"),
                        "tap_delta": behavior.get("tap_delta"),
                        "behavior_state": behavior.get("state"),
                        "prior_id": delta.get("prior_id"),
                    }
                )
            return {
                "instance_id": self.settings.instance_id,
                "phase": state.get("phase"),
                "locked_n": state["share_fit"].get("locked_n"),
                "candidate_n": state["share_fit"].get("candidate_n"),
                "parameter_candidate_n": state["share_fit"].get("parameter_candidate_n"),
                "behavior_candidate_n": state["share_fit"].get("behavior_candidate_n"),
                "lock_reason": state["share_fit"].get("lock_reason"),
                "completed": len(completed),
                "effective_interaction_minutes": EFFECTIVE_INTERACTION_MINUTES,
                "min_donors_for_auto": MIN_DONORS_FOR_AUTO,
                "stable_required": STABLE_REQUIRED,
                "rel_delta_threshold": REL_DELTA_THRESHOLD,
                "max_donors_for_auto": MAX_DONORS_FOR_AUTO,
                "behavior_window": BEHAVIOR_WINDOW,
                "opening_improve_epsilon": OPENING_IMPROVE_EPSILON,
                "opening_equiv_epsilon": OPENING_EQUIV_EPSILON,
                "plateau_persist_after_nonoverlap": PLATEAU_PERSIST_AFTER_NONOVERLAP,
                "confirm_horizon_donors": CONFIRM_HORIZON_DONORS,
                "rebound_confirm_windows": REBOUND_CONFIRM_WINDOWS,
                "tap_delta_tolerance": TAP_DELTA_TOLERANCE,
                "points": points,
                "series": {
                    "n": [row["n"] for row in points],
                    "taps_per_minute": [row["taps_per_minute"] for row in points],
                    "cumulative_mean_taps_per_minute": [
                        row["cumulative_mean_taps_per_minute"] for row in points
                    ],
                    "rel_delta": [row["rel_delta"] for row in points],
                    "correction_burden": [row["correction_burden"] for row in points],
                    "opening_correction_burden": [
                        row["opening_correction_burden"] for row in points
                    ],
                    "rolling_3_correction_burden": [
                        row["rolling_3_correction_burden"] for row in points
                    ],
                    "tap_delta": [row["tap_delta"] for row in points],
                },
            }

    def _read_state_readonly(self) -> dict[str, Any]:
        """Parse orchestration JSON without creating, migrating, or writing it."""
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _existing_prefix_prior(
        self, run_id: str, donor_ids: list[str]
    ) -> dict[str, Any] | None:
        for row in self.db.shared_priors(run_id):
            if (
                int(row.get("donor_count") or 0) == len(donor_ids)
                and int(
                    row.get("opportunity_budget")
                    if row.get("opportunity_budget") is not None
                    else -1
                )
                == SHARE_FIT_BUDGET
                and str(row.get("algorithm") or "") == DISTILHUBERT_PRIOR_ALGORITHM
                and self._prior_donor_ids(row) == donor_ids
            ):
                return row
        return None

    def existing_share_fit_weight_snapshot(self) -> dict[str, Any]:
        """Read the latest already-written share-fit prior. Never writes or moves files."""
        state = self._read_state_readonly()
        share = state.get("share_fit") or {}
        completed = [str(value) for value in (share.get("completed_donor_ids") or [])]
        locked_id = str(share.get("locked_prior_id") or "").strip()
        run_id = ""
        run = self.db.study_run(self.settings.public_enrollment_run_slug)
        if run is not None:
            run_id = str(run["id"])
        elif completed:
            participant = self.db.participant(completed[0])
            if participant is not None:
                run_id = str(participant["run_id"])
        prior = None
        source = None
        if locked_id:
            candidate = self.db.shared_prior(locked_id)
            if candidate is not None:
                prior = candidate
                source = "locked"
        if prior is None and run_id and completed:
            for length in range(len(completed), 0, -1):
                candidate = self._existing_prefix_prior(run_id, completed[:length])
                if candidate is not None:
                    prior = candidate
                    source = "latest_prefix"
                    break
        if prior is None:
            raise OrchestrationError("尚无已落盘的 Share 拟合权重，无法只读下载")
        head_path = Path(str(prior.get("encoder_path") or "")).resolve()
        pointer_path = Path(str(prior.get("main_path") or "")).resolve()
        if not head_path.is_file() or not pointer_path.is_file():
            raise OrchestrationError("Share 拟合权重文件缺失，拒绝生成以免改写存储")
        recorded_head = str(prior.get("encoder_sha256") or "")
        recorded_pointer = str(prior.get("main_sha256") or "")
        head_sha256 = file_sha256(head_path)
        pointer_sha256 = file_sha256(pointer_path)
        if recorded_head and recorded_head != head_sha256:
            raise OrchestrationError("Share 拟合头文件哈希与数据库记录不一致")
        if recorded_pointer and recorded_pointer != pointer_sha256:
            raise OrchestrationError("Share 拟合指针哈希与数据库记录不一致")
        return {
            "source": source,
            "prior_id": str(prior["id"]),
            "run_id": str(prior.get("run_id") or run_id),
            "donor_count": int(prior.get("donor_count") or 0),
            "donor_ids": self._prior_donor_ids(prior),
            "algorithm": str(prior.get("algorithm") or ""),
            "locked": source == "locked",
            "head_name": head_path.name,
            "pointer_name": pointer_path.name,
            "head_bytes": head_path.read_bytes(),
            "pointer_bytes": pointer_path.read_bytes(),
            "head_sha256": head_sha256,
            "pointer_sha256": pointer_sha256,
            "head_path": str(head_path),
            "pointer_path": str(pointer_path),
        }


def _cst_date_from_ms(start_ms: int) -> str:
    return datetime.fromtimestamp(
        int(start_ms) / 1000, SHARE_APPOINTMENT_TZ
    ).strftime("%Y-%m-%d")


def _operator_blocked_starts(share: dict[str, Any]) -> set[int]:
    starts: set[int] = set()
    for row in share.get("operator_blocked_slots") or []:
        if not isinstance(row, dict):
            continue
        start_ms = int(row.get("scheduled_start_ms") or 0)
        if start_ms > 0:
            starts.add(start_ms)
    return starts


def _operator_blocked_slot_records(share: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        {
            "scheduled_start_ms": int(row.get("scheduled_start_ms") or 0),
            "date_cst": _cst_date_from_ms(int(row.get("scheduled_start_ms") or 0)),
            "created_ms": int(row.get("created_ms") or 0),
        }
        for row in (share.get("operator_blocked_slots") or [])
        if isinstance(row, dict) and int(row.get("scheduled_start_ms") or 0) > 0
    ]
    return sorted(records, key=lambda row: row["scheduled_start_ms"])


def _share_fit_remaining_confidence(
    *,
    completed_n: int,
    min_n: int,
    remaining: int,
    locked: bool,
    stop_reason: str | None,
    trailing_stable: int,
    stable_required: int,
    behavior_state: str | None,
    tap_delta: float | None,
    improve_epsilon: float,
    last_rel_deltas: list[float],
    threshold: float,
) -> float:
    """0–1 confidence that the current remaining estimate matches the lock gates.

    This is a descriptive score from already-observed relΔ and opening-q
    windows. It does not write state or change recruitment.
    """
    if locked or stop_reason:
        return 1.0
    n_score = min(1.0, completed_n / max(min_n, 1))
    param_stable = min(1.0, trailing_stable / max(stable_required, 1))
    if last_rel_deltas:
        mean_rel = sum(abs(value) for value in last_rel_deltas) / len(last_rel_deltas)
        param_margin = max(
            0.0, min(1.0, 1.0 - (mean_rel / max(threshold * 3.0, 1e-9)))
        )
    else:
        param_margin = 0.0 if completed_n <= 0 else 0.25
    param_score = 0.5 * param_stable + 0.5 * param_margin
    state = str(behavior_state or "insufficient")
    if state == "plateau":
        opening_score = 0.92
    elif state == "rebound":
        opening_score = 0.88
    elif state == "baseline":
        opening_score = 0.55
    elif state == "improving":
        if tap_delta is not None:
            margin = min(
                1.0,
                abs(float(tap_delta)) / max(float(improve_epsilon) * 4.0, 1e-9),
            )
            opening_score = 0.32 + 0.28 * margin
        else:
            opening_score = 0.40
    else:
        opening_score = 0.12 + 0.28 * n_score
    remaining_score = (
        1.0 if remaining == 0 else max(0.2, 1.0 - remaining / max(min_n, 1))
    )
    bound_penalty = 0.12 if state == "improving" and remaining > 0 else 0.0
    confidence = (
        0.20 * n_score
        + 0.30 * param_score
        + 0.35 * opening_score
        + 0.15 * remaining_score
        - bound_penalty
    )
    if completed_n <= 0:
        confidence = min(confidence, 0.12)
    return round(max(0.0, min(1.0, confidence)), 3)


def _share_fit_parameter_gate(
    deltas: list[dict[str, Any]],
    *,
    completed_n: int,
    min_n: int,
    stable_required: int,
    threshold: float,
    persisted_candidate: int | None,
) -> tuple[int | None, int, int]:
    trailing_stable = 0
    for row in reversed(deltas):
        rel_delta = row.get("rel_delta")
        if rel_delta is None:
            break
        if float(rel_delta) < threshold:
            trailing_stable += 1
            continue
        break
    if deltas:
        if completed_n >= min_n and trailing_stable >= stable_required:
            parameter_candidate_n = completed_n - trailing_stable + 1
        else:
            parameter_candidate_n = None
    else:
        parameter_candidate_n = persisted_candidate
    if parameter_candidate_n is not None:
        parameter_additional = 0
    else:
        parameter_additional = max(
            0, min_n - completed_n, stable_required - trailing_stable
        )
    return parameter_candidate_n, trailing_stable, parameter_additional


def _share_fit_opening_gate(
    *,
    completed_n: int,
    min_n: int,
    opening_stats: list[dict[str, Any]] | None,
    behavior_history: list[dict[str, Any]],
    share: dict[str, Any],
) -> tuple[str | None, int | None, float | None, bool, int | None, bool, int | None]:
    confirm_windows = int(
        share.get("rebound_confirm_windows") or REBOUND_CONFIRM_WINDOWS
    )
    if opening_stats is not None:
        summary = _behavior_fit_summary(
            opening_stats,
            min_n=min_n,
            window=int(share.get("behavior_window") or BEHAVIOR_WINDOW),
            improve_epsilon=float(
                share.get("opening_improve_epsilon") or OPENING_IMPROVE_EPSILON
            ),
            rebound_tolerance=float(
                share.get("tap_delta_tolerance") or TAP_DELTA_TOLERANCE
            ),
            equiv_epsilon=float(
                share.get("opening_equiv_epsilon") or OPENING_EQUIV_EPSILON
            ),
            persist_after_nonoverlap=int(
                share.get("plateau_persist_after_nonoverlap")
                or PLATEAU_PERSIST_AFTER_NONOVERLAP
            ),
            rebound_confirm_windows=confirm_windows,
        )
        return (
            summary.get("current_state"),
            summary.get("candidate_n"),
            summary.get("current_tap_delta"),
            bool(summary.get("persist_ready")),
            summary.get("persist_need_n"),
            bool(summary.get("rebound_confirmed")),
            summary.get("rebound_need_n"),
        )
    extracted: list[dict[str, Any]] = []
    for row in behavior_history:
        burden = row.get("opening_correction_burden")
        if burden is None:
            extracted = []
            break
        extracted.append({"opening_correction_burden": float(burden)})
    if extracted and len(extracted) == completed_n:
        return _share_fit_opening_gate(
            completed_n=completed_n,
            min_n=min_n,
            opening_stats=extracted,
            behavior_history=[],
            share=share,
        )
    current = {}
    for row in behavior_history:
        if int(row.get("n") or 0) == completed_n:
            current = row
    if not current and behavior_history:
        last = behavior_history[-1]
        if int(last.get("n") or 0) == completed_n or completed_n <= 0:
            current = last
    persisted_candidate = share.get("behavior_candidate_n")
    if persisted_candidate is not None:
        persisted_candidate = int(persisted_candidate)
    tap_delta = current.get("tap_delta")
    window = int(share.get("behavior_window") or BEHAVIOR_WINDOW)
    persist_ready, persist_need_n = _plateau_persist_from_history(
        behavior_history,
        min_n=min_n,
        window=window,
        persist_after_nonoverlap=int(
            share.get("plateau_persist_after_nonoverlap")
            or PLATEAU_PERSIST_AFTER_NONOVERLAP
        ),
        equiv_epsilon=float(
            share.get("opening_equiv_epsilon") or OPENING_EQUIV_EPSILON
        ),
    )
    eligible = [
        row
        for row in behavior_history
        if isinstance(row, dict)
        and int(row.get("n") or 0) >= min_n
        and row.get("rolling_burden") is not None
    ]
    _, _, _, n_first = _terminal_plateau_from_rows(
        eligible,
        window=window,
        persist_after_nonoverlap=int(
            share.get("plateau_persist_after_nonoverlap")
            or PLATEAU_PERSIST_AFTER_NONOVERLAP
        ),
        equiv_epsilon=float(
            share.get("opening_equiv_epsilon") or OPENING_EQUIV_EPSILON
        ),
    )
    rebound_confirmed, rebound_need_n, _ = _rebound_confirm_from_rows(
        [
            row
            for row in behavior_history
            if isinstance(row, dict)
            and int(row.get("n") or 0) >= min_n
            and row.get("state") not in {None, "insufficient"}
        ],
        window=window,
        confirm_windows=confirm_windows,
        n_first=n_first,
    )
    return (
        current.get("state"),
        persisted_candidate,
        None if tap_delta is None else float(tap_delta),
        persist_ready,
        persist_need_n,
        rebound_confirmed,
        rebound_need_n,
    )


def _share_fit_remaining_core(
    share: dict[str, Any],
    *,
    opening_stats: list[dict[str, Any]] | None = None,
    apply_current_lock: bool = True,
) -> dict[str, Any]:
    donor_ids = [
        str(participant_id)
        for participant_id in (share.get("completed_donor_ids") or [])
        if participant_id
    ]
    completed_n = len(donor_ids) if donor_ids else int(share.get("completed") or 0)
    min_n = int(share.get("min_donors_for_auto") or MIN_DONORS_FOR_AUTO)
    stable_required = int(share.get("stable_required") or STABLE_REQUIRED)
    threshold = float(share.get("rel_delta_threshold") or REL_DELTA_THRESHOLD)
    max_n = int(share.get("max_donors_for_auto") or MAX_DONORS_FOR_AUTO)
    locked = bool(
        apply_current_lock
        and (share.get("locked_prior_id") or share.get("locked_n") is not None)
    )
    persisted_parameter = share.get("parameter_candidate_n")
    if persisted_parameter is not None:
        persisted_parameter = int(persisted_parameter)
    candidate_n = share.get("candidate_n")
    if candidate_n is not None:
        candidate_n = int(candidate_n)
    deltas = [
        row
        for row in (share.get("delta_history") or [])
        if isinstance(row, dict) and int(row.get("n") or 0) <= completed_n
    ]
    behavior_history = [
        row
        for row in (share.get("behavior_history") or [])
        if isinstance(row, dict) and int(row.get("n") or 0) <= completed_n
    ]
    prefix_stats = None
    if opening_stats is not None:
        prefix_stats = list(opening_stats)[:completed_n]
    parameter_candidate_n, trailing_stable, parameter_additional = (
        _share_fit_parameter_gate(
            deltas,
            completed_n=completed_n,
            min_n=min_n,
            stable_required=stable_required,
            threshold=threshold,
            persisted_candidate=persisted_parameter,
        )
    )
    (
        behavior_state,
        behavior_candidate_n,
        tap_delta,
        persist_ready,
        persist_need_n,
        rebound_confirmed,
        rebound_need_n,
    ) = _share_fit_opening_gate(
        completed_n=completed_n,
        min_n=min_n,
        opening_stats=prefix_stats,
        behavior_history=behavior_history,
        share=share,
    )
    if rebound_confirmed:
        behavior_additional = 0
    elif completed_n < min_n:
        behavior_additional = min_n - completed_n
    elif behavior_state == "rebound":
        extras = [1]
        if rebound_need_n is not None and int(rebound_need_n) > completed_n:
            extras.append(int(rebound_need_n) - completed_n)
        if persist_need_n is not None and int(persist_need_n) > completed_n:
            extras.append(int(persist_need_n) - completed_n)
        behavior_additional = max(extras)
    elif behavior_state == "improving":
        behavior_additional = 1
    elif persist_ready:
        behavior_additional = 0
    elif persist_need_n is not None and int(persist_need_n) > completed_n:
        behavior_additional = int(persist_need_n) - completed_n
    else:
        behavior_additional = 1
    confirm_horizon = int(
        share.get("confirm_horizon_donors") or CONFIRM_HORIZON_DONORS
    )
    stop_reason = _share_fit_stop_reason(
        completed_n=completed_n,
        max_n=max_n,
        behavior_state=behavior_state,
        parameter_candidate_n=parameter_candidate_n,
        persist_ready=persist_ready,
        rebound_confirmed=rebound_confirmed,
        confirm_horizon=confirm_horizon,
    )
    if locked or stop_reason:
        remaining = 0
    else:
        remaining = max(parameter_additional, behavior_additional)
    last_rel_deltas = [
        float(row["rel_delta"])
        for row in deltas
        if row.get("rel_delta") is not None
    ][-stable_required:]
    improve_epsilon = float(
        share.get("opening_improve_epsilon") or OPENING_IMPROVE_EPSILON
    )
    confidence = _share_fit_remaining_confidence(
        completed_n=completed_n,
        min_n=min_n,
        remaining=remaining,
        locked=locked,
        stop_reason=stop_reason,
        trailing_stable=trailing_stable,
        stable_required=stable_required,
        behavior_state=behavior_state,
        tap_delta=tap_delta,
        improve_epsilon=improve_epsilon,
        last_rel_deltas=last_rel_deltas,
        threshold=threshold,
    )
    reason_labels = {
        "manual": "管理员锁定",
        "parameter_and_behavior_plateau": "参数+开场平台",
        "behavior_rebound": "行为回弹",
        "max_donors": "达到50人上限",
        "confirm_horizon": "震荡确认上限",
    }
    if locked:
        summary = (
            f"已锁定拟合点 N*={share.get('locked_n')}"
            + (
                f"（{reason_labels.get(str(share.get('lock_reason') or ''), share.get('lock_reason'))}）"
                if share.get("lock_reason")
                else ""
            )
        )
    elif stop_reason:
        summary = (
            "已达到锁仓条件"
            f"（{reason_labels.get(stop_reason, stop_reason)}），预计无需再招募"
        )
    elif remaining == 0:
        summary = "按当前历史，拟合点条件已满足"
    else:
        summary = (
            f"按开场 q 中位闸门与参数 relΔ<{threshold:.0%} 预计还需 {remaining} 人"
        )
    selected_n = _select_share_fit_n(
        behavior_candidate_n, parameter_candidate_n, candidate_n
    )
    return {
        "readonly": True,
        "basis": "opening_q_median_and_parameter_rel_delta",
        "completed_n": completed_n,
        "min_donors_for_auto": min_n,
        "stable_required": stable_required,
        "rel_delta_threshold": threshold,
        "opening_improve_epsilon": improve_epsilon,
        "opening_equiv_epsilon": float(
            share.get("opening_equiv_epsilon") or OPENING_EQUIV_EPSILON
        ),
        "plateau_persist_after_nonoverlap": int(
            share.get("plateau_persist_after_nonoverlap")
            or PLATEAU_PERSIST_AFTER_NONOVERLAP
        ),
        "confirm_horizon_donors": confirm_horizon,
        "rebound_confirm_windows": int(
            share.get("rebound_confirm_windows") or REBOUND_CONFIRM_WINDOWS
        ),
        "persist_ready": persist_ready,
        "persist_need_n": persist_need_n,
        "rebound_confirmed": rebound_confirmed,
        "rebound_need_n": rebound_need_n,
        "tap_delta_tolerance": float(
            share.get("tap_delta_tolerance") or TAP_DELTA_TOLERANCE
        ),
        "max_donors_for_auto": max_n,
        "locked": locked,
        "locked_n": share.get("locked_n") if apply_current_lock else None,
        "lock_reason": share.get("lock_reason") if apply_current_lock else None,
        "candidate_n": selected_n,
        "parameter_candidate_n": parameter_candidate_n,
        "behavior_candidate_n": behavior_candidate_n,
        "current_behavior_state": behavior_state,
        "current_opening_delta": tap_delta,
        "parameter_trailing_stable": trailing_stable,
        "parameter_additional_donors": parameter_additional,
        "behavior_additional_donors": behavior_additional,
        "estimated_additional_donors": remaining,
        "confidence": confidence,
        "upper_bound_additional_donors": max(0, max_n - completed_n),
        "stop_reason_if_ready": stop_reason,
        "fit_point_reached": bool(locked or stop_reason or remaining == 0),
        "pending_questionnaire_count": len(
            share.get("auto_lock_pending_questionnaire_ids") or []
        ),
        "summary": summary,
    }


def _share_fit_remaining_estimate(
    share: dict[str, Any],
    *,
    opening_stats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read-only forecast of extra donors needed to reach the lock gates.

    Uses already-persisted donor counts and delta history. Opening gates are
    recomputed from opening-q stats when provided, otherwise from persisted
    behavior history. Does not refresh priors, write models, or change state.
    """
    current = _share_fit_remaining_core(
        share,
        opening_stats=opening_stats,
        apply_current_lock=True,
    )
    donor_ids = [
        str(participant_id)
        for participant_id in (share.get("completed_donor_ids") or [])
        if participant_id
    ]
    history: list[dict[str, Any]] = []
    for prefix_n in range(0, int(current["completed_n"]) + 1):
        prefix_share = {
            "completed_donor_ids": donor_ids[:prefix_n],
            "completed": prefix_n,
            "min_donors_for_auto": current["min_donors_for_auto"],
            "stable_required": current["stable_required"],
            "rel_delta_threshold": current["rel_delta_threshold"],
            "max_donors_for_auto": current["max_donors_for_auto"],
            "opening_improve_epsilon": current["opening_improve_epsilon"],
            "opening_equiv_epsilon": current.get("opening_equiv_epsilon"),
            "plateau_persist_after_nonoverlap": current.get(
                "plateau_persist_after_nonoverlap"
            ),
            "confirm_horizon_donors": current.get("confirm_horizon_donors"),
            "rebound_confirm_windows": current.get("rebound_confirm_windows"),
            "tap_delta_tolerance": current["tap_delta_tolerance"],
            "behavior_window": share.get("behavior_window") or BEHAVIOR_WINDOW,
            "delta_history": share.get("delta_history") or [],
            "behavior_history": share.get("behavior_history") or [],
            "parameter_candidate_n": None,
            "behavior_candidate_n": None,
            "candidate_n": None,
            "auto_lock_pending_questionnaire_ids": [],
        }
        point = _share_fit_remaining_core(
            prefix_share,
            opening_stats=opening_stats,
            apply_current_lock=False,
        )
        history.append(
            {
                "n": prefix_n,
                "estimated_additional_donors": point["estimated_additional_donors"],
                "confidence": point["confidence"],
                "current_behavior_state": point["current_behavior_state"],
                "parameter_trailing_stable": point["parameter_trailing_stable"],
                "parameter_candidate_n": point["parameter_candidate_n"],
                "behavior_candidate_n": point["behavior_candidate_n"],
                "parameter_additional_donors": point["parameter_additional_donors"],
                "behavior_additional_donors": point["behavior_additional_donors"],
            }
        )
    current["history"] = history
    return current


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-7
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for step in range(1, max_iter + 1):
        m2 = 2 * step
        aa = step * (b - step) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + step) * (qab + step) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(
        math.log(max(x, 1e-300)) * a
        + math.log(max(1.0 - x, 1e-300)) * b
        - ln_beta
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_two_tailed_p(t_stat: float, df: float) -> float:
    if df <= 0 or not math.isfinite(t_stat):
        return 0.0 if math.isfinite(t_stat) is False and df > 0 else 1.0
    x = df / (df + t_stat * t_stat)
    return min(1.0, max(0.0, _regularized_incomplete_beta(df / 2.0, 0.5, x)))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        rank = 0.5 * (index + 1 + end)
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    count = len(xs)
    if count < 2:
        return 0.0
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x < 1e-15 or den_y < 1e-15:
        return 0.0
    return max(-1.0, min(1.0, num / (den_x * den_y)))


def _spearman_trend(xs: list[float], ys: list[float | None]) -> dict[str, Any]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if y is not None and math.isfinite(float(y))
    ]
    count = len(pairs)
    if count < 3:
        return {
            "n": count,
            "rho": None,
            "p_value": None,
            "significant": False,
        }
    rho = _pearson_corr(
        _average_ranks([row[0] for row in pairs]),
        _average_ranks([row[1] for row in pairs]),
    )
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((count - 2) / max(1e-12, 1.0 - rho * rho))
        p_value = _student_t_two_tailed_p(t_stat, float(count - 2))
    return {
        "n": count,
        "rho": round(rho, 4),
        "p_value": round(p_value, 4),
        "significant": bool(p_value < SHARE_FIT_TREND_ALPHA),
    }


def _iqr_inlier_mask(values: list[float | None]) -> list[bool]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    present = [
        value is not None and math.isfinite(float(value)) for value in values
    ]
    if len(finite) < 4:
        return present
    q1, q3 = np.percentile(np.asarray(finite, dtype=np.float64), [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return present
    low, high = float(q1) - 1.5 * iqr, float(q3) + 1.5 * iqr
    return [
        bool(
            value is not None
            and math.isfinite(float(value))
            and low <= float(value) <= high
        )
        for value in values
    ]


def _trend_reading(
    *,
    rho: float | None,
    significant: bool,
    lower_is_better: bool,
) -> str:
    if rho is None:
        return "样本不足"
    improving = (rho < 0) if lower_is_better else (rho > 0)
    if significant:
        return "向好（显著）" if improving else "变差（显著）"
    if abs(rho) < 0.15:
        return "平（不显著）"
    if improving:
        return "弱向好（不显著）"
    return "略变差（不显著）"


def _empty_block_metric() -> dict[str, Any]:
    return {
        "opportunities": 0,
        "corrections": 0,
        "taps": 0,
        "taps_per_minute": None,
        "q": None,
    }


def _accumulate_share_fit_block_event(
    blocks: dict[int, dict[str, Any]],
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if payload.get("analysis_eligible") is False:
        return
    try:
        block = int(payload.get("active_block_index", payload.get("block", 0)) or 0)
    except (TypeError, ValueError):
        block = 0
    if block not in {1, 2, 3, 4, 5}:
        return
    metric = blocks.setdefault(block, _empty_block_metric())
    if event_type == "model.prediction":
        if payload.get("decision_eligible") is True and not payload.get("skipped"):
            metric["opportunities"] += 1
        return
    if event_type == "tap.state_transition_mistouch":
        return
    if payload.get("cancelled") is True:
        return
    if event_type in {"tap.manual_activate", "tap.interrupt"}:
        metric["taps"] += 1
        if payload.get("training_eligible") is True:
            metric["corrections"] += 1


def _finalize_block_metrics(blocks: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    finalized: dict[int, dict[str, Any]] = {}
    for block in range(1, 6):
        metric = dict(blocks.get(block) or _empty_block_metric())
        opportunities = int(metric["opportunities"])
        taps = int(metric["taps"])
        metric["taps_per_minute"] = (
            round(taps / SHARE_FIT_BLOCK_MINUTES, 6) if taps or opportunities else None
        )
        metric["q"] = (
            round(int(metric["corrections"]) / opportunities, 6)
            if opportunities > 0
            else None
        )
        finalized[block] = metric
    return finalized


def _share_fit_series_trend(
    donor_ns: list[int],
    values: list[float | None],
    codes: list[str],
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    mask = _iqr_inlier_mask(values)
    kept_n = [n for n, keep in zip(donor_ns, mask) if keep]
    kept_values = [value for value, keep in zip(values, mask) if keep]
    dropped = [
        code
        for code, keep, value in zip(codes, mask, values)
        if value is not None and math.isfinite(float(value)) and not keep
    ]
    trend = _spearman_trend(kept_n, kept_values)
    reading = _trend_reading(
        rho=trend["rho"],
        significant=bool(trend["significant"]),
        lower_is_better=lower_is_better,
    )
    finite = [float(value) for value in kept_values if value is not None]
    return {
        **trend,
        "dropped_n": len(dropped),
        "dropped_codes": dropped,
        "significant_label": "显著" if trend["significant"] else "不显著",
        "reading": reading,
        "inlier_mean": round(sum(finite) / len(finite), 4) if finite else None,
    }


def _opening_lock_burden(row: dict[str, Any]) -> float | None:
    value = row.get("opening_correction_burden")
    if value is None:
        value = row.get("correction_burden")
    return None if value is None else float(value)


def _select_share_fit_n(*candidates: int | None) -> int | None:
    values = [int(value) for value in candidates if value is not None]
    return max(values) if values else None


def _terminal_plateau_from_rows(
    eligible: list[dict[str, Any]],
    *,
    window: int,
    persist_after_nonoverlap: int,
    equiv_epsilon: float,
) -> tuple[int | None, bool, int | None, int | None]:
    """Return terminal-min start, persist flag, needed n, and first min n."""

    if not eligible:
        return None, False, None, None
    best = min(float(row["rolling_burden"]) for row in eligible)
    at_min = [
        int(row["n"])
        for row in eligible
        if float(row["rolling_burden"]) <= best + equiv_epsilon
    ]
    if not at_min:
        return None, False, None, None
    end = max(at_min)
    start = end
    at_min_set = set(at_min)
    while start - 1 in at_min_set:
        start -= 1
    n_first = min(at_min)
    persist_need_n = n_first + window + persist_after_nonoverlap - 1
    completed_n = max(int(row["n"]) for row in eligible)
    persist_ready = completed_n >= persist_need_n
    return start, persist_ready, persist_need_n, n_first


def _plateau_persist_from_history(
    history: list[dict[str, Any]],
    *,
    min_n: int,
    window: int,
    persist_after_nonoverlap: int,
    equiv_epsilon: float,
) -> tuple[bool, int | None]:
    eligible = [
        row
        for row in history
        if isinstance(row, dict)
        and row.get("rolling_burden") is not None
        and int(row.get("n") or 0) >= min_n
    ]
    _, persist_ready, persist_need_n, _ = _terminal_plateau_from_rows(
        eligible,
        window=window,
        persist_after_nonoverlap=persist_after_nonoverlap,
        equiv_epsilon=equiv_epsilon,
    )
    return persist_ready, persist_need_n


def _behavior_fit_summary(
    stats: list[dict[str, Any]],
    *,
    min_n: int = MIN_DONORS_FOR_AUTO,
    window: int = BEHAVIOR_WINDOW,
    improve_epsilon: float = OPENING_IMPROVE_EPSILON,
    rebound_tolerance: float = TAP_DELTA_TOLERANCE,
    tolerance: float | None = None,
    equiv_epsilon: float = OPENING_EQUIV_EPSILON,
    persist_after_nonoverlap: int = PLATEAU_PERSIST_AFTER_NONOVERLAP,
    rebound_confirm_windows: int = REBOUND_CONFIRM_WINDOWS,
) -> dict[str, Any]:
    """Select the terminal minimum Q3 run and classify the latest trend.

    The lock signal is each donor's first-block eligible correction rate.  The
    latest ``window`` donors are summarized by their median and compared with
    the best earlier median window.  A drop larger than ``improve_epsilon``
    keeps share-fit open.  A single rise larger than ``rebound_tolerance`` is
    diagnostic; automatic stop requires confirmed consecutive rebound after
    the first min plus one non-overlapping window.  The candidate is the
    start of the latest exact-min Q3 run.
    """

    if tolerance is not None:
        rebound_tolerance = float(tolerance)
    history: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    for index, row in enumerate(stats, start=1):
        opening_burden = _opening_lock_burden(row)
        entry: dict[str, Any] = {
            "n": index,
            "correction_burden": opening_burden,
            "opening_correction_burden": opening_burden,
            "rolling_burden": None,
            "best_prior_rolling_burden": None,
            "tap_delta": None,
            "state": "insufficient",
        }
        if index >= window:
            recent = [
                _opening_lock_burden(stats[position])
                for position in range(index - window, index)
            ]
            if all(value is not None for value in recent):
                entry["rolling_burden"] = float(
                    np.median(np.asarray(recent, dtype=np.float64))
                )
        if index >= min_n and entry["rolling_burden"] is not None:
            prior = [
                candidate
                for candidate in rolling_rows
                if int(candidate["n"]) >= min_n
            ]
            if prior:
                best_prior = min(float(candidate["rolling_burden"]) for candidate in prior)
                tap_delta = float(entry["rolling_burden"]) - best_prior
                entry["best_prior_rolling_burden"] = best_prior
                entry["tap_delta"] = tap_delta
                if tap_delta > rebound_tolerance:
                    entry["state"] = "rebound"
                elif tap_delta < -improve_epsilon:
                    entry["state"] = "improving"
                else:
                    entry["state"] = "plateau"
            else:
                entry["state"] = "baseline"
            rolling_rows.append(entry)
        history.append(entry)

    eligible = [
        row
        for row in rolling_rows
        if row.get("rolling_burden") is not None and int(row["n"]) >= min_n
    ]
    candidate_n, persist_ready, persist_need_n, n_first = _terminal_plateau_from_rows(
        eligible,
        window=window,
        persist_after_nonoverlap=persist_after_nonoverlap,
        equiv_epsilon=equiv_epsilon,
    )
    rebound_confirmed, rebound_need_n, n_first_rebound = _rebound_confirm_from_rows(
        eligible,
        window=window,
        confirm_windows=rebound_confirm_windows,
        n_first=n_first,
    )
    current = history[-1] if history else {}
    return {
        "history": history,
        "candidate_n": candidate_n,
        "current_state": current.get("state"),
        "current_tap_delta": current.get("tap_delta"),
        "persist_ready": persist_ready,
        "persist_need_n": persist_need_n,
        "n_first_min": n_first,
        "rebound_confirmed": rebound_confirmed,
        "rebound_need_n": rebound_need_n,
        "n_first_rebound": n_first_rebound,
    }


def _rebound_confirm_from_rows(
    eligible: list[dict[str, Any]],
    *,
    window: int,
    confirm_windows: int,
    n_first: int | None,
) -> tuple[bool, int | None, int | None]:
    """Confirm rebound only after a min plus consecutive rebound windows."""

    if n_first is None or not eligible:
        return False, None, None
    completed_n = max(int(row["n"]) for row in eligible)
    by_n = {int(row["n"]): row for row in eligible}
    trailing = 0
    cursor = completed_n
    while cursor in by_n and str(by_n[cursor].get("state") or "") == "rebound":
        trailing += 1
        cursor -= 1
    if trailing <= 0:
        return False, None, None
    n_first_rebound = completed_n - trailing + 1
    min_after_first = int(n_first) + int(window)
    need_n = max(n_first_rebound + int(confirm_windows) - 1, min_after_first)
    confirmed = (
        completed_n >= need_n
        and trailing >= int(confirm_windows)
        and completed_n >= min_after_first
    )
    return confirmed, need_n, n_first_rebound


def _share_fit_stop_reason(
    *,
    completed_n: int,
    max_n: int,
    behavior_state: str | None,
    parameter_candidate_n: int | None,
    persist_ready: bool = False,
    rebound_confirmed: bool = False,
    confirm_horizon: int = CONFIRM_HORIZON_DONORS,
) -> str | None:
    """Return the automatic stop reason, if any."""

    if completed_n >= max_n:
        return "max_donors"
    if rebound_confirmed:
        return "behavior_rebound"
    if completed_n >= confirm_horizon:
        return "confirm_horizon"
    if (
        persist_ready
        and parameter_candidate_n is not None
        and behavior_state == "plateau"
    ):
        return "parameter_and_behavior_plateau"
    return None


def seed_initial_head_from_prior(
    model: DistilHuBERTOnlineModel,
    prior_dir_or_pointer: Path,
) -> None:
    """Load averaged personal-head weights as a fresh starting point.

    Counters and personal replay buffers are reset so a new participant can
    learn (or freeze) from the shared start without inheriting donor counters.
    """
    pointer = Path(prior_dir_or_pointer)
    if pointer.is_dir():
        pointer = pointer / "distilhubert-prior.json"
    if not pointer.is_file():
        raise DistilHuBERTUnavailable("Shared DistilHuBERT prior pointer missing")
    try:
        raw = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistilHuBERTUnavailable("Shared DistilHuBERT prior pointer invalid") from exc
    head_path = (pointer.parent / str(raw.get("personal_head_path") or "")).resolve()
    if not head_path.is_file():
        raise DistilHuBERTUnavailable("Shared DistilHuBERT prior head missing")
    personal = _validated_head(head_path, model.release.feature_dim)
    if (
        not np.array_equal(personal.feature_mean, model.shared_head.feature_mean)
        or not np.array_equal(personal.feature_scale, model.shared_head.feature_scale)
    ):
        raise DistilHuBERTUnavailable("Shared prior changed the release feature scaler")
    for array in (personal.feature_mean, personal.feature_scale):
        array.setflags(write=False)
    model.personal_head = personal
    model.update_count = 0
    model.optimizer_step_count = 0
    model._replay_features = {0: [], 1: []}
    model._replay_seen = {0: 0, 1: 0}
