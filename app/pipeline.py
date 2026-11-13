"""Explicit study-wave orchestration and shared-prior handoff.

FormalPipeline validates wave definitions, completion budgets, compatible
checkpoints, and assignment readiness before releasing dependent waves.
Simulation access is controlled separately from the formal pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .db import StudyDB
from .integrity import file_sha256
from .matched import matched_assignment, matched_paths
from .priors import (
    PriorError,
    build_shared_prior,
    create_base_prior,
    delta_transport_checkpoint,
)
from .study_analysis import analyze_run


class PipelineError(RuntimeError):
    pass


class FormalPipeline:
    def __init__(
        self,
        db: StudyDB,
        settings: Settings,
        study: dict[str, Any],
        *,
        allow_simulation: bool = False,
    ):
        self.db = db
        self.settings = settings
        self.study = study
        self.rules = study["pipeline"]
        self.total_sessions = int(
            study.get("session_design", {}).get("total_sessions", 3)
        )
        self.session_days = tuple(range(1, self.total_sessions + 1))
        self.wave_sizes = [int(value) for value in self.rules["study_a_wave_sizes"]]
        self.wave_roles = [str(value) for value in self.rules["study_a_wave_roles"]]
        if len(self.wave_roles) != len(self.wave_sizes):
            raise PipelineError("Study A wave roles and sizes must have the same length")
        self.donor_checkpoints = [
            int(value) for value in self.rules["study_a_donor_checkpoints"]
        ]
        self.candidate_plans = {
            int(wave): {
                "cells": [tuple(int(value) for value in cell) for cell in plan["cells"]],
                "default_cell": tuple(int(value) for value in plan["default_cell"]),
            }
            for wave, plan in self.rules["candidate_probe_plans"].items()
        }
        self.selectable_cells = {
            tuple(int(value) for value in cell)
            for cell in self.rules["study_b_selectable_cells"]
        }
        self.study_b_initial_wave = len(self.wave_sizes) + 1
        self.study_b_reserve_wave = self.study_b_initial_wave + 1
        self.budget_kind = str(self.rules.get("budget_kind") or "activation_opportunity")
        self.budget_semantics = str(
            self.rules.get("budget_semantics") or "exact_count"
        )
        self.require_max_q_attainment = bool(
            self.rules.get("require_max_q_attainment", True)
        )
        level_key = (
            "q_feedback_levels"
            if self.budget_kind == "eligible_feedback"
            else "q_opportunity_levels"
        )
        self.q_levels = [int(value) for value in self.rules[level_key]]
        self.allow_simulation = allow_simulation
        self.bound_run_id: str | None = None
        self.bound_protocol_sha256: str | None = None

    @classmethod
    def from_run(
        cls,
        db: StudyDB,
        settings: Settings,
        run_slug: str,
        *,
        allow_simulation: bool = False,
    ) -> "FormalPipeline":
        run = db.study_run(run_slug)
        if run is None:
            raise PipelineError(f"Unknown study run: {run_slug}")
        protocol_text = str(run.get("protocol_json") or "")
        expected_sha256 = str(run.get("protocol_sha256") or "")
        if (
            not protocol_text
            or not expected_sha256
            or hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
            != expected_sha256
        ):
            raise PipelineError("Study run protocol snapshot integrity check failed")
        try:
            study = json.loads(protocol_text)
        except json.JSONDecodeError as exc:
            raise PipelineError("Study run protocol snapshot is invalid JSON") from exc
        if str(study.get("version") or "") != str(run["protocol_version"]):
            raise PipelineError("Study run protocol version does not match its snapshot")
        pipeline = cls(
            db,
            settings,
            study,
            allow_simulation=allow_simulation,
        )
        pipeline.bound_run_id = str(run["id"])
        pipeline.bound_protocol_sha256 = expected_sha256
        return pipeline

    def _budget_count(self, participant_id: str) -> int:
        if self.budget_kind == "eligible_feedback":
            return self.db.feedback_counts(participant_id)["eligible_feedbacks"]
        return self.db.learning_opportunity_count(participant_id)

    def _run(self, slug: str) -> dict[str, Any]:
        run = self.db.study_run(slug)
        if run is None:
            raise PipelineError(f"Unknown study run: {slug}")
        valid_mode = run["data_origin"] == "formal" and run["protocol_mode"] == "strict"
        if self.allow_simulation:
            valid_mode = valid_mode or (
                run["data_origin"] == "simulation" and run["protocol_mode"] == "compressed"
            )
        if not valid_mode or run["protocol_version"] != self.study["version"]:
            raise PipelineError("Pipeline operations require an approved current study run")
        if self.bound_run_id is not None and (
            str(run["id"]) != self.bound_run_id
            or str(run.get("protocol_sha256") or "")
            != self.bound_protocol_sha256
        ):
            raise PipelineError("Pipeline instance is bound to a different immutable study run")
        return run

    def _study_sessions_complete(self, db: StudyDB, participant_id: str) -> bool:
        sessions = db.sessions(participant_id)
        expected = {
            (day, role) for day in self.session_days for role in ("study", "companion")
        }
        actual = {
            (int(row["day"]), str(row["client_role"]))
            for row in sessions
            if row["state"] in {"completed", "ended"}
        }
        # New single-session runs require only their snapshotted session set.
        # Legacy runs may contain additional already-closed day rows; those
        # must not make an otherwise complete immutable run look incomplete.
        return expected.issubset(actual)

    def _participants(self, run_id: str) -> list[dict[str, Any]]:
        return self.db.participants_in_run(run_id)

    def _study_a(self, run_id: str) -> list[dict[str, Any]]:
        return sorted(
            [row for row in self._participants(run_id) if row["cohort"] == "study_a"],
            key=lambda row: str(row.get("assignment_id") or ""),
        )

    def _study_b(self, run_id: str) -> list[dict[str, Any]]:
        return sorted(
            [row for row in self._participants(run_id) if row["cohort"] == "study_b"],
            key=lambda row: str(row.get("neutral_id") or ""),
        )

    @staticmethod
    def _is_study_a_donor(participant: dict[str, Any]) -> bool:
        return str(participant.get("group_name") or "").lower() in {
            "donor",
            "seed_donor",
            "evaluator_donor",
        }

    def _study_a_donors(self, run_id: str) -> list[dict[str, Any]]:
        return [row for row in self._study_a(run_id) if self._is_study_a_donor(row)]

    @staticmethod
    def _payload(row: dict[str, Any]) -> dict[str, Any]:
        value = row.get("payload")
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}

    def _probe_execution(self, participant: dict[str, Any]) -> dict[str, Any]:
        assignments = self.db.prior_probe_assignments(
            participant_id=participant["id"], day=1
        )
        if not assignments:
            return {"required": 0, "completed": 0, "complete": True, "missing": []}
        sessions = [
            row
            for row in self.db.sessions(participant["id"])
            if int(row["day"]) == 1 and row["client_role"] == "study"
        ]
        ended_tokens: set[str] = set()
        started_tokens: set[str] = set()
        committed_tokens: set[str] = set()
        if sessions:
            snapshot = self.db.protocol_snapshot(participant["id"], sessions[0]["id"])
            for row in snapshot["events"]:
                payload = self._payload(row)
                if payload.get("phase") != "candidate_probe":
                    continue
                token = str(payload.get("policy_token") or "")
                if row["event_type"] == "phase.start":
                    started_tokens.add(token)
                elif row["event_type"] == "phase.end":
                    ended_tokens.add(token)
                elif row["event_type"] == "phase.server_committed":
                    committed_tokens.add(token)
        expected = {str(row["probe_token"]) for row in assignments}
        observed = started_tokens & ended_tokens & committed_tokens
        complete = expected.issubset(observed)
        return {
            "required": len(expected),
            "completed": len(expected & observed),
            "complete": complete,
            "missing": sorted(expected - observed),
        }

    def _require_probe_execution(self, participants: list[dict[str, Any]]) -> None:
        incomplete = [
            str(row.get("assignment_id"))
            for row in participants
            if not self._probe_execution(row)["complete"]
        ]
        if incomplete:
            raise PipelineError(
                "Blinded candidate-prior probes are incomplete: " + ", ".join(incomplete)
            )

    def _require_donors_ready(self, donors: list[dict[str, Any]]) -> None:
        incomplete = [row["assignment_id"] for row in donors if not self._study_sessions_complete(self.db, row["id"])]
        if incomplete:
            raise PipelineError(f"Study A donors have incomplete study sessions: {incomplete}")
        if self.require_max_q_attainment:
            max_q = max(self.q_levels)
            under_budget = [
                row["assignment_id"]
                for row in donors
                if self._budget_count(row["id"]) < max_q
            ]
            if under_budget:
                raise PipelineError(
                    f"Study A donors have fewer than {max_q} {self.budget_kind} updates: {under_budget}"
                )
        for budget in self.q_levels:
            snapshots = {
                row["participant_id"]: row
                for row in self.db.model_snapshots(
                    donors[0]["run_id"],
                    source="donor_shadow",
                    budget=budget,
                    budget_kind=self.budget_kind,
                )
            }
            missing = [row["assignment_id"] for row in donors if row["id"] not in snapshots]
            if missing:
                raise PipelineError(f"Missing q={budget} donor snapshots: {missing}")
            invalid = [
                row["assignment_id"]
                for row in donors
                if int(snapshots[row["id"]]["eligible_feedbacks"])
                != min(self._budget_count(row["id"]), budget)
            ]
            if self.budget_semantics == "per_donor_cap" and invalid:
                raise PipelineError(
                    f"Invalid q={budget} capped donor snapshots: {invalid}"
                )

    def _donor_readiness(self, donor: dict[str, Any]) -> dict[str, Any]:
        counts = self.db.feedback_counts(donor["id"])
        observed = self._budget_count(donor["id"])
        snapshot_rows = {
            int(row["opportunity_budget"]): row
            for row in self.db.model_snapshots(
                donor["run_id"], source="donor_shadow", budget_kind=self.budget_kind
            )
            if row["participant_id"] == donor["id"]
        }
        snapshots = sorted(snapshot_rows)
        missing_snapshots = sorted(set(self.q_levels) - set(snapshots))
        invalid_snapshots = sorted(
            budget
            for budget, row in snapshot_rows.items()
            if budget in self.q_levels
            and self.budget_semantics == "per_donor_cap"
            and int(row["eligible_feedbacks"]) != min(observed, budget)
        )
        sessions_complete = self._study_sessions_complete(self.db, donor["id"])
        return {
            "assignment_id": donor.get("assignment_id"),
            "sessions_complete": sessions_complete,
            "budget_kind": self.budget_kind,
            "budget_semantics": self.budget_semantics,
            "learning_opportunities": observed,
            "required_opportunities": max(self.q_levels),
            "max_q_attainment_required": self.require_max_q_attainment,
            "activation_opportunities": counts["activation_opportunities"],
            "eligible_feedbacks": counts["eligible_feedbacks"],
            "positive_feedbacks": counts["positive_feedbacks"],
            "negative_feedbacks": counts["negative_feedbacks"],
            "snapshots": snapshots,
            "missing_snapshots": missing_snapshots,
            "invalid_snapshots": invalid_snapshots,
            "cap_attainment": {
                str(budget): observed >= budget for budget in self.q_levels
            },
            "realized_feedback_by_cap": {
                str(budget): (
                    int(snapshot_rows[budget]["eligible_feedbacks"])
                    if budget in snapshot_rows
                    else None
                )
                for budget in self.q_levels
            },
            "eligible_for_prior": (
                sessions_complete
                and not missing_snapshots
                and not invalid_snapshots
                and (
                    not self.require_max_q_attainment
                    or observed >= max(self.q_levels)
                )
            ),
        }

    def _prior(self, run_id: str, donor_count: int, budget: int) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in self.db.shared_priors(run_id)
                if int(row["donor_count"]) == donor_count
                and int(row["opportunity_budget"]) == budget
            ),
            None,
        )

    def build_nested_priors(self, slug: str, donor_count: int) -> list[dict[str, Any]]:
        run = self._run(slug)
        donors = self._study_a_donors(run["id"])
        if donor_count < 1 or donor_count > len(donors):
            raise PipelineError("Invalid nested donor count")
        selected = donors[:donor_count]
        self._require_donors_ready(selected)
        priors = []
        for budget in self.q_levels:
            priors.append(
                build_shared_prior(
                    self.db,
                    self.settings,
                    run["id"],
                    [row["id"] for row in selected],
                    budget,
                    self.budget_kind,
                    self.budget_semantics,
                    str(self.rules["shared_prior_algorithm"]),
                )
            )
        self.db.record_pipeline_action(
            run["id"],
            "shared_prior.build_nested",
            actor="pipeline",
            payload={
                "donor_count": donor_count,
                "budget_kind": self.budget_kind,
                "budget_semantics": self.budget_semantics,
                "q_levels": self.q_levels,
                "prior_ids": [row["id"] for row in priors],
            },
        )
        return priors

    def _ensure_prior(
        self, run: dict[str, Any], donor_count: int, budget: int
    ) -> dict[str, Any]:
        if donor_count == 0 and budget == 0:
            return create_base_prior(self.db, self.settings, run["id"])
        prior = self._prior(run["id"], donor_count, budget)
        if prior is None:
            self.build_nested_priors(run["slug"], donor_count)
            prior = self._prior(run["id"], donor_count, budget)
        if prior is None:
            raise PipelineError(f"Unable to build candidate prior M({donor_count},{budget})")
        return prior

    @staticmethod
    def _balanced_probe_order(
        cells: list[tuple[int, int]], participant_index: int
    ) -> list[tuple[int, int]]:
        if len(cells) != 4:
            shift = participant_index % len(cells)
            return cells[shift:] + cells[:shift]
        base = [0, 1, 3, 2]
        shift = participant_index % 4
        indices = [((value + shift) % 4) for value in base]
        if (participant_index // 4) % 2:
            indices.reverse()
        return [cells[index] for index in indices]

    def _assign_study_a_wave(self, run: dict[str, Any], wave: int) -> list[dict[str, Any]]:
        participants = [
            row
            for row in self._study_a(run["id"])
            if int(row.get("enrollment_wave") or 0) == wave
        ]
        if len(participants) != self.wave_sizes[wave - 1]:
            raise PipelineError(f"Wave {wave} roster size does not match the locked protocol")
        expected_role = self.wave_roles[wave - 1]
        invalid_roles = [
            str(row.get("assignment_id"))
            for row in participants
            if str(row.get("group_name")) != expected_role
        ]
        if invalid_roles:
            raise PipelineError(
                f"Wave {wave} roster role must be {expected_role}: {invalid_roles}"
            )
        if wave == 1:
            base = create_base_prior(self.db, self.settings, run["id"])
            for participant in participants:
                if participant.get("prior_id") is None:
                    self.db.assign_prior(participant["id"], base["id"], actor="pipeline")
            return participants

        previous_n = self.donor_checkpoints[wave - 2]
        donors = self._study_a_donors(run["id"])[:previous_n]
        if len(donors) != previous_n:
            raise PipelineError(f"Expected {previous_n} completed donors before wave {wave}")
        self._require_donors_ready(donors)
        self._require_probe_execution(donors)
        plan = self.candidate_plans.get(wave)
        if plan is None:
            raise PipelineError(f"Wave {wave} has no locked candidate-probe plan")
        cells = list(plan["cells"])
        priors = {
            cell: self._ensure_prior(run, cell[0], cell[1]) for cell in set(cells)
        }
        default = priors[plan["default_cell"]]
        for index, participant in enumerate(participants):
            if participant.get("prior_id") is None:
                self.db.assign_prior(participant["id"], default["id"], actor="pipeline")
            ordered_cells = self._balanced_probe_order(cells, index)
            self.db.create_prior_probe_assignments(
                participant["id"],
                1,
                [priors[cell]["id"] for cell in ordered_cells],
            )
        self.db.record_pipeline_action(
            run["id"],
            "study_a.candidate_probes_frozen",
            actor="pipeline",
            payload={
                "wave": wave,
                "cells": [list(cell) for cell in cells],
                "default_cell": list(plan["default_cell"]),
                "participant_count": len(participants),
                "blinded": True,
            },
        )
        return participants

    def release_study_a_wave(self, slug: str, wave: int) -> dict[str, Any]:
        run = self._run(slug)
        if wave < 1 or wave > len(self.wave_sizes):
            raise PipelineError("Invalid Study A wave")
        participants = self._assign_study_a_wave(run, wave)
        for participant in participants:
            refreshed = self.db.participant(participant["id"])
            if refreshed and refreshed.get("release_state") != "released":
                self.db.set_release_state(
                    participant["id"],
                    "released",
                    actor="pipeline",
                    reason=f"Study A wave {wave} prerequisites passed",
                )
        return {
            "run_slug": slug,
            "wave": wave,
            "released": [row["neutral_id"] for row in participants],
        }

    def _write_json_atomic(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def _prepare_matched_models(
        self,
        run: dict[str, Any],
        selected_prior: dict[str, Any],
        donors: list[dict[str, Any]],
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        seed = int.from_bytes(hashlib.sha256(run["id"].encode()).digest()[:8], "big")
        snapshots = {
            (row["participant_id"], int(row["opportunity_budget"])): row
            for row in self.db.model_snapshots(
                run["id"], source="donor_shadow", budget_kind=self.budget_kind
            )
        }
        budgets = {
            int(day): int(value)
            for day, value in self.rules["matched_other_budget_by_day"].items()
            if int(day) in self.session_days
        }
        if set(budgets) != set(self.session_days):
            raise PipelineError("Matched-other budgets do not cover every study session")
        artifact_root = self.settings.data_dir / "matched" / run["id"]
        artifacts: dict[tuple[str, int], tuple[Path, Path]] = {}
        for donor in donors:
            for day, budget in budgets.items():
                snapshot = snapshots.get((donor["id"], budget))
                if snapshot is None:
                    raise PipelineError(
                        f"Matched model source is missing: {donor['assignment_id']} q={budget}"
                    )
                target_dir = artifact_root / donor["id"] / f"day-{day}"
                encoder = target_dir / "encoder.pth"
                main = target_dir / "main_model.pth"
                if not encoder.is_file():
                    delta_transport_checkpoint(
                        Path(self.settings.base_encoder_checkpoint),
                        Path(selected_prior["encoder_path"]),
                        Path(snapshot["encoder_path"]),
                        encoder,
                        str(self.rules["matched_other_algorithm"]),
                    )
                if not main.is_file():
                    delta_transport_checkpoint(
                        Path(self.settings.base_main_checkpoint),
                        Path(selected_prior["main_path"]),
                        Path(snapshot["main_path"]),
                        main,
                        str(self.rules["matched_other_algorithm"]),
                    )
                artifacts[(donor["id"], day)] = (encoder, main)

        assignments: dict[str, dict[str, Any]] = {}
        offset = seed % len(donors)
        for index, participant in enumerate(targets):
            assignments[participant["id"]] = {}
            for day, budget in budgets.items():
                donor = donors[(index + offset + (day - 1) * 11) % len(donors)]
                encoder, main = artifacts[(donor["id"], day)]
                assignments[participant["id"]][str(day)] = {
                    "donor_participant_id": donor["id"],
                    "matching_strata": {
                        "shared_prior_id": selected_prior["id"],
                        "opportunity_budget": budget,
                        "budget_kind": self.budget_kind,
                        "day": day,
                    },
                    "encoder_checkpoint": str(encoder.resolve()),
                    "encoder_sha256": file_sha256(encoder),
                    "main_checkpoint": str(main.resolve()),
                    "main_sha256": file_sha256(main),
                }
        manifest = {
            "version": "formal-matched-other-capped-donor-delta-v3",
            "approved": True,
            "run_id": run["id"],
            "run_slug": run["slug"],
            "assignment_rule": (
                "Deterministic balanced rotation over completed Study A donors; shared-prior "
                "plus capped donor-shadow delta at the preregistered day budget."
            ),
            "matching_variables": ["shared_prior_id", "opportunity_budget", "day"],
            "random_seed": seed,
            "frozen_at_ms": int(time.time() * 1000),
            "assignments": assignments,
        }
        manifest_path, sentinel_path = matched_paths(self.settings, run["id"])
        self._write_json_atomic(manifest_path, manifest)
        sentinel = {
            "passed": True,
            "created_ms": int(time.time() * 1000),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_version": manifest["version"],
            "assignments": {
                participant_id: {
                    day: {
                        "donor_participant_id": assignment["donor_participant_id"],
                        "encoder_sha256": assignment["encoder_sha256"],
                        "main_sha256": assignment["main_sha256"],
                    }
                    for day, assignment in day_assignments.items()
                }
                for participant_id, day_assignments in assignments.items()
            },
        }
        self._write_json_atomic(sentinel_path, sentinel)
        failures = []
        for participant in targets:
            for day in self.session_days:
                _, checks = matched_assignment(
                    self.settings, participant["id"], day, run["id"]
                )
                if not all(checks.values()):
                    failures.append(
                        {
                            "participant": participant["neutral_id"],
                            "day": day,
                            "failed": [name for name, passed in checks.items() if not passed],
                        }
                    )
        if failures:
            raise PipelineError(f"Matched-other self-test failed: {failures}")
        return manifest

    def _prepared_study_b_result(
        self,
        run: dict[str, Any],
        targets: list[dict[str, Any]],
        selected_n: int,
        selected_q: int,
    ) -> dict[str, Any] | None:
        prepared = any(
            row["action"] == "study_b.prepare"
            for row in self.db.pipeline_actions(run["id"])
        )
        if not prepared:
            return None
        selected = [
            row for row in self.db.shared_priors(run["id"]) if row["status"] == "selected"
        ]
        if len(selected) != 1:
            raise PipelineError("Frozen Study B preparation has no unique selected prior")
        prior = selected[0]
        if int(prior["donor_count"]) != selected_n or int(prior["opportunity_budget"]) != selected_q:
            raise PipelineError(
                "Study B was already frozen with a different n/q decision; it cannot be changed"
            )
        failures = []
        for participant in targets:
            if participant.get("prior_id") != prior["id"]:
                failures.append(f"{participant['neutral_id']}:prior")
                continue
            for day in self.session_days:
                _, checks = matched_assignment(
                    self.settings, participant["id"], day, run["id"]
                )
                if not all(checks.values()):
                    failures.append(f"{participant['neutral_id']}:day{day}")
        if failures:
            raise PipelineError(
                "Frozen Study B artifacts failed integrity verification: "
                + ", ".join(failures)
            )
        initial = [
            row
            for row in targets
            if int(row.get("enrollment_wave") or 0) == self.study_b_initial_wave
            and row.get("release_state") in {"released", "completed"}
        ]
        return {
            "run_slug": run["slug"],
            "selected_prior_id": prior["id"],
            "selected_n": prior["donor_count"],
            "selected_q": prior["opportunity_budget"],
            "matched_assignments": len(targets) * self.total_sessions,
            "released": [row["neutral_id"] for row in initial],
            "idempotent": True,
        }

    def prepare_study_b(
        self, slug: str, selected_n: int, selected_q: int, decision_record: str
    ) -> dict[str, Any]:
        run = self._run(slug)
        checkpoints = self.donor_checkpoints
        if selected_n not in checkpoints:
            raise PipelineError(f"selected_n must be one of the locked donor checkpoints: {checkpoints}")
        if selected_q not in self.q_levels:
            raise PipelineError(f"selected_q must be one of the locked {self.budget_kind} levels: {self.q_levels}")
        if (selected_n, selected_q) not in self.selectable_cells:
            raise PipelineError(
                f"M({selected_n},{selected_q}) was not in the locked candidate-evaluation grid"
            )
        decision_record = decision_record.strip()
        if len(decision_record) < 20:
            raise PipelineError(
                "A blinded saturation/precision decision record of at least 20 characters is required"
            )
        targets = self._study_b(run["id"])
        expected = 2 * (
            int(self.rules["study_b_initial_per_group"])
            + int(self.rules["study_b_reserve_per_group"])
        )
        if len(targets) != expected:
            raise PipelineError("Study B roster size does not match the locked protocol")
        existing = self._prepared_study_b_result(run, targets, selected_n, selected_q)
        if existing is not None:
            return existing
        roster = self._study_a(run["id"])
        if len(roster) != sum(self.wave_sizes):
            raise PipelineError("Study A roster is incomplete")
        incomplete_study_a = [
            str(row.get("assignment_id"))
            for row in roster
            if not self._study_sessions_complete(self.db, row["id"])
        ]
        if incomplete_study_a:
            raise PipelineError(
                "Study A holdout evaluation must finish before Study B: "
                + ", ".join(incomplete_study_a)
            )
        self._require_probe_execution(roster)
        donors = self._study_a_donors(run["id"])[:selected_n]
        self._require_donors_ready(donors)
        priors = self.build_nested_priors(slug, selected_n)
        selected = next(row for row in priors if int(row["opportunity_budget"]) == selected_q)
        if selected["status"] != "selected":
            self.db.select_shared_prior(
                run["id"],
                selected["id"],
                actor="pipeline",
                reason=decision_record,
            )
            selected = self.db.shared_prior(selected["id"])
            assert selected is not None
        for participant in targets:
            if participant.get("prior_id") is None:
                self.db.assign_prior(participant["id"], selected["id"], actor="pipeline")
        manifest = self._prepare_matched_models(run, selected, donors, targets)
        initial = [
            row
            for row in targets
            if int(row.get("enrollment_wave") or 0) == self.study_b_initial_wave
        ]
        for participant in initial:
            self.db.set_release_state(
                participant["id"],
                "released",
                actor="pipeline",
                reason="Study A complete; shared prior and matched-other assignments frozen",
            )
        self.db.record_pipeline_action(
            run["id"],
            "study_b.prepare",
            actor="pipeline",
            payload={
                "selected_prior_id": selected["id"],
                "selected_n": selected_n,
                "selected_q": selected_q,
                "decision_record": decision_record,
                "initial_released": [row["neutral_id"] for row in initial],
                "matched_manifest_sha256": file_sha256(
                    matched_paths(self.settings, run["id"])[0]
                ),
            },
        )
        return {
            "run_slug": slug,
            "selected_prior_id": selected["id"],
            "selected_n": selected["donor_count"],
            "selected_q": selected["opportunity_budget"],
            "decision_record": decision_record,
            "matched_assignments": sum(len(value) for value in manifest["assignments"].values()),
            "released": [row["neutral_id"] for row in initial],
            "idempotent": False,
        }

    def release_study_b_reserve(self, slug: str, reason: str) -> dict[str, Any]:
        run = self._run(slug)
        reserves = [
            row
            for row in self._study_b(run["id"])
            if int(row.get("enrollment_wave") or 0) == self.study_b_reserve_wave
        ]
        if len(reserves) != 2 * int(self.rules["study_b_reserve_per_group"]):
            raise PipelineError("Study B reserve roster is incomplete")
        for participant in reserves:
            checks = []
            for day in self.session_days:
                _, matched_checks = matched_assignment(
                    self.settings, participant["id"], day, participant["run_id"]
                )
                checks.extend(matched_checks.values())
            if not participant.get("prior_id") or not all(checks):
                raise PipelineError("Study B reserve is missing a prior or matched assignment")
            self.db.set_release_state(
                participant["id"], "released", actor="pipeline", reason=reason
            )
        return {"released": [row["neutral_id"] for row in reserves]}

    def _candidate_probe_status(self, run_id: str) -> dict[str, Any]:
        assignments = self.db.prior_probe_assignments(run_id=run_id, day=1)
        participants = {
            row["id"]: row for row in self._study_a(run_id)
        }
        execution = {
            participant_id: self._probe_execution(participant)
            for participant_id, participant in participants.items()
            if any(row["participant_id"] == participant_id for row in assignments)
        }
        cells: dict[tuple[int, int], dict[str, Any]] = {}
        for row in assignments:
            cell = (int(row["donor_count"]), int(row["opportunity_budget"]))
            summary = cells.setdefault(
                cell,
                {"n": cell[0], "q": cell[1], "assigned": 0, "completed": 0},
            )
            summary["assigned"] += 1
            missing = set(execution[row["participant_id"]]["missing"])
            if str(row["probe_token"]) not in missing:
                summary["completed"] += 1
        return {
            "blinded": True,
            "cells": sorted(cells.values(), key=lambda value: (value["n"], value["q"])),
            "participants_assigned": len(execution),
            "participants_complete": sum(value["complete"] for value in execution.values()),
            "identified_contrasts": {
                "q_at_fixed_n": ["M(12,2/4/8)", "M(20,4/8)", "M(28,4/8)"],
                "n_at_fixed_q": ["M(12/20/28,4)", "M(12/20/28,8)"],
                "equal_total_feedback": ["M(12,8) vs M(20,4)", "M(20,8) vs M(28,4)"],
            },
        }

    def status(self, slug: str) -> dict[str, Any]:
        run = self._run(slug)
        participants = self._participants(run["id"])
        waves: dict[str, Any] = {}
        max_wave = max(
            [int(row.get("enrollment_wave") or 0) for row in participants] or [0]
        )
        for wave in range(1, max_wave + 1):
            rows = [row for row in participants if int(row.get("enrollment_wave") or 0) == wave]
            wave_status: dict[str, Any] = {
                "count": len(rows),
                "released": sum(row.get("release_state") == "released" for row in rows),
                "complete": sum(self._study_sessions_complete(self.db, row["id"]) for row in rows),
            }
            if rows and all(row["cohort"] == "study_a" for row in rows):
                donor_rows = [row for row in rows if self._is_study_a_donor(row)]
                donor_readiness = [self._donor_readiness(row) for row in donor_rows]
                wave_status["eligible_for_prior"] = sum(
                    value["eligible_for_prior"] for value in donor_readiness
                )
                wave_status["prior_blockers"] = [
                    value for value in donor_readiness if not value["eligible_for_prior"]
                ]
            waves[str(wave)] = wave_status
        return {
            "run": run,
            "budget_kind": self.budget_kind,
            "budget_semantics": self.budget_semantics,
            "q_levels": self.q_levels,
            "waves": waves,
            "shared_priors": [
                {
                    "id": row["id"],
                    "n": row["donor_count"],
                    "q": row["opportunity_budget"],
                    "status": row["status"],
                }
                for row in self.db.shared_priors(run["id"])
            ],
            "snapshot_counts": {
                str(budget): len(
                    self.db.model_snapshots(
                        run["id"],
                        source="donor_shadow",
                        budget=budget,
                        budget_kind=self.budget_kind,
                    )
                )
                for budget in self.q_levels
            },
            "candidate_probes": self._candidate_probe_status(run["id"]),
            "study_b_estimands": {
                "shared_gain": "selected_shared_prior - M0 on held-out candidate probes",
                "user_accommodation": "Frozen(Day3-Day1)",
                "personalized_net_gain": "Personalized(Day3-Day1) - Frozen(Day3-Day1)",
                "specificity": "own_personalized - matched_other at equal feedback budget",
                "scope": "model/task/tap-conditioned predictive and behavioral contribution",
            },
            "actions": len(self.db.pipeline_actions(run["id"])),
        }

    def analysis(self, slug: str) -> dict[str, Any]:
        run = self._run(slug)
        return analyze_run(self.db, run["id"], self.study)
