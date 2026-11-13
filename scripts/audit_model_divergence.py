"""Read-only, fail-closed audit of M0/shared/personalized/frozen/matched checkpoint divergence for one study run.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sqlite3
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "tap_model_divergence_audit_v1"
FROZEN_GROUPS = {"frozen", "f"}
PERSONALIZED_GROUPS = {"personalized", "p"}
KNOWN_FLOAT_BUFFER_SUFFIXES = (
    "running_mean",
    "running_var",
)


class AuditFailure(RuntimeError):
    """A fail-closed model-divergence audit failure."""


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise AuditFailure(f"{label} must be a JSON object")
    return parsed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _json_object(path.read_text(encoding="utf-8"), label)
    except OSError as exc:
        raise AuditFailure(f"{label} could not be read: {path}") from exc


def _open_database_read_only(database: Path) -> sqlite3.Connection:
    _expect(database.is_file(), f"study database is missing: {database}")
    uri = f"{database.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=30)
    except sqlite3.Error as exc:
        raise AuditFailure(f"study database could not be opened read-only: {database}") from exc
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _require_schema(connection: sqlite3.Connection) -> None:
    required = {
        "study_runs": {"id", "slug", "protocol_json"},
        "participants": {
            "id",
            "run_id",
            "cohort",
            "group_name",
            "prior_id",
            "release_state",
            "assignment_id",
            "enrollment_wave",
            "created_ms",
        },
        "shared_priors": {
            "id",
            "run_id",
            "donor_count",
            "opportunity_budget",
            "status",
            "encoder_path",
            "main_path",
            "encoder_sha256",
            "main_sha256",
        },
        "model_snapshots": {
            "run_id",
            "participant_id",
            "source",
            "opportunity_budget",
            "encoder_path",
            "main_path",
            "encoder_sha256",
            "main_sha256",
            "created_ms",
            "base_prior_id",
        },
        "participant_calibrations": {
            "participant_id",
            "run_id",
            "update_count",
            "positive_updates",
            "negative_updates",
        },
        "model_counters": {
            "participant_id",
            "run_id",
            "eligible_feedbacks",
            "positive_feedbacks",
            "negative_feedbacks",
        },
        "events": {"participant_id", "event_type", "payload", "seq"},
    }
    for table, columns in required.items():
        actual = _table_columns(connection, table)
        _expect(bool(actual), f"required table is missing: {table}")
        missing = sorted(columns - actual)
        _expect(not missing, f"table {table} is missing columns: {', '.join(missing)}")


def _rows(
    connection: sqlite3.Connection, query: str, values: Iterable[Any] = ()
) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in connection.execute(query, tuple(values)).fetchall()]
    except sqlite3.Error as exc:
        raise AuditFailure(f"read-only database query failed: {exc}") from exc


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    expected_sha256: str | None

    def report(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "hash_matches_record": (
                self.expected_sha256 is None or self.sha256 == self.expected_sha256
            ),
        }


@dataclass(frozen=True)
class CheckpointPair:
    label: str
    encoder: Artifact
    main: Artifact

    @property
    def cache_key(self) -> tuple[str, str]:
        return (str(self.encoder.path), str(self.main.path))

    def report(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "encoder": self.encoder.report(),
            "main": self.main.report(),
        }


@dataclass
class LoadedCheckpoint:
    state: dict[str, Any]
    declared_parameters: tuple[str, ...] | None
    checkpoint_pair_id: str | None


class CheckpointInspector:
    def __init__(self, cache_size: int = 6):
        try:
            import torch
        except ImportError as exc:
            raise AuditFailure("PyTorch is required to inspect checkpoint tensors") from exc
        self.torch = torch
        self.cache_size = cache_size
        self.loaded: OrderedDict[str, LoadedCheckpoint] = OrderedDict()
        self.comparisons: dict[tuple[tuple[str, str], tuple[str, str]], dict[str, Any]] = {}

    def artifact(self, raw_path: str | Path, expected_sha256: str | None, label: str) -> Artifact:
        path = Path(raw_path)
        _expect(path.is_absolute(), f"{label} path must be absolute: {path}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise AuditFailure(f"{label} checkpoint is missing: {path}") from exc
        _expect(resolved.is_file(), f"{label} checkpoint is not a file: {resolved}")
        actual = file_sha256(resolved)
        expected = str(expected_sha256 or "").strip().lower() or None
        if expected is not None:
            _expect(len(expected) == 64, f"{label} has an invalid declared SHA-256")
            _expect(actual == expected, f"{label} SHA-256 does not match its immutable record")
        return Artifact(path=resolved, sha256=actual, expected_sha256=expected)

    def pair(
        self,
        label: str,
        encoder_path: str | Path,
        main_path: str | Path,
        encoder_sha256: str | None = None,
        main_sha256: str | None = None,
    ) -> CheckpointPair:
        return CheckpointPair(
            label=label,
            encoder=self.artifact(encoder_path, encoder_sha256, f"{label} encoder"),
            main=self.artifact(main_path, main_sha256, f"{label} main"),
        )

    @staticmethod
    def _declared_parameter_names(payload: dict[str, Any]) -> tuple[str, ...] | None:
        candidates = [
            payload.get("trainable_parameter_names"),
            payload.get("parameter_names"),
        ]
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend(
                [metadata.get("trainable_parameter_names"), metadata.get("parameter_names")]
            )
        for value in candidates:
            if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                return tuple(sorted(set(value)))
        return None

    def _load(self, artifact: Artifact) -> LoadedCheckpoint:
        key = str(artifact.path)
        cached = self.loaded.pop(key, None)
        if cached is not None:
            self.loaded[key] = cached
            return cached
        try:
            payload = self.torch.load(
                artifact.path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise AuditFailure(
                f"checkpoint cannot be safely loaded with weights_only=True: {artifact.path}"
            ) from exc
        _expect(
            file_sha256(artifact.path) == artifact.sha256,
            f"checkpoint changed while it was being audited: {artifact.path}",
        )
        _expect(isinstance(payload, dict), f"checkpoint payload is not a mapping: {artifact.path}")
        state = payload.get("model_state_dict")
        _expect(isinstance(state, dict) and state, f"model_state_dict is missing: {artifact.path}")
        loaded = LoadedCheckpoint(
            state=state,
            declared_parameters=self._declared_parameter_names(payload),
            checkpoint_pair_id=(
                str(payload["checkpoint_pair_id"])
                if payload.get("checkpoint_pair_id") is not None
                else None
            ),
        )
        self.loaded[key] = loaded
        while len(self.loaded) > self.cache_size:
            self.loaded.popitem(last=False)
            gc.collect()
        return loaded

    @staticmethod
    def _known_buffer(name: str) -> bool:
        return name.endswith(KNOWN_FLOAT_BUFFER_SUFFIXES) or name.endswith(
            "num_batches_tracked"
        )

    def compare_artifacts(self, reference: Artifact, candidate: Artifact) -> dict[str, Any]:
        reference_loaded = self._load(reference)
        candidate_loaded = self._load(candidate)
        reference_state = reference_loaded.state
        candidate_state = candidate_loaded.state
        reference_keys = set(reference_state)
        candidate_keys = set(candidate_state)
        _expect(
            reference_keys == candidate_keys,
            "checkpoint state dictionaries are incompatible: "
            f"missing_in_candidate={sorted(reference_keys - candidate_keys)[:8]}, "
            f"extra_in_candidate={sorted(candidate_keys - reference_keys)[:8]}",
        )
        for name in sorted(reference_keys):
            reference_value = reference_state[name]
            candidate_value = candidate_state[name]
            _expect(
                self.torch.is_tensor(reference_value)
                and self.torch.is_tensor(candidate_value),
                f"state entry is not a tensor: {name}",
            )
            _expect(
                reference_value.shape == candidate_value.shape,
                f"checkpoint tensor shape differs for {name}",
            )
            _expect(
                reference_value.dtype == candidate_value.dtype,
                f"checkpoint tensor dtype differs for {name}",
            )

        declared_reference = reference_loaded.declared_parameters
        declared_candidate = candidate_loaded.declared_parameters
        if declared_reference and declared_candidate:
            _expect(
                declared_reference == declared_candidate,
                "checkpoint trainable-parameter declarations do not match",
            )
        declared = declared_reference or declared_candidate
        if declared:
            names = list(declared)
            basis = "checkpoint_declared_trainable_parameters"
        else:
            names = [
                name
                for name in sorted(reference_keys)
                if self.torch.is_tensor(reference_state[name])
                and (
                    reference_state[name].is_floating_point()
                    or reference_state[name].is_complex()
                )
                and not self._known_buffer(name)
            ]
            basis = "floating_state_tensors_excluding_known_buffers"
        _expect(bool(names), "checkpoint has no comparable trainable floating tensors")

        squared_l2 = 0.0
        reference_squared_l2 = 0.0
        candidate_squared_l2 = 0.0
        max_abs_delta = 0.0
        elements = 0
        changed: list[str] = []
        for name in names:
            _expect(name in reference_state, f"declared parameter is missing: {name}")
            reference_tensor = reference_state[name]
            candidate_tensor = candidate_state[name]
            _expect(
                self.torch.is_tensor(reference_tensor)
                and self.torch.is_tensor(candidate_tensor),
                f"state entry is not a tensor: {name}",
            )
            _expect(
                reference_tensor.shape == candidate_tensor.shape,
                f"checkpoint tensor shape differs for {name}",
            )
            _expect(
                reference_tensor.dtype == candidate_tensor.dtype,
                f"checkpoint tensor dtype differs for {name}",
            )
            _expect(
                reference_tensor.is_floating_point() or reference_tensor.is_complex(),
                f"declared trainable tensor is not floating-point: {name}",
            )
            _expect(
                bool(self.torch.isfinite(reference_tensor).all())
                and bool(self.torch.isfinite(candidate_tensor).all()),
                f"checkpoint tensor contains NaN or infinity: {name}",
            )
            if reference_tensor.is_complex():
                reference_value = reference_tensor.detach().cpu().to(self.torch.complex128)
                candidate_value = candidate_tensor.detach().cpu().to(self.torch.complex128)
            else:
                reference_value = reference_tensor.detach().cpu().to(self.torch.float64)
                candidate_value = candidate_tensor.detach().cpu().to(self.torch.float64)
            delta = candidate_value - reference_value
            tensor_squared = float(delta.abs().square().sum().item())
            squared_l2 += tensor_squared
            reference_squared_l2 += float(reference_value.abs().square().sum().item())
            candidate_squared_l2 += float(candidate_value.abs().square().sum().item())
            max_abs_delta = max(
                max_abs_delta,
                float(delta.abs().max().item()) if delta.numel() else 0.0,
            )
            elements += int(delta.numel())
            if tensor_squared > 0:
                changed.append(name)

        _expect(
            all(math.isfinite(value) for value in (squared_l2, max_abs_delta)),
            "checkpoint L2 calculation is non-finite",
        )
        return {
            "comparison_basis": basis,
            "trainable_float_tensor_count": len(names),
            "trainable_float_element_count": elements,
            "changed_tensor_count": len(changed),
            "changed_tensor_examples": changed[:20],
            "l2_distance": math.sqrt(squared_l2),
            "reference_l2_norm": math.sqrt(reference_squared_l2),
            "candidate_l2_norm": math.sqrt(candidate_squared_l2),
            "max_abs_delta": max_abs_delta,
            "exact_tensor_equal": squared_l2 == 0.0,
        }

    def compare_pairs(
        self, reference: CheckpointPair, candidate: CheckpointPair
    ) -> dict[str, Any]:
        self.validate_pair(reference)
        self.validate_pair(candidate)
        key = (reference.cache_key, candidate.cache_key)
        if key in self.comparisons:
            return self.comparisons[key]
        encoder = self.compare_artifacts(reference.encoder, candidate.encoder)
        main = self.compare_artifacts(reference.main, candidate.main)
        combined_squared = encoder["l2_distance"] ** 2 + main["l2_distance"] ** 2
        result = {
            "reference": reference.label,
            "candidate": candidate.label,
            "encoder": encoder,
            "main": main,
            "combined_l2_distance": math.sqrt(combined_squared),
            "changed_tensor_count": encoder["changed_tensor_count"]
            + main["changed_tensor_count"],
            "exact_tensor_equal": encoder["exact_tensor_equal"]
            and main["exact_tensor_equal"],
        }
        self.comparisons[key] = result
        return result

    def validate_pair(self, pair: CheckpointPair) -> None:
        """Accept legacy pairs with no IDs, but reject partial or crossed new saves."""
        encoder_pair_id = self._load(pair.encoder).checkpoint_pair_id
        main_pair_id = self._load(pair.main).checkpoint_pair_id
        _expect(
            (encoder_pair_id is None) == (main_pair_id is None),
            f"{pair.label} checkpoint pair ID is present on only one artifact",
        )
        if encoder_pair_id is not None:
            _expect(
                bool(encoder_pair_id) and encoder_pair_id == main_pair_id,
                f"{pair.label} encoder/main checkpoint pair IDs do not match",
            )


def _pair_from_record(
    inspector: CheckpointInspector, record: dict[str, Any], label: str
) -> CheckpointPair:
    return inspector.pair(
        label,
        str(record.get("encoder_path") or ""),
        str(record.get("main_path") or ""),
        str(record.get("encoder_sha256") or ""),
        str(record.get("main_sha256") or ""),
    )


def _participant_label(participant: dict[str, Any]) -> str:
    return str(
        participant.get("neutral_id")
        or participant.get("assignment_id")
        or participant["id"]
    )


def _single_row(
    rows: list[dict[str, Any]], label: str, *, required: bool = True
) -> dict[str, Any] | None:
    if not rows and not required:
        return None
    _expect(len(rows) == 1, f"{label} must have exactly one row; found {len(rows)}")
    return rows[0]


def audit_model_divergence(
    data_dir: Path | str,
    run_slug: str,
    *,
    min_l2: float = 0.0,
) -> dict[str, Any]:
    """Audit one immutable run without mutating its database or checkpoint files."""

    _expect(bool(run_slug.strip()), "run slug is required")
    _expect(math.isfinite(min_l2) and min_l2 >= 0, "min_l2 must be finite and non-negative")
    root = Path(data_dir).resolve()
    _expect(root.is_dir(), f"data directory is missing: {root}")
    database = root / "study.sqlite3"
    connection = _open_database_read_only(database)
    inspector = CheckpointInspector()
    try:
        _require_schema(connection)
        run = _single_row(
            _rows(connection, "SELECT * FROM study_runs WHERE slug=?", (run_slug,)),
            f"study run {run_slug}",
        )
        assert run is not None
        run_id = str(run["id"])
        protocol = _json_object(str(run["protocol_json"]), "run protocol_json")
        pipeline = protocol.get("pipeline")
        _expect(isinstance(pipeline, dict), "run protocol is missing pipeline rules")
        matched_budgets = pipeline.get("matched_other_budget_by_day")
        _expect(
            isinstance(matched_budgets, dict) and matched_budgets,
            "run protocol is missing matched_other_budget_by_day",
        )
        try:
            expected_matched_budgets = {
                int(day): int(budget) for day, budget in matched_budgets.items()
            }
        except (TypeError, ValueError) as exc:
            raise AuditFailure("matched_other_budget_by_day is invalid") from exc
        session_design = protocol.get("session_design")
        total_sessions = (
            int(session_design.get("total_sessions") or 3)
            if isinstance(session_design, dict)
            else 3
        )
        expected_days = list(range(1, total_sessions + 1))
        _expect(
            sorted(expected_matched_budgets) == expected_days,
            "matched_other_budget_by_day must define exactly the protocol session days",
        )
        initialization = protocol.get("initialization")
        _expect(isinstance(initialization, dict), "run protocol is missing initialization")
        personal_rules = initialization.get("personal_calibration")
        _expect(isinstance(personal_rules, dict), "run protocol is missing personal_calibration")
        personal_calibration_required = bool(personal_rules.get("enabled"))

        priors = _rows(connection, "SELECT * FROM shared_priors WHERE run_id=?", (run_id,))
        base_record = _single_row(
            [
                row
                for row in priors
                if int(row["donor_count"]) == 0 and int(row["opportunity_budget"]) == 0
            ],
            "locked M0 prior",
        )
        selected_record = _single_row(
            [row for row in priors if str(row["status"]) == "selected"],
            "selected shared prior",
        )
        assert base_record is not None and selected_record is not None
        _expect(
            str(base_record["id"]) != str(selected_record["id"]),
            "selected shared prior cannot be the M0 prior",
        )
        base_pair = _pair_from_record(inspector, base_record, "locked_m0")
        selected_pair = _pair_from_record(inspector, selected_record, "selected_shared")
        shared_vs_base = inspector.compare_pairs(base_pair, selected_pair)
        _expect(
            shared_vs_base["combined_l2_distance"] > min_l2,
            "selected shared prior must differ from M0 in trainable floating tensors",
        )

        participants = _rows(
            connection,
            "SELECT * FROM participants WHERE run_id=? ORDER BY enrollment_wave, assignment_id, created_ms",
            (run_id,),
        )
        participant_ids = {str(row["id"]) for row in participants}
        released = [
            row
            for row in participants
            if str(row.get("release_state") or "").lower() in {"released", "completed"}
        ]
        personalized = [
            row
            for row in released
            if str(row["group_name"]).lower() in PERSONALIZED_GROUPS
        ]
        frozen = [
            row for row in released if str(row["group_name"]).lower() in FROZEN_GROUPS
        ]
        _expect(personalized, "run has no released/completed personalized participants")
        _expect(frozen, "run has no released/completed frozen participants")

        snapshots = _rows(
            connection,
            "SELECT rowid AS _rowid, * FROM model_snapshots WHERE run_id=? ORDER BY created_ms, rowid",
            (run_id,),
        )
        calibrations = {
            str(row["participant_id"]): row
            for row in _rows(
                connection,
                "SELECT * FROM participant_calibrations WHERE run_id=?",
                (run_id,),
            )
        }
        counters = {
            str(row["participant_id"]): row
            for row in _rows(
                connection,
                "SELECT * FROM model_counters WHERE run_id=?",
                (run_id,),
            )
        }

        personalized_reports: list[dict[str, Any]] = []
        personalized_pairs_by_budget: dict[str, dict[int, CheckpointPair]] = {}
        for participant in personalized:
            participant_id = str(participant["id"])
            label = _participant_label(participant)
            _expect(
                str(participant.get("prior_id") or "") == str(selected_record["id"]),
                f"personalized participant {label} is not bound to the selected shared prior",
            )
            participant_snapshots = [
                row
                for row in snapshots
                if str(row["participant_id"]) == participant_id
                and str(row["source"]) == "personalized"
            ]
            _expect(
                participant_snapshots,
                f"personalized participant {label} has no persisted personalized snapshot",
            )
            _expect(
                all(
                    str(row.get("base_prior_id") or "") == str(selected_record["id"])
                    for row in participant_snapshots
                ),
                f"personalized participant {label} snapshots are not bound to the selected prior",
            )
            snapshot_pairs = [
                _pair_from_record(
                    inspector,
                    row,
                    f"personalized_snapshot:{label}:q{row['opportunity_budget']}",
                )
                for row in participant_snapshots
            ]
            snapshots_by_budget = {
                int(row["opportunity_budget"]): pair
                for row, pair in zip(participant_snapshots, snapshot_pairs, strict=True)
            }
            for required_budget in sorted(set(expected_matched_budgets.values())):
                _expect(
                    required_budget in snapshots_by_budget,
                    f"personalized participant {label} is missing the q={required_budget} "
                    "snapshot required for matched comparison",
                )
            latest_snapshot_row = max(
                participant_snapshots,
                key=lambda row: (int(row["created_ms"]), int(row["_rowid"])),
            )
            latest_snapshot_index = participant_snapshots.index(latest_snapshot_row)
            latest_snapshot_pair = snapshot_pairs[latest_snapshot_index]
            model_dir = root / "models" / participant_id
            latest_pair = inspector.pair(
                f"personalized_latest:{label}",
                model_dir / "latest_encoder.pth",
                model_dir / "latest_main_model.pth",
            )
            latest_vs_prior = inspector.compare_pairs(selected_pair, latest_pair)
            snapshot_vs_prior = inspector.compare_pairs(selected_pair, latest_snapshot_pair)
            _expect(
                latest_vs_prior["combined_l2_distance"] > min_l2,
                f"personalized participant {label} latest checkpoint still equals its prior",
            )
            _expect(
                snapshot_vs_prior["combined_l2_distance"] > min_l2,
                f"personalized participant {label} persisted snapshot still equals its prior",
            )

            calibration = calibrations.get(participant_id)
            if personal_calibration_required:
                _expect(
                    calibration is not None,
                    f"personalized participant {label} has no persisted calibration state",
                )
                assert calibration is not None
                _expect(
                    int(calibration["update_count"]) > 0,
                    f"personalized participant {label} has no applied calibration updates",
                )
                _expect(
                    int(calibration["positive_updates"])
                    + int(calibration["negative_updates"])
                    == int(calibration["update_count"]),
                    f"personalized participant {label} calibration counters are inconsistent",
                )
            counter = counters.get(participant_id)
            _expect(
                counter is not None and int(counter["eligible_feedbacks"]) > 0,
                f"personalized participant {label} has no applied eligible feedback",
            )
            assert counter is not None
            _expect(
                int(counter["positive_feedbacks"]) + int(counter["negative_feedbacks"])
                == int(counter["eligible_feedbacks"]),
                f"personalized participant {label} feedback counters are inconsistent",
            )
            if personal_calibration_required:
                assert calibration is not None
                _expect(
                    int(calibration["update_count"]) == int(counter["eligible_feedbacks"]),
                    f"personalized participant {label} calibration and feedback counts differ",
                )
            personalized_pairs_by_budget[participant_id] = snapshots_by_budget
            personalized_reports.append(
                {
                    "participant_id": participant_id,
                    "label": label,
                    "cohort": participant["cohort"],
                    "prior_id": participant["prior_id"],
                    "latest_checkpoint": latest_pair.report(),
                    "latest_snapshot": latest_snapshot_pair.report(),
                    "personalized_snapshot_count": len(participant_snapshots),
                    "latest_vs_prior": latest_vs_prior,
                    "latest_snapshot_vs_prior": snapshot_vs_prior,
                    "latest_vs_latest_snapshot": inspector.compare_pairs(
                        latest_snapshot_pair, latest_pair
                    ),
                    "calibration": dict(calibration) if calibration is not None else None,
                    "model_counters": dict(counter),
                }
            )

        frozen_reports: list[dict[str, Any]] = []
        frozen_effective_pairs: dict[str, CheckpointPair] = {}
        for participant in frozen:
            participant_id = str(participant["id"])
            label = _participant_label(participant)
            _expect(
                str(participant.get("prior_id") or "") == str(selected_record["id"]),
                f"frozen participant {label} is not bound to the selected shared prior",
            )
            participant_snapshots = [
                row
                for row in snapshots
                if str(row["participant_id"]) == participant_id
                and str(row["source"]) == "personalized"
            ]
            _expect(
                not participant_snapshots,
                f"frozen participant {label} has forbidden personalized snapshots",
            )
            calibration = calibrations.get(participant_id)
            if calibration is not None:
                _expect(
                    int(calibration["update_count"]) == 0
                    and int(calibration["positive_updates"]) == 0
                    and int(calibration["negative_updates"]) == 0,
                    f"frozen participant {label} has forbidden personal calibration updates",
                )
            counter = counters.get(participant_id)
            if counter is not None:
                _expect(
                    int(counter["eligible_feedbacks"]) == 0
                    and int(counter["positive_feedbacks"]) == 0
                    and int(counter["negative_feedbacks"]) == 0,
                    f"frozen participant {label} has forbidden applied feedback updates",
                )

            event_rows = _rows(
                connection,
                "SELECT event_type, payload FROM events WHERE participant_id=? "
                "AND event_type IN ('model.ready','tap.manual_activate','tap.interrupt') "
                "ORDER BY seq",
                (participant_id,),
            )
            ready_fingerprints: list[dict[str, Any]] = []
            forbidden_training_events = 0
            for event in event_rows:
                payload = _json_object(
                    str(event["payload"]),
                    f"event payload for frozen participant {label}",
                )
                if event["event_type"] == "model.ready":
                    fingerprint = payload.get("model_fingerprint")
                    _expect(
                        isinstance(fingerprint, dict),
                        f"frozen participant {label} has malformed model.ready fingerprint",
                    )
                    ready_fingerprints.append(fingerprint)
                elif payload.get("training_applied") is True:
                    forbidden_training_events += 1
            _expect(
                ready_fingerprints,
                f"frozen participant {label} has no model.ready evidence",
            )
            expected_fingerprint = {
                "encoder_sha256": selected_pair.encoder.sha256,
                "main_sha256": selected_pair.main.sha256,
            }
            _expect(
                all(fingerprint == expected_fingerprint for fingerprint in ready_fingerprints),
                f"frozen participant {label} did not remain on the locked prior",
            )
            _expect(
                forbidden_training_events == 0,
                f"frozen participant {label} has training_applied tap events",
            )

            model_dir = root / "models" / participant_id
            encoder_path = model_dir / "latest_encoder.pth"
            main_path = model_dir / "latest_main_model.pth"
            _expect(
                encoder_path.exists() == main_path.exists(),
                f"frozen participant {label} has a partial personal checkpoint pair",
            )
            stray_pair: CheckpointPair | None = None
            stray_vs_prior: dict[str, Any] | None = None
            if encoder_path.exists():
                stray_pair = inspector.pair(
                    f"frozen_stray_latest:{label}", encoder_path, main_path
                )
                stray_vs_prior = inspector.compare_pairs(selected_pair, stray_pair)
                _expect(
                    stray_vs_prior["exact_tensor_equal"],
                    f"frozen participant {label} personal checkpoint differs from prior",
                )
            frozen_effective_pairs[participant_id] = selected_pair
            frozen_reports.append(
                {
                    "participant_id": participant_id,
                    "label": label,
                    "cohort": participant["cohort"],
                    "prior_id": participant["prior_id"],
                    "effective_checkpoint_source": "locked_selected_prior",
                    "effective_checkpoint": selected_pair.report(),
                    "stray_personal_checkpoint": stray_pair.report() if stray_pair else None,
                    "stray_personal_checkpoint_vs_prior": stray_vs_prior,
                    "personalized_snapshot_count": 0,
                    "calibration": dict(calibration) if calibration is not None else None,
                    "model_counters": dict(counter) if counter is not None else None,
                    "model_ready_event_count": len(ready_fingerprints),
                    "model_ready_fingerprints_match_prior": True,
                    "training_applied_event_count": 0,
                }
            )

        manifest_path = root / "matched" / run_id / "manifest.json"
        sentinel_path = root / "matched" / run_id / "self-test.json"
        _expect(manifest_path.is_file(), f"matched manifest is missing: {manifest_path}")
        _expect(sentinel_path.is_file(), f"matched self-test is missing: {sentinel_path}")
        manifest = _read_json(manifest_path, "matched manifest")
        sentinel = _read_json(sentinel_path, "matched self-test")
        manifest_sha = file_sha256(manifest_path)
        _expect(manifest.get("approved") is True, "matched manifest is not approved")
        _expect(str(manifest.get("run_id") or "") == run_id, "matched manifest run_id mismatch")
        _expect(
            str(manifest.get("run_slug") or "") == run_slug,
            "matched manifest run_slug mismatch",
        )
        _expect(bool(manifest.get("assignment_rule")), "matched assignment rule is missing")
        _expect(int(manifest.get("frozen_at_ms") or 0) > 0, "matched manifest is not frozen")
        _expect(sentinel.get("passed") is True, "matched self-test did not pass")
        _expect(
            str(sentinel.get("manifest_sha256") or "") == manifest_sha,
            "matched self-test manifest SHA-256 mismatch",
        )
        assignments = manifest.get("assignments")
        tested_assignments = sentinel.get("assignments")
        _expect(isinstance(assignments, dict), "matched manifest assignments are missing")
        _expect(
            isinstance(tested_assignments, dict),
            "matched self-test assignments are missing",
        )

        matched_reports: list[dict[str, Any]] = []
        target_participants = personalized + frozen
        for participant in target_participants:
            participant_id = str(participant["id"])
            label = _participant_label(participant)
            participant_assignments = assignments.get(participant_id)
            tested_participant = tested_assignments.get(participant_id)
            _expect(
                isinstance(participant_assignments, dict),
                f"matched assignments are missing for {label}",
            )
            _expect(
                isinstance(tested_participant, dict),
                f"matched self-test assignments are missing for {label}",
            )
            for day, expected_budget in sorted(expected_matched_budgets.items()):
                assignment = participant_assignments.get(str(day))
                tested = tested_participant.get(str(day))
                _expect(
                    isinstance(assignment, dict),
                    f"matched assignment is missing for {label} day {day}",
                )
                _expect(
                    isinstance(tested, dict),
                    f"matched self-test entry is missing for {label} day {day}",
                )
                donor_id = str(assignment.get("donor_participant_id") or "")
                _expect(
                    donor_id in participant_ids and donor_id != participant_id,
                    f"matched donor is invalid for {label} day {day}",
                )
                strata = assignment.get("matching_strata")
                _expect(
                    isinstance(strata, dict),
                    f"matched strata are missing for {label} day {day}",
                )
                _expect(
                    str(strata.get("shared_prior_id") or "") == str(selected_record["id"]),
                    f"matched shared prior mismatch for {label} day {day}",
                )
                _expect(
                    int(strata.get("opportunity_budget") or -1) == expected_budget,
                    f"matched budget mismatch for {label} day {day}",
                )
                matched_pair = inspector.pair(
                    f"matched:{label}:day{day}",
                    str(assignment.get("encoder_checkpoint") or ""),
                    str(assignment.get("main_checkpoint") or ""),
                    str(assignment.get("encoder_sha256") or ""),
                    str(assignment.get("main_sha256") or ""),
                )
                _expect(
                    tested.get("donor_participant_id") == donor_id
                    and tested.get("encoder_sha256") == matched_pair.encoder.sha256
                    and tested.get("main_sha256") == matched_pair.main.sha256,
                    f"matched self-test entry does not bind the artifact for {label} day {day}",
                )
                target_group = str(participant["group_name"]).lower()
                target_equal_budget_pair = (
                    personalized_pairs_by_budget[participant_id][expected_budget]
                    if target_group in PERSONALIZED_GROUPS
                    else frozen_effective_pairs[participant_id]
                )
                matched_reports.append(
                    {
                        "participant_id": participant_id,
                        "label": label,
                        "group": participant["group_name"],
                        "day": day,
                        "expected_budget": expected_budget,
                        "donor_participant_id": donor_id,
                        "checkpoint": matched_pair.report(),
                        "matched_vs_selected_shared": inspector.compare_pairs(
                            selected_pair, matched_pair
                        ),
                        "matched_vs_target_equal_budget_checkpoint": inspector.compare_pairs(
                            target_equal_budget_pair, matched_pair
                        ),
                    }
                )

        report = {
            "schema": SCHEMA,
            "passed": True,
            "read_only": True,
            "data_dir": str(root),
            "database": str(database),
            "run": {
                "id": run_id,
                "slug": run_slug,
                "data_origin": run.get("data_origin"),
                "protocol_version": run.get("protocol_version"),
                "protocol_sha256": run.get("protocol_sha256"),
            },
            "thresholds": {"minimum_required_l2_distance": min_l2},
            "base": {"prior_id": base_record["id"], "checkpoint": base_pair.report()},
            "shared": {
                "prior_id": selected_record["id"],
                "donor_count": selected_record["donor_count"],
                "opportunity_budget": selected_record["opportunity_budget"],
                "checkpoint": selected_pair.report(),
                "selected_shared_vs_m0": shared_vs_base,
            },
            "participants": {
                "scope": "released_or_completed_personalized_and_frozen",
                "personalized": personalized_reports,
                "frozen": frozen_reports,
            },
            "matched": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "self_test_path": str(sentinel_path),
                "assignment_count": len(matched_reports),
                "assignments": matched_reports,
            },
            "assertions": {
                "all_recorded_hashes_match": True,
                "shared_differs_from_m0": True,
                "all_personalized_checkpoints_differ_from_prior": True,
                "all_frozen_have_no_personal_updates": True,
                "all_frozen_effective_checkpoints_equal_prior": True,
                "all_matched_assignments_present_and_hash_bound": True,
            },
        }
        return report
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, fail-closed audit of M0/shared/personalized/frozen/matched "
            "checkpoint divergence for one study run."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-slug", required=True)
    parser.add_argument(
        "--min-l2",
        type=float,
        default=0.0,
        help="Require shared and personalized combined trainable-float L2 to exceed this value.",
    )
    args = parser.parse_args()
    try:
        report = audit_model_divergence(
            args.data_dir,
            args.run_slug,
            min_l2=args.min_l2,
        )
    except AuditFailure as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "passed": False,
                    "read_only": True,
                    "data_dir": str(args.data_dir.resolve()),
                    "run_slug": args.run_slug,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
