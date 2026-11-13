"""Regression tests for study analysis.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

from app.study_analysis import (
    _bootstrap_change_decomposition,
    _participant_day_changes,
)


def test_longitudinal_decomposition_uses_participant_level_changes() -> None:
    rows = [
        {"participant_id": "p1", "group": "personalized", "day": 1, "metric": 3.0},
        {"participant_id": "p1", "group": "personalized", "day": 3, "metric": 1.0},
        {"participant_id": "p2", "group": "personalized", "day": 1, "metric": 4.0},
        {"participant_id": "p2", "group": "personalized", "day": 3, "metric": 3.0},
        {"participant_id": "f1", "group": "frozen", "day": 1, "metric": 3.0},
        {"participant_id": "f1", "group": "frozen", "day": 3, "metric": 2.0},
        {"participant_id": "f2", "group": "frozen", "day": 1, "metric": 4.0},
        {"participant_id": "f2", "group": "frozen", "day": 3, "metric": 4.0},
    ]
    personalized = _participant_day_changes(rows, "personalized", "metric")
    frozen = _participant_day_changes(rows, "frozen", "metric")
    result = _bootstrap_change_decomposition(
        personalized, frozen, replicates=1_000, seed=17
    )

    assert personalized == {"p1": 2.0, "p2": 1.0}
    assert frozen == {"f1": 1.0, "f2": 0.0}
    assert result["personalized_participants"] == 2
    assert result["frozen_participants"] == 2
    assert result["personalized_day1_minus_day3"] == 1.5
    assert result["frozen_day1_minus_day3_proxy"] == 0.5
    assert result["difference_in_differences"] == 1.0


def test_longitudinal_decomposition_fails_closed_without_both_groups() -> None:
    result = _bootstrap_change_decomposition({"p1": 1.0}, {}, replicates=10)

    assert result["personalized_participants"] == 1
    assert result["frozen_participants"] == 0
    assert result["difference_in_differences"] is None
