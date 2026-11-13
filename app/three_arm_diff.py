"""Read-only three-arm difference board for admin.

Computes Kruskal-Wallis and Holm-adjusted pairwise Mann-Whitney tests
from already-loaded participant scores. Never writes study data.
"""

from __future__ import annotations

import math
from typing import Any


CONTRAST_ARMS = ("cold_learn", "shared_learn", "shared_frozen")
ARM_SHORT = {
    "cold_learn": "①",
    "shared_learn": "②",
    "shared_frozen": "③",
}
QUESTIONNAIRE_DIMENSION_LABELS = {
    "predictability": "Predictability 可预测性",
    "perceived_control": "Perceived control 掌控感",
    "correctability": "Correctability 可纠错性",
    "legibility": "Legibility 变化可理解性",
    "behavioral_adaptation": "Behavioral adaptation 低行为迁就",
    "response_accuracy": "Response accuracy 时机符合度",
    "cognitive_demand": "Cognitive demand 低认知负担",
    "annoyance": "Annoyance 低烦扰",
    "habitability": "Habitability 可习得性",
    "speed": "Speed 交互效率",
    "likeability": "Likeability 好感",
}
QUESTIONNAIRE_FACETS = ("trend", "overall", "construct")
QUESTIONNAIRE_FACET_LABELS = {
    "trend": "向好性",
    "overall": "整体性",
    "construct": "构念题",
}
QUESTIONNAIRE_FACET_DIMENSIONS = {
    "trend": (
        "response_accuracy",
        "likeability",
        "cognitive_demand",
        "annoyance",
        "habitability",
        "perceived_control",
        "speed",
    ),
    "overall": (
        "response_accuracy",
        "likeability",
        "cognitive_demand",
        "annoyance",
        "habitability",
        "speed",
    ),
    "construct": (
        "predictability",
        "perceived_control",
        "correctability",
        "legibility",
        "behavioral_adaptation",
    ),
}


def questionnaire_facet(item_id: str) -> str:
    text = str(item_id or "")
    if text.startswith("trend_"):
        return "trend"
    if text.startswith("overall_"):
        return "overall"
    return "construct"


def _normalize_questionnaire_groups(
    questionnaire_groups: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    empty = {facet: {} for facet in QUESTIONNAIRE_FACETS}
    if not questionnaire_groups:
        return empty
    keys = set(questionnaire_groups.keys())
    if keys <= set(QUESTIONNAIRE_FACETS):
        return {
            facet: dict(questionnaire_groups.get(facet) or {})
            for facet in QUESTIONNAIRE_FACETS
        }
    empty["construct"] = dict(questionnaire_groups)
    return empty


def significance_label(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "—"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[index]]
        ):
            end += 1
        average = (index + end + 2) / 2.0
        for cursor in range(index, end + 1):
            ranks[order[cursor]] = average
        index = end + 1
    return ranks


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(max(0.0, z) / math.sqrt(2.0))


def _chi2_sf(stat: float, df: int) -> float:
    if stat <= 0:
        return 1.0
    if df <= 1:
        return 2.0 * _normal_sf(math.sqrt(stat))
    if df == 2:
        return math.exp(-stat / 2.0)
    # Regularized upper gamma for small integer df via recurrence.
    x = stat / 2.0
    term = math.exp(-x)
    total = term
    for k in range(1, df // 2):
        term *= x / k
        total += term
    return min(1.0, max(0.0, total))


def kruskal_wallis(groups: list[list[float]]) -> dict[str, Any]:
    present = [list(group) for group in groups if len(group) >= 1]
    usable = [group for group in present if len(group) >= 2]
    if len(usable) < 2 or sum(len(group) for group in present) < 4:
        return {
            "H": None,
            "df": max(0, len(present) - 1),
            "p": None,
            "sig": "—",
            "note": "n不足",
        }
    values = [value for group in present for value in group]
    ranks = _average_ranks(values)
    offset = 0
    rank_sums: list[tuple[int, float]] = []
    for group in present:
        size = len(group)
        rank_sums.append((size, sum(ranks[offset : offset + size])))
        offset += size
    n = len(values)
    h = (12.0 / (n * (n + 1))) * sum(
        (rank_sum ** 2) / size for size, rank_sum in rank_sums
    ) - 3.0 * (n + 1)
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    tie = sum(count ** 3 - count for count in counts.values() if count > 1)
    denom = n ** 3 - n
    if tie and denom:
        h /= 1.0 - (tie / denom)
    df = len(present) - 1
    p = _chi2_sf(max(0.0, h), df)
    return {
        "H": round(h, 4),
        "df": df,
        "p": round(p, 4),
        "sig": significance_label(p),
        "note": None,
    }


def mann_whitney(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) < 2 or len(right) < 2:
        return {"U": None, "p": None, "sig": "—", "note": "n不足"}
    values = list(left) + list(right)
    ranks = _average_ranks(values)
    n1 = len(left)
    n2 = len(right)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    n = n1 + n2
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    tie = sum(count ** 3 - count for count in counts.values() if count > 1)
    variance = n1 * n2 * (n + 1) / 12.0
    if tie and n > 1:
        variance -= n1 * n2 * tie / (12.0 * n * (n - 1))
    if variance <= 0:
        return {"U": round(u, 3), "p": 1.0, "sig": "ns", "note": None}
    z = abs(u - n1 * n2 / 2.0) / math.sqrt(variance)
    p = min(1.0, 2.0 * _normal_sf(z))
    return {"U": round(u, 3), "p": round(p, 4), "sig": significance_label(p), "note": None}


def holm_adjust(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = [
        (index, row)
        for index, row in enumerate(rows)
        if isinstance(row.get("p"), (int, float))
    ]
    indexed.sort(key=lambda item: item[1]["p"])
    m = len(indexed)
    adjusted = [None] * len(rows)
    running = 0.0
    for rank, (index, row) in enumerate(indexed, start=1):
        value = min(1.0, row["p"] * (m - rank + 1))
        running = max(running, value)
        adjusted[index] = round(running, 4)
    result = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["p_holm"] = adjusted[index]
        item["sig"] = significance_label(adjusted[index])
        result.append(item)
    return result


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _as_rates(items: list[Any]) -> list[float]:
    rates: list[float] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("rate")
            if isinstance(value, (int, float)):
                rates.append(float(value))
        elif isinstance(item, (int, float)):
            rates.append(float(item))
    return rates


def _sum_int(items: list[Any], key: str) -> int | None:
    found = False
    total = 0
    for item in items:
        if isinstance(item, dict) and key in item:
            found = True
            total += int(item.get(key) or 0)
    return total if found else None


def _format_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _format_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _group_cells(
    groups: dict[str, list[Any]],
    *,
    formatter,
    include_counts: bool = False,
) -> dict[str, Any]:
    cells = {}
    for arm in CONTRAST_ARMS:
        items = list(groups.get(arm) or [])
        rates = _as_rates(items)
        mean = _mean(rates)
        cell = {
            "n": len(rates),
            "mean": None if mean is None else round(mean, 6),
            "mean_label": formatter(mean),
        }
        if include_counts:
            opportunities = _sum_int(items, "opportunities")
            behavior = _sum_int(items, "behavior_corrections")
            eligible = _sum_int(items, "eligible_corrections")
            raw_taps = _sum_int(items, "raw_taps")
            mistouches = _sum_int(items, "mistouches")
            training_rates = [
                float(item["training_rate"])
                for item in items
                if isinstance(item, dict)
                and isinstance(item.get("training_rate"), (int, float))
            ]
            training_mean = _mean(training_rates)
            cell.update(
                {
                    "opportunities": opportunities,
                    "behavior_corrections": behavior,
                    "eligible_corrections": eligible,
                    "raw_taps": raw_taps,
                    "mistouches": mistouches,
                    "training_mean": (
                        None if training_mean is None else round(training_mean, 6)
                    ),
                    "count_label": (
                        "—"
                        if opportunities is None
                        else (
                            f"机会 {opportunities} · 纠错 {behavior} · "
                            f"实际tap {raw_taps} · 训练合格 {eligible}"
                            + (
                                f" · 误触 {mistouches}"
                                if mistouches
                                else ""
                            )
                        )
                    ),
                    "training_label": (
                        "训练口径 —"
                        if training_mean is None
                        else f"训练口径 {_format_rate(training_mean)}"
                    ),
                }
            )
        cells[arm] = cell
    return cells


def _compare_groups(
    groups: dict[str, list[Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = [_as_rates(groups.get(arm) or []) for arm in CONTRAST_ARMS]
    omnibus = kruskal_wallis(ordered)
    pairs = []
    for left_index, left_arm in enumerate(CONTRAST_ARMS):
        for right_arm in CONTRAST_ARMS[left_index + 1 :]:
            left = _as_rates(groups.get(left_arm) or [])
            right = _as_rates(groups.get(right_arm) or [])
            test = mann_whitney(left, right)
            left_mean = _mean(left)
            right_mean = _mean(right)
            if left_mean is None or right_mean is None:
                direction = "—"
            elif left_mean > right_mean:
                direction = f"{ARM_SHORT[left_arm]}>{ARM_SHORT[right_arm]}"
            elif left_mean < right_mean:
                direction = f"{ARM_SHORT[left_arm]}<{ARM_SHORT[right_arm]}"
            else:
                direction = f"{ARM_SHORT[left_arm]}={ARM_SHORT[right_arm]}"
            pairs.append(
                {
                    "a": left_arm,
                    "b": right_arm,
                    "label": f"{ARM_SHORT[left_arm]}–{ARM_SHORT[right_arm]}",
                    "direction": direction,
                    "U": test["U"],
                    "p": test["p"],
                    "note": test["note"],
                }
            )
    return omnibus, holm_adjust(pairs)


def build_three_arm_difference_board(
    *,
    block_groups: dict[int, dict[str, list[Any]]],
    questionnaire_groups: dict[str, dict[str, list[Any]]],
    arm_labels: dict[str, str],
) -> dict[str, Any]:
    blocks = []
    for block in range(1, 6):
        groups = block_groups.get(block) or {arm: [] for arm in CONTRAST_ARMS}
        omnibus, pairwise = _compare_groups(groups)
        blocks.append(
            {
                "block": block,
                "metric": "participant_mean_behavior_correction_rate",
                "label": f"Block {block} 参与者均值纠错率",
                "groups": _group_cells(
                    groups, formatter=_format_rate, include_counts=True
                ),
                "omnibus": omnibus,
                "pairwise": pairwise,
            }
        )

    grouped = _normalize_questionnaire_groups(questionnaire_groups)
    questionnaire_by_facet: dict[str, list[dict[str, Any]]] = {}
    for facet in QUESTIONNAIRE_FACETS:
        rows = []
        for dimension in QUESTIONNAIRE_FACET_DIMENSIONS[facet]:
            groups = grouped[facet].get(dimension) or {
                arm: [] for arm in CONTRAST_ARMS
            }
            omnibus, pairwise = _compare_groups(groups)
            rows.append(
                {
                    "facet": facet,
                    "dimension": dimension,
                    "label": QUESTIONNAIRE_DIMENSION_LABELS[dimension],
                    "groups": _group_cells(groups, formatter=_format_score),
                    "omnibus": omnibus,
                    "pairwise": pairwise,
                }
            )
        questionnaire_by_facet[facet] = rows
    questionnaire = [
        row
        for facet in QUESTIONNAIRE_FACETS
        for row in questionnaire_by_facet[facet]
    ]

    return {
        "readonly": True,
        "writes_data": False,
        "method": (
            "Kruskal-Wallis 三组总检验；成对 Mann-Whitney，Holm 校正。"
            "只纳入端到端完成的三臂受试者。块表比较行为纠错率："
            "训练合格 tap，或仅因冻结臂不训练（activation_not_learning_current）"
            "的 tap，除以 eligible 机会。格子同时给出机会 / 纠错次数 / 实际 tap / "
            "训练合格次数。问卷分向好性 / 整体性 / 构念题三表，"
            "不把 trend 与 overall 合成同一维度。1–7，反向题已反向。"
        ),
        "alpha": 0.05,
        "arms": [
            {"arm": arm, "short": ARM_SHORT[arm], "label": arm_labels.get(arm, arm)}
            for arm in CONTRAST_ARMS
        ],
        "blocks": blocks,
        "questionnaire_trend": questionnaire_by_facet["trend"],
        "questionnaire_overall": questionnaire_by_facet["overall"],
        "questionnaire_construct": questionnaire_by_facet["construct"],
        "questionnaire": questionnaire,
    }
