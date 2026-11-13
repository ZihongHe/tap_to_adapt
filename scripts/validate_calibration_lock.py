"""Validate the locked calibration mechanics without generating labels.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def validate(study: dict[str, Any]) -> dict[str, Any]:
    initialization = study["initialization"]
    evidence = initialization["engineering_calibration_evidence"]
    personal = initialization["personal_calibration"]
    temperature = float(initialization["temperature"])
    global_bias = float(initialization["logit_bias"])
    threshold = float(initialization["activation_threshold"])
    cold_raw = float(evidence["cold_start_learning_raw_logit_median"])
    frozen_raw = float(evidence["frozen_raw_logit_reference"])
    cold_probability = sigmoid(cold_raw / temperature + global_bias)
    frozen_probability = sigmoid(frozen_raw / temperature + global_bias)

    schedule = initialization.get("network_update_schedule") or []
    limits = [int(item["eligible_feedbacks_lt"]) for item in schedule]
    scales = [float(item["scale"]) for item in schedule]
    learning_rate = float(personal["learning_rate"])
    lower = float(personal["minimum_logit_bias"])
    upper = float(personal["maximum_logit_bias"])
    positive_step = learning_rate * (1.0 - cold_probability)
    negative_step = learning_rate * (0.0 - frozen_probability)

    checks = {
        "protocol_is_v10_or_later": any(
            marker in str(study["version"]) for marker in ("-v10", "-v11")
        ),
        "temperature_is_positive": temperature > 0,
        "threshold_is_valid": 0 < threshold < 1,
        "cold_mode_is_near_threshold": 0.4 <= cold_probability <= 0.55,
        "overconfident_mode_is_compressed": 0.7 <= frozen_probability <= 0.9,
        "network_limits_are_increasing": limits == sorted(set(limits)),
        "network_scales_are_increasing": scales == sorted(scales)
        and all(0 <= value < 1 for value in scales),
        "first_feedbacks_are_intercept_only": bool(scales) and scales[0] == 0,
        "network_scale_controls_optimizer_learning_rate": "optimizer learning rate"
        in str(initialization.get("network_update_control") or "").lower(),
        "network_scale_interpolates_parameter_and_buffer_delta": "buffer delta"
        in str(initialization.get("network_update_control") or "").lower(),
        "positive_feedback_raises_intercept": positive_step > 0,
        "negative_feedback_lowers_intercept": negative_step < 0,
        "personal_intercept_is_bounded": lower < 0 < upper,
        "personal_intercept_excluded_from_shared_prior": personal.get(
            "shared_prior_inclusion"
        )
        is False,
        "runtime_label_quota_is_disabled": evidence.get("runtime_label_quota")
        is False,
        "engineering_evidence_not_formal": evidence.get("formal_use") is False,
    }
    return {
        "passed": all(checks.values()),
        "protocol_version": study["version"],
        "calibration_version": initialization["version"],
        "checks": checks,
        "derived": {
            "cold_start_reference_probability": cold_probability,
            "frozen_reference_probability": frozen_probability,
            "first_positive_intercept_step": positive_step,
            "first_negative_intercept_step": negative_step,
            "network_update_schedule": schedule,
            "personal_logit_bias_bounds": [lower, upper],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the locked calibration mechanics without generating labels."
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "content" / "study_protocol.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.protocol.read_text(encoding="utf-8")))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
