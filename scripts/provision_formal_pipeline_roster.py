"""Provision a held 36A+36B formal roster.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block(
    db: Any,
    *,
    run_slug: str,
    count: int,
    cohort: str,
    group: str,
    prefix: str,
    assignment_start: int,
    wave: int,
) -> list[dict[str, Any]]:
    codes = db.create_participants(
        count,
        cohort,
        group,
        data_origin="formal",
        protocol_mode="strict",
        run_slug=run_slug,
    )
    rows = []
    for offset, code in enumerate(codes):
        index = assignment_start + offset
        participant = db.participant_by_code(code)
        if participant is None:
            raise RuntimeError("Newly created formal participant could not be reloaded")
        rows.append(
            {
                "assignment_id": f"{prefix}{index:02d}",
                "participant_code": code,
                "participant_id": participant["id"],
                "cohort": cohort,
                "internal_group": group,
                "enrollment_wave": wave,
                "probe_schedule": "/".join(json.loads(participant["probe_schedule"])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision a held 36A+36B formal roster.")
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-formal-roster", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_formal_roster:
        raise SystemExit("Pass --acknowledge-formal-roster after checking the locked protocol")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB
    from app.priors import create_base_prior
    from app.protocol import Protocol

    db = StudyDB(settings)
    protocol = Protocol(settings)
    if db.study_run(args.run_slug) is not None:
        raise SystemExit("Run slug already exists")
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    os.chmod(output, 0o700)

    rows: list[dict[str, Any]] = []
    rows += block(
        db,
        run_slug=args.run_slug,
        count=12,
        cohort="study_a",
        group="seed_donor",
        prefix="A",
        assignment_start=1,
        wave=1,
    )
    rows += block(
        db,
        run_slug=args.run_slug,
        count=8,
        cohort="study_a",
        group="evaluator_donor",
        prefix="A",
        assignment_start=13,
        wave=2,
    )
    rows += block(
        db,
        run_slug=args.run_slug,
        count=8,
        cohort="study_a",
        group="evaluator_donor",
        prefix="A",
        assignment_start=21,
        wave=3,
    )
    rows += block(
        db,
        run_slug=args.run_slug,
        count=8,
        cohort="study_a",
        group="holdout_evaluator",
        prefix="A",
        assignment_start=29,
        wave=4,
    )
    rows += block(
        db,
        run_slug=args.run_slug,
        count=16,
        cohort="study_b",
        group="personalized",
        prefix="BP",
        assignment_start=1,
        wave=5,
    )
    rows += block(
        db,
        run_slug=args.run_slug,
        count=2,
        cohort="study_b",
        group="personalized",
        prefix="BP",
        assignment_start=17,
        wave=6,
    )
    rows += block(
        db,
        run_slug=args.run_slug,
        count=16,
        cohort="study_b",
        group="frozen",
        prefix="BF",
        assignment_start=1,
        wave=5,
    )
    rows += block(
        db,
        run_slug=args.run_slug,
        count=2,
        cohort="study_b",
        group="frozen",
        prefix="BF",
        assignment_start=17,
        wave=6,
    )
    neutral_values = [f"P{index:03d}" for index in range(1, len(rows) + 1)]
    secrets.SystemRandom().shuffle(neutral_values)
    for row, neutral_id in zip(rows, neutral_values, strict=True):
        row["neutral_id"] = neutral_id
        db.configure_participant(
            row["participant_id"],
            assignment_id=row["assignment_id"],
            neutral_id=neutral_id,
            enrollment_wave=int(row["enrollment_wave"]),
            release_state="held",
        )

    run = db.study_run(args.run_slug)
    assert run is not None
    base = create_base_prior(db, settings, run["id"])
    for row in rows:
        if row["assignment_id"].startswith("A") and int(row["enrollment_wave"]) == 1:
            db.assign_prior(row["participant_id"], base["id"], actor="roster-provisioner")

    internal = sorted(rows, key=lambda row: row["assignment_id"])
    blind = sorted(
        [
            {
                "neutral_id": row["neutral_id"],
                "participant_code": row["participant_code"],
                "release_wave": f"W{row['enrollment_wave']}",
                "status": "等待研究协调员开放",
                "day_1": "收到个人预约通知后进入",
                "day_2": "Day 1 完成后的下一个自然日，且至少间隔 12 小时",
                "day_3": "Day 2 完成后的下一个自然日，且至少间隔 12 小时",
                "companion": "每天与电脑端使用同一编号和同一天，电脑端开始前上线",
            }
            for row in rows
        ],
        key=lambda row: (int(row["release_wave"][1:]), row["neutral_id"]),
    )
    internal_path = output / "researcher_internal_assignments.csv"
    blind_path = output / "blinded_coordinator_roster.csv"
    write_csv(internal_path, internal)
    write_csv(blind_path, blind)

    instructions = """# 参与者操作指南

## 实验安排

本研究连续进行三天。请每天使用同一台电脑、同一部手机和研究团队单独发放的参与者编号。Day 2 必须在 Day 1 完成后的下一个自然日开始，且至少间隔 12 小时；Day 3 同理。每次请预留约 50 分钟。

## 开始前

1. 在安静、可独立说话的室内使用电脑 Chrome 或 Edge，允许网页访问麦克风。
2. 手机保持联网、关闭静音，媒体音量调至约 60%，放在距本人约 50 cm 的固定位置。
3. 电脑端和手机端输入同一参与者编号并选择同一天；等待两端均显示在线。
4. 完成网页中的网络、麦克风、浏览器和点击反应检查。检测未通过时不要强行继续。
5. 第 1 天还需完成一次开始前信息与操作理解检查；三天请使用同一台电脑和同一浏览器。

## 对话与轻触

从网页话题列表中自由选择希望讨论的话题，像日常对话一样自然说话，不需要表演特定情境。当你希望电脑端智能体现在开始回答时，轻触电脑页面；如果智能体已经开始响应，而你希望立即停止当前响应，也轻触同一区域。请按当时真实感受操作，不要为了增加或减少点击次数而改变行为。

部分时段手机会播放背景声音或进行对话。保持手机端运行，并遵循电脑页面提示；不要猜测或寻找系统内部条件。遇到断网、页面关闭或设备失败时暂停并联系研究团队，不要重复点击来补数据。

## 权利与隐私

不要在对话中透露姓名、手机号、身份证号、财务、医疗或他人隐私。你可以跳过任何话题、暂停或退出。是否保存原始音频、未来共享和参与研究是相互独立的选择，以网页知情同意书为准。
"""
    instruction_path = output / "participant_instructions.zh-CN.md"
    instruction_path.write_text(instructions, encoding="utf-8")

    manifest = {
        "version": "formal-roster-36A-36B-pipeline-v7",
        "created_ms": int(time.time() * 1000),
        "run_id": run["id"],
        "run_slug": run["slug"],
        "protocol_version": protocol.study_version,
        "data_origin": "formal",
        "protocol_mode": "strict",
        "total": len(rows),
        "counts": dict(Counter(f"{row['cohort']}:{row['internal_group']}" for row in rows)),
        "all_accounts_held": True,
        "base_prior_id": base["id"],
        "files": {
            path.name: sha256(path)
            for path in (internal_path, blind_path, instruction_path)
        },
        "contains_full_codes": True,
    }
    manifest_path = output / "roster_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (internal_path, blind_path, instruction_path, manifest_path):
        os.chmod(path, 0o600)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
