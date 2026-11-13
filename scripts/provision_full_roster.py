"""Provision the complete preregistered study roster.

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
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_a_wave(index: int) -> str:
    if index <= 20:
        return "initial_20"
    return f"sequential_batch_{((index - 21) // 4) + 1}_of_4"


def rows_for_block(
    db: Any,
    *,
    count: int,
    cohort: str,
    group: str,
    assignment_prefix: str,
    phase_label: str,
    analysis_use: str,
) -> list[dict[str, Any]]:
    codes = db.create_participants(count, cohort, group)
    rows: list[dict[str, Any]] = []
    for index, code in enumerate(codes, start=1):
        participant = db.participant_by_code(code)
        if participant is None:
            raise RuntimeError("A newly created participant could not be reloaded")
        schedules = json.loads(participant["probe_schedule"])
        if assignment_prefix == "A":
            wave = study_a_wave(index)
        elif assignment_prefix in {"BP", "BF"}:
            wave = "initial_32" if index <= 16 else "blinded_precision_reserve"
        else:
            wave = "engineering_pilot"
        rows.append(
            {
                "assignment_id": f"{assignment_prefix}{index:02d}",
                "participant_code": code,
                "participant_id": participant["id"],
                "cohort": cohort,
                "phase": phase_label,
                "internal_group": group,
                "enrollment_wave": wave,
                "day1_probe_order": schedules[0],
                "day2_probe_order": schedules[1],
                "day3_probe_order": schedules[2],
                "analysis_use": analysis_use,
            }
        )
    return rows


def markdown_document(rows: list[dict[str, Any]], created_ms: int) -> str:
    lines = [
        "# 参与者编号与职责说明（研究者私有）",
        "",
        "> 本文件包含完整登录编号，只能由研究团队保存，不得上传公开仓库、论文附件或群聊。参与者只接收自己的编号。身份联系方式必须保存在本系统之外。",
        "",
        f"生成时间戳：`{created_ms}`。编号总数：`{len(rows)}`。",
        "",
        "## 双端有效性规则",
        "",
        "1. 每名参与者连续三天使用同一个编号。",
        "2. 电脑端与手机伴随端必须输入同一编号并选择同一天。",
        "3. 两端 WebSocket 必须同时在线，手机声音必须显示就绪；后端成功激活后 timing 数据才有效。",
        "4. 任一端断线或心跳超过 15 秒，系统暂停 timing 数据并结束当前连续配对区间；恢复双端连接后才能继续。",
        "5. 不得把 Personalized/Frozen、probe 顺序或 Study A 的到达批次告知参与者。参与者面对的说明和任务保持一致。",
        "",
        "## 角色职责",
        "",
        "| 内部角色 | 人数 | 研究职责 | 是否进入正式分析 |",
        "|---|---:|---|---|",
        "| Pilot donor | 6 | 验证 Study A 三日流程、tap 更新和共享数据管线 | 否 |",
        "| Pilot personalized | 6 | 验证 Study B 个性化更新、保存和隔夜恢复 | 否 |",
        "| Pilot frozen | 6 | 验证冻结模型不更新及三日流程一致性 | 否 |",
        "| Study A donor | 36（20 起步，4 人一批） | 贡献共享先验饱和曲线的数据；按预注册停止规则决定是否启用后续批次 | 是 |",
        "| Study B Personalized | 18（16+2 reserve） | 三天保留并更新个人模型；参与每日 current/matched-other/VAD micro-block | 是 |",
        "| Study B Frozen | 18（16+2 reserve） | 三天使用冻结共享 prior；参与相同 micro-block 和流程 | 是 |",
        "",
        "## 编号映射",
        "",
        "`initial_20` 和 `initial_32` 先招募；`sequential_batch_*` 仅在 Study A 未满足饱和停止规则时依次启用；`blinded_precision_reserve` 只能按盲态 precision rule 启用。",
        "",
        "| 分配号 | 完整编号 | 阶段 | 内部长组 | 招募波次 | Day 1/2/3 probe 顺序 | 用途 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        orders = "/".join(
            [row["day1_probe_order"], row["day2_probe_order"], row["day3_probe_order"]]
        )
        lines.append(
            "| {assignment_id} | `{participant_code}` | {phase} | {internal_group} | "
            "{enrollment_wave} | `{orders}` | {analysis_use} |".format(orders=orders, **row)
        )
    lines.extend(
        [
            "",
            "## 发放检查",
            "",
            "- 每次只向一名参与者发送其单独编号和两个网址。",
            "- 发放前在独立身份映射表记录分配号，不在本系统存储姓名、手机号或支付账户。",
            "- Day 2/3 继续使用原编号，不生成新编号。",
            "- 参与者不得交换编号；发现双人共享或跨设备编号不一致时标记 protocol deviation。",
            "- Pilot 编号永远不能转成正式编号；正式参与者不得参加 Pilot。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision the complete preregistered study roster.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-formal-roster", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_formal_roster:
        raise SystemExit("Pass --acknowledge-formal-roster after checking the 36+36 design")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    output_dir = args.output_dir.resolve()
    markdown_path = output_dir / "participant_assignments.zh-CN.md"
    csv_path = output_dir / "participant_assignments.csv"
    manifest_path = output_dir / "roster_manifest.json"
    if any(path.exists() for path in (markdown_path, csv_path, manifest_path)):
        raise SystemExit(f"Refusing to overwrite an existing roster in {output_dir}")

    db = StudyDB(settings)
    existing_formal = [
        row for row in db.table_rows("participants") if row["cohort"] in {"study_a", "study_b"}
    ]
    if existing_formal:
        raise SystemExit(
            f"Refusing to create a second formal roster: {len(existing_formal)} formal rows already exist"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.chmod(0o700)
    except OSError:
        pass

    rows: list[dict[str, Any]] = []
    rows += rows_for_block(
        db,
        count=6,
        cohort="pilot",
        group="donor",
        assignment_prefix="PA",
        phase_label="Pilot / Study A pipeline",
        analysis_use="engineering_only",
    )
    rows += rows_for_block(
        db,
        count=6,
        cohort="pilot",
        group="personalized",
        assignment_prefix="PBP",
        phase_label="Pilot / Study B personalized",
        analysis_use="engineering_only",
    )
    rows += rows_for_block(
        db,
        count=6,
        cohort="pilot",
        group="frozen",
        assignment_prefix="PBF",
        phase_label="Pilot / Study B frozen",
        analysis_use="engineering_only",
    )
    rows += rows_for_block(
        db,
        count=36,
        cohort="study_a",
        group="donor",
        assignment_prefix="A",
        phase_label="Study A",
        analysis_use="formal_when_unlocked",
    )
    rows += rows_for_block(
        db,
        count=18,
        cohort="study_b",
        group="personalized",
        assignment_prefix="BP",
        phase_label="Study B",
        analysis_use="formal_when_unlocked",
    )
    rows += rows_for_block(
        db,
        count=18,
        cohort="study_b",
        group="frozen",
        assignment_prefix="BF",
        phase_label="Study B",
        analysis_use="formal_when_unlocked",
    )

    created_ms = int(time.time() * 1000)
    fields = list(rows[0].keys())
    with csv_path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown_document(rows, created_ms), encoding="utf-8")

    counts = Counter((row["cohort"], row["internal_group"]) for row in rows)
    manifest = {
        "version": "full-roster-36A-36B-plus-18pilot-v1",
        "created_ms": created_ms,
        "total": len(rows),
        "counts": {f"{cohort}:{group}": count for (cohort, group), count in counts.items()},
        "markdown_sha256": file_sha256(markdown_path),
        "csv_sha256": file_sha256(csv_path),
        "contains_full_access_codes": True,
        "participant_blinding_required": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (markdown_path, csv_path, manifest_path):
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    print(
        f"Provisioned {len(rows)} codes. Full codes were written only to {markdown_path} and {csv_path}."
    )


if __name__ == "__main__":
    main()
