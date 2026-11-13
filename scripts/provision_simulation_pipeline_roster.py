"""Provision an isolated v7 36A+36B compressed simulation roster.

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


def behavior_profile(neutral_id: str) -> str:
    profiles = ("ordinary", "patient", "impatient", "chatty", "uncertain")
    digest = hashlib.sha256(neutral_id.encode("utf-8")).digest()
    return profiles[int.from_bytes(digest[:4], "big") % len(profiles)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty roster: {path.name}")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def create_block(
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
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug=run_slug,
    )
    rows: list[dict[str, Any]] = []
    for offset, code in enumerate(codes):
        participant = db.participant_by_code(code)
        if participant is None:
            raise RuntimeError("New simulation participant could not be reloaded")
        index = assignment_start + offset
        schedule = json.loads(participant["probe_schedule"])
        rows.append(
            {
                "assignment_id": f"{prefix}{index:02d}",
                "participant_code": code,
                "participant_id": participant["id"],
                "cohort": cohort,
                "internal_group": group,
                "enrollment_wave": wave,
                "day1_probe_order": schedule[0],
                "day2_probe_order": schedule[1],
                "day3_probe_order": schedule[2],
                "analysis_use": "simulation_only",
                "data_origin": "simulation",
                "protocol_mode": "compressed",
                "run_slug": run_slug,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision an isolated v7 36A+36B compressed simulation roster."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_simulation_only:
        raise SystemExit(
            "Pass --acknowledge-simulation-only; generated records are never analysis eligible"
        )

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB
    from app.pipeline import FormalPipeline
    from app.protocol import Protocol

    db = StudyDB(settings)
    protocol = Protocol(settings)
    if db.study_run(args.run_slug) is not None:
        raise SystemExit(f"Run slug already exists: {args.run_slug}")
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    try:
        os.chmod(output, 0o700)
    except OSError:
        pass

    rows: list[dict[str, Any]] = []
    rows += create_block(
        db,
        run_slug=args.run_slug,
        count=12,
        cohort="study_a",
        group="seed_donor",
        prefix="A",
        assignment_start=1,
        wave=1,
    )
    rows += create_block(
        db,
        run_slug=args.run_slug,
        count=8,
        cohort="study_a",
        group="evaluator_donor",
        prefix="A",
        assignment_start=13,
        wave=2,
    )
    rows += create_block(
        db,
        run_slug=args.run_slug,
        count=8,
        cohort="study_a",
        group="evaluator_donor",
        prefix="A",
        assignment_start=21,
        wave=3,
    )
    rows += create_block(
        db,
        run_slug=args.run_slug,
        count=8,
        cohort="study_a",
        group="holdout_evaluator",
        prefix="A",
        assignment_start=29,
        wave=4,
    )
    for prefix, group in (("BP", "personalized"), ("BF", "frozen")):
        rows += create_block(
            db,
            run_slug=args.run_slug,
            count=16,
            cohort="study_b",
            group=group,
            prefix=prefix,
            assignment_start=1,
            wave=5,
        )
        rows += create_block(
            db,
            run_slug=args.run_slug,
            count=2,
            cohort="study_b",
            group=group,
            prefix=prefix,
            assignment_start=17,
            wave=6,
        )

    neutral_ids = [f"SIM{index:03d}" for index in range(1, 73)]
    secrets.SystemRandom().shuffle(neutral_ids)
    for row, neutral_id in zip(rows, neutral_ids, strict=True):
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
    pipeline = FormalPipeline(db, settings, protocol.study, allow_simulation=True)
    released = pipeline.release_study_a_wave(args.run_slug, 1)
    base_prior = next(
        prior
        for prior in db.shared_priors(run["id"])
        if int(prior["donor_count"]) == 0 and int(prior["opportunity_budget"]) == 0
    )

    internal_rows = sorted(rows, key=lambda row: row["assignment_id"])
    internal_path = output / "simulation_assignments.csv"
    write_csv(internal_path, internal_rows)

    blind_rows = sorted(
        [
            {
                "neutral_id": row["neutral_id"],
                "participant_code": row["participant_code"],
                "release_wave": f"W{row['enrollment_wave']}",
                "day1": "收到该波次开放通知后开始",
                "day2": "完成 Day 1 后按页面允许时间开始",
                "day3": "完成 Day 2 后按页面允许时间开始",
                "simulation_only": "yes",
            }
            for row in rows
        ],
        key=lambda row: (int(row["release_wave"][1:]), row["neutral_id"]),
    )
    blind_path = output / "blinded_simulation_roster.csv"
    write_csv(blind_path, blind_rows)

    runner_rows = [
        {
            "neutral_id": row["neutral_id"],
            "participant_code": row["participant_code"],
            "release_wave": f"W{row['enrollment_wave']}",
            "behavior_profile": behavior_profile(row["neutral_id"]),
            "analysis_use": "simulation_human_v7",
        }
        for row in sorted(
            rows,
            key=lambda value: (int(value["enrollment_wave"]), value["neutral_id"]),
        )
    ]
    runner_path = output / "synthetic_runner_roster.csv"
    write_csv(runner_path, runner_rows)
    wave_paths: list[Path] = []
    for wave in range(1, 7):
        wave_path = output / f"runner_W{wave}.csv"
        write_csv(
            wave_path,
            [row for row in runner_rows if row["release_wave"] == f"W{wave}"],
        )
        wave_paths.append(wave_path)

    participant_instructions = output / "participant_instructions.zh-CN.md"
    write_text(
        participant_instructions,
        """# 三日在线语音交互任务操作指引

1. 电脑使用 Chrome 或 Edge 打开 `http://127.0.0.1:8765/`；手机打开 `http://127.0.0.1:8765/companion`。
2. 两端输入同一个参与编号，选择同一个实验日。每天开始前保持两端同时在线，并按页面完成网络、麦克风、浏览器和点击反应检测。
3. 在电脑端从话题列表选择一个起始话题，自然交谈即可。需要时可主动换话题，不需要表演预设情境。
4. 当你希望电脑端智能体现在开始回答时，点击电脑页面；如果智能体已经响应，而你希望立即停止当前响应，也点击同一区域。只按当下真实感受操作，不按固定频率点击，也不要为了凑次数连点。
5. 页面要求手机端交流时，继续自然对话并保持背景声播放。页面暂停时，先恢复网络、声音或双端在线状态，再继续。
6. 完成当天问卷后再关闭两端。Day 2 和 Day 3 必须继续使用同一个编号；页面显示“尚未开放”表示所在批次仍在等待，不是网络故障。

请勿在对话中透露姓名、手机号、证件、财务、医疗或他人隐私。可以跳过任何不愿讨论的话题，也可以随时退出。""",
    )

    coordinator_instructions = output / "blinded_coordinator_schedule.zh-CN.md"
    write_text(
        coordinator_instructions,
        """# 盲化批次安排

协调员只使用 `blinded_simulation_roster.csv` 中的 neutral_id、participant_code 和 release_wave 联系参与者，不接触内部组别、模型坐标或研究假设。

| 波次 | 人数 | 何时介入 |
| --- | ---: | --- |
| W1 | 12 | 当前开放；分别完成 Day 1、Day 2、Day 3 |
| W2 | 8 | 后台确认 W2 released 后通知 |
| W3 | 8 | 后台确认 W3 released 后通知 |
| W4 | 8 | 后台确认 W4 released 后通知 |
| W5 | 32 | Study B 准备完成并开放后通知 |
| W6 | 4 | 仅在预注册替补条件触发后通知 |

协调员只提醒日期、双端同号、设备检测和完成状态；不建议何时点击，不解释比较条件，也不告知系统是否正在学习。""",
    )

    runner_instructions = output / "simulation_runner_protocol.zh-CN.md"
    write_text(
        runner_instructions,
        """# v7 模拟参与者执行约束

- 严格按 wave 执行；未释放编号预期返回 HTTP 423，不得绕过后台门禁。
- 每天先完成四项质量检测并保存一个起始话题，再激活双端会话。
- 执行者只接收对应的 `runner_W*.csv`；同一编号三天保持同一行为画像，但每次点击只能由当下交互感受触发。不得读取内部映射、组别或模型坐标。
- 正训练标签只来自 learning/current：等待时的自然手动激活为正例；模型激活后 2 秒内且用户已恢复说话的中断为时序负例。其余中断照常执行但保持 quarantined。
- 不以 q=2/4/8 为点击目标，不补计数。q 是每位贡献者的自然合格反馈上限 min(K_i,q)。
- Day 1 若出现多个“开场交流”阶段，按相同自然标准交互；不得猜测或硬对齐不同阶段。
- 每日必须产生至少一次真实 primary chat、服务端 TTS 和双端有效重叠；结束前确认 protocol-status complete。""",
    )

    manifest = {
        "version": "simulation-roster-36A-36B-v7",
        "created_ms": int(time.time() * 1000),
        "run_id": run["id"],
        "run_slug": run["slug"],
        "protocol_version": protocol.study_version,
        "protocol_sha256": run["protocol_sha256"],
        "data_origin": "simulation",
        "protocol_mode": "compressed",
        "total": len(rows),
        "counts": dict(
            Counter(f"{row['cohort']}:{row['internal_group']}" for row in rows)
        ),
        "analysis_eligible": False,
        "study_a_wave_sizes": protocol.study["pipeline"]["study_a_wave_sizes"],
        "study_a_wave_roles": protocol.study["pipeline"]["study_a_wave_roles"],
        "study_b_initial_per_group": protocol.study["pipeline"][
            "study_b_initial_per_group"
        ],
        "study_b_reserve_per_group": protocol.study["pipeline"][
            "study_b_reserve_per_group"
        ],
        "budget_kind": protocol.study["pipeline"]["budget_kind"],
        "q_levels": protocol.study["pipeline"]["q_feedback_levels"],
        "base_prior_id": base_prior["id"],
        "initial_released": released["released"],
        "formal_roster_untouched": True,
        "files": {
            internal_path.name: sha256(internal_path),
            blind_path.name: sha256(blind_path),
            runner_path.name: sha256(runner_path),
            participant_instructions.name: sha256(participant_instructions),
            coordinator_instructions.name: sha256(coordinator_instructions),
            runner_instructions.name: sha256(runner_instructions),
            **{path.name: sha256(path) for path in wave_paths},
        },
        "contains_full_codes": True,
    }
    manifest_path = output / "roster_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (
        internal_path,
        blind_path,
        runner_path,
        participant_instructions,
        coordinator_instructions,
        runner_instructions,
        *wave_paths,
        manifest_path,
    ):
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
