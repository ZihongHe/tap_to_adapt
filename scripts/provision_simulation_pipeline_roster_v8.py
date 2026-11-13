"""Command-line utility: provision simulation pipeline roster v8.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from provision_simulation_pipeline_roster import (
    behavior_profile,
    create_block,
    sha256,
    write_csv,
    write_text,
)


PARTICIPANT_GUIDE = """# 三日在线语音互动任务操作指引

1. 电脑使用 Chrome 或 Edge 打开 `http://127.0.0.1:8765/`；手机打开 `http://127.0.0.1:8765/companion`。
2. 两端输入同一参与编号并选择同一实验日。每天开始前保持两端同时在线，按页面完成网络、麦克风、浏览器和点击反应检测。
3. 在电脑端从话题列表选择一个起始话题，自然交谈即可；需要时可以主动换话题，不需要表演预设情境。
4. 当你希望电脑端智能体现在开始回答时，点击电脑页面；如果智能体已经响应，而你希望它立即停止当前回答，也点击同一区域。只按当下真实感受操作，不按固定频率点击，也不要为了凑次数连点。
5. 页面要求手机端交流时，继续自然对话并保持背景声播放。页面暂停时，先恢复网络、声音或双端在线状态，再继续。
6. 完成当天问卷后再关闭两端。Day 2 和 Day 3 必须继续使用同一编号；页面显示“尚未开放”表示所在批次仍在等待，不是网络故障。

请勿在对话中透露姓名、手机号、证件、财务、医疗或他人隐私。可以跳过任何不愿讨论的话题，也可以随时退出。"""


COORDINATOR_GUIDE = """# 盲化批次安排

协调员只使用 `blinded_simulation_roster.csv` 中的 neutral_id、participant_code 和 release_wave，不接触内部组别、模型坐标或研究假设。

| 波次 | 人数 | 何时介入 |
| --- | ---: | --- |
| W1 | 12 | 当前开放；分别完成 Day 1、Day 2、Day 3 |
| W2 | 8 | 后台确认 W2 released 后通知 |
| W3 | 8 | 后台确认 W3 released 后通知 |
| W4 | 8 | 后台确认 W4 released 后通知 |
| W5 | 32 | Study B 准备完成并开放后通知 |
| W6 | 4 | 仅在预注册替补条件触发后通知 |

协调员只提醒日期、双端同号、设备检测和完成状态；不建议何时点击，不解释比较条件，也不告知系统是否正在学习。"""


BLINDED_RUNNER_GUIDE = """# 模拟参与者盲化操作指南

你扮演一名真实参与者，而不是测试脚本。只读取当批 `runner_W*.csv` 中的 neutral_id、participant_code、behavior_profile 和本指南，不读取内部 assignment、group、prior、q、模型概率或研究假设。

## 每个编号

1. 使用相同编号分别打开电脑端和手机伴随端，并在两端选择同一天。
2. 完成页面要求的设备检测、话题选择和三日流程；每天至少完成一次真实 primary chat、一次服务端语音播放和一次有效双端重叠。
3. 从页面提供的话题中选择想聊的内容，自然说话、停顿、补充和换话题。
4. 只在主观上希望智能体现在回答时点击；智能体已经回答但你此刻希望它停止时，也按真实感受点击。不要为了制造某类结果、达到次数或修复后台指标而点击。
5. 同一编号三天保持同一行为画像，但不机械重复固定频率：ordinary 自然协作；patient 更愿意等待和思考；impatient 偏好简短及时的回复；chatty 更愿意展开和追问；uncertain 在决定是否点击时更犹豫。
6. 页面报错时记录原始错误并停止该日，不绕过门禁，不更改后台，不换成别人的编号继续。

不得根据已经产生的标签或统计结果改变后续行为，也不得刻意补齐正例、负例或任何 q 水平。"""


TRANSPORT_GUIDE = """# v9 模拟执行器传输与记录约束

- 严格按 wave 执行；未释放编号预期返回 HTTP 423，不绕过后台门禁。
- 行为决策模型只接收 `runner_W*.csv` 与 `blinded_simulated_user_guide.zh-CN.md`；本文件只供传输控制器和验收程序读取。
- 连续发送真实当前 PCM，不伪造 prediction 时间或 tap 的 client_ms；收到 `prediction.activate=true` 后才开始记录本地反应时间。
- 点击由模拟参与者当下判断触发。控制器不得读取 q 后补点击，也不得因某次反馈不合格而重试凑标签。
- 同一 `activation_id` 上的首次有效 interrupt 均计为负反馈，不以 thinking/speaking、反应时长或 tap 前是否恢复语音作为资格门槛。训练成功还须满足：`label=0`、`training_eligible=true`、`training_applied=true`、`training_audio_compatible=true`、`training_audio_samples>=8000`、`label_semantics=observable_activation_interrupt_v2`、`quarantined=false`。
- `activation_to_tap_ms`、`user_speech_resumed`、`interrupt_agent_state` 与置信度字段只用于连续遥测和分层分析，不得据此补点、重试或删除真实中断；音频过短或优化器未更新时保留服务端 quarantine 原因。
- 在关闭 WebSocket 前先停止 PCM 发送任务并等待其退出，避免 cleanup 阶段继续写入已关闭连接。
- 每日确认质量检测、双端同日在线、话题、primary chat、服务端 TTS、问卷和 protocol-status 均完整。原始失败尝试单独保留，不覆盖成功记录。
- q 是每位贡献者自然产生且实际应用的反馈上限 `min(K_i,q)`，不是点击目标。"""


def main(
    *,
    expected_protocol_version: str = "three-day-online-2026-07-18-v9",
    roster_version: str = "v9",
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Provision an isolated {roster_version} 36A+36B compressed simulation roster."
        )
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
    if protocol.study_version != expected_protocol_version:
        raise SystemExit(f"Refusing to provision with protocol {protocol.study_version}")
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

    neutral_prefix = roster_version.upper()
    neutral_ids = [f"{neutral_prefix}S{index:03d}" for index in range(1, 73)]
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

    internal_path = output / "simulation_assignments.csv"
    write_csv(internal_path, sorted(rows, key=lambda row: row["assignment_id"]))
    blind_rows = [
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
    ]
    blind_path = output / "blinded_simulation_roster.csv"
    write_csv(
        blind_path,
        sorted(blind_rows, key=lambda row: (int(row["release_wave"][1:]), row["neutral_id"])),
    )
    runner_rows = [
        {
            "neutral_id": row["neutral_id"],
            "participant_code": row["participant_code"],
            "release_wave": f"W{row['enrollment_wave']}",
            "behavior_profile": behavior_profile(row["neutral_id"]),
            "analysis_use": f"simulation_human_{roster_version}",
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

    guides = {
        "participant_instructions.zh-CN.md": PARTICIPANT_GUIDE,
        "blinded_coordinator_schedule.zh-CN.md": COORDINATOR_GUIDE,
        "blinded_simulated_user_guide.zh-CN.md": BLINDED_RUNNER_GUIDE,
        "simulation_runner_protocol.zh-CN.md": TRANSPORT_GUIDE.replace(
            "v9", roster_version
        ),
    }
    guide_paths: list[Path] = []
    for name, content in guides.items():
        path = output / name
        write_text(path, content)
        guide_paths.append(path)

    manifest = {
        "version": f"simulation-roster-36A-36B-{roster_version}",
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
        "feedback_application": protocol.study["feedback"][
            "eligible_feedback_application"
        ],
        "latency_clock": protocol.study["feedback"]["latency_clock"],
        "base_prior_id": base_prior["id"],
        "initial_released": released["released"],
        "formal_roster_untouched": True,
        "behavior_transport_prompt_separation": True,
        "files": {
            internal_path.name: sha256(internal_path),
            blind_path.name: sha256(blind_path),
            runner_path.name: sha256(runner_path),
            **{path.name: sha256(path) for path in guide_paths},
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
        *guide_paths,
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
