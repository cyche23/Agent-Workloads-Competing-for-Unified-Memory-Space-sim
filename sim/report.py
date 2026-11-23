"""Paired analysis of capacity pressure and same-budget policy gains."""

import json
from collections import Counter, defaultdict
from math import floor
from pathlib import Path

from .config import Experiment, MIB
from .trace import Trace

ARMS = ("fixed", "shared", "oracle_fifo", "oracle", "relaxed")
ORACLE_PAIRS = (
    ("fixed", "oracle"), ("shared", "oracle_fifo"), ("oracle_fifo", "oracle"),
    ("shared", "oracle"), ("oracle", "relaxed"),
)


def quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    left = floor(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def aggregate(rows: list[dict], agents: list[dict]) -> dict:
    complete = [row["e2e_s"] for row in agents if row["outcome"] == "completed"]
    makespan = sum(row["makespan_s"] for row in rows)
    failures: Counter = Counter()
    for row in rows:
        failures.update(json.loads(row["failure_reasons"]))
    sums = (
        "kv_memory_wait_s", "tool_memory_wait_s", "gpu_wait_s", "kv_evictions",
        "tool_evictions", "reprefill_tokens", "reprefill_gpu_s", "tool_restore_s",
        "tool_restores", "checkpoint_read_s", "checkpoint_write_s", "checkpoint_disk_queue_s",
        "kv_partition_blocked_requests", "tool_partition_blocked_requests",
        "gpu_wait_during_reprefill_s",
        "fifo_head_wait_s", "fifo_tail_wait_s", "resident_bypass_admits", "oracle_bypass_admits",
        "tool_request_kv_evictions", "kv_request_tool_evictions",
    )
    return {
        "submitted": len(agents),
        "completed": len(complete),
        "completion_rate": len(complete) / len(agents),
        "goodput_per_min": 60 * len(complete) / makespan if makespan else None,
        "e2e_p50_s": quantile(complete, 0.5),
        "e2e_p95_s": quantile(complete, 0.95),
        "failure_reasons": json.dumps(dict(sorted(failures.items()))),
        "max_partition_shortfall_bytes": max(row["max_partition_shortfall_bytes"] for row in rows),
        **{field: sum(row[field] for row in rows) for field in sums},
    }


def paired(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    baseline_ids = {(row["workload_id"], row["job_id"]): row for row in baseline_rows}
    candidate_ids = {(row["workload_id"], row["job_id"]): row for row in candidate_rows}
    assert baseline_ids.keys() == candidate_ids.keys()
    baseline_e2e, candidate_e2e = [], []
    deltas = []
    saved_wait_s, saved_tokens, saved_restore_s = 0.0, 0, 0.0
    for key, baseline in baseline_ids.items():
        candidate = candidate_ids[key]
        assert baseline["instance"] == candidate["instance"]
        assert baseline["arrival_ns"] == candidate["arrival_ns"]
        for field in baseline:
            if field.startswith("planned_"):
                assert baseline[field] == candidate[field]
        if baseline["outcome"] == candidate["outcome"] == "completed":
            baseline_e2e.append(baseline["e2e_s"])
            candidate_e2e.append(candidate["e2e_s"])
            deltas.append(baseline["e2e_s"] - candidate["e2e_s"])
            saved_wait_s += (
                baseline["kv_memory_wait_s"] + baseline["tool_memory_wait_s"]
                - candidate["kv_memory_wait_s"] - candidate["tool_memory_wait_s"]
            )
            saved_tokens += baseline["reprefill_tokens"] - candidate["reprefill_tokens"]
            saved_restore_s += baseline["tool_restore_s"] - candidate["tool_restore_s"]
    return {
        "matched_completed": len(deltas),
        "matched_e2e_saved_sum_s": sum(deltas) if deltas else None,
        "matched_e2e_saved_p50_s": quantile(deltas, 0.5),
        "matched_e2e_saved_p95_s": quantile(deltas, 0.95),
        "matched_p95_difference_s": quantile(baseline_e2e, 0.95) - quantile(candidate_e2e, 0.95) if deltas else None,
        "matched_memory_wait_saved_s": saved_wait_s if deltas else None,
        "matched_reprefill_tokens_saved": saved_tokens if deltas else None,
        "matched_tool_restore_saved_s": saved_restore_s if deltas else None,
    }


def build_comparisons(runs: list[dict], agents: list[dict]) -> list[dict]:
    by_level = defaultdict(lambda: defaultdict(list))
    by_case = defaultdict(list)
    for row in runs:
        by_level[row["scenario"], row["concurrency"]][row["policy"], row["inference_percent"]].append(row)
    for row in agents:
        by_case[row["case_id"]].append(row)
    comparisons = []
    for (scenario, concurrency), arms in by_level.items():
        cohorts = {
            key: [agent for run in rows for agent in by_case[run["case_id"]]]
            for key, rows in arms.items()
        }
        metrics = {key: aggregate(rows, cohorts[key]) for key, rows in arms.items()}
        best = max(
            (key for key in arms if key[0] == "fixed"),
            key=lambda key: (
                metrics[key]["completed"], metrics[key]["goodput_per_min"] or 0,
                -key[1],
            ),
        )
        arm_keys = {name: (name, None) for name in ARMS}
        arm_keys["fixed"] = best
        pairs = (("fixed", "shared"), ("shared", "relaxed"), *ORACLE_PAIRS)
        comparisons.append({
            "scenario": scenario, "concurrency": concurrency, "seed_count": len(arms[best]),
            "best_fixed_inference_percent": best[1],
            **{f"{arm}_{field}": value for arm, key in arm_keys.items() for field, value in metrics[key].items()},
            **{f"{baseline}_to_{candidate}_{field}": value
               for baseline, candidate in pairs
               for field, value in paired(cohorts[arm_keys[baseline]], cohorts[arm_keys[candidate]]).items()},
        })
    return comparisons


def display(value: float | None, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def pressure_summary(rows: list[dict], scenario: str) -> list[str]:
    lines = []
    for row in rows:
        if row["scenario"] != scenario:
            continue
        level = row["concurrency"]
        counts = row["fixed_kv_partition_blocked_requests"] + row["fixed_tool_partition_blocked_requests"]
        delta = row["shared_completed"] - row["fixed_completed"]
        extra_wait = row["fixed_to_shared_matched_memory_wait_saved_s"]
        extra_recompute = row["fixed_to_shared_matched_reprefill_tokens_saved"]
        extra_restore = row["fixed_to_shared_matched_tool_restore_saved_s"]
        if not counts:
            continue
        values = [f"{level} 并发出现 {counts} 次有全局余量的固定分区阻塞",
                  f"最大额度缺口 {row['fixed_max_partition_shortfall_bytes'] / MIB:.1f} MiB"]
        if delta:
            values.append(f"共享池完成数变化 {delta:+d}")
        if extra_wait is not None:
            values.append(f"成对完成样本累计内存等待节省 {extra_wait:+.1f} 秒")
            values.append(f"KV 重算减少 {extra_recompute:+,} tokens")
            values.append(f"工具恢复节省 {extra_restore:+.1f} 秒")
        lines.append("- " + "，".join(values) + "。")
    return lines


def write_report(
    path: Path, config: Experiment, traces: tuple[Trace, ...],
    comparisons: list[dict], cases: int, quick: bool,
) -> None:
    lines = [
        "# 统一内存容量竞争仿真", "",
        f"共完成 {cases} 组仿真。运行模式为{'快速检查' if quick else '完整实验矩阵'}。"
        f"主场景使用 {config.profile.dram_gib:g} GiB DRAM，"
        f"其中 {config.profile.static_gib:g} GiB 预留给模型和系统。", "",
        "## 完成数、吞吐与延迟", "",
        "共享池使用统一 FIFO 和全局 LRU。已驻留的可复用状态在新增占用有余量时可直接准入。"
        "宽预算组使用同一准入次序。历史 v1 的共享池按两侧分别排队和回收。", "",
        f"两个 oracle 组使用同一 DRAM 上限和 {config.oracle_window_s:g} 秒预演窗。"
        "`oracle_fifo` 保留共享池准入规则，`oracle` 同时评估队尾请求。"
        "决策优先避免任务失败，再降低累计端到端时间的预测值。实测结果给出这些可行策略"
        "在当前模型中已达到的性能，宽预算组用于诊断容量放宽后的表现。", "",
        "固定分区按完成数、goodput 依次择优。完成/提交按轨迹重放计数。Goodput 为每仿真分钟"
        "完成数，由各次试验完成数除以各次 makespan 之和。P95 使用各组的完成样本，"
        "下表各指标依次列出 fixed、shared、oracle_fifo、oracle、relaxed。", "",
        "| 场景 | 并发 | 固定推理份额 | 完成数 | Goodput | P95 秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        complete = " / ".join(str(row[f"{arm}_completed"]) for arm in ARMS)
        goodput = " / ".join(display(row[f"{arm}_goodput_per_min"]) for arm in ARMS)
        p95 = " / ".join(display(row[f"{arm}_e2e_p95_s"], 1) for arm in ARMS)
        lines.append(
            f"| {row['scenario']} | {row['concurrency']} | {row['best_fixed_inference_percent']}% |"
            f" {complete}（各提交 {row['fixed_submitted']}） | {goodput} | {p95} |"
        )

    failures = []
    for row in comparisons:
        for arm in ARMS:
            reasons = json.loads(row[f"{arm}_failure_reasons"])
            if reasons:
                detail = "，".join(f"`{reason}` × {count}" for reason, count in reasons.items())
                failures.append(f"| {row['scenario']} | {row['concurrency']} | {arm} | {detail} |")
    if failures:
        lines.extend([
            "", "| 场景 | 并发 | 策略 | 失败原因与次数 |", "| --- | ---: | --- | --- |", *failures,
        ])

    lines.extend([
        "", "## Oracle 配对对照", "",
        "各项节省按左侧策略减去右侧策略计算。任务延迟、内存等待、重算和工具恢复使用双方均完成的同一批任务。"
        "配对 P95 差使用这批任务分别计算两组的 P95。磁盘等待为各组所有 I/O 的累计排队时间。", "",
        "| 场景 | 并发 | 对照 | 完成数变化 | 成对完成 | 累计延迟节省 秒 | 中位节省 秒 | P95 差 秒 |"
        " 内存等待节省 秒 | 重算节省 tokens | 工具恢复节省 秒 | 磁盘排队节省 秒 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comparisons:
        for baseline, candidate in ORACLE_PAIRS:
            prefix = f"{baseline}_to_{candidate}_"
            delta = row[f"{candidate}_completed"] - row[f"{baseline}_completed"]
            disk = row[f"{baseline}_checkpoint_disk_queue_s"] - row[f"{candidate}_checkpoint_disk_queue_s"]
            lines.append(
                f"| {row['scenario']} | {row['concurrency']} | {baseline} → {candidate} | {delta:+d} |"
                f" {row[prefix + 'matched_completed']} |"
                f" {display(row[prefix + 'matched_e2e_saved_sum_s'], 1)} |"
                f" {display(row[prefix + 'matched_e2e_saved_p50_s'], 1)} |"
                f" {display(row[prefix + 'matched_p95_difference_s'], 1)} |"
                f" {display(row[prefix + 'matched_memory_wait_saved_s'], 1)} |"
                f" {display(row[prefix + 'matched_reprefill_tokens_saved'], 0)} |"
                f" {display(row[prefix + 'matched_tool_restore_saved_s'], 1)} | {disk:.1f} |"
            )

    lines.extend([
        "", "## 共享池准入与驱逐", "",
        "等待时间是所有申请的累计秒数。队尾等待从提交到首次成为队首或直接准入，"
        "队首等待从首次成为队首到准入或失败。", "",
        "| 场景 | 并发 | 策略 | 队尾等待 秒 | 队首等待 秒 | 驻留命中直通 | Oracle 越序 |"
        " 工具申请驱逐 KV | 推理申请驱逐工具 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comparisons:
        for arm in ("shared", "oracle_fifo", "oracle"):
            lines.append(
                f"| {row['scenario']} | {row['concurrency']} | {arm} |"
                f" {row[f'{arm}_fifo_tail_wait_s']:.1f} | {row[f'{arm}_fifo_head_wait_s']:.1f} |"
                f" {row[f'{arm}_resident_bypass_admits']} | {row[f'{arm}_oracle_bypass_admits']} |"
                f" {row[f'{arm}_tool_request_kv_evictions']} | {row[f'{arm}_kv_request_tool_evictions']} |"
            )
    lines.extend(["", "## 归因", ""])
    for scenario in ("burst", "staggered"):
        selected = [row for row in comparisons if row["scenario"] == scenario]
        if not selected:
            continue
        lines.append(f"### {'同刻到达' if scenario == 'burst' else '错峰到达'}")
        lines.append("")
        lines.extend(pressure_summary(selected, scenario) or ["最优固定分区在本组负载下没有观测到额度闲置。"])
        lines.append("")
        for row in selected:
            excess = row["shared_to_relaxed_matched_e2e_saved_p50_s"]
            if row["shared_reprefill_tokens"] or row["shared_tool_restores"] or (
                row["shared_completed"] < row["relaxed_completed"]
            ) or (excess is not None and excess > 0):
                lines.append(
                    f"- {row['concurrency']} 并发时，共享池重算 {row['shared_reprefill_tokens']:,} tokens，"
                    f"恢复工具 {row['shared_tool_restores']} 次，相对宽预算的配对中位额外延迟 "
                    f"{display(excess, 1)} 秒（n={row['shared_to_relaxed_matched_completed']}）。"
                )
        lines.append("")

    lines.extend([
        "## 轨迹与模型口径", "",
        "| 实例 | 推理轮数 | 工具调用 | 常驻 MiB | 工具峰值 MiB | 无窗口内样本的调用 | 历史状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for trace in traces:
        lines.append(
            f"| {trace.instance} | {len(trace.rounds)} | {sum(len(r.tools) for r in trace.rounds)} | "
            f"{trace.tool_resident / MIB:.1f} | {trace.tool_peak / MIB:.1f} | "
            f"{trace.uncovered_tools} | {trace.historical_status} |"
        )
    lines.extend([
        "",
        "历史 API 的 token 数用于估计本地上下文。批吞吐按配置，工具窗口按原轨迹重放，"
        "硬件容量分析针对当前内存账本。工具执行可重叠且持续时间固定。"
        "内核换页、CPU/带宽争用、驱动页池与跨请求前缀共享处于模型范围之外。",
        "",
        "完整固定比例和失败原因见 [runs.csv](runs.csv)，配对聚合见 [comparisons.csv](comparisons.csv)，"
        "逐 Agent 时间账本见 [agents.csv](agents.csv)。内存准入和回收链路见 [事件日志](events.jsonl.gz)。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
