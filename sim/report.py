"""Paired analysis of capacity pressure and fixed-partition losses."""

import json
from collections import Counter, defaultdict
from math import floor
from pathlib import Path

from .config import Experiment, MIB
from .trace import Trace


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
        "tool_restores", "checkpoint_read_s", "checkpoint_write_s",
        "kv_partition_blocked_requests", "tool_partition_blocked_requests",
        "gpu_wait_during_reprefill_s",
        "fifo_head_wait_s", "fifo_tail_wait_s", "resident_bypass_admits",
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


def paired(fixed: list[dict], other: list[dict]) -> dict:
    fixed_ids = {(row["workload_id"], row["job_id"]): row for row in fixed}
    shared_ids = {(row["workload_id"], row["job_id"]): row for row in other}
    assert fixed_ids.keys() == shared_ids.keys()
    fixed_e2e, other_e2e = [], []
    deltas = []
    saved_wait_s, saved_tokens, saved_restore_s = 0.0, 0, 0.0
    for key, baseline in fixed_ids.items():
        candidate = shared_ids[key]
        assert baseline["instance"] == candidate["instance"]
        assert baseline["arrival_ns"] == candidate["arrival_ns"]
        for field in baseline:
            if field.startswith("planned_"):
                assert baseline[field] == candidate[field]
        if baseline["outcome"] == candidate["outcome"] == "completed":
            fixed_e2e.append(baseline["e2e_s"])
            other_e2e.append(candidate["e2e_s"])
            deltas.append(baseline["e2e_s"] - candidate["e2e_s"])
            saved_wait_s += (
                baseline["kv_memory_wait_s"] + baseline["tool_memory_wait_s"]
                - candidate["kv_memory_wait_s"] - candidate["tool_memory_wait_s"]
            )
            saved_tokens += baseline["reprefill_tokens"] - candidate["reprefill_tokens"]
            saved_restore_s += baseline["tool_restore_s"] - candidate["tool_restore_s"]
    return {
        "matched_completed": len(deltas),
        "matched_e2e_saved_p50_s": quantile(deltas, 0.5),
        "matched_e2e_saved_p95_s": quantile(deltas, 0.95),
        "matched_p95_difference_s": quantile(fixed_e2e, 0.95) - quantile(other_e2e, 0.95) if deltas else None,
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
        shared, relaxed = ("shared", None), ("relaxed", None)
        fixed_shared = paired(cohorts[best], cohorts[shared])
        shared_relaxed = paired(cohorts[shared], cohorts[relaxed])
        comparisons.append({
            "scenario": scenario, "concurrency": concurrency, "seed_count": len(arms[best]),
            "best_fixed_inference_percent": best[1],
            **{f"{arm}_{field}": value for arm, key in
               (("fixed", best), ("shared", shared), ("relaxed", relaxed))
               for field, value in metrics[key].items()},
            **{f"fixed_to_shared_{field}": value for field, value in fixed_shared.items()},
            **{f"shared_to_relaxed_{field}": value for field, value in shared_relaxed.items()},
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
        "## 固定分区与共享池", "",
        "共享池使用统一 FIFO 和全局 LRU。已驻留的可复用状态在新增占用有余量时可直接准入。"
        "宽预算组使用同一准入次序。历史 v1 的共享池按两侧分别排队和回收。", "",
        "固定分区按完成数、goodput 依次择优。完成/提交按轨迹重放计数。Goodput 为每仿真分钟"
        "完成数，由各次试验完成数除以各次 makespan 之和。P95 使用各组的完成样本，"
        "成对 P95 差在双方均完成的同一批 Agent 上计算，正值表示共享池尾延迟更低。", "",
        "| 场景 | 并发 | 固定推理份额 | 完成数 固定/共享/宽预算 | Goodput 固定/共享/宽预算 |"
        " P95 秒 固定/共享/宽预算 | 共享池成对中位节省 秒 | 成对 P95 差 秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    arms = ("fixed", "shared", "relaxed")
    for row in comparisons:
        complete = " / ".join(str(row[f"{arm}_completed"]) for arm in arms)
        goodput = " / ".join(display(row[f"{arm}_goodput_per_min"]) for arm in arms)
        p95 = " / ".join(display(row[f"{arm}_e2e_p95_s"], 1) for arm in arms)
        paired_count = row["fixed_to_shared_matched_completed"]
        saved = display(row["fixed_to_shared_matched_e2e_saved_p50_s"], 1)
        p95_saved = display(row["fixed_to_shared_matched_p95_difference_s"], 1)
        lines.append(
            f"| {row['scenario']} | {row['concurrency']} | {row['best_fixed_inference_percent']}% |"
            f" {complete}（共 {row['fixed_submitted']}） | {goodput} | {p95} | {saved}（n={paired_count}） |"
            f" {p95_saved} |"
        )

    lines.extend([
        "", "## 共享池准入与驱逐", "",
        "等待时间是所有申请的累计秒数。队尾等待从提交到首次成为队首或直接准入，"
        "队首等待从首次成为队首到准入或失败。", "",
        "| 场景 | 并发 | 队尾等待 秒 | 队首等待 秒 | 驻留命中直通 | 工具申请驱逐 KV | 推理申请驱逐工具 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['scenario']} | {row['concurrency']} | "
            f"{row['shared_fifo_tail_wait_s']:.1f} | {row['shared_fifo_head_wait_s']:.1f} | "
            f"{row['shared_resident_bypass_admits']} | {row['shared_tool_request_kv_evictions']} | "
            f"{row['shared_kv_request_tool_evictions']} |"
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
