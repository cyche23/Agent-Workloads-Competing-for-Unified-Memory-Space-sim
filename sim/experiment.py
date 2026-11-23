"""Run paired sweeps and save the replay inputs and results."""

import csv
import gzip
import hashlib
import io
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from .config import Experiment, GIB, Profile, oracle_window_ticks, seconds, ticks
from .engine import Arrival, Policy, RunResult, Simulator
from .report import build_comparisons, quantile, write_report
from .trace import Trace, load_traces

PHASES = (
    "kv_memory_wait", "tool_memory_wait", "gpu_wait", "prefill", "decode",
    "reprefill", "tool_restore", "tool_execute",
)
JOB_COUNTS = (
    "rounds_executed", "tool_calls_executed", "prefill_tokens", "decode_tokens",
    "reused_prompt_tokens", "reprefill_tokens", "kv_evictions", "tool_evictions", "tool_restores",
    "kv_partition_blocked_requests", "tool_partition_blocked_requests", "resident_bypass_admits",
    "oracle_bypass_admits",
)
FIFO_TIMERS = ("fifo_tail_wait_ns", "fifo_head_wait_ns")
RUN_TIMERS = (
    "prefill_gpu_ns", "decode_gpu_ns", "reprefill_gpu_ns", "kv_eviction_ns",
    "tool_eviction_ns", "checkpoint_read_ns", "checkpoint_write_ns",
    "checkpoint_disk_queue_ns",
)
RUN_COUNTS = (
    "prefill_batches", "decode_batches", "max_prefill_batch", "max_decode_batch",
    "max_partition_shortfall_bytes", "tool_request_kv_evictions", "kv_request_tool_evictions",
)
POLICY_ADMISSION = {
    "fixed": "per_side_fifo",
    "shared": "global_fifo_with_resident_hits",
    "oracle_fifo": "global_fifo_with_resident_hits",
    "oracle": "bounded_lookahead_queue_reordering",
    "relaxed": "global_fifo_with_resident_hits",
}
POLICY_EVICTION = {
    "fixed": "per_side_lru",
    "shared": "global_lru",
    "oracle_fifo": "bounded_lookahead_wait_or_reclaim",
    "oracle": "bounded_lookahead_wait_or_reclaim",
    "relaxed": "none",
}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arrivals_for(
    traces: tuple[Trace, ...], concurrency: int, seed: int, stagger_max_s: float
) -> tuple[Arrival, ...]:
    rng = random.Random(seed)
    start_times = [ticks(rng.uniform(0, stagger_max_s)) for _ in range(concurrency)]
    first = min(start_times)
    return tuple(
        Arrival(index + 1, traces[(index + seed) % len(traces)], start - first)
        for index, start in enumerate(start_times)
    )


def scenarios(config: Experiment, quick: bool) -> list[tuple[str, Profile, tuple[int, ...], float]]:
    main = ("burst", config.profile, config.concurrency, 0.0)
    if quick:
        return [main]
    result = [main, ("staggered", config.profile, config.concurrency, config.stagger_max_s)]
    subset = config.sensitivity_concurrency
    if not subset:
        return result
    for capacity in config.sensitivity_dram_gib:
        result.append((f"dram_{capacity:g}gib", replace(config.profile, dram_gib=capacity), subset, 0.0))
    result.append((
        "prefix_sensitivity", replace(config.profile, prefix_fraction=config.sensitivity_prefix_fraction),
        subset, 0.0,
    ))
    for factor in config.sensitivity_restore_factors:
        adjusted = replace(
            config.profile, tool_read_s=config.profile.tool_read_s * factor,
            tool_restore_s=config.profile.tool_restore_s * factor,
        )
        result.append((f"restore_{factor:g}x", adjusted, subset, 0.0))
    return result


def planned_work(trace: Trace) -> dict:
    return {
        "planned_rounds": len(trace.rounds),
        "planned_prompt_tokens": sum(round_.prompt for round_ in trace.rounds),
        "planned_decode_tokens": sum(round_.output for round_ in trace.rounds),
        "planned_tool_calls": sum(len(round_.tools) for round_ in trace.rounds),
        "planned_tool_ns": sum(round_.tool_ns for round_ in trace.rounds),
    }


def measure_run(context: dict, result: RunResult) -> tuple[dict, list[dict]]:
    jobs = []
    for job in result.jobs:
        trace = job.arrival.trace
        planned = planned_work(trace)
        if job.outcome == "completed":
            assert job.counts["rounds_executed"] == planned["planned_rounds"]
            assert job.counts["tool_calls_executed"] == planned["planned_tool_calls"]
            assert job.counts["decode_tokens"] == planned["planned_decode_tokens"]
            assert sum(job.counts[field] for field in (
                "prefill_tokens", "reused_prompt_tokens", "reprefill_tokens",
            )) == planned["planned_prompt_tokens"]
            assert job.times["tool_execute"] == planned["planned_tool_ns"]
        assert job.terminal_at is not None
        assert job.counts["gpu_wait_during_reprefill_ns"] <= job.times["gpu_wait"]
        if context.get("policy") in ("shared", "oracle_fifo", "oracle", "relaxed"):
            assert sum(job.counts[field] for field in FIFO_TIMERS) == (
                job.times["kv_memory_wait"] + job.times["tool_memory_wait"]
            )
        jobs.append({
            **context,
            "job_id": job.id,
            "instance": trace.instance,
            "historical_status": trace.historical_status,
            "arrival_ns": job.arrival.at_ns,
            "terminal_ns": job.terminal_at,
            "outcome": job.outcome,
            "e2e_s": seconds(job.terminal_at - job.arrival.at_ns),
            **planned,
            **{field: job.counts[field] for field in JOB_COUNTS},
            **{f"{field}_s": seconds(job.times[field]) for field in PHASES},
            **{field.removesuffix("_ns") + "_s": seconds(job.counts[field]) for field in FIFO_TIMERS},
            "gpu_wait_during_reprefill_s": seconds(job.counts["gpu_wait_during_reprefill_ns"]),
        })
    makespan = seconds(max(row["terminal_ns"] for row in jobs) - min(row["arrival_ns"] for row in jobs))
    outcomes = Counter(row["outcome"] for row in jobs)
    completed = [row["e2e_s"] for row in jobs if row["outcome"] == "completed"]
    counts = dict(sorted((key, value) for key, value in outcomes.items() if key != "completed"))
    run = {
        **context,
        "submitted": len(jobs),
        "completed": len(completed),
        "completion_rate": len(completed) / len(jobs),
        "makespan_s": makespan,
        "drained_at_s": seconds(result.drained_at_ns),
        "goodput_per_min": 60 * len(completed) / makespan if makespan else None,
        "e2e_p50_s": quantile(completed, 0.5),
        "e2e_p95_s": quantile(completed, 0.95),
        "failure_reasons": json.dumps(counts, sort_keys=True),
        **{field: sum(row[field] for row in jobs) for field in planned_work(result.jobs[0].arrival.trace)},
        **{field: sum(row[field] for row in jobs) for field in JOB_COUNTS},
        **{f"{field}_s": sum(row[f"{field}_s"] for row in jobs) for field in PHASES},
        **{field: sum(row[field] for row in jobs)
           for field in ("fifo_head_wait_s", "fifo_tail_wait_s")},
        "gpu_wait_during_reprefill_s": sum(row["gpu_wait_during_reprefill_s"] for row in jobs),
        **{field.removesuffix("_ns") + "_s": seconds(result.counters.get(field, 0)) for field in RUN_TIMERS},
        **{field: result.counters.get(field, 0) for field in RUN_COUNTS},
        **result.peaks,
    }
    return run, jobs


def execute(
    config: Experiment, traces: tuple[Trace, ...], out: Path, quick: bool,
) -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    policies = [Policy("fixed", share) for share in config.inference_splits]
    policies.extend((
        Policy("shared"),
        Policy("oracle_fifo", oracle_window_s=config.oracle_window_s),
        Policy("oracle", oracle_window_s=config.oracle_window_s),
        Policy("relaxed"),
    ))
    seeds = config.seeds[:1] if quick else config.seeds
    runs, jobs, workloads, profiles = [], [], [], {}
    with (out / "events.jsonl.gz").open("wb") as file, gzip.GzipFile(
        fileobj=file, filename="", mode="wb", compresslevel=3, mtime=0,
    ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8") as events:

        def emit(row: dict) -> None:
            row["case_id"] = case_id
            events.write(json.dumps(row, separators=(",", ":")) + "\n")

        for scenario, profile, levels, stagger in scenarios(config, quick):
            print(f"{scenario} · {profile.dram_gib:g} GiB · concurrency {levels}", file=sys.stderr, flush=True)
            profiles[scenario] = {**asdict(profile), "stagger_max_s": stagger}
            for concurrency in levels:
                for seed in seeds:
                    workload_id = f"{scenario}/n{concurrency}/seed{seed}"
                    arrivals = arrivals_for(traces, concurrency, seed, stagger)
                    workloads.append({
                        "workload_id": workload_id,
                        "arrivals": [
                            {"job_id": a.job_id, "instance": a.trace.instance, "arrival_ns": a.at_ns}
                            for a in arrivals
                        ],
                    })
                    for policy in policies:
                        suffix = f"fixed{policy.inference_percent}" if policy.name == "fixed" else policy.name
                        case_id = f"{workload_id}/{suffix}"
                        context = {
                            "case_id": case_id, "workload_id": workload_id, "scenario": scenario,
                            "concurrency": concurrency, "seed": seed, "policy": policy.name,
                            "inference_percent": policy.inference_percent,
                        }
                        caps = policy.caps(profile.capacity)
                        emit({
                            "at_ns": 0, "event": "case_start", **context,
                            "total_dram_bytes": round(profile.dram_gib * GIB),
                            "static_bytes": round(profile.static_gib * GIB),
                            "dynamic_limit_bytes": None if policy.name == "relaxed" else profile.capacity,
                            "side_limit_bytes": caps,
                            "admission_policy": POLICY_ADMISSION[policy.name],
                            "eviction_policy": POLICY_EVICTION[policy.name],
                            **({"oracle_window_ns": oracle_window_ticks(policy.oracle_window_s)}
                               if policy.uses_oracle else {}),
                        })
                        simulation = Simulator(profile, policy, arrivals, emit).run()
                        run, participants = measure_run(context, simulation)
                        runs.append(run)
                        jobs.extend(participants)
    return runs, jobs, workloads, profiles


def run_experiment(config: Experiment, config_file: Path, out: Path, quick: bool = False) -> None:
    config.validate()
    for _, profile, _, _ in scenarios(config, quick):
        profile.validate()
    traces, source_hashes = load_traces(config.sources)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"status": "running"})
    try:
        write_json(out / "sources.json", {
            "source_sha256": source_hashes,
            "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        })
        run_config = {
            "experiment": asdict(config), "quick": quick,
            "policies": {name: {"admission": admission, "eviction": POLICY_EVICTION[name]}
                         for name, admission in POLICY_ADMISSION.items()},
        }
        write_json(out / "config.json", run_config)
        write_json(out / "traces.json", [asdict(trace) for trace in traces])
        runs, jobs, workloads, profiles = execute(config, traces, out, quick)
        write_json(out / "workloads.json", workloads)
        write_json(out / "config.json", {**run_config, "profiles_by_scenario": profiles})
        write_csv(out / "runs.csv", runs)
        write_csv(out / "agents.csv", jobs)
        comparisons = build_comparisons(runs, jobs)
        write_csv(out / "comparisons.csv", comparisons)
        write_report(out / "report.md", config, traces, comparisons, len(runs), quick)
        write_json(out / "status.json", {
            "status": "completed", "cases": len(runs), "agent_replays": len(jobs),
        })
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
