"""Load the numeric fields needed for replay, without reading tool commands."""

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import median

from .config import MIB, Sources, ticks


@dataclass(frozen=True)
class ToolCall:
    name: str
    action_id: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class Round:
    prompt: int
    output: int
    prefix: int = 0
    tool_ns: int = 0
    tools: tuple[ToolCall, ...] = ()
    response_id: str = ""


@dataclass(frozen=True)
class Trace:
    instance: str
    rounds: tuple[Round, ...]
    tool_resident: int = 0
    tool_peak: int = 0
    historical_status: str = ""
    idle_samples: int = 0
    active_samples: int = 0
    uncovered_tools: int = 0
    excluded_api_calls: int = 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merge_windows(calls: list[ToolCall]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((call.start_ns, call.end_ns) for call in calls):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def parse_calls(
    rows: list[dict[str, str]], response_ids: set[str]
) -> dict[str, list[ToolCall]]:
    calls = defaultdict(list)
    action_ids = set()
    for row in rows:
        if row["tool_name"] == "finish":
            continue
        start, end = ticks(float(row["start_s"])), ticks(float(row["end_s"]))
        response_id = row["llm_response_id"]
        if row["paired"] != "True" or response_id not in response_ids:
            raise ValueError(f"unpaired tool action {row['action_id']}")
        if end < start or row["action_id"] in action_ids:
            raise ValueError(f"invalid tool interval or duplicate action {row['action_id']}")
        action_ids.add(row["action_id"])
        calls[response_id].append(ToolCall(row["tool_name"], row["action_id"], start, end))
    return {key: sorted(values, key=lambda call: call.start_ns) for key, values in calls.items()}


def tool_footprint(
    samples: list[dict[str, str]], calls: list[ToolCall]
) -> tuple[int, int, int, int, int]:
    if not calls:
        return 0, 0, 0, 0, 0
    windows = merge_windows(calls)
    active, idle, timestamps = [], [], []
    for row in samples:
        if row["sample_ok"] != "1" or row["active_containers"] != "1":
            continue
        at = ticks(float(row["elapsed_s"]))
        if not windows[0][0] <= at < windows[-1][1]:
            continue
        mib = float(row["container_memory_mib"])
        if not isfinite(mib) or mib < 0:
            raise ValueError("invalid container memory sample")
        memory = round(mib * MIB)
        timestamps.append(at)
        if any(start <= at < end for start, end in windows):
            active.append(memory)
        else:
            idle.append(memory)
    if not active or not idle:
        raise ValueError("tool footprint needs both active and idle samples")
    resident, peak = round(median(idle)), max(active)
    if peak < resident:
        raise ValueError("sampled tool peak is below the resident estimate")
    uncovered = sum(not any(call.start_ns <= at < call.end_ns for at in timestamps) for call in calls)
    return resident, peak, len(idle), len(active), uncovered


def parse_trace(
    instance: str,
    token_rows: list[dict[str, str]],
    tool_rows: list[dict[str, str]],
    samples: list[dict[str, str]],
    status: str,
) -> Trace:
    linked = sorted(
        (row for row in token_rows if row["linked_to_dialogue"] == "True"),
        key=lambda row: int(row["dialogue_turn"]),
    )
    if not linked or [int(row["dialogue_turn"]) for row in linked] != list(range(1, len(linked) + 1)):
        raise ValueError(f"{instance}: missing or unordered dialogue turns")
    response_ids = {row["response_id"] for row in linked}
    if "" in response_ids or len(response_ids) != len(linked):
        raise ValueError(f"{instance}: invalid response IDs")
    by_response = parse_calls(tool_rows, response_ids)
    rounds = []
    previous_context = 0
    last_tool_end = float("-inf")
    for row in linked:
        prompt = int(row["prompt_tokens"])
        output = int(row["completion_tokens"])
        cache_read = int(row["cache_read_tokens"])
        if min(prompt, output, cache_read) < 0 or cache_read > prompt:
            raise ValueError(f"{instance}: invalid token counts")
        tools = by_response.get(row["response_id"], [])
        if tools and tools[0].start_ns < last_tool_end:
            raise ValueError(f"{instance}: overlapping tool windows across inference rounds")
        windows = merge_windows(tools)
        tool_ns = sum(end - start for start, end in windows)
        if windows:
            last_tool_end = windows[-1][1]
        rounds.append(Round(
            prompt, output, min(cache_read, previous_context, prompt),
            tool_ns, tuple(tools), row["response_id"],
        ))
        previous_context = prompt + output
    calls = [call for round_ in rounds for call in round_.tools]
    resident, peak, idle, active, uncovered = tool_footprint(samples, calls)
    return Trace(
        instance, tuple(rounds), resident, peak, status,
        idle, active, uncovered, len(token_rows) - len(linked),
    )


def load_traces(sources: Sources) -> tuple[tuple[Trace, ...], dict[str, str]]:
    token_rows = read_rows(sources.token_csv)
    files = [sources.token_csv]
    by_instance = defaultdict(list)
    for row in token_rows:
        by_instance[row["instance_id"]].append(row)
    traces = []
    for instance in sources.instances:
        path = sources.swe_batch / instance
        tools, resources, summary = (path / name for name in
                                     ("tool_calls.csv", "resources.csv", "summary.json"))
        files.extend((tools, resources, summary))
        metadata = json.loads(summary.read_text(encoding="utf-8"))
        status = "historical_error" if metadata["execution_errors"] else "finished"
        traces.append(parse_trace(
            instance, by_instance[instance], read_rows(tools), read_rows(resources), status,
        ))
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return tuple(traces), hashes
