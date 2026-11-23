"""Render physical and executing memory timelines from simulation event logs."""

import csv
import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import GIB, ticks

MINUTE_NS = ticks(60)


@dataclass(frozen=True)
class MemoryPoint:
    at_ns: int
    kv_resident: int
    tool_resident: int
    kv_executing: int | None
    tool_executing: int | None

    @property
    def resident(self) -> int:
        return self.kv_resident + self.tool_resident

    @property
    def executing(self) -> int | None:
        if self.kv_executing is None or self.tool_executing is None:
            return None
        return self.kv_executing + self.tool_executing


@dataclass(frozen=True)
class MinuteMemory:
    start_ns: int
    end_ns: int
    resident_byte_ns: int
    executing_byte_ns: int | None

    @property
    def resident_bytes(self) -> float:
        return self.resident_byte_ns / (self.end_ns - self.start_ns)

    @property
    def executing_bytes(self) -> float | None:
        if self.executing_byte_ns is None:
            return None
        return self.executing_byte_ns / (self.end_ns - self.start_ns)

    @property
    def executing_share(self) -> float | None:
        if self.executing_byte_ns is None or not self.resident_byte_ns:
            return None
        return self.executing_byte_ns / self.resident_byte_ns


@dataclass(frozen=True)
class Timeline:
    case_id: str
    budget: int
    points: tuple[MemoryPoint, ...]

    def mean_utilization(self, *, active: bool = False) -> float | None:
        duration = self.points[-1].at_ns - self.points[0].at_ns
        if not duration or (active and self.points[0].executing is None):
            return None
        byte_ns = sum(
            (point.executing if active else point.resident) * (following.at_ns - point.at_ns)
            for point, following in zip(self.points, self.points[1:])
        )
        return byte_ns / (duration * self.budget)

    def minute_averages(self) -> list[MinuteMemory]:
        averages = []
        index = 0
        stop_ns = self.points[-1].at_ns
        have_executing = self.points[0].executing is not None
        for start_ns in range(self.points[0].at_ns, stop_ns, MINUTE_NS):
            end_ns = min(start_ns + MINUTE_NS, stop_ns)
            resident = executing = 0
            while self.points[index].at_ns < end_ns:
                point, following = self.points[index], self.points[index + 1]
                duration = min(following.at_ns, end_ns) - max(point.at_ns, start_ns)
                resident += point.resident * duration
                if have_executing:
                    executing += point.executing * duration
                if following.at_ns > end_ns:
                    break
                index += 1
            averages.append(MinuteMemory(start_ns, end_ns, resident, executing if have_executing else None))
        return averages


@dataclass
class TimelineBuilder:
    case_id: str
    budget: int
    resident_only: bool
    snapshots: dict[int, MemoryPoint] = field(default_factory=dict)
    queued_read_deltas: Counter = field(default_factory=Counter)
    last_ns: int = 0

    def add_snapshot(self, row: dict) -> None:
        at = row["at_ns"]
        if at < self.last_ns:
            raise ValueError(f"{self.case_id}: event time moved backwards")
        self.last_ns = at
        kv = tool = None
        if not self.resident_only:
            try:
                kv = row["kv_executing_bytes"]
                tool = row["tool_inflight_bytes"]
                if row["event"] == "checkpoint_io_enqueue" and row["direction"] == "read":
                    start = row["start_ns"]
                    if start > at:
                        restore_bytes = row["restore_resident_bytes"]
                        self.queued_read_deltas[at] += restore_bytes
                        self.queued_read_deltas[start] -= restore_bytes
            except KeyError as error:
                raise ValueError(
                    f"{self.case_id}: execution snapshots are missing {error}. "
                    "Use --resident-only or replay with the current simulator."
                ) from error
        self.snapshots[at] = MemoryPoint(at, row["kv_resident_bytes"], row["tool_resident_bytes"], kv, tool)

    def finish(self) -> Timeline:
        if self.queued_read_deltas and max(self.queued_read_deltas) > self.last_ns:
            raise ValueError(f"{self.case_id}: a checkpoint read starts after the run drained")
        points = []
        waiting_bytes = 0
        snapshot = self.snapshots[0]
        for at in sorted(self.snapshots.keys() | self.queued_read_deltas.keys()):
            snapshot = self.snapshots.get(at, snapshot)
            waiting_bytes += self.queued_read_deltas[at]
            tool = snapshot.tool_executing
            if tool is not None:
                tool -= waiting_bytes
            point = MemoryPoint(at, snapshot.kv_resident, snapshot.tool_resident, snapshot.kv_executing, tool)
            if (waiting_bytes < 0 or min(point.kv_resident, point.tool_resident) < 0
                    or not 0 <= (point.kv_executing or 0) <= point.kv_resident
                    or not 0 <= (point.tool_executing or 0) <= point.tool_resident):
                raise ValueError(f"{self.case_id}: inconsistent memory snapshot at {at} ns")
            points.append(point)
        if waiting_bytes or points[-1].resident:
            raise ValueError(f"{self.case_id}: memory remains after the run drained")
        return Timeline(self.case_id, self.budget, tuple(points))


def read_timelines(log: Path, case_ids: list[str], resident_only: bool = False) -> list[Timeline]:
    wanted = set(case_ids)
    if not wanted or len(wanted) != len(case_ids):
        raise ValueError("select one or more distinct --case values")
    builders = {}
    selected = {}
    with gzip.open(log, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case_id = row["case_id"]
            if case_id not in wanted:
                continue
            if row["event"] == "case_start":
                budget = row["total_dram_bytes"] - row["static_bytes"]
                if budget <= 0 or case_id in builders:
                    raise ValueError(f"{case_id}: invalid case header")
                empty = None if resident_only else 0
                builders[case_id] = TimelineBuilder(
                    case_id, budget, resident_only, {0: MemoryPoint(0, 0, 0, empty, empty)},
                )
                continue
            builder = builders.get(case_id)
            if builder is None:
                raise ValueError(f"{case_id}: missing case header")
            builder.add_snapshot(row)
            if row["event"] == "drained":
                selected[case_id] = builder.finish()
                del builders[case_id]
                wanted.remove(case_id)
                if not wanted:
                    break
    if builders:
        raise ValueError(f"selected cases did not drain: {', '.join(sorted(builders))}")
    if wanted:
        raise ValueError(f"unknown case IDs: {', '.join(sorted(wanted))}")
    return [selected[case_id] for case_id in case_ids]


def write_timeline_csv(path: Path, timelines: list[Timeline]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "case_id", "at_ns", "kv_resident_bytes", "tool_resident_bytes", "resident_bytes",
            "kv_executing_bytes", "tool_executing_bytes", "executing_bytes", "dynamic_budget_bytes",
        ))
        for timeline in timelines:
            for point in timeline.points:
                writer.writerow((
                    timeline.case_id, point.at_ns, point.kv_resident, point.tool_resident, point.resident,
                    point.kv_executing, point.tool_executing, point.executing, timeline.budget,
                ))


def write_minute_csv(path: Path, timelines: list[Timeline], minute_series: list[list[MinuteMemory]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "case_id", "start_ns", "end_ns", "mean_resident_bytes", "mean_executing_bytes",
            "executing_share_of_resident", "dynamic_budget_bytes",
        ))
        for timeline, minutes in zip(timelines, minute_series):
            for minute in minutes:
                writer.writerow((
                    timeline.case_id, minute.start_ns, minute.end_ns, minute.resident_bytes,
                    minute.executing_bytes, minute.executing_share, timeline.budget,
                ))


def plot_memory(run: Path, case_ids: list[str], out: Path, resident_only: bool = False) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ValueError("install the plotting dependencies from requirements-viz.txt") from error

    timelines = read_timelines(run / "events.jsonl.gz", case_ids, resident_only)
    minute_series = [timeline.minute_averages() for timeline in timelines]
    out.mkdir(parents=True, exist_ok=False)
    figure = Figure(figsize=(12, 1.4 + 2.9 * len(timelines)), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(len(timelines) + 1, 1, height_ratios=[0.1] + [1] * len(timelines))
    header = figure.add_subplot(grid[0])
    header.set_axis_off()
    axes = [figure.add_subplot(grid[1])]
    for index in range(2, len(timelines) + 1):
        axes.append(figure.add_subplot(grid[index], sharex=axes[0], sharey=axes[0]))
    title = (
        "Physical memory by simulated minute" if resident_only
        else "Memory use by simulated minute"
    )
    figure.suptitle(title, fontsize=16, fontweight="bold")
    budgets = {timeline.budget for timeline in timelines}
    budget_label = "Dynamic budget"
    if len(budgets) == 1:
        budget_label += f" ({timelines[0].budget / GIB:g} GiB)"
    for axis, timeline, minutes in zip(axes, timelines, minute_series):
        time = [(minute.start_ns + minute.end_ns) / (2 * MINUTE_NS) for minute in minutes]
        resident = [minute.resident_bytes / GIB for minute in minutes]
        marker = "o" if len(minutes) == 1 else None
        axis.plot(time, resident, color="#17476a", linewidth=1.5, marker=marker, label="Physical resident")
        if not resident_only:
            executing = [minute.executing_bytes / GIB for minute in minutes]
            axis.fill_between(time, executing, resident, color="#d5e4ed", linewidth=0)
            axis.plot(time, executing, color="#af451e", linewidth=1.4, marker=marker, label="Executing")
        axis.axhline(timeline.budget / GIB, color="#6b7280", linestyle="--", linewidth=1,
                     label=budget_label)
        mean = timeline.mean_utilization()
        summary = "instantaneous run" if mean is None else f"mean resident / budget {mean:.1%}"
        mean_active = timeline.mean_utilization(active=True)
        if mean_active is not None:
            summary += f"   |   mean active / budget {mean_active:.1%}"
        if len(budgets) > 1:
            summary += f" ({timeline.budget / GIB:g} GiB budget)"
        axis.set_title(f"{timeline.case_id}   |   {summary}", loc="left", fontsize=10)
        axis.set_ylabel("Mean memory (GiB)")
        axis.grid(axis="y", color="#e5e7eb")
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    header.legend(handles, labels, loc="center", frameon=False, fontsize=9, ncols=len(handles))
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    end = max(timeline.points[-1].at_ns for timeline in timelines)
    axes[-1].set_xlim(0, end / MINUTE_NS if end else 1 / 60)
    axes[-1].set_ylim(bottom=0)
    axes[-1].set_xlabel("Simulated time (min)")
    figure.savefig(out / "memory_usage.png", dpi=160)
    write_timeline_csv(out / "memory_timeline.csv", timelines)
    write_minute_csv(out / "memory_minute.csv", timelines, minute_series)
