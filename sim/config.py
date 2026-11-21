"""Configuration shared by the simulator and experiment runner."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

MIB = 1024 ** 2
GIB = 1024 ** 3
TICKS_PER_SECOND = 10 ** 9


def ticks(seconds: float) -> int:
    return round(seconds * TICKS_PER_SECOND)


def seconds(tick_count: int) -> float:
    return tick_count / TICKS_PER_SECOND


@dataclass(frozen=True)
class Sources:
    token_csv: Path
    swe_batch: Path
    instances: tuple[str, ...]

    def validate(self) -> None:
        if not self.instances or len(self.instances) != len(set(self.instances)):
            raise ValueError("sources.instances must contain unique instance IDs")


@dataclass(frozen=True)
class Profile:
    dram_gib: float
    static_gib: float
    kv_bytes_per_token: int
    prefill_tps: tuple[float, ...]
    decode_tps: tuple[float, ...]
    max_batch: int
    prefill_chunk_tokens: int
    kv_evict_s: float
    tool_save_s: float
    tool_write_s: float
    tool_release_s: float
    tool_read_s: float
    tool_restore_s: float
    tool_restore_peak_mib: float
    prefix_fraction: float

    @property
    def capacity(self) -> int:
        return round(self.dram_gib * GIB) - round(self.static_gib * GIB)

    @property
    def restore_peak(self) -> int:
        return round(self.tool_restore_peak_mib * MIB)

    def validate(self) -> None:
        if not all(isfinite(x) and x >= 0 for x in (self.dram_gib, self.static_gib)):
            raise ValueError("DRAM and static memory must be finite and nonnegative")
        if self.capacity <= 0:
            raise ValueError("total DRAM must exceed static memory")
        for value in (self.kv_bytes_per_token, self.max_batch, self.prefill_chunk_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("KV bytes, max batch and prefill chunk must be positive integers")
        for rates in (self.prefill_tps, self.decode_tps):
            if len(rates) < self.max_batch or any(not isfinite(x) or x <= 0 for x in rates):
                raise ValueError("each batch size needs a finite, positive rate")
        costs = (
            self.kv_evict_s, self.tool_save_s, self.tool_write_s, self.tool_release_s,
            self.tool_read_s, self.tool_restore_s, self.tool_restore_peak_mib,
        )
        if any(not isfinite(cost) or cost < 0 for cost in costs):
            raise ValueError("recovery costs and restore memory must be finite and nonnegative")
        if not 0 <= self.prefix_fraction <= 1:
            raise ValueError("prefix_fraction must be in [0, 1]")


@dataclass(frozen=True)
class Experiment:
    sources: Sources
    profile: Profile
    concurrency: tuple[int, ...]
    inference_splits: tuple[int, ...]
    seeds: tuple[int, ...]
    stagger_max_s: float
    sensitivity_concurrency: tuple[int, ...]
    sensitivity_dram_gib: tuple[float, ...]
    sensitivity_prefix_fraction: float
    sensitivity_restore_factors: tuple[float, ...]

    def validate(self) -> None:
        self.sources.validate()
        self.profile.validate()
        for field in ("concurrency", "inference_splits", "seeds", "sensitivity_concurrency"):
            values = getattr(self, field)
            if any(type(value) is not int for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{field} must contain unique integers")
        if not self.concurrency or min(self.concurrency) < 1 or not self.seeds:
            raise ValueError("configure positive concurrency and at least one seed")
        if not self.inference_splits or any(not 0 < x < 100 for x in self.inference_splits):
            raise ValueError("inference_splits must contain percentages in (0, 100)")
        if any(x not in self.concurrency for x in self.sensitivity_concurrency):
            raise ValueError("sensitivity_concurrency must be a subset of concurrency")
        if not isfinite(self.stagger_max_s) or self.stagger_max_s < 0:
            raise ValueError("stagger_max_s must be finite and nonnegative")
        if not 0 <= self.sensitivity_prefix_fraction <= 1:
            raise ValueError("sensitivity_prefix_fraction must be in [0, 1]")
        for values in (self.sensitivity_dram_gib, self.sensitivity_restore_factors):
            if len(set(values)) != len(values) or any(not isfinite(x) or x <= 0 for x in values):
                raise ValueError("sensitivity values must be unique, finite and positive")
