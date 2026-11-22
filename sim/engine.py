"""Single-device discrete-event replay with explicit memory and I/O accounting."""

import heapq
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import chain, count, islice
from math import ceil
from typing import Callable, Literal

from .config import Profile, TICKS_PER_SECOND, ticks
from .trace import Round, Trace

Side = Literal["kv", "tool"]
Queue = Literal["kv", "tool", "global"]
SIDES: tuple[Side, Side] = ("kv", "tool")


@dataclass(frozen=True)
class Policy:
    name: Literal["fixed", "shared", "relaxed"]
    inference_percent: int | None = None

    @property
    def global_queue(self) -> bool:
        return self.name != "fixed"

    def caps(self, dynamic_bytes: int) -> dict[Side, int] | None:
        if self.name == "fixed":
            if self.inference_percent is None or not 0 < self.inference_percent < 100:
                raise ValueError("fixed policy requires an inference percentage in (0, 100)")
            kv = dynamic_bytes * self.inference_percent // 100
            return {"kv": kv, "tool": dynamic_bytes - kv}
        if self.name not in ("shared", "relaxed") or self.inference_percent is not None:
            raise ValueError("invalid shared or relaxed policy")
        return None


@dataclass(frozen=True)
class Arrival:
    job_id: int
    trace: Trace
    at_ns: int = 0


@dataclass
class Slot:
    state: str = "cold"
    resident: int = 0
    reserved: int = 0
    last_used: int = 0


@dataclass
class Job:
    arrival: Arrival
    slots: dict[Side, Slot] = field(default_factory=lambda: {side: Slot() for side in SIDES})
    turn: int = 0
    prefill_left: int = 0
    decode_left: int = 0
    reprefill_tokens: int = 0
    phase: str = ""
    phase_since: int = 0
    terminal_at: int | None = None
    outcome: str = ""
    times: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)

    @property
    def id(self) -> int:
        return self.arrival.job_id

    @property
    def round(self) -> Round:
        return self.arrival.trace.rounds[self.turn]


@dataclass(eq=False)
class Request:
    job: Job
    side: Side
    order: int
    submitted_at: int
    head_at: int | None = None
    last_block: tuple | None = None
    partition_blocked: bool = False


@dataclass
class Reclaim:
    request: Request
    needed: int
    remaining: int
    held: int


@dataclass(frozen=True)
class RunResult:
    jobs: tuple[Job, ...]
    counters: dict
    peaks: dict
    drained_at_ns: int


class Simulator:
    def __init__(
        self,
        profile: Profile,
        policy: Policy,
        arrivals: tuple[Arrival, ...],
        log: Callable[[dict], None] | None = None,
    ):
        profile.validate()
        if not arrivals or len({arrival.job_id for arrival in arrivals}) != len(arrivals):
            raise ValueError("arrivals require distinct job IDs")
        if any(arrival.at_ns < 0 or not arrival.trace.rounds for arrival in arrivals):
            raise ValueError("each arrival needs nonnegative time and at least one round")
        self.profile, self.policy, self.log = profile, policy, log
        self.caps = policy.caps(profile.capacity)
        self.jobs = tuple(Job(arrival) for arrival in arrivals)
        self.now = 0
        self.order = count()
        self.events: list[tuple[int, int, int, Callable, tuple]] = []
        queues = ("global",) if policy.global_queue else SIDES
        self.waiting: dict[Queue, deque[Request]] = {key: deque() for key in queues}
        self.reclaims: dict[Queue, Reclaim] = {}
        self.resident: Counter = Counter()
        self.reserved: Counter = Counter()
        self.held: Counter = Counter()
        self.tool_inflight_bytes = 0
        self.peaks: Counter = Counter()
        self.counters: Counter = Counter()
        self.prefill: deque[Job] = deque()
        self.decode: deque[Job] = deque()
        self.reprefill: deque[Job] = deque()
        self.gpu_batch: tuple[Job, ...] = ()
        self.reprefill_until = 0
        self.last_batch = "decode"
        self.disk_free_at = 0
        for job in sorted(self.jobs, key=lambda job: job.id):
            self.schedule(job.arrival.at_ns, self.submit, job, "kv", priority=1)

    def schedule(self, at: int, callback: Callable, *args, priority: int = 0) -> None:
        assert at >= self.now
        heapq.heappush(self.events, (at, priority, next(self.order), callback, args))

    def after(self, duration: int, callback: Callable, *args) -> None:
        self.schedule(self.now + duration, callback, *args)

    def charged(self, side: Side) -> int:
        return self.resident[side] + self.reserved[side] + self.held[side]

    def free_bytes(self) -> int:
        return self.profile.capacity - sum(self.charged(side) for side in SIDES)

    def check_memory(self) -> None:
        for side in SIDES:
            values = self.resident[side], self.reserved[side], self.held[side]
            assert min(values) >= 0
        kv, tool = self.charged("kv"), self.charged("tool")
        actual = self.resident["kv"] + self.resident["tool"]
        if self.policy.name != "relaxed":
            assert kv + tool <= self.profile.capacity
        if self.caps is not None:
            assert kv <= self.caps["kv"] and tool <= self.caps["tool"]
        self.peaks["charged_bytes"] = max(self.peaks["charged_bytes"], kv + tool)
        self.peaks["resident_bytes"] = max(self.peaks["resident_bytes"], actual)
        self.peaks["kv_charged_bytes"] = max(self.peaks["kv_charged_bytes"], kv)
        self.peaks["tool_charged_bytes"] = max(self.peaks["tool_charged_bytes"], tool)

    def record(self, event: str, job: Job | None = None, **fields) -> None:
        self.check_memory()
        if self.log is None:
            return
        row = {"at_ns": self.now, "event": event, **fields}
        if job is not None:
            row.update(job_id=job.id, turn=job.turn + 1)
        for side in SIDES:
            row[f"{side}_resident_bytes"] = self.resident[side]
            row[f"{side}_reserved_bytes"] = self.reserved[side]
            row[f"{side}_held_bytes"] = self.held[side]
        row["kv_executing_bytes"] = sum(job.slots["kv"].resident for job in self.gpu_batch)
        row["tool_inflight_bytes"] = self.tool_inflight_bytes
        self.log(row)

    def set_slot(self, job: Job, side: Side, state: str, resident: int, reserved: int = 0) -> None:
        slot = job.slots[side]
        assert resident >= 0 and reserved >= 0
        if side == "tool":
            if slot.state in ("active", "restoring"):
                self.tool_inflight_bytes -= slot.resident
            if state in ("active", "restoring"):
                self.tool_inflight_bytes += resident
        self.resident[side] += resident - slot.resident
        self.reserved[side] += reserved - slot.reserved
        slot.resident, slot.reserved, slot.state = resident, reserved, state
        self.check_memory()

    def phase(self, job: Job, phase: str) -> None:
        if job.phase:
            job.times[job.phase] += self.now - job.phase_since
        if phase == "gpu_wait" and self.now < self.reprefill_until:
            job.counts["gpu_wait_during_reprefill_ns"] += self.reprefill_until - self.now
        job.phase = phase
        job.phase_since = self.now

    def queue_for(self, side: Side) -> Queue:
        return "global" if self.policy.global_queue else side

    def submit(self, job: Job, side: Side) -> None:
        self.phase(job, f"{side}_memory_wait")
        request = Request(job, side, next(self.order), self.now)
        self.waiting[self.queue_for(side)].append(request)
        self.record("submit", job, side=side, request_order=request.order)

    def terminal(self, job: Job, outcome: str) -> None:
        self.phase(job, "")
        job.terminal_at = self.now
        job.outcome = outcome
        assert sum(job.times.values()) == self.now - job.arrival.at_ns
        for side in SIDES:
            if job.slots[side].state != "evicting":
                self.set_slot(job, side, "closed", 0)
        self.record("terminal", job, outcome=outcome)

    def target(self, request: Request) -> int:
        job, side = request.job, request.side
        if side == "kv":
            return (job.round.prompt + job.round.output) * self.profile.kv_bytes_per_token
        trace = job.arrival.trace
        if job.slots["tool"].state in ("evicted", "evicting"):
            return max(trace.tool_peak, trace.tool_resident, self.profile.restore_peak)
        return trace.tool_peak

    def admit(self) -> None:
        if self.policy.global_queue:
            queue = self.waiting["global"]
            while queue:
                head = queue[0]
                self.mark_head(head)
                reclaim = self.reclaims.get("global")
                # A settled transaction has already reserved enough for its head.
                if reclaim is None or reclaim.remaining:
                    if self.admit_resident_hit(queue):
                        continue
                if not self.try_admit(head):
                    return
            return

        # Fixed partitions have independent queues and can make progress separately.
        while True:
            heads = sorted(
                (queue[0] for queue in self.waiting.values() if queue),
                key=lambda request: request.order,
            )
            progressed = False
            for request in heads:
                if self.try_admit(request):
                    progressed = True
            if not progressed:
                return

    def mark_head(self, request: Request) -> None:
        if request.head_at is None:
            request.head_at = self.now
            self.record("queue_head", request.job, side=request.side, request_order=request.order)

    def request_wait(self, request: Request) -> dict:
        head_time = self.now if request.head_at is None else request.head_at
        tail_wait, head_wait = head_time - request.submitted_at, self.now - head_time
        if self.policy.global_queue:
            request.job.counts["fifo_tail_wait_ns"] += tail_wait
            request.job.counts["fifo_head_wait_ns"] += head_wait
        return {"request_order": request.order, "head_at_ns": request.head_at,
                "tail_wait_ns": tail_wait, "head_wait_ns": head_wait}

    def prefix_tokens(self, job: Job) -> int:
        return int(job.round.prefix * self.profile.prefix_fraction)

    def admit_resident_hit(self, queue: deque[Request]) -> bool:
        free = self.free_bytes()
        for request in islice(queue, 1, None):
            job, side = request.job, request.side
            slot = job.slots[side]
            if slot.state != "idle" or not slot.resident or (side == "kv" and not self.prefix_tokens(job)):
                continue
            target = self.target(request)
            if self.policy.name == "relaxed" or target - slot.resident <= free:
                # Existing reclaim holds are charged against free space.
                self.start_request(request, target, bypass=True)
                return True
        return False

    def try_admit(self, request: Request) -> bool:
        job, side = request.job, request.side
        key = self.queue_for(side)
        self.mark_head(request)
        slot = job.slots[side]
        if key in self.reclaims:
            reclaim = self.reclaims[key]
            assert reclaim.request is request
            if reclaim.remaining:
                return False
            assert reclaim.held == reclaim.needed
            self.held[side] -= reclaim.held
            del self.reclaims[key]
        target = self.target(request)
        if self.policy.name == "relaxed":
            self.start_request(request, target)
            return True

        cap = self.caps[side] if self.caps is not None else self.profile.capacity
        total_free = self.free_bytes()
        side_free = cap - self.charged(side) if self.caps is not None else total_free
        needed = max(0, target - slot.resident)
        if target > cap:
            self.waiting[key].popleft()
            stranded = self.caps is not None and needed <= total_free
            if stranded:
                job.counts[f"{side}_partition_blocked_requests"] += 1
                self.counters["max_partition_shortfall_bytes"] = max(
                    self.counters["max_partition_shortfall_bytes"], needed - side_free,
                )
            self.record(
                "over_cap", job, side=side, target_bytes=target, cap_bytes=cap,
                needed_bytes=needed, side_free_bytes=side_free, total_free_bytes=total_free,
                partition_stranded=stranded, **self.request_wait(request),
            )
            self.terminal(job, f"{side}_allocation_exceeds_cap")
            return True
        if slot.state != "evicting" and needed <= side_free:
            self.start_request(request, target)
            return True

        block_kind = "own_eviction" if slot.state == "evicting" else "capacity"
        block = (block_kind, needed, side_free, total_free)
        if block != request.last_block:
            request.last_block = block
            stranded = self.caps is not None and block_kind == "capacity" and side_free < needed <= total_free
            if stranded and not request.partition_blocked:
                request.partition_blocked = True
                job.counts[f"{side}_partition_blocked_requests"] += 1
            if stranded:
                self.counters["max_partition_shortfall_bytes"] = max(
                    self.counters["max_partition_shortfall_bytes"], needed - side_free,
                )
            self.record(
                "memory_blocked", job, side=side, reason=block_kind,
                needed_bytes=needed, side_free_bytes=side_free, total_free_bytes=total_free,
                partition_stranded=stranded, request_order=request.order,
            )
        if slot.state == "evicting":
            return False

        eligible_sides = SIDES if self.policy.global_queue else (side,)
        # Sorting is stable, so equal per-agent timestamps use SIDES order.
        candidates = sorted(
            ((other, victim_side) for other in self.jobs for victim_side in eligible_sides
             if (other is not job or victim_side != side)
             and other.slots[victim_side].state == "idle" and other.slots[victim_side].resident),
            key=lambda candidate: (candidate[0].slots[candidate[1]].last_used, candidate[0].id),
        )
        selected = []
        recoverable = side_free
        for victim, victim_side in candidates:
            selected.append((victim, victim_side))
            recoverable += victim.slots[victim_side].resident
            if recoverable >= needed:
                break
        if recoverable < needed:
            return False

        reclaim = Reclaim(request, needed, len(selected), side_free)
        self.reclaims[key] = reclaim
        self.held[side] += reclaim.held
        self.record(
            "reclaim_start", job, side=side, request_order=request.order, needed_bytes=needed,
            victim_ids=[victim.id for victim, _ in selected],
            victims=[{"job_id": victim.id, "side": victim_side} for victim, victim_side in selected],
        )
        for victim, victim_side in selected:
            self.start_eviction(victim, victim_side, reclaim)
        return True

    def start_request(self, request: Request, target: int, *, bypass: bool = False) -> None:
        job, side = request.job, request.side
        queue = self.waiting[self.queue_for(side)]
        if bypass:
            assert queue[0] is not request
            queue.remove(request)
            job.counts["resident_bypass_admits"] += 1
        else:
            head = queue.popleft()
            assert head is request
        slot = job.slots[side]
        old_state, old_resident = slot.state, slot.resident
        self.record(
            "admit", job, side=side, prior_state=old_state, target_bytes=target,
            reservation_delta_bytes=target - old_resident, resident_bypass=bypass,
            **self.request_wait(request),
        )
        if side == "kv":
            self.start_inference(job, target, old_state, old_resident)
        elif old_state == "evicted":
            self.start_restore(job, target)
        else:
            self.start_tool(job)

    def start_eviction(self, job: Job, side: Side, reclaim: Reclaim) -> None:
        slot = job.slots[side]
        slot.state = "evicting"
        job.counts[f"{side}_evictions"] += 1
        self.record("evict_start", job, side=side, bytes_to_free=slot.resident,
                    requester_side=reclaim.request.side, request_order=reclaim.request.order)
        if side == "kv":
            self.after(ticks(self.profile.kv_evict_s), self.finish_eviction, job, side, reclaim, self.now)
        else:
            self.after(ticks(self.profile.tool_save_s), self.saved, job, reclaim, self.now)

    def saved(self, job: Job, reclaim: Reclaim, started: int) -> None:
        self.record("checkpoint_saved", job)
        self.disk(job, "write", self.profile.tool_write_s, self.written, job, reclaim, started)

    def written(self, job: Job, reclaim: Reclaim, started: int) -> None:
        self.after(ticks(self.profile.tool_release_s), self.finish_eviction, job, "tool", reclaim, started)

    def finish_eviction(self, job: Job, side: Side, reclaim: Reclaim, started: int) -> None:
        released = job.slots[side].resident
        self.set_slot(job, side, "evicted", 0)
        hold = min(released, reclaim.needed - reclaim.held)
        reclaim.held += hold
        reclaim.remaining -= 1
        requester_side = reclaim.request.side
        self.held[requester_side] += hold
        self.counters[f"{side}_eviction_ns"] += self.now - started
        if requester_side != side:
            self.counters[f"{requester_side}_request_{side}_evictions"] += 1
        self.record(
            "evict_finish", job, side=side, released_bytes=released,
            duration_ns=self.now - started, requester_id=reclaim.request.job.id,
            requester_side=requester_side, request_order=reclaim.request.order,
        )

    def disk(self, job: Job, direction: str, duration_s: float, after: Callable, *args) -> None:
        start = max(self.now, self.disk_free_at)
        duration = ticks(duration_s)
        self.disk_free_at = start + duration
        self.counters[f"checkpoint_{direction}_ns"] += duration
        self.counters["checkpoint_disk_queue_ns"] += start - self.now
        read_memory = {"restore_resident_bytes": job.slots["tool"].resident} if direction == "read" else {}
        self.record(
            "checkpoint_io_enqueue", job, direction=direction, start_ns=start,
            end_ns=self.disk_free_at, **read_memory,
        )
        self.schedule(self.disk_free_at, self.finish_disk, job, direction, after, args)

    def finish_disk(self, job: Job, direction: str, after: Callable, args: tuple) -> None:
        self.record("checkpoint_io_finish", job, direction=direction)
        after(*args)

    def start_restore(self, job: Job, target: int) -> None:
        resident = max(job.arrival.trace.tool_resident, self.profile.restore_peak)
        self.set_slot(job, "tool", "restoring", resident, target - resident)
        self.phase(job, "tool_restore")
        job.counts["tool_restores"] += 1
        self.record("tool_restore_start", job)
        self.disk(job, "read", self.profile.tool_read_s, self.read, job)

    def read(self, job: Job) -> None:
        self.after(ticks(self.profile.tool_restore_s), self.finish_restore, job)

    def finish_restore(self, job: Job) -> None:
        self.record("tool_restore_finish", job, duration_ns=self.now - job.phase_since)
        self.start_tool(job)

    def start_tool(self, job: Job) -> None:
        self.set_slot(job, "tool", "active", job.arrival.trace.tool_peak)
        self.phase(job, "tool_execute")
        self.record("tool_start", job, duration_ns=job.round.tool_ns, call_count=len(job.round.tools))
        self.after(job.round.tool_ns, self.finish_tool, job)

    def finish_tool(self, job: Job) -> None:
        self.set_slot(job, "tool", "idle", job.arrival.trace.tool_resident)
        job.slots["tool"].last_used = self.now
        job.counts["tool_calls_executed"] += len(job.round.tools)
        self.record("tool_finish", job)
        self.advance(job)

    def start_inference(self, job: Job, target: int, old_state: str, old_resident: int) -> None:
        prefix = self.prefix_tokens(job)
        reused = prefix if old_state == "idle" else 0
        job.reprefill_tokens = prefix if old_state == "evicted" else 0
        job.counts["reused_prompt_tokens"] += reused
        assert reused * self.profile.kv_bytes_per_token <= old_resident
        resident = reused * self.profile.kv_bytes_per_token
        self.set_slot(job, "kv", "active", resident, target - resident)
        job.prefill_left = job.round.prompt - reused - job.reprefill_tokens
        job.decode_left = job.round.output
        self.record("kv_prepare", job, reuse_tokens=reused, restore_tokens=job.reprefill_tokens,
                    new_prefill_tokens=job.prefill_left)
        if job.reprefill_tokens:
            self.phase(job, "gpu_wait")
            self.reprefill.append(job)
        else:
            self.next_gpu_phase(job)

    def next_gpu_phase(self, job: Job) -> None:
        if job.prefill_left:
            self.phase(job, "gpu_wait")
            self.prefill.append(job)
        elif job.decode_left:
            self.phase(job, "gpu_wait")
            self.decode.append(job)
        else:
            self.finish_inference(job)

    def grow_kv(self, job: Job, tokens: int) -> None:
        slot = job.slots["kv"]
        delta = tokens * self.profile.kv_bytes_per_token
        self.set_slot(job, "kv", "active", slot.resident + delta, slot.reserved - delta)

    def finish_inference(self, job: Job) -> None:
        slot = job.slots["kv"]
        assert slot.reserved == 0
        self.set_slot(job, "kv", "idle", slot.resident)
        slot.last_used = self.now
        job.counts["rounds_executed"] += 1
        self.record("inference_finish", job)
        if job.round.tools or job.round.tool_ns:
            self.submit(job, "tool")
        else:
            self.advance(job)

    def advance(self, job: Job) -> None:
        if job.turn + 1 == len(job.arrival.trace.rounds):
            self.terminal(job, "completed")
        else:
            job.turn += 1
            self.submit(job, "kv")

    def dispatch(self) -> None:
        if self.gpu_batch:
            return
        if self.reprefill:
            job = self.reprefill.popleft()
            tokens = job.reprefill_tokens
            duration = max(1, ceil(TICKS_PER_SECOND * tokens / self.profile.prefill_tps[0]))
            self.gpu_batch = (job,)
            self.reprefill_until = self.now + duration
            for pending in chain(self.prefill, self.decode, self.reprefill):
                pending.counts["gpu_wait_during_reprefill_ns"] += duration
            self.phase(job, "reprefill")
            self.counters["reprefill_gpu_ns"] += duration
            self.record("reprefill_start", job, tokens=tokens, duration_ns=duration)
            self.after(duration, self.finish_reprefill, job)
        elif self.prefill and (not self.decode or self.last_batch == "decode"):
            self.start_batch("prefill")
        elif self.decode:
            self.start_batch("decode")

    def finish_reprefill(self, job: Job) -> None:
        self.gpu_batch = ()
        self.grow_kv(job, job.reprefill_tokens)
        job.counts["reprefill_tokens"] += job.reprefill_tokens
        self.record("reprefill_finish", job, tokens=job.reprefill_tokens)
        job.reprefill_tokens = 0
        self.next_gpu_phase(job)

    def start_batch(self, kind: str) -> None:
        queue = self.prefill if kind == "prefill" else self.decode
        batch = tuple(islice(queue, self.profile.max_batch))
        for job in batch:
            head = queue.popleft()
            assert head is job
            self.phase(job, kind)
        size = len(batch)
        if kind == "prefill":
            tokens = tuple(min(job.prefill_left, self.profile.prefill_chunk_tokens) for job in batch)
            duration = max(1, ceil(TICKS_PER_SECOND * sum(tokens) / self.profile.prefill_tps[size - 1]))
            steps = 1
        else:
            step_ns = max(1, ceil(TICKS_PER_SECOND * size / self.profile.decode_tps[size - 1]))
            steps = min(job.decode_left for job in batch)
            if self.prefill or self.decode:
                steps = 1
            elif self.events:
                # Slice identical decode steps without passing an external event
                # by more than the first decode step already in flight.
                steps = min(steps, max(1, (self.events[0][0] - self.now) // step_ns))
            duration = steps * step_ns
            tokens = (steps,) * size
        self.gpu_batch = batch
        self.last_batch = kind
        self.counters[f"{kind}_batches"] += steps
        self.counters[f"{kind}_gpu_ns"] += duration
        self.counters[f"max_{kind}_batch"] = max(self.counters[f"max_{kind}_batch"], size)
        self.record("batch_start", batch=kind, job_ids=[job.id for job in batch],
                    tokens=list(tokens), steps=steps, duration_ns=duration)
        self.after(duration, self.finish_batch, kind, batch, tokens)

    def finish_batch(self, kind: str, batch: tuple[Job, ...], tokens: tuple[int, ...]) -> None:
        self.gpu_batch = ()
        for job, amount in zip(batch, tokens):
            self.grow_kv(job, amount)
            if kind == "prefill":
                job.prefill_left -= amount
                job.counts["prefill_tokens"] += amount
            else:
                job.decode_left -= amount
                job.counts["decode_tokens"] += amount
        self.record("batch_finish", batch=kind, job_ids=[job.id for job in batch], tokens=list(tokens))
        for job in batch:
            self.next_gpu_phase(job)

    def run(self) -> RunResult:
        while True:
            while self.events and self.events[0][0] == self.now:
                _, _, _, callback, args = heapq.heappop(self.events)
                callback(*args)
            self.admit()
            if self.events and self.events[0][0] == self.now:
                continue
            self.dispatch()
            if not self.events:
                break
            self.now = self.events[0][0]

        blocked = [request for queue in self.waiting.values() for request in queue]
        assert not self.reclaims and not (self.prefill or self.decode or self.reprefill or self.gpu_batch)
        assert {request.job.id for request in blocked} == {
            job.id for job in self.jobs if job.terminal_at is None
        }
        # Diagnose every blocked head before terminal cleanup frees any memory.
        for request in blocked:
            self.record(
                "no_progress", request.job, side=request.side, target_bytes=self.target(request),
                last_block=request.last_block,
                queue_head_job_id=self.waiting[self.queue_for(request.side)][0].job.id,
                **self.request_wait(request),
            )
        for request in blocked:
            self.terminal(request.job, f"{request.side}_no_progress")
        assert all(self.charged(side) == 0 for side in SIDES)
        self.record("drained")
        return RunResult(self.jobs, dict(self.counters), dict(self.peaks), self.now)
