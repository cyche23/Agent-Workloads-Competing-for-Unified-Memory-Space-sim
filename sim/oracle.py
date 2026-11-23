"""Bounded counterfactual replay for same-budget memory admission."""

from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from math import ceil

from .config import Profile, TICKS_PER_SECOND, oracle_window_ticks, ticks
from .engine import SIDES, Arrival, Job, Policy, Request, Side, Simulator

REQUEST_LIMIT = 8
Victim = tuple[Job, Side]


@dataclass(frozen=True)
class Action:
    request: Request | None = None
    victims: tuple[Victim, ...] = ()
    resident_hit: bool = False

    @property
    def kind(self) -> str:
        if self.request is None:
            return "wait"
        return "reclaim" if self.victims else "admit"

    @property
    def identity(self) -> tuple:
        order = None if self.request is None else self.request.order
        return order, tuple(sorted((job.id, side) for job, side in self.victims))

    @property
    def release_bytes(self) -> int:
        return sum(job.slots[side].resident for job, side in self.victims)


@dataclass(frozen=True)
class Score:
    failures: int
    unfinished_ns: int
    remaining_service_ns: int

    @property
    def objective(self) -> tuple[int, int]:
        return self.failures, self.unfinished_ns + self.remaining_service_ns


class Oracle:
    def __init__(self, profile: Profile, policy: Policy, arrivals: tuple[Arrival, ...]):
        self.profile = profile
        self.window = oracle_window_ticks(policy.oracle_window_s)
        self.reorder = policy.name == "oracle"
        self.wait_state: tuple | None = None
        self.suffix_ns: dict[int, tuple[int, ...]] = {}
        self.next_tool: dict[int, tuple[int | None, ...]] = {}
        self.tool_release_ns = ticks(profile.tool_release_s)
        self.tool_save_ns = ticks(profile.tool_save_s)
        self.tool_write_ns = ticks(profile.tool_write_s)
        self.tool_read_ns = ticks(profile.tool_read_s)
        self.tool_restore_ns = ticks(profile.tool_restore_s)
        self.kv_release_ns = ticks(profile.kv_evict_s)
        for arrival in arrivals:
            suffix = [0]
            next_tool: list[int | None] = [None]
            for turn, round_ in reversed(list(enumerate(arrival.trace.rounds))):
                prefix = int(round_.prefix * profile.prefix_fraction)
                suffix.append(suffix[-1] + self.inference_ns(round_.prompt - prefix, round_.output) + round_.tool_ns)
                next_tool.append(turn if round_.tools or round_.tool_ns else next_tool[-1])
            self.suffix_ns[arrival.job_id] = tuple(reversed(suffix))
            self.next_tool[arrival.job_id] = tuple(reversed(next_tool))

    @staticmethod
    def pressure_state(sim: Simulator) -> tuple:
        # Converting a KV reservation to physical bytes during a batch does not
        # change runnable memory actions. Revisit on material state changes.
        return (
            tuple(request.order for request in sim.waiting["global"]),
            tuple((slot.state, slot.resident + slot.reserved)
                  for job in sim.jobs for slot in job.slots.values()),
        )

    def admit(self, sim: Simulator) -> None:
        queue = sim.waiting["global"]
        if self.wait_state is not None:
            if self.pressure_state(sim) == self.wait_state:
                return
            self.wait_state = None
        cap = sim.profile.capacity
        while queue:
            head = queue[0]
            sim.mark_head(head)
            reclaim = sim.reclaims.get("global")
            if reclaim is not None:
                if sim.advance_reclaim(queue, reclaim):
                    continue
                return

            hit = sim.resident_hit(queue)
            free = sim.free_bytes()
            target = sim.target(head)
            if not self.reorder or target > cap or sum(self.needed(sim, req) for req in queue) <= free:
                if hit is not None:
                    sim.start_request(hit, sim.target(hit), bypass=True)
                    continue
                if target > cap or self.needed(sim, head) <= free:
                    assert head.job.slots[head.side].state != "evicting"
                    sim.try_admit(head)
                    continue

            needed = self.needed(sim, head)
            if needed > free:
                sim.record_block(head, needed, free, free)
            can_wait = any(slot.state in ("active", "restoring")
                           for job in sim.jobs for slot in job.slots.values())
            actions, shared = self.candidates(sim, hit, can_wait)
            if not actions:
                return
            scores = [self.forecast(sim, action) for action in actions]
            best = min(
                range(len(actions)),
                key=lambda index: self.rank(actions[index], scores[index], index == shared),
            )
            action = actions[best]
            if sim.log is not None:
                sim.record(
                    "oracle_decision", head.job, side=head.side, request_order=head.order,
                    free_bytes=free, head_shortfall_bytes=max(0, needed - free),
                    queue=[{"request_order": req.order, "job_id": req.job.id, "side": req.side,
                            "needed_bytes": self.needed(sim, req)} for req in queue],
                    window_ns=self.window, selected=best, shared_action=shared,
                    candidates=[self.describe(sim, alternative, score)
                                for alternative, score in zip(actions, scores)],
                )
            if action.request is None:
                self.wait_state = self.pressure_state(sim)
                return
            self.apply(sim, action)

    @staticmethod
    def needed(sim: Simulator, request: Request) -> int:
        return max(0, sim.target(request) - request.job.slots[request.side].resident)

    def candidates(
        self, sim: Simulator, hit: Request | None, can_wait: bool,
    ) -> tuple[list[Action], int | None]:
        queue = sim.waiting["global"]
        free = sim.free_bytes()
        actions: list[Action] = []
        shared: int | None = None
        seen = set()
        requests = list(islice(queue, REQUEST_LIMIT if self.reorder else 1))
        if self.reorder and hit is not None and hit not in requests:
            requests.append(hit)
        for request in requests:
            if sim.target(request) > sim.profile.capacity or request.job.slots[request.side].state == "evicting":
                continue
            shortfall = self.needed(sim, request) - free
            if shortfall <= 0:
                if request is hit or (hit is None and request is queue[0]):
                    shared = len(actions)
                actions.append(Action(request, resident_hit=request is hit))
                continue
            for order, victims in enumerate(self.victim_sets(sim, request, shortfall)):
                action = Action(request, victims)
                if action.identity in seen:
                    continue
                seen.add(action.identity)
                if order == 0 and hit is None and request is queue[0]:
                    shared = len(actions)
                actions.append(action)
        if can_wait:
            if shared is None:
                shared = len(actions)
            actions.append(Action())
        return actions, shared

    def victim_sets(self, sim: Simulator, request: Request, shortfall: int) -> list[tuple[Victim, ...]]:
        lru = sim.eviction_candidates(request)
        if sum(job.slots[side].resident for job, side in lru) < shortfall:
            return []

        release = {(job.id, side): self.release_ns(sim, side) for job, side in lru}
        recovery = {(job.id, side): self.recovery_ns(job, side) for job, side in lru}

        def fastest(victim: Victim) -> tuple:
            job, side = victim
            key = job.id, side
            return release[key], recovery[key], -job.slots[side].resident

        def per_byte(victim: Victim) -> float:
            job, side = victim
            key = job.id, side
            return (release[key] + recovery[key]) / job.slots[side].resident

        groups = []
        for ranked in (lru, sorted(lru, key=fastest), sorted(lru, key=per_byte)):
            released = 0
            for index, (job, side) in enumerate(ranked, 1):
                released += job.slots[side].resident
                if released >= shortfall:
                    groups.append(tuple(ranked[:index]))
                    break
        return groups

    @staticmethod
    def rank(action: Action, score: Score, is_shared: bool) -> tuple:
        return (
            *score.objective, not is_shared, action.request is None,
            action.request.order if action.request is not None else 0,
            action.release_bytes, action.identity[1],
        )

    @staticmethod
    def apply(sim: Simulator, action: Action) -> None:
        request = action.request
        assert request is not None
        bypass = request is not sim.waiting["global"][0] and not action.resident_hit
        if action.victims:
            sim.start_reclaim(request, Oracle.needed(sim, request), sim.free_bytes(),
                              list(action.victims), oracle_bypass=bypass)
        else:
            sim.start_request(request, sim.target(request), bypass=action.resident_hit, oracle_bypass=bypass)

    def forecast(self, sim: Simulator, action: Action) -> Score:
        # Bound event callbacks are copied with their simulator and job objects.
        # Share the read-only replay inputs and omit live logging and recursion.
        memo = {id(sim.profile): sim.profile, id(self): None}
        if sim.log is not None:
            memo[id(sim.log)] = None
        memo.update((id(job.arrival), job.arrival) for job in sim.jobs)
        shadow = deepcopy(sim, memo)
        shadow.policy = Policy("shared")
        defer = None
        if action.request is None:
            state = self.pressure_state(shadow)
            defer = lambda: self.pressure_state(shadow) == state
        else:
            mapped = Action(memo[id(action.request)],
                            tuple((memo[id(job)], side) for job, side in action.victims),
                            action.resident_hit)
            self.apply(shadow, mapped)
        end = sim.now + self.window
        if shadow.replay(end, defer_admission=defer):
            shadow.finish_replay()
        return self.score(shadow, sim.now, end)

    def score(self, shadow: Simulator, start: int, end: int) -> Score:
        partial_gpu, restore_finish = self.inflight(shadow, end)
        failures = unfinished = remaining = 0
        for job in shadow.jobs:
            if job.arrival.at_ns > end or (job.terminal_at is not None and job.terminal_at < start):
                continue
            terminal = end if job.terminal_at is None else job.terminal_at
            unfinished += terminal - max(start, job.arrival.at_ns)
            if job.terminal_at is not None:
                failures += job.outcome != "completed"
                continue
            tool = job.slots["tool"]
            trace = job.arrival.trace
            failures += (
                tool.state in ("evicting", "evicted")
                and self.next_tool[job.id][job.turn] is not None
                and max(trace.tool_peak, trace.tool_resident, self.profile.restore_peak) > self.profile.capacity
            )
            remaining += self.remaining_ns(job, end, partial_gpu.get(job.id, 0), restore_finish.get(job.id))
        return Score(failures, unfinished, remaining)

    def inflight(self, sim: Simulator, end: int) -> tuple[dict[int, int], dict[int, int]]:
        partial_gpu, restore_finish = {}, {}
        for at, _, _, callback, args in sim.events:
            if callback == sim.finish_batch:
                kind, batch, tokens = args
                for job, amount in zip(batch, tokens):
                    service = self.inference_ns(amount, 0) if kind == "prefill" else self.inference_ns(0, amount)
                    partial_gpu[job.id] = service * (end - job.phase_since) // (at - job.phase_since)
            elif callback == sim.finish_reprefill:
                job, = args
                partial_gpu[job.id] = end - job.phase_since
            elif callback == sim.finish_restore:
                job, = args
                restore_finish[job.id] = at
            elif callback == sim.finish_disk and args[1] == "read":
                job = args[0]
                restore_finish[job.id] = at + self.tool_restore_ns
        return partial_gpu, restore_finish

    def remaining_ns(self, job: Job, end: int, partial_gpu: int, restore_finish: int | None) -> int:
        kv, tool = job.slots["kv"], job.slots["tool"]
        round_ = job.round
        remaining = self.suffix_ns[job.id][job.turn + 1]
        if not job.phase or job.phase == "kv_memory_wait":
            prefix = 0 if kv.state == "cold" else int(round_.prefix * self.profile.prefix_fraction)
            remaining += self.inference_ns(round_.prompt - prefix, round_.output) + round_.tool_ns
        elif kv.state == "active":
            remaining += (
                self.inference_ns(job.prefill_left + job.reprefill_tokens, job.decode_left)
                - partial_gpu + round_.tool_ns
            )
        else:
            elapsed = end - job.phase_since if tool.state == "active" else 0
            remaining += max(0, round_.tool_ns - elapsed)
        if restore_finish is not None:
            remaining += max(0, restore_finish - end)
        for side in SIDES:
            if job.slots[side].state in ("evicted", "evicting"):
                remaining += self.recovery_ns(job, side)
        return remaining

    def inference_ns(self, prefill: int, decode: int) -> int:
        return (ceil(prefill * TICKS_PER_SECOND / self.profile.prefill_tps[0])
                + ceil(decode * TICKS_PER_SECOND / self.profile.decode_tps[0]))

    def recovery_ns(self, job: Job, side: Side) -> int:
        turn = job.turn
        if side == "tool":
            return self.tool_read_ns + self.tool_restore_ns if self.next_tool[job.id][turn] is not None else 0
        if job.phase != "kv_memory_wait":
            turn += 1
        if turn >= len(job.arrival.trace.rounds):
            return 0
        prefix = int(job.arrival.trace.rounds[turn].prefix * self.profile.prefix_fraction)
        return self.inference_ns(prefix, 0)

    def release_ns(self, sim: Simulator, side: Side) -> int:
        if side == "kv":
            return self.kv_release_ns
        return max(self.tool_save_ns, sim.disk_free_at - sim.now) + self.tool_write_ns + self.tool_release_ns

    def describe(self, sim: Simulator, action: Action, score: Score) -> dict:
        request = action.request
        return {
            "action": action.kind,
            "request_order": None if request is None else request.order,
            "job_id": None if request is None else request.job.id,
            "side": None if request is None else request.side,
            "victims": [
                {"job_id": job.id, "side": side, "resident_bytes": job.slots[side].resident,
                 "estimated_release_ns": self.release_ns(sim, side),
                 "estimated_recovery_ns": self.recovery_ns(job, side)}
                for job, side in action.victims
            ],
            "failures": score.failures,
            "unfinished_ns": score.unfinished_ns,
            "remaining_service_ns": score.remaining_service_ns,
        }
