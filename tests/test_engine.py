import unittest
from dataclasses import replace

from sim.config import GIB, MIB, Profile, ticks
from sim.engine import Arrival, Policy, Simulator
from sim.trace import Round, ToolCall, Trace


def profile(capacity: int, **overrides) -> Profile:
    base = Profile(
        dram_gib=capacity / GIB,
        static_gib=0,
        kv_bytes_per_token=1,
        prefill_tps=(10, 20),
        decode_tps=(1, 2),
        max_batch=2,
        prefill_chunk_tokens=1024,
        kv_evict_s=0.2,
        tool_save_s=0.25,
        tool_write_s=0.3,
        tool_release_s=0.2,
        tool_read_s=0.5,
        tool_restore_s=2,
        tool_restore_peak_mib=0,
        prefix_fraction=1,
    )
    return replace(base, **overrides)


def window(prompt, output, duration, prefix=0):
    call = ToolCall("terminal", "test-tool", 0, ticks(duration))
    return Round(prompt, output, prefix, ticks(duration), (call,))


class SimulatorTests(unittest.TestCase):
    def run_case(self, config, policy, *arrivals):
        self.log = []
        result = Simulator(config, policy, arrivals, self.log.append).run()
        caps = policy.caps(config.capacity)
        previous_time = 0
        for row in self.log:
            sides = {
                side: sum(row[f"{side}_{field}_bytes"] for field in ("resident", "reserved", "held"))
                for side in ("kv", "tool")
            }
            self.assertGreaterEqual(row["at_ns"], previous_time)
            self.assertGreaterEqual(min(sides.values()), 0)
            if policy.name != "relaxed":
                self.assertLessEqual(sum(sides.values()), config.capacity)
            if caps is not None:
                self.assertLessEqual(sides["kv"], caps["kv"])
                self.assertLessEqual(sides["tool"], caps["tool"])
            previous_time = row["at_ns"]
        self.assertEqual(self.log[-1]["event"], "drained")
        self.assertEqual(self.log[-1]["kv_resident_bytes"] + self.log[-1]["tool_resident_bytes"], 0)
        return result

    def events(self, name, **fields):
        return [row for row in self.log if row["event"] == name
                and all(row.get(field) == value for field, value in fields.items())]

    def test_batched_and_continuous_decode(self):
        a = Trace("a", (Round(4, 4),))
        b = Trace("b", (Round(4, 2),))
        c = Trace("c", (Round(0, 1),))
        result = self.run_case(
            profile(40), Policy("shared"), Arrival(1, a), Arrival(2, b), Arrival(3, c, ticks(1)),
        )
        self.assertEqual([job.terminal_at for job in result.jobs], [ticks(4.4), ticks(3.4), ticks(2.4)])
        prefill = self.events("batch_start", batch="prefill")
        self.assertEqual(prefill[0]["job_ids"], [1, 2])
        self.assertEqual(prefill[0]["duration_ns"], ticks(0.4))
        decode = self.events("batch_start", batch="decode")
        self.assertEqual([row["job_ids"] for row in decode], [[1, 2], [3, 1], [2, 1], [1]])
        self.assertEqual(decode[1]["kv_executing_bytes"], 5)
        self.assertEqual(decode[1]["kv_resident_bytes"], 10)
        self.assertEqual(result.counters["max_decode_batch"], 2)
        self.assertEqual(result.counters["decode_batches"], 4)

    def test_retains_private_prefix_between_rounds(self):
        trace = Trace("retained", (window(8, 1, 2), Round(10, 0, 7)), 1, 2)
        result = self.run_case(profile(20), Policy("shared"), Arrival(1, trace))
        job = result.jobs[0]
        self.assertEqual(job.outcome, "completed")
        self.assertEqual(job.counts["prefill_tokens"], 11)
        self.assertEqual(job.counts["reused_prompt_tokens"], 7)
        self.assertEqual(job.counts["reprefill_tokens"], 0)
        self.assertEqual(job.terminal_at, ticks(4.1))
        self.assertEqual(self.events("tool_start")[0]["kv_resident_bytes"], 9)
        self.assertEqual(self.events("tool_start")[0]["kv_executing_bytes"], 0)
        self.assertEqual(self.events("tool_start")[0]["tool_resident_bytes"], 2)
        self.assertEqual(self.events("tool_start")[0]["tool_inflight_bytes"], 2)
        self.assertEqual(self.events("tool_finish")[0]["tool_inflight_bytes"], 0)

    def test_fixed_strands_free_capacity_with_individually_feasible_requests(self):
        job_trace = Trace("inference", (Round(6, 0),))
        arrivals = (Arrival(1, job_trace), Arrival(2, job_trace))
        fixed = self.run_case(profile(16), Policy("fixed", 50), *arrivals)
        self.assertEqual(fixed.jobs[1].times["kv_memory_wait"], ticks(0.6))
        blocked = self.events("memory_blocked", partition_stranded=True)
        self.assertEqual([(e["side_free_bytes"], e["total_free_bytes"]) for e in blocked], [(2, 10)])
        shared = self.run_case(profile(16), Policy("shared"), *arrivals)
        self.assertEqual(shared.jobs[1].times["kv_memory_wait"], 0)
        self.assertEqual(shared.jobs[1].terminal_at, ticks(0.6))
        self.assertEqual(fixed.jobs[1].terminal_at, ticks(1.2))

    def test_reprefill_is_exclusive_and_distinct_from_new_prefill(self):
        a = Trace("evicted", (window(8, 0, 2), Round(6, 0, 5)), 1, 1)
        b = Trace("busy", (Round(1, 7),))
        c = Trace("arrives_during_reprefill", (Round(1, 0),))
        result = self.run_case(
            profile(16), Policy("shared"), Arrival(1, a), Arrival(2, b, ticks(1)), Arrival(3, c, ticks(3.4)),
        )
        start, = self.events("evict_start", side="kv")
        finish, = self.events("evict_finish", side="kv")
        self.assertEqual(start["kv_resident_bytes"], 8)
        self.assertEqual(finish["at_ns"] - start["at_ns"], ticks(0.2))
        self.assertEqual(finish["released_bytes"], 8)
        b_admit, = self.events("admit", job_id=2)
        self.assertEqual(b_admit["at_ns"], finish["at_ns"])
        restore, = self.events("reprefill_start")
        restored, = self.events("reprefill_finish")
        self.assertEqual(restore["tokens"], 5)
        self.assertEqual(restored["at_ns"] - restore["at_ns"], ticks(0.5))
        batches = self.events("batch_start")
        self.assertFalse(any(restore["at_ns"] <= row["at_ns"] < restored["at_ns"] for row in batches))
        self.assertEqual(result.jobs[0].counts["prefill_tokens"], 9)
        self.assertEqual(result.jobs[0].counts["reprefill_tokens"], 5)
        self.assertEqual(result.jobs[2].counts["gpu_wait_during_reprefill_ns"], ticks(0.4))

    def test_save_write_release_and_nonblocking_restore(self):
        a = Trace("restore", (window(0, 0, 1), window(0, 10, 1)), 6, 8)
        b = Trace("displaces_a", (window(0, 0, 1),), 6, 8)
        c = Trace("inference", (Round(1, 1),))
        result = self.run_case(
            profile(100, tool_restore_peak_mib=9 / MIB), Policy("fixed", 90),
            Arrival(1, a), Arrival(2, b, ticks(2)), Arrival(3, c, ticks(11.5)),
        )
        start, = self.events("evict_start", side="tool")
        finish, = self.events("evict_finish", side="tool")
        self.assertEqual(start["tool_resident_bytes"], 6)
        self.assertEqual(finish["at_ns"] - start["at_ns"], ticks(0.75))
        self.assertEqual(self.events("tool_start", job_id=2)[0]["at_ns"], ticks(2.75))
        restore, = self.events("tool_restore_start")
        restored, = self.events("tool_restore_finish")
        self.assertEqual(restore["tool_resident_bytes"], 9)
        self.assertEqual(restore["tool_inflight_bytes"], 9)
        self.assertEqual(restored["at_ns"] - restore["at_ns"], ticks(2.5))
        self.assertLess(result.jobs[2].terminal_at, restored["at_ns"])
        self.assertGreater(result.jobs[2].terminal_at, restore["at_ns"])
        write, = self.events("checkpoint_io_enqueue", direction="write")
        read, = self.events("checkpoint_io_enqueue", direction="read")
        self.assertEqual(write["end_ns"] - write["start_ns"], ticks(0.3))
        self.assertEqual(read["end_ns"] - read["start_ns"], ticks(0.5))
        self.assertEqual(read["restore_resident_bytes"], 9)

    def test_checkpoint_writes_share_fifo_channel(self):
        a = Trace("small", (window(0, 0, 1), Round(0, 20)), 3, 4)
        b = Trace("large", (window(0, 0, 1),), 0, 9)
        result = self.run_case(
            profile(100), Policy("fixed", 90),
            Arrival(1, a), Arrival(2, a), Arrival(3, b, ticks(2)),
        )
        writes = self.events("checkpoint_io_enqueue", direction="write")
        self.assertEqual([row["job_id"] for row in writes], [1, 2])
        self.assertEqual(writes[0]["end_ns"], writes[1]["start_ns"])
        self.assertEqual(result.counters["checkpoint_disk_queue_ns"], ticks(0.3))
        self.assertEqual(self.events("tool_start", job_id=3)[0]["at_ns"], ticks(3.05))

    def test_shared_can_reclaim_own_kv_for_tool_with_a_real_hard_cap(self):
        big = Trace("hard_cap", (Round(12, 0),))
        over = self.run_case(profile(10), Policy("shared"), Arrival(1, big))
        self.assertEqual(over.jobs[0].outcome, "kv_allocation_exceeds_cap")
        trace = Trace("tool_after_inference", (window(6, 0, 1),), 5, 5)
        shared = self.run_case(profile(10), Policy("shared"), Arrival(1, trace))
        self.assertEqual(shared.jobs[0].outcome, "completed")
        evicted, = self.events("evict_finish", requester_side="tool", side="kv", job_id=1)
        self.assertEqual(evicted["requester_id"], 1)
        self.assertEqual(self.events("admit", side="tool")[0]["at_ns"], evicted["at_ns"])
        self.assertEqual(shared.counters["tool_request_kv_evictions"], 1)
        separate = self.run_case(profile(12), Policy("fixed", 40), Arrival(1, trace))
        self.assertEqual(separate.jobs[0].outcome, "kv_allocation_exceeds_cap")
        self.assertTrue(self.events("over_cap")[0]["partition_stranded"])
        waived = self.run_case(profile(10), Policy("relaxed"), Arrival(1, trace))
        self.assertEqual(waived.jobs[0].outcome, "completed")
        self.assertGreater(waived.peaks["charged_bytes"], 10)

    def test_global_lru_uses_both_sides_and_waits_for_physical_release(self):
        a = Trace("idle_tool", (window(4, 0, 1), Round(0, 6)), 4, 4)
        b = Trace("idle_kv", (window(3, 0, 3),), 1, 1)
        c = Trace("needs_both", (Round(11, 0),))
        result = self.run_case(
            profile(18), Policy("shared"), Arrival(1, a), Arrival(2, b), Arrival(3, c, ticks(1.5)),
        )
        reclaim, = self.events("reclaim_start")
        self.assertEqual(reclaim["victims"], [{"job_id": 2, "side": "kv"}, {"job_id": 1, "side": "tool"}])
        releases = self.events("evict_finish", requester_side="kv")
        self.assertEqual([row["at_ns"] for row in releases], [ticks(1.7), ticks(2.25)])
        admitted, = self.events("admit", job_id=3)
        self.assertEqual(admitted["at_ns"], releases[-1]["at_ns"])
        self.assertEqual(admitted["request_order"], reclaim["request_order"])
        self.assertEqual(result.counters["kv_request_tool_evictions"], 1)

    def test_fifo_blocks_a_smaller_cold_request_behind_head(self):
        anchor = Trace("anchor", (Round(0, 8),))
        head = Trace("head", (Round(6, 0),))
        tail = Trace("tail", (Round(2, 0),))
        result = self.run_case(
            profile(10), Policy("shared"), Arrival(1, anchor),
            Arrival(2, head, ticks(1)), Arrival(3, tail, ticks(2)),
        )
        admitted = self.events("admit")
        self.assertEqual([row["job_id"] for row in admitted], [1, 2, 3])
        self.assertEqual(admitted[-1]["at_ns"], ticks(8))
        self.assertEqual(result.jobs[2].counts["fifo_tail_wait_ns"], ticks(6))
        self.assertFalse(self.events("evict_start"))

    def test_resident_kv_bypasses_only_when_its_increment_fits(self):
        anchor = Trace("anchor", (Round(0, 8),))
        head = Trace("head", (Round(7, 0),))
        for prompt, prefix, expect_bypass in ((5, 4, True), (6, 4, False), (5, 0, False)):
            with self.subTest(prompt=prompt, prefix=prefix):
                warm = Trace("warm_kv", (window(4, 0, 2), Round(prompt, 0, prefix)), 1, 1)
                self.run_case(
                    profile(14), Policy("shared"), Arrival(1, warm),
                    Arrival(2, anchor), Arrival(3, head, ticks(1)),
                )
                admitted, = self.events("admit", job_id=1, side="kv", turn=2)
                if expect_bypass:
                    self.assertEqual(admitted["at_ns"], ticks(2.4))
                    self.assertTrue(admitted["resident_bypass"])
                    self.assertEqual(admitted["reservation_delta_bytes"], 1)
                    self.assertIsNone(admitted["head_at_ns"])
                else:
                    self.assertGreater(admitted["at_ns"], ticks(2.4))
                    self.assertFalse(self.events("admit", job_id=1, resident_bypass=True, at_ns=ticks(2.4)))

    def test_resident_tool_bypasses_fifo_and_avoids_restore(self):
        warm = Trace("warm_tool", (window(0, 0, 1), window(0, 1, 1)), 6, 7)
        anchor = Trace("anchor", (Round(0, 8),))
        head = Trace("head", (Round(10, 0),))
        result = self.run_case(
            profile(16), Policy("shared"), Arrival(1, warm),
            Arrival(2, anchor), Arrival(3, head, ticks(1.2)),
        )
        admitted, = self.events("admit", job_id=1, side="tool", turn=2)
        self.assertTrue(admitted["resident_bypass"])
        self.assertEqual(admitted["at_ns"], ticks(2))
        self.assertEqual(admitted["reservation_delta_bytes"], 1)
        self.assertEqual(result.jobs[0].counts["tool_restores"], 0)

    def test_resident_hit_cannot_spend_bytes_held_for_head(self):
        victim = Trace("idle_tool", (window(0, 0, 0.1), Round(0, 4)), 3, 3)
        warm = Trace("warm_kv", (window(4, 0, 2), Round(5, 0, 4)), 1, 1)
        head = Trace("head", (Round(4, 0),))
        self.run_case(
            profile(13), Policy("shared"), Arrival(1, victim),
            Arrival(2, warm), Arrival(3, head, ticks(2.2)),
        )
        reclaim, = self.events("reclaim_start", job_id=3)
        self.assertEqual(reclaim["kv_held_bytes"], 1)
        submitted, = self.events("submit", job_id=2, side="kv", turn=2)
        admitted, = self.events("admit", job_id=2, side="kv", turn=2)
        head_admit, = self.events("admit", job_id=3)
        self.assertLess(reclaim["at_ns"], submitted["at_ns"])
        self.assertLess(submitted["at_ns"], head_admit["at_ns"])
        self.assertGreaterEqual(admitted["at_ns"], head_admit["at_ns"])

    def test_oracle_waits_for_a_near_release_and_reclaims_for_a_long_wait(self):
        requester = Trace("requester", (window(3, 0, 1), Round(4, 1, 3)), 5, 5)
        anchor = Trace("anchor", (Round(0, 5),))
        for name in ("oracle_fifo", "oracle"):
            policy = Policy(name)
            with self.subTest(policy=name, release="near"):
                config = profile(10, decode_tps=(10, 20))
                arrivals = (Arrival(1, requester, ticks(0.4)), Arrival(2, anchor))
                baseline = self.run_case(config, Policy("shared"), *arrivals)
                result = self.run_case(config, policy, *arrivals)
                decision, = self.events("oracle_decision")
                self.assertEqual(decision["at_ns"], ticks(0.7))
                self.assertEqual(decision["candidates"][decision["selected"]]["action"], "wait")
                self.assertFalse(self.events("evict_start"))
                self.assertEqual(result.jobs[0].terminal_at, ticks(2))
                self.assertEqual(baseline.jobs[0].terminal_at - result.jobs[0].terminal_at, ticks(0.4))
            with self.subTest(policy=name, release="far"):
                result = self.run_case(profile(10), policy, Arrival(1, requester), Arrival(2, anchor))
                first = self.events("oracle_decision")[0]
                candidate = first["candidates"][first["selected"]]
                self.assertEqual(candidate["action"], "reclaim")
                self.assertEqual([(v["job_id"], v["side"]) for v in candidate["victims"]], [(1, "kv")])
                self.assertEqual(result.jobs[0].counts["reprefill_tokens"], 3)

    def test_oracle_selects_between_kv_and_tool_as_restore_cost_changes(self):
        tool = Trace("idle_tool", (window(4, 0, 1), window(0, 6, 1)), 4, 4)
        kv = Trace("idle_kv", (window(3, 0, 4), Round(3, 0, 3)), 1, 1)
        arrivals = (Arrival(1, tool), Arrival(2, kv), Arrival(3, Trace("new", (Round(7, 0),)), ticks(1.5)))
        for restore_s, expected in ((0, (1, "tool")), (4, (2, "kv"))):
            with self.subTest(restore_s=restore_s):
                config = profile(18, tool_save_s=0.05, tool_write_s=0.05, tool_release_s=0.05,
                                 tool_read_s=0, tool_restore_s=restore_s)
                result = self.run_case(config, Policy("oracle_fifo"), *arrivals)
                decision = self.events("oracle_decision")[0]
                choices = decision["candidates"]
                self.assertEqual({c["victims"][0]["side"] for c in choices if c["victims"]}, {"kv", "tool"})
                chosen, = choices[decision["selected"]]["victims"]
                self.assertEqual((chosen["job_id"], chosen["side"]), expected)
                eviction, = self.events("evict_start")
                self.assertEqual((eviction["job_id"], eviction["side"]), expected)
                self.assertTrue(all(job.outcome == "completed" for job in result.jobs))

    def test_oracle_scores_an_unavoidable_restore_over_cap_beyond_its_window(self):
        tool = Trace("idle_tool", (window(4, 0, 1), window(0, 6, 1)), 4, 4)
        kv = Trace("idle_kv", (window(3, 0, 4), Round(3, 0, 3)), 1, 1)
        arrivals = (Arrival(1, tool), Arrival(2, kv), Arrival(3, Trace("new", (Round(7, 0),)), ticks(1.5)))
        config = profile(18, tool_restore_peak_mib=19 / MIB, tool_save_s=0.05,
                         tool_write_s=0.05, tool_release_s=0.05, tool_read_s=0, tool_restore_s=0)
        result = self.run_case(config, Policy("oracle_fifo", oracle_window_s=0.1), *arrivals)
        decision = self.events("oracle_decision")[0]
        unsafe, = [c for c in decision["candidates"] if c["victims"] and c["victims"][0]["side"] == "tool"]
        self.assertEqual(unsafe["failures"], 1)
        self.assertEqual(decision["candidates"][decision["selected"]]["failures"], 0)
        self.assertTrue(all(job.outcome == "completed" for job in result.jobs))

    def test_oracle_cold_tail_can_reclaim_and_owns_the_released_bytes(self):
        anchor = Trace("anchor", (window(0, 0, 0.1), Round(0, 8)), 3, 3)
        head = Trace("head", (Round(7, 0),))
        tail = Trace("tail", (Round(4, 0),))
        arrivals = (Arrival(1, anchor), Arrival(2, head, ticks(1)), Arrival(3, tail, ticks(2)))
        fifo = self.run_case(profile(13), Policy("oracle_fifo"), *arrivals)
        result = self.run_case(profile(13), Policy("oracle"), *arrivals)
        reclaim, = self.events("reclaim_start", job_id=3)
        released, = self.events("evict_finish", requester_id=3)
        admitted, = self.events("admit", job_id=3)
        self.assertEqual(reclaim["kv_held_bytes"], 2)
        self.assertEqual(released["kv_held_bytes"], 4)
        self.assertEqual(admitted["at_ns"], released["at_ns"])
        self.assertEqual(admitted["request_order"], reclaim["request_order"])
        self.assertEqual(admitted["prior_state"], "cold")
        self.assertTrue(admitted["oracle_bypass"])
        self.assertFalse(admitted["resident_bypass"])
        self.assertIsNone(admitted["head_at_ns"])
        self.assertLess(admitted["at_ns"], self.events("admit", job_id=2)[0]["at_ns"])
        self.assertLess(sum(job.terminal_at for job in result.jobs), sum(job.terminal_at for job in fifo.jobs))

    def test_oracle_forces_reclaim_with_no_independent_progress_and_zero_release_time(self):
        trace = Trace("own_kv", (window(6, 0, 1),), 5, 5)
        result = self.run_case(profile(10, kv_evict_s=0), Policy("oracle"), Arrival(1, trace))
        decision, = self.events("oracle_decision")
        self.assertEqual([c["action"] for c in decision["candidates"]], ["reclaim"])
        released, = self.events("evict_finish")
        admitted, = self.events("admit", side="tool")
        self.assertEqual(released["at_ns"], admitted["at_ns"])
        self.assertEqual(result.jobs[0].outcome, "completed")

    def test_replay_is_deterministic(self):
        trace = Trace("same", (window(8, 1, 3), Round(7, 2, 6)), 1, 2)
        arrivals = (Arrival(2, trace, ticks(0.25)), Arrival(1, trace))
        for name in ("shared", "oracle_fifo", "oracle"):
            with self.subTest(policy=name):
                first = self.run_case(profile(17), Policy(name), *arrivals)
                log = self.log
                second = self.run_case(profile(17), Policy(name), *arrivals)
                self.assertEqual(self.log, log)
                self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
