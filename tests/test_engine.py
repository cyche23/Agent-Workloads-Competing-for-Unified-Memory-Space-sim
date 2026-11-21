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
        a = Trace("evicted", (window(8, 0, 2), Round(8, 0, 5)), 1, 1)
        b = Trace("busy", (Round(1, 7),))
        c = Trace("arrives_during_reprefill", (Round(1, 0),))
        result = self.run_case(
            profile(16), Policy("shared"), Arrival(1, a), Arrival(2, b, ticks(1)), Arrival(3, c, ticks(8.4)),
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
        self.assertEqual(result.jobs[0].counts["prefill_tokens"], 11)
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

    def test_distinguishes_fixed_quota_true_hard_cap_and_no_progress(self):
        big = Trace("hard_cap", (Round(12, 0),))
        over = self.run_case(profile(10), Policy("shared"), Arrival(1, big))
        self.assertEqual(over.jobs[0].outcome, "kv_allocation_exceeds_cap")
        immovable = Trace("no_progress", (window(6, 0, 1),), 5, 5)
        deadlock = self.run_case(profile(10), Policy("shared"), Arrival(1, immovable))
        self.assertEqual(deadlock.jobs[0].outcome, "tool_no_progress")
        self.assertEqual(self.events("no_progress")[0]["target_bytes"], 5)
        self.assertFalse(self.events("evict_start"))
        separate = self.run_case(profile(12), Policy("fixed", 40), Arrival(1, immovable))
        self.assertEqual(separate.jobs[0].outcome, "kv_allocation_exceeds_cap")
        self.assertTrue(self.events("over_cap")[0]["partition_stranded"])
        waived = self.run_case(profile(10), Policy("relaxed"), Arrival(1, immovable))
        self.assertEqual(waived.jobs[0].outcome, "completed")
        self.assertGreater(waived.peaks["charged_bytes"], 10)

    def test_replay_is_deterministic(self):
        trace = Trace("same", (window(8, 1, 3), Round(7, 2, 6)), 1, 2)
        arrivals = (Arrival(2, trace, ticks(0.25)), Arrival(1, trace))
        first = self.run_case(profile(17), Policy("shared"), *arrivals)
        log = self.log
        second = self.run_case(profile(17), Policy("shared"), *arrivals)
        self.assertEqual(self.log, log)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
