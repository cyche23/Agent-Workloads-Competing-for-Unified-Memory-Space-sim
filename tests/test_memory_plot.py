import gzip
import json
import tempfile
import unittest
from pathlib import Path

from sim.config import ticks
from sim.memory_plot import read_timelines, write_timeline_csv


CASE = "burst/n2/seed0/shared"


def header(case=CASE):
    return {
        "case_id": case, "at_ns": 0, "event": "case_start",
        "total_dram_bytes": 30, "static_bytes": 10,
    }


def snapshot(at, kv=0, tool=0, active_kv=0, inflight_tool=0, event="batch_start", **fields):
    return {
        "case_id": CASE, "at_ns": at, "event": event,
        "kv_resident_bytes": kv, "tool_resident_bytes": tool,
        "kv_executing_bytes": active_kv, "tool_inflight_bytes": inflight_tool,
        **fields,
    }


class TimelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.log = Path(temp.name) / "events.jsonl.gz"

    def write_log(self, *rows):
        with gzip.open(self.log, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def test_same_time_snapshots_and_checkpoint_queue(self):
        self.write_log(
            header(case="other"),
            header(),
            snapshot(0, 8, 2, 5, 2),
            snapshot(2, 8, 2, 4, 2),
            snapshot(2, 8, 6, 4, 6, "checkpoint_io_enqueue",
                     direction="read", start_ns=5, restore_resident_bytes=4),
            snapshot(7, 6, 4, 0, 4, "tool_start"),
            snapshot(8, 6, 2, 0, 0, "tool_finish"),
            snapshot(10, event="drained"),
        )
        timeline, = read_timelines(self.log, [CASE])
        self.assertEqual(timeline.budget, 20)
        self.assertEqual([(p.at_ns, p.resident, p.executing) for p in timeline.points], [
            (0, 10, 7), (2, 14, 6), (5, 14, 10), (7, 10, 4), (8, 8, 0), (10, 0, 0),
        ])
        self.assertAlmostEqual(timeline.mean_utilization(), 0.58)

    def test_legacy_requires_resident_only_and_exports_missing_values(self):
        legacy = snapshot(0, 8, 2)
        drained = snapshot(2, event="drained")
        for row in (legacy, drained):
            del row["kv_executing_bytes"]
            del row["tool_inflight_bytes"]
        self.write_log(header(), legacy, drained)
        with self.assertRaisesRegex(ValueError, "--resident-only"):
            read_timelines(self.log, [CASE])
        timeline, = read_timelines(self.log, [CASE], resident_only=True)
        self.assertEqual(timeline.points[0].resident, 10)
        self.assertIsNone(timeline.points[0].executing)
        self.assertIsNone(timeline.minute_averages()[0].executing_share)
        csv = self.log.with_name("timeline.csv")
        write_timeline_csv(csv, [timeline])
        self.assertIn(f"{CASE},0,8,2,10,,,,20", csv.read_text())

    def test_minute_means_use_interval_overlap_and_actual_final_duration(self):
        self.write_log(
            header(),
            snapshot(0, 4, active_kv=1),
            snapshot(ticks(10), 12, active_kv=6),
            snapshot(ticks(80)),
            snapshot(ticks(180), 6, active_kv=3),
            snapshot(ticks(185), event="drained"),
        )
        timeline, = read_timelines(self.log, [CASE])
        first, second, idle, tail = timeline.minute_averages()
        self.assertAlmostEqual(first.resident_bytes, 640 / 60)
        self.assertAlmostEqual(first.executing_bytes, 310 / 60)
        self.assertAlmostEqual(first.executing_share, 310 / 640)
        self.assertEqual((second.resident_bytes, second.executing_bytes), (4, 2))
        self.assertEqual((idle.resident_bytes, idle.executing_bytes, idle.executing_share), (0, 0, None))
        self.assertEqual((tail.start_ns, tail.end_ns), (ticks(180), ticks(185)))
        self.assertEqual((tail.resident_bytes, tail.executing_bytes, tail.executing_share), (6, 3, 0.5))

    def test_missing_and_incomplete_cases(self):
        self.write_log(header())
        with self.assertRaisesRegex(ValueError, "unknown case IDs"):
            read_timelines(self.log, ["other"])
        with self.assertRaisesRegex(ValueError, "did not drain"):
            read_timelines(self.log, [CASE])

    def test_zero_duration_and_invalid_execution_memory(self):
        self.write_log(header(), snapshot(0, event="drained"))
        timeline, = read_timelines(self.log, [CASE])
        self.assertIsNone(timeline.mean_utilization())
        self.assertEqual(timeline.minute_averages(), [])
        self.write_log(header(), snapshot(0, 1, active_kv=2), snapshot(1, event="drained"))
        with self.assertRaisesRegex(ValueError, "inconsistent memory"):
            read_timelines(self.log, [CASE])


if __name__ == "__main__":
    unittest.main()
