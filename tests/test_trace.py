import unittest

from sim.config import MIB, ticks
from sim.trace import parse_trace


def token(turn, response, prompt, completion, cached):
    return {
        "linked_to_dialogue": str(turn is not None),
        "dialogue_turn": str(turn),
        "response_id": response,
        "prompt_tokens": str(prompt),
        "completion_tokens": str(completion),
        "cache_read_tokens": str(cached),
    }


def action(id_, response, start, end):
    return {
        "tool_name": "terminal",
        "paired": "True",
        "llm_response_id": response,
        "action_id": id_,
        "start_s": str(start),
        "end_s": str(end),
    }


def sample(at, memory, ok="1", active="1"):
    return {
        "elapsed_s": str(at), "container_memory_mib": str(memory),
        "sample_ok": ok, "active_containers": active,
    }


class TraceTests(unittest.TestCase):
    def test_join_union_resident_peak_and_prefix(self):
        tokens = [
            token(2, "r2", 14, 1, 14),
            token(None, "extra", 4, 1, 0),
            token(1, "r1", 8, 2, 6),
            token(3, "r3", 15, 2, 11),
        ]
        tools = [
            action("b", "r1", 2, 4), action("a", "r1", 1, 3),
            action("c", "r1", 5, 6), action("d", "r2", 8, 8.1),
        ]
        resources = [
            sample(0, 100), sample(1.5, 17), sample(2.5, 20),
            sample(4, 9), sample(4.5, 11), sample(5, 14), sample(7, 99, "0"),
            sample(7.5, 99, active="0"), sample(8.2, 99),
        ]
        trace = parse_trace("test", tokens, tools, resources, "finished")
        self.assertEqual(trace.tool_resident, 10 * MIB)
        self.assertEqual(trace.tool_peak, 20 * MIB)
        self.assertEqual(trace.idle_samples, 2)
        self.assertEqual(trace.active_samples, 3)
        self.assertEqual(trace.uncovered_tools, 1)
        self.assertEqual(trace.excluded_api_calls, 1)
        self.assertEqual([r.prefix for r in trace.rounds], [0, 10, 11])
        self.assertEqual([r.tool_ns for r in trace.rounds], [ticks(4), ticks(0.1), 0])
        self.assertEqual([call.action_id for call in trace.rounds[0].tools], ["a", "b", "c"])

    def test_orphan_tool_action_fails_import(self):
        with self.assertRaisesRegex(ValueError, "unpaired tool action"):
            parse_trace("broken", [token(1, "good", 1, 1, 0)],
                        [action("a", "unknown", 1, 2)], [], "finished")


if __name__ == "__main__":
    unittest.main()
