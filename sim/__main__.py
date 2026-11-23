import argparse
import runpy
from pathlib import Path

from .config import Experiment
from .experiment import run_experiment
from .memory_plot import plot_memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SWE-bench agents under a DRAM budget.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run the matched experiment matrix")
    run.add_argument("--config", type=Path, required=True, help="Python script exposing CONFIG")
    run.add_argument("--out", type=Path, required=True, help="new run directory")
    run.add_argument(
        "--quick", action="store_true", help="run only burst at every configured concurrency with the first seed",
    )
    plot = subcommands.add_parser("plot-memory", help="render memory timelines from an existing run")
    plot.add_argument("--run", type=Path, required=True, help="run directory containing events.jsonl.gz")
    plot.add_argument("--case", action="append", required=True, help="case ID to plot (repeatable)")
    plot.add_argument("--out", type=Path, required=True, help="new plot directory")
    plot.add_argument("--resident-only", action="store_true", help="plot physical memory from a legacy log")
    args = parser.parse_args()
    try:
        if args.command == "plot-memory":
            plot_memory(args.run, args.case, args.out, args.resident_only)
            return
        config = runpy.run_path(str(args.config)).get("CONFIG")
        if not isinstance(config, Experiment):
            raise ValueError("the config script must define CONFIG as an Experiment")
        run_experiment(config, args.config, args.out, args.quick)
    except (OSError, ValueError) as error:
        parser.exit(2, f"sim: {error}\n")


if __name__ == "__main__":
    main()
