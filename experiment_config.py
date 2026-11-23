"""Editable inputs for the SWE-bench/OpenHands replay experiment."""

from pathlib import Path

from sim.config import Experiment, Profile, Sources

TRACE_ROOT = Path.home() / "codes" / "experiment-jetson"

CONFIG = Experiment(
    sources=Sources(
        token_csv=TRACE_ROOT / "20261117_swe_llm_trace_tokens/results/run_001/per_call_tokens.csv",
        swe_batch=TRACE_ROOT / "SWE_Bench_arm64/results/20261013_memory_heavy_resource_maxiter40",
        instances=(
            "pydata__xarray-6721",
            "scikit-learn__scikit-learn-25232",
            "astropy__astropy-14995",
        ),
    ),
    profile=Profile(
        dram_gib=32,
        static_gib=16,  # 按照1.7B模型运行在32GB设备上考虑
        kv_bytes_per_token=109 * 1024,  
        # Aggregate token/s indexed by batch size 1 through 8. Only size 1
        # is anchored to the historical Jetson replay.
        prefill_tps=(6000, 6375, 6750, 6750, 6750, 6750, 6750, 6750),
        decode_tps=(36, 72, 80, 88, 96, 104, 112, 120),
        max_batch=8,
        prefill_chunk_tokens=2048,
        kv_evict_s=0.200,
        tool_save_s=0.878,
        tool_write_s=1.263,
        tool_release_s=0.147,
        tool_read_s=1.263,
        tool_restore_s=0.490,
        tool_restore_peak_mib=376.125,
        prefix_fraction=1.0,
    ),
    concurrency=(1, 2, 3, 4, 6, 8),
    inference_splits=(50, 60, 70, 80, 90, 95),
    seeds=(0, 1, 2),
    stagger_max_s=10,
    sensitivity_concurrency=(4, 6, 8),
    sensitivity_dram_gib=(24,),
    sensitivity_prefix_fraction=0.5,
    sensitivity_restore_factors=(0.5, 2.0),
)
