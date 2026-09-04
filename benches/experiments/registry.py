# sb-cli experiment registry
# Maps experiment names to their directory paths (relative to benches/).
# Comment out entries to exclude from 'sb-cli collect'.
EXPERIMENTS: dict[str, str] = {
    "gemm_sc": "experiments/2026_09_01_15_18_50",
    "gemm_sc_baseline": "experiments/2026_09_01_15_35_11",
    "gemm-MINI-float32-007": "experiments/2026_09_01_16_29_53",
    "cpu_gemm_mini_ref": "experiments/2026_09_04_15_04_39",
    "cpu_gemm_mini_esp": "experiments/2026_09_04_15_07_01",
}
