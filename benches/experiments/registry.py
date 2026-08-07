# sb-cli experiment registry
# Maps experiment names to their directory paths (relative to benches/).
# Comment out entries to exclude from 'sb-cli collect'.
EXPERIMENTS: dict[str, str] = {
    "gemver-MINI-float32": "experiments/2026_07_29_14_57_58",
    "gemver-MINI-float32-000": "experiments/2026_07_29_15_07_39",
    "gemm_mini_test_aug": "experiments/2026_08_03_16_09_26",
}
