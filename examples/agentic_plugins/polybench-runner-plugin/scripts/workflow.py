import argparse
import json
import re
import subprocess
from pathlib import Path

import mlflow

PLUGIN_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_PROMPT = str(PLUGIN_DIR / "scripts" / "pythonworkflow.txt")
EXPERIMENT_NAME = "Polybench"

TRANSFORM_SCHEDULE_DIR = "transformation/transform_schedules"
KERNEL_STEPS_DIR = "transformation/kernel_steps"
ARTIFACTS_TRANSFORM = [
    "02_linalg_tile_ts.mlir",
    "04_affine_unroll_ts.mlir",
]

ARTIFACTS_KERNEL_STEPS = [
    "06_affine_unrolled.mlir",

]
BAMBU_SUMMARY = "transformation/bambu_summary.json"

def sync_workflow_prompt(prompt_file: str, kernel: str, dimension: str, target: str) -> None:
    path = Path(prompt_file)
    text = path.read_text()
    text = re.sub(r"(<Kernel>:\s*).*", rf"\g<1>{kernel}", text)
    text = re.sub(r"(<Dimension>:\s*).*", rf"\g<1>{dimension}", text)
    text = re.sub(r"(<Target>:\s*).*", rf"\g<1>{target}", text)
    path.write_text(text)


def find_experiment_dir(benches_root: Path) -> Path | None:
    """Return the most recently modified experiment directory under benches_root."""
    experiments = sorted(
        (benches_root / "experiments").iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for exp in experiments:
        if exp.is_dir() and not exp.is_symlink():
            return exp
    return None


@mlflow.trace(span_type="CHAT")
def run_workflow(prompt: str) -> dict:
    result = subprocess.run(
        [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--allowedTools", "Agent,Bash,Edit,Read,Skill,Write",
            "--exclude-dynamic-system-prompt-sections",
            "--plugin-dir", str(PLUGIN_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude exited with code {result.returncode}")

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("WARNING: could not parse claude output as JSON; skipping token/duration metrics")
        return {}

    for key in ("duration_ms", "duration_api_ms", "total_cost_usd"):
        if key in output:
            mlflow.log_metric(key, output[key])
            print(f"Logged metric: {key}={output[key]}")

    if "duration_ms" in output and "duration_api_ms" in output:
        duration_tool_ms = output["duration_ms"] - output["duration_api_ms"]
        mlflow.log_metric("duration_tool_ms", duration_tool_ms)
        print(f"Logged metric: duration_tool_ms={duration_tool_ms}")

    token_fields = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
    for model_key, usage in output.get("modelUsage", {}).items():
        if "haiku" in model_key:
            prefix = "haiku"
        elif "sonnet" in model_key:
            prefix = "sonnet"
        else:
            continue
        for field in token_fields:
            if field in usage:
                metric_name = f"{prefix}{field}"
                mlflow.log_metric(metric_name, usage[field])
                print(f"Logged metric: {metric_name}={usage[field]}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Polybench workflow via Claude Code CLI")
    parser.add_argument("--kernel", required=True, help="Kernel name (e.g. threemm)")
    parser.add_argument("--dimension", required=True, help="Dimension (e.g. TEST, SMALL, LARGE)")
    parser.add_argument("--target", required=True, choices=["Baseline", "Transformed", "Optimized"], help="Target type")
    parser.add_argument("--version", default="benchmark", help="Version label (default: benchmark)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = args.kernel
    dimension = args.dimension
    target = args.target
    version = args.version

    run_name = f"{kernel}_{dimension}_{target}"
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name), mlflow.start_span("workflow") as span:
        span.set_attributes({"kernel": kernel, "dimension": dimension, "target": target})
        sync_workflow_prompt(WORKFLOW_PROMPT, kernel, dimension, target)

        # --- Execute the workflow via Claude Code CLI ---
        claude_output = run_workflow(Path(WORKFLOW_PROMPT).read_text())

        # --- Locate the experiment directory produced by the workflow ---
        repo_root = PLUGIN_DIR.parent.parent.parent
        benches_root = repo_root / "benches"
        exp_dir = find_experiment_dir(benches_root)
        if exp_dir is None:
            print("WARNING: no experiment directory found; skipping artifact/param logging")
            return

        print(f"Using experiment directory: {exp_dir}")

        # --- Log params and metrics from bambu_summary.json ---
        params: dict = {
            "kernel": kernel,
            "dimension": dimension,
            "target": target,
            "version": version,
            "session_id": claude_output.get("session_id", ""),
        }
        summary_path = exp_dir / BAMBU_SUMMARY
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            for k in ("device", "clock_period", "memory_allocation_policy"):
                if k in summary:
                    params[k] = summary[k]
            if "total_cycles" in summary:
                mlflow.log_metric("total_cycles", summary["total_cycles"])
                print(f"Logged metric: total_cycles={summary['total_cycles']}")
        else:
            print(f"WARNING: {summary_path} not found; skipping param logging")
        mlflow.log_params(params)
        print(f"Logged params: {params}")

        # --- Log transform schedule artifacts ---
        sched_dir = exp_dir / TRANSFORM_SCHEDULE_DIR
        for artifact_name in ARTIFACTS_TRANSFORM:
            artifact_path = sched_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="transform_schedules")
                print(f"Logged artifact: {artifact_path}")
            else:
                print(f"WARNING: artifact not found: {artifact_path}")

        # --- Log kernel steps artifacts ---
        kernel_steps_dir = exp_dir / KERNEL_STEPS_DIR
        for artifact_name in ARTIFACTS_KERNEL_STEPS:
            artifact_path = kernel_steps_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="kernel_steps")
                print(f"Logged artifact: {artifact_path}")
            else:
                print(f"WARNING: artifact not found: {artifact_path}")


if __name__ == "__main__":
    main()
