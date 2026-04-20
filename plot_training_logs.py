#!/usr/bin/env python3
import argparse
import importlib
import re
from pathlib import Path


STEP_PATTERN = re.compile(
    r"step\s+(?P<step>\d+):\s+train\s+loss\s+(?P<train>[0-9]*\.?[0-9]+),\s+val\s+loss\s+(?P<val>[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def get_pyplot():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc


def parse_log_file(log_path: Path):
    steps = []
    train_losses = []
    val_losses = []

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = STEP_PATTERN.search(line)
        if not match:
            continue

        steps.append(int(match.group("step")))
        train_losses.append(float(match.group("train")))
        val_losses.append(float(match.group("val")))

    return steps, train_losses, val_losses


def parse_logs(log_files):
    runs = {}
    for log_file in log_files:
        steps, train_losses, val_losses = parse_log_file(log_file)
        if not steps:
            continue
        runs[log_file.stem] = {
            "steps": steps,
            "train": train_losses,
            "val": val_losses,
        }
    return runs


def plot_metric(plt, runs, metric_key, ylabel, title, output_path: Path):
    plt.figure(figsize=(9, 5))
    for run_name, data in runs.items():
        plt.plot(data["steps"], data[metric_key], label=run_name)

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def collect_log_files(inputs):
    collected = []
    for item in inputs:
        path = Path(item)
        if any(ch in item for ch in "*?[]"):
            collected.extend(Path().glob(item))
        elif path.is_dir():
            collected.extend(sorted(path.glob("*.log")))
        else:
            collected.append(path)

    unique_files = []
    seen = set()
    for file_path in collected:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(file_path)

    return sorted(unique_files)


def main():
    parser = argparse.ArgumentParser(
        description="Parse nanoGPT log files and plot train/validation loss vs. step.",
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Log files, directories, or globs (e.g., logs/*.log)",
    )
    parser.add_argument(
        "--outdir",
        default="plots",
        help="Directory for output figures (default: plots)",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Figure output format (default: png)",
    )
    args = parser.parse_args()

    plt = get_pyplot()

    log_files = collect_log_files(args.logs)
    existing_logs = [f for f in log_files if f.exists() and f.is_file()]
    if not existing_logs:
        raise SystemExit("No valid log files found.")

    runs = parse_logs(existing_logs)
    if not runs:
        raise SystemExit("No step/train/val entries found in the provided log files.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_out = outdir / f"train_loss_vs_step.{args.format}"
    val_out = outdir / f"val_loss_vs_step.{args.format}"

    plot_metric(
        plt=plt,
        runs=runs,
        metric_key="train",
        ylabel="Training Loss",
        title="Training Loss vs Step",
        output_path=train_out,
    )
    plot_metric(
        plt=plt,
        runs=runs,
        metric_key="val",
        ylabel="Validation Loss",
        title="Validation Loss vs Step",
        output_path=val_out,
    )

    print(f"Saved: {train_out}")
    print(f"Saved: {val_out}")


if __name__ == "__main__":
    main()
