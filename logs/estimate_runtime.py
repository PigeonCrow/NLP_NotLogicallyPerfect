#!/usr/bin/env python3
"""Estimate total training runtime (in minutes) from nanoGPT log files.

The script parses lines like:
    iter 250: loss 1.7989, time 42106.79ms, mfu 1.11%
and uses the observed average ms/iter to project the remaining runtime up to
`max_iters` (if present in the log).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from statistics import median
from typing import List, Optional

MAX_ITERS_RE = re.compile(r"^\s*max_iters\s*=\s*(\d+)")
LOG_INTERVAL_RE = re.compile(r"^\s*log_interval\s*=\s*(\d+)")
ITER_TIME_RE = re.compile(r"iter\s+(\d+):.*?time\s+([0-9]+(?:\.[0-9]+)?)ms")


@dataclass
class RuntimeEstimate:
    path: str
    max_iters: Optional[int]
    last_iter: Optional[int]
    observed_minutes: Optional[float]
    estimated_total_minutes: Optional[float]
    completed: Optional[bool]
    note: str = ""


def parse_log(path: str) -> RuntimeEstimate:
    max_iters: Optional[int] = None
    declared_log_interval: Optional[int] = None
    iter_times: List[tuple[int, float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            max_match = MAX_ITERS_RE.search(line)
            if max_match and max_iters is None:
                max_iters = int(max_match.group(1))

            log_interval_match = LOG_INTERVAL_RE.search(line)
            if log_interval_match and declared_log_interval is None:
                declared_log_interval = int(log_interval_match.group(1))

            iter_time_match = ITER_TIME_RE.search(line)
            if iter_time_match:
                iter_num = int(iter_time_match.group(1))
                time_ms = float(iter_time_match.group(2))
                iter_times.append((iter_num, time_ms))

    if not iter_times:
        return RuntimeEstimate(
            path=path,
            max_iters=max_iters,
            last_iter=None,
            observed_minutes=None,
            estimated_total_minutes=None,
            completed=None,
            note="No 'iter ... time ...ms' entries found",
        )

    observed_ms = sum(ms for _, ms in iter_times)
    observed_minutes = observed_ms / 1000.0 / 60.0

    first_iter = iter_times[0][0]
    last_iter = iter_times[-1][0]

    # Infer step size between logs when not explicitly declared.
    inferred_interval = declared_log_interval
    if inferred_interval is None:
        diffs = [
            iter_times[i][0] - iter_times[i - 1][0]
            for i in range(1, len(iter_times))
            if iter_times[i][0] - iter_times[i - 1][0] > 0
        ]
        if diffs:
            inferred_interval = int(median(diffs))

    if inferred_interval is None or inferred_interval <= 0:
        inferred_interval = 1

    covered_iters = max(last_iter - first_iter, inferred_interval)
    ms_per_iter = observed_ms / covered_iters

    estimated_total_minutes: Optional[float]
    completed: Optional[bool]

    if max_iters is not None:
        remaining_iters = max(0, max_iters - last_iter)
        estimated_total_ms = observed_ms + remaining_iters * ms_per_iter
        estimated_total_minutes = estimated_total_ms / 1000.0 / 60.0
        completed = last_iter >= max_iters
    else:
        estimated_total_minutes = observed_minutes
        completed = None

    return RuntimeEstimate(
        path=path,
        max_iters=max_iters,
        last_iter=last_iter,
        observed_minutes=observed_minutes,
        estimated_total_minutes=estimated_total_minutes,
        completed=completed,
    )


def resolve_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    # Deduplicate while preserving order.
    seen = set()
    unique_files = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique_files.append(path)
    return unique_files


def format_minutes(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate total runtime in minutes from nanoGPT log files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=["*.log"],
        help="Log files or glob patterns (default: *.log in current directory)",
    )
    args = parser.parse_args()

    files = resolve_files(args.files)
    if not files:
        raise SystemExit("No matching log files found.")

    estimates = [parse_log(path) for path in files]

    header = (
        f"{'file':35} {'last_iter':>9} {'max_iters':>9} "
        f"{'observed_min':>13} {'est_total_min':>13} {'status':>11}"
    )
    print(header)
    print("-" * len(header))

    for est in estimates:
        if est.note:
            status = "invalid"
        elif est.completed is True:
            status = "completed"
        elif est.completed is False:
            status = "in-progress"
        else:
            status = "unknown"

        print(
            f"{os.path.basename(est.path):35} "
            f"{str(est.last_iter if est.last_iter is not None else 'n/a'):>9} "
            f"{str(est.max_iters if est.max_iters is not None else 'n/a'):>9} "
            f"{format_minutes(est.observed_minutes):>13} "
            f"{format_minutes(est.estimated_total_minutes):>13} "
            f"{status:>11}"
        )

        if est.note:
            print(f"  note: {est.note}")


if __name__ == "__main__":
    main()
