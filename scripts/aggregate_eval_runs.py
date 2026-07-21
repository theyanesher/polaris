"""
Aggregate eval_results.csv files across a sweep folder into per-seed and
per-run-averaged tables.

Each run subfolder is expected to be named "<run_name>_seed<N>" and contain
an eval_results.csv with a trailing "summary" row holding num_success,
success_rate, avg_progress, avg_episode_length.

Usage:
    python polaris/scripts/aggregate_eval_runs.py <sweep_folder> [--out prefix]
    python polaris/scripts/aggregate_eval_runs.py runs/egoverse_franka_only_sweep_999_20260718_100550
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from math import sqrt
from pathlib import Path

SUMMARY_ROW = "summary"
SEED_RE = re.compile(r"^(?P<run_name>.+)_seed(?P<seed>\d+)$")

FIELDS = ["num_success", "success_rate", "avg_progress", "avg_episode_length"]

# Two-tailed 95% critical t-values by degrees of freedom (n_seeds - 1).
# Falls back to the normal-approximation value (1.96) beyond df=30.
T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def ci95_halfwidth(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    std = statistics.stdev(values)
    t_val = T_TABLE_95.get(n - 1, 1.96)
    return t_val * std / sqrt(n)


def load_summary(csv_path: Path) -> dict | None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    num_episodes = sum(1 for r in rows if r["episode"] != SUMMARY_ROW)
    summary_row = next((r for r in rows if r["episode"] == SUMMARY_ROW), None)
    if summary_row is None:
        return None
    return {"num_episodes": num_episodes, **{k: float(summary_row[k]) for k in FIELDS}}


def collect(sweep_folder: Path) -> list[dict]:
    records = []
    for run_dir in sorted(sweep_folder.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "eval_results.csv"
        if not csv_path.exists():
            continue
        match = SEED_RE.match(run_dir.name)
        run_name, seed = (match.group("run_name"), match.group("seed")) if match else (run_dir.name, "")
        summary = load_summary(csv_path)
        if summary is None:
            continue
        records.append({"run_name": run_name, "seed": seed, **summary})
    return sorted(records, key=lambda r: (r["run_name"], r["seed"]))


def build_avg_table(per_seed: list[dict]) -> list[dict]:
    by_run = defaultdict(list)
    for r in per_seed:
        by_run[r["run_name"]].append(r)

    avg_rows = []
    for run_name in sorted(by_run):
        rows = by_run[run_name]
        avg_row = {"run_name": run_name, "num_seeds": len(rows)}
        for k in FIELDS:
            values = [r[k] for r in rows]
            avg_row[k] = sum(values) / len(values)
            avg_row[f"{k}_ci95"] = ci95_halfwidth(values)
        avg_rows.append(avg_row)
    return avg_rows


def print_table(rows: list[dict], columns: list[str]):
    widths = {c: max(len(c), *(len(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        line = []
        for c in columns:
            v = r[c]
            v = f"{v:.4f}" if isinstance(v, float) else str(v)
            line.append(v.ljust(widths[c]))
        print("  ".join(line))


def write_csv(rows: list[dict], columns: list[str], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_folder", type=str, help="Folder containing per-seed run subfolders")
    parser.add_argument("--out", type=str, default=None, help="Output prefix (default: <sweep_folder>/eval_summary)")
    args = parser.parse_args()

    sweep_folder = Path(args.sweep_folder)
    per_seed = collect(sweep_folder)
    if not per_seed:
        print(f"No eval_results.csv found under {sweep_folder}")
        return

    avg_table = build_avg_table(per_seed)

    out_prefix = Path(args.out) if args.out else sweep_folder / "eval_summary"
    per_seed_path = out_prefix.with_name(out_prefix.name + "_per_seed.csv")
    avg_path = out_prefix.with_name(out_prefix.name + "_avg.csv")

    per_seed_cols = ["run_name", "seed", "num_episodes", *FIELDS]
    avg_cols = ["run_name", "num_seeds", *(c for k in FIELDS for c in (k, f"{k}_ci95"))]

    write_csv(per_seed, per_seed_cols, per_seed_path)
    write_csv(avg_table, avg_cols, avg_path)

    print("Per-seed results:")
    print_table(per_seed, per_seed_cols)
    print(f"\nSaved to {per_seed_path}")

    print("\nPer-run averages (across seeds):")
    print_table(avg_table, avg_cols)
    print(f"\nSaved to {avg_path}")


if __name__ == "__main__":
    main()
