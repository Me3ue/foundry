#!/usr/bin/env python3
"""Aggregate all completed RFD3 paper-matrix CA-RMSD evaluations.

Writes a sample-level table and target/condition summary with 95% bootstrap
confidence intervals. Missing/failed runs are explicitly listed rather than
silently omitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_mean_ci(values: list[float], iterations: int, rng: random.Random) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(iterations)]
    return quantile(means, 0.025), quantile(means, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20250308)
    args = parser.parse_args()
    metrics_paths = sorted(args.out_root.rglob("near_native_metrics.csv"))
    if not metrics_paths:
        parser.error("No near_native_metrics.csv files found.")

    rows, present_run_dirs = [], set()
    for metrics_path in metrics_paths:
        # expected layout: <root>/<target>/<condition>/<seed>/near_native_metrics.csv
        relative = metrics_path.relative_to(args.out_root)
        if len(relative.parts) != 4:
            print(f"WARNING: ignoring unexpected layout: {metrics_path}")
            continue
        target, condition, seed_dir, _ = relative.parts
        with metrics_path.open() as handle:
            for row in csv.DictReader(handle):
                row.update(target=target, condition=condition, seed=seed_dir.removeprefix("seed_"))
                row["ca_rmsd_angstrom"] = float(row["ca_rmsd_angstrom"])
                row["ca_coverage"] = float(row["ca_coverage"])
                row["passes_near_native_threshold"] = row["passes_near_native_threshold"].strip().lower() == "true"
                rows.append(row)
        present_run_dirs.add(metrics_path.parent)
    if not rows:
        parser.error("No evaluable rows found.")

    long_fields = ["target", "condition", "seed", "model", "ca_rmsd_angstrom", "matched_ca", "reference_ca", "ca_coverage", "passes_near_native_threshold"]
    with (args.out_root / "paper_metrics_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["target"], row["condition"])].append(row)
    rng = random.Random(args.bootstrap_seed)
    summary_fields = ["target", "condition", "n_samples", "n_seeds", "n_pass", "pass_rate", "ca_rmsd_mean_angstrom", "ca_rmsd_median_angstrom", "ca_rmsd_iqr_angstrom", "ca_rmsd_ci95_low_angstrom", "ca_rmsd_ci95_high_angstrom", "ca_rmsd_min_angstrom", "ca_rmsd_max_angstrom", "ca_coverage_mean", "ca_coverage_min"]
    summary_rows = []
    for (target, condition), group in sorted(grouped.items()):
        rmsd = [r["ca_rmsd_angstrom"] for r in group]
        coverage = [r["ca_coverage"] for r in group]
        low, high = bootstrap_mean_ci(rmsd, args.bootstrap_iterations, rng)
        summary_rows.append({
            "target": target, "condition": condition, "n_samples": len(group),
            "n_seeds": len({r["seed"] for r in group}),
            "n_pass": sum(r["passes_near_native_threshold"] for r in group),
            "pass_rate": sum(r["passes_near_native_threshold"] for r in group) / len(group),
            "ca_rmsd_mean_angstrom": statistics.mean(rmsd),
            "ca_rmsd_median_angstrom": statistics.median(rmsd),
            "ca_rmsd_iqr_angstrom": quantile(rmsd, .75) - quantile(rmsd, .25),
            "ca_rmsd_ci95_low_angstrom": low, "ca_rmsd_ci95_high_angstrom": high,
            "ca_rmsd_min_angstrom": min(rmsd), "ca_rmsd_max_angstrom": max(rmsd),
            "ca_coverage_mean": statistics.mean(coverage), "ca_coverage_min": min(coverage),
        })
    with (args.out_root / "paper_metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    expected_run_dirs = {path.parent for path in args.out_root.rglob("run_manifest.json")}
    missing = sorted(str(path) for path in expected_run_dirs - present_run_dirs)
    report = {"n_sample_rows": len(rows), "n_completed_runs": len(present_run_dirs),
              "runs_missing_evaluation": missing, "summary_file": "paper_metrics_summary.csv",
              "sample_file": "paper_metrics_samples.csv"}
    (args.out_root / "paper_metrics_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
