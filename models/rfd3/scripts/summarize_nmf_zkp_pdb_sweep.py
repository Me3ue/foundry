#!/usr/bin/env python
"""Create paper-ready holdout summaries from one NMF PDB sweep."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TAGS = ["baseline", "zkp_encoder", "zkp_proj", "zkp_head", "zkp_all"]


def bootstrap_ci(values: np.ndarray, seed: int = 42, n: int = 10000):
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def find_validation_csv(root: Path, tag: str) -> Path | None:
    paths = sorted((root / "train" / tag).rglob("validation_output_all_epochs.csv"))
    return paths[-1] if paths else None


def load_tag(root: Path, tag: str):
    path = find_validation_csv(root, tag)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    if "epoch" in df:
        df = df[df["epoch"] == df["epoch"].max()].copy()
    if "dataset" in df:
        df = df[df["dataset"] == "pdb_holdout"].copy()
    lddt_cols = [c for c in df.columns if "lddt" in c.lower()]
    preferred = [
        "lddt.mean_lddt_protein",
        "lddt.mean_lddt",
        "mean_lddt_protein",
        "mean_lddt",
    ]
    metric = next((c for c in preferred if c in df.columns), None)
    if metric is None:
        metric = next((c for c in lddt_cols if "protein" in c.lower()), None)
    if metric is None and len(lddt_cols) == 1:
        metric = lddt_cols[0]
    if metric is None:
        raise RuntimeError(f"{tag}: no lDDT column in {path}; columns={list(df.columns)}")
    out = df[["example_id", metric]].copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=[metric]).drop_duplicates("example_id")
    out = out.rename(columns={metric: "lddt"})
    return out, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    frames = {}
    paths = {}
    for tag in TAGS:
        frame, path = load_tag(args.sweep_dir, tag)
        if frame is not None:
            frames[tag] = frame
            paths[tag] = str(path)

    summary = []
    for tag, df in frames.items():
        x = df["lddt"].to_numpy(float)
        lo, hi = bootstrap_ci(x, seed=args.seed)
        summary.append({
            "setting": tag,
            "n": len(x),
            "coverage": len(x) / 256.0,
            "mean_lddt": np.mean(x),
            "sd_lddt": np.std(x, ddof=1) if len(x) > 1 else np.nan,
            "median_lddt": np.median(x),
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "validation_csv": paths[tag],
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.sweep_dir / "paper_summary.csv", index=False)

    # Preserve the exact per-example values used in all aggregate statistics.
    per_example = None
    for tag, df in frames.items():
        values = df.rename(columns={"lddt": f"{tag}_lddt"})
        per_example = values if per_example is None else per_example.merge(values, on="example_id", how="outer")
    if per_example is not None:
        per_example.to_csv(args.sweep_dir / "paper_per_example_lddt.csv", index=False)

    paired_rows = []
    if "baseline" in frames:
        base = frames["baseline"].rename(columns={"lddt": "baseline_lddt"})
        for tag, df in frames.items():
            if tag == "baseline":
                continue
            joined = base.merge(df.rename(columns={"lddt": f"{tag}_lddt"}), on="example_id")
            delta = (joined[f"{tag}_lddt"] - joined["baseline_lddt"]).to_numpy(float)
            lo, hi = bootstrap_ci(delta, seed=args.seed + 1)
            paired_rows.append({
                "setting": tag,
                "paired_n": len(delta),
                "paired_coverage": len(delta) / max(len(base), 1),
                "mean_delta_lddt": np.mean(delta) if len(delta) else np.nan,
                "sd_delta_lddt": np.std(delta, ddof=1) if len(delta) > 1 else np.nan,
                "median_delta_lddt": np.median(delta) if len(delta) else np.nan,
                "bootstrap_delta_ci95_low": lo,
                "bootstrap_delta_ci95_high": hi,
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(args.sweep_dir / "paper_paired_deltas.csv", index=False)

    lines = [
        "# Paper-level NMF PDB holdout summary", "",
        "Metric: per-example validation lDDT, with the exact same temporal holdout and crop configuration for every setting.",
        "Validation is run once after each task completes training.", "",
        "## Per-setting statistics", "",
    ]
    lines.append(summary_df.to_markdown(index=False) if not summary_df.empty else "No successful validation results.")
    lines += ["", "## Paired baseline deltas", ""]
    lines.append(paired_df.to_markdown(index=False) if not paired_df.empty else "No baseline pairing available.")
    (args.sweep_dir / "paper_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.sweep_dir / 'paper_summary.csv'}")
    print(f"Wrote {args.sweep_dir / 'paper_paired_deltas.csv'}")
    print(f"Wrote {args.sweep_dir / 'paper_per_example_lddt.csv'}")
    print(f"Wrote {args.sweep_dir / 'paper_summary.md'}")


if __name__ == "__main__":
    main()
