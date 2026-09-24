#!/usr/bin/env python
"""Create paper-ready holdout summaries from a single-layer NMF PDB sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def discover_tags(root: Path) -> list[str]:
    """Discover current single-layer jobs from the sweep output, not old group names."""
    train_root = root / "train"
    if not train_root.exists():
        return []
    tags = [p.name for p in train_root.iterdir() if p.is_dir()]
    return sorted(tags, key=lambda tag: (tag != "baseline", tag))


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
    if "example_id" not in df.columns:
        raise RuntimeError(f"{tag}: validation CSV has no example_id column: {path}")
    out = df[["example_id", metric]].copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=[metric]).drop_duplicates("example_id")
    out = out.rename(columns={metric: "lddt"})
    return out, path


def replacement_metadata(root: Path, tag: str) -> dict:
    candidates = [
        root / f"{tag}.nmf_replacements.json",
        *root.glob(f"*{tag}*.nmf_replacements.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        records = payload.get("records", [])
        names = [str(record.get("module_name", "")) for record in records]
        return {
            "replacement_json": str(path),
            "replaced_layers": "; ".join(names),
            "rank": payload.get("rank"),
            "nmf_params": sum(record.get("nmf_params", 0) or 0 for record in records),
        }
    return {
        "replacement_json": "",
        "replaced_layers": "baseline" if tag == "baseline" else "metadata_missing",
        "rank": "",
        "nmf_params": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tags = discover_tags(args.sweep_dir)
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for tag in tags:
        try:
            frame, path = load_tag(args.sweep_dir, tag)
        except RuntimeError as exc:
            skipped[tag] = str(exc)
            continue
        if frame is not None:
            frames[tag] = frame
            paths[tag] = str(path)
        else:
            skipped[tag] = "validation_output_all_epochs.csv not found"

    baseline_n = len(frames["baseline"]) if "baseline" in frames else max(
        (len(frame) for frame in frames.values()), default=0
    )
    summary = []
    for tag, df in frames.items():
        x = df["lddt"].to_numpy(float)
        lo, hi = bootstrap_ci(x, seed=args.seed)
        metadata = replacement_metadata(args.sweep_dir, tag)
        summary.append(
            {
                "setting": tag,
                "n": len(x),
                "coverage": len(x) / baseline_n if baseline_n else np.nan,
                "mean_lddt": np.mean(x),
                "sd_lddt": np.std(x, ddof=1) if len(x) > 1 else np.nan,
                "median_lddt": np.median(x),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "replaced_layers": metadata["replaced_layers"],
                "rank": metadata["rank"],
                "nmf_params": metadata["nmf_params"],
                "validation_csv": paths[tag],
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.sweep_dir / "paper_summary.csv", index=False)

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
            paired_rows.append(
                {
                    "setting": tag,
                    "paired_n": len(delta),
                    "paired_coverage": len(delta) / max(len(base), 1),
                    "mean_delta_lddt": np.mean(delta) if len(delta) else np.nan,
                    "sd_delta_lddt": np.std(delta, ddof=1) if len(delta) > 1 else np.nan,
                    "median_delta_lddt": np.median(delta) if len(delta) else np.nan,
                    "bootstrap_delta_ci95_low": lo,
                    "bootstrap_delta_ci95_high": hi,
                }
            )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(args.sweep_dir / "paper_paired_deltas.csv", index=False)

    lines = [
        "# Paper-level NMF PDB holdout summary",
        "",
        "Metric: per-example validation lDDT on the same temporal PDB holdout for every setting.",
        "Each current sweep job replaces one exact-matched Linear only.",
        "Validation is run once after each task completes training.",
        "",
        "## Discovered settings",
        "",
        ", ".join(tags) if tags else "No sweep jobs discovered.",
        "",
        "## Per-setting statistics",
        "",
    ]
    lines.append(summary_df.to_markdown(index=False) if not summary_df.empty else "No successful validation results.")
    lines += ["", "## Paired baseline deltas", ""]
    lines.append(paired_df.to_markdown(index=False) if not paired_df.empty else "No baseline pairing available.")
    if skipped:
        lines += ["", "## Skipped settings", ""]
        lines.extend(f"- `{tag}`: {reason}" for tag, reason in skipped.items())
    (args.sweep_dir / "paper_summary.md").write_text("\n".join(lines) + "\n")

    print(f"Discovered settings: {', '.join(tags) if tags else '(none)'}")
    print(f"Wrote {args.sweep_dir / 'paper_summary.csv'}")
    print(f"Wrote {args.sweep_dir / 'paper_paired_deltas.csv'}")
    print(f"Wrote {args.sweep_dir / 'paper_per_example_lddt.csv'}")
    print(f"Wrote {args.sweep_dir / 'paper_summary.md'}")
    if skipped:
        print(f"Skipped {len(skipped)} settings without usable holdout validation.")


if __name__ == "__main__":
    main()
