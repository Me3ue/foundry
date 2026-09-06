#!/usr/bin/env python
"""Collect PDB-NMF sweep metrics and write paper-ready tables + line plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PREFERRED = ["baseline", "zkp_encoder", "zkp_proj", "zkp_head", "zkp_all"]
COLORS = {
    "baseline": "#4C4C4C",
    "zkp_encoder": "#0072B2",
    "zkp_proj": "#009E73",
    "zkp_head": "#D55E00",
    "zkp_all": "#CC79A7",
}
CURVE_SPECS = [
    ("train/per_epoch_total_loss", "Total diffusion+sequence loss", False),
    ("train/per_epoch_mse_loss_mean", "Coordinate MSE", False),
    ("train/per_epoch_mean_lddt_protein", "Protein lDDT (train batches)", True),
    ("train/per_epoch_seq_recovery", "Sequence recovery", True),
    ("train/per_epoch_token_lvl_sequence_loss", "Token sequence CE", False),
    ("val/pdb_holdout/mean_lddt", "Holdout protein lDDT", True),
]


def _tag_from_name(name: str) -> str:
    n = name.lower()
    for tag in PREFERRED:
        if tag.replace("_", "-") in n or tag in n:
            return tag
    return name


def load_csv_metrics(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for csv_path in sweep_dir.rglob("metrics.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        # Hydra CSVLogger often writes a blank header row; keep numeric epoch/step.
        if "epoch" not in df.columns and "step" in df.columns:
            df["epoch"] = df["step"]
        tag = None
        parts = csv_path.parts
        for p in parts:
            if p in PREFERRED:
                tag = p
                break
        if tag is None:
            tag = csv_path.parent.name
        out[tag] = df
    return out


def last_numeric(df: pd.DataFrame, *keys: str):
    for key in keys:
        if key not in df.columns:
            continue
        s = pd.to_numeric(df[key], errors="coerce").dropna()
        if not s.empty:
            return float(s.iloc[-1])
    return None


def best_numeric(df: pd.DataFrame, key: str, higher_better: bool):
    if key not in df.columns:
        return None
    s = pd.to_numeric(df[key], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.max() if higher_better else s.min())


def style_axes(ax, ylabel: str, higher_better: bool):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, loc="best")
    if higher_better:
        ax.set_title(ylabel + "  (higher is better)", loc="left", fontsize=11)
    else:
        ax.set_title(ylabel + "  (lower is better)", loc="left", fontsize=11)


def plot_curves(metrics: dict[str, pd.DataFrame], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )
    for key, ylabel, higher_better in CURVE_SPECS:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        any_series = False
        for tag in PREFERRED:
            df = metrics.get(tag)
            if df is None or key not in df.columns:
                continue
            y = pd.to_numeric(df[key], errors="coerce")
            x = pd.to_numeric(df.get("epoch", df.get("step")), errors="coerce")
            mask = y.notna() & x.notna()
            if not mask.any():
                continue
            ax.plot(
                x[mask],
                y[mask],
                color=COLORS.get(tag, "black"),
                label=tag.replace("_", " "),
                linewidth=1.8,
                marker="o",
                markersize=3.2,
            )
            any_series = True
        if not any_series:
            plt.close(fig)
            continue
        style_axes(ax, ylabel, higher_better)
        fig.tight_layout()
        out = fig_dir / f"{key.replace('/', '_')}.pdf"
        png = fig_dir / f"{key.replace('/', '_')}.png"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        written.extend([out, png])

    # Combined 2x2 paper figure
    combo_keys = CURVE_SPECS[:4]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for ax, (key, ylabel, higher_better) in zip(axes.ravel(), combo_keys):
        for tag in PREFERRED:
            df = metrics.get(tag)
            if df is None or key not in df.columns:
                continue
            y = pd.to_numeric(df[key], errors="coerce")
            x = pd.to_numeric(df.get("epoch", df.get("step")), errors="coerce")
            mask = y.notna() & x.notna()
            if not mask.any():
                continue
            ax.plot(
                x[mask],
                y[mask],
                color=COLORS.get(tag, "black"),
                label=tag.replace("_", " "),
                linewidth=1.6,
            )
        style_axes(ax, ylabel, higher_better)
    fig.tight_layout()
    combo_pdf = fig_dir / "paper_training_curves.pdf"
    combo_png = fig_dir / "paper_training_curves.png"
    fig.savefig(combo_pdf, bbox_inches="tight")
    fig.savefig(combo_png, bbox_inches="tight")
    plt.close(fig)
    written.extend([combo_pdf, combo_png])
    return written


def write_tables(sweep_dir: Path, metrics: dict[str, pd.DataFrame], ckpt: str) -> tuple[Path, Path, Path]:
    rows = []
    for tag in [t for t in PREFERRED if t in metrics] + [t for t in metrics if t not in PREFERRED]:
        df = metrics[tag]
        summary = {}
        for p in sweep_dir.glob(f"{tag}*.run_summary.json"):
            try:
                summary = json.loads(p.read_text())
                break
            except Exception:
                continue
        rows.append(
            {
                "setting": tag,
                "status": summary.get("status", ""),
                "trainable_params": summary.get("trainable_params"),
                "final_train_loss": last_numeric(df, "train/per_epoch_total_loss"),
                "best_train_loss": best_numeric(df, "train/per_epoch_total_loss", False),
                "final_train_lddt": last_numeric(df, "train/per_epoch_mean_lddt_protein"),
                "best_train_lddt": best_numeric(df, "train/per_epoch_mean_lddt_protein", True),
                "final_seq_recovery": last_numeric(df, "train/per_epoch_seq_recovery"),
                "final_mse": last_numeric(df, "train/per_epoch_mse_loss_mean"),
                "holdout_lddt": last_numeric(df, "val/pdb_holdout/mean_lddt", "val/mean_lddt"),
            }
        )
    csv_path = sweep_dir / "paper_metrics.csv"
    md_path = sweep_dir / "paper_metrics.md"
    json_path = sweep_dir / "paper_metrics.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["setting"])
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps({"ckpt": ckpt, "rows": rows}, indent=2) + "\n")
    md = [
        "# PDB NMF capability summary",
        "",
        f"- Checkpoint: `{ckpt}`",
        f"- Sweep dir: `{sweep_dir}`",
        "",
        "Metrics are PDB reconstruction/denoising quality, not unconditional design.",
        "",
        "| " + " | ".join(rows[0].keys()) + " |" if rows else "",
        "| " + " | ".join(["---"] * len(rows[0])) + " |" if rows else "",
    ]
    for row in rows:
        md.append("| " + " | ".join("" if v is None else str(v) for v in row.values()) + " |")
    md_path.write_text("\n".join(md) + "\n")
    return csv_path, md_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir)
    metrics = load_csv_metrics(sweep_dir)
    fig_dir = sweep_dir / "figures"
    plots = plot_curves(metrics, fig_dir) if metrics else []
    tables = write_tables(sweep_dir, metrics, args.ckpt) if metrics else []
    print(f"Loaded {len(metrics)} runs from {sweep_dir}")
    for p in list(tables) + plots:
        print("Wrote", p)


if __name__ == "__main__":
    main()
