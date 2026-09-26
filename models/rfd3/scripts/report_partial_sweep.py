#!/usr/bin/env python
"""Paper-level report from a partially finished NMF/ZKP sweep.

Why this exists
---------------
The sweep writes its paper-level rollup (comparison_table, paper_summary,
figures/, paper_training_cost) only after the LAST job in the sequential loop
finishes. When the clock runs out and only some layers were trained -- or a job
was killed mid-training -- those files are never produced, even though every
per-job artefact (run.log, metrics.csv, val_metrics CSV, run_summary.json) is
already on disk.

This script reads whatever exists and builds the report anyway, but it never
pretends the missing parts are there: every setting carries an explicit
completeness label, and a killed job is reported as training-curves-only because
the holdout validation runs once, after fit, and therefore does not exist for it.

Definitions are imported from the existing scripts (summarize_nmf_zkp_pdb_sweep
for the lDDT/bootstrapping, plot_nmf_zkp_pdb_metrics for the curve keys) so the
numbers here are identical to the ones the full pipeline would produce.

Usage
-----
    python models/rfd3/scripts/report_partial_sweep.py --out OUT_DIR \\
        [--discover LOG_ROOT] [SWEEP_DIR ...]

    # one sweep dir produced by a single invocation
    python models/rfd3/scripts/report_partial_sweep.py --out /tmp/report \\
        /backup01/zzj/logs/train_nmf_zkp_pdb/sweep_nmf_zkp_pdb_2026-09-26_08-00-00

    # several dirs (baseline / encoder / head run separately)
    python models/rfd3/scripts/report_partial_sweep.py --out /tmp/report \\
        --discover /backup01/zzj/logs/train_nmf_zkp_pdb

When the same tag exists in more than one dir, a copy that has a validation CSV
wins over one without; ties go to the earliest dir listed (newest first for
--discover). Nothing is copied between dirs: the report reads them in place.

Outputs (all inside --out)
--------------------------
    partial_report.md              main human-readable report
    partial_summary.csv            one row per setting, every field merged
    partial_paired_deltas.csv      per-example delta vs baseline + bootstrap CI
    partial_per_example_lddt.csv   wide per-example lDDT matrix
    partial_status.json            machine-readable inventory of what was found
    figures/partial_holdout_lddt.{pdf,png}
    figures/partial_paired_deltas.{pdf,png}
    figures/partial_training_progress.{pdf,png}
    figures/partial_training_curves.{pdf,png}   (only if metrics.csv exists)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the official definitions instead of re-deriving them, so a number in
# this report always equals the number in paper_summary.csv.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_nmf_zkp_pdb_metrics as splot  # noqa: E402
import summarize_nmf_zkp_pdb_sweep as ssum  # noqa: E402

SUMMARY_SUFFIX = ".run_summary.json"
STEP_RE = re.compile(r"\.step(\d+)$")
BOX = "\u2502\u2503|"
DEFAULT_LOG_ROOT = Path("/backup01/zzj/logs/train_nmf_zkp_pdb")

# Which proof family each single-layer job belongs to (same grouping the sweep
# script prints in comparison_table.md).
FAMILY_OF = {
    "baseline": ("none", "full fine-tune of the released checkpoint"),
    "process_s_init": ("encoder", "input-integrity M"),
    "process_z_init": ("encoder", "input-integrity M"),
    "process_c": ("encoder", "input-integrity M"),
    "process_s_trunk": ("encoder", "input-integrity M"),
    "transition_post_token_l1": ("encoder", "input-integrity M"),
    "transition_post_atom_l1": ("encoder", "input-integrity M"),
    "process_pll": ("proj", "intermediate-state M"),
    "project_pll": ("proj", "intermediate-state M"),
    "process_n_atom": ("proj", "intermediate-state M"),
    "process_n_token": ("proj", "intermediate-state M"),
    "process_single_l": ("proj", "intermediate-state M"),
    "process_z": ("proj", "intermediate-state M"),
    "to_r_update": ("head", "output-consistency M"),
    "sequence_head": ("head", "output-consistency M"),
}
FAMILY_MEMBERS = {
    "encoder": [t for t, (f, _) in FAMILY_OF.items() if f == "encoder"],
    "proj": [t for t, (f, _) in FAMILY_OF.items() if f == "proj"],
    "head": [t for t, (f, _) in FAMILY_OF.items() if f == "head"],
}

INK = "#1a1a1a"
MUTED = "#5a5a5a"
BASELINE_COLOR = "#4C4C4C"
ACCENT = "#0072B2"
WARN = "#D55E00"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame) -> str:
    """to_markdown needs tabulate; degrade to plain text instead of crashing."""
    if df.empty:
        return "_(no rows)_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def fmt(value, digits: int = 4, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) or isinstance(value, np.floating):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return str(value)


def tag_of_summary_file(path: Path) -> tuple[str, int]:
    stem = path.name
    if stem.endswith(SUMMARY_SUFFIX):
        stem = stem[: -len(SUMMARY_SUFFIX)]
    match = STEP_RE.search(stem)
    return (stem[: match.start()], int(match.group(1))) if match else (stem, -1)


def parse_run_log(path: Path) -> dict:
    """Recover per-epoch progress/timing from the captured stdout.

    train_logging prints an `Epoch N Summary` rich table every epoch; the box
    drawing characters are stripped by the regex and a value wrapped onto the
    following line is still accepted. This is the only timing source for a job
    that was killed before train_lora wrote its run_summary.json.
    """
    out: dict = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")

    def series(label: str) -> list[float]:
        pattern = re.compile(
            re.escape(label)
            + rf"[^\S\n]*[{BOX}]?[^\S\n]*(?:\n[^\S\n]*[{BOX}]?[^\S\n]*)?"
            + r"(-?[0-9]+(?:\.[0-9]+)?)"
        )
        return [float(m.group(1)) for m in pattern.finditer(text)]

    elapsed = series("Elapsed Time (s)")
    per_batch = series("Mean Time per Batch (s)")
    steps = series("Total Optimizer Steps")
    out["log_epochs"] = len(elapsed)
    out["log_total_seconds"] = float(sum(elapsed)) if elapsed else None
    out["log_mean_s_per_batch"] = float(np.mean(per_batch)) if per_batch else None
    out["log_last_optimizer_steps"] = int(steps[-1]) if steps else None
    out["log_per_epoch_seconds"] = elapsed
    max_epochs = re.search(r"max_epochs[=:\s]+(\d+)", text)
    out["log_planned_max_epochs"] = int(max_epochs.group(1)) if max_epochs else None
    traceback_count = text.count("Traceback (most recent call last)")
    out["log_tracebacks"] = traceback_count
    return out


def load_csv_metrics_from(root: Path) -> dict[str, pd.DataFrame]:
    """Per-tag metrics.csv, tolerating several runs of the same tag.

    splot.load_csv_metrics keys by directory name, which silently keeps only one
    copy when a tag ran twice; prefer the highest epoch count instead.
    """
    out: dict[str, pd.DataFrame] = {}
    for tag, df in splot.load_csv_metrics(root).items():
        epoch_col = "epoch" if "epoch" in df.columns else None
        if tag in out and epoch_col:
            prev = out[tag]
            if epoch_col in prev.columns and len(prev) >= len(df):
                continue
        out[tag] = df
    return out


class Setting:
    """Everything known about one sweep tag, wherever it was found."""

    def __init__(self, tag: str, source: Path):
        self.tag = tag
        self.source = source
        self.family, self.proof_target = FAMILY_OF.get(tag, ("(unmapped)", ""))
        self.validation_csv: Path | None = None
        self.metrics_csv: Path | None = None
        self.run_log: Path | None = None
        self.run_summary_path: Path | None = None
        self.summary: dict = {}
        self.log_info: dict = {}
        self.metrics: pd.DataFrame | None = None
        self.frame: pd.DataFrame | None = None  # per-example lDDT
        self.lddt_col: str | None = None

    # ---- loading ---------------------------------------------------------- #
    def load(self) -> None:
        self.validation_csv = ssum.find_validation_csv(self.source, self.tag)
        if self.validation_csv is not None:
            frame, path = ssum.load_tag(self.source, self.tag)
            self.frame, self.validation_csv = frame, path
            self.lddt_col = self._lddt_column(path)
        metrics_paths = sorted((self.source / "train" / self.tag).rglob("metrics.csv"))
        if metrics_paths:
            self.metrics_csv = metrics_paths[-1]
        log_paths = sorted(self.source.glob(f"{self.tag}.run.log"))
        if log_paths:
            self.run_log = log_paths[-1]
        summaries = [
            p
            for p in self.source.glob(f"*{SUMMARY_SUFFIX}")
            if tag_of_summary_file(p)[0] == self.tag
        ]
        if summaries:
            best = max(summaries, key=lambda p: (tag_of_summary_file(p)[1], p.name))
            self.run_summary_path = best
            try:
                self.summary = json.loads(best.read_text(encoding="utf-8"))
            except Exception:
                self.summary = {}
        self.log_info = parse_run_log(self.run_log) if self.run_log else {}
        if self.metrics_csv is not None:
            try:
                self.metrics = pd.read_csv(self.metrics_csv)
            except Exception:
                self.metrics = None

    @staticmethod
    def _lddt_column(path: Path) -> str | None:
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            return None
        for name in (
            "lddt.mean_lddt_protein",
            "lddt.mean_lddt",
            "mean_lddt_protein",
            "mean_lddt",
        ):
            if name in cols:
                return name
        return next((c for c in cols if "lddt" in c.lower() and "protein" in c.lower()), None)

    # ---- derived ---------------------------------------------------------- #
    @property
    def status(self) -> str:
        if self.frame is not None and self.run_summary_path is not None:
            return "complete"
        if self.frame is not None:
            return "validated-no-summary"
        if self.metrics is not None:
            return "training-only (killed before holdout)"
        return "no-artifacts"

    @property
    def has_holdout(self) -> bool:
        return self.frame is not None

    @property
    def epochs_completed(self):
        if self.metrics is not None and "epoch" in self.metrics.columns:
            s = pd.to_numeric(self.metrics["epoch"], errors="coerce").dropna()
            if not s.empty:
                return int(s.max())
        return self.log_info.get("log_epochs") or None

    @property
    def wall_seconds(self):
        cost = (self.summary or {}).get("cost", {}) or {}
        if cost.get("wall_time_seconds") is not None:
            return float(cost["wall_time_seconds"]), "run_summary.json"
        if self.log_info.get("log_total_seconds"):
            return float(self.log_info["log_total_seconds"]), "run.log (sum of epoch elapsed)"
        return None, ""

    def cost_field(self, key: str):
        return ((self.summary or {}).get("cost", {}) or {}).get(key)

    def param(self, key: str, alt: str | None = None):
        value = self.summary.get(key)
        if value is None and alt:
            value = self.summary.get(alt)
        return value

    def replacement(self) -> dict:
        if self.tag == "baseline":
            # The baseline job disables NMF entirely, so there is no replacement
            # metadata and no rank to report.
            return {"rank": None, "nmf_params": None, "replaced_layers": "baseline (no replacement)"}
        meta = ssum.replacement_metadata(self.source, self.tag)
        if meta.get("replaced_layers") in ("", "metadata_missing"):
            nmf_cfg = (self.summary or {}).get("nmf") or {}
            rank = nmf_cfg.get("rank")
            meta["replaced_layers"] = "metadata_missing"
            meta["rank"] = rank
            if rank:
                meta["nmf_params"] = None
        return meta


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def discover(sweep_dirs: list[Path]) -> tuple[dict[str, Setting], list[tuple[str, Path, str]]]:
    """Pick one source dir per tag and note any copies that were passed over."""
    chosen: dict[str, Setting] = {}
    notes: list[tuple[str, Path, str]] = []
    for directory in sweep_dirs:
        tags: set[str] = set()
        train_root = directory / "train"
        if train_root.exists():
            tags.update(p.name for p in train_root.iterdir() if p.is_dir())
        for path in directory.glob(f"*{SUMMARY_SUFFIX}"):
            tags.add(tag_of_summary_file(path)[0])
        for path in directory.glob("*.exit_code"):
            tags.add(path.name[: -len(".exit_code")])
        for tag in sorted(tags):
            if tag.endswith(".step") or tag.startswith("."):
                continue
            probe = Setting(tag, directory)
            probe.validation_csv = ssum.find_validation_csv(directory, tag)
            has_metric = bool(sorted((directory / "train" / tag).rglob("metrics.csv")))
            if tag in chosen:
                current = chosen[tag]
                if probe.validation_csv is not None and current.frame is None:
                    notes.append((tag, current.source, f"superseded by {directory} (that copy has a validation CSV)"))
                    chosen[tag] = probe
                else:
                    notes.append((tag, directory, f"ignored (kept {current.source})"))
                continue
            if probe.validation_csv is None and not has_metric:
                notes.append((tag, directory, "ignored (no validation CSV and no metrics.csv)"))
                continue
            chosen[tag] = probe
    for setting in chosen.values():
        setting.load()
    return chosen, notes


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def build_summary(settings: dict[str, Setting], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    baseline = settings.get("baseline")
    baseline_n = len(baseline.frame) if baseline is not None and baseline.frame is not None else None
    if baseline_n is None:
        baseline_n = max((len(s.frame) for s in settings.values() if s.frame is not None), default=0) or None

    rows, per_example, paired_rows = [], None, []
    for tag, setting in settings.items():
        row: dict = {
            "setting": tag,
            "family": setting.family,
            "proof_target": setting.proof_target,
            "status": setting.status,
        }
        if setting.frame is not None:
            x = setting.frame["lddt"].to_numpy(float)
            lo, hi = ssum.bootstrap_ci(x, seed=seed)
            row.update(
                {
                    "n": len(x),
                    "coverage": (len(x) / baseline_n) if baseline_n else np.nan,
                    "mean_lddt": float(np.mean(x)),
                    "sd_lddt": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                    "median_lddt": float(np.median(x)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "lddt_column": setting.lddt_col,
                }
            )
            values = setting.frame.rename(columns={"lddt": f"{tag}_lddt"})
            per_example = values if per_example is None else per_example.merge(values, on="example_id", how="outer")

        if baseline is not None and baseline.frame is not None and setting.frame is not None and tag != "baseline":
            joined = baseline.frame.rename(columns={"lddt": "baseline_lddt"}).merge(
                setting.frame.rename(columns={"lddt": f"{tag}_lddt"}), on="example_id"
            )
            delta = (joined[f"{tag}_lddt"] - joined["baseline_lddt"]).to_numpy(float)
            lo, hi = ssum.bootstrap_ci(delta, seed=seed + 1)
            row.update(
                {
                    "paired_n": len(delta),
                    "paired_coverage": len(delta) / max(len(baseline.frame), 1),
                    "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
                    "sd_delta": float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan,
                    "median_delta": float(np.median(delta)) if len(delta) else np.nan,
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                }
            )
            paired_rows.append(
                {
                    "setting": tag,
                    "family": setting.family,
                    "paired_n": len(delta),
                    "mean_delta_lddt": row["mean_delta"],
                    "sd_delta_lddt": row["sd_delta"],
                    "median_delta_lddt": row["median_delta"],
                    "bootstrap_delta_ci95_low": lo,
                    "bootstrap_delta_ci95_high": hi,
                    "baseline_n": len(baseline.frame),
                }
            )

        # training-side numbers
        if setting.metrics is not None:
            df = setting.metrics
            row.update(
                {
                    "epochs_completed": setting.epochs_completed,
                    "final_train_loss": splot.last_numeric(df, "train/per_epoch_total_loss"),
                    "best_train_loss": splot.best_numeric(df, "train/per_epoch_total_loss", False),
                    "final_train_lddt": splot.last_numeric(df, "train/per_epoch_mean_lddt_protein"),
                    "best_train_lddt": splot.best_numeric(df, "train/per_epoch_mean_lddt_protein", True),
                    "final_seq_recovery": splot.last_numeric(df, "train/per_epoch_seq_recovery"),
                    "final_mse": splot.last_numeric(df, "train/per_epoch_mse_loss_mean"),
                }
            )
        else:
            row.update(
                {
                    "epochs_completed": setting.epochs_completed,
                    "final_train_loss": None,
                    "best_train_loss": None,
                    "final_train_lddt": None,
                    "best_train_lddt": None,
                    "final_seq_recovery": None,
                    "final_mse": None,
                }
            )

        wall, wall_source = setting.wall_seconds
        trainable = setting.param("trainable_params")
        total = setting.param("total_params")
        row.update(
            {
                "optimizer_steps": setting.summary.get("global_step")
                or setting.log_info.get("log_last_optimizer_steps"),
                "planned_epochs": setting.log_info.get("log_planned_max_epochs"),
                "wall_time_seconds": wall,
                "wall_time_source": wall_source,
                "mean_s_per_batch": setting.log_info.get("log_mean_s_per_batch"),
                "gpu_peak_gib": (setting.cost_field("gpu_peak_allocated_bytes") or 0) / 2**30
                if setting.cost_field("gpu_peak_allocated_bytes")
                else None,
                "cpu_max_rss_gib": (setting.cost_field("cpu_max_rss_bytes") or 0) / 2**30
                if setting.cost_field("cpu_max_rss_bytes")
                else None,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_fraction": (trainable / total) if trainable and total else None,
                "rank": setting.replacement().get("rank"),
                "nmf_params": setting.replacement().get("nmf_params"),
                "replaced_layers": setting.replacement().get("replaced_layers"),
            }
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["_f"] = summary["family"].map({"none": 1, "encoder": 2, "proj": 3, "head": 4}).fillna(9)
        summary["_c"] = ~summary["status"].eq("complete")
        summary = summary.sort_values(["_f", "_c", "setting"]).drop(columns=["_f", "_c"]).reset_index(drop=True)
        # Keep counts integral in the CSV; a single NaN would otherwise turn the
        # whole column into floats (8.0 instead of 8).
        for column in ("n", "paired_n", "epochs_completed", "planned_epochs", "optimizer_steps",
                       "trainable_params", "total_params", "nmf_params", "rank"):
            if column in summary.columns:
                summary[column] = pd.to_numeric(summary[column], errors="coerce")
                if column != "rank":  # rank may legitimately be absent for baseline
                    summary[column] = summary[column].astype("Int64")
        # Keep the CSV readable: 6 decimals is well beyond what lDDT needs and
        # avoids values like 0.39000000000000007.
        float_cols = summary.select_dtypes(include="float").columns
        summary[float_cols] = summary[float_cols].round(6)
        summary["status_detail"] = refine_status(summary)
    paired = pd.DataFrame(paired_rows)
    return summary, paired, per_example


def refine_status(summary: pd.DataFrame) -> pd.Series:
    """Separate "ran out of schedule" from "schedule finished, holdout missing".

    A job killed between the last epoch and the holdout pass has full training
    curves but no validation CSV, which reads very differently from one killed
    half-way through training. Compare against the longest completed run.
    """
    if "epochs_completed" not in summary.columns:
        return summary["status"]
    complete = summary[summary["status"] == "complete"]["epochs_completed"].dropna()
    reference = int(complete.max()) if not complete.empty else None
    detail = []
    for _, row in summary.iterrows():
        status, epochs = row["status"], row.get("epochs_completed")
        if status != "complete" and reference and pd.notna(epochs) and float(epochs) >= 0.98 * reference:
            detail.append(f"trained {int(epochs)} epochs, holdout missing")
        else:
            detail.append(status)
    return pd.Series(detail, index=summary.index)


def family_coverage(settings: dict[str, Setting]) -> pd.DataFrame:
    rows = []
    for family, members in [("encoder", FAMILY_MEMBERS["encoder"]), ("proj", FAMILY_MEMBERS["proj"]), ("head", FAMILY_MEMBERS["head"])]:
        done = [t for t in members if settings.get(t) is not None and settings[t].has_holdout]
        partial = [
            t
            for t in members
            if settings.get(t) is not None and not settings[t].has_holdout and settings[t].metrics is not None
        ]
        rows.append(
            {
                "family": family,
                "layers_total": len(members),
                "layers_with_holdout": len(done),
                "done": ", ".join(done) if done else "-",
                "training_only": ", ".join(partial) if partial else "-",
                "missing": ", ".join(t for t in members if t not in done and t not in partial) or "-",
            }
        )
    rows.append(
        {
            "family": "baseline",
            "layers_total": 1,
            "layers_with_holdout": 1 if (settings.get("baseline") and settings["baseline"].has_holdout) else 0,
            "done": "baseline" if (settings.get("baseline") and settings["baseline"].has_holdout) else "-",
            "training_only": "-",
            "missing": "-" if (settings.get("baseline") and settings["baseline"].has_holdout) else "baseline",
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures (light theme, ink on white)
# --------------------------------------------------------------------------- #
def _figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 140,
        }
    )


def _save(fig, path: Path, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        target = path.with_suffix(f".{ext}")
        fig.savefig(target, bbox_inches="tight", facecolor="white")
        written.append(target)
    plt.close(fig)


def plot_holdout(summary: pd.DataFrame, out_dir: Path, settings: dict[str, Setting]) -> list[Path]:
    df = summary[summary["mean_lddt"].notna()] if "mean_lddt" in summary else summary.iloc[0:0]
    if df.empty:
        return []
    written: list[Path] = []
    _figure_style()
    tags = list(df["setting"])
    y = np.arange(len(tags))[::-1]
    colors = [BASELINE_COLOR if t == "baseline" else ACCENT if settings[t].status == "complete" else WARN for t in tags]
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(tags) + 1.5))
    for i, (tag, yy, color) in enumerate(zip(tags, y, colors)):
        mean = float(df.loc[df["setting"] == tag, "mean_lddt"].iloc[0])
        lo = float(df.loc[df["setting"] == tag, "ci95_low"].iloc[0])
        hi = float(df.loc[df["setting"] == tag, "ci95_high"].iloc[0])
        ax.barh(yy, mean, height=0.55, color=color, alpha=0.85, zorder=2)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yy, yy], color=INK, lw=1.2, zorder=3)
            ax.plot([lo, lo], [yy - 0.1, yy + 0.1], color=INK, lw=1.2, zorder=3)
            ax.plot([hi, hi], [yy - 0.1, yy + 0.1], color=INK, lw=1.2, zorder=3)
        frame = settings[tag].frame
        if frame is not None:
            pts = frame["lddt"].to_numpy(float)
            jitter = (np.linspace(-1, 1, len(pts)) * 0.18) if len(pts) > 1 else np.zeros(1)
            ax.scatter(pts, yy + jitter, s=9, color=INK, alpha=0.45, zorder=4, linewidths=0)
    if "baseline" in list(df["setting"]):
        ax.axvline(float(df.loc[df["setting"] == "baseline", "mean_lddt"].iloc[0]), color=BASELINE_COLOR, ls="--", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.set_xlabel("holdout lDDT (mean with bootstrap 95% CI, per-example dots)")
    ax.set_title("Holdout lDDT by replaced layer" + ("  (orange = training-only, no holdout)" if any(s.status != "complete" for s in settings.values()) else ""), loc="left", fontsize=10)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, None)
    _save(fig, out_dir / "figures" / "partial_holdout_lddt", written)
    return written


def plot_paired(paired: pd.DataFrame, out_dir: Path) -> list[Path]:
    if paired.empty:
        return []
    written: list[Path] = []
    _figure_style()
    tags = list(paired["setting"])
    y = np.arange(len(tags))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(tags) + 1.5))
    for tag, yy in zip(tags, y):
        row = paired.loc[paired["setting"] == tag].iloc[0]
        mean = float(row["mean_delta_lddt"])
        lo, hi = float(row["bootstrap_delta_ci95_low"]), float(row["bootstrap_delta_ci95_high"])
        color = ACCENT if mean >= 0 else WARN
        ax.barh(yy, mean, height=0.55, color=color, alpha=0.85, zorder=2)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yy, yy], color=INK, lw=1.2, zorder=3)
            ax.plot([lo, lo], [yy - 0.1, yy + 0.1], color=INK, lw=1.2, zorder=3)
            ax.plot([hi, hi], [yy - 0.1, yy + 0.1], color=INK, lw=1.2, zorder=3)
    ax.axvline(0.0, color=BASELINE_COLOR, ls="--", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.set_xlabel("paired ΔlDDT vs baseline (mean with bootstrap 95% CI)")
    ax.set_title("Effect of replacing one layer (same holdout examples)", loc="left", fontsize=10)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir / "figures" / "partial_paired_deltas", written)
    return written


def plot_progress(summary: pd.DataFrame, settings: dict[str, Setting], out_dir: Path) -> list[Path]:
    df = summary[summary.get("epochs_completed").notna()] if "epochs_completed" in summary else summary.iloc[0:0]
    if df.empty:
        return []
    written: list[Path] = []
    _figure_style()
    tags = list(df["setting"])
    y = np.arange(len(tags))[::-1]
    planned = df["planned_epochs"].dropna() if "planned_epochs" in df else pd.Series(dtype=float)
    if not planned.empty:
        target, target_label = float(planned.max()), f"planned {int(planned.max())}"
    else:
        # No MAX_EPOCHS in the artefacts; the longest run is the best available
        # reference for "how far the others got".
        target, target_label = float(df["epochs_completed"].max()), f"longest run {int(df['epochs_completed'].max())}"
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(tags) + 1.5))
    for tag, yy in zip(tags, y):
        row = df.loc[df["setting"] == tag].iloc[0]
        done = float(row["epochs_completed"])
        complete = settings[tag].status == "complete"
        ax.barh(yy, done, height=0.55, color=ACCENT if complete else WARN, alpha=0.85, zorder=2)
        ax.annotate(f"{int(done)}" + ("" if complete else "  (interrupted)"), (done, yy), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.axvline(target, color=BASELINE_COLOR, ls="--", lw=0.9, zorder=1)
    ax.annotate(target_label, (target, len(tags) - 0.4), xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.set_xlabel("epochs completed (from metrics.csv)")
    ax.set_title("Training progress — how much of the schedule actually ran", loc="left", fontsize=10)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, None)
    _save(fig, out_dir / "figures" / "partial_training_progress", written)
    return written


def plot_curves(settings: dict[str, Setting], out_dir: Path) -> list[Path]:
    metrics = {tag: s.metrics for tag, s in settings.items() if s.metrics is not None}
    if not metrics:
        return []
    _figure_style()
    written: list[Path] = []
    keys = [spec for spec in splot.CURVE_SPECS if not spec[0].startswith("val/")]
    ncols, nrows = 2, int(np.ceil(len(keys) / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 2.5 * nrows), squeeze=False)
    order = splot.ordered_tags(metrics)
    for ax, (key, ylabel, higher_better) in zip(axes.ravel(), keys):
        drew = False
        for index, tag in enumerate(order):
            df = metrics[tag]
            column = splot.resolve_key(df, key)
            if column is None:
                continue
            x = pd.to_numeric(df["epoch"], errors="coerce") if "epoch" in df.columns else pd.RangeIndex(len(df))
            y = pd.to_numeric(df[column], errors="coerce")
            if y.notna().sum() == 0:
                continue
            ax.plot(x, y, lw=1.4, color=splot.color_for(tag, index), label=tag, marker="o", ms=2.2)
            drew = True
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  ({'higher' if higher_better else 'lower'} is better)", loc="left", fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if drew:
            ax.legend(frameon=False, fontsize=7, loc="best")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Training curves — whatever epochs were completed", x=0.02, ha="left", fontsize=11)
    _save(fig, out_dir / "figures" / "partial_training_curves", written)
    return written


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def draft_paragraph(summary: pd.DataFrame, coverage: pd.DataFrame, settings: dict[str, Setting]) -> list[str]:
    base = summary[summary["setting"] == "baseline"]
    others = summary[(summary["setting"] != "baseline") & summary["mean_lddt"].notna()] if "mean_lddt" in summary else summary.iloc[0:0]
    if base.empty or others.empty:
        return [
            "> _Not enough completed settings for a results paragraph "
            "(a holdout evaluation for the baseline plus at least one replaced layer is required)._"
        ]
    b = base.iloc[0]
    every = summary[summary["mean_lddt"].notna()]
    n = int(every["n"].min())
    # Parameter counts come from a *replaced* layer, not from the baseline: the
    # whole point of the sentence is how few factors are trained.
    replaced = every[every["setting"] != "baseline"]
    trainable = replaced["trainable_params"].dropna().iloc[0] if not replaced["trainable_params"].dropna().empty else None
    total = b["total_params"]
    lines = [
        f"> We replaced one linear layer at a time by a sign-preserving rank-"
        f"{int(every['rank'].dropna().iloc[0]) if every['rank'].notna().any() else 'k'} non-negative "
        f"factorisation of that layer's weight and fine-tuned only the factors"
        + (f" ({int(trainable):,} of {int(total):,} parameters)" if trainable and pd.notna(total) else "")
        + f", keeping the rest of the network frozen. Holdout lDDT was evaluated on the same "
        f"{n} temporally held-out PDB examples for every setting.",
        f"> The fully fine-tuned baseline reached {b['mean_lddt']:.4f} "
        f"[{b['ci95_low']:.4f}, {b['ci95_high']:.4f}].",
    ]
    parts = []
    for _, row in others.iterrows():
        if np.isfinite(row.get("mean_delta", np.nan)):
            parts.append(f"`{row['setting']}` {row['mean_delta']:+.4f} [{row['delta_ci95_low']:+.4f}, {row['delta_ci95_high']:+.4f}]")
    if parts:
        lines.append("> Per-layer paired changes were: " + "; ".join(parts) + ".")
    lines.append(
        "> Coverage: "
        + "; ".join(f"{r.family} {r.layers_with_holdout}/{r.layers_total}" for r in coverage.itertuples())
        + ". Results for the remaining layers were not available at the time of writing."
    )
    return lines


def write_report(
    out_dir: Path,
    sweep_dirs: list[Path],
    settings: dict[str, Setting],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    per_example: pd.DataFrame | None,
    coverage: pd.DataFrame,
    notes: list[tuple[str, Path, str]],
    figures: list[Path],
    seed: int,
) -> Path:
    complete = summary[summary["status"] == "complete"] if not summary.empty else summary
    lines = [
        "# Partial NMF/ZKP sweep report",
        "",
        f"- Sweep dir(s): {', '.join('`' + str(d) + '`' for d in sweep_dirs)}",
        f"- Settings discovered: **{len(summary)}** ({len(complete)} with a completed holdout evaluation)",
        f"- Bootstrap resamples: {10000}, seed {seed}",
        "",
        "## How to read this report",
        "",
        "- A setting is `complete` only when it has **both** a holdout validation CSV and a "
        "`run_summary.json`. Only these numbers are paper-grade.",
        "- `training-only` settings were interrupted during training. The trainer validates once, "
        "*after* fit, and periodic validation is disabled in this sweep, so a killed job has "
        "**training curves but no holdout number**. Its rows are filled with `-`. A job that "
        "reached the full schedule but was stopped before the holdout pass is labelled "
        "`trained N epochs, holdout missing` instead, which is a different gap to close.",
        "- lDDT definitions are imported from `summarize_nmf_zkp_pdb_sweep.py`"
        " (per-example protein lDDT at the final epoch, `dataset == pdb_holdout`), so these "
        "numbers match `paper_summary.csv` exactly where both exist.",
        "- Timing falls back to the `Epoch N Summary` tables in `<tag>.run.log` when "
        "`run_summary.json` is absent; the source is stated per row in `wall_time_source`.",
        "",
        "## Status of every setting",
        "",
    ]
    status_cols = [
        "setting",
        "family",
        "proof_target",
        "status_detail" if "status_detail" in summary.columns else "status",
        "epochs_completed",
        "planned_epochs",
        "optimizer_steps",
    ]
    lines.append(md_table(summary[[c for c in status_cols if c in summary.columns]]))
    lines += ["", "## Family coverage", "", md_table(coverage)]
    lines += ["", "## Holdout lDDT (paper-grade rows only)", ""]
    val_cols = ["setting", "n", "coverage", "mean_lddt", "sd_lddt", "median_lddt", "ci95_low", "ci95_high"]
    val_df = summary[[c for c in val_cols if c in summary.columns]] if not summary.empty else summary
    val_df = val_df[val_df["mean_lddt"].notna()] if "mean_lddt" in val_df else val_df
    lines.append(md_table(val_df.drop(columns=["coverage"], errors="ignore")))
    lines += ["", "## Paired change relative to the baseline", ""]
    lines.append(md_table(paired) if not paired.empty else "_No baseline pairing available._")
    lines += ["", "## Training progress and cost", ""]
    cost_cols = [
        "setting",
        "epochs_completed",
        "final_train_lddt",
        "best_train_lddt",
        "final_train_loss",
        "final_seq_recovery",
        "wall_time_seconds",
        "wall_time_source",
        "mean_s_per_batch",
        "gpu_peak_gib",
        "trainable_params",
        "trainable_fraction",
        "rank",
        "replaced_layers",
    ]
    cost_df = summary[[c for c in cost_cols if c in summary.columns]]
    if "wall_time_seconds" in cost_df:
        cost_df = cost_df.copy()
        cost_df["wall_time_seconds"] = cost_df["wall_time_seconds"].map(lambda v: fmt(v, 1))
        cost_df["mean_s_per_batch"] = cost_df["mean_s_per_batch"].map(lambda v: fmt(v, 3))
        cost_df["gpu_peak_gib"] = cost_df["gpu_peak_gib"].map(lambda v: fmt(v, 2))
        cost_df["trainable_fraction"] = cost_df["trainable_fraction"].map(lambda v: fmt(v, 6))
    lines.append(md_table(cost_df))
    lines += ["", "## Auto-drafted results paragraph", "", "_Numbers are filled in from the tables above; edit the prose before use._", ""]
    lines += draft_paragraph(summary, coverage, settings)
    lines += ["", "## Files", ""]
    for name in (
        "partial_summary.csv",
        "partial_paired_deltas.csv",
        "partial_per_example_lddt.csv",
        "partial_status.json",
    ):
        lines.append(f"- `{out_dir / name}`")
    for path in figures:
        lines.append(f"- `{path}`")
    if notes:
        lines += ["", "## Source-dir resolution notes", ""]
        lines += [f"- `{tag}`: {reason} — `{directory}`" for tag, directory, reason in notes]
    lines += [
        "",
        "## What is still missing",
        "",
    ]
    missing = []
    for r in coverage.itertuples():
        if r.layers_with_holdout < r.layers_total:
            missing.append(f"- `{r.family}`: {r.layers_total - r.layers_with_holdout} layer(s) without a holdout number ({r.missing})")
    lines += missing or ["- Nothing: every configured layer has a holdout evaluation."]
    report = out_dir / "partial_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dirs", nargs="*", type=Path, help="sweep dir(s); may be empty with --discover")
    ap.add_argument("--out", type=Path, required=True, help="directory to write the report into")
    ap.add_argument("--discover", type=Path, nargs="?", const=DEFAULT_LOG_ROOT, default=None,
                    help=f"merge every sweep_nmf_zkp_pdb_* under this root (default {DEFAULT_LOG_ROOT})")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dirs: list[Path] = []
    if args.discover is not None:
        if not args.discover.is_dir():
            print(f"ERROR: --discover root does not exist: {args.discover}", file=sys.stderr)
            return 1
        # newest first, matching merge_sweep_reports.sh
        dirs += [p for p in sorted(args.discover.glob("sweep_nmf_zkp_pdb_*"), key=lambda p: p.stat().st_mtime, reverse=True) if p.is_dir()]
    dirs += [d for d in args.sweep_dirs]
    seen: set[Path] = set()
    dirs = [d for d in dirs if not (d in seen or seen.add(d))]
    if not dirs:
        print("ERROR: pass at least one sweep dir, or --discover ROOT", file=sys.stderr)
        return 1

    settings, notes = discover(dirs)
    if not settings:
        print("No settings found (no validation CSV, metrics.csv or run_summary.json).", file=sys.stderr)
        return 1
    print(f"Discovered {len(settings)} setting(s): {', '.join(settings)}")
    for tag, setting in settings.items():
        print(f"  {tag:28s} {setting.status:34s} from {setting.source}")

    summary, paired, per_example = build_summary(settings, args.seed)
    coverage = family_coverage(settings)
    args.out.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.out / "partial_summary.csv", index=False)
    paired.to_csv(args.out / "partial_paired_deltas.csv", index=False)
    if per_example is not None:
        per_example.to_csv(args.out / "partial_per_example_lddt.csv", index=False)

    figures: list[Path] = []
    figures += plot_holdout(summary, args.out, settings)
    figures += plot_paired(paired, args.out)
    figures += plot_progress(summary, settings, args.out)
    figures += plot_curves(settings, args.out)

    inventory = {
        "sweep_dirs": [str(d) for d in dirs],
        "seed": args.seed,
        "settings": {
            tag: {
                "status": s.status,
                "family": s.family,
                "source_dir": str(s.source),
                "validation_csv": str(s.validation_csv) if s.validation_csv else None,
                "metrics_csv": str(s.metrics_csv) if s.metrics_csv else None,
                "run_log": str(s.run_log) if s.run_log else None,
                "run_summary_json": str(s.run_summary_path) if s.run_summary_path else None,
                "epochs_completed": s.epochs_completed,
                "holdout_examples": int(len(s.frame)) if s.frame is not None else 0,
                "log_info": {k: v for k, v in s.log_info.items() if k != "log_per_epoch_seconds"},
            }
            for tag, s in settings.items()
        },
        "source_notes": [{"tag": t, "dir": str(d), "reason": r} for t, d, r in notes],
    }
    (args.out / "partial_status.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    report = write_report(args.out, dirs, settings, summary, paired, per_example, coverage, notes, figures, args.seed)
    print()
    print(f"Wrote {report}")
    print(f"Wrote {args.out / 'partial_summary.csv'}")
    print(f"Wrote {args.out / 'partial_paired_deltas.csv'}")
    print(f"Wrote {args.out / 'partial_status.json'}")
    for path in figures:
        print(f"Wrote {path}")
    incomplete = summary[summary["status"] != "complete"]
    if not incomplete.empty:
        label = "status_detail" if "status_detail" in incomplete.columns else "status"
        print()
        print(f"NOTE: {len(incomplete)} setting(s) are not complete and carry no holdout number:")
        for _, row in incomplete.iterrows():
            print(f"  {row['setting']}: {row[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
