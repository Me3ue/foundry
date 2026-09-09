#!/usr/bin/env python
"""Summarize a fixed, per-example RFD3 NMF evaluation for paper reporting.

The evaluator deliberately operates on per-example metric CSV files rather than
training curves.  It retains only the intersection of examples that succeeded
for baseline and every compared variant, aggregates multiple diffusion samples
within each example, then reports paired baseline deltas with deterministic
non-parametric bootstrap confidence intervals.

Expected layout (one directory per setting)::

    evaluation_root/
      baseline/val_metrics/*.csv
      zkp_encoder/val_metrics/*.csv
      zkp_proj/val_metrics/*.csv
      zkp_head/val_metrics/*.csv

The CSVs emitted by StoreValidationMetricsInDFCallback already contain
``example_id`` plus numerical metric columns.  A manifest containing every
attempted example ID is optional but strongly recommended: it permits explicit
success/failure-rate reporting instead of silently ignoring failed examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PREFERRED = ("baseline", "zkp_encoder", "zkp_proj", "zkp_head", "zkp_all")
# Metrics available from RFD3's loss/design/backbone/sidechain/H-bond metric
# stack.  Missing metrics are omitted rather than represented as zero.
PRIMARY_METRICS = (
    "mean_lddt",
    "mean_lddt_protein",
    "mse_loss_mean",
    "seq_recovery",
    "lowest_t_seq_recovery",
    "token_lvl_sequence_loss",
    "frac_clashing",
    "frac_backbone_clashing",
    "frac_floating",
    "fraction_chainbreaks",
    "mean_clash_percent",
    "mean_valid_sidechain_percent",
    "mean_unintended_bonds_percent",
)


def discover_metrics_csvs(setting_dir: Path) -> list[Path]:
    candidates = list(setting_dir.glob("**/val_metrics/*.csv"))
    if not candidates:
        candidates = list(setting_dir.glob("**/*validation*.csv"))
    return sorted(candidates)


def read_setting(setting_dir: Path) -> pd.DataFrame:
    files = discover_metrics_csvs(setting_dir)
    if not files:
        raise FileNotFoundError(f"No per-example validation CSV found under {setting_dir}")
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc
        if "example_id" not in frame.columns:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"No CSV under {setting_dir} contains an example_id column")
    frame = pd.concat(frames, ignore_index=True)
    # Multiple validation epochs/files are possible. Keep the final epoch only
    # so that each evaluation checkpoint contributes once.
    if "epoch" in frame.columns:
        epoch = pd.to_numeric(frame["epoch"], errors="coerce")
        if epoch.notna().any():
            frame = frame.loc[epoch == epoch.max()].copy()
    frame["example_id"] = frame["example_id"].astype(str)
    return frame


def numeric_metric_columns(frame: pd.DataFrame) -> list[str]:
    ignored = {"epoch", "step", "global_step", "model_idx", "sample_idx"}
    cols: list[str] = []
    for col in frame.columns:
        if col in ignored or col == "example_id":
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            cols.append(col)
    return cols


def metric_column_map(frame: pd.DataFrame, metrics: Iterable[str]) -> dict[str, str]:
    """Resolve metric names from flattened callback columns."""
    mapping: dict[str, str] = {}
    for metric in metrics:
        if metric in frame.columns:
            mapping[metric] = metric
            continue
        matches = [c for c in frame.columns if c.endswith(f".{metric}")]
        if matches:
            mapping[metric] = matches[0]
    return mapping


def aggregate_per_example(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    mapping = metric_column_map(frame, metrics)
    if not mapping:
        return pd.DataFrame(columns=["example_id"])
    out = frame[["example_id", *mapping.values()]].copy()
    out.rename(columns={source: metric for metric, source in mapping.items()}, inplace=True)
    for col in mapping:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # Mean across diffusion samples/models only after retaining all samples.
    return out.groupby("example_id", as_index=False)[list(mapping)].mean()


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_sign_permutation_p(values: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided randomization p-value for paired deltas, without scipy."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    if n_perm <= 0:
        return float("nan")
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(values)))
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1.0 + np.sum(permuted >= observed)) / (n_perm + 1.0))


def _finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    pair = pd.concat([x, y], axis=1).apply(pd.to_numeric, errors="coerce").to_numpy()
    pair = pair[np.isfinite(pair).all(axis=1)]
    if pair.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return pair[:, 0], pair[:, 1]
def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-FDR q-values in the original order, ignoring NaNs."""
    valid = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    q_values = [float("nan")] * len(p_values)
    running = 1.0
    for reverse_rank, (index, p) in enumerate(sorted(valid, key=lambda item: item[1], reverse=True), start=1):
        running = min(running, p * len(valid) / reverse_rank)
        q_values[index] = float(min(running, 1.0))
    return q_values


def load_manifest(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("example_ids", data.get("examples", []))
        return {str(x["example_id"] if isinstance(x, dict) else x) for x in data}
    frame = pd.read_csv(path)
    column = "example_id" if "example_id" in frame.columns else frame.columns[0]
    return set(frame[column].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=None, help="CSV/JSON of every attempted example_id")
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutation", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    root: Path = args.evaluation_root
    settings = args.settings or [p.name for p in root.iterdir() if p.is_dir()]
    settings = [s for s in PREFERRED if s in settings] + [s for s in settings if s not in PREFERRED]
    if "baseline" not in settings:
        raise ValueError("A baseline directory is required for paired NMF comparisons")

    raw = {setting: read_setting(root / setting) for setting in settings}
    # Resolve flattened callback names such as
    # ``general_metrics.mean_lddt_protein`` before finding common metrics.
    resolved = {setting: metric_column_map(df, PRIMARY_METRICS) for setting, df in raw.items()}
    common_primary = set.intersection(*(set(mapping) for mapping in resolved.values()))
    # Include any additional common flattened numerical columns for future
    # metrics, while keeping the named paper metrics first.
    available_columns = set.intersection(*(set(numeric_metric_columns(df)) for df in raw.values()))
    extra = sorted(
        {
            column.rsplit(".", 1)[-1]
            for column in available_columns
            if column.rsplit(".", 1)[-1] not in PRIMARY_METRICS
        }
    )
    metrics = [m for m in PRIMARY_METRICS if m in common_primary] + extra
    if not metrics:
        raise ValueError("No common numerical per-example metrics were found across settings")

    per_example = {setting: aggregate_per_example(df, metrics) for setting, df in raw.items()}
    # Pair each variant with baseline independently. Requiring the intersection
    # of every setting would make the estimate depend on whether an unrelated
    # variant failed and can bias a partial sweep.
    baseline_ids = set(per_example["baseline"].example_id)
    common_ids = sorted(baseline_ids)
    if not common_ids:
        raise ValueError("No successful baseline examples")
    manifest_ids = load_manifest(args.manifest)
    attempted = len(manifest_ids) if manifest_ids is not None else None

    out_dir = root / "paper_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"example_id": common_ids}).to_csv(out_dir / "common_successful_examples.csv", index=False)

    rng = np.random.default_rng(args.seed)
    overall_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    paired_p_values: list[float] = []
    base = per_example["baseline"].set_index("example_id")
    failures: list[dict[str, object]] = []

    for setting, frame in per_example.items():
        indexed = frame.set_index("example_id")
        succeeded = set(indexed.index)
        failures.append({
            "setting": setting,
            "attempted_examples": attempted,
            "successful_examples": len(succeeded),
            "common_paired_examples": len(common_ids),
            "failure_count": None if attempted is None else attempted - len(succeeded),
            "success_rate": None if attempted is None else len(succeeded) / attempted,
        })
        values = indexed.reindex(common_ids)
        for metric in metrics:
            x = pd.to_numeric(values[metric], errors="coerce").to_numpy(dtype=float)
            finite_x = x[np.isfinite(x)]
            lo, hi = bootstrap_mean_ci(finite_x, rng, args.bootstrap)
            overall_rows.append({
                "setting": setting, "metric": metric, "n_attempted": attempted,
                "n_successful": len(succeeded), "n_paired": int(len(finite_x)),
                "missing_or_nonfinite": int(len(x) - len(finite_x)),
                "mean": float(np.mean(finite_x)) if len(finite_x) else float("nan"),
                "std": float(np.std(finite_x, ddof=1)) if len(finite_x) > 1 else float("nan"),
                "median": float(np.median(finite_x)) if len(finite_x) else float("nan"),
                "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
            })
            if setting != "baseline":
                pair_ids = sorted(set(base.index) & set(indexed.index))
                bx, vx = _finite_pair(base.loc[pair_ids, metric], indexed.loc[pair_ids, metric])
                delta = vx - bx
                dlo, dhi = bootstrap_mean_ci(delta, rng, args.bootstrap)
                p_value = paired_sign_permutation_p(delta, args.permutation, rng)
                paired_rows.append({
                    "setting": setting, "baseline": "baseline", "metric": metric,
                    "n_paired": len(delta), "mean_delta": float(np.mean(delta)) if len(delta) else float("nan"),
                    "median_delta": float(np.median(delta)) if len(delta) else float("nan"),
                    "std_delta": float(np.std(delta, ddof=1)) if len(delta) > 1 else float("nan"),
                    "bootstrap_ci95_low": dlo, "bootstrap_ci95_high": dhi,
                    "permutation_p": p_value,
                    "fraction_variant_greater": float(np.mean(delta > 0)) if len(delta) else float("nan"),
                    "fraction_variant_lower": float(np.mean(delta < 0)) if len(delta) else float("nan"),
                    "fraction_tied": float(np.mean(delta == 0)) if len(delta) else float("nan"),
                })
                paired_p_values.append(p_value)

    for row, q_value in zip(paired_rows, benjamini_hochberg(paired_p_values)):
        row["permutation_q_fdr"] = q_value

    pd.DataFrame(overall_rows).to_csv(out_dir / "metric_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out_dir / "paired_baseline_deltas.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "evaluation_coverage.csv", index=False)
    protocol = {
        "bootstrap_replicates": args.bootstrap,
        "permutation_replicates": args.permutation,
        "bootstrap_seed": args.seed,
        "aggregation": "mean across diffusion samples, then paired per-example comparison",
        "paired_population": "baseline-successful examples intersected independently with each variant; finite metric pairs only",
        "n_common_successful_examples": len(common_ids),
        "manifest_provided": manifest_ids is not None,
        "metrics": metrics,
        "settings": settings,
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote paper evaluation summary to {out_dir}")
    print(f"Paired successful examples: {len(common_ids)}")


if __name__ == "__main__":
    main()
