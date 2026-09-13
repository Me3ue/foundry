#!/usr/bin/env python
"""Preflight PDB interface rows through the RFD3 dataset transform pipeline.

This deliberately does not construct a model or run inference.  It loads the
same parquet/parser/transform used by the experiment, visits every selected
row, and records transform exceptions (and atomworks fallback rows) by PDB ID.

Holdout screening always uses the validation transform (``is_inference=true``).
That is the path ``validation_loop`` runs.  A successful transform is not
enough: validation then builds ``X_noisy_L = coord_atom_lvl_to_be_noised +
noise`` and rejects the example if that tensor or ``feats`` contains NaNs.
Preflight now performs the same checks so structures such as 9hql cannot slip
through.  When ``max_atoms_in_crop`` is set (as in ``pdb_holdout.yaml``),
validation also crops to that atom budget so the pairwise atom embedding fits
on an 80GB GPU.

``--training-conditions`` is retained for compatibility but no longer flips the
pipeline into training mode.  Training fills unresolved coordinates after a
crop and skips the inference-only fill; using that path would hide the NaNs
that used to crash uncropped validation.

The filter expression is only written with ``--write-filter``.  The operation
is deterministic and replaces the single ``pdb_id not in [...]`` line in the
specified YAML file with the union of its existing IDs and failures.

Important: preflight intentionally removes every existing ``pdb_id not in``
filter from the in-memory dataset config before loading the parquet.  This
ensures that a previous run's blacklist cannot hide a structure from a new
full preflight.  The YAML file itself is only changed when ``--write-filter``
is supplied.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Prefer the in-repo packages over the conda site-packages copies.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _src in (
    _REPO_ROOT / "src",
    _REPO_ROOT / "models" / "rfd3" / "src",
    _REPO_ROOT / "models" / "rfd3na" / "src",
):
    _src_str = str(_src)
    if _src.exists() and _src_str not in sys.path:
        sys.path.insert(0, _src_str)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


def _example_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("example_id", "id"):
            if key in value and isinstance(value[key], (str, int)):
                return str(value[key])
        for child in value.values():
            found = _example_id(child)
            if found:
                return found
    return None


def _pdb_id(row: Any) -> str:
    value = row.get("pdb_id", "") if hasattr(row, "get") else ""
    return str(value).strip().lower()


def _normalise_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:2000]


def _error_group(error: str) -> str:
    """Return a compact transform-stage group for the final summary."""
    match = re.search(r"stage [`']([^`']+)[`']", error)
    if match:
        return match.group(1)
    if error.startswith("fallback returned"):
        return "fallback"
    if "X_noisy_L" in error or "Tensor contains NaNs" in error:
        return "network_input_nans"
    if "NaN detected in `feats`" in error:
        return "feats_nans"
    return error.split(":", 1)[0] or "unknown"


def _as_example_dict(value: Any) -> dict[str, Any]:
    """Unwrap a dataset wrapper result into a single example dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], dict):
            return value[0]
        for item in value:
            if isinstance(item, dict) and "coord_atom_lvl_to_be_noised" in item:
                return item
    raise AssertionError(
        f"could not interpret transform output as an example dict: {type(value)!r}"
    )


def _contains_nans(x: Any, *, msg: str = "") -> str | None:
    """Return the first NaN path, using the same wording as the trainer."""
    if isinstance(x, torch.Tensor):
        if x.is_floating_point() and torch.isnan(x).any():
            return ": ".join(filter(bool, [msg, "Tensor contains NaNs!"]))
        return None
    if isinstance(x, np.ndarray):
        if x.size and np.issubdtype(x.dtype, np.floating) and np.isnan(x).any():
            return ": ".join(filter(bool, [msg, "Numpy array contains NaNs!"]))
        return None
    if isinstance(x, float):
        if np.isnan(x):
            return ": ".join(filter(bool, [msg, "float is NaN!"]))
        return None
    if isinstance(x, dict):
        for key, value in x.items():
            found = _contains_nans(value, msg=".".join(filter(bool, [msg, str(key)])))
            if found:
                return found
        return None
    if isinstance(x, (list, tuple)):
        for idx, value in enumerate(x):
            found = _contains_nans(value, msg=".".join(filter(bool, [msg, str(idx)])))
            if found:
                return found
        return None
    return None


def _check_validation_network_inputs(example: dict[str, Any]) -> None:
    """Mirror ``RFD3Trainer._assemble_network_inputs`` validation NaN checks.

    Training may replace unresolved coordinates with zeros.  Validation does
    not: a transform that returns without raising can still produce an
    ``X_noisy_L`` that crashes ``validation_loop``.
    """
    if not isinstance(example, dict):
        raise AssertionError(
            f"transform output is not an example dict: {type(example)!r}"
        )
    example_id = example.get("example_id", "unknown")
    coords = example.get("coord_atom_lvl_to_be_noised")
    noise = example.get("noise")
    if coords is None or noise is None:
        raise AssertionError(
            "transform output missing coord_atom_lvl_to_be_noised or noise "
            f"for example_id: {example_id}"
        )
    try:
        x_noisy = coords + noise
    except Exception as exc:
        raise AssertionError(
            f"could not form X_noisy_L for example_id: {example_id}: {exc}"
        ) from exc
    found = _contains_nans(
        x_noisy,
        msg=f"network_input (X_noisy_L) for example_id: {example_id}",
    )
    if found:
        raise AssertionError(found)
    feats = example.get("feats")
    if feats is not None:
        found = _contains_nans(
            feats,
            msg=f"NaN detected in `feats` for example_id: {example_id}",
        )
        if found:
            raise AssertionError(found)


def _load_config(args: argparse.Namespace) -> DictConfig:
    config_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(
            config_name="validate",
            overrides=[f"experiment={args.experiment}"],
            return_hydra_config=True,
        )
    # Config interpolation uses the ``hydra:`` resolver (for example in
    # output/cache paths).  Standalone compose() does not populate HydraConfig,
    # unlike a @hydra.main entry point, so install the composed config before
    # instantiating any nodes.
    from hydra.core.hydra_config import HydraConfig
    HydraConfig.instance().set_config(cfg)
    transform_cfg = cfg.datasets.val.pdb_holdout.dataset.transform
    # Always match validation_loop.  TrainingRoute fills unresolved coords
    # after cropping; validation does not, so is_inference=False would hide
    # the X_noisy_L NaNs that abort holdout evaluation.
    transform_cfg.is_inference = True
    if args.training_conditions:
        print(
            "Note: --training-conditions no longer sets is_inference=False. "
            "Holdout preflight always uses the validation transform so that "
            "unresolved-coordinate NaNs are visible."
        )
        transform_cfg.train_conditions = cfg.datasets.global_transform_args.train_conditions
    if args.max_examples is not None:
        cfg.datasets.val.pdb_holdout.max_examples = args.max_examples
    return cfg


def _write_filter(path: Path, failed: set[str]) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?m)^(\s*-\s*[\"']pdb_id not in )(.*?)([\"']\s*)$")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find a pdb_id exclusion filter in {path}")
    old_ids = set(ast.literal_eval(match.group(2)))
    ids = sorted({str(x).lower() for x in old_ids} | failed)
    ids_repr = "[" + ", ".join(f"'{pdb_id}'" for pdb_id in ids) + "]"
    replacement = f"{match.group(1)}{ids_repr}{match.group(3)}"
    path.write_text(text[: match.start()] + replacement + text[match.end():], encoding="utf-8")
    print(f"updated {path}: {ids}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="nmf_zkp_pdb")
    parser.add_argument("--output", type=Path, default=Path("pdb_transform_preflight.csv"))
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing interfaces_df.parquet")
    parser.add_argument("--pdb-mirror", type=Path, default=None, help="Directory containing PDB/CIF files")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--training-conditions", action="store_true")
    parser.add_argument("--write-filter", type=Path, default=None)
    parser.add_argument(
        "--tracebacks",
        action="store_true",
        help="Print the full traceback for every failed example (very verbose)",
    )
    args = parser.parse_args()

    cfg = _load_config(args)
    if args.data_dir is not None:
        cfg.paths.data.pdb_parquet_dir = str(args.data_dir)
    if args.pdb_mirror is not None:
        cfg.paths.data.pdb_data_dir = str(args.pdb_mirror)
    dataset_cfg = cfg.datasets.val.pdb_holdout
    # Full preflight must not inherit a blacklist produced by an earlier run.
    # Remove only PDB-exclusion predicates; retain all scientific constraints
    # such as date, resolution, protein count, and inter-molecule status.
    filters = dataset_cfg.dataset.dataset.filters
    kept_filters = [
        str(filter_expr)
        for filter_expr in filters
        if "pdb_id not in" not in str(filter_expr).lower()
    ]
    removed_filters = len(filters) - len(kept_filters)
    dataset_cfg.dataset.dataset.filters = kept_filters
    if removed_filters:
        print(f"Removed {removed_filters} historical PDB blacklist filter(s) for full preflight.")
    wrapper_cfg = dataset_cfg.dataset
    dataset = hydra.utils.instantiate(wrapper_cfg.dataset)
    transform = hydra.utils.instantiate(wrapper_cfg.transform)
    wrapper_cfg = {"_target_": "atomworks.ml.datasets.StructuralDatasetWrapper",
                   "dataset": dataset,
                   "dataset_parser": hydra.utils.instantiate(wrapper_cfg.dataset_parser),
                   "transform": transform,
                   "save_failed_examples_to_dir": None,
                   "cif_parser_args": wrapper_cfg.cif_parser_args}
    wrapper = hydra.utils.instantiate(wrapper_cfg, _recursive_=False)
    rows = getattr(dataset, "data", None)
    if rows is None:
        raise RuntimeError("Instantiated validation dataset has no .data table")

    results: list[dict[str, str | int | bool]] = []
    failures: dict[str, list[str]] = defaultdict(list)
    error_groups: Counter[str] = Counter()
    try:
        from tqdm import tqdm
        iterator = tqdm(
            range(len(rows)),
            total=len(rows),
            desc="Preflighting transforms",
            unit="row",
            dynamic_ncols=True,
        )
    except ImportError:
        print("tqdm is not installed; continuing without a progress bar.")
        iterator = range(len(rows))

    for index in iterator:
        row = rows.iloc[index]
        requested_id = str(row.get("example_id", index))
        pdb_id = _pdb_id(row)
        returned_id = None
        try:
            value = wrapper[index]
            returned_id = _example_id(value)
            fallback = returned_id is not None and returned_id != requested_id
            if fallback:
                ok, error = False, "fallback returned " + returned_id
                error_groups[_error_group(error)] += 1
            else:
                _check_validation_network_inputs(_as_example_dict(value))
                ok, error = True, ""
        except Exception as exc:
            ok, error = False, _normalise_exception(exc)
            error_groups[_error_group(error)] += 1
            if args.tracebacks:
                traceback.print_exc()
            else:
                print(f"\nFAILED {pdb_id} ({requested_id}): {error}")
        if not ok:
            failures[pdb_id].append(error)
            if not args.tracebacks and error.startswith("fallback returned"):
                print(f"\nFAILED {pdb_id} ({requested_id}): {error}")
        results.append({"index": index, "example_id": requested_id, "pdb_id": pdb_id,
                        "ok": ok, "returned_example_id": returned_id or "", "error": error})
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                ok=sum(bool(item["ok"]) for item in results),
                failed=len(results) - sum(bool(item["ok"]) for item in results),
                failed_pdbs=len(failures),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(results).to_csv(args.output, index=False)
    args.output.with_suffix(".failed_pdbs.json").write_text(
        json.dumps({k: v for k, v in sorted(failures.items())}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = set(failures)
    print(f"checked {len(results)} rows; failed rows={sum(not bool(x['ok']) for x in results)}; failed pdbs={sorted(failed)}")
    if error_groups:
        print("failure groups:")
        for group, count in error_groups.most_common():
            print(f"  {count:>6}  {group}")
    if args.write_filter and failed:
        _write_filter(args.write_filter, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
