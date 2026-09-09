#!/usr/bin/env python
"""Preflight PDB interface rows through the RFD3 dataset transform pipeline.

This deliberately does not construct a model or run inference.  It loads the
same parquet/parser/transform used by the experiment, visits every selected
row, and records transform exceptions (and atomworks fallback rows) by PDB ID.
Use ``--training-conditions`` to exercise ``SampleConditioningType`` as the
training loader does; without it the normal inference/validation transform is
used.

The filter expression is only written with ``--write-filter``.  The operation
is deterministic and replaces the single ``pdb_id not in [...]`` line in the
specified YAML file with the union of its existing IDs and failures.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
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
    return error.split(":", 1)[0] or "unknown"


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
    if args.training_conditions:
        # In pdb_holdout.yaml, the StructuralDatasetWrapper config is stored
        # under dataset; transform is therefore dataset.transform (not
        # pdb_holdout.transform).
        transform_cfg = cfg.datasets.val.pdb_holdout.dataset.transform
        transform_cfg.is_inference = False
        # The training route needs the same condition definitions as training.
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
    replacement = f"{match.group(1)}{json.dumps(ids)}{match.group(3)}"
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
        try:
            value = wrapper[index]
            returned_id = _example_id(value)
            fallback = returned_id is not None and returned_id != requested_id
            error = "fallback returned " + returned_id if fallback else ""
            ok = not fallback
        except Exception as exc:
            ok, returned_id, error = False, None, _normalise_exception(exc)
            error_groups[_error_group(error)] += 1
            if args.tracebacks:
                traceback.print_exc()
            else:
                print(f"\nFAILED {pdb_id} ({requested_id}): {error}")
        if not ok:
            failures[pdb_id].append(error)
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
