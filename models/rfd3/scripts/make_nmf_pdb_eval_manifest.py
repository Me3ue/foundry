#!/usr/bin/env python
"""Create a deterministic, structure-level PDB evaluation manifest.

This is deliberately separate from training.  The manifest is built from the
metadata parquet with a strict temporal split and one structure per PDB/assembly
selection.  The downstream evaluator must still run the RFD3 transform and
record ``transform_ok``/``transform_error`` for every row; a metadata manifest
alone cannot prove that coordinates are finite.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def stable_key(row: pd.Series) -> str:
    return f"{row.pdb_id}|{row.assembly_id}|{row.pn_unit_1_iid}|{row.pn_unit_2_iid}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquet", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--train-cutoff", default="2024-12-16")
    p.add_argument("--eval-end", default="2025-01-01")
    p.add_argument("--max-examples", type=int, default=256)
    args = p.parse_args()

    columns = [
        "example_id", "pdb_id", "assembly_id", "deposition_date", "resolution",
        "num_polymer_pn_units", "n_prot", "cluster", "is_inter_molecule",
        "pn_unit_1_iid", "pn_unit_2_iid",
    ]
    df = pd.read_parquet(args.parquet, columns=columns)
    dates = pd.to_datetime(df["deposition_date"], errors="coerce")
    mask = (
        (dates >= pd.Timestamp(args.train_cutoff))
        & (dates < pd.Timestamp(args.eval_end))
        & (df["resolution"] < 2.5)
        & (df["num_polymer_pn_units"] <= 4)
        & (df["n_prot"] >= 1)
        & df["cluster"].notna()
        & df["is_inter_molecule"].astype(bool)
    )
    out = df.loc[mask].copy()
    # These are known parser/transform failures from the training logs. Keep
    # them in an audit file but never in the evaluation manifest.
    bad_pdbs = {"7puh", "9hq9", "9hqj", "6ufk", "6jji", "7atg", "2cse", "6ic7"}
    out["pdb_id"] = out["pdb_id"].astype(str).str.lower()
    rejected = out[out["pdb_id"].isin(bad_pdbs)].copy()
    out = out[~out["pdb_id"].isin(bad_pdbs)].copy()
    out["stable_key"] = out.apply(stable_key, axis=1)
    out["stable_hash"] = out["stable_key"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    out = out.sort_values(["stable_hash", "stable_key"]).drop_duplicates("stable_key")
    out = out.head(args.max_examples).copy()
    out["transform_ok"] = pd.NA
    out["transform_error"] = pd.NA
    out["evaluation_status"] = "requires_transform_preflight"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    rejected.to_csv(args.output.with_name(args.output.stem + ".rejected_known_failures.csv"), index=False)
    print(f"wrote {len(out)} candidates to {args.output}")
    print(f"rejected known failures: {len(rejected)}")


if __name__ == "__main__":
    main()
