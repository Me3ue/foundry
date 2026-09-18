#!/usr/bin/env python3
"""Evaluate RFD3 near-native outputs with transparent, non-leaking metrics.

Reports CA-aligned RMSD and coverage for every final model.  Optionally runs
US-align for TM-score.  It deliberately does not call a low-RMSD partial-
diffusion result a de-novo success: input-coordinate retention makes such a
claim invalid.  Use this table, raw structures, run_manifest.json, and an
independent geometry check (e.g. MolProbity) when reporting results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def biotite_io():
    """Import optional RFD3 evaluation dependencies only when evaluation runs."""
    try:
        from biotite.structure import AtomArrayStack, superimpose
        from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure
        from biotite.structure.io.pdbx import CIFFile, get_structure as get_cif_structure
    except ImportError as error:
        raise RuntimeError(
            "Biotite is required for evaluation; run this from the Foundry/RFD3 environment."
        ) from error
    return AtomArrayStack, superimpose, PDBFile, get_pdb_structure, CIFFile, get_cif_structure


def load_ca(path: Path) -> dict[tuple[str, int, str], np.ndarray]:
    """Load first model CA coordinates keyed by chain, residue ID, insertion code."""
    AtomArrayStack, _, PDBFile, get_pdb_structure, CIFFile, get_cif_structure = biotite_io()
    if path.name.endswith(".cif.gz"):
        with gzip.open(path, "rt") as handle:
            array = get_cif_structure(CIFFile.read(handle))
    elif path.suffix.lower() == ".cif":
        array = get_cif_structure(CIFFile.read(path))
    else:
        array = get_pdb_structure(PDBFile.read(path))
    if isinstance(array, AtomArrayStack):
        array = array[0]
    categories = set(array.get_annotation_categories())
    ins = array.ins_code if "ins_code" in categories else np.full(array.array_length(), "")
    mask = (array.atom_name == "CA") & np.isfinite(array.coord).all(axis=1)
    result = {}
    for chain, resid, icode, coord in zip(array.chain_id[mask], array.res_id[mask], ins[mask], array.coord[mask]):
        key = (str(chain), int(resid), str(icode).strip())
        # Alternate locations are resolved deterministically by retaining first atom.
        result.setdefault(key, coord)
    return result


def rmsd(reference: dict, model: dict) -> tuple[float, int, float]:
    common = sorted(set(reference) & set(model))
    if len(common) < 3:
        return float("nan"), len(common), 0.0
    ref = np.asarray([reference[key] for key in common])
    mob = np.asarray([model[key] for key in common])
    _, superimpose, _, _, _, _ = biotite_io()
    _, transform = superimpose(ref, mob)
    fitted = transform.apply(mob)
    return float(np.sqrt(np.mean(np.sum((fitted - ref) ** 2, axis=1)))), len(common), len(common) / len(reference)


def usalign(reference: Path, model: Path, executable: str) -> float:
    """Extract TM-score normalized by reference length from US-align stdout."""
    exe = shutil.which(executable) or executable
    if not Path(exe).exists():
        raise FileNotFoundError(f"US-align executable not found: {executable}")
    result = subprocess.run([exe, str(model), str(reference), "-mol", "protein"],
                            text=True, capture_output=True, check=True)
    # US-align prints two normalization choices; the second is normalized by chain_2/reference.
    matches = re.findall(r"TM-score=\s*([0-9.]+)", result.stdout)
    if len(matches) < 2:
        raise RuntimeError("Could not parse TM-score from US-align output")
    return float(matches[1])


def target_for_output(path: Path) -> str | None:
    # Example: near_native_inputs_1ABR_near_native_0_model_0.cif.gz
    match = re.search(r"_([0-9A-Za-z]{4})_near_native(?:_|$)", path.name)
    return match.group(1).upper() if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--usalign", metavar="PATH", help="Optional US-align binary.")
    parser.add_argument("--max-ca-rmsd", type=float, default=2.0,
                        help="Near-native CA-RMSD threshold (Å), reported not imposed.")
    parser.add_argument("--min-ca-coverage", type=float, default=0.95)
    args = parser.parse_args()
    manifest_path = args.run_dir / "run_manifest.json"
    if not manifest_path.exists():
        parser.error(f"Missing {manifest_path}; run the generation script first.")
    manifest = json.loads(manifest_path.read_text())
    references = {name.upper(): Path(info["input"]) for name, info in manifest["targets"].items()}
    reference_ca = {name: load_ca(path) for name, path in references.items()}
    models = sorted(path for path in args.run_dir.glob("*.cif*") if "_noisy_" not in path.name and "_denoised_" not in path.name and not path.name.endswith(".json"))
    rows = []
    for model in models:
        target = target_for_output(model)
        if target not in references:
            continue
        value, n_ca, coverage = rmsd(reference_ca[target], load_ca(model))
        row = {"target": target, "model": str(model), "ca_rmsd_angstrom": value,
               "matched_ca": n_ca, "reference_ca": len(reference_ca[target]), "ca_coverage": coverage,
               "passes_near_native_threshold": bool(coverage >= args.min_ca_coverage and value <= args.max_ca_rmsd)}
        if args.usalign:
            row["tm_score_reference_normalized"] = usalign(references[target], model, args.usalign)
        rows.append(row)
    if not rows:
        parser.error("No final RFD3 CIF outputs matching manifest target names were found.")
    fields = list(rows[0])
    output = args.run_dir / "near_native_metrics.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    passing = sum(row["passes_near_native_threshold"] for row in rows)
    print(f"Wrote {output}: {passing}/{len(rows)} pass CA-RMSD <= {args.max_ca_rmsd} Å and coverage >= {args.min_ca_coverage:.0%}.")
    print("For publication: report all samples, fixed-coordinate fraction, input SHA256/checkpoint/seed,"
          " and independent all-atom stereochemistry plus H-bond geometry validation.")

if __name__ == "__main__":
    main()
