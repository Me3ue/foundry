#!/usr/bin/env python3
"""Post-process one RFD3 run with anchor-aware structure metrics.

This complements evaluate_near_native_rfd3.py. It reports all-residue,
fixed-anchor, and non-anchor CA RMSD, plus output/trajectory completeness.
It does not claim hydrogen-bond geometry recovery; use HBPLUS separately for
that because RFD3 H-bond conditioning is atom-wise rather than pairwise.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

import numpy as np


def readers():
    from biotite.structure import AtomArrayStack, superimpose
    from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb
    from biotite.structure.io.pdbx import CIFFile, get_structure as get_cif
    return AtomArrayStack, superimpose, PDBFile, get_pdb, CIFFile, get_cif


def load_ca(path: Path) -> dict[tuple[str, int], np.ndarray]:
    AtomArrayStack, _, PDBFile, get_pdb, CIFFile, get_cif = readers()
    if path.name.endswith(".cif.gz"):
        with gzip.open(path, "rt") as handle:
            array = get_cif(CIFFile.read(handle))
    elif path.suffix.lower() == ".cif":
        array = get_cif(CIFFile.read(path))
    else:
        array = get_pdb(PDBFile.read(path))
    if isinstance(array, AtomArrayStack):
        array = array[0]
    mask = (array.atom_name == "CA") & np.isfinite(array.coord).all(axis=1)
    result = {}
    for chain, resid, coord in zip(array.chain_id[mask], array.res_id[mask], array.coord[mask]):
        result.setdefault((str(chain), int(resid)), coord)
    return result


def aligned_rmsd(reference, model, keys):
    _, superimpose, *_ = readers()
    keys = sorted(set(keys) & set(reference) & set(model))
    if len(keys) < 3:
        return float("nan"), len(keys)
    ref = np.asarray([reference[k] for k in keys])
    mob = np.asarray([model[k] for k in keys])
    _, transform = superimpose(ref, mob)
    fitted = transform.apply(mob)
    return float(np.sqrt(np.mean(np.sum((fitted - ref) ** 2, axis=1)))), len(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    spec_path = args.run_dir / "near_native_inputs.json"
    specs = json.loads(spec_path.read_text())
    target = next(iter(manifest["targets"]))
    spec = specs[f"{target}_near_native"]
    reference = load_ca(Path(manifest["targets"][target]["input"]))
    anchor_keys = set()
    for selector in spec.get("select_fixed_atoms", {}):
        match = re.fullmatch(r"([A-Za-z])(\d+)", selector)
        if match:
            anchor_keys.add((match.group(1), int(match.group(2))))
    all_keys = set(reference)
    non_anchor = all_keys - anchor_keys
    models = sorted(
        p for p in args.run_dir.glob("*_model_*.cif.gz")
        if "_noisy_" not in p.name and "_denoised_" not in p.name
    )
    rows = []
    for model_path in models:
        model = load_ca(model_path)
        all_rmsd, n_all = aligned_rmsd(reference, model, all_keys)
        anchor_rmsd, n_anchor = aligned_rmsd(reference, model, anchor_keys)
        free_rmsd, n_free = aligned_rmsd(reference, model, non_anchor)
        stem = model_path.name.replace(".cif.gz", "")
        model_index = stem.rsplit("_model_", 1)[-1]
        noisy = args.run_dir / f"{stem.replace('_model_', '_noisy_model_')}.cif.gz"
        denoised = args.run_dir / f"{stem.replace('_model_', '_denoised_model_')}.cif.gz"
        rows.append({
            "target": target, "model": str(model_path), "model_index": model_index,
            "reference_ca": len(reference), "matched_ca": n_all,
            "ca_coverage": n_all / len(reference),
            "anchor_residues": len(anchor_keys), "non_anchor_residues": len(non_anchor),
            "all_ca_rmsd_angstrom": all_rmsd,
            "anchor_ca_rmsd_angstrom": anchor_rmsd,
            "non_anchor_ca_rmsd_angstrom": free_rmsd,
            "noisy_trajectory_exists": noisy.is_file(),
            "denoised_trajectory_exists": denoised.is_file(),
        })
    if not rows:
        parser.error("No final model CIF files found.")
    output = args.run_dir / "near_native_metrics_anchor_aware.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {output} ({len(rows)} models; {len(anchor_keys)} anchor residues).")


if __name__ == "__main__":
    main()
