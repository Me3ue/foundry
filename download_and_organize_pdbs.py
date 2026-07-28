#!/usr/bin/env python3
from pathlib import Path
import json
import os
import requests

from Bio.PDB import MMCIFParser, PDBIO, Select


PDB_IDS = [
    "3BTA", "5N0B", "1MDT", "1DM0", "2AAI", "1ABR", "4UY2",
    "4JTA", "1QLX", "7UMQ", "5OQV", "5O3L", "6A6B"
]

BASE_URL = "https://files.rcsb.org/download"
RAW_DIR = Path("raw_pdb")
CLEAN_DIR = Path("clean_pdb")
META_DIR = Path("metadata")


class ProteinOnlySelect(Select):
    def accept_residue(self, residue):
        hetflag = residue.get_id()[0]
        # keep standard residues only, skip waters and ligands
        return hetflag == " "


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, out_path: Path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def clean_structure(cif_path: Path, out_pdb_path: Path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb_path), select=ProteinOnlySelect())


def extract_metadata(cif_path: Path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))

    model = next(structure.get_models())
    chains = list(model.get_chains())

    residue_count = 0
    atom_count = 0
    chain_ids = []
    for chain in chains:
        chain_ids.append(chain.id)
        for residue in chain:
            hetflag = residue.get_id()[0]
            if hetflag == " ":
                residue_count += 1
                atom_count += len(residue)

    return {
        "pdb_id": cif_path.stem.upper(),
        "source": f"{BASE_URL}/{cif_path.stem.upper()}.cif",
        "num_chains": len(chains),
        "chain_ids": chain_ids,
        "num_protein_residues": residue_count,
        "num_protein_atoms": atom_count,
        "raw_file": str(cif_path),
        "clean_file": str(CLEAN_DIR / f"{cif_path.stem.upper()}.pdb"),
    }


def main():
    ensure_dirs()

    summary = []

    for pdb_id in PDB_IDS:
        pdb_id = pdb_id.upper()
        raw_path = RAW_DIR / f"{pdb_id}.cif"
        clean_path = CLEAN_DIR / f"{pdb_id}.pdb"
        meta_path = META_DIR / f"{pdb_id}.json"

        url = f"{BASE_URL}/{pdb_id}.cif"
        print(f"Downloading {pdb_id} ...")
        try:
            download_file(url, raw_path)
        except Exception as e:
            print(f"  failed to download {pdb_id}: {e}")
            continue

        print(f"Cleaning {pdb_id} ...")
        try:
            clean_structure(raw_path, clean_path)
        except Exception as e:
            print(f"  failed to clean {pdb_id}: {e}")
            continue

        try:
            meta = extract_metadata(raw_path)
            meta["status"] = "ok"
        except Exception as e:
            meta = {
                "pdb_id": pdb_id,
                "status": "metadata_failed",
                "error": str(e),
                "raw_file": str(raw_path),
                "clean_file": str(clean_path),
            }

        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary.append(meta)

        print(f"  done: {pdb_id}")

    (META_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nAll done.")
    print(f"Raw files   -> {RAW_DIR}/")
    print(f"Clean files -> {CLEAN_DIR}/")
    print(f"Metadata    -> {META_DIR}/")


if __name__ == "__main__":
    main()