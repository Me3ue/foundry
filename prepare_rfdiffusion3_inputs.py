#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

from Bio.PDB import MMCIFParser, PDBIO, Select

PDB_IDS = [
    "3BTA", "5N0B", "1MDT", "1DM0", "2AAI", "1ABR", "4UY2",
    "4JTA", "1QLX", "7UMQ", "5OQV", "5O3L", "6A6B"
]

ROOT = Path(".")
RAW_DIR = ROOT / "raw_pdb"
RF_DIR = ROOT / "rfdiffusion3_inputs"
META_DIR = ROOT / "metadata"
ANALYSIS_DIR = ROOT / "analysis"


class ProteinChainSelect(Select):
    def __init__(self, keep_chain_id: str | None = None):
        self.keep_chain_id = keep_chain_id

    def accept_chain(self, chain):
        if self.keep_chain_id is None:
            return 1
        return chain.id == self.keep_chain_id

    def accept_residue(self, residue):
        return residue.get_id()[0] == " "


class BackboneOnlySelect(Select):
    def accept_atom(self, atom):
        return atom.get_name() in {"N", "CA", "C", "O"}


def choose_primary_chain(meta: dict) -> str | None:
    chains = meta.get("chain_stats", [])
    protein_chains = [c for c in chains if c.get("is_protein")]
    if not protein_chains:
        return None
    return max(protein_chains, key=lambda c: c.get("residues", 0)).get("chain_id")


def write_selection_pdb(raw_path: Path, out_path: Path, chain_id: str | None = None, backbone_only: bool = False):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(raw_path.stem, str(raw_path))
    io = PDBIO()
    io.set_structure(structure)
    if backbone_only:
        class SelectBoth(ProteinChainSelect):
            def accept_atom(self, atom):
                return atom.get_name() in {"N", "CA", "C", "O"}
        io.save(str(out_path), select=SelectBoth(chain_id))
    else:
        io.save(str(out_path), select=ProteinChainSelect(chain_id))


def make_metadata_entry(pdb_id: str, meta: dict, chain_id: str | None) -> dict:
    return {
        "pdb_id": pdb_id,
        "source_cif": str(RAW_DIR / f"{pdb_id}.cif"),
        "selected_chain": chain_id,
        "oligomer_class": meta.get("oligomer_class"),
        "size_class": meta.get("size_class"),
        "rough_fold_class": meta.get("rough_fold_class"),
        "protein_chains": meta.get("protein_chains"),
        "total_chains": meta.get("num_chains"),
    }


def main() -> None:
    RF_DIR.mkdir(parents=True, exist_ok=True)
    (RF_DIR / "backbone_only").mkdir(parents=True, exist_ok=True)
    (RF_DIR / "single_chain").mkdir(parents=True, exist_ok=True)
    (RF_DIR / "metadata").mkdir(parents=True, exist_ok=True)

    registry = []

    for pdb_id in PDB_IDS:
        meta_path = ANALYSIS_DIR / f"{pdb_id}.json"
        raw_path = RAW_DIR / f"{pdb_id}.cif"
        if not meta_path.exists() or not raw_path.exists():
            registry.append({"pdb_id": pdb_id, "status": "missing_input"})
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chain_id = choose_primary_chain(meta)
        if chain_id is None:
            registry.append({"pdb_id": pdb_id, "status": "no_protein_chain"})
            continue

        single_chain_out = RF_DIR / "single_chain" / f"{pdb_id}_{chain_id}.pdb"
        backbone_out = RF_DIR / "backbone_only" / f"{pdb_id}_{chain_id}_bb.pdb"

        write_selection_pdb(raw_path, single_chain_out, chain_id=chain_id, backbone_only=False)
        write_selection_pdb(raw_path, backbone_out, chain_id=chain_id, backbone_only=True)

        entry = make_metadata_entry(pdb_id, meta, chain_id)
        (RF_DIR / "metadata" / f"{pdb_id}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        registry.append({**entry, "status": "ok"})
        print(f"prepared {pdb_id} chain {chain_id}")

    (RF_DIR / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"wrote {RF_DIR / 'registry.json'}")


if __name__ == "__main__":
    main()
