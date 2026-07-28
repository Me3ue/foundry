#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from collections import Counter, defaultdict

from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

PDB_IDS = [
    "3BTA", "5N0B", "1MDT", "1DM0", "2AAI", "1ABR", "4UY2",
    "4JTA", "1QLX", "7UMQ", "5OQV", "5O3L", "6A6B"
]

ROOT = Path(".")
RAW_DIR = ROOT / "raw_pdb"
CLEAN_DIR = ROOT / "clean_pdb"
META_DIR = ROOT / "metadata"
OUT_DIR = ROOT / "analysis"

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}

HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "PRO"}


@dataclass
class ChainStats:
    pdb_id: str
    chain_id: str
    residues: int
    atoms: int
    first_residue: str | None
    last_residue: str | None
    sequence: str
    composition: dict
    is_protein: bool


@dataclass
class StructureStats:
    pdb_id: str
    exists_raw: bool
    exists_clean: bool
    num_models: int
    num_chains: int
    protein_chains: int
    total_residues: int
    total_atoms: int
    protein_residues: int
    hetero_residues: int
    water_residues: int
    missing_residues_hint: int
    chain_stats: list
    oligomer_class: str
    size_class: str
    rough_fold_class: str
    notes: list


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def classify_oligomer(num_protein_chains: int, total_chains: int) -> str:
    if num_protein_chains <= 1:
        return "monomer"
    if num_protein_chains == 2:
        return "dimer"
    if num_protein_chains <= 4:
        return "oligomer_3_to_4"
    return "oligomer_5plus"


def classify_size(total_residues: int) -> str:
    if total_residues < 100:
        return "small"
    if total_residues < 250:
        return "medium"
    return "large"


def rough_fold_class(chain_stats: list[ChainStats]) -> str:
    if not chain_stats:
        return "unknown"
    avg_hydrophobic = sum(sum(v for aa, v in cs.composition.items() if aa in HYDROPHOBIC) for cs in chain_stats) / max(1, sum(cs.residues for cs in chain_stats))
    if avg_hydrophobic > 0.45:
        return "likely_globular"
    return "mixed_or_elongated"


def chain_sequence(chain) -> tuple[str, Counter, int, int, str | None, str | None, bool]:
    residues = []
    comp = Counter()
    atoms = 0
    first = None
    last = None
    protein_like = 0
    for residue in chain:
        hetflag, resseq, icode = residue.get_id()
        if hetflag != " ":
            continue
        resname = residue.get_resname().strip().upper()
        if first is None:
            first = f"{resname}{resseq}{icode.strip()}"
        last = f"{resname}{resseq}{icode.strip()}"
        comp[resname] += 1
        atoms += len(residue)
        if resname in AA3:
            protein_like += 1
            try:
                residues.append(seq1(resname))
            except Exception:
                residues.append("X")
        else:
            residues.append("X")
    is_protein = protein_like > 0 and protein_like / max(1, len(residues)) > 0.8
    return "".join(residues), comp, len(residues), atoms, first, last, is_protein


def analyze_structure(pdb_id: str) -> StructureStats:
    raw_path = RAW_DIR / f"{pdb_id}.cif"
    clean_path = CLEAN_DIR / f"{pdb_id}.pdb"
    exists_raw = raw_path.exists()
    exists_clean = clean_path.exists()

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(raw_path if exists_raw else clean_path))
    model = next(structure.get_models())

    chain_stats: list[ChainStats] = []
    total_residues = total_atoms = protein_residues = hetero_residues = water_residues = 0

    for chain in model:
        seq, comp, residues, atoms, first, last, is_protein = chain_sequence(chain)
        if residues == 0 and atoms == 0:
            continue
        chain_stats.append(
            ChainStats(
                pdb_id=pdb_id,
                chain_id=chain.id,
                residues=residues,
                atoms=atoms,
                first_residue=first,
                last_residue=last,
                sequence=seq,
                composition=dict(comp),
                is_protein=is_protein,
            )
        )
        for residue in chain:
            hetflag = residue.get_id()[0]
            total_atoms += len(residue)
            if hetflag == " ":
                total_residues += 1
                protein_residues += 1
            elif hetflag == "W":
                water_residues += 1
                total_residues += 1
            else:
                hetero_residues += 1
                total_residues += 1

    protein_chains = sum(1 for cs in chain_stats if cs.is_protein)
    num_chains = len(chain_stats)
    notes = []
    if protein_chains == 0:
        notes.append("no_protein_chain_detected")
    if num_chains > protein_chains:
        notes.append("contains_nonprotein_or_hybrid_chains")
    if protein_chains > 1:
        notes.append("likely_oligomeric")
    if any(cs.residues < 30 for cs in chain_stats if cs.is_protein):
        notes.append("has_short_protein_chain")

    return StructureStats(
        pdb_id=pdb_id,
        exists_raw=exists_raw,
        exists_clean=exists_clean,
        num_models=1,
        num_chains=num_chains,
        protein_chains=protein_chains,
        total_residues=total_residues,
        total_atoms=total_atoms,
        protein_residues=protein_residues,
        hetero_residues=hetero_residues,
        water_residues=water_residues,
        missing_residues_hint=0,
        chain_stats=[asdict(cs) for cs in chain_stats],
        oligomer_class=classify_oligomer(protein_chains, num_chains),
        size_class=classify_size(protein_residues),
        rough_fold_class=rough_fold_class(chain_stats),
        notes=notes,
    )


def main() -> None:
    ensure_dirs()
    results = []
    by_class = defaultdict(list)

    for pdb_id in PDB_IDS:
        try:
            stats = analyze_structure(pdb_id)
            obj = asdict(stats)
            results.append(obj)
            by_class[stats.oligomer_class].append(pdb_id)
            (OUT_DIR / f"{pdb_id}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
            print(f"analyzed {pdb_id}")
        except Exception as e:
            err = {"pdb_id": pdb_id, "status": "failed", "error": str(e)}
            results.append(err)
            (OUT_DIR / f"{pdb_id}.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
            print(f"failed {pdb_id}: {e}")

    summary = {
        "structures": results,
        "oligomer_groups": {k: v for k, v in by_class.items()},
        "counts": {
            "total": len(results),
            "monomer": len(by_class.get("monomer", [])),
            "dimer": len(by_class.get("dimer", [])),
            "oligomer_3_to_4": len(by_class.get("oligomer_3_to_4", [])),
            "oligomer_5plus": len(by_class.get("oligomer_5plus", [])),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
