#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any

from Bio.PDB import PDBParser

PDB_IDS = [
    "3BTA", "5N0B", "1MDT", "1DM0", "2AAI", "1ABR", "4UY2",
    "4JTA", "1QLX", "7UMQ", "5OQV", "5O3L", "6A6B"
]

ROOT = Path(".").resolve()
INPUT_DIR = ROOT / "rfdiffusion3_inputs"
PLAN_DIR = INPUT_DIR / "plans"
DESIGN_DIR = INPUT_DIR / "design_specs"
SINGLE_CHAIN_DIR = INPUT_DIR / "single_chain"


@dataclass
class DesignSpec:
    input: str
    select_fixed_atoms: Any
    partial_t: float | None
    select_unfixed_sequence: Any
    symmetry: Any
    dialect: int = 2
    extra: dict[str, Any] | None = None


def parse_residue_ids(pdb_path: Path, chain_id: str) -> list[int]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    chain = next((c for c in model if c.id == chain_id), None)
    if chain is None:
        raise ValueError(f"chain {chain_id} not found in {pdb_path}")
    residue_ids = []
    for residue in chain:
        hetflag, resseq, _icode = residue.get_id()
        if hetflag == " ":
            residue_ids.append(int(resseq))
    if not residue_ids:
        raise ValueError(f"no standard residues found in {pdb_path}")
    return sorted(residue_ids)


def clamp_ranges(ranges: list[list[int]], valid_ids: list[int]) -> list[list[int]]:
    if not valid_ids:
        return []
    valid_set = set(valid_ids)
    clamped: list[list[int]] = []
    for start, end in ranges:
        s = max(min(start, end), valid_ids[0])
        e = min(max(start, end), valid_ids[-1])
        if s > e:
            continue
        current = None
        for resi in range(s, e + 1):
            if resi in valid_set:
                if current is None:
                    current = [resi, resi]
                else:
                    current[1] = resi
            else:
                if current is not None:
                    clamped.append(current)
                    current = None
        if current is not None:
            clamped.append(current)
    return clamped


def ranges_to_contig(chain_id: str, ranges: list[list[int]]) -> str | bool:
    if not ranges:
        return False
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(f"{chain_id}{start}")
        else:
            parts.append(f"{chain_id}{start}-{end}")
    return ",".join(parts)


def fixed_atoms_map(chain_id: str, ranges: list[list[int]]) -> dict[str, str]:
    fixed: dict[str, str] = {}
    for start, end in ranges:
        for resi in range(start, end + 1):
            fixed[f"{chain_id}{resi}"] = "BKBN"
    return fixed


def build_spec(pdb_id: str, plan: dict[str, Any]) -> DesignSpec:
    chain_id = plan["selected_chain"]
    input_pdb = (SINGLE_CHAIN_DIR / f"{pdb_id}_{chain_id}.pdb").resolve()
    valid_ids = parse_residue_ids(input_pdb, chain_id)

    fixed_ranges = clamp_ranges(plan.get("fixed_ranges", []), valid_ids)
    diffuse_ranges = clamp_ranges(plan.get("diffuse_ranges", []), valid_ids)
    symmetry = plan.get("symmetry")

    fixed_atoms = fixed_atoms_map(chain_id, fixed_ranges)
    diffuse_contig = ranges_to_contig(chain_id, diffuse_ranges)

    if pdb_id in {"1QLX", "5N0B"}:
        partial_t = 4.0
    elif pdb_id in {"3BTA", "2AAI", "1ABR"}:
        partial_t = 6.0
    elif pdb_id in {"1MDT", "4UY2"}:
        partial_t = 7.5
    else:
        partial_t = 9.0

    sym = None
    if symmetry:
        sym = {"kind": "symmetry", "type": symmetry}

    return DesignSpec(
        input=str(input_pdb),
        select_fixed_atoms=fixed_atoms,
        partial_t=partial_t,
        select_unfixed_sequence=diffuse_contig,
        symmetry=sym,
        extra={
            "pdb_id": pdb_id,
            "selected_chain": chain_id,
            "valid_residue_ids": valid_ids,
            "fixed_ranges": fixed_ranges,
            "diffuse_ranges": diffuse_ranges,
            "note": plan.get("rationale"),
        },
    )


def main() -> None:
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for pdb_id in PDB_IDS:
        plan_path = PLAN_DIR / f"{pdb_id}.json"
        if not plan_path.exists():
            summary.append({"pdb_id": pdb_id, "status": "missing_plan"})
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        spec = build_spec(pdb_id, plan)
        out_path = DESIGN_DIR / f"{pdb_id}.json"
        out_path.write_text(json.dumps({pdb_id: asdict(spec)}, indent=2), encoding="utf-8")
        summary.append({"pdb_id": pdb_id, "status": "ok", "input": str(out_path)})
        print(f"wrote {out_path}")

    (DESIGN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {DESIGN_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
