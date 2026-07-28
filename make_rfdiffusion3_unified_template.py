#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any

from Bio.PDB import PDBParser

PDB_IDS = [
    "3BTA", "5N0B", "1MDT", "1DM0", "2AAI", "1ABR", "4UY2",
    "4JTA", "1QLX", "7UMQ", "5OQV", "5O3L", "6A6B",
]

ROOT = Path(".").resolve()
BASE_DIR = ROOT / "rfdiffusion3_inputs"
TEMPLATE_DIR = BASE_DIR / "unified_templates"
SINGLE_CHAIN_DIR = BASE_DIR / "single_chain"


@dataclass
class UnifiedTemplate:
    input: str
    select_fixed_atoms: dict[str, str]
    select_unfixed_sequence: Any
    partial_t: float
    symmetry: Any
    dialect: int = 2
    extra: dict[str, Any] | None = None


def parse_valid_residues(pdb_path: Path, chain_id: str) -> list[int]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    chain = next((c for c in model if c.id == chain_id), None)
    if chain is None:
        raise ValueError(f"chain {chain_id} not found in {pdb_path}")
    residues = []
    for residue in chain:
        hetflag, resseq, _icode = residue.get_id()
        if hetflag != " ":
            continue
        if not all(atom in residue for atom in ("N", "CA", "C", "O")):
            continue
        residues.append(int(resseq))
    if not residues:
        raise ValueError(f"no standard residues with full backbone atoms found in {pdb_path}")
    return sorted(set(residues))


def contiguous_segments(ids: list[int]) -> list[list[int]]:
    if not ids:
        return []
    ids = sorted(set(ids))
    segments: list[list[int]] = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        segments.append([start, prev])
        start = prev = x
    segments.append([start, prev])
    return segments


def merge_segments(segments: list[list[int]]) -> list[list[int]]:
    if not segments:
        return []
    normalized = sorted([[int(s), int(e)] for s, e in segments], key=lambda x: x[0])
    merged = [normalized[0][:]]
    for s, e in normalized[1:]:
        last = merged[-1]
        if s <= last[1] + 1:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return merged


def slice_by_fraction(valid_ids: list[int], start_frac: float, end_frac: float) -> list[list[int]]:
    if not valid_ids:
        return []
    n = len(valid_ids)
    start_idx = max(0, min(n - 1, int(round(n * start_frac))))
    end_idx = max(start_idx, min(n - 1, int(round(n * end_frac)) - 1))
    if start_idx > end_idx:
        return []
    return [[valid_ids[start_idx], valid_ids[end_idx]]]


def trim_to_existing_segments(candidate: list[list[int]], valid_ids: list[int]) -> list[list[int]]:
    valid = set(valid_ids)
    out: list[list[int]] = []
    for s, e in candidate:
        cur = None
        for resi in range(int(s), int(e) + 1):
            if resi in valid:
                if cur is None:
                    cur = [resi, resi]
                else:
                    cur[1] = resi
            else:
                if cur is not None:
                    out.append(cur)
                    cur = None
        if cur is not None:
            out.append(cur)
    return merge_segments(out)


def expand_ranges(ranges: list[list[int]]) -> set[int]:
    expanded: set[int] = set()
    for s, e in ranges:
        expanded.update(range(int(s), int(e) + 1))
    return expanded


def complement_segments(valid_ids: list[int], fixed_ranges: list[list[int]]) -> list[list[int]]:
    fixed = expand_ranges(fixed_ranges)
    return contiguous_segments([resi for resi in valid_ids if resi not in fixed])


def choose_core_from_segments(segments: list[list[int]], start_frac: float, end_frac: float) -> list[list[int]]:
    if not segments:
        return []
    # Prefer the longest continuous segment to avoid holes inside the fixed core.
    seg = max(segments, key=lambda x: x[1] - x[0] + 1)
    start, end = seg
    length = end - start + 1
    core_start = start + int(round(length * start_frac))
    core_end = start + int(round(length * end_frac)) - 1
    core_start = max(start, min(core_start, end))
    core_end = max(core_start, min(core_end, end))
    return [[core_start, core_end]]


def select_window(valid_ids: list[int], window_size: int) -> list[int]:
    if len(valid_ids) <= window_size:
        return valid_ids
    mid = len(valid_ids) // 2
    half = window_size // 2
    start = max(0, mid - half)
    end = min(len(valid_ids), start + window_size)
    start = max(0, end - window_size)
    return valid_ids[start:end]


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


def fixed_atom_dict(chain_id: str, fixed_ranges: list[list[int]]) -> dict[str, str]:
    fixed: dict[str, str] = {}
    for start, end in fixed_ranges:
        for resi in range(int(start), int(end) + 1):
            fixed[f"{chain_id}{resi}"] = "BKBN"
    return fixed


def choose_template_plan(pdb_id: str, meta: dict[str, Any], valid_ids: list[int]) -> dict[str, Any]:
    chain_stats = [c for c in meta.get("chain_stats", []) if c.get("is_protein")]
    if not chain_stats:
        raise ValueError(f"no protein chains found for {pdb_id}")
    primary = max(chain_stats, key=lambda c: c.get("residues", 0))
    chain_id = primary["chain_id"]
    n = len(valid_ids)

    if n < 20:
        raise ValueError(f"too few valid residues in {pdb_id}:{chain_id}")

    # Only mutate about 40 residues; keep the rest fixed.
    diffuse_window = select_window(valid_ids, 40)
    diffuse_ranges = contiguous_segments(diffuse_window)
    fixed_ranges = complement_segments(valid_ids, diffuse_ranges)
    if not diffuse_ranges or not fixed_ranges:
        raise ValueError(f"invalid fixed/diffuse split in {pdb_id}:{chain_id}")

    symmetry = None
    if meta.get("oligomer_class") in {"oligomer_3_to_4", "oligomer_5plus"}:
        symmetry = {"kind": "symmetry", "type": "Cn"}

    return {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "fixed_ranges": fixed_ranges,
        "diffuse_ranges": diffuse_ranges,
        "window_residue_ids": diffuse_window,
        "partial_t": 1.0,
        "symmetry": symmetry,
    }


def main() -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for pdb_id in PDB_IDS:
        meta_path = ROOT / "analysis" / f"{pdb_id}.json"
        if not meta_path.exists():
            summary.append({"pdb_id": pdb_id, "status": "missing_analysis"})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chain_id = max([c for c in meta.get("chain_stats", []) if c.get("is_protein")], key=lambda c: c.get("residues", 0))["chain_id"]
        input_pdb = (SINGLE_CHAIN_DIR / f"{pdb_id}_{chain_id}.pdb").resolve()
        if not input_pdb.exists():
            summary.append({"pdb_id": pdb_id, "status": "missing_input", "input": str(input_pdb)})
            continue

        valid_ids = parse_valid_residues(input_pdb, chain_id)
        plan = choose_template_plan(pdb_id, meta, valid_ids)
        fixed_ranges = merge_segments(plan["fixed_ranges"])
        diffuse_ranges = merge_segments(plan["diffuse_ranges"])
        if not fixed_ranges or not diffuse_ranges:
            summary.append({"pdb_id": pdb_id, "status": "invalid_ranges", "fixed_ranges": fixed_ranges, "diffuse_ranges": diffuse_ranges})
            continue

        spec = UnifiedTemplate(
            input=str(input_pdb),
            select_fixed_atoms=fixed_atom_dict(chain_id, fixed_ranges),
            select_unfixed_sequence=ranges_to_contig(chain_id, diffuse_ranges),
            partial_t=float(plan["partial_t"]),
            symmetry=plan["symmetry"],
            extra={
                "pdb_id": pdb_id,
                "selected_chain": chain_id,
                "valid_residue_ids": valid_ids,
                "window_residue_ids": plan["window_residue_ids"],
                "fixed_ranges": fixed_ranges,
                "diffuse_ranges": diffuse_ranges,
                "symmetry_enabled": bool(plan["symmetry"]),
                "note": "fixed core + diffused complement with aligned experiment semantics",
            },
        )

        out_path = TEMPLATE_DIR / f"{pdb_id}.json"
        out_path.write_text(json.dumps({pdb_id: asdict(spec)}, indent=2), encoding="utf-8")
        summary.append({"pdb_id": pdb_id, "status": "ok", "output": str(out_path)})
        print(f"wrote {out_path}")

    (TEMPLATE_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {TEMPLATE_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
