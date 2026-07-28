#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Any

PDB_IDS = [
    "3BTA", "5N0B", "1MDT", "1DM0", "2AAI", "1ABR", "4UY2",
    "4JTA", "1QLX", "7UMQ", "5OQV", "5O3L", "6A6B"
]

ROOT = Path(".")
ANALYSIS_DIR = ROOT / "analysis"
RF_PLAN_DIR = ROOT / "rfdiffusion3_inputs" / "plans"


@dataclass
class ResiduePlan:
    pdb_id: str
    selected_chain: str
    chain_length: int
    fixed_ranges: list[list[int]]
    diffuse_ranges: list[list[int]]
    rationale: str
    symmetry: str | None = None


def contig_to_ranges(start: int, end: int) -> list[list[int]]:
    if start > end:
        return []
    return [[start, end]]


def complement_ranges(length: int, fixed: list[list[int]]) -> list[list[int]]:
    fixed_set = set()
    for s, e in fixed:
        fixed_set.update(range(s, e + 1))
    ranges = []
    run_start = None
    for i in range(1, length + 1):
        if i not in fixed_set:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                ranges.append([run_start, i - 1])
                run_start = None
    if run_start is not None:
        ranges.append([run_start, length])
    return ranges


def choose_primary_chain(meta: dict) -> dict[str, Any]:
    chains = [c for c in meta.get("chain_stats", []) if c.get("is_protein")]
    if not chains:
        raise ValueError(f"no protein chains found for {meta.get('pdb_id')}")
    return max(chains, key=lambda c: c.get("residues", 0))


def build_plan(meta: dict) -> ResiduePlan:
    pdb_id = meta["pdb_id"]
    chain = choose_primary_chain(meta)
    chain_id = chain["chain_id"]
    length = int(chain["residues"])

    fixed: list[list[int]] = []
    symmetry: str | None = None
    rationale = ""

    # Conservative defaults based on target class.
    if pdb_id == "1QLX":
        fixed = contig_to_ranges(10, 95)
        rationale = "small monomer: keep the compact core fixed and diffuse both termini."
    elif pdb_id in {"5N0B", "3BTA"}:
        fixed = contig_to_ranges(max(8, length // 5), min(length - 8, (length * 4) // 5))
        rationale = "single-chain globular/elongated target: freeze the central scaffold, diffuse edge loops and termini."
    elif pdb_id in {"1MDT", "2AAI", "1ABR"}:
        fixed = contig_to_ranges(max(10, length // 6), min(length - 10, (length * 5) // 6))
        rationale = "dimeric target: keep the main fold core fixed; later add interface-specific masking if needed."
    elif pdb_id == "4UY2":
        fixed = contig_to_ranges(25, 175)
        rationale = "mixed oligomer: preserve the longest chain core and diffuse external loops / short auxiliary chains in a later pass."
    elif pdb_id in {"1DM0", "4JTA"}:
        fixed = contig_to_ranges(20, max(20, min(length - 20, length - 25)))
        rationale = "large oligomer: keep the repeated/scaffold-like middle region fixed and diffuse chain ends; add symmetry after baseline."
    elif pdb_id in {"7UMQ", "5OQV", "5O3L", "6A6B"}:
        fixed = contig_to_ranges(8, length - 8)
        symmetry = "Cn"
        rationale = "highly repetitive oligomer: fix the core of each repeat unit and use symmetry-conditioned partial diffusion."
    else:
        fixed = contig_to_ranges(max(8, length // 4), min(length - 8, (length * 3) // 4))
        rationale = "generic conservative scaffold plan."

    diffuse = complement_ranges(length, fixed)

    return ResiduePlan(
        pdb_id=pdb_id,
        selected_chain=chain_id,
        chain_length=length,
        fixed_ranges=fixed,
        diffuse_ranges=diffuse,
        rationale=rationale,
        symmetry=symmetry,
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    RF_PLAN_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for pdb_id in PDB_IDS:
        meta_path = ANALYSIS_DIR / f"{pdb_id}.json"
        if not meta_path.exists():
            summary.append({"pdb_id": pdb_id, "status": "missing_analysis"})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        plan = build_plan(meta)
        payload = asdict(plan)
        write_json(RF_PLAN_DIR / f"{pdb_id}.json", payload)
        summary.append(payload)
        print(f"planned {pdb_id}: chain {plan.selected_chain}, fixed {plan.fixed_ranges}, diffuse {plan.diffuse_ranges}")

    write_json(RF_PLAN_DIR / "summary.json", summary)
    print(f"wrote {RF_PLAN_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
