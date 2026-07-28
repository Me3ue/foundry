#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import gzip
import json
import math
import re
import shutil
import subprocess
from typing import Iterable

from Bio.PDB import MMCIFParser, PDBParser, Superimposer

ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "rfdiffusion3_inputs" / "unified_templates"
OUT_ROOT = ROOT / "logs" / "inference_outs"
REPORT_DIR = ROOT / "logs" / "rfdiffusion3_batch"
CSV_PATH = REPORT_DIR / "structure_metrics.csv"
JSON_PATH = REPORT_DIR / "structure_metrics.json"
MD_PATH = REPORT_DIR / "structure_metrics.md"

VARIANTS = ["baseline", "fixed", "partial", "symmetry"]


@dataclass
class MetricRow:
    pdb_id: str
    variant: str
    status: str
    reference: str | None = None
    generated: str | None = None
    chain_id: str | None = None
    fixed_core: str | None = None
    diffuse: str | None = None
    rmsd_ca: float | None = None
    rmsd_fixed_ca: float | None = None
    rmsd_diffuse_ca: float | None = None
    tm_score: float | None = None
    local_stability: str | None = None
    note: str | None = None


def load_templates() -> dict[str, dict]:
    data = {}
    if not TEMPLATE_DIR.exists():
        return data
    for path in TEMPLATE_DIR.glob("*.json"):
        if path.name == "summary.json":
            continue
        try:
            obj = json.loads(path.read_text())
            pdb_id = next(iter(obj))
            data[pdb_id] = obj[pdb_id]
        except Exception:
            continue
    return data


def parse_range_list(ranges: Iterable[Iterable[int]]) -> list[int]:
    ids: list[int] = []
    for start, end in ranges:
        ids.extend(list(range(int(start), int(end) + 1)))
    return ids


def fixed_residue_ids(template: dict) -> set[int]:
    ids: set[int] = set()
    for rng in template.get("extra", {}).get("fixed_ranges", []):
        ids.update(range(int(rng[0]), int(rng[1]) + 1))
    return ids


def diffuse_residue_ids(template: dict) -> set[int]:
    ids: set[int] = set()
    for rng in template.get("extra", {}).get("diffuse_ranges", []):
        ids.update(range(int(rng[0]), int(rng[1]) + 1))
    return ids


def format_ranges(template: dict, key: str, chain_id: str | None) -> str | None:
    ranges = template.get("extra", {}).get(key, [])
    if not ranges:
        return None
    chain = chain_id or template.get("extra", {}).get("selected_chain") or ""
    parts = []
    for start, end in ranges:
        if int(start) == int(end):
            parts.append(f"{chain}{int(start)}")
        else:
            parts.append(f"{chain}{int(start)}-{int(end)}")
    return ", ".join(parts)


def read_structure(path: Path):
    if path.name.lower().endswith(".cif.gz"):
        parser = MMCIFParser(QUIET=True)
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return parser.get_structure(path.stem, fh)
    if path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def first_structure_file(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None
    patterns = ("*.pdb", "*.cif", "*.mmcif", "*.cif.gz")
    candidates = []
    for pat in patterns:
        candidates.extend(out_dir.rglob(pat))
    if not candidates:
        return None
    def rank(p: Path):
        name = p.name.lower()
        score = 0
        if "model_0" in name:
            score += 100
        if "denoised" in name:
            score += 10
        if name.endswith(".cif.gz"):
            score += 5
        return (-score, len(p.parts), str(p))
    return sorted(candidates, key=rank)[0]


def get_chain(structure, chain_id: str):
    model = next(structure.get_models())
    if chain_id in model:
        return model[chain_id]
    return next(model.get_chains())


def atom_map(chain, atom_name: str = "CA") -> dict[int, object]:
    mapping = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        resi = residue.id[1]
        if atom_name in residue:
            mapping[resi] = residue[atom_name]
    return mapping


def collect_atoms(chain, res_ids: Iterable[int], atom_name: str = "CA") -> list[object]:
    amap = atom_map(chain, atom_name)
    atoms = []
    for rid in res_ids:
        atom = amap.get(int(rid))
        if atom is not None:
            atoms.append(atom)
    return atoms


def fit_transform(ref_atoms, gen_atoms):
    if len(ref_atoms) < 3 or len(ref_atoms) != len(gen_atoms):
        return None
    sup = Superimposer()
    sup.set_atoms(ref_atoms, gen_atoms)
    return sup.rotran


def rmsd_after_transform(ref_atoms, gen_atoms, rotran) -> float | None:
    if len(ref_atoms) < 3 or len(ref_atoms) != len(gen_atoms) or rotran is None:
        return None
    rot, tran = rotran
    total = 0.0
    for ref_atom, gen_atom in zip(ref_atoms, gen_atoms):
        gx, gy, gz = gen_atom.get_coord()
        x = rot[0][0] * gx + rot[0][1] * gy + rot[0][2] * gz + tran[0]
        y = rot[1][0] * gx + rot[1][1] * gy + rot[1][2] * gz + tran[1]
        z = rot[2][0] * gx + rot[2][1] * gy + rot[2][2] * gz + tran[2]
        rx, ry, rz = ref_atom.get_coord()
        total += (x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2
    return float(math.sqrt(total / len(ref_atoms)))


def tm_score_from_aligned_atoms(ref_atoms, gen_atoms, length_norm: int, rotran=None) -> float | None:
    if len(ref_atoms) < 3 or len(ref_atoms) != len(gen_atoms) or length_norm <= 0:
        return None
    if rotran is None:
        rotran = fit_transform(ref_atoms, gen_atoms)
    if rotran is None:
        return None
    rot, tran = rotran
    total = 0.0
    for ref_atom, gen_atom in zip(ref_atoms, gen_atoms):
        gx, gy, gz = gen_atom.get_coord()
        x = rot[0][0] * gx + rot[0][1] * gy + rot[0][2] * gz + tran[0]
        y = rot[1][0] * gx + rot[1][1] * gy + rot[1][2] * gz + tran[1]
        z = rot[2][0] * gx + rot[2][1] * gy + rot[2][2] * gz + tran[2]
        rx, ry, rz = ref_atom.get_coord()
        d = math.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)
        total += 1.0 / (1.0 + (d / 1.24) ** 2)
    return float(total / length_norm)


def tm_align_score(gen_path: Path, ref_path: Path, ref_atoms=None, gen_atoms=None, rotran=None) -> float | None:
    exe = shutil.which("TMalign") or shutil.which("TM-align") or shutil.which("USalign")
    if exe:
        try:
            proc = subprocess.run([exe, str(gen_path), str(ref_path)], capture_output=True, text=True, check=False)
            text = proc.stdout + "\n" + proc.stderr
        except Exception:
            text = ""
        for line in text.splitlines():
            m = re.search(r"TM-score\s*=\s*([0-9.]+)", line)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    if ref_atoms is None or gen_atoms is None:
        return None
    length_norm = max(len(ref_atoms), len(gen_atoms), 1)
    return tm_score_from_aligned_atoms(ref_atoms, gen_atoms, length_norm)


def local_stability_label(rmsd_fixed: float | None, rmsd_diffuse: float | None, tm: float | None) -> str | None:
    if rmsd_fixed is None and rmsd_diffuse is None and tm is None:
        return None
    fixed_ok = rmsd_fixed is not None and rmsd_fixed < 2.0
    diffuse_moved = rmsd_diffuse is not None and rmsd_diffuse >= 2.0
    tm_ok = tm is not None and tm >= 0.5
    if fixed_ok and tm_ok and diffuse_moved:
        return "good"
    if fixed_ok and tm_ok:
        return "moderate"
    return "weak"


def process_one(pdb_id: str, variant: str, template: dict) -> MetricRow:
    chain_id = template.get("extra", {}).get("selected_chain")
    ref_path = Path(template.get("input", ""))
    out_dir = OUT_ROOT / pdb_id / variant
    gen_path = first_structure_file(out_dir)

    fixed_ranges = format_ranges(template, "fixed_ranges", chain_id)
    diffuse_ranges = format_ranges(template, "diffuse_ranges", chain_id)

    if not ref_path.exists():
        return MetricRow(
            pdb_id=pdb_id,
            variant=variant,
            status="missing_reference",
            reference=str(ref_path),
            generated=str(gen_path) if gen_path else None,
            chain_id=chain_id,
            fixed_core=fixed_ranges,
            diffuse=diffuse_ranges,
            note="reference structure not found",
        )
    if gen_path is None or not gen_path.exists():
        return MetricRow(
            pdb_id=pdb_id,
            variant=variant,
            status="missing_output",
            reference=str(ref_path),
            generated=None,
            chain_id=chain_id,
            fixed_core=fixed_ranges,
            diffuse=diffuse_ranges,
            note="no generated structure file found",
        )

    try:
        ref_struct = read_structure(ref_path)
        gen_struct = read_structure(gen_path)
        ref_chain = get_chain(ref_struct, chain_id) if chain_id else next(next(ref_struct.get_models()).get_chains())
        gen_chain = get_chain(gen_struct, chain_id) if chain_id else next(next(gen_struct.get_models()).get_chains())
    except Exception as e:
        return MetricRow(
            pdb_id=pdb_id,
            variant=variant,
            status="parse_error",
            reference=str(ref_path),
            generated=str(gen_path),
            chain_id=chain_id,
            fixed_core=fixed_ranges,
            diffuse=diffuse_ranges,
            note=str(e),
        )

    ref_all = atom_map(ref_chain, "CA")
    gen_all = atom_map(gen_chain, "CA")
    common = sorted(set(ref_all) & set(gen_all))
    ref_atoms = [ref_all[i] for i in common]
    gen_atoms = [gen_all[i] for i in common]
    all_rotran = fit_transform(ref_atoms, gen_atoms)
    rmsd_all = rmsd_after_transform(ref_atoms, gen_atoms, all_rotran)

    fixed_ids = sorted(fixed_residue_ids(template))
    diffuse_ids = sorted(diffuse_residue_ids(template))
    ref_fixed_atoms = collect_atoms(ref_chain, fixed_ids, "CA")
    gen_fixed_atoms = collect_atoms(gen_chain, fixed_ids, "CA")
    ref_diff_atoms = collect_atoms(ref_chain, diffuse_ids, "CA")
    gen_diff_atoms = collect_atoms(gen_chain, diffuse_ids, "CA")

    rmsd_fixed = rmsd_after_transform(ref_fixed_atoms, gen_fixed_atoms, all_rotran)
    rmsd_diff = rmsd_after_transform(ref_diff_atoms, gen_diff_atoms, all_rotran)
    tm = tm_align_score(gen_path, ref_path, ref_atoms=ref_atoms, gen_atoms=gen_atoms, rotran=all_rotran)

    return MetricRow(
        pdb_id=pdb_id,
        variant=variant,
        status="ok",
        reference=str(ref_path),
        generated=str(gen_path),
        chain_id=chain_id,
        fixed_core=fixed_ranges,
        diffuse=diffuse_ranges,
        rmsd_ca=rmsd_all,
        rmsd_fixed_ca=rmsd_fixed,
        rmsd_diffuse_ca=rmsd_diff,
        tm_score=tm,
        local_stability=local_stability_label(rmsd_fixed, rmsd_diff, tm),
    )


def render_markdown(rows: list[MetricRow]) -> str:
    lines = []
    lines.append("# RFdiffusion3 structure metrics")
    lines.append("")
    lines.append("| PDB | Variant | Status | RMSD(CA) | RMSD(fixed) | RMSD(diffuse) | TM-score | Local stability |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.pdb_id} | {r.variant} | {r.status} | {'' if r.rmsd_ca is None else f'{r.rmsd_ca:.6f}'} | "
            f"{'' if r.rmsd_fixed_ca is None else f'{r.rmsd_fixed_ca:.6f}'} | "
            f"{'' if r.rmsd_diffuse_ca is None else f'{r.rmsd_diffuse_ca:.6f}'} | "
            f"{'' if r.tm_score is None else f'{r.tm_score:.6f}'} | {r.local_stability or ''} |"
        )
    return "\n".join(lines)


def main() -> None:
    templates = load_templates()
    rows: list[MetricRow] = []
    for pdb_id, template in sorted(templates.items()):
        for variant in VARIANTS:
            rows.append(process_one(pdb_id, variant, template))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    JSON_PATH.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    MD_PATH.write_text(render_markdown(rows), encoding="utf-8")

    print(f"wrote {CSV_PATH}")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
