#!/usr/bin/env python3
"""计算论文里的几何指标：把 RFD3 设计结构与对应的折叠预测配对后逐条打分。

    python 70_metrics_geometry.py
    python 70_metrics_geometry.py --experiment exp2_ppi --limit 20

产物：out/metrics/geometry.csv
列：experiment, condition, design, fold, metric 名...

配对规则：折叠输出的目录名以设计文件名为前缀（60_fold.py 的命名约定），
所以用 `<设计名>` 去 out/folds/<exp>/<cond>/ 下做前缀匹配即可。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import common, metrics  # noqa: E402

DESIGN_PATTERNS = ("*_denoised_model_*.cif.gz", "*_denoised_model_*.cif",
                   "*_model_*.cif.gz", "*_model_*.cif")

FIELDS = [
    "experiment", "condition", "design", "fold", "status",
    "target_aligned_binder_rmsd",
    "dna_aligned_protein_rmsd",
    "backbone_rmsd",
    "ligand_rmsd",
    "motif_all_atom_rmsd",
    "ligand_rasa",
    "n_hbonds",
    "n_clashes",
    "n_interface_residues",
    "interface_contact_preserved",
    "interface_charge_preserved",
    "com_shift",
    "subunit_rmsd",
    "n_chains",
    "note",
]


# ------------------------------------------------------------------ 辅助 ---


def find_designs(run_dir: Path) -> list[Path]:
    for pat in DESIGN_PATTERNS:
        hits = sorted(run_dir.rglob(pat))
        if hits:
            return hits
    return []


def find_folds(folds_dir: Path, design_stem: str) -> list[Path]:
    """找出该骨架对应的全部折叠模型（每条骨架有 4-8 条 MPNN 序列）。"""
    if not folds_dir.exists():
        return []
    hits = sorted(folds_dir.rglob(f"*{design_stem}*_model.cif"))
    if hits:
        return hits
    return [c for c in sorted(folds_dir.rglob(f"*{design_stem}*"))
            if c.is_file() and c.suffix in (".cif", ".pdb")]


# 论文口径：一条骨架的得分取它所有 MPNN 序列里最好的那条
# （DNA："The minimum RMSD for each backbone was taken as a representative"）
MIN_KEYS = {"target_aligned_binder_rmsd", "dna_aligned_protein_rmsd",
            "backbone_rmsd", "ligand_rmsd", "motif_all_atom_rmsd", "subunit_rmsd",
            "n_clashes"}
MAX_KEYS = {"ligand_rasa", "n_hbonds", "n_interface_residues", "n_chains"}


def aggregate(per_fold: list[dict]) -> dict:
    """把同一骨架的多条序列聚合成一行（RMSD 取最小，其余取最大）。"""
    ok = [p for p in per_fold if p.get("status") == "ok"]
    if not ok:
        first = per_fold[0] if per_fold else {"status": "no_fold"}
        return {**first, "n_folds": len(per_fold)}
    out: dict = {"status": "ok", "n_folds": len(per_fold)}
    for key in MIN_KEYS | MAX_KEYS:
        vals = [p.get(key) for p in ok]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue
        out[key] = min(vals) if key in MIN_KEYS else max(vals)
    return out


NUCLEIC_RES = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U",
               "RA", "RC", "RG", "RU"}


def chain_roles(structure) -> dict[str, str]:
    """判断每条链是 protein / dna / ligand。"""
    roles: dict[str, str] = {}
    for chain in metrics.first_model(structure).get_chains():
        n_prot = n_nuc = n_het = 0
        for residue in chain:
            name = residue.get_resname().strip().upper()
            if residue.id[0] != " ":
                n_het += 1
            elif name in NUCLEIC_RES:
                n_nuc += 1
            else:
                n_prot += 1
        if n_nuc > n_prot:
            roles[chain.id] = "dna"
        elif n_prot > 0:
            roles[chain.id] = "protein"
        else:
            roles[chain.id] = "ligand"
    return roles


def chain_ca(chain) -> np.ndarray:
    idx = metrics.atom_index(chain, ["CA"])
    return np.stack([idx[k] for k in sorted(idx)]) if idx else np.zeros((0, 3))


def trim_terminal(ca: np.ndarray, n_trim: int = 3) -> np.ndarray:
    if len(ca) <= 2 * n_trim + 3:
        return ca
    return ca[n_trim:len(ca) - n_trim]


def pick_protein_chains(structure) -> list:
    roles = chain_roles(structure)
    return [c for c in metrics.first_model(structure).get_chains()
            if roles.get(c.id) == "protein"]


def pick_dna_chains(structure) -> list:
    roles = chain_roles(structure)
    return [c for c in metrics.first_model(structure).get_chains()
            if roles.get(c.id) == "dna"]


def all_hetero_atoms(structure) -> dict[tuple[str, int, str], np.ndarray]:
    out = {}
    for chain in metrics.first_model(structure).get_chains():
        for residue in chain:
            if residue.id[0] == " ":
                continue
            for atom in residue.get_atoms():
                out[(chain.id, int(residue.id[1]), atom.get_name())] = \
                    np.asarray(atom.get_coord(), dtype=float)
    return out


def paired_atoms(d_left: dict, d_right: dict):
    keys = sorted(set(d_left) & set(d_right))
    if not keys:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    return (np.stack([d_left[k] for k in keys]),
            np.stack([d_right[k] for k in keys]), keys)


def interface_preservation(d_chain, f_chain,
                           d_partner: np.ndarray, f_partner: np.ndarray) -> dict:
    """Fig. S9a-b：界面接触在 MPNN 之后是否保留、以及电荷类别是否不变。"""
    if len(d_partner) == 0 or len(f_partner) == 0:
        return {}
    d_if = metrics.interface_residues(d_chain, d_partner)
    f_if = metrics.interface_residues(f_chain, f_partner)
    if not d_if:
        return {}
    frac, charge = metrics.preserved_contacts(
        d_if, f_if, metrics.charge_map(d_chain), metrics.charge_map(f_chain))
    out = {"n_interface_residues": len(d_if)}
    if frac is not None:
        out["interface_contact_preserved"] = frac
    if charge is not None:
        out["interface_charge_preserved"] = charge
    return out


# ------------------------------------------------------------ 各实验指标 ---


def pick_binder(chains) -> object:
    """挑出 binder：优先链 A（RFD3 生成的链就是 A），否则取最短的蛋白链。"""
    for c in chains:
        if c.id == "A":
            return c
    return min(chains, key=lambda c: len(chain_ca(c)))


def metric_ppi(design: Path, fold: Path) -> dict:
    """Fig. 3a：对齐靶点 Cα（预测 vs 设计），再算 binder Cα RMSD。"""
    ds, fs = metrics.read_structure(design), metrics.read_structure(fold)
    d_prot, f_prot = pick_protein_chains(ds), pick_protein_chains(fs)
    if len(d_prot) < 2 or len(f_prot) < 2:
        return {"status": "cannot_split_chains"}
    d_chain, f_chain = pick_binder(d_prot), pick_binder(f_prot)
    d_target = [c for c in d_prot if c.id != d_chain.id][0]
    f_target = [c for c in f_prot if c.id != f_chain.id][0]

    d_t = metrics.atom_index(d_target, ["CA"])
    f_t = metrics.atom_index(f_target, ["CA"])
    d_b = metrics.atom_index(d_chain, ["CA"])
    f_b = metrics.atom_index(f_chain, ["CA"])
    d_tc, f_tc, _ = paired_atoms(d_t, f_t)
    d_bc, f_bc, _ = paired_atoms(d_b, f_b)
    rmsd = metrics.aligned_rmsd(d_tc, f_tc, d_bc, f_bc)
    d_t_all = all_protein_coords(d_target)
    f_t_all = all_protein_coords(f_target)
    out = {"status": "ok", "target_aligned_binder_rmsd": rmsd}
    out.update(interface_preservation(d_chain, f_chain, d_t_all, f_t_all))
    return out


def metric_dna(design: Path, fold: Path) -> dict:
    """Fig. 3b：对齐 DNA 磷酸原子，算蛋白 Cα RMSD（裁剪末端 loop）。"""
    ds, fs = metrics.read_structure(design), metrics.read_structure(fold)
    d_dna = pick_dna_chains(ds)
    f_dna = pick_dna_chains(fs)
    d_prot, f_prot = pick_protein_chains(ds), pick_protein_chains(fs)
    if not d_dna or not f_dna or not d_prot or not f_prot:
        return {"status": "missing_dna_or_protein"}
    d_c, f_c = d_dna[0], f_dna[0]
    d_p = metrics.atom_index(d_c, list(metrics.PHOSPHATE_ATOMS))
    f_p = metrics.atom_index(f_c, list(metrics.PHOSPHATE_ATOMS))
    d_pc, f_pc, _ = paired_atoms(d_p, f_p)
    d_ca = trim_terminal(chain_ca(d_prot[0]))
    f_ca = trim_terminal(chain_ca(f_prot[0]))
    n = min(len(d_ca), len(f_ca))
    rmsd = metrics.aligned_rmsd(d_pc, f_pc, d_ca[:n], f_ca[:n])
    out = {"status": "ok", "dna_aligned_protein_rmsd": rmsd}
    out.update(interface_preservation(
        d_prot[0], f_prot[0],
        np.stack(list(metrics.atom_index(d_c).values())) if d_p else np.zeros((0, 3)),
        np.stack(list(metrics.atom_index(f_c).values())) if f_p else np.zeros((0, 3))))
    return out


def all_protein_coords(chain) -> np.ndarray:
    """链上全部蛋白重原子坐标（算 RASA / 氢键必须用全原子，不能用骨架）。"""
    coords = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        for atom in residue.get_atoms():
            coords.append(np.asarray(atom.get_coord(), dtype=float))
    return np.stack(coords) if coords else np.zeros((0, 3))


def all_backbone_coords(chain) -> np.ndarray:
    """链上骨架原子（N, CA, C, O）坐标 —— 论文的 clash 判据是配体 vs 骨架。"""
    coords = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        for atom in residue.get_atoms():
            if atom.get_name() in ("N", "CA", "C", "O"):
                coords.append(np.asarray(atom.get_coord(), dtype=float))
    return np.stack(coords) if coords else np.zeros((0, 3))


def metric_small_molecule(design: Path, fold: Path) -> dict:
    """Fig. 3c：蛋白骨架(N,CA,C)对齐后，算配体重原子的 RMSD。"""
    ds, fs = metrics.read_structure(design), metrics.read_structure(fold)
    d_prot = pick_protein_chains(ds)
    f_prot = pick_protein_chains(fs)
    if not d_prot or not f_prot:
        return {"status": "no_protein"}
    d_bb = metrics.atom_index(d_prot[0], ["N", "CA", "C"])
    f_bb = metrics.atom_index(f_prot[0], ["N", "CA", "C"])
    d_bbc, f_bbc, _ = paired_atoms(d_bb, f_bb)

    d_het, f_het = all_hetero_atoms(ds), all_hetero_atoms(fs)
    d_names = {k[2] for k in d_het}
    f_names = {k[2] for k in f_het}
    common_names = d_names & f_names
    bb_rmsd = metrics.aligned_rmsd(d_bbc, f_bbc, d_bbc, f_bbc)
    if not common_names:
        return {"status": "no_ligand_in_fold", "backbone_rmsd": bb_rmsd}
    d_l = {k: v for k, v in d_het.items() if k[2] in common_names}
    f_l = {k: v for k, v in f_het.items() if k[2] in common_names}
    d_lc, f_lc, _ = paired_atoms(d_l, f_l)

    # RASA / 氢键用整条链的全原子（含侧链），否则会严重高估暴露度
    protein_atoms = all_protein_coords(d_prot[0])
    rasa = metrics.ligand_rasa(d_lc, protein_atoms) if len(d_lc) else None
    hb = metrics.hbond_pairs(d_lc, protein_atoms)
    out = {"status": "ok", "backbone_rmsd": bb_rmsd, "ligand_rmsd":
           metrics.aligned_rmsd(d_bbc, f_bbc, d_lc, f_lc),
           "ligand_rasa": rasa, "n_hbonds": hb}
    out.update(interface_preservation(d_prot[0], f_prot[0], d_lc, f_lc))
    return out



def metric_enzyme(design: Path, fold: Path) -> dict:
    """Fig. 3d：对齐 motif 骨架，算 motif 全原子 RMSD。"""
    ds, fs = metrics.read_structure(design), metrics.read_structure(fold)
    d_prot, f_prot = pick_protein_chains(ds), pick_protein_chains(fs)
    if not d_prot or not f_prot:
        return {"status": "no_protein"}
    d_all = metrics.atom_index(d_prot[0])
    f_all = metrics.atom_index(f_prot[0])
    bb = ["N", "CA", "C", "O"]
    d_bbc, f_bbc, keys = paired_atoms(
        {(k[0], k[1]): v for k, v in d_all.items() if k[1] in bb},
        {(k[0], k[1]): v for k, v in f_all.items() if k[1] in bb})
    d_het, f_het = all_hetero_atoms(ds), all_hetero_atoms(fs)
    if d_het and f_het:
        d_lc, f_lc, _ = paired_atoms(d_het, f_het)
        rmsd = metrics.aligned_rmsd(d_bbc, f_bbc, d_lc, f_lc)
        if rmsd is not None:
            # 论文的成功判据还要求"配体与预测骨架原子无 clash"
            f_bb_only = all_backbone_coords(f_prot[0])
            out = {"status": "ok", "motif_all_atom_rmsd": rmsd}
            out["n_clashes"] = metrics.count_clashes(f_lc, f_bb_only)
            return out
    # 没有配体时退化为整体骨架 RMSD（仅作粗筛）
    return {"status": "protein_only", "motif_all_atom_rmsd":
            metrics.aligned_rmsd(d_bbc, f_bbc, d_bbc, f_bbc)}


def metric_unconditional(design: Path, fold: Path) -> dict:
    """§3：无条件设计 vs 预测的整体 Cα RMSD（Kabsch 叠合）。

    论文口径："designs have at least one sequence that is predicted by AF3 to
    fold within 1.5 Å RMSD to the backbone of the design model"。
    """
    ds, fs = metrics.read_structure(design), metrics.read_structure(fold)
    d_prot, f_prot = pick_protein_chains(ds), pick_protein_chains(fs)
    if not d_prot or not f_prot:
        return {"status": "no_protein"}
    d_ca, f_ca = chain_ca(d_prot[0]), chain_ca(f_prot[0])
    n = min(len(d_ca), len(f_ca))
    if n < 3:
        return {"status": "too_short"}
    return {"status": "ok",
            "backbone_rmsd": metrics.aligned_rmsd(d_ca[:n], f_ca[:n], d_ca[:n], f_ca[:n])}


def metric_symmetry(design: Path, _fold: Path | None) -> dict:
    """Fig. 2g：统计链数并计算首末两条链叠合后的 Cα RMSD。"""
    ds = metrics.read_structure(design)
    chains = pick_protein_chains(ds)
    if len(chains) < 2:
        return {"status": "not_multichain", "n_chains": len(chains)}
    a, b = chain_ca(chains[0]), chain_ca(chains[1])
    n = min(len(a), len(b))
    rmsd = metrics.aligned_rmsd(a[:n], b[:n], a[:n], b[:n])
    return {"status": "ok", "n_chains": len(chains), "subunit_rmsd": rmsd}


def metric_conditioning(design: Path, fold: Path | None, condition: str) -> dict:
    """Fig. 2d-f：氢键数量、配体 RASA、质心偏移。"""
    ds = metrics.read_structure(design)
    d_prot = pick_protein_chains(ds)
    d_het = all_hetero_atoms(ds)
    out: dict = {"status": "ok"}
    if d_prot and d_het:
        protein_atoms = all_protein_coords(d_prot[0])
        d_lc = np.stack(list(d_het.values()))
        out["n_hbonds"] = metrics.hbond_pairs(d_lc, protein_atoms)
        out["ligand_rasa"] = metrics.ligand_rasa(d_lc, protein_atoms)
    if "com" in condition and d_prot:
        ca = chain_ca(d_prot[0])
        if len(ca):
            out["com_shift"] = float(np.linalg.norm(ca.mean(axis=0)))
    return out


def dispatch(experiment: str, design: Path, fold: Path | None,
             condition: str) -> dict:
    if fold is None and not experiment.startswith("exp6"):
        return {"status": "no_fold"}
    try:
        if experiment.startswith("exp1"):
            return metric_unconditional(design, fold)
        if experiment.startswith("exp2"):
            return metric_ppi(design, fold)
        if experiment.startswith("exp3"):
            return metric_dna(design, fold)
        if experiment.startswith("exp4"):
            return metric_small_molecule(design, fold)
        if experiment.startswith("exp5"):
            return metric_enzyme(design, fold)
        if experiment.startswith("exp6"):
            return metric_symmetry(design, fold)
        if experiment.startswith("exp7"):
            return metric_conditioning(design, fold, condition)
        if experiment.startswith("exp9"):
            return metric_enzyme(design, fold)
    except Exception as exc:  # noqa: BLE001
        return {"status": f"error: {type(exc).__name__}: {exc}"}
    return {"status": "not_applicable"}


def score_one(exp: str, cond: str, design: Path) -> dict:
    """算一个设计的全部几何指标（供进程池调用）。"""
    stem = design.name.split(".")[0]
    folds = find_folds(common.FOLDS_DIR / exp / cond, stem)
    if folds:
        res = aggregate([dispatch(exp, design, f, cond) for f in folds])
    else:
        res = dispatch(exp, design, None, cond)
    row = {k: "" for k in FIELDS}
    row.update({"experiment": exp, "condition": cond, "design": stem,
                "fold": f"{len(folds)} folds" if folds else ""})
    for k, v in res.items():
        if k in row:
            row[k] = "" if v is None else v
    return row


def _score_task(task):
    """进程池的入口（参数必须可 pickle）。"""
    exp, cond, design = task
    try:
        return score_one(exp, cond, design)
    except Exception as exc:  # noqa: BLE001
        row = {k: "" for k in FIELDS}
        row.update({"experiment": exp, "condition": cond,
                    "design": design.name.split(".")[0],
                    "status": f"worker_error: {type(exc).__name__}: {exc}"})
        return row


def main() -> int:
    ap = argparse.ArgumentParser(description="计算论文几何指标")
    ap.add_argument("--experiment", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="每个条件最多算多少条")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("N_WORKERS", "8")),
                    help="并行进程数（解析 CIF / 算 SASA 都是 CPU 活）")
    args = ap.parse_args()

    common.ensure_dirs()
    out_path = common.METRICS_DIR / "geometry.csv"
    rows: list[dict] = []

    tasks: list[tuple[str, str, Path]] = []
    if common.DESIGNS_DIR.exists():
        for exp_dir in sorted(p for p in common.DESIGNS_DIR.iterdir() if p.is_dir()):
            exp = exp_dir.name
            if args.experiment and exp not in args.experiment:
                continue
            for cond_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                cond = cond_dir.name
                designs = find_designs(cond_dir)
                if args.limit:
                    designs = designs[:args.limit]
                if not designs:
                    continue
                print(f"[{exp}/{cond}] {len(designs)} 个设计")
                tasks.extend((exp, cond, d) for d in designs)

    if tasks:
        workers = max(1, args.workers)
        print(f"\n共 {len(tasks)} 个设计待打分，用 {workers} 个进程并行\n")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_score_task, tasks, chunksize=4):
                rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{len(rows)} 条记录，其中 {n_ok} 条成功配对折叠输出")
    print(f"wrote {out_path}")
    if rows and n_ok == 0:
        print("提示：没有匹配到折叠结果，检查 WITH_FOLD / FOLD_BACKEND 是否开启。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
