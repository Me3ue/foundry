#!/usr/bin/env python3
"""对 RFD3 生成的骨架做 MPNN 序列设计（论文里每条骨架配 4 或 8 条序列）。

    python 50_sequence_design.py --experiment exp2_ppi --n-seqs 4
    python 50_sequence_design.py --experiment exp4_small_molecule --model ligand_mpnn --n-seqs 8
    python 50_sequence_design.py --all --n-seqs 2          # 全部实验

产物：out/sequences/<experiment>/<condition>/<design_stem>/*.{fa,cif}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import common  # noqa: E402

RC_ENV_BIN = Path(os.environ.get("RC_ENV_BIN", "/home/zhangzijian/anaconda3/envs/rc/bin"))
MPNN = RC_ENV_BIN / "mpnn"

DESIGN_PATTERNS = ("*_denoised_model_*.cif.gz", "*_denoised_model_*.cif",
                   "*_model_*.cif.gz", "*_model_*.cif", "*.cif.gz", "*.cif")

# 哪些实验用 LigandMPNN（有非蛋白原子），哪些用 ProteinMPNN
LIGAND_EXPERIMENTS = {"exp3_dna", "exp4_small_molecule", "exp5_enzyme_ame",
                      "exp7_conditioning"}
PROTEIN_EXPERIMENTS = {"exp1_unconditional", "exp2_ppi", "exp6_symmetry"}

SPEC_BY_EXPERIMENT = {
    "exp1_unconditional": ["exp1_unconditional.json"],
    "exp2_ppi": ["exp2_ppi.json"],
    "exp3_dna": ["exp3_dna_rigid.json", "exp3_dna_diffused.json"],
    "exp4_small_molecule": ["exp4_sm_fixed.json", "exp4_sm_diffused.json"],
    "exp5_enzyme_ame": ["exp5_ame.json"],
    "exp6_symmetry": ["exp6_symmetry.json"],
}


def find_designs(run_dir: Path) -> list[Path]:
    for pat in DESIGN_PATTERNS:
        hits = sorted(run_dir.rglob(pat))
        if hits:
            return hits
    return []


NUCLEIC_RES = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U",
               "RA", "RC", "RG", "RU"}

# 这些实验里 binder 只是结构中的一条链，MPNN 只应重设计它
BINDER_ONLY_EXPERIMENTS = {"exp2_ppi", "exp3_dna", "exp4_small_molecule",
                           "exp7_conditioning"}


def _chain_kinds(path: Path) -> dict[str, str]:
    """返回 {chain_id: 'protein'|'dna'|'ligand'}。"""
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except Exception:  # noqa: BLE001
        return {}
    parser = (MMCIFParser(QUIET=True) if path.suffix in (".cif", ".mmcif")
              or path.name.endswith(".cif.gz") else PDBParser(QUIET=True))
    try:
        if path.name.endswith(".cif.gz"):
            import gzip
            with gzip.open(path, "rt", errors="ignore") as fh:
                structure = parser.get_structure(path.stem, fh)
        else:
            structure = parser.get_structure(path.stem, str(path))
    except Exception:  # noqa: BLE001
        return {}
    kinds: dict[str, str] = {}
    for chain in next(structure.get_models()).get_chains():
        n_prot = n_nuc = 0
        for residue in chain:
            name = residue.get_resname().strip().upper()
            if residue.id[0] != " ":
                continue
            if name in NUCLEIC_RES:
                n_nuc += 1
            else:
                n_prot += 1
        kinds[chain.id] = "dna" if n_nuc > n_prot else ("protein" if n_prot else "other")
    return kinds


def designed_chains_for(design: Path, experiment: str, spec_entry: dict | None) -> str | None:
    """推断哪条链是 RFD3 新生成的 binder。

    规则：设计结构里的蛋白链 减去 输入靶点里的链 = binder。
    Fig. 3a/3b 的做法都是只重设计 binder、保留靶点/DNA 界面。
    """
    if experiment not in BINDER_ONLY_EXPERIMENTS:
        return None
    kinds = _chain_kinds(design)
    if not kinds:
        return None
    protein_chains = [cid for cid, k in kinds.items() if k == "protein"]
    if len(protein_chains) <= 1:
        return None
    target_chains: set[str] = set()
    if spec_entry and spec_entry.get("input"):
        target_chains = set(_chain_kinds(Path(spec_entry["input"])))
    binder = [c for c in protein_chains if c not in target_chains]
    if not binder:
        return None
    return ",".join(binder)


def fixed_residues_for(experiment: str, design_key: str) -> str | None:
    """从设计规格里取出被固定的残基（酶设计需要固定 motif 侧链）。"""
    for spec_name in SPEC_BY_EXPERIMENT.get(experiment, []):
        spec_path = common.INPUTS_DIR / "specs" / spec_name
        spec = common.load_json(spec_path)
        if not spec:
            continue
        entry = spec.get(design_key)
        if not entry:
            continue
        keys = list((entry.get("select_fixed_atoms") or {}).keys())
        keys = [k for k in keys if k and not k.isalpha()]  # 去掉纯配体代码
        if keys:
            return json.dumps(keys)
    return None


def spec_entry_for(experiment: str, design_key: str) -> dict | None:
    for spec_name in SPEC_BY_EXPERIMENT.get(experiment, []):
        spec = common.load_json(common.INPUTS_DIR / "specs" / spec_name)
        if spec and design_key in spec:
            return spec[design_key]
    return None


def run_mpnn(model_type: str, struct: Path, out_dir: Path, n_seqs: int,
             fixed: str | None, ckpt: Path, seed: int = 0,
             designed_chains: str | None = None) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [str(MPNN),
            "--model_type", model_type,
            "--checkpoint_path", str(ckpt),
            "--is_legacy_weights", "True",
            "--structure_path", str(struct),
            "--out_directory", str(out_dir),
            "--write_fasta", "True",
            "--write_structures", "True",
            "--batch_size", str(n_seqs),
            "--number_of_batches", "1",
            "--seed", str(seed)]
    if fixed:
        args += ["--fixed_residues", fixed]
    elif designed_chains:
        args += ["--designed_chains", designed_chains]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    [fail] {struct.name}: "
              f"{(proc.stderr or proc.stdout).strip().splitlines()[-1:]}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="批量 MPNN 序列设计")
    ap.add_argument("--experiment", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="处理 out/designs 下全部实验")
    ap.add_argument("--model", choices=["auto", "protein_mpnn", "ligand_mpnn"],
                    default="auto")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("N_WORKERS", "8")),
                    help="并行进程数（MPNN 是 CPU 任务，大内存机器直接开 NPROC/2）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    experiments = None
    if not args.all:
        experiments = args.experiment if args.experiment else sorted(
            p.name for p in common.DESIGNS_DIR.iterdir() if p.is_dir()
        ) if common.DESIGNS_DIR.exists() else []

    ckpt = Path(os.environ.get(
        "LIGANDMPNN_CKPT",
        str(Path(os.environ.get("MPNN_CKPT_DIR", Path.home() / ".foundry/checkpoints"))
            / "ligandmpnn_v_32_010_25.pt")))
    protein_ckpt = Path(os.environ.get(
        "PROTEINMPNN_CKPT",
        str(Path(os.environ.get("MPNN_CKPT_DIR", Path.home() / ".foundry/checkpoints"))
            / "proteinmpnn_v_48_020.pt")))

    if not MPNN.exists():
        print(f"找不到 mpnn 可执行文件: {MPNN}", file=sys.stderr)
        return 1

    # 先把所有待办任务收集起来，再丢进进程池
    tasks: list[tuple[str, Path, Path, str, str | None, str | None, Path]] = []
    for exp in experiments:
        exp_dir = common.DESIGNS_DIR / exp
        if not exp_dir.exists():
            print(f"[skip] {exp}: 目录不存在 {exp_dir}")
            continue
        for cond_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            designs = find_designs(cond_dir)
            if not designs:
                continue
            print(f"[{exp}/{cond_dir.name}] {len(designs)} 个骨架")
            for design in designs:
                stem = design.name.split(".")[0]
                out_dir = common.SEQS_DIR / exp / cond_dir.name / stem
                if out_dir.exists() and not args.overwrite:
                    hits = list(out_dir.glob("*.fa")) + list(out_dir.glob("*.fasta"))
                    if hits:
                        continue
                mtype = args.model
                if mtype == "auto":
                    mtype = "ligand_mpnn" if exp in LIGAND_EXPERIMENTS else "protein_mpnn"
                ck = ckpt if mtype == "ligand_mpnn" else protein_ckpt
                if not Path(ck).exists():
                    print(f"  [skip] 缺少 MPNN 权重 {ck}")
                    return 1
                key = _design_key(stem, exp)
                fixed = fixed_residues_for(exp, key)
                chains = None
                if not fixed:
                    chains = designed_chains_for(design, exp, spec_entry_for(exp, key))
                tasks.append((mtype, design, out_dir, stem, fixed, chains, Path(ck)))

    if not tasks:
        print("没有需要处理的设计（都已经有序列了）。")
        return 0

    n_ok = n_fail = 0
    workers = max(1, args.workers)
    print(f"\n共 {len(tasks)} 个骨架待设计，用 {workers} 个进程并行\n")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, t, args.n_seqs, args.seed): t for t in tasks
        }
        for fut in as_completed(futures):
            mtype, design, _, stem, fixed, chains, _ = futures[fut]
            try:
                ok = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [err] {stem}: {type(exc).__name__}: {exc}")
                ok = False
            n_ok += int(ok)
            n_fail += int(not ok)
            if ok:
                detail = f"fixed={fixed}" if fixed else (
                    f"designed_chains={chains}" if chains else "全部重设计")
                print(f"  ok  {stem}  ({mtype}, {args.n_seqs} 条序列, {detail})")

    print(f"\n完成：成功 {n_ok}，失败 {n_fail}")
    return 0


def _run_one(task, n_seqs: int, seed: int) -> bool:
    mtype, design, out_dir, _stem, fixed, chains, ckpt = task
    return run_mpnn(mtype, design, out_dir, n_seqs, fixed, ckpt,
                    seed=seed, designed_chains=chains)


def _design_key(stem: str, experiment: str) -> str:
    """从设计文件名里还原出规格里的 JSON key。

    RFD3 输出名形如 `<prefix>_<jsonkey>_<batch>_model_<n>`，
    这里用规格文件的 key 列表做最长前缀匹配。
    """
    for spec_name in SPEC_BY_EXPERIMENT.get(experiment, []):
        spec = common.load_json(common.INPUTS_DIR / "specs" / spec_name)
        if not spec:
            continue
        for key in spec:
            if key in stem:
                return key
    return stem


if __name__ == "__main__":
    raise SystemExit(main())
