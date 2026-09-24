#!/usr/bin/env python3
"""对 MPNN 设计出的序列做结构预测（自洽性 / self-consistency 评估）。

论文用 AlphaFold3 做这一步（酶用 Chai-1）。本仓库可以本地跑的等价后端是 RF3。
脚本负责把 MPNN 的序列组装成各后端的输入格式，然后调用对应命令。

    FOLD_BACKEND=rf3  python 60_fold.py --experiment exp2_ppi
    FOLD_BACKEND=none python 60_fold.py            # 只生成输入文件，不跑预测

产物：
    out/folds/<experiment>/<condition>/rf3_inputs.json
    out/folds/<experiment>/<condition>/**/*_summary_confidences.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import common  # noqa: E402

RC_ENV_BIN = Path(os.environ.get("RC_ENV_BIN", "/home/zhangzijian/anaconda3/envs/rc/bin"))
RF3 = RC_ENV_BIN / "rf3"
RF3_CKPT = os.environ.get(
    "RF3_CKPT",
    str(Path.home() / ".foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt"))

SPEC_BY_EXPERIMENT = {
    "exp1_unconditional": ["exp1_unconditional.json"],
    "exp2_ppi": ["exp2_ppi.json"],
    "exp3_dna": ["exp3_dna_rigid.json", "exp3_dna_diffused.json"],
    "exp4_small_molecule": ["exp4_sm_fixed.json", "exp4_sm_diffused.json"],
    "exp5_enzyme_ame": ["exp5_ame.json"],
    "exp6_symmetry": ["exp6_symmetry.json"],
}

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "X": "UNK",
}


def read_first_fasta(path: Path) -> str | None:
    seq_parts: list[str] = []
    started = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if started:
                    break
                started = True
                continue
            if started and line:
                seq_parts.append(line)
    return "".join(seq_parts) or None


def collect_sequences(seq_dir: Path) -> list[tuple[str, str]]:
    """返回 [(sample_name, sequence)]。"""
    out: list[tuple[str, str]] = []
    for fa in sorted(list(seq_dir.glob("*.fa")) + list(seq_dir.glob("*.fasta"))):
        seq = read_first_fasta(fa)
        if seq:
            out.append((fa.stem, seq))
    return out


def protein_sequence_from_pdb(path: Path, chain_id: str | None = None) -> str | None:
    try:
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.PDB.Polypeptide import three_to_one
    except Exception:  # noqa: BLE001
        return None
    parser = (MMCIFParser(QUIET=True) if Path(path).suffix in (".cif", ".mmcif")
              else PDBParser(QUIET=True))
    try:
        structure = parser.get_structure(Path(path).stem, str(path))
    except Exception:  # noqa: BLE001
        return None
    model = next(structure.get_models())
    chains = [c for c in model.get_chains() if chain_id is None or c.id == chain_id]
    for chain in chains:
        letters = []
        for residue in chain:
            if residue.id[0] != " ":
                continue
            try:
                letters.append(three_to_one(residue.get_resname().strip().upper()))
            except Exception:  # noqa: BLE001
                letters.append("X")
        if len(letters) >= 20:
            return "".join(letters)
    return None


def spec_for(experiment: str) -> dict:
    merged: dict = {}
    for name in SPEC_BY_EXPERIMENT.get(experiment, []):
        merged.update(common.load_json(common.INPUTS_DIR / "specs" / name) or {})
    return merged


def build_components(experiment: str, seq: str, key: str, spec: dict,
                     targets: dict) -> list[dict]:
    """按实验类型组装 RF3 的 components。"""
    comps: list[dict] = [{"seq": seq, "chain_id": "A"}]
    entry = spec.get(key, {})

    if experiment == "exp2_ppi":
        target = targets["ppi"].get(key)
        if target:
            tseq = protein_sequence_from_pdb(Path(target["input"]))
            if tseq:
                comps.append({"seq": tseq, "chain_id": "B"})

    elif experiment == "exp3_dna":
        for rec in targets["dna"]:
            if rec["name"] == key.rsplit("_", 1)[0]:
                comps.append({"seq": rec["dna_sequence"], "chain_id": "B"})
                break

    elif experiment in ("exp4_small_molecule", "exp5_enzyme_ame",
                        "exp7_conditioning"):
        lig = entry.get("ligand")
        if lig:
            for code in str(lig).split(","):
                code = code.strip()
                if code:
                    comps.append({"ccd_code": code})

    return comps


def load_targets() -> dict:
    ppi = common.load_json(common.INPUTS_DIR / "targets" / "ppi_targets.json") or []
    if not ppi:
        spec = common.load_json(common.INPUTS_DIR / "specs" / "exp2_ppi.json") or {}
        ppi = [{"name": k, "input": v.get("input")} for k, v in spec.items()]
    dna = common.load_json(common.INPUTS_DIR / "targets" / "dna_targets.json") or []
    return {"ppi": {t["name"]: t for t in ppi}, "dna": dna}


def main() -> int:
    ap = argparse.ArgumentParser(description="批量结构预测（自洽性评估）")
    ap.add_argument("--experiment", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="每个条件最多折叠多少条序列")
    ap.add_argument("--backend", default=os.environ.get("FOLD_BACKEND", "rf3"))
    ap.add_argument("--shards", type=int,
                    default=int(os.environ.get("FOLD_SHARDS", "6")),
                    help="把折叠任务拆成几片（≈ 用几张卡）")
    ap.add_argument("--parallel", default=os.environ.get("FOLD_PARALLEL_GPUS", "auto"),
                    help="GPU 池并行度；auto = 按空闲显存决定")
    args = ap.parse_args()

    backend = args.backend
    experiments = args.experiment or (
        sorted(p.name for p in common.SEQS_DIR.iterdir() if p.is_dir())
        if common.SEQS_DIR.exists() else [])
    targets = load_targets()
    written = 0
    jobs: list[str] = []

    for exp in experiments:
        exp_seq_dir = common.SEQS_DIR / exp
        if not exp_seq_dir.exists():
            print(f"[skip] {exp}: 还没有序列（先跑 50_sequence_design.py）")
            continue
        spec = spec_for(exp)
        for cond_dir in sorted(p for p in exp_seq_dir.iterdir() if p.is_dir()):
            samples: list[tuple[str, str]] = []
            for design_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
                key = _design_key(design_dir.name, spec)
                for name, seq in collect_sequences(design_dir):
                    samples.append((f"{design_dir.name}__{name}", seq))
            if not samples:
                continue
            if args.limit:
                samples = samples[:args.limit]

            out_dir = common.FOLDS_DIR / exp / cond_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = []
            for name, seq in samples:
                key = _design_key(name, spec)
                payload.append({
                    "name": name,
                    "components": build_components(exp, seq, key, spec, targets),
                })
            spec_path = out_dir / "fold_inputs.json"
            spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written += 1
            print(f"[{exp}/{cond_dir.name}] {len(payload)} 条序列 -> {spec_path}")

            if backend != "rf3":
                print(f"  backend={backend}：仅生成输入文件，请交给 {backend} 运行")
                continue
            if not RF3.exists():
                print(f"  [skip] 找不到 rf3 可执行文件 {RF3}")
                continue

            # 分片：把一次大 fold 拆成 N 份，铺到 N 张卡上同时跑。
            # RF3 自己也能多卡分流，但显式分片更可控、日志也更清楚。
            n_shards = max(1, min(args.shards, len(payload)))
            for i in range(n_shards):
                chunk = payload[i::n_shards]
                shard_path = out_dir / f"fold_inputs_shard{i}.json"
                shard_path.write_text(json.dumps(chunk, indent=2), encoding="utf-8")
                ckpt_part = f" ckpt_path={RF3_CKPT}" if Path(RF3_CKPT).exists() else ""
                jobs.append(f'"{RF3}" fold inputs={shard_path} out_dir={out_dir}'
                            f"{ckpt_part}")

    print(f"\n共写出 {written} 个折叠输入文件，{len(jobs)} 个折叠任务")
    if not jobs:
        return 0

    pool = common.PAPER_ROOT / "lib" / "gpu_pool.py"
    if not pool.exists():
        print("找不到 lib/gpu_pool.py，改为串行执行：")
        for job in jobs:
            subprocess.run(job, shell=True, cwd=str(common.PAPER_ROOT))
        return 0

    env = dict(os.environ)
    cmd = [sys.executable, str(pool), "--jobs-file", "-",
           "--min-free-mem", env.get("FOLD_MIN_FREE_MEM", "20000"),
           "--job-mem", env.get("FOLD_JOB_MEM", "20000")]
    if args.parallel and args.parallel != "auto":
        cmd += ["--parallel", str(args.parallel)]
    print(f"提交折叠任务到 GPU 池（shards={args.shards}，parallel={args.parallel}）")
    proc = subprocess.run(cmd, input="\n".join(jobs), text=True,
                          cwd=str(common.PAPER_ROOT), env=env)
    return proc.returncode


def _design_key(stem: str, spec: dict) -> str:
    for key in spec:
        if key in stem:
            return key
    return stem


if __name__ == "__main__":
    raise SystemExit(main())
