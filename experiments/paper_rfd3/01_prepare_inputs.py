#!/usr/bin/env python3
"""准备论文所有实验需要输入结构和 RFD3 设计规格。

    python 01_prepare_inputs.py                 # 全部准备
    python 01_prepare_inputs.py --only ppi dna  # 只准备指定部分
    python 01_prepare_inputs.py --no-download   # 只用本地已有文件

产物（都在 out/inputs/ 下）：
    ppi/<target>.pdb           蛋白靶点（裁剪后）
    dna/<pdb>_dna.pdb          仅含 DNA 链的靶点
    sm/<ligand>.pdb            仅含配体（+可选邻近残基）的靶点
    ame/<case>.pdb             酶活性位点 motif
    targets/*.json             从 PDB 自动提取的目标清单（链号、残基范围、序列）
    specs/*.json              可直接喂给 `rfd3 design` 的设计规格
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import common  # noqa: E402

try:
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
except Exception as exc:  # noqa: BLE001
    print(f"需要 biopython：{exc}", file=sys.stderr)
    raise

NUCLEIC = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "I",
           "RA", "RC", "RG", "RU"}

# ---------------------------------------------------------------- 默认清单 ---
# 论文 §3.1 / Fig. 3a 的五个靶点。
#
# 权威定义在仓库自带的配置里：
#   models/rfd3/configs/datasets/val/val_examples/bcov_ppi_easy_medium_with_ori.yaml
#   （以及带氢键条件的 bpem_ori_hb.yaml）—— 用 02_extract_repo_benchmarks.py 导出成
#   out/inputs/specs/exp2_ppi_paper.json
# 靶点的 PDB ID 由训练排除清单反推：
#   models/rfd3/configs/datasets/train/pdb/rfd3_train_interface.yaml
#   -> ['3di3', '5o45', '1z92', '2gy5', '4zxb']
#
# 下面这张表说明：benchmark 里的文件是作者自己裁剪+重新编号过的，需要从 PDB 重新裁。
#   benchmark_chain / offset：benchmark 编号 = 沉积编号 - offset，链号也换过。
#   verified：是否已用仓库的 tutorial 示例交叉验证过（示例里用的是沉积编号）。
PPI_PAPER_TARGETS = {
    "pdl1": {
        "pdb_id": "5O45", "deposited_chain": "A", "offset": 16, "verified": True,
        # benchmark 里的 B40/B99/B107 = 沉积编号 A56/A115/A123
    },
    "insulinr": {
        "pdb_id": "4ZXB", "deposited_chain": "E", "offset": 5, "verified": True,
        # benchmark 里的 B59/B83/B91 = 沉积编号 E64/E88/E96
    },
    "tie2": {"pdb_id": "2GY5", "deposited_chain": "B", "offset": 0, "verified": False},
    "il2ra": {"pdb_id": "1Z92", "deposited_chain": "B", "offset": 0, "verified": False},
    # il7ra(3di3) 只在训练排除清单里出现，benchmark 定义没随仓库发布 —— 需要自己按
    # 同样的格式补一份（hotspots 用原子级、contig 用 100-100,/0,<chain>1-<N>）
}

# 反过来也能用仓库里现成的教程结构（已经是沉积编号）直接跑
PPI_BUILTIN = {
    "pdl1": {
        "input": str(common.RFD3_INPUT_PDBS / "5o45_cropped.pdb"),
        "contig": "50-120,/0,A17-131",
        "hotspots": {"A56": "CG,OH", "A115": "CG,SD", "A123": "CD2,OH"},
    },
    "insulinr": {
        "input": str(common.RFD3_INPUT_PDBS / "4zxb_cropped.pdb"),
        "contig": "40-120,/0,E6-155",
        "hotspots": {"E64": "CD2,CZ", "E88": "CG,CZ", "E96": "CD1,CZ"},
    },
}

# 论文 §3.2 / Fig. 3b 的三个 DNA 靶点（训练集外序列）
DNA_PDBS = ["7RTE", "7N5U", "7M5W"]

# 论文 §3.3 / Fig. 3c 的四个配体。
# IAI / OQO 仓库自带；FAD / SAM 需要从 PDB 里裁出来。
SM_LIGANDS = {
    "IAI": {"source": str(common.RFD3_INPUT_PDBS / "IAI.pdb"),
            "ligand": "IAI", "length": "180-180"},
    "OQO": {"source": str(common.RFD3_INPUT_PDBS / "7v11.pdb"),
            "ligand": "OQO", "length": "180-180"},
    # 下面两个默认从 PDB 裁剪；换成你手上任意含该配体的结构即可
    "FAD": {"pdb": "1FAD", "ligand": "FAD", "length": "180-180"},
    "SAM": {"pdb": "1SAM", "ligand": "SAM", "length": "180-180"},
}

BINDER_LENGTH_DNA = (100, 130)


# ------------------------------------------------------------------ 工具 ---


def download_pdb(pdb_id: str, dest: Path, allow_download: bool = True) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    if not allow_download:
        print(f"  [skip] {pdb_id}: 本地无 {dest} 且禁止下载")
        return None
    url = common.RCSB_DOWNLOAD.format(pdb_id=pdb_id.upper())
    print(f"  下载 {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 下载 {pdb_id} 失败: {exc}")
        dest.unlink(missing_ok=True)
        return None
    return dest


def read_any(path: Path):
    parser = MMCIFParser(QUIET=True) if path.suffix in (".cif", ".mmcif") else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def chain_kind(chain) -> str:
    """判断链是 protein / dna / rna / other。"""
    counts = {"protein": 0, "nucleic": 0, "other": 0}
    for residue in chain:
        name = residue.get_resname().strip().upper()
        if name in NUCLEIC:
            counts["nucleic"] += 1
        elif residue.id[0] == " " and len(residue) >= 3:
            counts["protein"] += 1
        else:
            counts["other"] += 1
    if counts["nucleic"] > 0 and counts["nucleic"] >= counts["protein"]:
        return "nucleic"
    if counts["protein"] > 0:
        return "protein"
    return "other"


class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id


class NucleicSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname().strip().upper() in NUCLEIC


class LigandSelect(Select):
    def __init__(self, resname):
        self.resname = resname.upper()

    def accept_residue(self, residue):
        return residue.get_resname().strip().upper() == self.resname


class LigandPlusNeighbors(Select):
    """保留配体 + 距其 5 Å 内的蛋白残基，便于 RFD3 解析口袋环境。"""

    def __init__(self, resname, neighbor_residues):
        self.resname = resname.upper()
        self.neighbors = neighbor_residues

    def accept_residue(self, residue):
        name = residue.get_resname().strip().upper()
        if name == self.resname:
            return True
        key = (residue.get_parent().id, residue.id[1])
        return key in self.neighbors


def save(structure, select, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dest), select=select)
    return dest


def nucleic_sequence(chain) -> str:
    table = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U",
             "A": "A", "C": "C", "G": "G", "U": "U", "I": "I"}
    return "".join(table.get(r.get_resname().strip().upper(), "N") for r in chain)


# --------------------------------------------------------------- 各准备步骤 ---


def parse_target_segment(contig: str) -> tuple[str, int, int] | None:
    """从 contig 里取出靶点链号和残基范围，例如 '100-100,/0,B1-150' -> ('B',1,150)。"""
    m = re.search(r"([A-Za-z]+)(\d+)-(\d+)", contig or "")
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def shift_contig(contig: str, new_chain: str, offset: int) -> str:
    seg = parse_target_segment(contig)
    if seg is None:
        return contig
    old_chain, lo, hi = seg
    return contig.replace(f"{old_chain}{lo}-{hi}",
                          f"{new_chain}{lo + offset}-{hi + offset}")


def shift_hotspots(hotspots: dict, new_chain: str, offset: int) -> dict:
    out = {}
    for res, atoms in (hotspots or {}).items():
        m = re.match(r"([A-Za-z]+)(\d+)", str(res))
        if not m:
            continue
        out[f"{new_chain}{int(m.group(2)) + offset}"] = atoms
    return out


class ChainRangeSelect(Select):
    """只保留指定链的指定残基号区间。"""

    def __init__(self, chain_id: str, lo: int, hi: int):
        self.chain_id, self.lo, self.hi = chain_id, lo, hi

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " " and self.lo <= residue.id[1] <= self.hi


def validate_hotspots(path: Path, chain_id: str, hotspots: dict) -> list[str]:
    """检查热点残基是否存在、且确实含有被指定的原子名。"""
    problems: list[str] = []
    try:
        structure = read_any(path)
    except Exception as exc:  # noqa: BLE001
        return [f"解析失败: {exc}"]
    model = next(structure.get_models())
    if chain_id not in model:
        return [f"链 {chain_id} 不存在"]
    have: dict[int, set[str]] = {}
    for residue in model[chain_id]:
        have[int(residue.id[1])] = {a.get_name() for a in residue.get_atoms()}
    for res, atoms in hotspots.items():
        m = re.match(r"[A-Za-z]+(\d+)", str(res))
        if not m:
            continue
        rid = int(m.group(1))
        if rid not in have:
            problems.append(f"{res} 不存在")
            continue
        for atom in str(atoms).split(","):
            atom = atom.strip()
            if atom and atom not in have[rid]:
                problems.append(f"{res} 没有原子 {atom}")
    return problems


def prepare_ppi_paper(allow_download: bool) -> list[dict]:
    """按仓库自带的 benchmark 定义，从 PDB 重建论文 §3.1 的靶点结构。

    benchmark 里的输入文件是作者裁剪 + 重新编号过的，仓库没随附；
    这里用 PPI_PAPER_TARGETS 里的「沉积链 + 编号偏移」把它还原成沉积编号，
    并用热点原子名做自校验（对不上就跳过并提示用户手工指定 offset）。
    """
    spec_path = common.INPUTS_DIR / "specs" / "exp2_ppi_paper.json"
    if not spec_path.exists():
        print(f"  [skip] 没有 {spec_path}，先运行 02_extract_repo_benchmarks.py")
        return []
    bench = common.load_json(spec_path) or {}
    targets: list[dict] = []

    for name, cfg in PPI_PAPER_TARGETS.items():
        entry = bench.get(name)
        if not entry:
            print(f"  [skip] {name}: benchmark 定义里没有这个靶点")
            continue
        seg = parse_target_segment(entry.get("contig", ""))
        if seg is None:
            print(f"  [skip] {name}: 无法解析 contig {entry.get('contig')}")
            continue
        _, lo, hi = seg
        dep_chain, offset = cfg["deposited_chain"], cfg["offset"]

        dest = common.INPUTS_DIR / "ppi" / f"{name}.pdb"
        if not dest.exists():
            raw = common.INPUTS_DIR / "raw" / f"{cfg['pdb_id'].lower()}.pdb"
            if not download_pdb(cfg["pdb_id"], raw, allow_download):
                continue
            try:
                structure = read_any(raw)
                save(structure, ChainRangeSelect(dep_chain, lo + offset, hi + offset),
                     dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {name}: 裁剪失败 {exc}")
                continue

        hotspots = shift_hotspots(entry.get("select_hotspots", {}), dep_chain, offset)
        problems = validate_hotspots(dest, dep_chain, hotspots)
        if problems:
            print(f"  [warn] {name}: 热点校验不通过（{'；'.join(problems[:3])}）")
            print(f"         如果 benchmark 编号偏移不是 {offset}，请在 "
                  f"PPI_PAPER_TARGETS 里改 offset 后重跑。")
            continue

        targets.append({
            "name": name,
            "input": str(dest),
            "contig": shift_contig(entry["contig"], dep_chain, offset),
            "hotspots": hotspots,
            "hbond_donors": shift_hotspots(
                entry.get("select_hbond_donor", {}), dep_chain, offset),
            "hbond_acceptors": shift_hotspots(
                entry.get("select_hbond_acceptor", {}), dep_chain, offset),
        })
        tag = "已验证" if cfg["verified"] else "偏移待确认"
        print(f"  {name}: {dest}  链{dep_chain} {lo + offset}-{hi + offset}  ({tag})")
    return targets


def prepare_ppi(allow_download: bool) -> dict:
    print("[ppi] 蛋白靶点")

    # 优先用仓库自带的论文 benchmark 定义（5 个靶点）
    targets = prepare_ppi_paper(allow_download)

    # 兜底 / 补充：仓库 tutorial 里现成的两个裁剪结构（沉积编号，可直接跑）
    if len(targets) < len(PPI_PAPER_TARGETS):
        print("  用仓库 tutorial 的现成结构补充：")
        for name, cfg in PPI_BUILTIN.items():
            if any(t["name"] == name for t in targets):
                continue
            src = Path(cfg["input"])
            if not src.exists():
                print(f"  [warn] 缺少 {src}，跳过 {name}")
                continue
            dest = common.INPUTS_DIR / "ppi" / f"{name}_tutorial.pdb"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
            targets.append({"name": name, "input": str(dest),
                            "contig": cfg["contig"], "hotspots": cfg["hotspots"]})
            print(f"  {name}: {dest}")

    # 用户自定义靶点
    extra = common.INPUTS_DIR / "ppi_extra_targets.json"
    if not extra.exists():
        extra.write_text(json.dumps({
            "_说明": ("论文 §3.1 共 5 个靶点：pdl1 / insulinr / tie2 / il2ra / il7ra。"
                      "前四个可以从仓库自带的 benchmark 定义重建；il7ra(3di3) 的 "
                      "benchmark 定义没有随仓库发布，需要在这里手工补。"),
            "_格式": {
                "input": "out/inputs/ppi/il7ra.pdb（自备的裁剪结构）",
                "contig": "<binder长度范围>,/0,<靶点链><起>-<止>",
                "hotspots": {"B40": "CG,CZ", "B99": "CG,SD"},
            },
            "_也可以": ("如果你拿到了作者原始的裁剪文件，直接把 input 指过去、"
                        "contig 用 benchmark 原样的 '100-100,/0,B1-N' 即可。"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  提示：补齐 il7ra 等自定义靶点请编辑 {extra}")
    else:
        user = json.loads(extra.read_text(encoding="utf-8"))
        for name, cfg in user.items():
            if name.startswith("_"):
                continue
            if not Path(cfg["input"]).exists():
                print(f"  [skip] {name}: {cfg['input']} 不存在")
                continue
            targets.append({"name": name, **cfg})
            print(f"  {name}: {cfg['input']}")

    spec = common.spec_ppi(targets)
    common.write_spec(common.INPUTS_DIR / "specs" / "exp2_ppi.json", spec)
    return {"targets": targets}


def prepare_dna(allow_download: bool) -> dict:
    print("[dna] DNA 靶点")
    records = []
    for pdb_id in DNA_PDBS:
        raw = common.INPUTS_DIR / "raw" / f"{pdb_id.lower()}.pdb"
        if not download_pdb(pdb_id, raw, allow_download):
            continue
        try:
            structure = read_any(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 解析 {pdb_id} 失败: {exc}")
            continue
        model = next(structure.get_models())
        dna_chains = [c for c in model.get_chains() if chain_kind(c) == "nucleic"]
        if not dna_chains:
            print(f"  [skip] {pdb_id}: 没找到 DNA 链")
            continue
        for chain in dna_chains:
            dest = common.INPUTS_DIR / "dna" / f"{pdb_id.lower()}_{chain.id}_dna.pdb"
            save(structure, NucleicSelect(), dest)
            ids = [r.id[1] for r in chain]
            records.append({
                "name": f"{pdb_id.lower()}_{chain.id}",
                "pdb_id": pdb_id,
                "input": str(dest),
                "dna_chain": chain.id,
                "dna_first": min(ids),
                "dna_last": max(ids),
                "dna_sel": f"{chain.id}{min(ids)}-{max(ids)}",
                "dna_sequence": nucleic_sequence(chain),
                "n_residues": len(ids),
                "ori_token": [-8, -8, 8],
                "length": f"{BINDER_LENGTH_DNA[0]}-{BINDER_LENGTH_DNA[1]}",
            })
            print(f"  {pdb_id} 链{chain.id}: {len(ids)} nt, "
                  f"序列={nucleic_sequence(chain)[:40]}")

    for rec in records:
        rec["contig"] = (f"{rec['dna_chain']}{rec['dna_first']}-{rec['dna_last']},/0,"
                         f"{BINDER_LENGTH_DNA[0]}-{BINDER_LENGTH_DNA[1]}")

    target_file = common.INPUTS_DIR / "targets" / "dna_targets.json"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"  目标清单 -> {target_file}")

    for diffused in (False, True):
        spec = common.spec_dna(records, diffused=diffused)
        name = "exp3_dna_diffused.json" if diffused else "exp3_dna_rigid.json"
        common.write_spec(common.INPUTS_DIR / "specs" / name, spec)
    return {"targets": records}


def prepare_small_molecule(allow_download: bool) -> dict:
    print("[sm] 小分子配体")
    records = []
    for code, cfg in SM_LIGANDS.items():
        if "source" in cfg and Path(cfg["source"]).exists():
            src, structure = Path(cfg["source"]), read_any(Path(cfg["source"]))
        elif "pdb" in cfg:
            raw = common.INPUTS_DIR / "raw" / f"{cfg['pdb'].lower()}.pdb"
            if not download_pdb(cfg["pdb"], raw, allow_download):
                continue
            src, structure = raw, read_any(raw)
        else:
            print(f"  [skip] {code}: 没有可用来源")
            continue

        ligand_atoms = []
        for residue in next(structure.get_models()).get_residues():
            if residue.get_resname().strip().upper() == cfg["ligand"]:
                ligand_atoms = [a.get_name() for a in residue.get_atoms()]
                break
        if not ligand_atoms:
            print(f"  [skip] {code}: {src.name} 里没有配体 {cfg['ligand']}")
            continue

        dest = common.INPUTS_DIR / "sm" / f"{code}.pdb"
        save(structure, LigandSelect(cfg["ligand"]), dest)
        records.append({
            "name": code,
            "input": str(dest),
            "ligand": cfg["ligand"],
            "length": cfg["length"],
            "atoms": ligand_atoms,
        })
        print(f"  {code}: {len(ligand_atoms)} 个重原子 -> {dest}")

    target_file = common.INPUTS_DIR / "targets" / "sm_ligands.json"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"  目标清单 -> {target_file}")

    for mode in ("fixed", "diffused"):
        spec = common.spec_small_molecule(records, mode=mode)
        common.write_spec(common.INPUTS_DIR / "specs" / f"exp4_sm_{mode}.json", spec)
    return {"targets": records}


def prepare_enzyme(allow_download: bool) -> dict:
    """AME benchmark：41 个 PDB 活性位点。

    数据来自 RFdiffusion2（Ahern et al.）的 Atomic Motif Enzyme benchmark，
    不在本仓库内。这里给出一份可直接编辑的清单模板，并把仓库自带的示例
    enzyme_design.json 转成同样的结构，保证流程可以先跑通。
    """
    print("[enzyme] AME 活性位点")
    cases_file = common.INPUTS_DIR / "targets" / "ame_cases.json"
    cases_file.parent.mkdir(parents=True, exist_ok=True)

    if not cases_file.exists():
        # 用仓库示例初始化，同时给出真实 AME 清单的填写模板
        example_input = common.RFD3_INPUT_PDBS / "M0255_1mg5.pdb"
        seed = [{
            "name": "M0255_demo",
            "input": str(example_input),
            "ligand": "NAI,ACT",
            "unindex": "A108,A139,A152,A156",
            "length": "180-200",
            "fixed_atoms": {
                "A108": "ND2,CG", "A139": "OG,CB,CA",
                "A152": "OH,CZ", "A156": "NZ,CE,CD",
                "ACT": "OXT", "NAI": "",
            },
            "n_islands": 1,
        }]
        cases_file.write_text(json.dumps({
            "_说明": [
                "AME benchmark 的 41 个案例定义在 RFdiffusion2 论文的补充材料里，",
                "官方数据在 https://github.com/RosettaCommons/RFdiffusion （rf2 分支 / AME 目录）。",
                "把它整理成这个 JSON 数组即可：每个案例一条记录。",
                "字段：name / input / ligand / unindex / length / fixed_atoms / n_islands / symmetry_id",
                "n_islands = 该案例的 residue islands 数量，用于复现 Fig. 3d 的分组曲线。",
            ],
            "_seed_example": seed,
        }, indent=2), encoding="utf-8")
        print(f"  [需要人工补齐] {cases_file}")

    data = json.loads(cases_file.read_text(encoding="utf-8"))
    cases = data if isinstance(data, list) else data.get("cases") or data.get("_seed_example") or []
    cases = [c for c in cases if isinstance(c, dict) and "input" in c]
    if not cases:
        print("  [skip] 暂无可用 AME 案例")
        return {"cases": []}

    spec = common.spec_enzyme(cases)
    common.write_spec(common.INPUTS_DIR / "specs" / "exp5_ame.json", spec)
    sym_cases = [c for c in cases if c.get("symmetry_id")]
    if sym_cases:
        common.write_spec(common.INPUTS_DIR / "specs" / "exp5_ame_symmetry.json",
                          common.spec_enzyme(sym_cases, subset="symmetry"))
    return {"cases": cases}


def prepare_symmetry(lengths: list[int] | None = None) -> dict:
    print("[symmetry] 对称规格")
    import os
    ids = os.environ.get("SYMMETRY_IDS", "C3 C5 D2").split()
    spec = common.spec_symmetry(ids, length=100)
    common.write_spec(common.INPUTS_DIR / "specs" / "exp6_symmetry.json", spec)
    return {"ids": ids}


def prepare_unconditional() -> dict:
    print("[uncond] 无条件单体规格")
    import os
    lengths = [int(x) for x in os.environ.get("UNCOND_LENGTHS", "100 150 200").split()]
    spec = common.spec_unconditional(lengths)
    common.write_spec(common.INPUTS_DIR / "specs" / "exp1_unconditional.json", spec)
    # η 扫描（Fig. S1c）
    for eta in ("1.0", "1.5", "2.0", "3.0"):
        common.write_spec(common.INPUTS_DIR / "specs" / f"exp1_eta_{eta}.json", spec)
    return {"lengths": lengths}


def main() -> int:
    ap = argparse.ArgumentParser(description="准备 RFD3 论文实验输入")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["ppi", "dna", "sm", "enzyme", "symmetry", "uncond"])
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    common.ensure_dirs()
    (common.INPUTS_DIR / "specs").mkdir(parents=True, exist_ok=True)
    (common.INPUTS_DIR / "targets").mkdir(parents=True, exist_ok=True)

    steps = args.only or ["uncond", "ppi", "dna", "sm", "enzyme", "symmetry"]
    allow = not args.no_download
    for step in steps:
        {"uncond": prepare_unconditional,
         "ppi": lambda: prepare_ppi(allow),
         "dna": lambda: prepare_dna(allow),
         "sm": lambda: prepare_small_molecule(allow),
         "enzyme": lambda: prepare_enzyme(allow),
         "symmetry": lambda: prepare_symmetry()}[step]()

    print("\n输入准备完成。下一步：")
    print("  ./10_exp1_unconditional.sh      # §3 无条件单体")
    print("  ./11_exp2_ppi.sh                # §3.1 蛋白结合蛋白")
    print("  ./12_exp3_dna.sh                # §3.2 DNA 结合蛋白")
    print("  ./13_exp4_small_molecule.sh     # §3.3 小分子结合蛋白")
    print("  ./14_exp5_enzyme.sh             # §3.4 酶设计")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
