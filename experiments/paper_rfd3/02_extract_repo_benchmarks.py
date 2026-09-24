#!/usr/bin/env python3
"""把仓库里自带的 benchmark 定义（Hydra 验证集配置）导出成可运行的 RFD3 输入规格。

论文的评测数据集定义**就在这个仓库的 config 里**，不需要自己去翻补充材料：

    models/rfd3/configs/datasets/val/*.yaml                 各个 in silico benchmark
    models/rfd3/configs/datasets/val/val_examples/*.yaml    PPI benchmark（内联定义）
    models/rfd3/configs/datasets/train/pdb/rfd3_train_*.yaml 训练时的排除清单
                                                             -> 反推出 holdout 测试集

本脚本做三件事：
  1. 扫描上面所有配置，输出一份「benchmark -> 需要的数据文件 -> 仓库里有没有」的清单
  2. 把 PPI benchmark（bcov_af3_ppi 系列）翻译成 dialect-2 的 RFD3 输入 JSON
     （注意：配置里用的是遗留字段名 atom_level_hotspots / hbond_donors，
       运行时要改成 select_hotspots / select_hbond_donor / select_hbond_acceptor）
  3. 从训练过滤器里提取 holdout 的 PDB ID 清单

    python 02_extract_repo_benchmarks.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import common  # noqa: E402

try:
    import yaml
except ImportError as exc:  # noqa: BLE001
    print(f"需要 PyYAML（Foundry 环境里自带）：{exc}", file=sys.stderr)
    raise

VAL_DIR = common.FOUNDRY_ROOT / "models" / "rfd3" / "configs" / "datasets" / "val"
VAL_EXAMPLES = VAL_DIR / "val_examples"
TRAIN_DIR = common.FOUNDRY_ROOT / "models" / "rfd3" / "configs" / "datasets" / "train" / "pdb"

# 遗留字段名（dialect 1）-> 当前字段名（dialect 2）
# 配置里用的是左侧这套；`rfd3 design` 的输入 JSON 要用右侧这套。
DIALECT2_NAMES = {
    "atom_level_hotspots": "select_hotspots",
    "hbond_donors": "select_hbond_donor",
    "hbond_acceptors": "select_hbond_acceptor",
}

# 论文 §3.1 的五个靶点（PDB ID 由训练排除清单反推，见 main 里的 holdout 提取）
PAPER_PPI_TARGETS = {
    "pdl1": "5o45",
    "insulinr": "4zxb",
    "tie2": "2gy5",
    "il2ra": "1z92",
    "il7ra": "3di3",  # 只出现在训练排除清单里，benchmark 定义未随仓库发布
}


def flatten_selection(value) -> str:
    """把 {"CG,CZ": 1} 或 "CG,CZ" 统一成 "CG,CZ"。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        atoms: list[str] = []
        for k in value:
            atoms.extend(str(k).split(","))
        return ",".join(a.strip() for a in atoms if a.strip())
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def to_dialect2(entry: dict, global_args: dict | None = None) -> dict:
    """把 val_examples 里的条目翻译成 dialect-2 输入规格。

    顺序：先铺 global_args（默认值），再用条目自己的字段覆盖。
    """
    out: dict = {"dialect": 2}
    for key, value in (global_args or {}).items():
        if value is not None:
            out[key] = value
    for key in ("input", "contig", "length", "ligand", "unindex",
                "infer_ori_strategy", "is_non_loopy", "ori_token",
                "redesign_motif_sidechains", "partial_t"):
        if key in entry and entry[key] is not None:
            out[key] = entry[key]
    for legacy, new in DIALECT2_NAMES.items():
        if legacy in entry and entry[legacy]:
            out[new] = {res: flatten_selection(sel)
                        for res, sel in entry[legacy].items()}
    return out


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _default_names(node) -> list[str]:
    """把 `defaults:` 里的一项（字符串或单键 dict）解析成配置文件名列表。"""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [str(v) for v in node.values() if isinstance(v, str)]
    return []


def merge_with_defaults(path: Path, seen: set[Path] | None = None) -> dict:
    """把 `defaults:` 里的父配置（同目录下的 yaml）合并进来。

    Hydra 的 defaults 是相对当前配置文件所在目录解析的，所以 val_examples/ 下的
    配置会去找 val_examples/ 里的父配置。
    """
    seen = seen or set()
    if path in seen:
        return {}
    seen.add(path)
    data = load_yaml(path)
    merged: dict = {}
    for default in data.get("defaults", []) or []:
        for name in _default_names(default):
            if name == "_self_":
                continue
            parent = path.parent / f"{name}.yaml"
            if parent.exists():
                merged.update(merge_with_defaults(parent, seen))
    for key, value in data.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def scan_val_catalog() -> list[dict]:
    """扫描 val/ 下所有配置，列出它们各自需要哪个数据文件。"""
    rows: list[dict] = []
    for path in sorted(VAL_DIR.glob("*.yaml")):
        data = load_yaml(path)
        dataset = data.get("dataset", {}) or {}
        raw = dataset.get("data")
        if isinstance(raw, str):
            data_path = re.sub(r"\$\{[^}]+\}", "<design_benchmark_data_dir>", raw)
        elif raw is None:
            data_path = "(继承/内联)"
        else:
            data_path = "(内联 dict)"
        rows.append({
            "config": f"val/{path.name}",
            "name": dataset.get("name", ""),
            "data": data_path,
            "subset_to_keys": dataset.get("subset_to_keys") or "",
            "eval_every_n": dataset.get("eval_every_n", ""),
        })
    for path in sorted(VAL_EXAMPLES.glob("*.yaml")):
        rows.append({"config": f"val/val_examples/{path.name}",
                     "name": "（内联 PPI 定义）",
                     "data": "内联在 YAML 里",
                     "subset_to_keys": "",
                     "eval_every_n": ""})
    return rows


def extract_holdout_ids() -> dict:
    """从训练过滤器反推 holdout 的 PDB ID。"""
    result: dict = {"excluded_pdb_ids": [], "cutoff": None, "source": []}
    for path in sorted(TRAIN_DIR.glob("rfd3_train_*.yaml")):
        text = path.read_text(encoding="utf-8")
        result["source"].append(str(path.relative_to(common.FOUNDRY_ROOT)))
        for m in re.finditer(r"pdb_id not in \[([^\]]+)\]", text):
            ids = re.findall(r'"([0-9a-zA-Z]{4})"', m.group(1))
            for i in ids:
                if i.lower() not in result["excluded_pdb_ids"]:
                    result["excluded_pdb_ids"].append(i.lower())
        m = re.search(r"deposition_date < '([^']+)'", text)
        if m:
            result["cutoff"] = m.group(1)
    return result


def write_catalog(rows: list[dict], holdout: dict) -> Path:
    path = common.INPUTS_DIR / "targets" / "repo_benchmark_catalog.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 仓库自带的 benchmark 定义清单", "",
             "由 `02_extract_repo_benchmarks.py` 自动生成。", "",
             "## 验证集配置 -> 需要的数据文件", "",
             "| 配置文件 | benchmark 名 | 数据文件 | 只取这些 key | 评估频率 |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['config']}` | {r['name']} | `{r['data']}` | "
                     f"{r['subset_to_keys']} | {r['eval_every_n']} |")
    lines += ["", "## 从训练过滤器反推出的 holdout 测试集", "",
              f"- 训练时间截断：`deposition_date < {holdout['cutoff']}`", "",
              "- 明确排除的 PDB ID（= 论文的 held-out 靶点）：", ""]
    for pid in holdout["excluded_pdb_ids"]:
        guess = [f"{k} ({v})" for k, v in PAPER_PPI_TARGETS.items()
                 if v.lower() == pid]
        label = ", ".join(guess) if guess else "DNA 结合物靶点"
        lines.append(f"  - `{pid}` — {label}")
    lines += ["", f"- 来源：{'、'.join(holdout['source'])}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="导出仓库自带的 benchmark 定义")
    ap.add_argument("--ppi-config",
                    default="val_examples/bcov_ppi_easy_medium_with_ori.yaml",
                    help="PPI benchmark 的基准配置文件（相对 val/）")
    args = ap.parse_args()

    common.ensure_dirs()
    specs_dir = common.INPUTS_DIR / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 清单 ---
    rows = scan_val_catalog()
    holdout = extract_holdout_ids()
    catalog = write_catalog(rows, holdout)
    print(f"[catalog] {catalog}  （{len(rows)} 个验证集配置）")
    print(f"[catalog] holdout PDB ID: {holdout['excluded_pdb_ids']}")
    print(f"[catalog] 训练时间截断: {holdout['cutoff']}")

    # --- 2. PPI benchmark -> dialect-2 输入 JSON ---
    base_path = VAL_DIR / args.ppi_config
    if not base_path.exists():
        print(f"[warn] 找不到 {base_path}")
    else:
        for name, out_name in (
            (args.ppi_config, "exp2_ppi_paper.json"),
            ("val_examples/bpem_ori_hb.yaml", "exp2_ppi_paper_hbond.json"),
        ):
            src = VAL_DIR / name
            if not src.exists():
                continue
            raw = merge_with_defaults(src)
            global_args = raw.get("global_args", {}) or {}
            spec: dict = {}
            for key, entry in raw.items():
                if key in ("global_args", "defaults") or not isinstance(entry, dict):
                    continue
                if "input" not in entry and "contig" not in entry:
                    continue
                dialect2 = to_dialect2(entry, global_args)
                # 原来的 input 指向作者的内网路径（/projects/ml/...）。
                # 换成 01_prepare_inputs.py 会生成的本地路径，两个脚本才能串起来。
                dialect2["input"] = str(
                    common.INPUTS_DIR / "ppi" / f"{key}.pdb")
                spec[key] = dialect2
            if spec:
                target = specs_dir / out_name
                common.write_spec(target, spec)
                paper = [k for k in spec if k in PAPER_PPI_TARGETS]
                print(f"          其中属于论文 §3.1 的五个靶点: {paper or '（无）'}")

    # --- 3. holdout 靶点的 PDB ID ---
    holdout_file = common.INPUTS_DIR / "targets" / "holdout_pdb_ids.json"
    holdout_file.write_text(json.dumps({
        **holdout,
        "paper_ppi_targets": PAPER_PPI_TARGETS,
        "_说明": ("这些 PDB ID 被显式排除在训练之外，因此是论文的 held-out 评测靶点。"
                  "DNA 靶点（7rte/7n5u/7m5w）与 PPI 靶点共用同一份排除清单。"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[holdout] {holdout_file}")

    print("\n下一步：")
    print("  python 01_prepare_inputs.py --only ppi    # 下载并整理 PPI 靶点结构")
    print("  详细的数据集获取说明见 DATASETS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
