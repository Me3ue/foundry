#!/usr/bin/env python3
"""为 RFD3 的 NMF / LoRA 微调裁一个「训练子集」——不必把 82 GB 全库搬到服务器。

为什么可以裁
------------
RFD3 训练集的两个 parquet 只是**元数据索引**：

    interfaces_df.parquet   687 万行 / 6,079,530 行通过训练 filters
    pn_units_df.parquet     381 万行 / 3,526,663 行通过训练 filters

而 `InterfacesDFParser` / `PNUnitsDFParser` 会**完全忽略 parquet 里的 `path` 列**，
改用模板自己拼路径（见 atomworks/ml/datasets/parsers/default_metadata_row_parsers.py）：

    {base_dir}/{pdb_id[1:3]}/{pdb_id}{file_extension}
    例：5o45 -> {base_dir}/o4/5o45.cif.gz

于是只要**同时裁两样东西**、并保持它们一致：

    ① 裁 parquet 的行              -> 子集 parquet
    ② 裁镜像里的 cif.gz            -> 子集镜像（保持分卷布局）

训练就能照常跑。`calculate_weights_for_pdb_dataset_df` 在传入的 df 内部重新算
cluster size，所以采样权重会在子集内重新归一化 —— baseline 与各个 NMF 变体
用同一子集，横向对比（论文要的正是相对差）依然成立；只有绝对 lDDT 数值
会和完整训练集不同。

子命令
------
    plan      只做统计：训练集有多大、裁到 N 行需要多少 pdb_id、占多少磁盘
    extract   真的裁：导出子集 parquet + 子集镜像 + manifest + env 片段
    verify    校验子集是否自洽（parquet 里每个 pdb_id 在子集镜像里都存在）

常用例子
--------
    # 0) 先看规模（不改任何文件，秒级）
    python 04_training_subset.py plan --parquet-dir /media/zzj/Data/pdb_metadata_latest

    # 1) 裁一个 2048 行 / 子集的微调集（约几百 MB）
    python 04_training_subset.py extract \
        --parquet-dir /media/zzj/Data/pdb_metadata_latest \
        --mirror      /media/zzj/Data/pdb_mirror \
        --out         out/train_subset_2k \
        --rows 2048

    # 2) 校验
    python 04_training_subset.py verify --out out/train_subset_2k

    # 3) 拷到服务器（只传一个目录）
    rsync -avP out/train_subset_2k/ server:/data/$USER/rfd3_train_subset/

    # 4) 服务器上跑 NMF sweep 时指过去
    #    PARQUET=/data/$USER/rfd3_train_subset/metadata \
    #    PDB_MIRROR=/data/$USER/rfd3_train_subset/mirror \
    #    N_EXAMPLES=128 \
    #    bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh

重要：extract 默认会把**验证集（holdout）用到的 pdb_id 一起打进去**，
否则训练时 validation_loop 会因为找不到结构而报错。来源是
experiments/paper_rfd3/out/inputs/... 或仓库根的 evaluation_manifest.csv。
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 与 train/pdb/rfd3_train_interface.yaml + rfd3_train_pn_unit.yaml 保持一致的
# 训练 filters。改这里等于改训练口径，务必与 yaml 同步。
# ---------------------------------------------------------------------------
TRAIN_CUTOFF = "2024-12-16"
MAX_RESOLUTION = 9.0
MAX_POLYMER_PN_UNITS = 300
# rfd3_train_interface.yaml / rfd3_train_pn_unit.yaml 里的两条 pdb_id 排除
EXCLUDED_PDB_IDS = ["7rte", "7m5w", "7n5u", "3di3", "5o45", "1z92", "2gy5", "4zxb"]

PARQUET_FILES = ("interfaces_df.parquet", "pn_units_df.parquet")

# 子集 parquet 要保留哪些列 —— 直接对应两个训练配置的 `columns_to_load`
# （train/pdb/af3_train_interface.yaml 与 af3_train_pn_unit.yaml）。
# 全列有 60 列、687 万行，整表读入会吃掉几十 GB 内存（本机实测被 OOM kill）；
# 按官方 columns_to_load 裁列既省内存，也和训练时加载的列完全一致。
INTERFACE_COLUMNS = [
    "example_id", "pdb_id", "assembly_id", "deposition_date", "resolution",
    "num_polymer_pn_units", "method", "cluster", "n_prot", "n_nuc", "n_ligand",
    "n_peptide",
    "pn_unit_1_iid", "pn_unit_2_iid",
    "pn_unit_1_non_polymer_res_names", "pn_unit_2_non_polymer_res_names",
    "is_inter_molecule", "all_pn_unit_iids_after_processing", "involves_loi",
    "path",  # 解析器用模板拼路径、其实用不到，留着方便调试
]
PN_UNIT_COLUMNS = [
    "example_id", "pdb_id", "assembly_id", "deposition_date", "resolution",
    "num_polymer_pn_units", "method", "cluster", "n_prot", "n_nuc", "n_ligand",
    "n_peptide", "total_num_atoms_in_unprocessed_assembly",
    "q_pn_unit_iid", "q_pn_unit_non_polymer_res_names",
    "all_pn_unit_iids_after_processing", "q_pn_unit_is_loi",
    "path",
]
COLUMNS_BY_SPLIT = {"interfaces": INTERFACE_COLUMNS, "pn_units": PN_UNIT_COLUMNS}

# 训练 filters 里用到的列（别名：不同源可能是 _x / _q_ 前缀，见下）
COMMON_COLS = ["pdb_id", "deposition_date", "resolution", "num_polymer_pn_units", "cluster"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT_MANIFEST = REPO_ROOT / "evaluation_manifest.csv"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def import_pandas():
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit(
            "需要 pandas / pyarrow。请改用装了它们的 Python 运行，例如：\n"
            "  <conda-env>/bin/python 04_training_subset.py ...\n"
            "（本机是 /home/zzj/anaconda3/envs/rc/bin/python）"
        )
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        raise SystemExit("需要 pyarrow：pip install pyarrow")
    return pd


@functools.lru_cache(maxsize=1)
def af3_excluded_regex() -> str:
    """从 atomworks 取 AF3 排除配体正则；取不到就退化到空串（不禁用该 filter）。

    atomworks 在导入时会往 stderr 打两行环境变量提示，这里顺手吞掉。
    """
    import contextlib
    import io
    try:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            from atomworks.constants import AF3_EXCLUDED_LIGANDS_REGEX as r
        return r
    except Exception:  # noqa: BLE001
        print("[warn] 无法从 atomworks 导入 AF3_EXCLUDED_LIGANDS_REGEX，"
              "本次不做配体排除过滤（会让子集略微偏宽）", file=sys.stderr)
        return ""


def mirror_relpath(pdb_id: str, ext: str = ".cif.gz") -> Path:
    p = pdb_id.lower()
    return Path(p[1:3]) / f"{p}{ext}"


def find_in_mirror(mirror: Path, pdb_id: str) -> Path | None:
    p = pdb_id.lower()
    for ext in (".cif.gz", ".cif", ".pdb.gz", ".pdb"):
        cand = mirror / mirror_relpath(p, ext)
        if cand.exists():
            return cand
    return None


# --------------------------------------------------------------- 过滤逻辑 ---


def load_split(pd, parquet_dir: Path, name: str, columns: list[str] | None = None):
    path = parquet_dir / name
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    return pd.read_parquet(path, columns=columns)


def filter_interface(pd, df, exclude_ligands: bool = True):
    """对应 train/pdb/rfd3_train_interface.yaml。"""
    import pandas as pd_mod
    dates = pd_mod.to_datetime(df["deposition_date"], errors="coerce")
    mask = (
        ~df["pdb_id"].astype(str).str.lower().isin(EXCLUDED_PDB_IDS)
        & (dates < pd_mod.Timestamp(TRAIN_CUTOFF))
        & (df["resolution"] < MAX_RESOLUTION)
        & (df["num_polymer_pn_units"] <= MAX_POLYMER_PN_UNITS)
        & df["cluster"].notna()
    )
    if "is_inter_molecule" in df.columns:
        mask &= df["is_inter_molecule"].astype("boolean").fillna(False)
    regex = af3_excluded_regex() if exclude_ligands else ""
    if regex:
        for col in ("pn_unit_1_non_polymer_res_names", "pn_unit_2_non_polymer_res_names"):
            if col in df.columns:
                hit = (df[col].notna()
                       & df[col].astype(str).str.contains(regex, regex=True))
                mask &= ~hit
    return df.loc[mask]


def filter_pn_unit(pd, df, exclude_ligands: bool = True):
    """对应 train/pdb/rfd3_train_pn_unit.yaml。"""
    import pandas as pd_mod
    dates = pd_mod.to_datetime(df["deposition_date"], errors="coerce")
    mask = (
        ~df["pdb_id"].astype(str).str.lower().isin(EXCLUDED_PDB_IDS)
        & (dates < pd_mod.Timestamp(TRAIN_CUTOFF))
        & (df["resolution"] < MAX_RESOLUTION)
        & (df["num_polymer_pn_units"] <= MAX_POLYMER_PN_UNITS)
        & df["cluster"].notna()
    )
    regex = af3_excluded_regex() if exclude_ligands else ""
    col = "q_pn_unit_non_polymer_res_names"
    if regex and col in df.columns:
        hit = df[col].notna() & df[col].astype(str).str.contains(regex, regex=True)
        mask &= ~hit
    return df.loc[mask]


def estimate_cif_size(mirror: Path, per_dir: int = 25, max_dirs: int = 40,
                      seed: int = 0) -> float:
    """从镜像里抽样估平均 cif.gz 大小（字节）。

    两个必须的讲究（本机实测踩过）：
      1. **跨一级子目录均匀取样**：RCSB 分卷目录大小极不均匀（`00/` 等几乎空，
         多数目录上百个文件），只盯一头会低估近一半；
      2. **目录内随机取样**，不能取 `iterdir()` 的前 N 个 —— 那是 inode 顺序，
         恰好偏向小文件（实测低估 40%）。
    """
    dirs = sorted(p for p in mirror.iterdir() if p.is_dir())
    if not dirs:
        return 337 * 1024.0
    rng = random.Random(seed)
    step = max(1, len(dirs) // max_dirs)
    sizes: list[int] = []
    for d in dirs[::step]:
        try:
            entries = [f for f in d.iterdir() if f.is_file()]
        except OSError:
            continue
        if not entries:
            continue
        k = min(per_dir, len(entries))
        for f in rng.sample(entries, k):
            try:
                sizes.append(f.stat().st_size)
            except OSError:
                continue
        if len(sizes) >= max_dirs * per_dir:
            break
    if not sizes:
        return 337 * 1024.0
    return sum(sizes) / len(sizes)


def load_holdout_ids(manifest: Path | None) -> set[str]:
    """验证集（pdb_holdout.yaml）用到的 pdb_id —— 子集必须包含它们。"""
    if manifest is None or not Path(manifest).exists():
        return set()
    ids: set[str] = set()
    with Path(manifest).open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("pdb_id") or "").strip().lower()
            if pid:
                ids.add(pid)
    return ids


# ---------------------------------------------------------------- 抽样策略 ---


def sample_rows(pd, df, rows: int, per_cluster: int, seed: int):
    """先按 cluster 限额去冗余，再从候选池随机抽到目标行数。

    限额的意义：同一 cluster 里的行近同源，小数据集里若被同一簇灌满，
    分布就废了。默认 per_cluster=1（每簇最多 1 行），抽不够时自动放宽。

    实现上只在"位置索引"上循环、最后才 `iloc` 复制选中的那几行 ——
    对几百行的子集没必要把 563 万行的表整体 shuffle 一遍（会多占几 GB）。
    """
    import numpy as np

    n = len(df)
    if rows <= 0 or rows >= n:
        return df

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    clusters = df["cluster"].to_numpy()

    counts: dict = {}
    chosen: list[int] = []
    for pos in order:
        c = clusters[pos]
        if counts.get(c, 0) < per_cluster:
            counts[c] = counts.get(c, 0) + 1
            chosen.append(int(pos))
            if len(chosen) >= rows:
                break

    if len(chosen) < rows:
        # 限额太紧，剩下的从没选中的行里随机补
        taken = np.zeros(n, dtype=bool)
        taken[chosen] = True
        pool = np.flatnonzero(~taken)
        need = min(rows - len(chosen), len(pool))
        if need > 0:
            chosen.extend(int(i) for i in rng.choice(pool, size=need, replace=False))

    return df.iloc[sorted(chosen)].copy()


# --------------------------------------------------------------- 子命令 ---


def cmd_plan(args) -> int:
    pd = import_pandas()
    parquet_dir = Path(args.parquet_dir)
    mirror = Path(args.mirror)

    print(f"parquet 目录: {parquet_dir}")
    print(f"镜像目录   : {mirror}  {'(存在)' if mirror.exists() else '(不存在)'}")
    print()

    # 只读过滤+抽样需要的列：全列 parquet 有 60 列、1.9 GB，没必要全拉
    keep = {"interfaces": ["pdb_id", "deposition_date", "resolution",
                           "num_polymer_pn_units", "cluster", "is_inter_molecule",
                           "pn_unit_1_non_polymer_res_names",
                           "pn_unit_2_non_polymer_res_names"],
            "pn_units": ["pdb_id", "deposition_date", "resolution",
                         "num_polymer_pn_units", "cluster",
                         "q_pn_unit_non_polymer_res_names"]}

    filtered_frames: dict[str, "object"] = {}
    n_filtered: dict[str, int] = {}
    n_all: dict[str, int] = {}

    for name in PARQUET_FILES:
        path = parquet_dir / name
        if not path.exists():
            print(f"[skip] {name} 不存在")
            continue
        split = "interfaces" if "interfaces" in name else "pn_units"
        print(f"读取 {name} ...")
        import pyarrow.parquet as pq_mod
        available = set(pq_mod.ParquetFile(path).schema_arrow.names)
        cols = [c for c in keep[split] if c in available]
        df = pd.read_parquet(path, columns=cols or keep[split])
        n_all[name] = len(df)
        filtered = (filter_interface(pd, df) if split == "interfaces"
                    else filter_pn_unit(pd, df))
        del df
        filtered_frames[split] = filtered
        n_filtered[name] = len(filtered)
        print(f"  总行数        : {n_all[name]:,}")
        print(f"  训练 filters 后: {len(filtered):,}")
        print(f"  唯一 pdb_id    : {filtered['pdb_id'].nunique():,}")
        print(f"  唯一 cluster   : {filtered['cluster'].nunique():,}")
        print()

    if not mirror.exists():
        print("[warn] 镜像不存在，无法估算体积")
        return 0

    avg = estimate_cif_size(mirror)
    print(f"镜像平均单文件: {human(avg)}")
    print()
    print("| 每 split 行数 | unique pdb_id（两 split 并集）| 估 cif 体积 |")
    print("|---|---|---|")
    sizes = (256, 512, 1024, 2048, 4096, 8192, 32768)
    for rows in sizes:
        ids: set[str] = set()
        skipped = False
        for split, filtered in filtered_frames.items():
            if rows >= len(filtered):
                skipped = True
                ids |= {str(x).lower() for x in filtered["pdb_id"].unique()}
                continue
            sub = sample_rows(pd, filtered, rows, args.per_cluster, args.seed)
            ids |= {str(x).lower() for x in sub["pdb_id"].unique()}
        note = "（已含全部）" if skipped else ""
        print(f"| {rows:,} | {len(ids):,}{note} | {human(len(ids) * avg)} |")

    full_ids: set[str] = set()
    for filtered in filtered_frames.values():
        full_ids |= {str(x).lower() for x in filtered["pdb_id"].unique()}
    print(f"| 全量（不裁） | {len(full_ids):,} | "
          f"{human(len(full_ids) * avg)}（≈ 全库 82 GB）|")

    holdout = load_holdout_ids(Path(args.holdout_manifest) if args.holdout_manifest
                               else DEFAULT_HOLDOUT_MANIFEST)
    if holdout:
        files = [f for i in holdout if (f := find_in_mirror(mirror, i))]
        size = sum(f.stat().st_size for f in files)
        print()
        print(f"另外必须带上验证集: {len(holdout)} 个 pdb_id，"
              f"本机镜像里找到 {len(files)} 个，{human(size)}")
    return 0


def cmd_extract(args) -> int:
    pd = import_pandas()
    parquet_dir = Path(args.parquet_dir)
    mirror = Path(args.mirror)
    out = Path(args.out)
    if not mirror.exists():
        raise SystemExit(f"镜像不存在: {mirror}")

    meta_dir = out / "metadata"
    mirror_out = out / "mirror"
    meta_dir.mkdir(parents=True, exist_ok=True)
    mirror_out.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {out}")
    print(f"目标行数: {args.rows if args.rows else '全量'}  / 每 cluster 最多 "
          f"{args.per_cluster} 行  / seed={args.seed}")
    print()

    needed_ids: dict[str, str] = {}
    summary = []
    for name in PARQUET_FILES:
        path = parquet_dir / name
        if not path.exists():
            print(f"[skip] {name} 不存在")
            continue
        split = "interface" if "interfaces" in name else "pn_unit"
        split_key = "interfaces" if "interfaces" in name else "pn_units"
        print(f"[{split}] 读取 {name} ...")
        import pyarrow.parquet as pq_mod
        available = set(pq_mod.ParquetFile(path).schema_arrow.names)
        if args.keep_all_columns:
            use_cols = None
            print("  [--keep-all-columns] 读取全列（内存需求可达几十 GB）")
        else:
            use_cols = [c for c in COLUMNS_BY_SPLIT[split_key] if c in available]
            missing = [c for c in COLUMNS_BY_SPLIT[split_key] if c not in available]
            if missing:
                print(f"  [note] 源 parquet 没有这些列，已跳过: {missing}")
            print(f"  只读取 {len(use_cols)} 列（对应训练配置的 columns_to_load）")
        df = pd.read_parquet(path, columns=use_cols)
        filtered = (filter_interface(pd, df) if split == "interface"
                    else filter_pn_unit(pd, df))
        del df
        n_filt = len(filtered)
        sub = sample_rows(pd, filtered, args.rows, args.per_cluster, args.seed)
        del filtered
        n_ids = sub["pdb_id"].nunique()
        print(f"  训练 filters 后 {n_filt:,} 行 -> 抽 {len(sub):,} 行 / "
              f"{n_ids:,} 个 pdb_id / {sub['cluster'].nunique():,} 个 cluster")

        dest = meta_dir / name
        sub.to_parquet(dest, index=False)
        print(f"  写出 {dest}  ({human(dest.stat().st_size)})")
        for pid in sub["pdb_id"].astype(str).str.lower().unique():
            needed_ids.setdefault(pid, f"train:{split}")
        summary.append({"file": name, "split": split,
                        "rows_filtered": n_filt, "rows_kept": len(sub),
                        "pdb_ids": int(n_ids),
                        "clusters": int(sub["cluster"].nunique())})
        print()

    # 验证集的结构必须一起带上，否则 validation_loop 会崩
    manifest = Path(args.holdout_manifest) if args.holdout_manifest else DEFAULT_HOLDOUT_MANIFEST
    holdout = load_holdout_ids(manifest)
    if holdout:
        for pid in holdout:
            needed_ids.setdefault(pid, "holdout")
        print(f"[holdout] 从 {manifest} 带入 {len(holdout)} 个 pdb_id")
    else:
        print(f"[holdout] 没找到 {manifest}；如果训练配置启用了 pdb_holdout "
              f"验证，请用 --holdout-manifest 指定")

    for pid in args.extra_ids or []:
        needed_ids.setdefault(pid.lower(), "extra")

    # 抽镜像
    print(f"\n[mirror] 需要 {len(needed_ids):,} 个结构的 cif")
    rows = []
    missing = []
    total = 0
    for i, (pid, why) in enumerate(sorted(needed_ids.items()), 1):
        src = find_in_mirror(mirror, pid)
        if src is None:
            missing.append(pid)
            rows.append({"pdb_id": pid, "reason": why, "source": "", "bytes": 0,
                         "status": "missing"})
            continue
        rel = mirror_relpath(pid, src.suffix if not src.name.endswith(".cif.gz") else ".cif.gz")
        dest = mirror_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        if args.link:
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
        else:
            shutil.copy2(src, dest)
        size = dest.stat().st_size
        total += size
        rows.append({"pdb_id": pid, "reason": why, "source": str(src),
                     "bytes": size, "status": "ok"})
        if i % 500 == 0:
            print(f"  ... {i:,}/{len(needed_ids):,}")

    print(f"  已放 {len(rows) - len(missing):,} 个文件，{human(total)}")
    if missing:
        print(f"  [warn] 镜像里缺 {len(missing)} 个: {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")
        print(f"         用 --download-missing 可以从 RCSB 补（需联网）")

    if missing and args.download_missing:
        import urllib.request
        for pid in list(missing):
            dest = mirror_out / mirror_relpath(pid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://files.rcsb.org/download/{pid}.cif.gz"
            try:
                urllib.request.urlretrieve(url, dest)
                total += dest.stat().st_size
                missing.remove(pid)
                print(f"  [下载] {pid}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [下载失败] {pid}: {exc}")

    # manifest
    man = out / "subset_manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "reason", "source", "bytes", "status"])
        w.writeheader()
        w.writerows(rows)

    # env 片段 + 说明
    env = out / "env.subset.sh"
    env.write_text(
        "#!/usr/bin/env bash\n"
        "# 由 04_training_subset.py 生成：把 RFD3 微调指向这个子集。\n"
        "# 用法：  source env.subset.sh  然后照常跑 run_nmf_zkp_pdb_sweep.sh\n"
        f'export PARQUET="{meta_dir.resolve()}"\n'
        f'export PDB_MIRROR="{mirror_out.resolve()}"\n'
        f'export DATA="$PDB_MIRROR"\n'
        f'export N_EXAMPLES="${{N_EXAMPLES:-{min(args.rows or 128, 128)}}}"\n'
        "# CCD 与权重不在这里，仍需单独准备（CCD 约 1.7 GB、ckpt 约 2.7 GB）\n"
        'export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-/media/zzj/Data/ccd_mirror}"\n'
        'export CCD_PATH="$CCD_MIRROR_PATH"\n',
        encoding="utf-8",
    )

    readme = out / "README.txt"
    readme.write_text(
        "RFD3 微调训练子集\n"
        "=================\n\n"
        f"行数   : {args.rows if args.rows else '全量'}\n"
        f"pdb_id : {len(needed_ids):,}（含验证集）\n"
        f"体积   : {human(total)}\n"
        f"seed   : {args.seed}（固定 seed，重跑结果一致）\n\n"
        "目录结构（AtomWorks 按 {base_dir}/{pdb_id[1:3]}/{pdb_id}.cif.gz 查找）：\n"
        "  metadata/interfaces_df.parquet\n"
        "  metadata/pn_units_df.parquet\n"
        "  mirror/{xx}/{pdb_id}.cif.gz\n"
        "  subset_manifest.csv\n"
        "  env.subset.sh\n\n"
        "服务器上跑：\n"
        "  source env.subset.sh\n"
        "  bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh\n\n"
        "注意\n"
        "----\n"
        "1. 子集内 cluster size 会重算，采样权重随之变化。baseline 与各 NMF 变体\n"
        "   必须用同一子集，横向对比才有效；绝对 lDDT 不能与完整训练集结果直接比。\n"
        "2. 验证集（pdb_holdout）用到的结构已一并打入，不要删。\n"
        "3. 每子集统计：\n"
        + "".join(f"   - {s['file']}: 过滤后 {s['rows_filtered']:,} 行 -> 保留 "
                  f"{s['rows_kept']:,} 行 / {s['pdb_ids']:,} pdb_id / "
                  f"{s['clusters']:,} cluster\n" for s in summary),
        encoding="utf-8",
    )

    print(f"\n写出:")
    print(f"  {man}")
    print(f"  {env}")
    print(f"  {readme}")
    print(f"\n总体积: {human(total)}（另需 CCD ~1.7 GB + ckpt ~2.7 GB）")
    if missing:
        print(f"\n仍有 {len(missing)} 个结构缺失，verify 会报错，请先补上。")
        return 1
    return 0


def cmd_verify(args) -> int:
    pd = import_pandas()
    out = Path(args.out)
    meta_dir, mirror = out / "metadata", out / "mirror"
    if not meta_dir.exists() or not mirror.exists():
        raise SystemExit(f"{out} 不像一个子集目录（缺 metadata/ 或 mirror/）")

    print(f"校验 {out}")
    total = 0
    ok = 0
    missing: list[str] = []
    for name in PARQUET_FILES:
        p = meta_dir / name
        if not p.exists():
            print(f"[skip] {p} 不存在")
            continue
        df = pd.read_parquet(p, columns=["pdb_id"])
        ids = sorted({str(x).lower() for x in df["pdb_id"].unique()})
        miss = [i for i in ids if find_in_mirror(mirror, i) is None]
        print(f"  {name}: {len(df):,} 行 / {len(ids):,} pdb_id / "
              f"缺 {len(miss)} 个")
        ok += len(ids) - len(miss)
        total += len(ids)
        missing += miss

    print(f"\n覆盖率: {ok:,}/{total:,} = {ok / max(total, 1) * 100:.2f}%")
    if missing:
        print(f"缺失（前 30）: {sorted(set(missing))[:30]}")
        print("用 extract --download-missing 补，或把这批 pdb_id 从子集里剔除。")
        return 1
    print("✓ 子集自洽，可以拷到服务器上用")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="裁一个 RFD3 微调训练子集（parquet + 镜像）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_args(p):
        p.add_argument("--parquet-dir", default="/media/zzj/Data/pdb_metadata_latest",
                       help="含 interfaces_df.parquet / pn_units_df.parquet 的目录")
        p.add_argument("--mirror", default="/media/zzj/Data/pdb_mirror",
                       help="本地 PDB 镜像根目录")
        p.add_argument("--rows", type=int, default=2048,
                       help="每个训练 parquet 保留多少行（0 = 全量）")
        p.add_argument("--per-cluster", type=int, default=1,
                       help="同一 cluster 最多保留几行（默认 1，防近同源冗余）")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--holdout-manifest", default=None,
                       help="验证集清单 csv（默认仓库根 evaluation_manifest.csv）")

    p_plan = sub.add_parser("plan", help="只统计规模，不改文件")
    common_args(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_ex = sub.add_parser("extract", help="真的裁子集")
    common_args(p_ex)
    p_ex.add_argument("--out", required=True, help="输出目录")
    p_ex.add_argument("--link", action="store_true",
                      help="用硬链接代替复制（同盘时省空间；跨盘会自动退回复制）")
    p_ex.add_argument("--download-missing", action="store_true",
                      help="本地缺的结构从 RCSB 下载补上（需联网）")
    p_ex.add_argument("--extra-ids", nargs="*", default=None,
                      help="额外要塞进子集的 pdb_id")
    p_ex.add_argument("--keep-all-columns", action="store_true",
                      help="子集 parquet 保留全部 60 列（默认只保留训练配置需要的列；"
                           "全列读入可能吃掉几十 GB 内存）")
    p_ex.set_defaults(func=cmd_extract)

    p_v = sub.add_parser("verify", help="校验子集自洽性")
    p_v.add_argument("--out", required=True)
    p_v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
