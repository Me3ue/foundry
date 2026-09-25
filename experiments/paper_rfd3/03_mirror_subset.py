#!/usr/bin/env python3
"""从本地 PDB 镜像里抽子集 —— 不用把 100 GB 全同步到服务器。

背景
----
论文 §3 的全部 in silico benchmark 加起来只需要 **几十个结构文件**，
而完整的 RCSB PDB 镜像约 100 GB（包含全库近 20 万个结构）。
本脚本按需从本地镜像里把用得上的那几个抽出来，保持 RCSB 的目录约定：

    {mirror_root}/{pdb_id 的中间两位}/{pdb_id}.cif.gz
                     例：5o45 -> o4/5o45.cif.gz

抽出来的小镜像直接 rsync 到服务器、把 PDB_MIRROR_PATH 指过去即可，
AtomWorks / RFD3 会像用完整镜像一样读它。

子命令
------
    audit      清点本地镜像的规模和目录布局
    paper      自动推导"论文实验需要的全部结构"并抽取
    ids        从文件/命令行给的 PDB ID 列表抽取
    manifest   从 evaluation_manifest.csv（或任意带 pdb_id 列的 csv/parquet）抽取

常用例子
--------
    # 1) 先看看本地镜像长什么样、多大
    python 03_mirror_subset.py audit --mirror /media/zzj/Data/pdb_mirror

    # 2) 抽论文需要的（约几十 MB），直接写成服务器可用的小镜像
    python 03_mirror_subset.py paper \
        --mirror /media/zzj/Data/pdb_mirror \
        --out    out/pdb_mirror_subset

    # 3) 本地缺的从 RCSB 补上
    python 03_mirror_subset.py paper --out out/pdb_mirror_subset --download-missing

    # 4) 拷到服务器（只传这一个目录）
    rsync -avP out/pdb_mirror_subset/  server:/data/$USER/pdb_mirror/

    # 5) 服务器上指过去
    export PDB_MIRROR_PATH=/data/$USER/pdb_mirror
    # 或跑 hydra 时： paths.data.pdb_data_dir=/data/$USER/pdb_mirror
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import common  # noqa: E402

DEFAULT_MIRROR = str(common.PDB_MIRROR)
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
# 允许的扩展名，按优先级排列（.cif.gz 是镜像标准，其余是兼容情况）
EXTENSIONS = common.STRUCT_EXTENSIONS
mirror_relpath = common.mirror_relpath
find_in_mirror = common.find_in_mirror

# ---------------------------------------------------------------------------
# 论文实验真正需要的结构（PDB ID -> 用途）。
# 这是"最小可行集"，全量镜像里的其它 19 万个结构都跟论文 §3 无关。
# ---------------------------------------------------------------------------
PAPER_PDB_IDS: dict[str, str] = {
    # §3.1 蛋白结合蛋白（五个 held-out 靶点 + 仓库 benchmark 里的额外三个）
    "5o45": "PPI 靶点 PD-L1",
    "4zxb": "PPI 靶点 InsulinR",
    "2gy5": "PPI 靶点 Tie2",
    "1z92": "PPI 靶点 IL-2Ra",
    "3di3": "PPI 靶点 IL-7Ra（benchmark 定义未随仓库发布）",
    "2x1w": "bcov benchmark 额外靶点 VEGFR",
    "1yjd": "bcov benchmark 额外靶点 CD28",
    "1lqs": "bcov benchmark 额外靶点 IL-10Rb",
    # §3.2 DNA 结合蛋白（三个训练集外靶点）
    "7rte": "DNA 靶点 1",
    "7n5u": "DNA 靶点 2",
    "7m5w": "DNA 靶点 3",
    # §3.3 小分子（配体来源结构；IAI/OQO 仓库自带，FAD/SAM 需要自备任意含它的结构）
    "7v11": "OQO 配体来源（仓库自带裁剪版）",
    "1fad": "FAD 配体来源（可换成任意含 FAD 的结构）",
    "1sam": "SAM 配体来源（可换成任意含 SAM 的结构）",
    # §4 实验的计算部分
    "1euv": "Ulp-1，半胱氨酸水解酶 motif 来源",
    "1ctt": "M0097，AME 示例活性位点",
    # RFD3 官方 tutorial 用到的
    "1bna": "dsDNA 教程示例",
    "5o4d": "ssDNA 教程示例",
    "1q75": "RNA 教程示例",
    "2r5z": "蛋白-DNA 复合物教程示例",
    "1mg5": "M0255 酶设计教程示例",
}


@dataclass
class Result:
    pdb_id: str
    purpose: str = ""
    status: str = "found"
    source: str = ""
    dest: str = ""
    size_bytes: int = 0
    sha256: str = ""

    @property
    def size_human(self) -> str:
        return human(self.size_bytes)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def human_id(pdb_id: str) -> str:
    return pdb_id.lower()


def sha256_of(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            if limit and fh.tell() > limit:
                break
    return h.hexdigest()


def copy_or_link(src: Path, dest: Path, link: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if link:
        try:
            os.link(src, dest)
            return
        except OSError:
            pass  # 跨盘 / 不支持硬链接时退回复制
    shutil.copy2(src, dest)


def download(pdb_id: str, dest: Path) -> bool:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = RCSB_URL.format(pdb_id=human_id(pdb_id))
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    下载失败 {url}: {exc}")
        dest.unlink(missing_ok=True)
        return False


# --------------------------------------------------------------- 各子命令 ---


def cmd_audit(args) -> int:
    mirror = Path(args.mirror)
    if not mirror.exists():
        print(f"镜像不存在：{mirror}", file=sys.stderr)
        return 1
    print(f"镜像根目录: {mirror}")

    dirs = [d for d in mirror.iterdir() if d.is_dir()]
    print(f"  一级子目录: {len(dirs)} 个")
    if dirs:
        sample = ", ".join(sorted(d.name for d in dirs)[:12])
        print(f"  例如: {sample}{' ...' if len(dirs) > 12 else ''}")

    total = files = 0
    for path in mirror.rglob("*"):
        if path.is_file():
            files += 1
            try:
                total += path.stat().st_size
            except OSError:
                pass
    print(f"  文件数: {files}")
    print(f"  总体积: {human(total)}")
    # 抽一份论文子集大概能省多少
    est = len(PAPER_PDB_IDS) * 4 * 1024 * 1024
    if total > est:
        print(f"\n论文 §3 需要的结构数: {len(PAPER_PDB_IDS)} 个"
              f"（粗估 {human(est)}，只占镜像的 {est / total * 100:.2f}%）")
    else:
        print(f"\n论文 §3 需要的结构数: {len(PAPER_PDB_IDS)} 个（粗估 {human(est)}）")
    return 0


def _collect_ids(args) -> dict[str, str]:
    """按子命令收集 {pdb_id: 用途}。"""
    if args.cmd == "paper":
        ids = dict(PAPER_PDB_IDS)
        # 顺便把仓库 benchmark 定义里出现的额外靶点也带上
        holdout = common.load_json(common.INPUTS_DIR / "targets" / "holdout_pdb_ids.json")
        if holdout:
            for pid in holdout.get("excluded_pdb_ids", []):
                ids.setdefault(pid, "训练排除清单（held-out）")
        return ids

    if args.cmd == "ids":
        ids: dict[str, str] = {}
        for pid in args.pdb_ids or []:
            ids[pid.lower()] = "命令行指定"
        for f in args.ids_file or []:
            for line in Path(f).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids[line.split()[0].lower()] = "清单文件"
        return ids

    if args.cmd == "manifest":
        path = Path(args.from_manifest)
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        ids = {}
        if path.suffix == ".csv":
            with path.open() as fh:
                for row in csv.DictReader(fh):
                    pid = (row.get("pdb_id") or "").strip().lower()
                    if pid:
                        ids[pid] = row.get("evaluation_status", "") or "manifest"
        else:
            import pandas as pd  # 只为 parquet 才需要

            df = pd.read_parquet(path, columns=["pdb_id"])
            for pid in df["pdb_id"].astype(str).str.lower().unique():
                ids[pid] = "manifest"
        return ids

    raise SystemExit(f"未知子命令 {args.cmd}")


def run_extract(args, ids: dict[str, str]) -> int:
    mirror = Path(args.mirror)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"镜像: {mirror}")
    print(f"输出: {out}")
    print(f"待抽取: {len(ids)} 个结构\n")

    results: list[Result] = []
    for pdb_id, purpose in sorted(ids.items()):
        res = Result(pdb_id=pdb_id, purpose=purpose)
        src = find_in_mirror(mirror, pdb_id) if mirror.exists() else None

        if src is None and args.download_missing:
            ext = ".cif.gz"
            dest = out / mirror_relpath(pdb_id, ext)
            if download(pdb_id, dest):
                res.status, res.source = "downloaded", "RCSB"
                res.dest = str(dest)
                res.size_bytes = dest.stat().st_size
                res.sha256 = sha256_of(dest)
                print(f"  ↓ {pdb_id}  从 RCSB 下载  {res.size_human}")
            else:
                res.status = "missing"
                print(f"  ! {pdb_id}  本地没有，下载也失败")
        elif src is None:
            res.status = "missing"
            print(f"  ! {pdb_id}  本地镜像里没找到")
        else:
            ext = "".join(src.suffixes[-2:]) if src.name.endswith(".gz") else src.suffix
            dest = out / mirror_relpath(pdb_id, ext)
            copy_or_link(src, dest, args.link)
            res.status = "linked" if args.link else "copied"
            res.source = str(src)
            res.dest = str(dest)
            res.size_bytes = dest.stat().st_size
            if not args.fast:
                res.sha256 = sha256_of(dest)
            print(f"  ✓ {pdb_id}  {res.size_human:>10}  {res.status}")
        results.append(res)

    # ---- 汇总 ----
    ok = [r for r in results if r.status != "missing"]
    missing = [r for r in results if r.status == "missing"]
    total = sum(r.size_bytes for r in ok)

    print(f"\n成功 {len(ok)} / {len(results)}，合计 {human(total)}")
    if missing:
        print(f"缺失 {len(missing)} 个: {', '.join(r.pdb_id for r in missing)}")
        print("  可以在服务器上直接补：")
        print(f"  python 03_mirror_subset.py paper --mirror {mirror} "
              f"--out {out} --download-missing")
        if not args.keep_missing:
            pass

    # ---- 清单 ----
    man = out / "mirror_subset_manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "purpose", "status",
                                           "size_bytes", "sha256", "source", "dest"])
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in w.fieldnames})
    print(f"清单: {man}")

    readme = out / "README.txt"
    readme.write_text(
        "这是从完整 PDB 镜像里抽出来的最小子集，供 RFdiffusion3 论文实验使用。\n"
        "目录结构遵循 RCSB 约定：{PDB ID 的中间两位}/{PDB ID}.cif.gz\n\n"
        "在服务器上把路径指过来即可：\n"
        "    export PDB_MIRROR_PATH=<这个目录的绝对路径>\n"
        "    # 或在 hydra 里覆盖： paths.data.pdb_data_dir=<这个目录>\n\n"
        f"共 {len(ok)} 个结构，{human(total)}。明细见 mirror_subset_manifest.csv。\n",
        encoding="utf-8")
    print(f"说明: {readme}")
    return 0 if not missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从本地 PDB 镜像抽子集（避免同步 100 GB）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["audit", "paper", "ids", "manifest"])
    ap.add_argument("--mirror", default=DEFAULT_MIRROR,
                    help=f"本地镜像根目录（默认 {DEFAULT_MIRROR}）")
    ap.add_argument("--out", default=str(common.INPUTS_DIR.parent / "pdb_mirror_subset"),
                    help="子集输出目录")
    ap.add_argument("--pdb-ids", nargs="*", default=None, help="cmd=ids 时的 ID 列表")
    ap.add_argument("--ids-file", nargs="*", default=None,
                    help="cmd=ids 时的清单文件（一行一个 ID）")
    ap.add_argument("--from-manifest", default=str(common.FOUNDRY_ROOT / "evaluation_manifest.csv"),
                    help="cmd=manifest 时的 csv/parquet 路径")
    ap.add_argument("--link", action="store_true",
                    help="用硬链接代替复制（同一块盘上几乎不占额外空间）")
    ap.add_argument("--fast", action="store_true", help="跳过 sha256 计算")
    ap.add_argument("--download-missing", action="store_true",
                    help="本地镜像里没有的，从 RCSB 逐文件下载")
    ap.add_argument("--keep-missing", action="store_true", help="（占位，无实际作用）")
    args = ap.parse_args()

    if args.cmd == "audit":
        return cmd_audit(args)

    ids = _collect_ids(args)
    if not ids:
        print("没有要抽取的 ID。", file=sys.stderr)
        return 1
    return run_extract(args, ids)


if __name__ == "__main__":
    raise SystemExit(main())
