# 数据集：怎么获取、怎么组织

> 配套文档：`README.md`（实验手册）、`COMMANDS.md`（设备规模 + 标准命令）。
>
> **一个关键结论先说**：论文 §3 所有 in silico benchmark 的**输入定义本来就在这个仓库里**
> （`models/rfd3/configs/datasets/val/`），不需要去翻补充材料。真正缺的只是那些
> 被作者裁剪过的靶点结构文件（需要从 RCSB 重新下载并裁剪，方法见 §2）。
> `02_extract_repo_benchmarks.py` 会自动把它们导出成可运行的输入 JSON。

---

## 0. 数据全景

| # | 数据集 | 用途 | 来源 | 目标位置（脚本里的变量） |
|---|---|---|---|---|
| 1 | PDB mmCIF 镜像 | 训练 / holdout 评测 | RCSB rsync | `$PDB_MIRROR_PATH` → `paths.data.pdb_data_dir` |
| 2 | PDB 元数据 parquet | 训练 / holdout 索引 | AtomWorks 官方归档 | → `paths.data.pdb_parquet_dir` |
| 3 | CCD 镜像 | 配体/修饰残基定义 | PDBe CCD rsync | `$CCD_MIRROR_PATH` |
| 4 | AF2 蒸馏集 | 训练（monomer distillation） | ESM-IF 论文的 AF2 预测 | → `paths.data.monomer_distillation_*` |
| 5 | benchmark 输入 JSON | §3 全部 in silico 评测 | **仓库自带配置**（本页 §1） | → `paths.data.design_benchmark_data_dir` |
| 6 | holdout 靶点结构 | §3.1 / §3.2 评测 | RCSB 下载 + 裁剪 | `out/inputs/ppi`、`out/inputs/dna` |
| 7 | 小分子配体结构 | §3.3 评测 | 仓库自带 / RCSB | `out/inputs/sm` |
| 8 | AME 案例 | §3.4 评测 | RFdiffusion2 仓库 | → `out/inputs/targets/ame_cases.json` |
| 9 | MSA | AF3 / RF3 / Chai 折叠 | MMseqs2 / ColabFold | `$COLABFOLD_*` / `$LOCAL_MSA_DIRS` |
| 10 | PDB100 | Fig. S4b 新颖性（FoldSeek） | FoldSeek | 任意，`foldseek` 建库 |

---

## 1. 论文的 benchmark 定义就在仓库里

`models/rfd3/configs/datasets/val/` 下的每一个 YAML 就是一个 benchmark：

| 配置文件 | benchmark | 需要的数据文件 |
|---|---|---|
| `val/unconditional.yaml` | 无条件单体（§3 开头 / Fig. S1c） | `<design_benchmark_data_dir>/monomer.json` |
| `val/unconditional_deep.yaml` | 更深的无条件评测 | `.../unconditional_deep.json` |
| `val/indexed.yaml` | 有索引的 motif scaffolding | `.../indexed.json` |
| `val/unindexed.yaml` | 无索引原子 motif | `.../unindexed.json` |
| **`val/mcsa_41.yaml`** | **AME 酶 benchmark（§3.4 / Fig. 3d）** | `.../mcsa_41.json` |
| `val/mcsa_41_short_rigid.yaml` | 刚性短 motif 变体（RFD2 对照路线） | `.../mcsa_41_short_rigid_new.json` |
| **`val/dna_binder_design5.yaml`** | **DNA 结合物（§3.2 / Fig. 3b）** | `.../dna_binder.json` |
| `val/dna_binder_long.yaml` | DNA 长变体，`subset_to_keys: [7rte_sequence_only, 7rte_with_structure]` | `.../tests/dna.json` |
| `val/dna_binder_short.yaml` | DNA 短变体，同一组 key | `.../rfd3/tests/test_data/dna.json` |
| **`val/sm_binder_hbonds.yaml`** | **小分子 + 氢键条件（§3.3 / Fig. 3c，Fig. 2d）** | `.../sm_binder_hbonds.json`，`subset_to_keys: [FAD, IAI, OQO, SAM]` |
| `val/sm_binder_hbonds_short.yaml` | 短版本，key 变成 `FAD_1..3 / IAI_1..3` | `.../sm_binder_hbonds_sampled.json` |
| **`val/val_examples/bcov_ppi_easy_medium_with_ori.yaml`** | **PPI 结合蛋白（§3.1 / Fig. 3a）** | **内联在 YAML 里，不需要外部文件** |
| `val/val_examples/bpem_ori_hb.yaml` | PPI + 氢键条件（Fig. 2d） | 内联 |
| `val/val_examples/*_varying_lengths.yaml` / `_spoof_helical_bundle.yaml` | PPI 变长 / 螺旋束变体 | 内联 |
| `val/pdb_holdout.yaml` | 时间外推 holdout | `<pdb_parquet_dir>/interfaces_df.parquet` |
| `val/ppi_inference.yaml` | 通用 PPI 推理入口 | 需在命令行给 `dataset.data=...` |

> `dna_binder_long.yaml` 的两个 key 命名很说明问题：
> **`7rte_with_structure`** 就是论文 Fig. 3b 的 **"+"**（给定 DNA 构象），
> **`7rte_sequence_only`** 就是 **"−"**（只给序列，构象一起采样）。

> **注意**：配置里用的是遗留字段名（`atom_level_hotspots`、`hbond_donors`、
> `hbond_acceptors`），而 `rfd3 design` 的输入 JSON 要用 dialect-2 的
> `select_hotspots` / `select_hbond_donor` / `select_hbond_acceptor`。
> 两者含义相同，只是命名新旧之别（见 `rfd3/inference/legacy_input_parsing.py`）。

把这个转换自动化：

```bash
python 02_extract_repo_benchmarks.py
# 产出：
#   out/inputs/specs/exp2_ppi_paper.json          PPI benchmark（dialect 2，可直接跑）
#   out/inputs/specs/exp2_ppi_paper_hbond.json    加上氢键条件的版本
#   out/inputs/targets/repo_benchmark_catalog.md  上面那张表的自动版本
#   out/inputs/targets/holdout_pdb_ids.json       反推出的 holdout 靶点清单
```

### benchmark 输入 JSON 的格式

放在 `${paths.data.design_benchmark_data_dir}` 下，就是一个 key → 输入规格的字典，
和 `rfd3 design inputs=<某个 json>` 用的是同一套 schema：

```json
{
  "uncond_92mer": { "input": null, "length": "92-92" },

  "pdl1": {
    "dialect": 2,
    "infer_ori_strategy": "hotspots",
    "input": "/abs/path/to/target.pdb",
    "contig": "100-100,/0,B1-115",
    "length": "215-215",
    "select_hotspots": { "B40": "CG,CZ", "B99": "CG,SD", "B107": "CG,CZ" },
    "redesign_motif_sidechains": false
  }
}
```

`val/pdb_holdout.yaml` 里的 `subset_to_keys` 就是用来只跑其中几个 key 的
（例如小分子 benchmark 只取 FAD / IAI / OQO / SAM 四个）。

---

## 2. holdout 测试集：论文到底用了哪些靶点

训练配置 `models/rfd3/configs/datasets/train/pdb/rfd3_train_interface.yaml` 里写得很清楚：

```yaml
- 'pdb_id not in ["7rte", "7m5w", "7n5u"]'                    # DNA 靶点
- 'pdb_id not in ["3di3", "5o45", "1z92", "2gy5", "4zxb"]'    # PPI 靶点
- "deposition_date < '2024-12-16'"                             # 训练时间截断
```

对应关系：

| PDB ID | 靶点 | 类别 | 备注 |
|---|---|---|---|
| `5o45` | **PD-L1** | PPI | benchmark 里叫 `5o45_pdl1.pdb` |
| `4zxb` | **InsulinR** | PPI | `insulin_target.pdb` |
| `2gy5` | **Tie2** | PPI | `tie2_2gy5_official_B.pdb` |
| `1z92` | **IL-2Ra** | PPI | `il2ra_1z92_B.pdb` |
| `3di3` | **IL-7Ra** | PPI | ⚠️ 只在排除清单里，benchmark 定义未随仓库发布 |
| `7rte` / `7n5u` / `7m5w` | **DNA 结合物 ×3** | DNA | 直接从 RCSB 下 |

另外 benchmark 里还有 4 个非论文靶点（`vegfr` / `rbd` / `cd28` / `il10ra`），
可以拿来当额外测试集。

### ⚠️ 靶点需要重新裁剪：benchmark 文件用的是重编号

仓库里 `bcov_ppi_easy_medium_with_ori.yaml` 指向的是作者内部的裁剪文件
（`/projects/ml/aa_design/benchmarks/bcov_af3_ppi_benchmark/...`），这些文件不在仓库里，
而且**编号和链号都被改过**。可以用仓库自带的 tutorial 结构交叉验证出偏移量：

| 靶点 | benchmark 写法 | 仓库 tutorial 写法 | 沉积链 | 编号偏移 |
|---|---|---|---|---|
| PD-L1 | `B1-115`，热点 `B40/B99/B107` | `A17-131`，热点 `A56/A115/A123` | A | **+16** ✅ 已核对 |
| InsulinR | `B1-150`，热点 `B59/B83/B91` | `E6-155`，热点 `E64/E88/E96` | E | **+5** ✅ 已核对 |
| Tie2 | `B1-188` | — | B | 0（待确认） |
| IL-2Ra | `B1-122` | — | B | 0（待确认） |
| IL-7Ra | 无定义 | — | — | 需手工补 |

`01_prepare_inputs.py` 已经把这张表写进 `PPI_PAPER_TARGETS`：它会下载沉积结构、
按 `偏移` 裁出目标链、把热点残基号平移过去，**并用热点的原子名做自校验**
（例如 `CG,SD` 必须落在一个 MET 上）。校验不过就跳过并提示你改 offset。

```bash
python 02_extract_repo_benchmarks.py      # 先导出 benchmark 定义
python 01_prepare_inputs.py --only ppi    # 下载 + 裁剪 + 校验，写入 out/inputs/ppi/
```

跑完检查 `out/inputs/ppi/`：应该有 `pdl1.pdb`、`insulinr.pdb`、`tie2.pdb`、`il2ra.pdb`；
IL-7Ra 的输入结构通过 `out/inputs/ppi_extra_targets.json` 手工补。

### 三个 DNA 靶点

不需要裁剪，脚本直接从 RCSB 拉全结构、抽出 DNA 链：

```bash
python 01_prepare_inputs.py --only dna
# out/inputs/dna/7rte_a_dna.pdb 等，同时记录链号/残基范围/序列
# out/inputs/targets/dna_targets.json
```

论文里 DNA 靶点的 binder 长度是 100–130 aa；contig 形如 `A1-12,/0,100-130`。

---

## 3. PDB 镜像（训练 / 全量评测）

AtomWorks 自带 rsync 同步工具：

```bash
# 全量镜像（2025-08 时约 100 GB）
atomworks pdb sync /path/to/pdb_mirror

# 只拉几个结构（做小实验够用）
atomworks pdb sync /path/to/pdb_mirror --pdb-id 5o45 --pdb-id 4zxb
atomworks pdb sync /path/to/pdb_mirror --pdb-ids-file my_ids.txt   # 一行一个 ID

# CCD（配体 / 修饰残基定义）
atomworks ccd sync /path/to/ccd_mirror
atomworks ccd sync /path/to/ccd_mirror --ccd-code IAI --ccd-code OQO
```

**目录约定**（必须遵守，否则 AtomWorks 找不到文件）：

```
pdb_mirror/
├── a2/
│   └── 1a2b.cif.gz        # 1a2b -> <中间两位>/<id>.cif.gz
├── 5o/
│   └── 5o45.cif.gz
└── ...

ccd_mirror/
└── I/
    └── IAI/
        └── IAI.cif        # IAI -> <首字母>/<code>/<code>.cif
```

写完记得在 `foundry/.env` 里登记：

```bash
PDB_MIRROR_PATH=/path/to/pdb_mirror
CCD_MIRROR_PATH=/path/to/ccd_mirror
```

配置侧的落点（`models/rfd3/configs/paths/data/default.yaml`）：

```yaml
pdb_data_dir: /media/zzj/Data/          # 镜像根目录
pdb_parquet_dir: /media/zzj/Data/pdb_metadata_latest   # 元数据 parquet 目录
cif_cache_dir: /media/zzj/Data/pdb_mirror              # 解析缓存
```

> 只跑推理（不做训练/微调）的话，**不需要全量镜像**：`rfd3 design` 直接吃你给的
> PDB/CIF 路径，配体走 biotite 内置 CCD 兜底。全量镜像只对训练和
> `pdb_holdout.yaml` 那类"从 parquet 里批量取样"的评测才是必需的。

---

## 4. 元数据 parquet

训练/批量评测不直接扫文件系统，而是读两份 parquet 索引：

```bash
atomworks setup metadata /path/to/metadata_dir
# 官方归档，解压到 <dir>/shared/...；把 pdb_parquet_dir 指向对应目录即可
```

需要的文件与必需列：

`interfaces_df.parquet`（用于 `af3_train_interface.yaml`、`rfd3_train_interface.yaml`、`pdb_holdout.yaml`）

| 列 | 说明 |
|---|---|
| `example_id` | 主键，形如 `{['pdb','interfaces']}{9ml8}{1}{['G_1','H_1']}` |
| `pdb_id` / `assembly_id` / `deposition_date` / `resolution` / `method` | 基本元信息 |
| `num_polymer_pn_units` / `n_prot` / `n_nuc` / `n_ligand` / `n_peptide` | 组成统计 |
| `cluster` | 序列相似度簇（过滤要用 `cluster.notnull()`） |
| `pn_unit_1_iid` / `pn_unit_2_iid` | 界面两侧的 pn_unit 标识 |
| `pn_unit_1_non_polymer_res_names` / `pn_unit_2_...` | 界面配体名（用来排除 AF3 黑名单配体） |
| `is_inter_molecule` | 是否跨分子界面 |
| `all_pn_unit_iids_after_processing` / `involves_loi` | 处理后的单元清单 / 是否含 LOI |

`pn_units_df.parquet`（用于 `af3_train_pn_unit.yaml`）多一列 `q_pn_unit_*` 系列、
`total_num_atoms_in_unprocessed_assembly`、`q_pn_unit_is_loi`。

> 你工作区里的 `evaluation_manifest.csv` 就是这份 parquet 的等价物
> （列：`example_id, pdb_id, assembly_id, deposition_date, resolution,
> num_polymer_pn_units, n_prot, cluster, is_inter_molecule, pn_unit_1_iid,
> pn_unit_2_iid, ...`）。要做 `pdb_holdout.yaml` 那类时间外推评测，
> 可以直接把它转成 parquet 并补上缺失的列。

---

## 5. 蒸馏数据集（训练用）

`models/rfd3/configs/datasets/train/rfd3_monomer_distillation.yaml`：

```yaml
dataset:
  data: ${paths.data.monomer_distillation_parquet_dir}/af2_distillation_facebook.parquet
  columns_to_load: [example_id, path]
```

只要两列：

| 列 | 说明 |
|---|---|
| `example_id` | 形如 `{['monomer_distillation']}{top7_1enh}{1}{[]}` |
| `path` | 对应的 CIF 绝对路径 |

配套的 CIF 目录由 `monomer_distillation_data_dir` 指定。
来源是 **ESM-IF 论文的 AF2 预测集**（Hsu et al., *Learning inverse folding from millions
of predicted structures*, ICML 2022）—— Metagenomic Atlas / Facebook 系列预测结构。
本工作区 `single_structure_data/` 里已经有一个最小可跑版本
（`af2_distillation_facebook.parquet` + `1enh.cif` / `1qys.cif` + `monomer.json`），
可以当格式样板。

> RFD3NA 额外需要一个 TF–DNA 蒸馏集（`na_complex_distillation.yaml` 里的
> `/projects/ml/prot_dna/transcriptionFactor_distillation_rf3.newDL.csv`），
> 列同样是 `example_id` + `path`。跑 RFD3（非 NA）用不到。

---

## 6. 配体与 CCD

论文 §3.3 的四个配体：

| 配体 | 来源 | 备注 |
|---|---|---|
| `IAI` | 仓库自带 `models/rfd3/docs/input_pdbs/IAI.pdb` | 短名 |
| `OQO` | 仓库自带 `models/rfd3/docs/input_pdbs/7v11.pdb` | 从中抽链 |
| `FAD` | RCSB（任意含 FAD 的结构） | PDB 中常见 |
| `SAM` | RCSB（任意含 SAM 的结构） | PDB 中常见 |

```bash
python 01_prepare_inputs.py --only sm
# out/inputs/sm/IAI.pdb / OQO.pdb / FAD.pdb / SAM.pdb（只保留配体的 HETATM）
# out/inputs/targets/sm_ligands.json（记录配体原子名清单，RASA 条件要用）
```

单配体 CCD 定义按需拉：

```bash
atomworks ccd sync /path/to/ccd_mirror --ccd-code IAI --ccd-code OQO --ccd-code FAD --ccd-code SAM
```

RFD3 的配体条件：`ligand`（3 字母代码）、`select_fixed_atoms`（`""` 表示一起扩散，
`ALL` 表示刚体固定）、`select_buried` / `select_exposed`（RASA 条件）。

---

## 7. MSA 数据（折叠后端用）

| 后端 | MSA 获取方式 |
|---|---|
| **AF3** | 需要本地数据库（uniref90 / mgnify / bfd / uniref30 / pdb_seqres）+ jackhmmer、nhmmer。用官方 `build_data` 脚本下载（几百 GB） |
| **RF3** | 支持 `.a3m` / `.fasta`，**必须自己准备或先算好**；不支持在线搜 MSA |
| **Chai-1** | `--use-msa-server` 走 ColabFold 公共 MMseqs2 服务；或本地 `--msa-directory` |

AtomWorks 提供了一套 MSA 工具链：

```bash
atomworks msa generate   # 用 MMseqs2 从序列生成 MSA
atomworks msa find       # 在已有目录里按序列找 MSA
atomworks msa filter     # 用 HHfilter 去冗余
atomworks msa organize   # 整理成标准目录结构
```

对应的 `.env` 变量：

```bash
LOCAL_MSA_DIRS=                      # 本地 MSA 目录（冒号分隔）
MMSEQS2_PATH=/path/to/mmseqs         # 生成/搜索 MSA
HHFILTER_PATH=/path/to/hhfilter      # 过滤 MSA
COLABFOLD_LOCAL_DB_PATH_GPU=         # 本地 ColabFold 库
COLABFOLD_LOCAL_DB_PATH_CPU=
COLABFOLD_NET_DB_PATH_GPU=           # 网络回退（可能 IO 抖动）
COLABFOLD_NET_DB_PATH_CPU=
```

RF3 的 JSON 输入里每个链给 `msa_path` 即可：

```json
{"seq": "MSEQ...", "chain_id": "A", "msa_path": "/path/to/A.a3m"}
```

> 论文没有对 MSA 做特殊说明；PPI/DNA/小分子这些设计任务的评估通常不依赖 MSA
> （RF3 无 MSA 也能给出合理的 pTM）。追求严谨的话按上表配好。

---

## 8. 外部评估用的数据库

```bash
# FoldSeek —— Fig. S4b 新颖性（对 PDB 的最相似结构的 TM-score）
foldseek databases PDB100 /path/to/pdb100 tmp      # 建库
foldseek easy-search design.pdb /path/to/pdb100 out.m8 tmp \
  --format-output "query,target,alntmscore"

# 3DNA —— Fig. S3g 的 DNA 大沟/小沟宽度
#   装好后写 .env: X3DNA_PATH=/path/to/x3dna-v2.4

# Rosetta —— Fig. S4c 的 ΔΔG（DDGnoRepack），需要 license

# HBPLUS —— Fig. 2d 的氢键统计 + RFD3 的氢键条件
#   装好后写 .env: HBPLUS_PATH=/path/to/hbplus
```

---

## 9. 推荐的目录布局

```
experiments/paper_rfd3/out/
├── inputs/
│   ├── raw/                     # 从 RCSB 下下来的原始结构
│   ├── ppi/                     # 裁剪好的 PPI 靶点
│   ├── dna/                     # 抽出来的 DNA 链
│   ├── sm/                      # 只有配体的结构
│   ├── ame/                     # AME 活性位点 motif
│   ├── motifs/                  # §4 实验用的 motif（DBRFD3 / 半胱氨酸水解酶）
│   ├── targets/                 # 靶点清单 JSON + benchmark 目录
│   │   ├── dna_targets.json
│   │   ├── sm_ligands.json
│   │   ├── ame_cases.json
│   │   ├── holdout_pdb_ids.json
│   │   └── repo_benchmark_catalog.md
│   └── specs/                   # 可直接喂给 rfd3 design 的输入 JSON
├── designs/                     # RFD3 采样输出
├── sequences/                   # MPNN 序列设计输出
├── folds/                       # AF3 / Chai / RF3 折叠输出
├── metrics/                     # 逐设计 / 逐条件的指标 CSV
└── reports/                     # 汇总报告 + 图
```

环境变量对照（`env.sh` 里全部可覆盖，`paths.data.*` 是 Hydra 侧的名字）：

| env.sh 变量 | Hydra `paths.data.*` | 含义 |
|---|---|---|
| `INPUTS_DIR` | `design_benchmark_data_dir`（可指过来） | benchmark 输入与整理后的结构 |
| `DESIGNS_DIR` | `out_dir` | 采样输出 |
| `MPNN_CKPT_DIR` | — | MPNN 权重目录 |
| （`.env` 里）`PDB_MIRROR_PATH` | `pdb_data_dir` | PDB 镜像根 |
| （`.env` 里）`CCD_MIRROR_PATH` | — | CCD 镜像根 |
| — | `pdb_parquet_dir` | 元数据 parquet |

---

## 10. 最小清单：只想复现 §3 的 in silico benchmark

按顺序执行，**不需要 PDB 全量镜像、不需要元数据 parquet、不需要训练数据**：

```bash
cd experiments/paper_rfd3

# 1) 从仓库配置导出 benchmark 定义
python 02_extract_repo_benchmarks.py

# 2) 下载并整理评测输入
python 01_prepare_inputs.py            # 全部
# 或分别来：--only ppi / dna / sm / enzyme / symmetry / uncond / ppi dna

# 3) 跑（见 README.md / COMMANDS.md）
./run_all.sh
```

最终你需要的外部资源只有：

- **RCSB 网络下载**（PDB 结构与 DNA 靶点，几 MB 级）
- **MPNN 权重**两个 `.pt`（约 10 MB）
- **RFD3 权重**（`foundry install base-models`）
- **一个折叠后端**（RF3 权重约 1 GB，或 AF3 / Chai-1）
- **AME 的 41 个案例**（去 RFdiffusion2 仓库取，见 `COMMANDS.md` §B.2 实验 5）

一共几十 MB 到几 GB，一台单卡机器就能跑完整条 in silico 流程。
