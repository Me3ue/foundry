# RFdiffusion3 论文实验复现手册

> 论文：**De novo Design of All-atom Biomolecular Interactions with RFdiffusion3**
> (Butcher, Krishna, Mitra, Brent, Li, Corley et al., bioRxiv 2025.09.18.676967)
>
> 本目录把论文正文（§3 *In silico results*、§4 *Experiments*）里的**每一个实验**都翻译成
> 可直接执行的命令，并配了一套一键运行 + 自动汇总的脚本。论文原文文本已抽取到
> `docs/paper/rfd3_paper.txt` 便于对照。
>
> 📄 **另见 [`COMMANDS.md`](./COMMANDS.md)** —— 论文的**设备规模**（GPU 型号、参数量、
> 每个实验的骨架数/序列数/预测次数）+ **不带任何本机降级处理的标准命令**
> （官方 AF3 / Chai-1 / RFdiffusion2 / Rosetta 调用方式）。
>
> 📄 **[`DATASETS.md`](./DATASETS.md)** —— **数据集怎么获取与组织**：论文 §3 全部
> benchmark 的输入定义其实就在 `models/rfd3/configs/datasets/val/` 里（含 holdout
> 靶点的 PDB ID 和编号偏移）、PDB/CCD 镜像、元数据 parquet、蒸馏集、MSA、目录布局。

---

## 0. 三分钟上手

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3

# 0) 从仓库自带配置里导出论文 benchmark 定义（PPI 五个靶点 + holdout 清单）
python 02_extract_repo_benchmarks.py

# 0b) 准备输入结构。不必同步 100 GB 的全量 PDB 镜像：
#     默认逐文件从 RCSB 下载，论文 §3 全部实验一共只要 21 个结构（几十 MB）。
python 01_prepare_inputs.py
#     若已从本地镜像抽了子集（见 DATASETS.md §3.1），改成：
#     python 01_prepare_inputs.py --mirror /data/$USER/pdb_mirror --no-download

# 1) 先做一次环境自检（检查 GPU / rfd3 / mpnn / 权重路径）
source ./env.sh && source ./lib.sh && check_env

# 2) 冒烟测试：每个条件 8 个骨架，跑通全流程
./run_all.sh

# 3) 论文规模（预计数天，见 §6 硬件适配）
SCALE=paper ./run_all.sh

# 4) 只汇总（任何时候都能单独跑）
python 90_summarize.py
```

产物全部落在 `out/`：

| 路径 | 内容 |
|---|---|
| `out/designs/<实验>/<条件>/` | RFD3 生成的骨架（含 denoised cif.gz） |
| `out/sequences/…` | ProteinMPNN / LigandMPNN 设计的序列 |
| `out/folds/…` | 自洽性折叠（RF3/AF3/Chai）的预测与置信度 |
| `out/metrics/per_design.csv` | 每个设计一行的全部原始指标 |
| `out/metrics/summary_by_condition.csv` | 每个实验条件一行的论文口径通过率 |
| `out/metrics/geometry.csv` | 几何指标（各种 RMSD / RASA / 氢键 / 界面） |
| `out/reports/paper_experiments_report.md` | **与论文数字自动对照的汇总报告** |
| `out/reports/fig1d_speed.svg` | Fig. 1d 速度曲线 |

---

## 1. 论文实验 ↔ 脚本 ↔ 论文出处 对照表

| 论文位置 | 实验 | 脚本 | 能否本地跑 |
|---|---|---|---|
| §3 开头 / Fig. S1c | 无条件单体生成、η 扫描 | `10_exp1_unconditional.sh` | ✅ 完全可复现 |
| §3.1 / Fig. 3a / Fig. S2 | 蛋白结合蛋白（5 靶点 × 400 骨架） | `11_exp2_ppi.sh` | ✅（靶点结构需自备 3 个） |
| §3.2 / Fig. 3b / Fig. S3 | DNA 结合蛋白（3 靶点，rigid / diffused） | `12_exp3_dna.sh` | ✅ 完全可复现 |
| §3.3 / Fig. 3c / Fig. S4 | 小分子结合蛋白（FAD/SAM/IAI/OQO） | `13_exp4_small_molecule.sh` | ⚠️ RFdiffusionAA 基线需外部仓库 |
| §3.4 / Fig. 3d / Fig. S5 | 酶设计 AME benchmark（41 个活性位点） | `14_exp5_enzyme.sh` | ⚠️ AME 案例清单需自备 |
| Fig. 2g / Fig. S6 | 对称设计（D2/C3/C5/C7、C2 对称酶） | `15_exp6_symmetry.sh` / `14 … --symmetry` | ✅ 完全可复现 |
| Fig. 2d / 2e / 2f | 氢键条件、RASA 埋藏、质心条件 | `16_exp7_conditioning.sh` | ⚠️ 氢键需 HBPLUS |
| Fig. 1d | RFD1/2/3 推理速度标定 | `17_exp8_speed.sh` | ⚠️ 只有 RFD3 那条曲线 |
| §4 / Fig. 4 / S7 / S8 | DNA 结合物、半胱氨酸水解酶 | `18_exp9_wetlab_insilico.sh` | ⚠️ 仅计算部分；湿实验不可复现 |
| Fig. S9 | MPNN 后界面接触/电荷/组成保留 | `70_metrics_geometry.py` | ✅ |

> **重要提醒**：论文正文没有附补充方法（Supplemental Methods / Section 3.5 / 4.2 不在本 PDF 里），
> 所以少数超参数（AME 的 41 个案例定义、DBRFD3 的 motif 文件、筛选阈值细节）需要从
> 论文的补充材料或作者仓库补齐。脚本在这些地方都会明确提示，并用占位文件保证流程不中断。

---

## 2. 前置条件

### 2.1 已就绪的环境

本机已经具备（脚本默认值就指向它们）：

| 项目 | 路径 |
|---|---|
| Foundry/RFD3 Python 环境 | `/home/zhangzijian/anaconda3/envs/rc`（含 `rfd3` / `mpnn` / `rf3` / `rfd3na` 命令） |
| RFD3 权重 | `/media/zzj/Data/rfd3_latest.ckpt` |
| HBPLUS | `/home/zhangzijian/protein/HBPLUS/hbplus/hbplus` |
| TMalign（多样性聚类用） | `/usr/bin/TMalign` |

所有路径都能用环境变量覆盖，例如：

```bash
RC_ENV_BIN=/path/to/venv/bin RFD3_CKPT=/path/to/ckpt ./run_all.sh
```

### 2.2 还需要下载的两个权重

```bash
mkdir -p ~/.foundry/checkpoints && cd ~/.foundry/checkpoints

# ProteinMPNN（蛋白-蛋白 / DNA 用）——论文里 4 条序列/骨架
wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt

# LigandMPNN（小分子 / 酶 / DNA 用）——论文里 8 条序列/骨架
wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt

# RF3（本地自洽性折叠后端，代替 AF3）
wget http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt
```

> ⚠️ `models/mpnn/README.md` 明确写了：**benchmark 对比请用原始官方仓库的权重**，
> Foundry 里的 MPNN 是重实现版本。要严格对齐论文的序列设计结果，建议用
> [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) /
> [LigandMPNN](https://github.com/dauparas/LigandMPNN) 原仓库的权重。

### 2.3 评估后端（论文用的是 AF3 / Chai-1）

论文的所有"通过率"都依赖 PAE / pTM / ipTM，只有结构预测模型能给。三个选择：

| 后端 | 设置方式 | 说明 |
|---|---|---|
| **RF3**（默认） | `FOLD_BACKEND=rf3` | 本仓库自带，输出 AF3 风格的 `*_summary_confidences.json`，**推荐先用它跑通全流程** |
| **AlphaFold3** | `FOLD_BACKEND=af3` | 论文原版。权重需向 Google DeepMind 申请（`alphafold3` 仓库 / AF3 Server） |
| **Chai-1** | `FOLD_BACKEND=chai` | 开源 AF3 复现，论文的酶设计用的就是它 |

选 `af3`/`chai` 时脚本只负责生成输入文件（`out/folds/<实验>/<条件>/fold_inputs.json`），
把文件交给对应环境跑完、结果放回同目录即可，`90_summarize.py` 会自动识别。

### 2.4 可选的外部工具

```bash
# FoldSeek —— Fig. S4b 的"新颖性"（每个设计对 PDB 的最相似 TM-score）
conda install -c bioconda foldseek

# Rosetta —— Fig. S4c 的 ΔΔG（DDGnoRepack）
# 需 Rosetta license，装好后把 `ddg_monomer` / `InterfaceAnalyzer` 加进 PATH

# 3DNA (x3dna-dssr) —— Fig. S3g 的 DNA 大沟/小沟宽度
# 装好后在 foundry/.env 里设置 X3DNA_PATH

# RDKit —— Fig. S4d 的配体构象 RMSD（每个配体 50 个构象）
pip install rdkit
```

---

## 3. 论文的采样参数（全部实验共用）

论文 Fig. S4 脚注明确写出，除非另有说明全部实验用：

| 参数 | 值 | 对应 CLI 覆盖 |
|---|---|---|
| step scale **η** | 1.5 | `inference_sampler.step_scale=1.5` |
| noise level **γ0** | 0.6 | `inference_sampler.gamma_0=0.6` |
| 去噪步数 | 200 | `inference_sampler.num_timesteps=200` |
| recycling | 2（checkpoint 默认） | `inference_sampler.n_recycle=2` |
| CFG scale（仅小分子 diffused） | 2.0 | `inference_sampler.cfg_scale=2.0` + `use_classifier_free_guidance=True` |
| 序列数 / 骨架 | PPI、DNA：4；小分子、酶：8 | `--n-seqs` |

这些正好就是 RFD3 的默认值（见 `models/rfd3/configs/inference_engine/rfdiffusion3.yaml`），
所以复现论文时**不要**套用官方文档给 PPI 的"生产推荐值"（`step_scale=3, gamma_0=0.2`）——
那个命中率更高但多样性更低，和论文 Fig. 3a 的口径不是一回事。

### 论文的通过判据（脚本已内置，见 `lib/metrics.py`）

| 实验 | 判据 |
|---|---|
| PPI（§3.1） | min inter-chain PAE ≤ 1.5 **且** binder pTM ≥ 0.8 **且** target-aligned binder Cα RMSD < 2.5 Å |
| DNA（§3.2） | DNA 磷酸对齐后的蛋白 Cα RMSD（裁剪末端）< 5 Å；分档 <1.5 / 1.5-3 / 3-5 Å |
| 小分子（§3.3） | backbone RMSD ≤ 1.5 Å **且** ligand RMSD ≤ 5 Å **且** min chain-pair PAE ≤ 1.5 **且** ipTM ≥ 0.8 |
| 酶（§3.4） | motif backbone-aligned motif **all-atom** RMSD < 1.5 Å |

**一条骨架取最好的一条序列**（论文原文：*"The minimum RMSD for each backbone was taken as a representative to score each backbone"*），
`70_metrics_geometry.py` 的 `aggregate()` 正是这么做的。

---

## 4. 手动逐条命令（不想用一键脚本时）

### 实验 1 —— 无条件单体（§3 / Fig. S1c）

```bash
python 01_prepare_inputs.py --only uncond
# 生成 out/inputs/specs/exp1_eta_1.5.json：
#   {"uncond_L100": {"length":100,"is_non_loopy":true}, ...}

rfd3 design \
  out_dir=out/designs/exp1_unconditional/eta_1.5 \
  inputs=out/inputs/specs/exp1_eta_1.5.json \
  ckpt_path=/media/zzj/Data/rfd3_latest.ckpt \
  diffusion_batch_size=4 n_batches=2 \
  inference_sampler.step_scale=1.5 \
  inference_sampler.gamma_0=0.6 \
  inference_sampler.num_timesteps=200 \
  prevalidate_inputs=True skip_existing=True
```

论文数字：长度 100-200 aa，**98% 的设计至少有 1 条序列折叠到 1.5 Å 以内**（8 条 ProteinMPNN 序列）。
Fig. S1c 的结论：η 越大 → designability 越高、diversity 越低；论文选 η=1.5。

### 实验 2 —— 蛋白结合蛋白（§3.1 / Fig. 3a）

```bash
python 02_extract_repo_benchmarks.py      # 先从仓库配置导出 benchmark 定义
python 01_prepare_inputs.py --only ppi    # 下载沉积结构 → 裁出靶点链 → 校验热点

# 400 条骨架 = diffusion_batch_size 8 × n_batches 50
rfd3 design \
  out_dir=out/designs/exp2_ppi/all \
  inputs=out/inputs/specs/exp2_ppi.json \
  ckpt_path=/media/zzj/Data/rfd3_latest.ckpt \
  diffusion_batch_size=8 n_batches=50 \
  inference_sampler.step_scale=1.5 inference_sampler.gamma_0=0.6
```

**靶点的权威定义就在仓库里**：`models/rfd3/configs/datasets/val/val_examples/`
`bcov_ppi_easy_medium_with_ori.yaml`（含原子级热点）与 `bpem_ori_hb.yaml`（含氢键条件）。
论文的五个靶点 = 训练配置里显式排除的那五个 PDB ID：
`5o45`(PD-L1)、`4zxb`(InsulinR)、`2gy5`(Tie2)、`1z92`(IL-2Ra)、`3di3`(IL-7Ra)。

⚠️ benchmark 里的输入文件被作者**裁剪 + 重编号**过，且没随仓库发布。
`01_prepare_inputs.py` 会按已验证的编号偏移（PD-L1 +16、InsulinR +5）从 RCSB 沉积结构重建，
并用热点的原子名自校验；IL-7Ra 的定义缺失，需在 `out/inputs/targets/ppi_extra_targets.json` 手工补。
**细节与验证过程见 [`DATASETS.md` §2](./DATASETS.md)。**

论文数字：RFD3 在未聚类的 4/5 个靶点、聚类后的 **5/5** 个靶点上优于 RFD1；
TM-score 0.6 聚类后平均 **8.2 个成功簇 vs RFD1 的 1.4 个**。

### 实验 3 —— DNA 结合蛋白（§3.2 / Fig. 3b）

```bash
python 01_prepare_inputs.py --only dna
# 自动下载 7RTE / 7N5U / 7M5W，抽出 DNA 链，写好 rigid / diffused 两套规格

for setting in rigid diffused; do
  rfd3 design \
    out_dir=out/designs/exp3_dna/$setting \
    inputs=out/inputs/specs/exp3_dna_${setting}.json \
    ckpt_path=/media/zzj/Data/rfd3_latest.ckpt \
    diffusion_batch_size=8 n_batches=50
done
```

* `rigid`：`"select_fixed_atoms": {"A1-12": "ALL"}` —— DNA 构象取自晶体结构（论文的 **+**）
* `diffused`：`"select_fixed_atoms": {"A1-12": ""}` —— 只给序列，DNA 构象与蛋白联合采样（论文的 **−**）

论文数字：<5 Å 的通过率 **单体 8.67%、二体 6.67%**；固定界面时 6.5% / 5.5%。

### 实验 4 —— 小分子结合蛋白（§3.3 / Fig. 3c）

```bash
python 01_prepare_inputs.py --only sm

# (a) 配体刚体固定
rfd3 design out_dir=out/designs/exp4_small_molecule/fixed \
  inputs=out/inputs/specs/exp4_sm_fixed.json ckpt_path=... \
  diffusion_batch_size=8 n_batches=50

# (b) 配体坐标一起扩散 + RASA=buried + CFG=2
rfd3 design out_dir=out/designs/exp4_small_molecule/diffused \
  inputs=out/inputs/specs/exp4_sm_diffused.json ckpt_path=... \
  diffusion_batch_size=8 n_batches=50 \
  inference_sampler.use_classifier_free_guidance=True \
  inference_sampler.cfg_scale=2.0
```

论文数字：RFD3 在**全部 4 个配体**上都优于 RFdiffusionAA；
RFD3 的设计更有多样性、更远离训练集、Rosetta ΔΔG 更低。

### 实验 5 —— 酶设计 AME benchmark（§3.4 / Fig. 3d）

**AME 协议的权威定义在 RFdiffusion2 的论文里**（Ahern et al., Nature Methods 2025,
doi:10.1038/s41592-025-02975-x），RFD3 论文 §3.5 直接引用它。原文流程：

1. 取 **M-CSA**（Mechanism and Catalytic Site Atlas）的全部酶，用 **PARITY 数据库**
   交叉比对，筛出"所有反应物都在结构里"的条目；
2. 加质量过滤 → **41 个案例**；
3. 对 M-CSA 标注的每个催化残基，**随机取该残基的一个原子子图**作为催化原子；
4. 每个案例 **生成 100 条骨架** → **LigandMPNN 配 8 条序列** → **Chai-1 折叠**；
5. **成功判据**：至少一条序列的 **5 个 Chai-1 diffusion sample 中有一个**，把生成
   motif 残基的**骨架**对齐到预测坐标后，**所有原子侧链 RMSD < 1.5 Å**，
   且配体与预测骨架原子**无 clash**。

代码与案例清单：**https://github.com/RosettaCommons/RFdiffusion2/**
（`rf_diffusion/benchmark/open_source_demo.json` 里有可直接跑的 demo）

```bash
python 01_prepare_inputs.py --only enzyme
# ⚠️ 会生成 out/inputs/targets/ame_cases.json 模板，需要你把 41 个案例填进去
```

每个案例一条记录：

```json
{
  "name": "M0097_1ctt",
  "input": "out/inputs/ame/M0097_1ctt.pdb",
  "ligand": "LIG",
  "unindex": "A108,A139,A152,A156",
  "length": "180-200",
  "fixed_atoms": {"A108": "ND2,CG", "A139": "OG,CB,CA"},
  "n_islands": 2,
  "symmetry_id": null
}
```

* `unindex` 是论文的关键机制：把催化残基的**序列位置交给模型自己找**（无索引原子作为额外 token）。
* `n_islands` = 该案例的 residue islands 数量，用于画 Fig. 3d 的分组曲线。
* 加 `"symmetry_id": "C2"` 就进入 Fig. S6 的对称子集。

论文数字：41 个案例中 **37 个优于 RFD2（90%）**；>4 个 residue islands 的难案例
**通过率 15% vs RFD2 的 4%**（n=12）。

### 实验 6 —— 对称设计（Fig. 2g / Fig. S6c-d）

```bash
SYMMETRY_IDS="D2 C3 C5 C7" ./15_exp6_symmetry.sh
```

对应规格：`{"sym_C5": {"length":100, "is_non_loopy":true, "symmetry":{"id":"C5"}}}`
论文数字：D2/C3/C5/C7 的 AF3 Cα RMSD 分别为 0.832 / 0.450 / 0.614 / 0.539 Å。

### 实验 7 —— 原子级条件控制（Fig. 2d / 2e / 2f）

```bash
ONLY=hbond ./16_exp7_conditioning.sh   # 氢键供体/受体（需 HBPLUS）
ONLY=rasa  ./16_exp7_conditioning.sh   # RASA 埋藏分档
ONLY=com   ./16_exp7_conditioning.sh   # 质心条件
```

* 氢键：`select_hbond_acceptor` / `select_hbond_donor`，成对比较
  「无条件 / 有条件 / 有条件+CFG」三种设置的氢键比例。
  论文数字：小分子 **26.67% → 32.67% → 36.67%**；DNA **11% → 11.3% → 12.5%**。
* RASA：`select_buried` / `select_exposed`（`spec_rasa_grade` 给的是
  all_buried / half / all_exposed 三档）。
* 质心：`ori_token` 两个相反方向，生成的结构应聚成两簇。

### 实验 8 —— 推理速度（Fig. 1d）

```bash
LENGTHS="50 100 150 200 250 300" REPEATS=3 ./17_exp8_speed.sh
python 91_plot_speed.py     # 输出 out/reports/fig1d_speed.svg
```

要画完整的 RFD1/RFD2/RFD3 三条曲线，需要分别克隆
[RFdiffusion](https://github.com/RosettaCommons/RFDiffusion) 和 RFdiffusion2 的官方仓库，
把它们的 wall-clock 按同样的 CSV 格式（`backend,length,repeat,wallclock_s,…`）
追加到 `out/metrics/speed_scaling.csv` 即可，画图脚本会自动多画一条线。

### 实验 9 —— §4 实验的计算部分（Fig. 4 / Fig. S7 / Fig. S8）

```bash
./18_exp9_wetlab_insilico.sh
```

* **(a) DNA 结合物两阶段流程**：先把靶 DNA 序列的 AF3 预测结构当输入采样
  （`exp9_dbrfd3_stage1.json`），再把与 DNA 接触的 motif 固定、重采样其余骨架
  （`exp9_dbrfd3_stage2.json`）。论文靶序列 **CGAGAACATAGTCG**，
  最终 DBRFD3 的 EC50 = 5.89 ± 2.15 µM。
* **(b) 半胱氨酸水解酶**：motif 来自 Ulp-1（PDB **1EUV**）的 Cys-His-Asp 三联体 + Gln +
  底物 4-methylumbelliferyl phenyl acetate 的第一个四面体中间体几何。
  把整理好的 motif 放到 `out/inputs/motifs/cysteine_hydrolase_C6.pdb` 即可开跑。
  论文筛了 **190 个设计 → 35 个多轮转化 → 最优 C6 的 kcat/Km = 3557**。

> 湿实验（酵母表面展示滴定、IVTT 荧光筛选、Michaelis-Menten 动力学）不在计算可复现范围内，
> 脚本结尾会打印论文的对照数字，方便你把计算侧结果和论文对起来。

---

## 5. 汇总与对照

```bash
python 90_summarize.py                    # 全部实验
python 90_summarize.py --experiment exp2_ppi exp3_dna
```

`out/reports/paper_experiments_report.md` 里第一张表就是
「论文口径指标 / 论文报告数字 / 本次复现结果」三列对照，直接能看出差在哪。
`lib/summarize.py` 顶部的 `PAPER_REFERENCE` 字典集中保存了所有论文数字，
要加新指标改那里就行。

---

## 6. 硬件适配与规模

### 6.1 本机配置（6× RTX A6000 + 503 GB 内存）

`env.sh` 已经按这套硬件调好，**不需要手工改任何东西**：

| 配置项 | 取值 | 原因 |
|---|---|---|
| `LOW_MEMORY` | `False` | A6000 有 48 GB，不需要分块 tokenization 的降级路径 |
| `DIFFUSION_BATCH_SIZE` | `8` | 论文原版默认值；只影响吞吐不影响采样分布 |
| `MAX_PARALLEL_GPUS` | `auto` | 按当前空闲显存决定并行度 |
| `GPU_POOL_MIN_FREE` | 25000 MiB | 空闲不足 25 GB 的卡不再往上排任务 |
| `GPU_JOB_MEM` | 25000 MiB | 单任务预扣显存，避免超卖 |
| `N_WORKERS` | `nproc/2`（上限 32） | MPNN / 几何指标的进程数 |
| `OMP_NUM_THREADS` | 4 | 防止 N 个进程 × M 线程互相抢 CPU |

**每个实验内部把条件拆开铺到多张卡上**（由 `lib/gpu_pool.py` 调度）：

| 实验 | 并行任务数 | 每任务 |
|---|---|---|
| 1 无条件 | 4 | 每个 η |
| 2 PPI | 4–5 | 每个靶点 |
| 3 DNA | 6 | 2 设置 × 3 靶点 |
| 4 小分子 | 8 | 2 条件 × 4 配体 |
| 5 酶 AME | **41** | 每个活性位点案例 |
| 6 对称 | 4 | D2/C3/C5/C7 |
| 7 条件控制 | 13 | 各对照组 |
| 8 速度 | 1（串行） | 测速必须独占 GPU |
| 9 湿实验计算 | 3 | 两阶段 + 水解酶 |

实验之间**串行**执行——每个实验自己就会吃满 GPU 池，再并行只会互相抢卡。

### 6.2 常用调法

```bash
# 只用几张指定的卡（比如别人在占 0/1/3/5）
GPUS=2,4 ./run_all.sh

# 显存充足时提高吞吐（不改采样分布，只缩短墙钟时间）
DIFFUSION_BATCH_SIZE=16 ./run_all.sh

# 前端调小并行度，避免打扰同机的其他人
MAX_PARALLEL_GPUS=2 GPU_POOL_MIN_FREE=40000 ./run_all.sh

# 产物别放在 /dev/shm（上千个 CIF 会吃内存）
OUT=/scratch/$USER/rfd3_paper ./run_all.sh
```

想看当前分配情况：`source ./env.sh && source ./lib.sh && gpu_table`

### 6.3 规模档位

| SCALE | 骨架数/条件 | MPNN 序列数/骨架 | 说明 |
|---|---|---|---|
| `quick`（默认） | 8 | 2 | 冒烟测试，验证流程通畅 |
| `half` | 200 | 4 | 半规模，指标已有统计意义 |
| `paper` | 400 | 4（小分子/酶为 8） | 论文规模 |

显存不够时的兜底：`LOW_MEMORY=True DIFFUSION_BATCH_SIZE=1`。
减小 batch 只是把一次 forward 的样本数变小，总骨架数由 `n_batches` 控制，
**科学口径不变**（`rfd3_design` 会自动把 N 个骨架拆成 `ceil(N/batch)` 次）。

---

## 7. 目录结构

```
experiments/paper_rfd3/
├── README.md                  # 本文件：实验手册
├── COMMANDS.md                # 论文设备规模 + 标准运行指令（无本机降级）
├── DATASETS.md                # 数据集怎么获取与组织（含仓库自带 benchmark 的位置）
├── env.sh                     # 全部路径与超参数（含硬件自动探测，可环境变量覆盖）
├── lib.sh                     # bash 工具：日志 / rfd3_design_cmd / run_on_gpus / check_env
├── lib/
│   ├── gpu_pool.py            # 多卡调度器：按显存把任务铺到 6 张 A6000 上
│   ├── common.py              # 路径约定 + 论文各实验的 RFD3 输入规格构造
│   ├── metrics.py             # 几何原语(aligned_rmsd/TM 并行聚类)、RASA、氢键、clash、判据
│   └── summarize.py           # 扫描 out/ → CSV + Markdown 对照报告
├── 01_prepare_inputs.py       # 下载并整理输入（本地缓存 → 本地镜像 → RCSB 逐文件下载）
├── 02_extract_repo_benchmarks.py  # 从仓库 val/ 配置导出论文 benchmark 定义
├── 03_mirror_subset.py        # 从 100 GB 本地镜像抽几十 MB 子集，不用整盘同步
├── 10…18_*.sh                 # 实验 1-9（各自内部多卡并行）
├── 50_sequence_design.py      # 批量 MPNN 序列设计（多进程，自动只重设计 binder）
├── 60_fold.py                 # 批量结构预测（分片 + 多卡，rf3 / af3 / chai）
├── 70_metrics_geometry.py     # 各种 aligned-RMSD / 界面 / RASA / 氢键 / clash（多进程）
├── 90_summarize.py            # 汇总
├── 91_plot_speed.py           # Fig. 1d 曲线（纯 Python 输出 SVG，无依赖）
├── run_all.sh                 # 一键跑全部 + 汇总
└── out/                       # 全部产物（布局见 DATASETS.md §9）
```

---

## 8. 常见问题

**Q：`rfd3: command not found`**
`env.sh` 默认指向 `/home/zhangzijian/anaconda3/envs/rc/bin`。换了环境就设 `RC_ENV_BIN=<你的>/bin`。

**Q：某个任务报 `OutOfMemoryError: CUDA out of memory`**
A6000 48 GB 正常情况下不会 OOM（`GPU_JOB_MEM` 已按 25 GB 预扣）。真遇到时按顺序试：

1. `GPU_JOB_MEM=35000 ./run_all.sh` —— 提高单任务预扣，让调度器少往同一张卡塞；
2. `DIFFUSION_BATCH_SIZE=4` —— 减小单次 forward 的样本数（总骨架数仍由 `n_batches` 控制，口径不变）；
3. `GPU_POOL_MIN_FREE=40000` —— 只往明显空闲的卡上排；
4. 如果是**别的程序**占了卡：`GPUS=2,4 ./run_all.sh` 指定专用卡。

**Q：跑得比预期慢 / GPU 没吃满**
先 `source ./env.sh && source ./lib.sh && gpu_table` 看有几张卡真空闲。
如果只有 1–2 张卡满足 `GPU_POOL_MIN_FREE=25000`，并行度自然只有 1–2；
可以调低门槛：`GPU_POOL_MIN_FREE=15000 ./run_all.sh`。

**Q：`90_summarize.py` 里一堆 `no_fold`**
说明折叠这一环没跑。检查 `WITH_FOLD=1`、`FOLD_BACKEND` 是否正确，
或者看 `out/folds/<实验>/<条件>/fold_inputs.json` 有没有生成。
`FOLD_BACKEND=none` 时只生成输入文件，指标需要外部后端回填。

**Q：几何指标全是空**
先跑 `python 70_metrics_geometry.py`。它负责把设计结构和折叠结果配对；
如果 `geometry.csv` 里 `status` 是 `no_fold`，还是折叠没跑。

**Q：怎么把 AF3/Chai 的结果接进来？**
把它们的输出放到 `out/folds/<实验>/<条件>/` 下，文件名规则
`<设计文件名>_model.cif` + `<设计文件名>_summary_confidences.json`，
`90_summarize.py` 会自动按前缀配对。

**Q：为什么 AME / 半胱氨酸水解酶的 motif 文件要我自己准备？**
这两处的具体原子清单写在论文的补充材料（Section 3.5 / 4.2）里，不在正文 PDF 中。
本目录的 `01_prepare_inputs.py` 和 `18_exp9_wetlab_insilico.sh` 会生成模板并提示格式。
