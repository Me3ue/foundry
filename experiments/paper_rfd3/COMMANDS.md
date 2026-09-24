# 论文设备规模 + 标准运行指令参考

> 本文档只讲**正常/标准**的跑法：官方安装方式、官方命令、论文的原始规模。
> 不包含任何为小显存 GPU 做的降级处理（那些在 `README.md` §6 和 `env.sh` 里）。
>
> 标注说明：**[论文原文]** = 论文正文明确写出；**[论文推算]** = 按论文描述加总；
> **[官方文档]** = 来自 Foundry / AF3 / Chai-1 / RFdiffusion2 的官方文档。
>
> 📄 数据的获取与组织（PDB 镜像、元数据 parquet、benchmark 输入、蒸馏集、MSA）
> 见 [`DATASETS.md`](./DATASETS.md)。

---

## A. 论文中的设备规模

### A.1 硬件

| 项目 | 值 | 出处 |
|---|---|---|
| 推理速度标定的 GPU | **NVIDIA A6000** | [论文原文] Fig. 1d 图注："runtimes are measured on NVIDIA A6000 GPUs" |
| 训练用卡型/卡数/总卡时 | 论文正文未给出 | 补充方法（Supplemental Methods）里有，不在这份 PDF 中 |

> 论文只在 Fig. 1d 明确点了 A6000。完整的训练/推理算力预算写在补充材料里，
> 需要去 bioRxiv 的 Supplementary 文件或作者仓库取。

### A.2 模型规模 [论文原文]

| 指标 | RFD3 | 对照 |
|---|---|---|
| 可训练参数 | **168 M** | AF3 ≈ 350 M |
| 条件模块 | Pairformer 从 48 层缩到 **2 层** | AF3 用 48 层 |
| 三角乘法 / 三角注意力 | **完全省略** | RFD1/RFD2/AF3 都用了 |
| 每个残基的原子表示 | 固定 **14 个原子**（4 骨架 + 10 侧链，Trp 侧链原子数） | — |
| 推理提速 | 比 RFD2 快约 **10×**，比 RFD1 快一个数量级 | — |
| 训练数据 | PDB 全复合物（收录至 **2024 年 12 月**）+ Hsu et al. 的 **AF2 蒸馏集** | — |
| 训练流程 | 先混合预训练（AF2 预测 + PDB），再在更大比例的 DNA / PPI 上微调 | — |

### A.3 每个实验的采样规模

统一采样参数 [论文原文, Fig. S4 脚注]：**η = 1.5、γ0 = 0.6、200 步去噪**
（小分子 diffused-ligand 额外开 CFG，scale = 2.0）。

| 实验 | 骨架数 | 序列数/骨架 | 预测方法 | 结构预测总次数 |
|---|---|---|---|---|
| §3 无条件单体 | 96（长度 100–250） | 8（ProteinMPNN） | AF3 | ≈ 770 |
| §3.1 PPI | **5 靶点 × 400 = 2 000** | 4（ProteinMPNN） | AF3 | **8 000** |
| §3.1 的 RFD1 对照 | 5 × 400 = 2 000 | 4 | AF3 | 8 000 |
| §3.2 DNA（rigid + diffused 两设置） | 3 靶点 × 400 × 2 = **2 400** | 4（LigandMPNN） | AF3 | **9 600** |
| §3.3 小分子（fixed + diffused 两条件） | 4 配体 × 400 × 2 = **3 200** | 8（LigandMPNN） | AF3 | **25 600** |
| §3.3 的 RFdiffusionAA 基线 | 4 × 400 = 1 600 | 8 | AF3 | 12 800 |
| §3.4 酶 AME | **41 案例 × 100 = 4 100** | 8（LigandMPNN） | **Chai-1**（每序列 5 个 diffusion sample） | **32 800** |
| Fig. S9 界面分析 | 每个 design case 400 个结构 | 4 或 8 | — | — |
| §4 半胱氨酸水解酶 | 190 个设计送筛 | — | AF3 | — |
| §4 DNA 结合物 | 5 个设计送合成 | 4（LigandMPNN） | AF3 | — |

**合计规模 [论文推算]**：AF3 / Chai-1 的结构预测调用量在 **7.5 万次量级**；
RFD3 本身的采样量约 **1.2 万条骨架**。

> 注意：正文 §3.2 第一段提到"对每个序列生成 **100** 个结构"（那是 DNA 结合物的
> 初步测试），而 Fig. 3b 图注写的是 **400 backbones per target**。两个数字对应的
> 是不同的实验阶段，别混。

**关键结论：整个 benchmark 的算力瓶颈在结构预测（AF3/Chai-1），不在 RFD3。**
RFD3 只用一张 A6000 就能扛；7.5 万次 AF3/Chai 折叠才是真正吃卡的部分。

### A.4 论文自己报告的算力优势 [论文原文]

> "RFD3 achieves improved performance compared to prior approaches on a range of
> in silico benchmarks **with one tenth the computational cost**."

即同样的 benchmark，RFD3 的总算力约为前作（RFD1/RFD2）的 1/10。

### A.5 用于评估的软件

| 用途 | 论文用的工具 | 官方文档推荐的硬件 |
|---|---|---|
| 自洽性折叠（PPI / DNA / 小分子） | **AlphaFold3** | A100 80 GB（单卡可预测到约 5 120 残基）；官方示例用 2×A100 |
| 自洽性折叠（酶 / AME） | **Chai-1** | A100 80 GB 或 H100 80 GB；A10/A30 可跑小复合物；RTX 4090 也可 |
| 序列设计 | ProteinMPNN / LigandMPNN | CPU 即可 |
| 界面/氢键分析 | HBPLUS、Biotite（SASA） | CPU |
| DNA 沟宽 | 3DNA Analyze | CPU |
| 新颖性 | FoldSeek | CPU |
| ΔΔG | Rosetta（DDGnoRepack） | CPU，需 license |

---

## B. 标准运行指令

### B.0 安装与权重

```bash
# 全部三个模型（RFD3 + RF3 + ProteinMPNN/LigandMPNN）
pip install "rc-foundry[all]"

# 只装 RFD3
pip install "rc-foundry[rfd3]"

# 下载基础模型权重（默认落到 ~/.foundry/checkpoints）
foundry install base-models --checkpoint-dir ~/.foundry/checkpoints

# 核对
foundry list-available
foundry list-installed
```

MPNN 权重单独下载（论文用的是官方原版权重）：

```bash
# ProteinMPNN：PPI / DNA 用，论文每条骨架 4 条序列
wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt

# LigandMPNN：小分子 / 酶 / DNA 用，论文每条骨架 8 条序列
wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt
```

HBPLUS（氢键条件必需）：

```bash
# https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html
# 装好后在 foundry/.env 里设置：
HBPLUS_PATH=/path/to/hbplus
```

### B.1 通用四步流程

论文所有 in silico 实验都是同一套骨架：

```bash
# 1. RFD3 采样骨架
rfd3 design out_dir=<out> inputs=<spec.json> ckpt_path=<rfd3.ckpt>

# 2. ProteinMPNN / LigandMPNN 设计序列
mpnn --model_type protein_mpnn \
     --checkpoint_path <proteinmpnn.pt> \
     --is_legacy_weights True \
     --structure_path <design.cif> \
     --out_directory <seq_out> \
     --write_fasta True --write_structures True \
     --batch_size 4 --number_of_batches 1

# 3. 折叠（AF3 / Chai-1 / RF3）
python run_alphafold.py --json_path=<af3_input.json> \
     --model_dir=<params> --db_dir=<dbs> --output_dir=<fold_out>

# 4. 算指标（比较设计结构与预测结构）
#    见 experiments/paper_rfd3/70_metrics_geometry.py
```

**所有 rfd3 参数用 hydra 语法 `arg=value`（不是 `--arg value`）。**

### B.2 逐实验的标准命令（论文规模）

#### 实验 1 — 无条件单体（§3 / Fig. S1c）

```bash
rfd3 design \
  out_dir=out/exp1/eta_1.5 \
  inputs=specs/exp1_eta_1.5.json \
  ckpt_path=~/.foundry/checkpoints/rfd3_latest.ckpt

# 规格（8 条 ProteinMPNN 序列/骨架）
# {"uncond_L100": {"length": 100, "is_non_loopy": true},
#  "uncond_L150": {"length": 150, "is_non_loopy": true},
#  "uncond_L200": {"length": 200, "is_non_loopy": true}}
```

#### 实验 2 — 蛋白结合蛋白（§3.1）

```bash
# 400 条骨架 = diffusion_batch_size 8 × n_batches 50
rfd3 design \
  out_dir=out/exp2/pdl1 \
  inputs=specs/pdl1.json \
  ckpt_path=~/.foundry/checkpoints/rfd3_latest.ckpt \
  diffusion_batch_size=8 n_batches=50

# 规格
# {"pdl1": {"dialect": 2,
#           "infer_ori_strategy": "hotspots",
#           "input": "5o45_cropped.pdb",
#           "contig": "50-120,/0,A17-131",
#           "select_hotspots": {"A56": "CG,OH", "A115": "CG,SD", "A123": "CD2,OH"},
#           "is_non_loopy": true}}
```

五个靶点：PD-L1、InsulinR、IL-7Ra、Tie2（PDB 2GY5）、IL-2Ra。
**权威定义在仓库里**：`models/rfd3/configs/datasets/val/val_examples/bcov_ppi_easy_medium_with_ori.yaml`
（原子级热点）与 `bpem_ori_hb.yaml`（氢键条件版）；论文的五个靶点就是训练配置里显式排除的
五个 PDB ID（`5o45` / `4zxb` / `2gy5` / `1z92` / `3di3`）。
靶点结构需要从 RCSB 重新裁剪（benchmark 文件被作者重编号过），详见
[`DATASETS.md` §2](./DATASETS.md)。

#### 实验 3 — DNA 结合蛋白（§3.2）

```bash
# 每个靶点 400 条骨架 × 2 个设置（rigid / diffused）
for setting in rigid diffused; do
  rfd3 design \
    out_dir=out/exp3/${setting} \
    inputs=specs/exp3_dna_${setting}.json \
    ckpt_path=~/.foundry/checkpoints/rfd3_latest.ckpt \
    diffusion_batch_size=8 n_batches=50
done

# rigid   : "select_fixed_atoms": {"A1-12": "ALL"}   DNA 构象取自晶体（论文的 +）
# diffused: "select_fixed_atoms": {"A1-12": ""}      DNA 构象联合采样（论文的 −）
```

靶点：7RTE、7N5U、7M5W。

#### 实验 4 — 小分子结合蛋白（§3.3）

```bash
# (a) 配体刚体固定
rfd3 design out_dir=out/exp4/fixed \
  inputs=specs/exp4_sm_fixed.json ckpt_path=<ckpt> \
  diffusion_batch_size=8 n_batches=50

# (b) 配体坐标一起扩散 + RASA 埋藏 + CFG 2.0
rfd3 design out_dir=out/exp4/diffused \
  inputs=specs/exp4_sm_diffused.json ckpt_path=<ckpt> \
  diffusion_batch_size=8 n_batches=50 \
  inference_sampler.use_classifier_free_guidance=True \
  inference_sampler.cfg_scale=2.0
```

#### 实验 5 — 酶设计 AME benchmark（§3.4）

**AME 协议的完整定义来自 RFdiffusion2 的论文（Nature Methods 2025, s41592-025-02975-x），
RFD3 论文的 §3.5 就是引用它。原文描述的建库与评估流程：**

1. 取 **M-CSA**（Mechanism and Catalytic Site Atlas）里的全部酶，
   用 **PARITY 数据库**交叉比对，筛出所有反应物都在结构里的条目；
2. 再加质量过滤 → 得到 **41 个案例**；
3. 对 M-CSA 标注的每个催化残基，**随机取残基的一个子图**作为催化原子；
4. 每个案例 **生成 100 条骨架**，用 **LigandMPNN 配 8 条序列**，再用 **Chai-1 折叠**；
5. **成功判据**：至少一条序列的 **5 个 Chai-1 diffusion sample 中有一个**，
   把生成 motif 残基的骨架对齐到预测坐标后，**所有原子侧链 RMSD < 1.5 Å**，
   且配体与预测的骨架原子**没有 clash**。

```bash
rfd3 design \
  out_dir=out/exp5/ame \
  inputs=specs/exp5_ame.json \
  ckpt_path=<ckpt> \
  diffusion_batch_size=8 n_batches=13        # 13 × 8 ≈ 每案例 100 条

# 每个案例的规格（unindex 是 RFD3 的关键机制：序列位置交给模型自己找）
# {"M0097_1ctt": {"input": "ame/M0097_1ctt.pdb",
#                 "ligand": "LIG",
#                 "unindex": "A108,A139,A152,A156",
#                 "length": "180-200",
#                 "select_fixed_atoms": {"A108": "ND2,CG", "A139": "OG,CB,CA"}}}
```

AME 案例清单与代码在 **https://github.com/RosettaCommons/RFdiffusion2/**。

RFD2 / RFdiffusionAA 作为对照基线，用官方 apptainer 镜像跑：

```bash
# RFdiffusion2（RFD2 对照 + AME 原始结果）
git clone https://github.com/RosettaCommons/RFdiffusion2/
cd RFdiffusion2 && export PYTHONPATH="$PWD" && python setup.py

apptainer exec --nv rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif \
  rf_diffusion/benchmark/pipeline.py \
  --config-name=open_source_demo \
  stop_step=''          # 一路跑到 LigandMPNN + Chai-1

# RFdiffusionAA（小分子基线）
# https://github.com/baker-laboratory/RoseTTAFold-All-Atom
```

#### 实验 6 — 对称设计（Fig. 2g / Fig. S6）

```bash
rfd3 design out_dir=out/exp6/C5 inputs=specs/sym_C5.json ckpt_path=<ckpt>

# {"sym_C5": {"length": 100, "is_non_loopy": true, "symmetry": {"id": "C5"}}}
# C2 对称酶额外加 "is_symmetric_motif": true
```

#### 实验 7 — 原子级条件控制（Fig. 2d/2e/2f）

```bash
# (a) 氢键供体/受体条件（需 HBPLUS）
rfd3 design out_dir=out/exp7/hbond inputs=specs/exp7_hbond.json ckpt_path=<ckpt>

# (b) RASA 埋藏条件
#     "select_buried":  {"IAI": "C22,C23,..."}   应埋藏的原子
#     "select_exposed": {"IAI": "..."}           应暴露的原子

# (c) 质心条件：用 ori_token 指定生成蛋白相对靶点的质心
#     {"dna_comA": {"input": "...", "contig": "A1-12,/0,110-130",
#                   "length": "110-130", "ori_token": [-8, -8, 8]}}
```

#### 实验 8 — 推理速度（Fig. 1d）

论文在 **A6000** 上按长度扫 runtime，对比 RFD1 / RFD2 / RFD3：

```bash
for L in 50 100 150 200 250 300; do
  /usr/bin/time -v rfd3 design \
    out_dir=out/exp8/L${L} \
    inputs=specs/L${L}.json \
    ckpt_path=<ckpt> \
    diffusion_batch_size=1 n_batches=1 \
    inference_sampler.num_timesteps=200
done

# RFD1 基线：https://github.com/RosettaCommons/RFDiffusion
# RFD2 基线：https://github.com/RosettaCommons/RFdiffusion2
```

#### 实验 9 — §4 实验的计算部分

```bash
# DNA 结合物两阶段（论文靶序列 CGAGAACATAGTCG）
# 阶段 1：固定 AF3 预测的靶 DNA 构象采样
rfd3 design out_dir=out/exp9/dbrfd3/stage1 inputs=specs/dbrfd3_stage1.json ckpt_path=<ckpt>
# 阶段 2：固定与 DNA 接触的 motif，重采样其余骨架
rfd3 design out_dir=out/exp9/dbrfd3/stage2 inputs=specs/dbrfd3_stage2.json ckpt_path=<ckpt>

# 半胱氨酸水解酶：Ulp-1（PDB 1EUV）的 Cys-His-Asp + Gln + 底物 TI1 几何
rfd3 design out_dir=out/exp9/cys inputs=specs/cys_hydrolase.json \
  ckpt_path=<ckpt> diffusion_batch_size=8 n_batches=24   # ≈ 190 个设计
```

### B.3 外部工具的标准命令

```bash
# ---- 序列设计（MPNN，argparse 风格）----
mpnn --model_type ligand_mpnn \
     --checkpoint_path ligandmpnn_v_32_010_25.pt \
     --is_legacy_weights True \
     --structure_path design.cif \
     --out_directory seq_out \
     --write_fasta True --write_structures True \
     --batch_size 8 --number_of_batches 1

# ---- 结构预测：AlphaFold3 ----
python run_alphafold.py \
  --json_path=fold_input.json \
  --model_dir=/path/to/af3_params \
  --db_dir=/path/to/af3_databases \
  --output_dir=/path/to/af_output
# 可拆分 CPU/GPU：--norun_inference / --norun_data_pipeline
# CPU-only（慢约 100×）：--jax_backend=cpu --flash_attention_implementation=xla

# ---- 结构预测：Chai-1（论文酶设计用的）----
pip install chai_lab
chai-lab fold --use-msa-server input.fasta output_folder
# 关键参数：--num-diffn-timesteps 200 --num-diffn-samples 5 --seed <n>
# 注意：output_folder 必须为空目录

# ---- 结构预测：RF3（Foundry 自带，AF3 的开源等价物）----
rf3 fold inputs=fold_input.json out_dir=fold_out ckpt_path=rf3.ckpt

# ---- 新颖性（Fig. S4b）----
foldseek easy-search design.pdb pdb100_db out.m8 tmp --format-output "query,target,alntmscore"

# ---- ΔΔG（Fig. S4c，DDGnoRepack，需 Rosetta license）----
Rosetta/main/source/bin/ddg_monomer.linuxgccrelease \
  -in:file:s complex.pdb -ddg:iterations 1 -ddg:repack_rounds 0

# ---- DNA 沟宽（Fig. S3g）----
x3dna-dssr -i=design.pdb --analyze

# ---- 氢键（Fig. 2d）----
hbplus complex.pdb
```

---

## C. 一句话总结

RFD3 侧很轻（168 M 参数，一张 A6000 即可）；
论文的算力规模实际上由**约 7.5 万次 AF3 / Chai-1 折叠**决定，
其中 PPI（8 000）、DNA（9 600）、小分子（25 600，含基线 12 800）、
酶（32 800）四块是大头。复现时把折叠做成批处理队列（AF3/Chai 一个进程吃多个输入，
避免反复加载模型）是省时间的关键。
