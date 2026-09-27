# 单张 RTX A6000 跑完论文复现实验 —— 执行手册

> 适用：只有 **1 张 A6000（49140 MiB，如 GPU 3）** 可用，
> 项目在 `/backup01/zzj/protein/foundry`，
> 数据在 `/dev/shm/{pdb_mirror, ccd_mirror, pdb_metadata_latest}`。
> 多卡版本见 `README.md` / `COMMANDS.md`。

---

## 0. 你要跑的实验 ↔ 脚本对照

| 论文位置 | 实验 | 脚本 |
|---|---|---|
| §3 / Fig. S1c | 无条件单体 + η 扫描 | `10_exp1_unconditional.sh` |
| §3.1 / Fig. 3a | 蛋白结合蛋白（PPI） | `11_exp2_ppi.sh` |
| §3.2 / Fig. 3b | DNA 结合蛋白（rigid vs diffused） | `12_exp3_dna.sh` |
| §3.3 / Fig. 3c | 小分子结合蛋白 | `13_exp4_small_molecule.sh` |
| §3.4 / Fig. 3d | 酶设计 AME benchmark | `14_exp5_enzyme.sh` ⚠️ 需补输入 |
| Fig. 2g / S6 | 对称设计 D2/C3/C5/C7 | `15_exp6_symmetry.sh` |
| Fig. 2d/2e/2f | 氢键 / RASA / 质心条件 | `16_exp7_conditioning.sh` |
| Fig. 1d | 推理速度标定 | `17_exp8_speed.sh` |
| §4 / Fig. 4 | 湿实验的 in silico 部分 | `18_exp9_wetlab_insilico.sh` ⚠️ 需补 motif |

实验 1-4、6-8 的输入由 `01_prepare_inputs.py` 全自动准备。
**实验 5 需要 41 个 AME 案例定义，实验 9 需要 motif 文件** —— PDF 里没有补充方法，
缺了会跳过（脚本会明确提示缺什么）。

---

## 1. 一条命令跑完全部

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3 && \
GPUS=3 SINGLE_GPU=1 SCALE=quick PROBE=1 \
./run_paper_seq.sh 2>&1 | tee paper_run_$(date +%F_%H%M).log
```

这条命令按顺序做 6 件事：

1. **环境自检** —— GPU / rfd3 / mpnn / 权重 / `/dev/shm` 数据是否就位
2. **显存探测**（`PROBE=1`）—— 实测这张卡能开多大 batch，写进 `out/vram_profile.json`
3. **准备输入** —— 从仓库配置导出论文 benchmark 定义 + 整理 PDB 结构
4. **跑 9 个实验** —— 骨架采样 → 序列设计 → 折叠，逐个实验推进
5. **几何指标** —— RMSD / 界面 / RASA / 氢键 / clash（多进程并行）
6. **汇总报告** —— `out/reports/paper_experiments_report.md`，第一张表就是
   「论文口径指标 | 论文报告数字 | 本次复现」三列对照

跑完打印一份**实验状态表**，并告诉你还缺哪些手工输入。

### 分档推进（强烈建议）

单卡上不要一上来就跑论文规模。**分三次**，每次 `SCALE` 提一级：

```bash
# 第 1 次：冒烟，验证全链路通（约 1 小时）
GPUS=3 SINGLE_GPU=1 SCALE=quick PROBE=1 ./run_paper_seq.sh

# 第 2 次：半规模，指标有统计意义（约 1-3 天）
GPUS=3 SINGLE_GPU=1 SCALE=half ./run_paper_seq.sh

# 第 3 次：论文原规模（折叠单卡要数周，见 §3）
GPUS=3 SINGLE_GPU=1 SCALE=paper WITH_FOLD=0 ./run_paper_seq.sh
```

### 分阶段跑（可断点续跑）

`run_paper_seq.sh` 把流程拆成带完成标记的阶段，中断后用 `RESUME=1` 接着跑：

```bash
PHASE=sample ./run_paper_seq.sh    # 只采骨架
PHASE=seq    ./run_paper_seq.sh    # 只做序列设计
PHASE=fold   ./run_paper_seq.sh    # 只做折叠
PHASE=report ./run_paper_seq.sh    # 只出指标 + 报告
RESUME=1 ./run_paper_seq.sh        # 跳过已完成的阶段
```

单卡上「先出全部骨架、确认质量后再统一折叠」比「每个实验一路做完」更稳妥。

---

## 2. 三个提速杠杆（按性价比）

### 杠杆 1：把显存吃满 —— `05_probe_vram.sh`

手册里的 `diffusion_batch_size=8` 是为"小卡也能跑"写的保守值。扩散采样的每条样本
**互相独立**，batch 只影响吞吐、**不改采样分布**，49 GB 卡上白白浪费 2/3。

```bash
GPU_ID=3 ./05_probe_vram.sh              # 20-40 分钟，只做一次
GPU_ID=3 ./05_probe_vram.sh --dry-run    # 先看会执行什么，不真跑
```

递进试 `diffusion_batch_size ∈ {8,16,24,32,48,64,96}`，每点真跑一次最小 design，
用 `nvidia-smi` 每 2 秒轮询记录**相对任务启动前的峰值增量**（卡上有别人进程也能用），
到第一个 OOM 停。推荐值写进 `$OUT/vram_profile.json`，之后 `source ./env.sh`
会**自动读回**，所有脚本零改动生效。

手工覆盖（优先级更高）：`DIFFUSION_BATCH_SIZE=32 ./run_paper_seq.sh`

### 杠杆 2：CPU worker 拉满

32 核机器上，DataLoader worker 是"喂饱 GPU"的关键（atomworks 的特征化很重）：

```bash
NUM_WORKERS=12 PREFETCH=6 ./run_paper_seq.sh
```

序列设计 / 几何指标这类纯 CPU 阶段用 `N_WORKERS`（默认 `nproc/2`）。

### 杠杆 3：`TMPDIR` 挪出 `/tmp`

⚠️ **这是长跑杀手**：torch 的 DataLoader worker 每 epoch 会在 `$TMPDIR` 下泄漏一个
`pymp-xxxx` 目录，小 tmpfs 几百 epoch 后写满，然后报
`OSError: [Errno 28] ... '/tmp/pymp-xxxx'`，**batch 不再送达、主进程看起来像卡住**。

`env.sh` 已把 `TMPDIR` 指到 `$OUT/_worker_tmp`，并默认 `PERSISTENT_WORKERS=1`
（worker 不再每 epoch 重生，从根上堵住）。数据在 `/dev/shm` 上也已自动识别。

---

## 3. 时间预算（单卡，粗估）

| 阶段 | quick | half | paper |
|---|---|---|---|
| 骨架采样 | ~10 分钟 | 3-8 小时 | 6-16 小时 |
| 序列设计（MPNN） | ~5 分钟 | 1-3 小时 | 2-6 小时 |
| **折叠（结构预测）** | ~20 分钟 | **1-3 天** | **数周** ❌ |
| 指标 + 汇总 | 分钟级 | 小时级 | 小时级 |

参照：本仓库历史日志里 `diffusion_batch_size=8, n_batches=1` 的近原生 design
单次约 **42 秒**（含 ~25 秒进程启动 + 模型加载），跑在 5070 Ti Laptop 上 ——
A6000 更快、显存大 3 倍。

**结论：RFD3 采样很快，AF3/RF3 折叠才是瓶颈。** 单卡上 paper 规模的全量折叠
（论文约 7.5 万次预测）不现实，建议：

- 先 `WITH_FOLD=0` 出全部骨架 + 序列，确认质量；
- 再只折叠通过几何初筛的子集；
- 或用 `FOLD_BACKEND=none` 交给别的机器/时段分批做。

---

## 4. 单卡并行度原则

一张卡上**不要**同时跑两个吃显存的任务 —— 会互相 OOM。正确做法：

```bash
SINGLE_GPU=1 ./run_paper_seq.sh      # 自动挑空闲显存最多的卡 + 并行度锁 1
GPUS=3      ./run_paper_seq.sh       # 或者显式指定卡
```

`SINGLE_GPU=1` 会把 `MAX_PARALLEL_GPUS` 锁成 1，所有条件**串行**铺在这一张卡上。
CPU 侧的并行不受影响（`N_WORKERS` / `NUM_WORKERS` 照常拉满，与 GPU 重叠利用）。

---

## 5. 需要你手工补的输入

PDF 里没有 Supplemental Methods，这两项必须自己补：

**实验 5 —— AME 的 41 个活性位点案例**
协议出自 RFdiffusion2 的 Nature Methods 论文（Ahern et al., doi:10.1038/s41592-025-02975-x），
代码与案例清单在 <https://github.com/RosettaCommons/RFdiffusion2/>。
模板由 `01_prepare_inputs.py` 生成到
`out/inputs/targets/ame_cases.json`（含 `unindex` / `fixed_atoms` / `n_islands` 字段），补齐后重跑实验 5。

**实验 9 —— DBRFD3 与半胱氨酸水解酶的 motif**
放到 `out/inputs/motifs/`。缺文件时脚本会用占位文件跑通流程，但指标无意义。

**PPI 的另外 3 个靶点**（Tie2 / IL-7Ra / IL-2Ra）：`02_extract_repo_benchmarks.py`
已从仓库配置里导出定义，`01_prepare_inputs.py` 会尝试从镜像重建；
仓库自带 PD-L1 与 InsulinR 两个裁剪结构。IL-7Ra 的 benchmark 定义确实缺失，需手工补。

---

## 6. 常见坑（脚本大多已处理，知道一下）

| 现象 | 原因 | 处理 |
|---|---|---|
| `OSError: [Errno 28] ... pymp-xxxx`，之后像卡住 | worker temp 目录泄漏、`/tmp` 满 | `env.sh` 把 `TMPDIR` 挪到 `$OUT/_worker_tmp`；`PERSISTENT_WORKERS=1` 默认开 |
| `CUDA out of memory` | batch 太大 | 降 `DIFFUSION_BATCH_SIZE`（推理）；跑 `05_probe_vram.sh` 找边界 |
| `nvidia-smi` 报驱动通信失败但命令存在 | 错误写进 **stdout**，被当成显存值 | `05_probe_vram.sh` 已校验返回纯数字 |
| `ModuleNotFoundError: No module named 'Bio'` | 缺 biopython | 装进运行环境：`$PY -m pip install biopython` |
| `ModuleNotFoundError: No module named 'pandas'` | 选错解释器（在 `(base)` 里跑） | 用 `RC_ENV_BIN` 指到装了 foundry 的环境 |
| 图/表里 lDDT 是空的 | 验证列名是 `val/pdb_holdout/lddt.mean_lddt_protein` 而非 `val/mean_lddt` | 仓库根的 `fix_nmf_zkp_eval_errors.patch` 修的就是这个，确认已应用 |
| 结果写进 `/dev/shm` 吃掉内存 | `OUT` 落在 tmpfs | `env.sh` 检测到 tmpfs 会自动改到 `$HOME/foundry_experiments`；也可 `OUT=/backup01/$USER/rfd3_out` |

---

## 7. 最短路径（照着抄）

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3

# 一次性：探出这张卡的显存上限（20-40 分钟）
GPU_ID=3 ./05_probe_vram.sh

# 冒烟：验证全链路（约 1 小时）
GPUS=3 SINGLE_GPU=1 SCALE=quick ./run_paper_seq.sh

# 半规模：出有统计意义的数据
GPUS=3 SINGLE_GPU=1 SCALE=half ./run_paper_seq.sh

# 报告在 out/reports/paper_experiments_report.md
```
