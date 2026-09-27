# 单张 RTX A6000 跑完全部复现任务 —— 执行手册

> 适用：只有 **1 张 A6000（49140 MiB）** 可用，项目在 `/backup01/zzj/protein/foundry`，
> 数据在 `/dev/shm/{pdb_mirror, ccd_mirror, pdb_metadata_latest}`。
> 多卡版本见 `README.md` / `COMMANDS.md`。

---

## 0. 先看结论

| 问题 | 答案 |
|---|---|
| 单卡能跑完全部吗？ | **骨架生成 + 序列设计 + NMF 微调：能。** 论文规模的**折叠**（7.5 万次结构预测）：不能，要数周 |
| 时间花在哪？ | **折叠 ≫ 微调 > 骨架生成**。RFD3 采样很快，AF3/RF3/Chai-1 才是瓶颈 |
| 单卡最大的提速手段 | ① 显存吃满（大 batch）② CPU worker 拉满 ③ 数据在 /dev/shm |
| 建议顺序 | 探显存 → 冒烟 → **NMF 微调（你的主线）** → 论文 §3 半规模 → 折叠分批 |

实测参照（本仓库历史日志）：**L≈200 的近原生设计，`diffusion_batch_size=8` 单次约 42 秒**
（含 ~25 秒进程启动 + 模型加载；纯采样约 8 条/15-20 秒）。这是 5070 Ti Laptop 的数字，
A6000 更快、显存大 3 倍，实际会更好。

---

## 1. 三个提速杠杆（按性价比排序）

### 杠杆 1：把显存吃满 —— `05_probe_vram.sh`

手册里的 `diffusion_batch_size=8` 是论文为了稳妥写的。扩散采样的每条样本**互相独立**，
batch 只影响吞吐、不影响采样分布，A6000 49 GB 通常能开好几倍。

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3
source ./env.sh && source ./lib.sh

# 用哪张卡（你的是 3 号）。约 20-40 分钟，会真跑若干次最小任务
GPU_ID=3 ./05_probe_vram.sh

# 只想先看会执行什么，不真跑：
GPU_ID=3 ./05_probe_vram.sh --dry-run
```

它做的事：递进试 `diffusion_batch_size`（8→16→24→32→48→64→96）和训练侧的
`(batch, crop_size, max_atoms_in_crop)`，每点用 `nvidia-smi` 轮询记录**峰值显存**，
到第一个 OOM 停止，把推荐值写进 `$OUT/vram_profile.json`。

之后 `source ./env.sh` 会**自动读回**这个文件，所有实验脚本无需改参数就用了最优配置：

```bash
source ./env.sh && source ./lib.sh && check_env   # 看硬件 + 加载的 batch
```

想手工覆盖就直接设环境变量，优先级更高：
`DIFFUSION_BATCH_SIZE=24 CROP_SIZE=384 MAX_ATOMS_IN_CROP=3840 ./run_all.sh`

### 杠杆 2：CPU worker 拉满

32 核机器上，DataLoader 的 worker 是"喂饱 GPU"的关键（尤其训练时每个 batch 都要
做 cif 解析 + transform）。

```bash
NUM_WORKERS=12 PREFETCH=6 ./run_all.sh
```

序列设计 / 几何指标这类纯 CPU 阶段用 `N_WORKERS`（默认 `nproc/2`）。

### 杠杆 3：数据放在 /dev/shm + TMPDIR 挪出 /tmp

`env.sh` 已自动探测 `/dev/shm/pdb_mirror`、`/dev/shm/ccd_mirror`、`/dev/shm/pdb_metadata_latest`，
并把 `TMPDIR` 指到 `$OUT/_worker_tmp`。

⚠️ **`TMPDIR` 千万别留在 `/tmp`**：torch 的 DataLoader worker 每 epoch 会在
`$TMPDIR` 下泄漏一个 `pymp-xxxx` 目录，小 tmpfs 几百 epoch 后写满，然后报
`OSError: [Errno 28] ... '/tmp/pymp-xxxx'`，**batch 不再送达、主进程看起来像卡住**。
`env.sh` 已经处理，长跑时再加 `PERSISTENT_WORKERS=1`（默认已开）从根上堵住。

---

## 2. 推荐执行顺序（含时间预算）

### 步骤 0 —— 环境自检（1 分钟）

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3
source ./env.sh && source ./lib.sh
check_env
```

会打印：GPU 表 + 可用卡数、`N_WORKERS`、RFD3/MPNN/RF3 可执行文件、权重路径、
`/dev/shm` 数据是否就位、当前生效的 batch 与采样参数。**有任何一项是 `!` 就先解决它。**

### 步骤 1 —— 探显存（20-40 分钟，只做一次）

见杠杆 1。这一步的收益是剩下所有任务都省时间。

### 步骤 2 —— 冒烟跑通全链路（0.5-2 小时）

```bash
SINGLE_GPU=1 SCALE=quick ./run_all.sh
```

每个条件 8 条骨架，验证 9 个实验 + 序列设计 + 折叠 + 汇总能串起来。
产出 `out/reports/paper_experiments_report.md`（第一张表就是"论文数字 vs 本次复现"）。

### 步骤 3 —— NMF 微调 sweep（你的主线，**建议优先做**）

这条线不依赖 `experiments/`，脚本在 `models/rfd3/scripts/`。默认路径已经指向
`/dev/shm`，只要覆盖几项：

```bash
cd /backup01/zzj/protein/foundry

GPU=3 \
PYTHON=/backup01/zzj/rc-cu128/bin/python \
LOG_ROOT=/backup01/zzj/protein/foundry/logs/train_nmf_zkp_pdb \
HBPLUS_PATH=/backup01/zzj/protein/HBPLUS/hbplus/hbplus \
DATA=/dev/shm/pdb_metadata_latest \
PARQUET=/dev/shm/pdb_metadata_latest \
PDB_MIRROR=/dev/shm/pdb_mirror \
CCD_MIRROR_PATH=/dev/shm/ccd_mirror CCD_PATH=/dev/shm/ccd_mirror \
NUM_WORKERS=12 PREFETCH=6 \
N_EXAMPLES=128 MAX_EPOCHS=580 SEED=42 \
INCLUDE_BASELINE=1 LAYER_SET=encoder \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TMPDIR=/backup01/zzj/protein/foundry/logs/_worker_tmp \
bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh 2>&1 | tee nmf_sweep.log
```

**显存吃满**：上面 `DIFFUSION_BS` / `CROP_SIZE` / `MAX_ATOMS` 没给就还是论文口径
（4 / 256 / 1920）。按探测结果提高到 e.g. `DIFFUSION_BS=16 CROP_SIZE=384 MAX_ATOMS_IN_CROP=3840`：

```bash
DIFFUSION_BS=16 CROP_SIZE=384 MAX_ATOMS=3840 NUM_WORKERS=12 ...
```

> ⚠️ **训练 batch 与推理 batch 性质不同**：推理 batch 改了不影响结果；训练 batch
> 改变了梯度平均，**baseline 与所有 NMF/LoRA 变体必须用同一个值**，否则对比无效。
> 一旦定下就不要再改，中途改了就整轮重跑。

`LAYER_SET` 选哪一组：`encoder`（6 个 job，输入整形层）/ `proj`（6）/ `head`（2）/
`all`（14）。**建议先 `LAYER_SET=one LAYER=token_initializer.process_s_init.1` 单点试一轮**
（1 个 job），确认能出 `val_metrics/validation_output_all_epochs.csv` 再铺全量。

中途 kill 也能出数据：

```bash
python models/rfd3/scripts/report_partial_sweep.py \
  --out logs/train_nmf_zkp_pdb/sweep_nmf_zkp_pdb_<STAMP>
```

### 步骤 4 —— 论文 §3 骨架生成（半规模 3-8 小时 / 全规模 1-2 天）

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3
SINGLE_GPU=1 SCALE=half WITH_SEQ=1 WITH_FOLD=0 ./run_all.sh
```

**先 `WITH_FOLD=0`**：骨架 + 序列设计很便宜，折叠很贵。等骨架出齐、确认没问题再折叠。

只要某几个实验：

```bash
SINGLE_GPU=1 SCALE=half WITH_FOLD=0 ONLY="2 3 4" ./run_all.sh   # PPI / DNA / 小分子
```

### 步骤 5 —— 折叠（按需，单卡这是瓶颈）

- 全量折叠（7.5 万次预测）单卡要**数周**，不要一次铺开。
- 建议：① 只折叠通过几何初筛的子集；② 分批挂后台跑；③ 或者用 `FOLD_BACKEND=none`
  先出骨架，把折叠放到别的机器/别的时段。

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3
SINGLE_GPU=1 FOLD_BACKEND=rf3 WITH_FOLD=1 SCALE=half ./run_all.sh
```

---

## 3. 单卡并行度怎么设

一张卡上**不要**同时跑两个吃显存的任务 —— 会互相 OOM。正确做法是：

```bash
SINGLE_GPU=1 MAX_PARALLEL_GPUS=1 ./run_all.sh     # 所有条件串行铺在这一张卡上
```

`env.sh` 的 `SINGLE_GPU=1` 会自动挑空闲显存最多的卡，并把 `MAX_PARALLEL_GPUS` 锁成 1。
要指定卡就 `GPUS=3`。

CPU 侧的并行不受影响：`N_WORKERS`（MPNN/指标）和 `NUM_WORKERS`（DataLoader）
照常拉满，它们和 GPU 是重叠利用的。

---

## 4. 常见坑（都在脚本里处理了，但知道一下）

| 现象 | 原因 | 处理 |
|---|---|---|
| `OSError: [Errno 28] ... pymp-xxxx`，之后像卡住 | DataLoader worker 的 temp 目录泄漏，`/tmp` 写满 | `env.sh` 把 `TMPDIR` 挪到 `$OUT/_worker_tmp`；加 `PERSISTENT_WORKERS=1` |
| `CUDA out of memory` | batch / crop / atoms 太大 | 降 `DIFFUSION_BATCH_SIZE`（推理）或 `DIFFUSION_BS`+`CROP_SIZE`（训练）；跑 `05_probe_vram.sh` 找边界 |
| `nvidia-smi` 报"driver 通信失败"但命令存在 | 错误被写到 **stdout**，被当成显存值 | `05_probe_vram.sh` 已校验返回的是纯数字 |
| 训练启动即退出、0 个 step | `MAX_EPOCHS=1` 时 ckpt 载入后 `current_epoch=1>=1`，训练循环整体跳过 | 想真的训就要 `MAX_EPOCHS>=2` |
| 图/表里 lDDT 是空的 | 验证列名不是 `val/mean_lddt`，而是 `val/pdb_holdout/lddt.mean_lddt_protein` | 仓库根的 `fix_nmf_zkp_eval_errors.patch` 修的就是这个，确认已应用 |
| 结果写到 `/dev/shm` 里被吃掉内存 | `OUT` 落在 tmpfs | `env.sh` 检测到 tmpfs 会自动改到 `$HOME/foundry_experiments`；也可 `OUT=/backup01/$USER/rfd3_out` |

---

## 5. 一句话的最优配置

```bash
cd /backup01/zzj/protein/foundry/experiments/paper_rfd3

# 一次性：探出这张卡的显存上限
GPU_ID=3 ./05_probe_vram.sh

# 之后所有任务都用同一套前缀
source ./env.sh && source ./lib.sh          # 自动加载 vram_profile.json
export SINGLE_GPU=1                          # 单卡、串行、吃满 CPU
export NUM_WORKERS=12 PREFETCH=6
export PERSISTENT_WORKERS=1

check_env                                     # 确认生效
SCALE=quick ./run_all.sh                      # 冒烟
SCALE=half  WITH_FOLD=0 ./run_all.sh          # 出骨架 + 序列
```
