#!/usr/bin/env bash
# =============================================================================
# RFdiffusion3 论文实验复现 —— 全局配置
#
# 已按 6× RTX A6000（49 GB/卡）+ 大内存机器调好：
#   * 关掉小显存降级（low_memory_mode=False），用论文原版的 diffusion_batch_size=8
#   * 多个实验条件并行铺到多张空闲卡上（lib/gpu_pool.py）
#   * MPNN 序列设计 / 几何指标用多进程并行
#
# 所有变量都可以用环境变量覆盖，例如：
#   GPUS=2,4 ./run_all.sh                     # 只用 2 号和 4 号卡
#   MAX_PARALLEL_GPUS=6 SCALE=paper ./run_all.sh
#   DIFFUSION_BATCH_SIZE=16 ./10_exp1_unconditional.sh
#   OUT=/scratch/$USER/rfd3_paper ./run_all.sh
# =============================================================================
set -Eeuo pipefail

# 某些批处理环境（lsf/slurm 的裸 shell）不会带 HOME，补一下
if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || true)"
fi
export HOME="${HOME:-/root}"

# ---------------------------------------------------------------- 目录布局 ---
# 本文件位于 <repo>/experiments/paper_rfd3/env.sh
PAPER_ROOT="${PAPER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FOUNDRY_ROOT="${FOUNDRY_ROOT:-$(cd "$PAPER_ROOT/../.." && pwd)}"
export PAPER_ROOT FOUNDRY_ROOT

# 产物目录：默认放在仓库里；如果仓库本身在 tmpfs（例如 /dev/shm）就挪到 home，
# 因为上千个 CIF 写进 /dev/shm 会吃内存。
if [[ -z "${OUT:-}" ]]; then
  fstype="$(df -P --output=fstype "$FOUNDRY_ROOT" 2>/dev/null | tail -1 | tr -d ' ' || true)"
  case "$fstype" in
    tmpfs|ramfs) OUT="$HOME/foundry_experiments" ;;
    *)           OUT="$PAPER_ROOT/out" ;;
  esac
fi
INPUTS_DIR="$OUT/inputs"                 # 整理好的输入结构 + 生成的设计规格
LOGS_DIR="$OUT/logs"                     # 每个 (实验, 条件, 种子) 的运行日志
DESIGNS_DIR="$OUT/designs"               # rfd3 design 的输出
SEQS_DIR="$OUT/sequences"                # MPNN 序列设计输出
FOLDS_DIR="$OUT/folds"                   # 结构预测（自洽性）输出
METRICS_DIR="$OUT/metrics"               # 指标 CSV/JSON
REPORTS_DIR="$OUT/reports"               # 汇总报告

# ------------------------------------------------------- Foundry 运行环境 ---
# 已经安装好 rfd3 / mpnn / rf3 / foundry 入口的 Python 环境 bin 目录。
# 若你用的是 venv，改成 <venv>/bin 即可。
export RC_ENV_BIN="${RC_ENV_BIN:-/home/zhangzijian/anaconda3/envs/rc/bin}"
PY="${PY:-$RC_ENV_BIN/python}"
RF_ENV_NAME="${RF_ENV_NAME:-rc}"

# --------------------------------------------------------------- 模型权重 ---
RFD3_CKPT="${RFD3_CKPT:-/media/zzj/Data/rfd3_latest.ckpt}"
RF3_CKPT="${RF3_CKPT:-$HOME/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt}"
MPNN_CKPT_DIR="${MPNN_CKPT_DIR:-$HOME/.foundry/checkpoints}"
PROTEINMPNN_CKPT="${PROTEINMPNN_CKPT:-$MPNN_CKPT_DIR/proteinmpnn_v_48_020.pt}"
LIGANDMPNN_CKPT="${LIGANDMPNN_CKPT:-$MPNN_CKPT_DIR/ligandmpnn_v_32_010_25.pt}"
HBPLUS="${HBPLUS:-/home/zhangzijian/protein/HBPLUS/hbplus/hbplus}"

# =============================================================== 硬件配置 ===
# ------------------------------------------------------------ CPU 并行度 ---
NPROC="$(nproc 2>/dev/null || echo 16)"
# MPNN 序列设计 / 几何指标这类 CPU 任务的进程数。内存充足（500 GB 级）时
# 开到 NPROC/2 通常最划算；MKL 等数值库自身也会多线程，别开太满。
N_WORKERS="${N_WORKERS:-$(( NPROC / 2 > 32 ? 32 : (NPROC / 2 < 4 ? 4 : NPROC / 2) ))}"

# 每个 worker 的线程数：设为 1 可以让 N_WORKERS 个进程真正吃满 CPU，
# 避免 N 个进程 × M 线程互相抢。想单进程跑快就 OMP_NUM_THREADS=$NPROC。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

# --------------------------------------------------------------- 多卡调度 ---
# GPUS：只用这些卡（逗号分隔）。留空 = 自动挑空闲的。
GPUS="${GPUS:-}"
# 同时跑几个任务（= 占用几张卡）。auto = 按当前满足显存门槛的卡数决定。
MAX_PARALLEL_GPUS="${MAX_PARALLEL_GPUS:-auto}"
# 显卡空闲显存低于这个值（MiB）就不再往上排任务。
GPU_POOL_MIN_FREE="${GPU_POOL_MIN_FREE:-25000}"
# 单个 RFD3 任务预计占用多少显存（MiB）。A6000 48 GB 上 batch=8、L≈200 大约 15-25 GB。
GPU_JOB_MEM="${GPU_JOB_MEM:-25000}"

# ------------------------------------------------------------ 显存与批量 ---
# A6000 有 48 GB，不需要低显存降级。
LOW_MEMORY="${LOW_MEMORY:-False}"
if [[ "$LOW_MEMORY" == "True" ]]; then LOW_MEM_FLAG="--low_memory"; else LOW_MEM_FLAG=""; fi
# 论文默认 diffusion_batch_size=8。A6000 上可以往上抬（16/32）来缩短墙钟时间，
# 但**只影响吞吐，不影响采样分布**；要与论文数字严格对齐就保持 8。
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-8}"
# 论文用 2 次 recycle（checkpoint 默认）；显存够也可以留着。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ------------------------------------------------------------ 论文采样参数 ---
# 论文 Fig. S4 脚注："step scale η = 1.5, noise level γ0 = 0.6, 200 denoising steps"
STEP_SCALE="${STEP_SCALE:-1.5}"          # η
GAMMA_0="${GAMMA_0:-0.6}"                # γ0
NUM_TIMESTEPS="${NUM_TIMESTEPS:-200}"
N_RECYCLE="${N_RECYCLE:-2}"
CFG_SCALE="${CFG_SCALE:-2.0}"            # 小分子 diffused-ligand 条件用 2.0

# ------------------------------------------------------------------ 规模 ---
# quick = 冒烟测试；half = 半规模；paper = 论文规模
SCALE="${SCALE:-quick}"
case "$SCALE" in
  quick) N_BACKBONES="${N_BACKBONES:-8}"    ; MPNN_SEQS="${MPNN_SEQS:-2}" ;;
  half)  N_BACKBONES="${N_BACKBONES:-200}"  ; MPNN_SEQS="${MPNN_SEQS:-4}" ;;
  paper) N_BACKBONES="${N_BACKBONES:-400}"  ; MPNN_SEQS="${MPNN_SEQS:-4}" ;;
  *) echo "未知 SCALE=$SCALE（可选 quick|half|paper）" >&2; exit 1 ;;
esac
SEEDS="${SEEDS:-0}"

# 论文里 LigandMPNN 每条骨架配 8 条序列（小分子 / 酶），
# 蛋白-蛋白与 DNA 用 4 条（ProteinMPNN / LigandMPNN）。
LIGAND_MPNN_SEQS="${LIGAND_MPNN_SEQS:-8}"

# AME benchmark：论文口径是每个案例 100 条骨架
# （来源：RFdiffusion2 论文 Ahern et al., Nat Methods 2025 的 AME benchmark 一节）
ENZYME_BACKBONES_PER_CASE="${ENZYME_BACKBONES_PER_CASE:-100}"

# --------------------------------------------------------- 论文规模速查 ---
#   §3   无条件单体   96 个（长度 100-250）        8 条 ProteinMPNN 序列
#   §3.1 PPI          5 靶点 × 400 骨架            4 条 ProteinMPNN 序列
#   §3.2 DNA          3 靶点 × 400 骨架 × 2 设置   4 条 LigandMPNN 序列
#   §3.3 小分子       4 配体 × 400 骨架 × 2 条件   8 条 LigandMPNN 序列
#   §3.4 酶 AME       41 案例 × 100 骨架           8 条 LigandMPNN 序列（Chai-1 评估）
#   合计约 1.2 万条骨架 → 约 7.5 万次 AF3 / Chai-1 结构预测
#   推理速度标定在 NVIDIA A6000 上完成（Fig. 1d）—— 与本机同型号，可直比
#   完整设备规模 + 标准命令：见 COMMANDS.md

# ------------------------------------------------------------- 目标清单 ---
# 蛋白-蛋白结合物（论文 §3.1 / Fig. 3a）。用 02_extract_repo_benchmarks.py 导出定义，
# 再由 01_prepare_inputs.py 从 PDB 重建。
PPI_TARGETS="${PPI_TARGETS:-pdl1 insulinr tie2 il2ra}"

# DNA 结合物（论文 §3.2 / Fig. 3b）：三个训练集外序列。
DNA_TARGETS="${DNA_TARGETS:-7rte 7n5u 7m5w}"

# 小分子结合物（论文 §3.3 / Fig. 3c）：FAD、SAM 常见；IAI、OQO 罕见。
SM_LIGANDS="${SM_LIGANDS:-FAD SAM IAI OQO}"

# 对称性（Fig. 2g / Fig. S6c-d）
SYMMETRY_IDS="${SYMMETRY_IDS:-D2 C3 C5 C7}"

# 无条件单体长度（§3 正文：100-200 aa）
UNCOND_LENGTHS="${UNCOND_LENGTHS:-100 150 200}"

# ------------------------------------------------------------------ 工具 ---
# 结构预测后端：rf3（Foundry 自带）/ af3 / chai / none（只生成输入不折叠）
# 6 张 A6000 上 AF3 与 Chai-1 都能一块卡一个进程地铺开。
FOLD_BACKEND="${FOLD_BACKEND:-rf3}"
# 折叠阶段并行几张卡
FOLD_PARALLEL_GPUS="${FOLD_PARALLEL_GPUS:-$MAX_PARALLEL_GPUS}"

# 汇总报告语言
REPORT_LANG="${REPORT_LANG:-zh}"

# ------------------------------------------------------------ 目录初始化 ---
mkdir -p "$INPUTS_DIR" "$LOGS_DIR" "$DESIGNS_DIR" "$SEQS_DIR" \
         "$FOLDS_DIR" "$METRICS_DIR" "$REPORTS_DIR" "$LOGS_DIR/gpu_pool"

export PYTHONPATH="$FOUNDRY_ROOT/src:$FOUNDRY_ROOT/models/rfd3/src:$FOUNDRY_ROOT/models/rf3/src:$FOUNDRY_ROOT/models/mpnn/src:${PYTHONPATH:-}"
export PAPER_ROOT FOUNDRY_ROOT OUT N_WORKERS GPUS MAX_PARALLEL_GPUS
