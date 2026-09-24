#!/usr/bin/env bash
# =============================================================================
# RFdiffusion3 论文实验复现 —— 全局配置
#
# 所有变量都可以通过环境变量覆盖，例如：
#   RFD3_CKPT=/path/to/ckpt.ckpt ./run_all.sh
#   SCALE=quick ./run_all.sh              # 冒烟测试（每条件 8 个骨架）
#   SCALE=paper ./run_all.sh              # 论文规模（每条件 400 个骨架）
# =============================================================================
set -Eeuo pipefail

# ---------------------------------------------------------------- 目录布局 ---
# 本文件位于 <repo>/experiments/paper_rfd3/env.sh
PAPER_ROOT="${PAPER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FOUNDRY_ROOT="${FOUNDRY_ROOT:-$(cd "$PAPER_ROOT/../.." && pwd)}"
export PAPER_ROOT FOUNDRY_ROOT

OUT="${OUT:-$PAPER_ROOT/out}"            # 所有实验产物
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
export RC_ENV_BIN="${RC_ENV_BIN:-/home/zzj/anaconda3/envs/rc/bin}"
PY="${PY:-$RC_ENV_BIN/python}"
RF_ENV_NAME="${RF_ENV_NAME:-rc}"

# --------------------------------------------------------------- 模型权重 ---
RFD3_CKPT="${RFD3_CKPT:-/media/zzj/Data/rfd3_latest.ckpt}"
RF3_CKPT="${RF3_CKPT:-$HOME/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt}"
MPNN_CKPT_DIR="${MPNN_CKPT_DIR:-$HOME/.foundry/checkpoints}"
PROTEINMPNN_CKPT="${PROTEINMPNN_CKPT:-$MPNN_CKPT_DIR/proteinmpnn_v_48_020.pt}"
LIGANDMPNN_CKPT="${LIGANDMPNN_CKPT:-$MPNN_CKPT_DIR/ligandmpnn_v_32_010_25.pt}"
HBPLUS="${HBPLUS:-/home/zzj/protein/HBPLUS/hbplus/hbplus}"

# ------------------------------------------------------------ 论文采样参数 ---
# 论文 Fig. S4 脚注明确给出："step scale η = 1.5, noise level γ0 = 0.6,
# 200 denoising steps（除非另有说明）"。这些就是 RFD3 的默认值。
STEP_SCALE="${STEP_SCALE:-1.5}"          # η
GAMMA_0="${GAMMA_0:-0.6}"                # γ0
NUM_TIMESTEPS="${NUM_TIMESTEPS:-200}"
CFG_SCALE="${CFG_SCALE:-2.0}"            # 小分子 diffused-ligand 条件用 2.0
N_RECYCLE="${N_RECYCLE:-2}"

# 小显存 GPU 建议打开：分块 tokenization，显著降低峰值显存。
# 参考：本机 11.5 GiB 显存曾出现 OOM，low_memory_mode=True 可缓解。
LOW_MEMORY="${LOW_MEMORY:-True}"
if [[ "$LOW_MEMORY" == "True" ]]; then LOW_MEM_FLAG="--low_memory"; else LOW_MEM_FLAG=""; fi

# 单次 forward 的样本数。显存不足时降到 1-2。
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-4}"

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
#   推理速度标定在 NVIDIA A6000 上完成（Fig. 1d）
#   完整设备规模 + 不带降级处理的标准命令：见 COMMANDS.md

# ------------------------------------------------------------- 目标清单 ---
# 蛋白-蛋白结合物（论文 §3.1 / Fig. 3a）。仓库自带两个裁剪好的靶点；
# tie2 / il7ra / il2ra 需要用 prepare_inputs.py 的 targets 配置补齐。
PPI_TARGETS="${PPI_TARGETS:-pdl1 insulinr}"

# DNA 结合物（论文 §3.2 / Fig. 3b）：三个训练集外序列。
DNA_TARGETS="${DNA_TARGETS:-7rte 7n5u 7m5w}"

# 小分子结合物（论文 §3.3 / Fig. 3c）：FAD、SAM 常见；IAI、OQO 罕见。
SM_LIGANDS="${SM_LIGANDS:-IAI}"

# 对称性（Fig. 2g / Fig. S6c-d）
SYMMETRY_IDS="${SYMMETRY_IDS:-C3 C5 D2}"

# 无条件单体长度（§3 正文：100-200 aa）
UNCOND_LENGTHS="${UNCOND_LENGTHS:-100 150 200}"

# ------------------------------------------------------------------ 工具 ---
# 结构预测后端：rf3（默认，开源可本地跑）/ af3 / chai / none（跳过）
FOLD_BACKEND="${FOLD_BACKEND:-rf3}"

# 汇总报告里是否附带中文说明
REPORT_LANG="${REPORT_LANG:-zh}"

# ------------------------------------------------------------ 目录初始化 ---
mkdir -p "$INPUTS_DIR" "$LOGS_DIR" "$DESIGNS_DIR" "$SEQS_DIR" \
         "$FOLDS_DIR" "$METRICS_DIR" "$REPORTS_DIR"

export PYTHONPATH="$FOUNDRY_ROOT/src:$FOUNDRY_ROOT/models/rfd3/src:$FOUNDRY_ROOT/models/rf3/src:$FOUNDRY_ROOT/models/mpnn/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
