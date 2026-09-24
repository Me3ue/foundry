#!/usr/bin/env bash
# =============================================================================
# 一键运行论文正文的全部实验，并在最后汇总成表。
#
# 已按 6×RTX A6000 调好：每个实验内部把条件拆开铺到多张卡上，
# 实验之间串行（否则会互相抢卡、把显存算崩）。
#
#   ./run_all.sh                     # 冒烟测试规模（每条件 8 个骨架）
#   SCALE=paper ./run_all.sh         # 论文规模（每条件 400 个骨架）
#   WITH_SEQ=0 WITH_FOLD=0 ./run_all.sh   # 只采样骨架，不做序列设计和折叠
#   ONLY="2 3" GPUS=2,4 ./run_all.sh # 只跑实验 2/3，且只用 2、4 号卡
#
# 快速自检（不跑实验，只看硬件与路径）：
#   source ./env.sh && source ./lib.sh && check_env
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

export WITH_SEQ="${WITH_SEQ:-1}"
export WITH_FOLD="${WITH_FOLD:-1}"
ONLY="${ONLY:-1 2 3 4 5 6 7 8 9}"

START=$(date +%s)
check_env

hdr "步骤 0：从仓库配置里导出论文 benchmark 定义"
"$PY" 02_extract_repo_benchmarks.py || warn "benchmark 导出失败（PPI 会退回 tutorial 结构）"

hdr "步骤 0b：准备输入结构"
"$PY" 01_prepare_inputs.py

run_step() {
  local id="$1" script="$2"
  if [[ " $ONLY " != *" $id "* ]]; then
    warn "跳过实验 $id（ONLY=$ONLY）"
    return 0
  fi
  hdr "实验 $id -> $script"
  if ! bash "$script"; then
    warn "实验 $id 失败，继续后续实验"
  fi
}

run_step 1 ./10_exp1_unconditional.sh
run_step 2 ./11_exp2_ppi.sh
run_step 3 ./12_exp3_dna.sh
run_step 4 ./13_exp4_small_molecule.sh
run_step 5 ./14_exp5_enzyme.sh
run_step 6 ./15_exp6_symmetry.sh
run_step 7 ./16_exp7_conditioning.sh
run_step 8 ./17_exp8_speed.sh
run_step 9 ./18_exp9_wetlab_insilico.sh

hdr "步骤 70：几何指标（RMSD / 界面 / RASA / 氢键 / clash，${N_WORKERS} 进程并行）"
"$PY" 70_metrics_geometry.py --workers "$N_WORKERS" || warn "几何指标计算失败"

hdr "步骤 90：汇总"
"$PY" 90_summarize.py

hdr "步骤 91：Fig. 1d 速度曲线"
"$PY" 91_plot_speed.py || warn "速度数据还没生成（实验 8 未跑）"

ELAPSED=$(( $(date +%s) - START ))
hdr "全部完成，用时 $((ELAPSED/3600))h$(((ELAPSED%3600)/60))m"
ok "报告: $REPORTS_DIR/paper_experiments_report.md"
ok "逐条件统计: $METRICS_DIR/summary_by_condition.csv"
ok "逐设计明细: $METRICS_DIR/per_design.csv"
ok "GPU 池日志: $LOGS_DIR/gpu_pool/"
