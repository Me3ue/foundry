#!/usr/bin/env bash
# =============================================================================
# 实验 1 —— 无条件蛋白生成（论文 §3 "IN SILICO RESULTS" 开头 + Fig. S1c）
#
# 论文做法：
#   * 生成长度 100-200 aa 的全新骨架
#   * 每条骨架用 ProteinMPNN 设计 8 条序列（本脚本默认按 SCALE 缩放）
#   * 用 AF3 判据统计"可设计性"：至少 1 条序列折叠到 1.5 Å RMSD 以内
#   * Fig. S1c 扫描 step scale η，观察 可设计性 ↑ / 多样性 ↓ 的权衡，
#     论文最终选定 η = 1.5
#
# 多卡：4 个 η 值直接铺到 4 张空闲卡上同时跑（A6000 48 GB 跑 L≤200 的批次很轻松）。
#
# 运行：
#   ./10_exp1_unconditional.sh
#   SCALE=paper ./10_exp1_unconditional.sh              # 论文规模
#   ETAS="1.0 1.5 2.0 3.0" GPUS=2,4 ./10_exp1_unconditional.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 1：无条件生成（§3 / Fig. S1c）"
SPEC="$INPUTS_DIR/specs/exp1_unconditional.json"
[[ -f "$SPEC" ]] || die "缺少 $SPEC，请先运行 python 01_prepare_inputs.py"
DEST="$DESIGNS_DIR/exp1_unconditional"
ETAS="${ETAS:-1.0 1.5 2.0 3.0}"

# 每个 (η, seed) 一个任务 -> 铺到多张卡
build_jobs() {
  local eta seed
  for eta in $ETAS; do
    for seed in $SEEDS; do
      rfd3_design_cmd "$DEST/eta_${eta}/seed_${seed}" "$SPEC" "$N_BACKBONES" \
        "inference_sampler.step_scale=$eta" "seed=$seed"
      printf '\n'
      record_run exp1_unconditional "eta_${eta}" "$N_BACKBONES" "seed=$seed"
    done
  done
}

build_jobs | run_on_gpus "$MAX_PARALLEL_GPUS" \
  || warn "部分任务失败（日志见 $LOGS_DIR/gpu_pool/）"

# ---- 序列设计 + 自洽性折叠 ----
if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 1：ProteinMPNN 序列设计（${N_WORKERS} 进程并行）"
  "$PY" 50_sequence_design.py --experiment exp1_unconditional \
        --model protein_mpnn --n-seqs "$MPNN_SEQS" --workers "$N_WORKERS" \
        || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 1：结构预测（自洽性，最多并行 $FOLD_PARALLEL_GPUS 卡）"
  "$PY" 60_fold.py --experiment exp1_unconditional \
        --parallel "$FOLD_PARALLEL_GPUS" || warn "折叠失败"
fi

hdr "实验 1 完成"
ok "设计结果: $DEST"
ok "汇总:     python 90_summarize.py --experiment exp1_unconditional"
