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
# 运行：
#   ./10_exp1_unconditional.sh
#   SCALE=paper ./10_exp1_unconditional.sh          # 论文规模
#   ETAS="1.0 1.5 2.0 3.0" ./10_exp1_unconditional.sh
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
for eta in $ETAS; do
  for seed in $SEEDS; do
    run_dir="$DEST/eta_${eta}/seed_${seed}"
    log="$LOGS_DIR/exp1_eta_${eta}_seed${seed}.log"
    log "η=$eta  seed=$seed  骨架数=$N_BACKBONES"
    ( time rfd3_design "$run_dir" "$SPEC" "$N_BACKBONES" \
        inference_sampler.step_scale="$eta" seed="$seed" ) 2>&1 | tee "$log" || warn "η=$eta 运行异常"
    record_run exp1_unconditional "eta_${eta}" "$N_BACKBONES" "seed=$seed"
  done
done

# ---- 可选：序列设计 + 自洽性折叠 ----
if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 1：ProteinMPNN 序列设计"
  "$PY" 50_sequence_design.py --experiment exp1_unconditional \
        --model protein_mpnn --n-seqs "$MPNN_SEQS" || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 1：结构预测（自洽性）"
  "$PY" 60_fold.py --experiment exp1_unconditional || warn "折叠失败"
fi

hdr "实验 1 完成"
ok "设计结果: $DEST"
ok "汇总:     python 90_summarize.py --experiment exp1_unconditional"
