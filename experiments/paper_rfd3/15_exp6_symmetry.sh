#!/usr/bin/env bash
# =============================================================================
# 实验 6 —— 对称设计（论文 Fig. 2g / Fig. S6c-d）
#
# 论文做法：
#   * 用对称噪声初始化，在推理时对 diffusion 模块的输出做对称化
#   * 展示 D2 / C3 / C5 / C7 四种对称，AF3 Cα RMSD 分别为
#     0.832 / 0.450 / 0.614 / 0.539 Å
#   * Fig. S6c-d 还测了"对称约束但不给 motif"的骨架成功率
#
# 多卡：D2/C3/C5/C7 四个对称条件各占一张卡。
#
# 运行：
#   ./15_exp6_symmetry.sh
#   SYMMETRY_IDS="D2 C3 C5 C7" SCALE=paper ./15_exp6_symmetry.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 6：对称设计（Fig. 2g / Fig. S6c-d）"
SPEC="$INPUTS_DIR/specs/exp6_symmetry.json"
[[ -f "$SPEC" ]] || die "缺少 $SPEC，请先运行 python 01_prepare_inputs.py"
DEST="$DESIGNS_DIR/exp6_symmetry"

build_jobs() {
  local key path seed s keep
  while IFS=$'\t' read -r key path; do
    # 规格 key 形如 sym_C3；只有 SYMMETRY_IDS 里列出的才跑
    keep=1
    for s in $SYMMETRY_IDS; do [[ "$key" == "sym_${s}" ]] && keep=0; done
    [[ $keep -eq 0 ]] || continue
    for seed in $SEEDS; do
      rfd3_design_cmd "$DEST/uncond/$key/seed_${seed}" "$path" "$N_BACKBONES" \
        "seed=$seed"
      printf '\n'
      record_run exp6_symmetry "uncond_sym" "$N_BACKBONES" "sym=$key seed=$seed"
    done
  done < <(split_spec "$SPEC" "$INPUTS_DIR/specs/exp6_split")
}

log "对称类型: $SYMMETRY_IDS"
build_jobs | run_on_gpus "$MAX_PARALLEL_GPUS" \
  || warn "部分任务失败（日志见 $LOGS_DIR/gpu_pool/）"

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 6：序列设计（同源寡聚体加对称约束，${N_WORKERS} 进程并行）"
  "$PY" 50_sequence_design.py --experiment exp6_symmetry \
        --model protein_mpnn --n-seqs "$MPNN_SEQS" --workers "$N_WORKERS" \
        || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 6：结构预测"
  "$PY" 60_fold.py --experiment exp6_symmetry --parallel "$FOLD_PARALLEL_GPUS" \
        || warn "折叠失败"
fi

hdr "实验 6 完成"
ok "设计结果: $DEST"
ok "对称性验证：70_metrics_geometry.py 会算各亚基间的 Cα RMSD"
