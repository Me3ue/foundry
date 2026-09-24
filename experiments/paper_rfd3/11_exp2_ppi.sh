#!/usr/bin/env bash
# =============================================================================
# 实验 2 —— 蛋白结合蛋白（论文 §3.1 / Fig. 3a / Fig. S2）
#
# 论文做法：
#   * 5 个治疗相关靶点：PD-L1、InsulinR、IL-7Ra、Tie2、IL-2Ra
#   * 每个靶点生成 400 条骨架（= diffusion_batch_size 8 × n_batches 50）
#   * 每条骨架用 ProteinMPNN 设计 4 条序列
#   * AF3 通过判据（来自 Zambaldi et al. [8]）：
#         min inter-chain PAE ≤ 1.5
#         binder pTM        ≥ 0.8
#         target-aligned binder Cα RMSD < 2.5 Å
#   * 再用 TM-score 0.6 做 complete-linkage 聚类，统计"成功簇数"
#     （论文：RFD3 平均 8.2 个成功簇 vs RFD1 1.4 个）
#
# 多卡：**每个靶点占一张卡**，5 个靶点正好铺满 5 张 A6000。
#       参数只影响吞吐不影响统计口径，所以分卡跑与原版等价。
#
# 备注：论文的 benchmark 用 η=1.5, γ0=0.6（与全局默认一致）。
#       官方文档给的"生产推荐"是 step_scale=3, gamma_0=0.2，命中率更高但更同质。
#
# 运行：
#   ./11_exp2_ppi.sh
#   SCALE=paper PPI_TARGETS="pdl1 insulinr tie2 il2ra il7ra" ./11_exp2_ppi.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 2：蛋白结合蛋白（§3.1 / Fig. 3a）"
SPEC="$INPUTS_DIR/specs/exp2_ppi.json"
[[ -f "$SPEC" ]] || die "缺少 $SPEC，请先运行 02_extract_repo_benchmarks.py + 01_prepare_inputs.py"
DEST="$DESIGNS_DIR/exp2_ppi"

# 按靶点拆成独立规格，这样每个靶点可以单独占一张卡
SPLIT_DIR="$INPUTS_DIR/specs/exp2_split"
mapfile -t PAIRS < <(split_spec "$SPEC" "$SPLIT_DIR")
[[ ${#PAIRS[@]} -gt 0 ]] || die "$SPEC 里没有靶点"

build_jobs() {
  local pair key path seed
  for pair in "${PAIRS[@]}"; do
    key="${pair%%$'\t'*}"; path="${pair#*$'\t'}"
    # 用户没指定就用全部靶点
    if [[ -n "$PPI_TARGETS" && " $PPI_TARGETS " != *" $key "* ]]; then continue; fi
    for seed in $SEEDS; do
      rfd3_design_cmd "$DEST/$key/seed_${seed}" "$path" "$N_BACKBONES" "seed=$seed"
      printf '\n'
      record_run exp2_ppi "$key" "$N_BACKBONES" "seed=$seed"
    done
  done
}

build_jobs | run_on_gpus "$MAX_PARALLEL_GPUS" \
  || warn "部分靶点失败（日志见 $LOGS_DIR/gpu_pool/）"

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 2：ProteinMPNN 序列设计（4 条/骨架，${N_WORKERS} 进程并行）"
  "$PY" 50_sequence_design.py --experiment exp2_ppi \
        --model protein_mpnn --n-seqs "$MPNN_SEQS" --workers "$N_WORKERS" \
        || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 2：结构预测（AF3 等价判据需要 PAE / pTM）"
  "$PY" 60_fold.py --experiment exp2_ppi --parallel "$FOLD_PARALLEL_GPUS" \
        || warn "折叠失败"
fi

hdr "实验 2 完成"
ok "设计结果: $DEST"
ok "汇总:     python 90_summarize.py --experiment exp2_ppi"
