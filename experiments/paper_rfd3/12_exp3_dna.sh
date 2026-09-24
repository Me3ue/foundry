#!/usr/bin/env bash
# =============================================================================
# 实验 3 —— DNA 结合蛋白（论文 §3.2 / Fig. 3b / Fig. S3）
#
# 论文做法：
#   * 三个训练集外的 DNA 靶点：7RTE、7N5U、7M5W
#   * 每个靶点 400 条骨架，每条 LigandMPNN 配 4 条序列
#   * 两个设置（Fig. 3b 的 + / −）：
#       rigid    —— DNA 构象取自输入 PDB，全原子固定（+）
#       diffused —— 只给 DNA 序列，模型与蛋白一起采样 DNA 构象（−）
#   * AF3 折叠后用"DNA-aligned RMSD"判成败：
#       对齐 DNA 磷酸原子(P, OP1, OP2) → 算蛋白 Cα 的 RMSD → 裁剪 N/C 端 loop
#     分档统计：<1.5 Å / 1.5-3 Å / 3-5 Å
#   * 论文结论：单体 8.67% / 二体 6.67% 通过率（<5 Å）
#
# 多卡：2 设置 × 3 靶点 = 6 个任务，正好铺满 6 张 A6000。
#
# 运行：
#   ./12_exp3_dna.sh
#   SCALE=paper ./12_exp3_dna.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 3：DNA 结合蛋白（§3.2 / Fig. 3b）"
for f in exp3_dna_rigid.json exp3_dna_diffused.json; do
  [[ -f "$INPUTS_DIR/specs/$f" ]] || die "缺少 $INPUTS_DIR/specs/$f，请先运行 python 01_prepare_inputs.py"
done
DEST="$DESIGNS_DIR/exp3_dna"

build_jobs() {
  local setting spec pair key path seed
  for setting in rigid diffused; do
    spec="$INPUTS_DIR/specs/exp3_dna_${setting}.json"
    while IFS=$'\t' read -r key path; do
      # 按 DNA_TARGETS 过滤（key 形如 7rte_a_rigid）
      key_matches "$key" $DNA_TARGETS || continue
      for seed in $SEEDS; do
        rfd3_design_cmd "$DEST/$setting/$key/seed_${seed}" "$path" "$N_BACKBONES" \
          "seed=$seed"
        printf '\n'
        record_run exp3_dna "$setting" "$N_BACKBONES" "target=$key seed=$seed"
      done
    done < <(split_spec "$spec" "$INPUTS_DIR/specs/exp3_split")
  done
}

build_jobs | run_on_gpus "$MAX_PARALLEL_GPUS" \
  || warn "部分任务失败（日志见 $LOGS_DIR/gpu_pool/）"

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 3：LigandMPNN 序列设计（4 条/骨架，${N_WORKERS} 进程并行）"
  "$PY" 50_sequence_design.py --experiment exp3_dna \
        --model ligand_mpnn --n-seqs "$MPNN_SEQS" --workers "$N_WORKERS" \
        || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 3：结构预测"
  "$PY" 60_fold.py --experiment exp3_dna --parallel "$FOLD_PARALLEL_GPUS" \
        || warn "折叠失败"
fi

hdr "实验 3 完成"
ok "设计结果: $DEST"
ok "下一步：python 70_metrics_geometry.py --experiment exp3_dna（算 DNA-aligned RMSD）"
