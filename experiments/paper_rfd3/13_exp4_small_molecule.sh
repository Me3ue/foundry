#!/usr/bin/env bash
# =============================================================================
# 实验 4 —— 小分子结合蛋白（论文 §3.3 / Fig. 3c / Fig. S4）
#
# 论文做法：
#   * 4 个配体：FAD、SAM（PDB 常见）、IAI、OQO（罕见）
#   * 每个条件 400 条骨架，每条 LigandMPNN 配 8 条序列
#   * 三种条件（Fig. 3c 的柱子）：
#       RFdiffusionAA        —— 前作方法（外部仓库，本脚本不跑）
#       fixed ligand         —— 用 PDB 里的晶体构象，配体刚体固定
#       diffused ligand      —— 配体坐标一起扩散 + RASA=buried + CFG scale 2
#   * AF3 通过判据：
#         backbone RMSD ≤ 1.5 Å、ligand RMSD ≤ 5 Å
#         min chain-pair PAE ≤ 1.5、ipTM ≥ 0.8
#   * 附加分析（Fig. S4b-d）：FoldSeek 新颖性、Rosetta ΔΔG、
#                               RDKit 构象 RMSD（每个配体 50 个构象）
#
# 多卡：2 条件 × 4 配体 = 8 个任务，按空闲显存自动铺到 6 张卡上。
#       diffused 模式的 CFG 会让单次显存略高，GPU_JOB_MEM 已按 25 GB 预留。
#
# 运行：
#   ./13_exp4_small_molecule.sh
#   SCALE=paper SM_LIGANDS="FAD SAM IAI OQO" ./13_exp4_small_molecule.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 4：小分子结合蛋白（§3.3 / Fig. 3c）"
for f in exp4_sm_fixed.json exp4_sm_diffused.json; do
  [[ -f "$INPUTS_DIR/specs/$f" ]] || die "缺少 $INPUTS_DIR/specs/$f，请先运行 python 01_prepare_inputs.py"
done
DEST="$DESIGNS_DIR/exp4_small_molecule"

build_jobs() {
  local mode spec key path seed
  for mode in fixed diffused; do
    spec="$INPUTS_DIR/specs/exp4_sm_${mode}.json"
    while IFS=$'\t' read -r key path; do
      key_matches "$key" $SM_LIGANDS || continue
      for seed in $SEEDS; do
        if [[ "$mode" == "diffused" ]]; then
          # 论文："diffused-ligand binders were generated with the RASA
          #        condition set to buried and a CFG scale of 2"
          rfd3_design_cmd "$DEST/$mode/$key/seed_${seed}" "$path" "$N_BACKBONES" \
            "seed=$seed" \
            "inference_sampler.use_classifier_free_guidance=True" \
            "inference_sampler.cfg_scale=$CFG_SCALE"
        else
          rfd3_design_cmd "$DEST/$mode/$key/seed_${seed}" "$path" "$N_BACKBONES" \
            "seed=$seed"
        fi
        printf '\n'
        record_run exp4_small_molecule "$mode" "$N_BACKBONES" "ligand=$key seed=$seed"
      done
    done < <(split_spec "$spec" "$INPUTS_DIR/specs/exp4_split")
  done
}

build_jobs | run_on_gpus "$MAX_PARALLEL_GPUS" \
  || warn "部分任务失败（日志见 $LOGS_DIR/gpu_pool/）"

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 4：LigandMPNN 序列设计（8 条/骨架，${N_WORKERS} 进程并行）"
  "$PY" 50_sequence_design.py --experiment exp4_small_molecule \
        --model ligand_mpnn --n-seqs "$LIGAND_MPNN_SEQS" --workers "$N_WORKERS" \
        || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 4：结构预测（最多并行 $FOLD_PARALLEL_GPUS 卡）"
  "$PY" 60_fold.py --experiment exp4_small_molecule \
        --parallel "$FOLD_PARALLEL_GPUS" || warn "折叠失败"
fi

hdr "实验 4 完成"
ok "设计结果: $DEST"
warn "RFdiffusionAA 基线需在 https://github.com/baker-laboratory/RoseTTAFold-All-Atom 单独跑"
warn "Rosetta ΔΔG（DDGnoRepack）与 FoldSeek 新颖性见 COMMANDS.md §B.3"
