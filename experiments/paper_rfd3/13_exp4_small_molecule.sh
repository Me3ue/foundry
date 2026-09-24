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
# 运行：
#   SM_LIGANDS="IAI" ./13_exp4_small_molecule.sh
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

for mode in fixed diffused; do
  for seed in $SEEDS; do
    run_dir="$DEST/$mode/seed_${seed}"
    log="$LOGS_DIR/exp4_sm_${mode}_seed${seed}.log"
    extra=()
    if [[ "$mode" == "diffused" ]]; then
      # 论文："diffused-ligand binders were generated with the RASA condition
      #        set to buried and a CFG scale of 2"
      extra+=(inference_sampler.use_classifier_free_guidance=True
              inference_sampler.cfg_scale="$CFG_SCALE")
    fi
    log "模式=$mode  每配体骨架数=$N_BACKBONES  seed=$seed"
    ( time rfd3_design "$run_dir" "$INPUTS_DIR/specs/exp4_sm_${mode}.json" \
        "$N_BACKBONES" seed="$seed" "${extra[@]}" ) 2>&1 \
        | tee "$log" || warn "$mode 运行异常"
    record_run exp4_small_molecule "$mode" "$N_BACKBONES" "seed=$seed"
  done
done

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 4：LigandMPNN 序列设计（8 条/骨架）"
  "$PY" 50_sequence_design.py --experiment exp4_small_molecule \
        --model ligand_mpnn --n-seqs "$LIGAND_MPNN_SEQS" || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 4：结构预测"
  "$PY" 60_fold.py --experiment exp4_small_molecule || warn "折叠失败"
fi

hdr "实验 4 完成"
ok "设计结果: $DEST"
warn "RFdiffusionAA 基线需在 https://github.com/baker-laboratory/RoseTTAFold-All-Atom 单独跑"
warn "Rosetta ΔΔG（DDGnoRepack）与 FoldSeek 新颖性见 README「外部工具」一节"
