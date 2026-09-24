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
# 备注：论文的 benchmark 用 η=1.5, γ0=0.6（与全局默认一致）。
#       官方文档给的"生产推荐"是 step_scale=3, gamma_0=0.2，命中率更高但更同质。
#       想复现官方推荐可以：STEP_SCALE=3 GAMMA_0=0.2 ./11_exp2_ppi.sh
#
# 运行：
#   PPI_TARGETS="pdl1 insulinr" ./11_exp2_ppi.sh
#   SCALE=paper PPI_TARGETS="pdl1 insulinr tie2 il7ra il2ra" ./11_exp2_ppi.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 2：蛋白结合蛋白（§3.1 / Fig. 3a）"
SPEC="$INPUTS_DIR/specs/exp2_ppi.json"
[[ -f "$SPEC" ]] || die "缺少 $SPEC，请先运行 python 01_prepare_inputs.py"
DEST="$DESIGNS_DIR/exp2_ppi"

# 只保留用户选中的靶点，生成一个筛选后的规格文件
"$PY" - "$SPEC" "$INPUTS_DIR/specs/exp2_ppi_selected.json" $PPI_TARGETS <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
keep = set(sys.argv[3:])
sub = {k: v for k, v in spec.items() if k in keep}
json.dump(sub, open(sys.argv[2], "w"), indent=2)
print(f"选中靶点: {list(sub)}")
PY
SPEC_SEL="$INPUTS_DIR/specs/exp2_ppi_selected.json"

for seed in $SEEDS; do
  run_dir="$DEST/all/seed_${seed}"
  log="$LOGS_DIR/exp2_ppi_seed${seed}.log"
  log "每靶点骨架数=$N_BACKBONES  seed=$seed"
  ( time rfd3_design "$run_dir" "$SPEC_SEL" "$N_BACKBONES" seed="$seed" ) 2>&1 \
      | tee "$log" || warn "RFD3 运行异常"
  record_run exp2_ppi "all" "$N_BACKBONES" "seed=$seed"
done

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 2：ProteinMPNN 序列设计（4 条/骨架）"
  "$PY" 50_sequence_design.py --experiment exp2_ppi \
        --model protein_mpnn --n-seqs "${MPNN_SEQS}" || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 2：结构预测（AF3 等价判据需要 PAE / pTM）"
  "$PY" 60_fold.py --experiment exp2_ppi || warn "折叠失败"
fi

hdr "实验 2 完成"
ok "设计结果: $DEST"
ok "汇总:     python 90_summarize.py --experiment exp2_ppi"
