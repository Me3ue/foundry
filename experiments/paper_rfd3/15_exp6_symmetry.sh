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
# 运行：
#   SYMMETRY_IDS="D2 C3 C5 C7" ./15_exp6_symmetry.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 6：对称设计（Fig. 2g / Fig. S6c-d）"
SPEC="$INPUTS_DIR/specs/exp6_symmetry.json"
[[ -f "$SPEC" ]] || die "缺少 $SPEC，请先运行 python 01_prepare_inputs.py"
DEST="$DESIGNS_DIR/exp6_symmetry"

"$PY" - "$SPEC" "$INPUTS_DIR/specs/exp6_symmetry_sel.json" $SYMMETRY_IDS <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
keep = [f"sym_{s}" for s in sys.argv[3:]]
sub = {k: v for k, v in spec.items() if k in keep}
json.dump(sub, open(sys.argv[2], "w"), indent=2)
print(f"选中对称: {list(sub)}")
PY

for seed in $SEEDS; do
  run_dir="$DEST/uncond/seed_${seed}"
  log="$LOGS_DIR/exp6_symmetry_seed${seed}.log"
  log "对称=$SYMMETRY_IDS  骨架数=$N_BACKBONES  seed=$seed"
  ( time rfd3_design "$run_dir" "$INPUTS_DIR/specs/exp6_symmetry_sel.json" \
      "$N_BACKBONES" seed="$seed" ) 2>&1 | tee "$log" || warn "对称运行异常"
  record_run exp6_symmetry "uncond_sym" "$N_BACKBONES" "ids=$SYMMETRY_IDS seed=$seed"
done

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 6：序列设计（同源寡聚体加对称约束）"
  "$PY" 50_sequence_design.py --experiment exp6_symmetry \
        --model protein_mpnn --n-seqs "$MPNN_SEQS" || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 6：结构预测"
  "$PY" 60_fold.py --experiment exp6_symmetry || warn "折叠失败"
fi

hdr "实验 6 完成"
ok "设计结果: $DEST"
ok "对称性验证：70_metrics_geometry.py 会计算各亚基间的 Cα RMSD"
