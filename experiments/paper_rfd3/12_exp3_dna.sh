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
# 运行：
#   ./12_exp3_dna.sh
#   SCALE=paper DNA_TARGETS="7rte 7n5u 7m5w" ./12_exp3_dna.sh
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

# 按 DNA_TARGETS 过滤（规格 key 形如 7rte_a_rigid）
filter_spec() {
  "$PY" - "$1" "$2" $DNA_TARGETS <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
keep = tuple(sys.argv[3:])
sub = {k: v for k, v in spec.items() if any(k.startswith(p) for p in keep)}
json.dump(sub, open(sys.argv[2], "w"), indent=2)
print(f"{sys.argv[2]}: {list(sub)}")
PY
}
filter_spec "$INPUTS_DIR/specs/exp3_dna_rigid.json"    "$INPUTS_DIR/specs/exp3_dna_rigid_sel.json"
filter_spec "$INPUTS_DIR/specs/exp3_dna_diffused.json" "$INPUTS_DIR/specs/exp3_dna_diffused_sel.json"

for setting in rigid diffused; do
  for seed in $SEEDS; do
    run_dir="$DEST/$setting/seed_${seed}"
    log="$LOGS_DIR/exp3_dna_${setting}_seed${seed}.log"
    log "设置=$setting  每靶点骨架数=$N_BACKBONES  seed=$seed"
    ( time rfd3_design "$run_dir" "$INPUTS_DIR/specs/exp3_dna_${setting}_sel.json" \
        "$N_BACKBONES" seed="$seed" ) 2>&1 | tee "$log" || warn "$setting 运行异常"
    record_run exp3_dna "$setting" "$N_BACKBONES" "seed=$seed"
  done
done

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 3：LigandMPNN 序列设计（4 条/骨架）"
  "$PY" 50_sequence_design.py --experiment exp3_dna \
        --model ligand_mpnn --n-seqs "${MPNN_SEQS}" || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 3：结构预测"
  "$PY" 60_fold.py --experiment exp3_dna || warn "折叠失败"
fi

hdr "实验 3 完成"
ok "设计结果: $DEST"
ok "下一步：用 70_metrics_geometry.py 计算 DNA-aligned RMSD，再 90_summarize.py"
