#!/usr/bin/env bash
# =============================================================================
# 实验 5 —— 酶设计 / AME benchmark（论文 §3.4 / Fig. 3d / Fig. S5 / Fig. S6）
#
# 论文做法：
#   * Atomic Motif Enzyme (AME) benchmark：41 个来自 PDB 的活性位点
#   * 用 `unindex` 把 motif 残基的序列位置交给模型自己找：
#       无索引原子 -> 额外 token，只包含被固定的原子
#   * 每条骨架用 LigandMPNN 配 8 条序列
#   * 用 Chai-1（开源的 AF3 复现）判定：
#       motif backbone-aligned motif all-atom RMSD < 1.5 Å
#   * Fig. 3d 按 residue islands 数量分组（1,2,3,4,>4）
#   * Fig. S5c：41 个案例中 37 个优于 RFD2（90%）
#   * Fig. S6：C2 对称子集，用天然对称变换把 motif 摆到两个亚基上
#               （需要 symmetric noise + symmetry.id）
#
# ⚠️ 前置：AME 的 41 个案例定义在 RFdiffusion2 论文补充材料里，不在本仓库。
#    请先把它们整理成 out/inputs/targets/ame_cases.json（模板已自动生成）。
#
# 运行：
#   ./14_exp5_enzyme.sh
#   ./14_exp5_enzyme.sh --symmetry        # 追加 C2 对称子集
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 5：酶设计 AME benchmark（§3.4 / Fig. 3d）"
[[ -f "$INPUTS_DIR/specs/exp5_ame.json" ]] || \
  die "缺少 $INPUTS_DIR/specs/exp5_ame.json，请先运行 python 01_prepare_inputs.py 并补齐 ame_cases.json"
DEST="$DESIGNS_DIR/exp5_enzyme_ame"

# 论文口径：每案例 100 条骨架（RFdiffusion2 论文的 AME benchmark）。
# 冒烟测试时默认降到 N_BACKBONES，完整复现用 ENZYME_BACKBONES_PER_CASE=100。
if [[ "$SCALE" == "paper" ]]; then N_ENZYME="$ENZYME_BACKBONES_PER_CASE"; else N_ENZYME="$N_BACKBONES"; fi

for seed in $SEEDS; do
  run_dir="$DEST/ame/seed_${seed}"
  log="$LOGS_DIR/exp5_ame_seed${seed}.log"
  log "每个活性位点骨架数=$N_ENZYME  seed=$seed （论文口径：每案例 100 条）"
  ( time rfd3_design "$run_dir" "$INPUTS_DIR/specs/exp5_ame.json" \
      "$N_ENZYME" seed="$seed" ) 2>&1 | tee "$log" || warn "RFD3 运行异常"
  record_run exp5_enzyme_ame "ame" "$N_ENZYME" "seed=$seed"
done

# ---- Fig. S6：C2 对称子集 ----
if [[ "${1:-}" == "--symmetry" || "${WITH_SYMMETRY:-0}" == "1" ]]; then
  if [[ -f "$INPUTS_DIR/specs/exp5_ame_symmetry.json" ]]; then
    hdr "实验 5b：C2 对称 AME 子集（Fig. S6）"
    for seed in $SEEDS; do
      run_dir="$DEST/ame_symmetry/seed_${seed}"
      ( time rfd3_design "$run_dir" "$INPUTS_DIR/specs/exp5_ame_symmetry.json" \
          "$N_BACKBONES" seed="$seed" ) 2>&1 \
          | tee "$LOGS_DIR/exp5_ame_symmetry_seed${seed}.log" || warn "对称子集运行异常"
      record_run exp5_enzyme_ame "ame_symmetry" "$N_BACKBONES" "seed=$seed"
    done
  else
    warn "没有 C2 子集规格（ame_cases.json 里给案例加 symmetry_id 字段即可）"
  fi
fi

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 5：LigandMPNN 序列设计（8 条/骨架，固定 motif）"
  "$PY" 50_sequence_design.py --experiment exp5_enzyme_ame \
        --model ligand_mpnn --n-seqs "$LIGAND_MPNN_SEQS" || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 5：结构预测"
  warn "motif all-atom RMSD 需要在折叠时把 motif 作为条件（Chai-1 / AF3 with motif）。"
  warn "本仓库默认的 RF3 后端只折叠蛋白本体，用于 pLDDT/pTM 的粗筛。"
  "$PY" 60_fold.py --experiment exp5_enzyme_ame || warn "折叠失败"
fi

hdr "实验 5 完成"
ok "设计结果: $DEST"
