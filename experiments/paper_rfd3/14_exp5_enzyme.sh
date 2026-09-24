#!/usr/bin/env bash
# =============================================================================
# 实验 5 —— 酶设计 / AME benchmark（论文 §3.4 / Fig. 3d / Fig. S5 / Fig. S6）
#
# 论文做法：
#   * Atomic Motif Enzyme (AME) benchmark：41 个来自 PDB 的活性位点
#   * 用 `unindex` 把 motif 残基的序列位置交给模型自己找：
#       无索引原子 -> 额外 token，只包含被固定的原子
#   * 每个案例生成 100 条骨架，用 LigandMPNN 配 8 条序列
#   * 用 Chai-1 判定（5 个 diffusion sample 取一个）：
#       motif 骨架对齐后，所有原子侧链 RMSD < 1.5 Å，且配体与骨架无 clash
#   * Fig. 3d 按 residue islands 数量分组（1,2,3,4,>4）
#   * Fig. S5c：41 个案例中 37 个优于 RFD2（90%）
#   * Fig. S6：C2 对称子集，用天然对称变换把 motif 摆到两个亚基上
#
# 多卡：**41 个案例拆成 41 个独立任务**，6 张 A6000 一起跑，
#       是最能吃满机器的一个实验。
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

# 论文口径：每案例 100 条骨架（RFdiffusion2 论文的 AME benchmark）
if [[ "$SCALE" == "paper" ]]; then N_ENZYME="$ENZYME_BACKBONES_PER_CASE"; else N_ENZYME="$N_BACKBONES"; fi

build_jobs() {
  local key path seed
  while IFS=$'\t' read -r key path; do
    for seed in $SEEDS; do
      rfd3_design_cmd "$DEST/ame/$key/seed_${seed}" "$path" "$N_ENZYME" "seed=$seed"
      printf '\n'
      record_run exp5_enzyme_ame "ame" "$N_ENZYME" "case=$key seed=$seed"
    done
  done < <(split_spec "$INPUTS_DIR/specs/exp5_ame.json" "$INPUTS_DIR/specs/exp5_split")
}

log "每案例骨架数=$N_ENZYME"
build_jobs | run_on_gpus "$MAX_PARALLEL_GPUS" \
  || warn "部分案例失败（日志见 $LOGS_DIR/gpu_pool/）"

# ---- Fig. S6：C2 对称子集 ----
if [[ "${1:-}" == "--symmetry" || "${WITH_SYMMETRY:-0}" == "1" ]]; then
  if [[ -f "$INPUTS_DIR/specs/exp5_ame_symmetry.json" ]]; then
    hdr "实验 5b：C2 对称 AME 子集（Fig. S6）"
    build_sym_jobs() {
      local key path seed
      while IFS=$'\t' read -r key path; do
        for seed in $SEEDS; do
          rfd3_design_cmd "$DEST/ame_symmetry/$key/seed_${seed}" "$path" \
            "$N_ENZYME" "seed=$seed"
          printf '\n'
          record_run exp5_enzyme_ame "ame_symmetry" "$N_ENZYME" "case=$key seed=$seed"
        done
      done < <(split_spec "$INPUTS_DIR/specs/exp5_ame_symmetry.json" \
                          "$INPUTS_DIR/specs/exp5_sym_split")
    }
    build_sym_jobs | run_on_gpus "$MAX_PARALLEL_GPUS" || warn "对称子集部分失败"
  else
    warn "没有 C2 子集规格（ame_cases.json 里给案例加 symmetry_id 字段即可）"
  fi
fi

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 5：LigandMPNN 序列设计（8 条/骨架，${N_WORKERS} 进程并行）"
  "$PY" 50_sequence_design.py --experiment exp5_enzyme_ame \
        --model ligand_mpnn --n-seqs "$LIGAND_MPNN_SEQS" --workers "$N_WORKERS" \
        || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 5：结构预测"
  warn "motif all-atom RMSD 需要在折叠时把 motif 作为条件（Chai-1 / AF3 with motif）。"
  warn "本仓库默认的 RF3 后端只折叠蛋白本体，用于 pLDDT/pTM 的粗筛。"
  "$PY" 60_fold.py --experiment exp5_enzyme_ame --parallel "$FOLD_PARALLEL_GPUS" \
        || warn "折叠失败"
fi

hdr "实验 5 完成"
ok "设计结果: $DEST"
