#!/usr/bin/env bash
# Verify that the NMF/ZKP evaluation-path fixes are present in this checkout.
#
# Why this exists: the fixes were also shipped as a unified diff
# (`fix_nmf_zkp_eval_errors.patch`). If the working tree already contains them,
# `git apply` refuses with "patch does not apply" for every file -- that is not
# an error, it just means there is nothing left to do. This script answers the
# real question: "is each fix actually in place?"
#
# Usage:
#   bash models/rfd3/scripts/verify_nmf_zkp_fixes.sh
# Exit code 0 = every fix present.

set -uo pipefail

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "cannot cd to ${REPO_ROOT}" >&2; exit 2; }

missing=0

chk() {
  local file="$1" needle="$2" label="$3"
  if [[ ! -f "$file" ]]; then
    printf '  [FILE MISSING] %s  (%s)\n' "$label" "$file"
    missing=$((missing + 1))
    return
  fi
  if grep -qF -- "$needle" "$file"; then
    printf '  [OK]      %s\n' "$label"
  else
    printf '  [MISSING] %s  (%s)\n' "$label" "$file"
    missing=$((missing + 1))
  fi
}

echo "repo: ${REPO_ROOT}"
echo
echo "损失函数 / 指标 (models/rfd3/src/rfd3/metrics/losses.py)"
chk models/rfd3/src/rfd3/metrics/losses.py \
  'crd_mask_L = loss_input["crd_mask_L"]' \
  "crd_mask_L 已恢复 (NameError: name 'crd_mask_L' is not defined)"
chk models/rfd3/src/rfd3/metrics/losses.py \
  'if D < 2:' \
  "D=1 不再产生 NaN 的 mse_loss_low/high_t"
chk models/rfd3/src/rfd3/metrics/losses.py \
  'subset_lddt' \
  "子集 lDDT 按子集配对数重归一化 (+ 空子集返回 NaN)"
chk models/rfd3/src/rfd3/metrics/losses.py \
  'if int((w_seq > 0).sum()) == 0:' \
  "sequence_valid_mask 全 0 时不再 NaN"

echo
echo "训练/报告 (models/rfd3/src/rfd3/train_lora.py)"
chk models/rfd3/src/rfd3/train_lora.py \
  'def _get_run_summary_path' \
  "run summary 路径函数已恢复 (原来只留下悬空函数体)"
chk models/rfd3/src/rfd3/train_lora.py \
  '_unwrap_for_parameter_count' \
  "EMA shadow 导致的 total_params 翻倍已修正"
chk models/rfd3/src/rfd3/train_lora.py \
  '_mean_lddt_from_validation_csv' \
  "holdout lDDT 从逐样本验证 CSV 回读 (不再被硬编码列名吞掉)"

echo
echo "指标/绘图"
chk models/rfd3/src/rfd3/metrics/hbonds_metrics.py \
  'conditioning_base import get_motif_features' \
  "hbond 指标缺失的 get_motif_features 导入已补"
chk models/rfd3/scripts/plot_nmf_zkp_pdb_metrics.py \
  'KEY_ALIASES' \
  "绘图脚本接受真实的验证指标列名"

echo
echo "sweep 脚本 (models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh)"
chk models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh \
  '_hbplus_candidate' \
  "HBPLUS 路径改为探测存在的二进制"
chk models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh \
  'EXTRA_OVERRIDES' \
  "新增 EXTRA_OVERRIDES 逃生口 (零训练步评估等)"

echo
if [[ "${missing}" == "0" ]]; then
  echo "全部修复就位 (11/11)。无需再打补丁。"
  exit 0
fi
echo "${missing} 项缺失。可选处理："
echo "  git apply -3 fix_nmf_zkp_eval_errors.patch          # 三方合并，容忍上下文漂移"
echo "  patch -p1 --forward --fuzz=3 < fix_nmf_zkp_eval_errors.patch"
echo "若两条命令都失败，说明这个 checkout 的基线不同，请提供："
echo "  git log --oneline -1; git status --short; git diff --stat"
exit 1
