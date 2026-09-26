#!/usr/bin/env bash
# Merge several NMF/ZKP sweep directories into one combined report.
#
# Why: each `run_nmf_zkp_pdb_sweep.sh` invocation writes its own timestamped
# sweep dir, so running families on separate GPUs (or `LAYER_SET=encoder` /
# `head` / `all` in parallel) produces several independent reports. This script
# copies the *small* report inputs of every sweep dir into one merged dir and
# regenerates the combined summary + figures there.
#
# Only the files the report needs are copied (per-example validation CSV,
# per-epoch metrics.csv, and the top-level run_summary / nmf_replacements /
# exit_code). Multi-GB `val_structures/` dumps are intentionally skipped, and
# plain copies are used instead of symlinks because pathlib's rglob stopped
# following directory symlinks in Python 3.13.
#
# Usage:
#   bash models/rfd3/scripts/merge_sweep_reports.sh OUT_DIR SWEEP_DIR [SWEEP_DIR ...]
#   bash models/rfd3/scripts/merge_sweep_reports.sh OUT_DIR --discover [LOG_ROOT]
#
#   # --discover merges every sweep_nmf_zkp_pdb_* under LOG_ROOT
#   # (default /backup01/zzj/logs/train_nmf_zkp_pdb); the most recent dir first,
#   # so its version of a duplicated tag wins.
#
# Examples:
#   LOG=/backup01/zzj/logs/train_nmf_zkp_pdb
#   bash models/rfd3/scripts/merge_sweep_reports.sh "$LOG/sweep_combined" --discover "$LOG"
#   bash models/rfd3/scripts/merge_sweep_reports.sh /tmp/m $LOG/sweep_A $LOG/sweep_B
#
# Notes:
#   * Give the sweep dir that contains `train/baseline` the highest priority
#     (first, or newest) so the paired-delta table can be produced.
#   * Run this only after the jobs have finished; tags without a validation CSV
#     are listed under "Skipped settings" in paper_summary.md.
#   * The summary is written into OUT_DIR; the source dirs are never modified.

set -uo pipefail

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "cannot cd to ${REPO_ROOT}" >&2; exit 2; }

if [[ $# -lt 2 ]]; then
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

OUT_DIR="$1"
shift
SWEEP_DIRS=()

if [[ "${1:-}" == "--discover" ]]; then
  DISCOVER_ROOT="${2:-/backup01/zzj/logs/train_nmf_zkp_pdb}"
  if [[ ! -d "${DISCOVER_ROOT}" ]]; then
    echo "ERROR: --discover root does not exist: ${DISCOVER_ROOT}" >&2
    exit 1
  fi
  # newest first so the most recent (likely complete) run wins on duplicates
  while IFS= read -r dir; do
    [[ -d "${dir}/train" ]] && SWEEP_DIRS+=("${dir}")
  done < <(ls -1dt "${DISCOVER_ROOT}"/sweep_nmf_zkp_pdb_* 2>/dev/null)
else
  SWEEP_DIRS=("$@")
fi

if [[ ${#SWEEP_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no sweep directories to merge." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/train"
echo "Merging ${#SWEEP_DIRS[@]} sweep dir(s) into ${OUT_DIR}"

for dir in "${SWEEP_DIRS[@]}"; do
  if [[ ! -d "${dir}/train" ]]; then
    echo "  SKIP (no train/): ${dir}" >&2
    continue
  fi
  for f in "${dir}"/*.run_summary.json "${dir}"/*.nmf_replacements.json "${dir}"/*.exit_code; do
    [[ -e "${f}" ]] && { cp -f "${f}" "${OUT_DIR}/" 2>/dev/null || true; }
  done
  # Per-tag priority is "first sweep dir wins": a duplicated tag (e.g. encoder
  # layers present in both the encoder-only and the LAYER_SET=all run) keeps the
  # copy from the highest-priority dir, which should be the one with baseline.
  ( cd "${dir}/train" \
    && find . -type f \( -name validation_output_all_epochs.csv -o -name metrics.csv \) -print0 \
    | while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        [[ -e "${OUT_DIR}/train/${rel}" ]] && continue
        mkdir -p "${OUT_DIR}/train/$(dirname "${rel}")"
        cp -f "${dir}/train/${rel}" "${OUT_DIR}/train/${rel}"
      done )
  echo "  ${dir}: $(ls -1 "${dir}/train" 2>/dev/null | wc -l) tag dir(s) seen"
done

echo "Merged tags: $(ls -1 "${OUT_DIR}/train" 2>/dev/null | wc -l)"
echo "  $(ls -1 "${OUT_DIR}/train" 2>/dev/null | tr '\n' ' ')"

PYTHON="${PYTHON:-python}"
if [[ -x "/backup01/zzj/rc-cu128/bin/python" ]] && ! "${PYTHON}" -c 'import pandas' >/dev/null 2>&1; then
  PYTHON="/backup01/zzj/rc-cu128/bin/python"
fi

echo
echo "Regenerating the combined report with ${PYTHON}"
"${PYTHON}" models/rfd3/scripts/summarize_nmf_zkp_pdb_sweep.py "${OUT_DIR}" --seed "${SEED:-42}" || {
  echo "WARNING: summarize failed (is pandas installed in ${PYTHON}?)" >&2; }
"${PYTHON}" models/rfd3/scripts/plot_nmf_zkp_pdb_metrics.py "${OUT_DIR}" --ckpt "${CKPT:-}" || \
  echo "WARNING: plotting failed; the CSV/MD tables above are still valid." >&2

echo
echo "Combined report:"
echo "  ${OUT_DIR}/paper_summary.md          (per-setting lDDT + bootstrap CI)"
echo "  ${OUT_DIR}/paper_paired_deltas.csv   (baseline-paired deltas; needs a baseline tag)"
echo "  ${OUT_DIR}/paper_per_example_lddt.csv"
echo "  ${OUT_DIR}/paper_metrics.md"
echo "  ${OUT_DIR}/figures/paper_training_curves.pdf"
