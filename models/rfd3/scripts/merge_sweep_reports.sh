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
# Options:
#   --discover [ROOT]   merge every sweep_nmf_zkp_pdb_* under ROOT (default
#                       /backup01/zzj/logs/train_nmf_zkp_pdb)
#   --drop-incomplete   skip tags without a validation CSV. Use this when a run
#                       was killed mid-sweep and you do not want the truncated
#                       training curves of the jobs that never finished.
#   --exclude A,B       skip tags matching these names/globs (comma separated),
#                       e.g. --exclude 'process_pll,project_pll'
#
# Examples:
#   LOG=/backup01/zzj/logs/train_nmf_zkp_pdb
#   bash models/rfd3/scripts/merge_sweep_reports.sh "$LOG/sweep_combined" --discover "$LOG"
#   bash models/rfd3/scripts/merge_sweep_reports.sh /tmp/m $LOG/sweep_A $LOG/sweep_B
#   # a LAYER_SET=all run killed after the baseline: keep baseline, drop the
#   # half-finished proj jobs, and take encoder/head from their own runs
#   bash models/rfd3/scripts/merge_sweep_reports.sh "$LOG/sweep_combined" \
#     --drop-incomplete "$LOG/sweep_enc" "$LOG/sweep_head" "$LOG/sweep_all"
#
# Notes:
#   * Run this only after the jobs have finished.
#   * Duplicated tags (e.g. encoder layers present in both an encoder-only run
#     and a LAYER_SET=all run) are resolved per tag: a finished copy always wins
#     over a truncated one, then the earliest dir on the command line wins (for
#     --discover, the newest sweep dir). So the dir holding `train/baseline` does
#     not need special handling -- baseline is only produced by one run.
#   * Tags that are kept without a validation CSV are listed under "Skipped
#     settings" in paper_summary.md, but their training curves still appear in
#     paper_metrics.md and figures/ unless you pass --drop-incomplete.
#   * The summary is written into OUT_DIR; the source dirs are never modified.

set -uo pipefail

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "cannot cd to ${REPO_ROOT}" >&2; exit 2; }

if [[ $# -lt 2 ]]; then
  sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

EXCLUDE_PATTERNS=()
DROP_INCOMPLETE=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --drop-incomplete)
      DROP_INCOMPLETE=1; shift ;;
    --exclude)
      IFS=',' read -r -a _pats <<<"${2:-}"; EXCLUDE_PATTERNS+=("${_pats[@]}"); shift 2 ;;
    --exclude=*)
      IFS=',' read -r -a _pats <<<"${1#*=}"; EXCLUDE_PATTERNS+=("${_pats[@]}"); shift ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"}

tag_excluded() {
  local tag="$1" pat
  for pat in ${EXCLUDE_PATTERNS[@]+"${EXCLUDE_PATTERNS[@]}"}; do
    # Unquoted on purpose: --exclude accepts globs such as 'process_*'.
    # shellcheck disable=SC2254
    case "${tag}" in ${pat}) return 0 ;; esac
  done
  return 1
}

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

# ---------------------------------------------------------------------------
# Pick ONE source dir per tag. The tag directory name is stable, but the run
# directory inside it (train/<tag>/<timestamp>_<job>/...) is not, so a
# file-level "skip if the destination exists" test never fires for duplicates
# and both copies would land in the merged dir -- the summarizer's rglob then
# silently picked the last one in sort order rather than the intended dir.
# Selection rule instead:
#   1. a dir whose copy has a validation CSV (a finished job) beats one without;
#   2. ties and all-incomplete cases go to the earliest dir on the command line
#      (for --discover, the newest sweep dir).
# This matters when a run was killed mid-sweep: a truncated job in one dir must
# never shadow the complete copy of the same layer in another.
# ---------------------------------------------------------------------------
declare -A TAG_SOURCE=()   # tag -> chosen dir
declare -A TAG_COMPLETE=() # tag -> 1 if the chosen copy has a validation CSV
TAG_ORDER=()

for dir in "${SWEEP_DIRS[@]}"; do
  [[ -d "${dir}/train" ]] || continue
  for tag_dir in "${dir}"/train/*/; do
    [[ -d "${tag_dir}" ]] || continue
    tag="$(basename "${tag_dir}")"
    tag_excluded "${tag}" && continue
    if find "${tag_dir}" -name validation_output_all_epochs.csv -print -quit | grep -q .; then
      complete=1
    else
      complete=0
    fi
    if [[ -z "${TAG_SOURCE[${tag}]:-}" ]]; then
      TAG_SOURCE["${tag}"]="${dir}"
      TAG_COMPLETE["${tag}"]="${complete}"
      TAG_ORDER+=("${tag}")
    elif [[ "${complete}" == "1" && "${TAG_COMPLETE[${tag}]}" == "0" ]]; then
      TAG_SOURCE["${tag}"]="${dir}"
      TAG_COMPLETE["${tag}"]="1"
    fi
  done
done

incomplete=()
skipped_incomplete=()
for tag in "${TAG_ORDER[@]}"; do
  dir="${TAG_SOURCE[${tag}]}"
  if [[ "${TAG_COMPLETE[${tag}]}" == "0" ]]; then
    incomplete+=("${tag}")
    if [[ "${DROP_INCOMPLETE}" == "1" ]]; then
      skipped_incomplete+=("${tag}")
      printf '  %-28s %-10s %s\n' "${tag}" "DROPPED" "${dir}"
      continue
    fi
    printf '  %-28s %-10s %s\n' "${tag}" "PARTIAL" "${dir}"
  else
    printf '  %-28s %-10s %s\n' "${tag}" "ok" "${dir}"
  fi
  ( cd "${dir}/train/${tag}" \
    && find . -type f \( -name validation_output_all_epochs.csv -o -name metrics.csv \) -print0 \
    | while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        mkdir -p "${OUT_DIR}/train/${tag}/$(dirname "${rel}")"
        cp -f "${dir}/train/${tag}/${rel}" "${OUT_DIR}/train/${tag}/${rel}"
      done )
  # Top-level per-tag artefacts come from the same dir that provided the tag.
  for f in "${dir}/${tag}.run_summary.json" "${dir}/${tag}.nmf_replacements.json" \
           "${dir}/${tag}.exit_code" "${dir}/${tag}.run.log"; do
    [[ -e "${f}" ]] && cp -f "${f}" "${OUT_DIR}/"
  done
done

echo "Merged tags: $(ls -1 "${OUT_DIR}/train" 2>/dev/null | wc -l)"
echo "  $(ls -1 "${OUT_DIR}/train" 2>/dev/null | tr '\n' ' ')"
if [[ ${#incomplete[@]} -gt 0 ]]; then
  echo
  if [[ ${#skipped_incomplete[@]} -gt 0 ]]; then
    echo "Dropped ${#skipped_incomplete[@]} incomplete tag(s) (--drop-incomplete): ${skipped_incomplete[*]}"
  fi
  kept_incomplete=()
  for t in "${incomplete[@]}"; do
    tag_excluded "${t}" && continue
    [[ " ${skipped_incomplete[*]-} " == *" ${t} "* ]] && continue
    kept_incomplete+=("${t}")
  done
  if [[ ${#kept_incomplete[@]} -gt 0 ]]; then
    echo "WARNING: ${#kept_incomplete[@]} tag(s) have no validation CSV (killed or failed mid-run):"
    echo "  ${kept_incomplete[*]}"
    echo "  They appear in paper_metrics.md / the training-curve figure but are"
    echo "  reported as \"Skipped settings\" in paper_summary.md. Pass"
    echo "  --drop-incomplete (or --exclude) to leave them out entirely."
  fi
fi

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
