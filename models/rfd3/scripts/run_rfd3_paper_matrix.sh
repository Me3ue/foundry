#!/usr/bin/env bash
# Reproducible RFD3 constrained-reconstruction experiment matrix.
# 13 targets x 6 conditions x 3 seeds x 8 samples = 1,872 final models.
# Each final model has noisy and denoised CIF trajectories.
set -Eeuo pipefail

ROOT="${ROOT:-/backup01/zzj/protein/foundry}"
CKPT="${CKPT:-/dev/shm/rfd3_latest.ckpt}"
HBPLUS="${HBPLUS:-/home/zhangzijian/protein/HBPLUS/hbplus/hbplus}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/rfd3_paper_matrix}"
PDB_DIR="${PDB_DIR:-${ROOT}/inputs/pdb}"
PYTHON="${PYTHON:-python}"
RUNNER="${ROOT}/models/rfd3/scripts/run_near_native_rfd3.py"
EVALUATOR="${ROOT}/models/rfd3/scripts/evaluate_near_native_rfd3.py"
# Run the proven small target first as an end-to-end smoke test. Override the
# full ordered target list through TARGETS_OVERRIDE="1QLX 5OQV ..." if needed.
if [[ -n "${TARGETS_OVERRIDE:-}" ]]; then
  read -r -a TARGETS <<< "$TARGETS_OVERRIDE"
else
  TARGETS=(1QLX 5OQV 1ABR 2AAI 4UY2 7UMQ 5O3L 6A6B 1MDT 1DM0 3BTA 5N0B 4JTA)
fi
SEEDS=(20250308 20250309 20250310)
SAMPLES="${SAMPLES:-8}"
TIMESTEPS="${TIMESTEPS:-200}"

# The runner defaults to chunked pairwise inference (--low-memory-mode) so the
# same scientific configuration can run on both 80 GB and smaller GPUs. Set
# SAMPLES=1 for a non-scientific memory smoke test, then restore SAMPLES=8.
CONDITIONS=(
  "C1_strong_noHB|1.0|5|off"
  "C2_strong_HB|1.0|5|on"
  "C3_medium_noHB|2.0|10|off"
  "C4_medium_HB|2.0|10|on"
  "C5_weak_noHB|5.0|20|off"
  "C6_weak_HB|5.0|20|on"
  "C7_noanchor_noHB|2.0|0|off"
  "C8_noanchor_HB|2.0|0|on"
)

[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
[[ -x "$HBPLUS" ]] || { echo "Missing HBPLUS: $HBPLUS" >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi
mkdir -p "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for target in "${TARGETS[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    IFS='|' read -r name partial_t anchor_stride hbond <<< "$condition"
    for seed in "${SEEDS[@]}"; do
      run_dir="$OUT_ROOT/$target/$name/seed_$seed"
      mkdir -p "$run_dir"
      echo "=== target=$target condition=$name seed=$seed ==="
      cmd=("$PYTHON" "$RUNNER" --ids "$target" --out-dir "$run_dir"
           --structures-dir "$PDB_DIR" --checkpoint "$CKPT"
           --designs-per-target "$SAMPLES" --partial-t "$partial_t" --sequence-mode fixed
           --anchor-stride "$anchor_stride" --timesteps "$TIMESTEPS" --seed "$seed" --low-memory-mode)
      [[ "$hbond" == on ]] && cmd+=(--hbplus "$HBPLUS")
      "${cmd[@]}" 2>&1 | tee "$run_dir/run.log"
      "$PYTHON" "$EVALUATOR" --run-dir "$run_dir" 2>&1 | tee "$run_dir/evaluate.log"
    done
  done
done
