#!/usr/bin/env bash
# One-click replacement-NMF sweep on ZKP-suitable layers, trained on PDB.
#
# Groups follow models/rfd3/docs/rfd3_structure_and_nmf_analysis.md:
#   1) zkp_encoder  input-shaping / encoder interface  (best input-integrity M)
#   2) zkp_proj     token/atom projection mixers       (intermediate-state M)
#   3) zkp_head     output heads                       (output-consistency M)
#
# Optional:
#   INCLUDE_BASELINE=1  also run a no-NMF PDB baseline
#   INCLUDE_ALL=1       also run the union of all ZKP-suitable layers
#
# Usage:
#   bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh
# Optional env:
#   DATA=... PARQUET=... PDB_MIRROR=... LOG_ROOT=... CKPT=... PYTHON=python SEED=42 MAX_EPOCHS=5

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
# Training/preflight must import the in-repo rfd3/foundry sources, not the
# stale copies installed into the conda env's site-packages.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/models/rfd3/src:${REPO_ROOT}/models/rfd3na/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA="${DATA:-/media/zzj/Data/pdb_metadata_latest}"
PARQUET="${PARQUET:-${DATA}}"
# CIF/PDB mirror is separate from metadata parquet. The metadata dir only has
# interfaces_df.parquet / pn_units_df.parquet / rfd3_latest.ckpt.
PDB_MIRROR="${PDB_MIRROR:-/media/zzj/Data/pdb_mirror}"
# H-bond featurization (calculate_hbonds=0.2) shells out to HBPLUS. The binary
# on this machine lives under /root, not the old /home/zzj path baked into env.
export HBPLUS_PATH="${HBPLUS_PATH:-/root/protein/HBPLUS/hbplus/hbplus}"
LOG_ROOT="${LOG_ROOT:-/root/protein/foundry/logs/train_nmf_zkp_pdb}"
CKPT="${CKPT:-/media/zzj/Data/pdb_metadata_latest/rfd3_latest.ckpt}"
# Prefer the project rc environment when PYTHON is not explicitly supplied.
# The system/base Python may resolve to a different site-packages tree and can
# silently hide a broken dependency installation.
if [[ -z "${PYTHON+x}" ]]; then
  if [[ -x "/opt/conda/envs/rc/bin/python" ]]; then
    PYTHON="/opt/conda/envs/rc/bin/python"
  else
    PYTHON="python"
  fi
fi

# Fail once, before launching every sweep job, if the runtime environment is
# unusable.  In particular, pandas is imported by foundry's logging module.
if ! "${PYTHON}" - <<'PY'
import pandas
import foundry
import rfd3
print(f"Using Python: {__import__('sys').executable}")
print(f"Using pandas: {pandas.__file__}")
print(f"Using foundry: {foundry.__file__}")
print(f"Using rfd3: {rfd3.__file__}")
PY
then
  echo "ERROR: Python environment failed the pandas/foundry/rfd3 import check." >&2
  echo "Repair the environment or rerun with PYTHON=/path/to/python." >&2
  exit 1
fi

SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-580}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
INCLUDE_ALL="${INCLUDE_ALL:-0}"
SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_DIR="${SWEEP_DIR:-${LOG_ROOT}/sweep_nmf_zkp_pdb_${SWEEP_STAMP}}"
TABLE_OUT="${TABLE_OUT:-${SWEEP_DIR}/comparison_table.md}"
CSV_OUT="${CSV_OUT:-${SWEEP_DIR}/comparison_table.csv}"
mkdir -p "$SWEEP_DIR"

# 128 examples/epoch gives a less noisy estimate of the training distribution
# while keeping the four-setting sweep tractable. Override with N_EXAMPLES for
# a controlled ablation.
COMMON_OVERRIDES=(
  "paths.data.pdb_data_dir=${PDB_MIRROR}"
  "paths.data.pdb_parquet_dir=${PARQUET}"
  "paths.log_dir=${SWEEP_DIR}"
  "logger=csv"
  "seed=${SEED}"
  "trainer.max_epochs=${MAX_EPOCHS}"
  "datasets.diffusion_batch_size_train=${DIFFUSION_BS:-4}"
  "datasets.crop_size=${CROP_SIZE:-256}"
  "datasets.max_atoms_in_crop=${MAX_ATOMS:-1920}"
  "trainer.n_examples_per_epoch=${N_EXAMPLES:-128}"
  # Disable periodic validation; train_lora.py performs exactly one explicit
  # validation after fit completes.
  "trainer.validate_every_n_epochs=1000000000"
  "dataloader.train.dataloader_params.num_workers=${NUM_WORKERS:-8}"
  "dataloader.train.dataloader_params.prefetch_factor=${PREFETCH:-4}"
)

JOBS=()
if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
  JOBS+=("baseline|nmf_zkp_pdb|name=baseline ckpt_config.path=${CKPT} ckpt_config.reset_optimizer=true")
fi
JOBS+=(
  "zkp_encoder|nmf_zkp_encoder_pdb|name=zkp_encoder"
  "zkp_proj|nmf_zkp_proj_pdb|name=zkp_proj"
  "zkp_head|nmf_zkp_head_pdb|name=zkp_head"
)
if [[ "${INCLUDE_ALL}" == "1" ]]; then
  JOBS+=("zkp_all|nmf_zkp_all_pdb|name=zkp_all")
fi

run_one() {
  local tag="$1"
  local experiment="$2"
  local extras="$3"
  local run_log="${SWEEP_DIR}/${tag}.run.log"
  echo "============================================================"
  echo "[$(date '+%F %T')] START ${tag}  experiment=${experiment}"
  echo "============================================================"

  local extra_arr=()
  if [[ -n "${extras}" ]]; then
    # shellcheck disable=SC2206
    extra_arr=(${extras})
  fi

  set +e
  ${PYTHON} models/rfd3/src/rfd3/train_lora.py \
    "experiment=${experiment}" \
    "${COMMON_OVERRIDES[@]}" \
    "${extra_arr[@]}" \
    2>&1 | tee "${run_log}"
  local rc=${PIPESTATUS[0]}
  set -e

  # Require every configured job to produce a successful holdout evaluation.
  # A zero exit code without val_metrics is not a valid paper-comparison run.
  local validation_dir="${SWEEP_DIR}/train/${tag}"
  if [[ "${rc}" == "0" ]] && ! find "${validation_dir}" -type f -path '*/val_metrics/validation_output_all_epochs.csv' -print -quit | grep -q .; then
    echo "${tag}: missing validation_output_all_epochs.csv; marking job failed" >&2
    rc=2
  fi
  echo "${rc}" > "${SWEEP_DIR}/${tag}.exit_code"
  echo "[$(date '+%F %T')] END ${tag} exit=${rc}"
  return 0
}

overall_failed=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag experiment extras <<<"${job}"
  run_one "${tag}" "${experiment}" "${extras}"
  job_exit="$(cat "${SWEEP_DIR}/${tag}.exit_code")"
  if [[ "${job_exit}" != "0" ]]; then
    overall_failed=1
  fi
done

"${PYTHON}" - <<'PY' "$SWEEP_DIR" "$TABLE_OUT" "$CSV_OUT" "$CKPT"
import csv
import json
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
table_out = Path(sys.argv[2])
csv_out = Path(sys.argv[3])
ckpt = sys.argv[4]

preferred = ["baseline", "zkp_encoder", "zkp_proj", "zkp_head", "zkp_all"]
found = [p.name.replace(".exit_code", "") for p in sorted(sweep_dir.glob("*.exit_code"))]
tags = [t for t in preferred if t in found] + [t for t in found if t not in preferred]


def load_summary(tag: str):
    candidates = [sweep_dir / f"{tag}.run_summary.json", *sweep_dir.glob("*.run_summary.json")]
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        name = str(data.get("name", ""))
        if p.name.startswith(tag) or tag in name or tag.replace("_", "-") in name:
            data["_summary_path"] = str(p)
            return data
    exit_code = (sweep_dir / f"{tag}.exit_code").read_text().strip() if (sweep_dir / f"{tag}.exit_code").exists() else "?"
    return {
        "status": "failed" if exit_code not in {"0", ""} else "missing",
        "trainable_params": None,
        "total_params": None,
        "best_val_lddt": None,
        "best_val_loss": None,
        "_summary_path": "summary_json_missing",
    }


def fmt(v, digits=4):
    if v is None or v == "":
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def load_replacements(tag: str) -> str:
    for p in [
        sweep_dir / f"{tag}.nmf_replacements.json",
        sweep_dir / f"nmf-zkp-{tag.replace('zkp_', '')}-pdb.nmf_replacements.json",
        *sweep_dir.glob(f"*{tag}*.nmf_replacements.json"),
    ]:
        if not p.exists():
            continue
        try:
            rep = json.loads(p.read_text())
        except Exception:
            continue
        records = rep.get("records", [])
        if not records:
            return str(p)
        return " ; ".join(
            f"{x.get('module_name')}[{x.get('module_type')}] "
            f"in={x.get('in_features')} out={x.get('out_features')} "
            f"rank={x.get('rank')} params={x.get('nmf_params')}"
            for x in records
        )
    return ""


headers = [
    "setting",
    "status",
    "best_val_lddt",
    "best_val_loss",
    "final_train_loss",
    "best_train_loss",
    "final_train_lddt",
    "best_train_lddt",
    "final_seq_recovery",
    "final_coordinate_mse",
    "trainable_params",
    "total_params",
    "trainable_fraction",
    "proof_family",
    "replaced_layers",
]
family = {
    "baseline": "none",
    "zkp_encoder": "input-integrity M",
    "zkp_proj": "intermediate-state M",
    "zkp_head": "output-consistency M",
    "zkp_all": "union of ZKP-suitable layers",
}

rows = []
md = [
    "# ZKP-oriented NMF PDB sweep",
    "",
    f"- Baseline checkpoint: `{ckpt}`",
    f"- Sweep dir: `{sweep_dir}`",
    "",
    "## Layer groups",
    "",
    "- `zkp_encoder`: `transition_post_token`, `transition_post_atom`, `process_s_init`, `process_z_init`, `process_c`, `process_s_trunk`",
    "- `zkp_proj`: `process_pll`, `project_pll`, `upcast.project`, `downcast.project`, `process_n`",
    "- `zkp_head`: `to_r_update`, `sequence_head`",
    "- excluded: `to_q` / `to_k` / `to_v` / `to_b` / `to_g`, `process_r`, `process_a`",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]

for tag in tags:
    s = load_summary(tag)
    exit_code = (sweep_dir / f"{tag}.exit_code").read_text().strip() if (sweep_dir / f"{tag}.exit_code").exists() else ""
    status = s.get("status") or ("success" if exit_code == "0" else f"exit={exit_code}")
    row = {
        "setting": tag,
        "status": status,
        # These remain empty when the sweep intentionally has no validation
        # loader. Never substitute training metrics into validation fields.
        "best_val_lddt": fmt(s.get("best_val_lddt")),
        "best_val_loss": fmt(s.get("best_val_loss")),
        "final_train_loss": fmt(s.get("final_train_loss") or s.get("metrics", {}).get("train/per_epoch_total_loss")),
        "best_train_loss": fmt(s.get("best_epoch_loss")),
        "final_train_lddt": fmt(s.get("final_train_lddt") or s.get("metrics", {}).get("train/per_epoch_mean_lddt_protein")),
        "best_train_lddt": fmt(s.get("best_epoch_lddt")),
        "final_seq_recovery": fmt(s.get("final_train_seq_recovery") or s.get("metrics", {}).get("train/per_epoch_seq_recovery")),
        "final_coordinate_mse": fmt(s.get("final_train_coordinate_mse") or s.get("metrics", {}).get("train/per_epoch_mse_loss_mean")),
        "trainable_params": s.get("trainable_params"),
        "total_params": s.get("total_params"),
        "trainable_fraction": fmt(
            (s.get("trainable_params") / s.get("total_params"))
            if s.get("trainable_params") is not None and s.get("total_params")
            else None
        ),
        "proof_family": family.get(tag, ""),
        "replaced_layers": load_replacements(tag),
    }
    rows.append(row)
    md.append("| " + " | ".join("" if row.get(h) is None else str(row.get(h)) for h in headers) + " |")

md.extend(
    [
        "",
        "## Notes",
        "",
        "- Training uses the PDB interface dataset (`rfd3_train_interface`), not `single_structure_data`.",
        "- Encoder-group NMF is injected into the full model (`apply_to_token_initializer=true`).",
        "- Head-group NMF stays on `diffusion_module` only.",
        "- The middle square `M` of each replaced layer is the intended ZKP proof target.",
        "- Validation metrics are intentionally blank for this training-only sweep; do not interpret training metrics as generalization metrics.",
        "- The final holdout validation runs once after training; periodic validation is disabled.",
        "- The generated paper_summary.csv/.md report per-example lDDT mean, SD, median, bootstrap 95% CI, paired baseline deltas, and coverage.",
        "",
    ]
)

table_out.write_text("\n".join(md) + "\n")
with csv_out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=headers)
    w.writeheader()
    w.writerows(rows)
print("\n".join(md))
print(f"\nWrote: {table_out}")
print(f"Wrote: {csv_out}")
PY

# Build paper-level statistics from the per-example holdout CSVs. This is
# intentionally run after all jobs, and never substitutes training metrics for
# validation metrics.
if ! "${PYTHON}" models/rfd3/scripts/summarize_nmf_zkp_pdb_sweep.py "${SWEEP_DIR}" --seed "${SEED}"; then
  echo "Paper summary generation failed; validation CSVs may be missing lDDT." >&2
  overall_failed=1
fi

"${PYTHON}" models/rfd3/scripts/plot_nmf_zkp_pdb_metrics.py "${SWEEP_DIR}" --ckpt "${CKPT}" || true

echo ""
echo "Done. Comparison table:"
echo "  ${TABLE_OUT}"
echo "  ${CSV_OUT}"
echo "Paper figures (if metrics.csv exists):"
echo "  ${SWEEP_DIR}/figures/paper_training_curves.pdf"
echo "  ${SWEEP_DIR}/paper_metrics.md"
echo "Paper full-metric evaluation:"
echo "  ${SWEEP_DIR}/paper_summary/metric_summary.csv"
echo "  ${SWEEP_DIR}/paper_summary/paired_baseline_deltas.csv"
echo "  ${SWEEP_DIR}/paper_summary/evaluation_coverage.csv"

echo "Training cost summary:"
echo "  ${SWEEP_DIR}/paper_training_cost.csv"
echo "  ${SWEEP_DIR}/paper_training_cost.md"

if [[ "${overall_failed}" != "0" ]]; then
  echo "Sweep finished with one or more failed jobs; see *.run.log and *.exit_code." >&2
  exit 1
fi
