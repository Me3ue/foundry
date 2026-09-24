#!/usr/bin/env bash
# One-click replacement-NMF sweep: one linear layer per job, trained on PDB.
#
# Each job exact-matches a single 2D Linear and replaces only that layer
# (nmf.match_mode=exact, nmf.max_replacements=1). Parent modules such as
# Transition are never replaced as a group.
#
# LAYER_SET:
#   encoder  input-shaping linears (default)
#   proj     token/atom projection linears
#   head     output-head linears
#   all      union of the three families
#   one      a single LAYER=module.path job
#
# Optional:
#   INCLUDE_BASELINE=1  also run a no-NMF PDB baseline
#
# Usage:
#   bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh
#   LAYER_SET=one LAYER=token_initializer.process_pll bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh
# Optional env:
#   DATA=... PARQUET=... PDB_MIRROR=... LOG_ROOT=... CKPT=... PYTHON=python SEED=42 MAX_EPOCHS=5
#   GPU=2  (default: use idle RTX A6000 GPU 2; set GPU=4 if needed)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
# Training/preflight must import the in-repo rfd3/foundry sources, not the
# stale copies installed into the conda env's site-packages.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/models/rfd3/src:${REPO_ROOT}/models/rfd3na/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA="${DATA:-/dev/shm/pdb_metadata_latest}"
PARQUET="${PARQUET:-/dev/shm/pdb_metadata_latest}"
# CIF/PDB mirror is separate from metadata parquet. The metadata dir only has
# interfaces_df.parquet / pn_units_df.parquet / rfd3_latest.ckpt.
PDB_MIRROR="${PDB_MIRROR:-/dev/shm/pdb_mirror}"
# AtomWorks resolves the chemical component dictionary through this environment variable.
export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-/dev/shm/ccd_mirror}"
export CCD_PATH="${CCD_PATH:-/dev/shm/ccd_mirror}"
# H-bond featurization (calculate_hbonds=0.2) shells out to HBPLUS. The binary
# on this machine lives under /root, not the old /home/zhangzijian path baked into env.
export HBPLUS_PATH="${HBPLUS_PATH:-/root/protein/HBPLUS/hbplus/hbplus}"
# GPU 2 is currently idle on the supplied machine. Override with GPU=4 if
# another free A6000 is preferred; do not use busy GPUs 0, 1, 3, or 5.
GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="${GPU}"
LOG_ROOT="${LOG_ROOT:-/root/protein/foundry/logs/train_nmf_zkp_pdb}"
CKPT="${CKPT:-/dev/shm/pdb_metadata_latest/rfd3_latest.ckpt}"
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
  "save_checkpoints=false"
  "trainer.checkpoint_every_n_epochs=1000000000"
  "datasets.diffusion_batch_size_train=${DIFFUSION_BS:-4}"
  "datasets.crop_size=${CROP_SIZE:-256}"
  "datasets.max_atoms_in_crop=${MAX_ATOMS:-1920}"
  "trainer.n_examples_per_epoch=${N_EXAMPLES:-128}"
  # Disable periodic validation; train_lora.py performs exactly one explicit
  # validation after fit completes.
  "trainer.validate_every_n_epochs=1000000000"
  "dataloader.train.dataloader_params.num_workers=${NUM_WORKERS:-2}"
  "dataloader.train.dataloader_params.prefetch_factor=${PREFETCH:-2}"
)

LAYER_SET="${LAYER_SET:-encoder}"
LAYER="${LAYER:-}"

layer_jobs_for() {
  local family="$1"
  case "${family}" in
    encoder)
      printf '%s\n' \
        "process_s_init|token_initializer.process_s_init.1" \
        "process_z_init|token_initializer.process_z_init.1" \
        "process_c|diffusion_module.process_c.1" \
        "process_s_trunk|token_initializer.process_s_trunk.1" \
        "transition_post_token_l1|token_initializer.transition_post_token.linear_1" \
        "transition_post_atom_l1|token_initializer.transition_post_atom.linear_1"
      ;;
    proj)
      printf '%s\n' \
        "process_pll|token_initializer.process_pll" \
        "project_pll|token_initializer.project_pll" \
        "process_n_atom|diffusion_module.process_n.0.1" \
        "process_n_token|diffusion_module.process_n.1.1" \
        "process_single_l|token_initializer.process_single_l.1" \
        "process_z|token_initializer.process_z.1"
      ;;
    head)
      printf '%s\n' \
        "to_r_update|diffusion_module.to_r_update.1" \
        "sequence_head|diffusion_module.sequence_head.linear"
      ;;
    *)
      echo "Unknown LAYER_SET family: ${family}" >&2
      return 1
      ;;
  esac
}

JOBS=()
if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
  JOBS+=("baseline|nmf_zkp_pdb|name=baseline +nmf.enabled=false ckpt_config.path=${CKPT} ckpt_config.reset_optimizer=true")
fi

append_layer_jobs() {
  local family="$1"
  local spec tag path
  while IFS= read -r spec; do
    [[ -z "${spec}" ]] && continue
    IFS='|' read -r tag path <<<"${spec}"
    JOBS+=("${tag}|nmf_zkp_single_layer_pdb|name=${tag} nmf.target_keywords=[${path}] nmf.match_mode=exact nmf.max_replacements=1 nmf.apply_to_token_initializer=true")
  done < <(layer_jobs_for "${family}")
}

case "${LAYER_SET}" in
  encoder) append_layer_jobs encoder ;;
  proj) append_layer_jobs proj ;;
  head) append_layer_jobs head ;;
  all)
    append_layer_jobs encoder
    append_layer_jobs proj
    append_layer_jobs head
    ;;
  one)
    if [[ -z "${LAYER}" ]]; then
      echo "LAYER_SET=one requires LAYER=<module.path>, e.g. LAYER=token_initializer.process_pll" >&2
      exit 1
    fi
    tag="$(echo "${LAYER}" | tr './' '__')"
    JOBS+=("${tag}|nmf_zkp_single_layer_pdb|name=${tag} nmf.target_keywords=[${LAYER}] nmf.match_mode=exact nmf.max_replacements=1 nmf.apply_to_token_initializer=true")
    ;;
  *)
    echo "Unknown LAYER_SET=${LAYER_SET}. Use encoder, proj, head, all, or one." >&2
    exit 1
    ;;
esac

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

preferred = ["baseline"]
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
    "process_s_init": "input-integrity M",
    "process_z_init": "input-integrity M",
    "process_c": "input-integrity M",
    "process_s_trunk": "input-integrity M",
    "transition_post_token_l1": "input-integrity M",
    "transition_post_atom_l1": "input-integrity M",
    "process_pll": "intermediate-state M",
    "project_pll": "intermediate-state M",
    "process_n_atom": "intermediate-state M",
    "process_n_token": "intermediate-state M",
    "process_single_l": "intermediate-state M",
    "process_z": "intermediate-state M",
    "to_r_update": "output-consistency M",
    "sequence_head": "output-consistency M",
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
    "- Each job replaces **exactly one** Linear via `match_mode=exact` and `max_replacements=1`.",
    "- Encoder defaults: `process_s_init.1`, `process_z_init.1`, `process_c.1`, `process_s_trunk.1`, `transition_post_token.linear_1`, `transition_post_atom.linear_1`.",
    "- Projection defaults: `process_pll`, `project_pll`, `process_n.{0,1}.1`, `process_single_l.1`, `process_z.1`.",
    "- Head defaults: `to_r_update.1`, `sequence_head.linear`.",
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
        "- Training uses the PDB dataset, not `single_structure_data`.",
        "- One Linear is replaced per job. Parent Transition / Sequential names are not expanded.",
        "- The middle square `M` of that single layer is the intended ZKP proof target.",
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
echo "  ${SWEEP_DIR}/paper_summary.csv"
echo "  ${SWEEP_DIR}/paper_paired_deltas.csv"
echo "  ${SWEEP_DIR}/paper_per_example_lddt.csv"

echo "Training cost summary:"
echo "  ${SWEEP_DIR}/paper_training_cost.csv"
echo "  ${SWEEP_DIR}/paper_training_cost.md"

if [[ "${overall_failed}" != "0" ]]; then
  echo "Sweep finished with one or more failed jobs; see *.run.log and *.exit_code." >&2
  exit 1
fi
