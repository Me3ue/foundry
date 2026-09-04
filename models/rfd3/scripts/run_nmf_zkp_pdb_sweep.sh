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
#   DATA=... PARQUET=... LOG_ROOT=... CKPT=... PYTHON=python SEED=42 MAX_EPOCHS=5

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/media/zzj/Data/pdb_metadata_latest}"
PARQUET="${PARQUET:-${DATA}}"
LOG_ROOT="${LOG_ROOT:-/home/zzj/protein/foundry/logs/train_nmf_zkp_pdb}"
CKPT="${CKPT:-/media/zzj/Data/pdb_metadata_latest/rfd3_latest.ckpt}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
INCLUDE_ALL="${INCLUDE_ALL:-0}"
SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_DIR="${SWEEP_DIR:-${LOG_ROOT}/sweep_nmf_zkp_pdb_${SWEEP_STAMP}}"
TABLE_OUT="${TABLE_OUT:-${SWEEP_DIR}/comparison_table.md}"
CSV_OUT="${CSV_OUT:-${SWEEP_DIR}/comparison_table.csv}"
mkdir -p "$SWEEP_DIR"

COMMON_OVERRIDES=(
  "paths.data.pdb_data_dir=${DATA}"
  "paths.data.pdb_parquet_dir=${PARQUET}"
  "paths.log_dir=${SWEEP_DIR}"
  "logger=csv"
  "seed=${SEED}"
  "trainer.max_epochs=${MAX_EPOCHS}"
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

  echo "${rc}" > "${SWEEP_DIR}/${tag}.exit_code"
  echo "[$(date '+%F %T')] END ${tag} exit=${rc}"
  return 0
}

for job in "${JOBS[@]}"; do
  IFS='|' read -r tag experiment extras <<<"${job}"
  run_one "${tag}" "${experiment}" "${extras}"
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
    "trainable_params",
    "total_params",
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
        "best_val_lddt": fmt(s.get("best_val_lddt")),
        "best_val_loss": fmt(s.get("best_val_loss")),
        "trainable_params": s.get("trainable_params"),
        "total_params": s.get("total_params"),
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

echo ""
echo "Done. Comparison table:"
echo "  ${TABLE_OUT}"
echo "  ${CSV_OUT}"
