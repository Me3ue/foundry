#!/usr/bin/env bash
# Robust replacement-NMF sweep on single_structure + baseline comparison table.
# Runs one baseline checkpoint evaluation plus the four requested NMF targets.
# Usage (from repo root):
#   bash models/rfd3/scripts/run_nmf_single_structure_sweep.sh
# Optional env:
#   DATA=... LOG_ROOT=... CKPT=... PYTHON=python SEED=42 MAX_EPOCHS=20 TOPK=3

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/home/zzj/protein/foundry/single_structure_data}"
LOG_ROOT="${LOG_ROOT:-/home/zzj/protein/foundry/logs/train_nmf}"
CKPT="${CKPT:-/media/zzj/Data/rfd3_latest.ckpt}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
TOPK="${TOPK:-3}"
SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_DIR="${SWEEP_DIR:-${LOG_ROOT}/sweep_nmf_single_structure_${SWEEP_STAMP}}"
TABLE_OUT="${TABLE_OUT:-${SWEEP_DIR}/comparison_table.md}"
CSV_OUT="${CSV_OUT:-${SWEEP_DIR}/comparison_table.csv}"
EVAL_DIR="${EVAL_DIR:-${SWEEP_DIR}/evaluation}"
CKPT_DIR="${CKPT_DIR:-${SWEEP_DIR}/checkpoints}"
mkdir -p "$SWEEP_DIR" "$EVAL_DIR" "$CKPT_DIR"

COMMON_OVERRIDES=(
  "paths.data.monomer_distillation_parquet_dir=${DATA}"
  "paths.data.monomer_distillation_data_dir=${DATA}"
  "paths.data.design_benchmark_data_dir=${DATA}"
  "paths.log_dir=${SWEEP_DIR}"
  "logger=csv"
  "seed=${SEED}"
  "trainer.max_epochs=${MAX_EPOCHS}"
)

# name|experiment|extra hydra overrides (space-separated, optional)
JOBS=(
  "baseline|lora_finetune|lora.enabled=false ckpt_path=${CKPT}"
  "nmf_fc1_only|nmf_fc1_only_single_structure|"
  "nmf_fc2_only|nmf_fc2_only_single_structure|"
  "nmf_to_o_only|nmf_to_o_only_single_structure|"
  "nmf_to_g_only|nmf_to_g_only_single_structure|"
)

run_one() {
  local tag="$1"
  local experiment="$2"
  local extras="$3"
  local run_log="${SWEEP_DIR}/${tag}.run.log"
  local eval_json="${EVAL_DIR}/${tag}.json"
  local final_ckpt="${CKPT_DIR}/${tag}.last.ckpt"
  echo "============================================================"
  echo "[$(date '+%F %T')] START ${tag}  experiment=${experiment}"
  echo "============================================================"

  # shellcheck disable=SC2206
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

  # Best-effort checkpoint capture from the most recent run directory.
  local hydra_dir=""
  hydra_dir="$(
    find "${SWEEP_DIR}" -type f -path '*/lightning_logs/*/metrics.csv' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | awk '{print $2}' \
      | xargs -r dirname \
      | xargs -r dirname \
      | xargs -r dirname || true
  )"
  if [[ -n "${hydra_dir}" ]]; then
    local best_ckpt
    best_ckpt="$(find "${hydra_dir}" -type f \( -name '*.ckpt' -o -name 'last.ckpt' \) 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${best_ckpt}" ]]; then
      cp -f "${best_ckpt}" "${final_ckpt}" 2>/dev/null || true
      printf '%s\n' "${best_ckpt}" > "${CKPT_DIR}/${tag}.source_ckpt.txt"
    fi
  fi

  # Per-run best-effort summary.
  "${PYTHON}" - <<'PY' "${SWEEP_DIR}" "${tag}" "${eval_json}" "${rc}" "${TOPK}"
import csv
import json
import re
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
tag = sys.argv[2]
out = Path(sys.argv[3])
exit_code = int(sys.argv[4]) if sys.argv[4].isdigit() else sys.argv[4]
topk_n = int(sys.argv[5])

log_path = sweep_dir / f"{tag}.run.log"
text = log_path.read_text(errors="replace") if log_path.exists() else ""

info = {
    "tag": tag,
    "exit_code": exit_code,
    "trainable": None,
    "val_mean_lddt": None,
    "val_total_loss": None,
    "topk": [],
    "notes": [],
}

m = re.search(r"NMF enabled: trainable params=([\d,]+)", text)
if m:
    info["trainable"] = int(m.group(1).replace(",", ""))
else:
    m = re.search(r"trainable params=([\d,]+)", text)
    if m:
        info["trainable"] = int(m.group(1).replace(",", ""))

m = re.findall(r"Mean Lddt Protein\s*[│|]\s*([0-9.]+)", text)
if m:
    try:
        info["val_mean_lddt"] = float(m[-1])
    except ValueError:
        pass
m = re.findall(r"Total Loss\s*[│|]\s*([0-9.]+)", text)
if m:
    try:
        info["val_total_loss"] = float(m[-1])
    except ValueError:
        pass

for pat in ["Traceback", "CUDA out of memory", "RuntimeError", "AssertionError", "Error executing job with overrides"]:
    if pat in text:
        info["notes"].append(pat)

# Try to infer top-k from metrics.csv if present.
metrics_csv = None
cands = list(sweep_dir.glob(f"**/{tag}*/lightning_logs/*/metrics.csv"))
if not cands:
    cands = list(sweep_dir.glob("**/lightning_logs/*/metrics.csv"))
if cands:
    metrics_csv = max(cands, key=lambda p: p.stat().st_mtime)

if metrics_csv and metrics_csv.exists():
    with metrics_csv.open() as f:
        rows = list(csv.DictReader(f))
    scored = []
    for row in rows:
        epoch = row.get("epoch") or row.get("trainer/global_step") or row.get("step")
        try:
            epoch = int(float(epoch)) if epoch not in (None, "") else None
        except ValueError:
            epoch = None
        loss = row.get("val/total_loss") or row.get("train/per_epoch_total_loss")
        lddt = row.get("val/mean_lddt") or row.get("train/per_epoch_mean_lddt_protein")
        try:
            loss_f = float(loss) if loss not in (None, "") else None
        except ValueError:
            loss_f = None
        try:
            lddt_f = float(lddt) if lddt not in (None, "") else None
        except ValueError:
            lddt_f = None
        if epoch is not None and (loss_f is not None or lddt_f is not None):
            scored.append({"epoch": epoch, "loss": loss_f, "lddt": lddt_f})
    info["topk"] = sorted(
        scored,
        key=lambda r: (
            float("inf") if r["loss"] is None else r["loss"],
            float("-inf") if r["lddt"] is None else -r["lddt"],
            r["epoch"],
        ),
    )[:topk_n]

out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(info, indent=2, ensure_ascii=False))
PY

  echo "[$(date '+%F %T')] END ${tag} exit=${rc} hydra_dir=${hydra_dir:-unknown}"
  return 0
}

for job in "${JOBS[@]}"; do
  IFS='|' read -r tag experiment extras <<<"${job}"
  run_one "${tag}" "${experiment}" "${extras}"
done

"${PYTHON}" - <<'PY' "$SWEEP_DIR" "$TABLE_OUT" "$CSV_OUT" "$CKPT" "$EVAL_DIR"
import csv
import json
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
table_out = Path(sys.argv[2])
csv_out = Path(sys.argv[3])
ckpt = sys.argv[4]
eval_dir = Path(sys.argv[5])

tags = ["baseline", "nmf_fc1_only", "nmf_fc2_only", "nmf_to_o_only", "nmf_to_g_only"]


def load_eval_summary(tag: str):
    p = eval_dir / f"{tag}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"exit_code": None, "trainable": None, "val_mean_lddt": None, "val_total_loss": None, "topk": [], "notes": []}


def fmt(v, digits=4):
    if v is None or v == "":
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def delta(cur, base_v, higher_better=True):
    if cur is None or base_v is None:
        return ""
    try:
        c, b = float(cur), float(base_v)
    except (TypeError, ValueError):
        return ""
    d = c - b
    if higher_better:
        flag = "↑" if d > 1e-6 else ("↓" if d < -1e-6 else "≈")
    else:
        flag = "↓" if d < -1e-6 else ("↑" if d > 1e-6 else "≈")
    return f"{d:+.4f} {flag}"

rows = []
for tag in tags:
    s = load_eval_summary(tag)
    rows.append({"setting": tag, **s})

base = rows[0]
base_lddt = base.get("val_mean_lddt")
base_loss = base.get("val_total_loss")

headers = ["setting", "exit_code", "val_mean_lddt", "Δlddt_vs_baseline", "val_total_loss", "Δloss_vs_baseline", "trainable", "topk_epochs", "topk_val_loss", "topk_val_lddt", "replaced_layers", "notes"]
md = [
    "# NMF single-structure sweep report",
    "",
    f"- Baseline checkpoint: `{ckpt}`",
    f"- Sweep dir: `{sweep_dir}`",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]

csv_rows = []
for r in rows:
    topk_epochs = ", ".join(str(x.get("epoch")) for x in r.get("topk", []) if x.get("epoch") is not None)
    topk_loss = ", ".join(fmt(x.get("loss")) for x in r.get("topk", []) if x.get("loss") is not None)
    topk_lddt = ", ".join(fmt(x.get("lddt")) for x in r.get("topk", []) if x.get("lddt") is not None)
    # Read NMF replacement JSON written by train_lora.py
    replaced_layers = ""
    rep_json = sweep_dir / f"{r['setting']}.nmf_replacements.json"
    if rep_json.exists():
        try:
            rep = json.loads(rep_json.read_text())
            layers = []
            for item in rep.get("records", []):
                layers.append(
                    f"{item.get('module_name')}[{item.get('module_type')}] in={item.get('in_features')} out={item.get('out_features')} rank={item.get('rank')} params={item.get('nmf_params')}"
                )
            replaced_layers = " ; ".join(layers)
        except Exception:
            replaced_layers = str(rep_json)
    row = {
        "setting": r["setting"],
        "exit_code": r.get("exit_code"),
        "val_mean_lddt": fmt(r.get("val_mean_lddt")),
        "Δlddt_vs_baseline": "" if r["setting"] == "baseline" else delta(r.get("val_mean_lddt"), base_lddt, True),
        "val_total_loss": fmt(r.get("val_total_loss")),
        "Δloss_vs_baseline": "" if r["setting"] == "baseline" else delta(r.get("val_total_loss"), base_loss, False),
        "trainable": r.get("trainable"),
        "topk_epochs": topk_epochs,
        "topk_val_loss": topk_loss,
        "topk_val_lddt": topk_lddt,
        "replaced_layers": replaced_layers,
        "notes": ", ".join(r.get("notes", [])) if r.get("notes") else "",
    }
    csv_rows.append(row)
    md.append("| " + " | ".join(str(row[h]) if row[h] is not None else "" for h in headers) + " |")

md.extend([
    "",
    "## Details",
    "",
    "| setting | exit_code | notes |",
    "| --- | --- | --- |",
])
for r in rows:
    md.append(f"| {r['setting']} | {r.get('exit_code')} | {', '.join(r.get('notes', [])) if r.get('notes') else ''} |")

md.extend([
    "",
    "## How to read",
    "",
    "- `baseline` = original checkpoint with NMF disabled; this is the pre-finetune reference.",
    "- `Δlddt_vs_baseline`: higher is better.",
    "- `Δloss_vs_baseline`: lower is better.",
    "- `replaced_layers` comes from the JSON record written at training time.",
    "- `topk_*` is the best-k epoch summary inferred from the run metrics when available.",
    "- Failure in one run does not stop the sweep.",
    "",
])

table_out.write_text("\n".join(md) + "\n")
with csv_out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=headers)
    w.writeheader()
    w.writerows(csv_rows)

print("\n".join(md))
print(f"\nWrote: {table_out}")
print(f"Wrote: {csv_out}")
PY

echo ""
echo "Done. Comparison table:"
echo "  ${TABLE_OUT}"
echo "  ${CSV_OUT}"
