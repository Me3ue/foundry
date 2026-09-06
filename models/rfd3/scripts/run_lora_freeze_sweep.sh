#!/usr/bin/env bash
# Robust LoRA / freeze sweep with baseline comparison and report-friendly summary.
# Runs one baseline checkpoint evaluation plus the four requested LoRA freeze experiments.
# Usage (from repo root):
#   bash models/rfd3/scripts/run_lora_freeze_sweep.sh
# Optional env:
#   DATA=... LOG_ROOT=... CKPT=... PYTHON=python SEED=42 MAX_EPOCHS=20

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-~/protein/foundry/single_structure_data}"
LOG_ROOT="${LOG_ROOT:-~/protein/foundry/logs/train}"
CKPT="${CKPT:-~/protein/foundry/models/rfd3/rfd3_latest.ckpt}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_DIR="${SWEEP_DIR:-${LOG_ROOT}/sweep_lora_freeze_${SWEEP_STAMP}}"
TABLE_OUT="${TABLE_OUT:-${SWEEP_DIR}/comparison_table.md}"
CSV_OUT="${CSV_OUT:-${SWEEP_DIR}/comparison_table.csv}"
EVAL_DIR="${EVAL_DIR:-${SWEEP_DIR}/evaluation}"
CKPT_DIR="${CKPT_DIR:-${SWEEP_DIR}/checkpoints}"
TOPK="${TOPK:-3}"

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
  "lora_freeze_input|lora_freeze_input|"
  "lora_freeze_diffusion|lora_freeze_diffusion|"
  "lora_freeze_encoder|lora_freeze_encoder|"
  "lora_head_only|lora_head_only|"
)

run_one() {
  local tag="$1"
  local experiment="$2"
  local extras="$3"
  local run_log="${SWEEP_DIR}/${tag}.run.log"
  local marker="${SWEEP_DIR}/${tag}.hydra_run_dir.txt"
  local eval_summary="${EVAL_DIR}/${tag}.json"
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
    echo "${hydra_dir}" > "${marker}"
  fi
  echo "${rc}" > "${SWEEP_DIR}/${tag}.exit_code"

  # Best-effort per-run evaluation summary: parse the latest metrics table and log one JSON blob.
  "${PYTHON}" - <<'PY' "${SWEEP_DIR}" "${tag}" "${eval_summary}" "${rc}"
import csv
import json
import re
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
tag = sys.argv[2]
out = Path(sys.argv[3])
exit_code = int(sys.argv[4]) if sys.argv[4].isdigit() else sys.argv[4]

marker = sweep_dir / f"{tag}.hydra_run_dir.txt"
metrics_csv = None
if marker.exists():
    root = Path(marker.read_text().strip())
    cands = list(root.glob("**/lightning_logs/*/metrics.csv"))
    if cands:
        metrics_csv = max(cands, key=lambda p: p.stat().st_mtime)

log_path = sweep_dir / f"{tag}.run.log"
text = log_path.read_text(errors="replace") if log_path.exists() else ""


def last_nonempty(rows, key):
    last = None
    for row in rows:
        v = row.get(key, "")
        if v is None or v == "":
            continue
        try:
            last = float(v)
        except ValueError:
            last = v
    return last

metrics = {}
topk = []
if metrics_csv and metrics_csv.exists():
    with metrics_csv.open() as f:
        rows = list(csv.DictReader(f))
    for key in [
        "train/per_epoch_total_loss",
        "train/per_epoch_mean_lddt_protein",
        "train/per_epoch_seq_recovery",
        "train/per_epoch_mse_loss_mean",
        "train/batch_mean/total_loss",
        "train/batch_mean/mean_lddt_protein",
        "val/mean_lddt",
        "val/total_loss",
    ]:
        metrics[key] = last_nonempty(rows, key)
    scored = []
    for row in rows:
        epoch = row.get("epoch") or row.get("trainer/global_step") or row.get("step")
        try:
            epoch = int(float(epoch)) if epoch is not None and epoch != "" else None
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
    topk = sorted(
        scored,
        key=lambda r: (
            float("inf") if r["loss"] is None else r["loss"],
            float("-inf") if r["lddt"] is None else -r["lddt"],
            r["epoch"],
        ),
    )[:3]

summary = {
    "tag": tag,
    "exit_code": exit_code,
    "metrics_csv": str(metrics_csv) if metrics_csv else "",
    "metrics": metrics,
    "topk": topk,
    "trainable_after_lora": None,
    "trainable_after_freeze": None,
    "notes": [],
}

m = re.search(r"LoRA trainable parameter summary: trainable params=([\d,]+)", text)
if m:
    summary["trainable_after_lora"] = int(m.group(1).replace(",", ""))
m = re.search(r"Post-freeze trainable parameter summary: trainable params=([\d,]+)", text)
if m:
    summary["trainable_after_freeze"] = int(m.group(1).replace(",", ""))

for pat in [
    r"Traceback \(most recent call last\):",
    r"CUDA out of memory",
    r"RuntimeError:",
    r"AssertionError:",
    r"Error executing job with overrides",
]:
    if re.search(pat, text):
        summary["notes"].append(pat)

out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

  if [[ -n "${hydra_dir}" ]]; then
    local best_ckpt
    best_ckpt="$(find "${hydra_dir}" -type f \( -name '*.ckpt' -o -name 'last.ckpt' \) 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${best_ckpt}" ]]; then
      cp -f "${best_ckpt}" "${final_ckpt}" 2>/dev/null || true
      printf '%s\n' "${best_ckpt}" > "${CKPT_DIR}/${tag}.source_ckpt.txt"
    fi
  fi

  echo "[$(date '+%F %T')] END ${tag} exit=${rc} hydra_dir=${hydra_dir:-unknown}"
  return 0
}

for job in "${JOBS[@]}"; do
  IFS='|' read -r tag experiment extras <<<"${job}"
  run_one "${tag}" "${experiment}" "${extras}"
done

# Aggregate metrics from each tag's newest metrics.csv under SWEEP_DIR.
"${PYTHON}" - <<'PY' "$SWEEP_DIR" "$TABLE_OUT" "$CSV_OUT" "$CKPT" "$EVAL_DIR"
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
table_out = Path(sys.argv[2])
csv_out = Path(sys.argv[3])
ckpt = sys.argv[4]
eval_dir = Path(sys.argv[5])

tags = [
    "baseline",
    "lora_freeze_input",
    "lora_freeze_diffusion",
    "lora_freeze_encoder",
    "lora_head_only",
]

metric_keys = [
    "train/per_epoch_total_loss",
    "train/per_epoch_mean_lddt_protein",
    "train/per_epoch_seq_recovery",
    "train/per_epoch_mse_loss_mean",
    "train/batch_mean/total_loss",
    "train/batch_mean/mean_lddt_protein",
    "val/mean_lddt",
    "val/total_loss",
]


def last_nonempty(rows, key):
    last = None
    for row in rows:
        v = row.get(key, "")
        if v is None or v == "":
            continue
        try:
            last = float(v)
        except ValueError:
            last = v
    return last


def find_metrics_csv(tag: str):
    marker = sweep_dir / f"{tag}.hydra_run_dir.txt"
    if marker.exists():
        root = Path(marker.read_text().strip())
        cands = list(root.glob("**/lightning_logs/*/metrics.csv"))
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    return None


def load_eval_summary(tag: str):
    p = eval_dir / f"{tag}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"exit_code": None, "metrics_csv": "", "metrics": {}, "topk": [], "trainable_after_lora": None, "trainable_after_freeze": None, "notes": []}


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


rows_out = []
for tag in tags:
    summary = load_eval_summary(tag)
    metrics_path = find_metrics_csv(tag)
    metrics = {k: None for k in metric_keys}
    if metrics_path and metrics_path.exists():
        with metrics_path.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
        for k in metric_keys:
            metrics[k] = last_nonempty(data, k)

    eval_metrics = summary.get("metrics", {}) or {}
    combined = {k: eval_metrics.get(k, metrics.get(k)) for k in metric_keys}

    rows_out.append(
        {
            "setting": tag,
            "status": "ok" if str(summary.get("exit_code")) in {"0", "None", ""} else f"fail({summary.get('exit_code')})",
            "exit_code": summary.get("exit_code"),
            "metrics_csv": summary.get("metrics_csv") or str(metrics_path) if metrics_path else "",
            "trainable_after_lora": summary.get("trainable_after_lora"),
            "trainable_after_freeze": summary.get("trainable_after_freeze"),
            "val_mean_lddt": combined.get("val/mean_lddt"),
            "val_total_loss": combined.get("val/total_loss"),
            "per_epoch_total_loss": combined.get("train/per_epoch_total_loss"),
            "per_epoch_mean_lddt_protein": combined.get("train/per_epoch_mean_lddt_protein"),
            "per_epoch_seq_recovery": combined.get("train/per_epoch_seq_recovery"),
            "per_epoch_mse_loss_mean": combined.get("train/per_epoch_mse_loss_mean"),
            "batch_mean_total_loss": combined.get("train/batch_mean/total_loss"),
            "batch_mean_mean_lddt_protein": combined.get("train/batch_mean/mean_lddt_protein"),
            "topk": summary.get("topk", []),
            "notes": ", ".join(summary.get("notes", [])) if summary.get("notes") else "",
        }
    )

base = next((r for r in rows_out if r["setting"] == "baseline"), None)
base_lddt = None
base_loss = None
if base:
    base_lddt = base["val_mean_lddt"] if base["val_mean_lddt"] is not None else base["per_epoch_mean_lddt_protein"]
    base_loss = base["val_total_loss"] if base["val_total_loss"] is not None else base["per_epoch_total_loss"]

headers = [
    "setting",
    "status",
    "val_mean_lddt",
    "Δlddt_vs_baseline",
    "val_total_loss",
    "Δloss_vs_baseline",
    "seq_recovery",
    "topk_epochs",
    "topk_val_loss",
    "topk_val_lddt",
    "trainable_after_freeze",
    "trainable_after_lora",
    "notes",
]

md_lines = [
    "# LoRA freeze sweep report",
    "",
    f"- Baseline checkpoint: `{ckpt}`",
    f"- Sweep dir: `{sweep_dir}`",
    "",
    "## Summary",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]

csv_rows = []
for r in rows_out:
    cur_lddt = r["val_mean_lddt"] if r["val_mean_lddt"] is not None else r["per_epoch_mean_lddt_protein"]
    cur_loss = r["val_total_loss"] if r["val_total_loss"] is not None else r["per_epoch_total_loss"]
    topk_epochs = ", ".join(str(item.get("epoch")) for item in r.get("topk", []) if item.get("epoch") is not None)
    topk_loss = ", ".join(fmt(item.get("loss")) for item in r.get("topk", []) if item.get("loss") is not None)
    topk_lddt = ", ".join(fmt(item.get("lddt")) for item in r.get("topk", []) if item.get("lddt") is not None)
    row = {
        "setting": r["setting"],
        "status": r["status"],
        "val_mean_lddt": fmt(cur_lddt),
        "Δlddt_vs_baseline": "" if r["setting"] == "baseline" else delta(cur_lddt, base_lddt, True),
        "val_total_loss": fmt(cur_loss),
        "Δloss_vs_baseline": "" if r["setting"] == "baseline" else delta(cur_loss, base_loss, False),
        "seq_recovery": fmt(r["per_epoch_seq_recovery"]),
        "topk_epochs": topk_epochs,
        "topk_val_loss": topk_loss,
        "topk_val_lddt": topk_lddt,
        "trainable_after_freeze": r["trainable_after_freeze"],
        "trainable_after_lora": r["trainable_after_lora"],
        "notes": r["notes"],
    }
    csv_rows.append(row)
    md_lines.append("| " + " | ".join(str(row[h]) if row[h] is not None else "" for h in headers) + " |")

md_lines.extend(
    [
        "",
        "## Details",
        "",
        "| setting | exit_code | metrics_csv | per_epoch_total_loss | per_epoch_mean_lddt_protein | per_epoch_seq_recovery | batch_mean_total_loss | batch_mean_mean_lddt_protein | topk_epochs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
)
for r in rows_out:
    topk_epochs = ", ".join(str(item.get("epoch")) for item in r.get("topk", []) if item.get("epoch") is not None)
    md_lines.append(
        "| "
        + " | ".join(
            [
                r["setting"],
                str(r["exit_code"]),
                r["metrics_csv"],
                fmt(r["per_epoch_total_loss"]),
                fmt(r["per_epoch_mean_lddt_protein"]),
                fmt(r["per_epoch_seq_recovery"]),
                fmt(r["batch_mean_total_loss"]),
                fmt(r["batch_mean_mean_lddt_protein"]),
                topk_epochs,
            ]
        )
        + " |"
    )

md_lines.extend(
    [
        "",
        "## How to read",
        "",
        "- `baseline` = original checkpoint with LoRA disabled; this is the pre-finetune reference.",
        "- `Δlddt_vs_baseline`: higher is better.",
        "- `Δloss_vs_baseline`: lower is better.",
        "- `status` summarizes whether the train job exited cleanly; failure in one job does not stop the sweep.",
        "- `topk_epochs` lists the top-k epochs ranked by lowest validation loss, then highest validation LDDT.",
        "- `notes` records obvious failure patterns from the run log (e.g. OOM / Traceback).",
        "",
    ]
)

table_out.write_text("\n".join(md_lines) + "\n")
with csv_out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=headers)
    w.writeheader()
    w.writerows(csv_rows)

print("\n".join(md_lines))
print(f"\nWrote: {table_out}")
print(f"Wrote: {csv_out}")
PY

echo ""
echo "Done. Comparison table:"
echo "  ${TABLE_OUT}"
echo "  ${CSV_OUT}"
