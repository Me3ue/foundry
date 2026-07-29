#!/usr/bin/env bash
# Robust PDB LoRA / freeze sweep with structure evaluation and report-friendly summary.
# Runs one baseline checkpoint evaluation plus the four requested PDB LoRA freeze experiments.
# Usage (from repo root):
#   bash models/rfd3/scripts/run_lora_freeze_sweep_pdb.sh
# Optional env:
#   DATA=... LOG_ROOT=... CKPT=... PYTHON=python SEED=42 MAX_EPOCHS=5 EVAL_PER_EXPERIMENT=1

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/media/zzj/Data/pdb_metadata_latest}"
LOG_ROOT="${LOG_ROOT:-/home/zzj/protein/foundry/logs/train_pdb}"
CKPT="${CKPT:-/media/zzj/Data/pdb_metadata_latest/rfd3_latest.ckpt}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
EVAL_PER_EXPERIMENT="${EVAL_PER_EXPERIMENT:-1}"
SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_DIR="${SWEEP_DIR:-${LOG_ROOT}/sweep_lora_freeze_pdb_${SWEEP_STAMP}}"
TABLE_OUT="${TABLE_OUT:-${SWEEP_DIR}/comparison_table.md}"
CSV_OUT="${CSV_OUT:-${SWEEP_DIR}/comparison_table.csv}"
EVAL_DIR="${EVAL_DIR:-${SWEEP_DIR}/evaluation}"
CKPT_DIR="${CKPT_DIR:-${SWEEP_DIR}/checkpoints}"
GEN_DIR="${GEN_DIR:-${SWEEP_DIR}/generated_structures}"
TOPK="${TOPK:-3}"
mkdir -p "$SWEEP_DIR" "$EVAL_DIR" "$CKPT_DIR" "$GEN_DIR"

COMMON_OVERRIDES=(
  "paths.data.pdb_data_dir=${DATA}"
  "paths.data.pdb_parquet_dir=${DATA}"
  "paths.log_dir=${SWEEP_DIR}"
  "logger=csv"
  "seed=${SEED}"
  "trainer.max_epochs=${MAX_EPOCHS}"
)

JOBS=(
  "baseline|lora_finetune_pdb|lora.enabled=false ckpt_path=${CKPT}"
  "lora_freeze_input|lora_freeze_input_pdb|"
  "lora_freeze_diffusion|lora_freeze_diffusion_pdb|"
  "lora_freeze_encoder|lora_freeze_encoder_pdb|"
  "lora_head_only|lora_head_only_pdb|"
)

prepare_eval_set() {
  local eval_root="${SWEEP_DIR}/eval_set"
  mkdir -p "${eval_root}/single_chain" "${eval_root}/reference"
  local registry="${REPO_ROOT}/rfdiffusion3_inputs/registry.json"
  if [[ ! -f "${registry}" ]]; then
    echo "Preparing RFdiffusion3 input registry for evaluation set..."
    "${PYTHON}" "${REPO_ROOT}/prepare_rfdiffusion3_inputs.py"
  fi
  if [[ ! -f "${registry}" ]]; then
    echo "Could not find or create ${registry}" >&2
    return 1
  fi
  "${PYTHON}" - <<'PY' "$registry" "$eval_root"
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1])
out_root = Path(sys.argv[2])
entries = json.loads(registry.read_text())
for e in entries:
    if e.get("status") != "ok":
        continue
    pdb_id = e["pdb_id"]
    chain = e["selected_chain"]
    single = Path("rfdiffusion3_inputs") / "single_chain" / f"{pdb_id}_{chain}.pdb"
    ref_dst = out_root / "reference" / f"{pdb_id}_{chain}.pdb"
    if single.exists() and not ref_dst.exists():
        ref_dst.write_text(single.read_text())
print(out_root)
PY
}

run_one() {
  local tag="$1"
  local experiment="$2"
  local extras="$3"
  local run_log="${SWEEP_DIR}/${tag}.run.log"
  local marker="${SWEEP_DIR}/${tag}.hydra_run_dir.txt"
  local eval_summary="${EVAL_DIR}/${tag}.json"
  local final_ckpt="${CKPT_DIR}/${tag}.last.ckpt"
  local eval_root="${SWEEP_DIR}/eval_set"

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

  [[ -n "${hydra_dir}" ]] && echo "${hydra_dir}" > "${marker}"
  echo "${rc}" > "${SWEEP_DIR}/${tag}.exit_code"

  "${PYTHON}" - <<'PY' "${SWEEP_DIR}" "${tag}" "${eval_summary}" "${rc}"
import csv, json, re, sys
from pathlib import Path
sweep_dir = Path(sys.argv[1]); tag = sys.argv[2]; out = Path(sys.argv[3]); exit_code = int(sys.argv[4]) if sys.argv[4].isdigit() else sys.argv[4]
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
    last=None
    for row in rows:
        v=row.get(key,"")
        if v in (None,""):
            continue
        try: last=float(v)
        except ValueError: last=v
    return last
metrics={}; topk=[]
if metrics_csv and metrics_csv.exists():
    with metrics_csv.open() as f: rows=list(csv.DictReader(f))
    for key in ["train/per_epoch_total_loss","train/per_epoch_mean_lddt_protein","train/per_epoch_seq_recovery","train/per_epoch_mse_loss_mean","train/batch_mean/total_loss","train/batch_mean/mean_lddt_protein","val/mean_lddt","val/total_loss"]:
        metrics[key]=last_nonempty(rows,key)
    scored=[]
    for row in rows:
        epoch=row.get("epoch") or row.get("trainer/global_step") or row.get("step")
        try: epoch=int(float(epoch)) if epoch not in (None,"") else None
        except ValueError: epoch=None
        loss=row.get("val/total_loss") or row.get("train/per_epoch_total_loss")
        lddt=row.get("val/mean_lddt") or row.get("train/per_epoch_mean_lddt_protein")
        try: loss_f=float(loss) if loss not in (None,"") else None
        except ValueError: loss_f=None
        try: lddt_f=float(lddt) if lddt not in (None,"") else None
        except ValueError: lddt_f=None
        if epoch is not None and (loss_f is not None or lddt_f is not None):
            scored.append({"epoch":epoch,"loss":loss_f,"lddt":lddt_f})
    topk=sorted(scored, key=lambda r:(float('inf') if r["loss"] is None else r["loss"], float('-inf') if r["lddt"] is None else -r["lddt"], r["epoch"]))[:3]
summary={"tag":tag,"exit_code":exit_code,"metrics_csv":str(metrics_csv) if metrics_csv else "","metrics":metrics,"topk":topk,"trainable_after_lora":None,"trainable_after_freeze":None,"notes":[]}
m=re.search(r"LoRA trainable parameter summary: trainable params=([\d,]+)", text)
if m: summary["trainable_after_lora"]=int(m.group(1).replace(",",""))
m=re.search(r"Post-freeze trainable parameter summary: trainable params=([\d,]+)", text)
if m: summary["trainable_after_freeze"]=int(m.group(1).replace(",",""))
for pat in [r"Traceback \(most recent call last\):",r"CUDA out of memory",r"RuntimeError:",r"AssertionError:",r"Error executing job with overrides"]:
    if re.search(pat, text): summary["notes"].append(pat)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
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

  if [[ "${EVAL_PER_EXPERIMENT}" == "1" ]]; then
    local ref_root="${SWEEP_DIR}/eval_set/reference"
    local gen_root="${hydra_dir:-}"
    local eval_out_dir="${GEN_DIR}/${tag}"
    mkdir -p "${eval_out_dir}"
    if [[ -n "${hydra_dir}" ]]; then
      "${PYTHON}" "${REPO_ROOT}/compute_rfdiffusion3_structure_metrics.py" \
        --reference-root "${ref_root}" \
        --generated-root "${hydra_dir}" \
        --out-root "${eval_out_dir}" \
        --tag "${tag}" \
        --limit 20 \
        --chain-from-registry "${SWEEP_DIR}/eval_set"
    fi
  fi

  echo "[$(date '+%F %T')] END ${tag} exit=${rc} hydra_dir=${hydra_dir:-unknown}"
  return 0
}

prepare_eval_set
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag experiment extras <<<"${job}"
  run_one "${tag}" "${experiment}" "${extras}"
done

"${PYTHON}" - <<'PY' "$SWEEP_DIR" "$TABLE_OUT" "$CSV_OUT" "$CKPT" "$EVAL_DIR"
import csv, json, sys
from pathlib import Path
sweep_dir = Path(sys.argv[1]); table_out = Path(sys.argv[2]); csv_out = Path(sys.argv[3]); ckpt = sys.argv[4]; eval_dir = Path(sys.argv[5])
tags=["baseline","lora_freeze_input","lora_freeze_diffusion","lora_freeze_encoder","lora_head_only"]

def load_eval_summary(tag):
    p=eval_dir / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else {"exit_code":None,"metrics":{},"topk":[],"trainable_after_lora":None,"trainable_after_freeze":None,"notes":[]}

def fmt(v,d=4):
    if v is None or v=="": return ""
    return f"{v:.{d}f}" if isinstance(v,(int,float)) else str(v)

def delta(cur,base_v,higher_better=True):
    if cur is None or base_v is None: return ""
    try: c,b=float(cur),float(base_v)
    except: return ""
    d=c-b
    flag=("↑" if d>1e-6 else ("↓" if d<-1e-6 else "≈")) if higher_better else ("↓" if d<-1e-6 else ("↑" if d>1e-6 else "≈"))
    return f"{d:+.4f} {flag}"
rows=[]
for tag in tags:
    s=load_eval_summary(tag); m=s.get("metrics",{}) or {}; topk=s.get("topk",[]) or []
    rows.append({"setting":tag,"status":"ok" if str(s.get("exit_code")) in {"0","None",""} else f"fail({s.get('exit_code')})","exit_code":s.get("exit_code"),"metrics_csv":s.get("metrics_csv") or "","trainable_after_lora":s.get("trainable_after_lora"),"trainable_after_freeze":s.get("trainable_after_freeze"),"val_mean_lddt":m.get("val/mean_lddt"),"val_total_loss":m.get("val/total_loss"),"per_epoch_total_loss":m.get("train/per_epoch_total_loss"),"per_epoch_mean_lddt_protein":m.get("train/per_epoch_mean_lddt_protein"),"per_epoch_seq_recovery":m.get("train/per_epoch_seq_recovery"),"topk":topk,"notes":", ".join(s.get("notes",[])) if s.get("notes") else ""})
base=next((r for r in rows if r["setting"]=="baseline"),None)
base_lddt=base["val_mean_lddt"] if base else None
base_loss=base["val_total_loss"] if base else None
if base_lddt is None and base: base_lddt=base["per_epoch_mean_lddt_protein"]
if base_loss is None and base: base_loss=base["per_epoch_total_loss"]
headers=["setting","status","val_mean_lddt","Δlddt_vs_baseline","val_total_loss","Δloss_vs_baseline","seq_recovery","rmsd_ca","tm_score","topk_epochs","topk_val_loss","topk_val_lddt","trainable_after_freeze","trainable_after_lora","notes"]
md=["# PDB LoRA freeze sweep report","",f"- Baseline checkpoint: `{ckpt}`",f"- Sweep dir: `{sweep_dir}`","","## Summary","","| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
csv_rows=[]
for r in rows:
    cur_lddt=r["val_mean_lddt"] if r["val_mean_lddt"] is not None else r["per_epoch_mean_lddt_protein"]
    cur_loss=r["val_total_loss"] if r["val_total_loss"] is not None else r["per_epoch_total_loss"]
    topk_epochs=", ".join(str(x.get("epoch")) for x in r.get("topk",[]) if x.get("epoch") is not None)
    topk_loss=", ".join(fmt(x.get("loss")) for x in r.get("topk",[]) if x.get("loss") is not None)
    topk_lddt=", ".join(fmt(x.get("lddt")) for x in r.get("topk",[]) if x.get("lddt") is not None)
    rmsd_ca=tm_score=""
    rmsd_path=sweep_dir / "generated_structures" / r["setting"] / "structure_metrics.csv"
    if rmsd_path.exists():
        with rmsd_path.open() as f:
            rr=list(csv.DictReader(f))
        if rr:
            vals=[row for row in rr if row.get("status")=="ok"]
            if vals:
                rmsd_ca=vals[-1].get("rmsd_ca","")
                tm_score=vals[-1].get("tm_score","")
    row={"setting":r["setting"],"status":r["status"],"val_mean_lddt":fmt(cur_lddt),"Δlddt_vs_baseline":"" if r["setting"]=="baseline" else delta(cur_lddt,base_lddt,True),"val_total_loss":fmt(cur_loss),"Δloss_vs_baseline":"" if r["setting"]=="baseline" else delta(cur_loss,base_loss,False),"seq_recovery":fmt(r["per_epoch_seq_recovery"]),"rmsd_ca":fmt(rmsd_ca),"tm_score":fmt(tm_score),"topk_epochs":topk_epochs,"topk_val_loss":topk_loss,"topk_val_lddt":topk_lddt,"trainable_after_freeze":r["trainable_after_freeze"],"trainable_after_lora":r["trainable_after_lora"],"notes":r["notes"]}
    csv_rows.append(row)
    md.append("| "+" | ".join(str(row[h]) if row[h] is not None else "" for h in headers)+" |")
md.extend(["","## Details","","| setting | exit_code | metrics_csv | per_epoch_total_loss | per_epoch_mean_lddt_protein | per_epoch_seq_recovery |","| --- | --- | --- | --- | --- | --- |"])
for r in rows:
    md.append("| "+" | ".join([r["setting"],str(r["exit_code"]),r["metrics_csv"],fmt(r["per_epoch_total_loss"]),fmt(r["per_epoch_mean_lddt_protein"]),fmt(r["per_epoch_seq_recovery"])] )+" |")
md.extend(["","## How to read","","- `baseline` = original checkpoint with LoRA disabled; this is the pre-finetune reference.","- `Δlddt_vs_baseline`: higher is better.","- `Δloss_vs_baseline`: lower is better.","- `rmsd_ca` and `tm_score` are populated from structure evaluation when generated structures are available.","- `status` summarizes whether the train job exited cleanly; failure in one job does not stop the sweep.",""])
table_out.write_text("\n".join(md)+"\n")
with csv_out.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=headers); w.writeheader(); w.writerows(csv_rows)
print("\n".join(md))
print(f"\nWrote: {table_out}")
print(f"Wrote: {csv_out}")
PY

echo ""
echo "Done. Comparison table:"
echo "  ${TABLE_OUT}"
echo "  ${CSV_OUT}"
