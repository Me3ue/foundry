#!/usr/bin/env bash
# Final-report replacement-NMF sweep on single_structure.
#
# This sweep is designed for paper-style comparison. It runs:
#   1) baseline (no NMF)
#   2) FFN / transition family
#   3) projection / mixing family
#   4) head + projection family
#
# The script captures:
#   - final exit status
#   - run summary JSON
#   - trainable parameter count
#   - best/final validation metrics (from run_summary.json)
#   - replacement JSON for the layers that were actually swapped
#
# Usage:
#   bash models/rfd3/scripts/run_nmf_final_single_structure_sweep.sh
# Optional env:
#   DATA=... LOG_ROOT=... CKPT=... PYTHON=python SEED=42 MAX_EPOCHS=20

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/home/zzj/protein/foundry/single_structure_data}"
LOG_ROOT="${LOG_ROOT:-/home/zzj/protein/foundry/logs/train_nmf}"
CKPT="${CKPT:-/media/zzj/Data/rfd3_latest.ckpt}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-590}"
SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_DIR="${SWEEP_DIR:-${LOG_ROOT}/sweep_nmf_final_single_structure_${SWEEP_STAMP}}"
TABLE_OUT="${TABLE_OUT:-${SWEEP_DIR}/comparison_table.md}"
CSV_OUT="${CSV_OUT:-${SWEEP_DIR}/comparison_table.csv}"
mkdir -p "$SWEEP_DIR"

COMMON_OVERRIDES=(
  "paths.data.monomer_distillation_parquet_dir=${DATA}"
  "paths.data.monomer_distillation_data_dir=${DATA}"
  "paths.data.design_benchmark_data_dir=${DATA}"
  "paths.log_dir=${SWEEP_DIR}"
  "logger=csv"
  "seed=${SEED}"
  "trainer.max_epochs=${MAX_EPOCHS}"
)

JOBS=(
  "baseline|lora_finetune|name=baseline lora.enabled=false ckpt_path=${CKPT}"
  "ffn_core|nmf_ffn_core_single_structure|name=ffn_core"
  "proj_mix|nmf_proj_mix_single_structure|name=proj_mix"
  "head_plus_proj|nmf_head_plus_proj_single_structure|name=head_plus_proj"
)

run_one() {
  local tag="$1"
  local experiment="$2"
  local extras="$3"
  local run_log="${SWEEP_DIR}/${tag}.run.log"
  local final_ckpt="${SWEEP_DIR}/${tag}.last.ckpt"
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
tags = ["baseline", "ffn_core", "proj_mix", "head_plus_proj"]


def candidate_summaries(tag: str):
    direct = sweep_dir / f"{tag}.run_summary.json"
    if direct.exists():
        yield direct
    yield from sweep_dir.glob("*.run_summary.json")


def candidate_metrics_csvs(tag: str, summary: dict | None = None):
    if summary:
        m = summary.get("metrics_csv")
        if m:
            p = Path(m)
            if p.exists():
                yield p
        run_dir = summary.get("run_dir")
        if run_dir:
            p = Path(run_dir)
            yield from p.glob("**/metrics.csv")
            yield from p.glob("**/lightning_logs/**/metrics.csv")
    yield from sweep_dir.glob(f"train/{tag}/**/metrics.csv")
    yield from sweep_dir.glob(f"**/{tag}/**/metrics.csv")
    yield from sweep_dir.glob("**/lightning_logs/**/metrics.csv")


def load_summary(tag: str):
    for p in candidate_summaries(tag):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        name = str(data.get("name", ""))
        if p.name.startswith(tag) or tag in name or tag.replace("_", "-") in name or name.replace("-", "_") == tag:
            data["_summary_path"] = str(p)
            return data
    return {
        "status": "missing",
        "run_dir": "",
        "trainable_params": None,
        "total_params": None,
        "best_val_lddt": None,
        "best_val_loss": None,
        "final_val_lddt": None,
        "final_val_loss": None,
        "best_epoch": None,
        "best_epoch_loss": None,
        "best_epoch_lddt": None,
        "final_epoch": None,
        "final_epoch_loss": None,
        "final_epoch_lddt": None,
        "nmf": {},
        "metrics_csv": "",
        "_summary_path": "summary_json_missing",
    }


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
    flag = "↑" if (d > 1e-6 and higher_better) or (d < -1e-6 and not higher_better) else ("↓" if (d < -1e-6 and higher_better) or (d > 1e-6 and not higher_better) else "≈")
    return f"{d:+.4f} {flag}"


def best_metric(row, metric_prefix: str):
    best_epoch_key = f"best_epoch_{metric_prefix}"
    best_val_key = f"best_val_{metric_prefix}"
    return row.get(best_epoch_key) if row.get(best_epoch_key) is not None else row.get(best_val_key)


def final_metric(row, metric_prefix: str):
    final_epoch_key = f"final_epoch_{metric_prefix}"
    final_val_key = f"final_val_{metric_prefix}"
    return row.get(final_epoch_key) if row.get(final_epoch_key) is not None else row.get(final_val_key)


def verdict(cur, base_v, higher_better=True):
    d = delta(cur, base_v, higher_better)
    if not d:
        return "n/a"
    val = float(cur) - float(base_v)
    if abs(val) <= 1e-4:
        return "unchanged"
    return "better" if ((val > 0 and higher_better) or (val < 0 and not higher_better)) else "worse"


def overall_verdict(best_lddt_status, best_loss_status, final_lddt_status, final_loss_status):
    statuses = [best_lddt_status, best_loss_status, final_lddt_status, final_loss_status]
    if any(s == "n/a" for s in statuses):
        return "inconclusive"
    if all(s == "unchanged" for s in statuses):
        return "unchanged"
    if all(s == "better" for s in statuses):
        return "better"
    if all(s == "worse" for s in statuses):
        return "worse"
    return "mixed"

rows = []
for tag in tags:
    s = load_summary(tag)
    rows.append({"setting": tag, **s})

base = rows[0]
base_lddt = base.get("best_val_lddt")
base_loss = base.get("best_val_loss")

for r in rows:
    if r.get("best_val_lddt") is None or r.get("best_val_loss") is None:
        best_lddt = None
        best_loss = None
        best_epoch = None
        best_epoch_lddt = None
        best_epoch_loss = None
        final_epoch = None
        final_epoch_lddt = None
        final_epoch_loss = None
        for m in candidate_metrics_csvs(r["setting"], r):
            try:
                import csv as _csv
                rows_csv = list(_csv.DictReader(m.open()))
            except Exception:
                continue
            if not rows_csv:
                continue
            def last_nonempty(keys):
                val = None
                for row in rows_csv:
                    for key in keys:
                        v = row.get(key, "")
                        if v in (None, ""):
                            continue
                        try:
                            val = float(v)
                            break
                        except ValueError:
                            continue
                return val
            def best_row():
                scored = []
                for row in rows_csv:
                    epoch = row.get("epoch") or row.get("step")
                    try:
                        epoch = int(float(epoch)) if epoch not in (None, "") else None
                    except ValueError:
                        epoch = None
                    if epoch is None:
                        continue
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
                    if loss_f is not None or lddt_f is not None:
                        scored.append((epoch, loss_f, lddt_f))
                if not scored:
                    return None
                scored.sort(key=lambda x: (float('inf') if x[1] is None else x[1], float('-inf') if x[2] is None else -x[2], x[0]))
                return scored[0]
            br = best_row()
            if br is not None:
                best_epoch, best_epoch_loss, best_epoch_lddt = br
            final_row = rows_csv[-1]
            try:
                final_epoch = int(float(final_row.get("epoch", final_row.get("step", 0))))
            except Exception:
                final_epoch = None
            final_epoch_loss = last_nonempty(["val/total_loss", "train/per_epoch_total_loss"])
            final_epoch_lddt = last_nonempty(["val/mean_lddt", "train/per_epoch_mean_lddt_protein"])
            best_lddt = last_nonempty(["val/mean_lddt", "train/per_epoch_mean_lddt_protein"])
            best_loss = last_nonempty(["val/total_loss", "train/per_epoch_total_loss"])
            break
        if best_lddt is not None:
            r["best_val_lddt"] = best_lddt
        if best_loss is not None:
            r["best_val_loss"] = best_loss
        r["best_epoch"] = best_epoch
        r["best_epoch_loss"] = best_epoch_loss
        r["best_epoch_lddt"] = best_epoch_lddt
        r["final_epoch"] = final_epoch
        r["final_epoch_loss"] = final_epoch_loss
        r["final_epoch_lddt"] = final_epoch_lddt

headers = [
    "setting",
    "status",
    "summary_json",
    "run_dir",
    "best_val_lddt",
    "delta_lddt_vs_baseline",
    "best_val_loss",
    "delta_loss_vs_baseline",
    "best_epoch",
    "best_epoch_loss",
    "best_epoch_lddt",
    "final_epoch",
    "final_epoch_loss",
    "final_epoch_lddt",
    "trainable_params",
    "total_params",
    "replaced_layers",
]

metric_analysis = []

md = [
    "# Final NMF single-structure sweep report",
    "",
    f"- Baseline checkpoint: `{ckpt}`",
    f"- Sweep dir: `{sweep_dir}`",
    "",
    "## Summary",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
    "",
    "### Interpretation",
    "",
    "- `baseline` is the original checkpoint with NMF disabled.",
    "- `ffn_core` is the conservative replacement family (transition / FFN style layers).",
    "- `proj_mix` adds non-attention projection and mixing layers.",
    "- `head_plus_proj` further adds the output head.",
    "- `best_val_*` and `final_val_*` come from `run_summary.json` written by `train_lora.py`.",
    "- `replaced_layers` lists every swapped module together with dimensions and NMF parameter count.",
    "",
]

csv_rows = []
for r in rows:
    status = r.get("status", "unknown")
    summary_path = r.get("_summary_path", "summary_json_missing")
    rep_json = sweep_dir / f"{r['setting']}.nmf_replacements.json"
    replaced_layers = ""
    if rep_json.exists():
        try:
            rep = json.loads(rep_json.read_text())
            replaced_layers = " ; ".join(
                f"{x.get('module_name')}[{x.get('module_type')}] in={x.get('in_features')} out={x.get('out_features')} rank={x.get('rank')} params={x.get('nmf_params')}"
                for x in rep.get("records", [])
            )
        except Exception:
            replaced_layers = str(rep_json)

    row = {
        "setting": r["setting"],
        "status": status,
        "summary_json": summary_path,
        "run_dir": r.get("run_dir", ""),
        "best_val_lddt": fmt(r.get("best_val_lddt")),
        "delta_lddt_vs_baseline": "" if r["setting"] == "baseline" else delta(r.get("best_val_lddt"), base_lddt, True),
        "best_val_loss": fmt(r.get("best_val_loss")),
        "delta_loss_vs_baseline": "" if r["setting"] == "baseline" else delta(r.get("best_val_loss"), base_loss, False),
        "best_epoch": r.get("best_epoch"),
        "best_epoch_loss": fmt(r.get("best_epoch_loss")),
        "best_epoch_lddt": fmt(r.get("best_epoch_lddt")),
        "final_epoch": r.get("final_epoch"),
        "final_epoch_loss": fmt(r.get("final_epoch_loss")),
        "final_epoch_lddt": fmt(r.get("final_epoch_lddt")),
        "trainable_params": r.get("trainable_params"),
        "total_params": r.get("total_params"),
        "replaced_layers": replaced_layers,
    }
    csv_rows.append(row)
    md.append("| " + " | ".join(str(row.get(h, "")) if row.get(h, "") is not None else "" for h in headers) + " |")

    if r["setting"] != "baseline":
        best_lddt_status = verdict(r.get("best_val_lddt"), base_lddt, higher_better=True)
        best_loss_status = verdict(r.get("best_val_loss"), base_loss, higher_better=False)
        final_lddt_status = verdict(r.get("final_val_lddt"), base.get("final_val_lddt"), higher_better=True)
        final_loss_status = verdict(r.get("final_val_loss"), base.get("final_val_loss"), higher_better=False)
        metric_analysis.append({
            "setting": r["setting"],
            "best_lddt_status": best_lddt_status,
            "best_loss_status": best_loss_status,
            "final_lddt_status": final_lddt_status,
            "final_loss_status": final_loss_status,
            "overall": overall_verdict(best_lddt_status, best_loss_status, final_lddt_status, final_loss_status),
            "lddt_delta": delta(r.get("best_val_lddt"), base_lddt, True),
            "loss_delta": delta(r.get("best_val_loss"), base_loss, False),
            "final_lddt_delta": delta(r.get("final_val_lddt"), base.get("final_val_lddt"), True),
            "final_loss_delta": delta(r.get("final_val_loss"), base.get("final_val_loss"), False),
            "best_epoch": r.get("best_epoch"),
            "final_epoch": r.get("final_epoch"),
            "best_epoch_lddt": r.get("best_epoch_lddt"),
            "best_epoch_loss": r.get("best_epoch_loss"),
            "final_epoch_lddt": r.get("final_epoch_lddt"),
            "final_epoch_loss": r.get("final_epoch_loss"),
        })

md.extend([
    "",
    "## Metric-by-metric analysis",
    "",
])

for item in metric_analysis:
    md.extend([
        f"### {item['setting']} — {item['overall']}",
        "",
        f"- Best `lddt` vs baseline: {item['lddt_delta']} ({item['best_lddt_status']})",
        f"- Best `loss` vs baseline: {item['loss_delta']} ({item['best_loss_status']})",
        f"- Final `lddt` vs baseline: {item['final_lddt_delta']} ({item['final_lddt_status']})",
        f"- Final `loss` vs baseline: {item['final_loss_delta']} ({item['final_loss_status']})",
        f"- Best epoch: {item['best_epoch']}",
        f"- Final epoch: {item['final_epoch']}",
        f"- Best epoch metrics: loss={fmt(item['best_epoch_loss'])}, lddt={fmt(item['best_epoch_lddt'])}",
        f"- Final epoch metrics: loss={fmt(item['final_epoch_loss'])}, lddt={fmt(item['final_epoch_lddt'])}",
        "",
    ])

md.extend([
    "## Run details",
    "",
    "| setting | status | summary_json | run_dir | ckpt |",
    "| --- | --- | --- | --- | --- |",
])
for r in rows:
    summary_path = r.get("_summary_path", "summary_json_missing")
    ckpt_path = sweep_dir / "checkpoints" / f"{r['setting']}.last.ckpt"
    md.append(f"| {r['setting']} | {r.get('status', 'unknown')} | `{summary_path}` | `{r.get('run_dir', '')}` | `{ckpt_path}` |")

md.extend([
    "",
    "## Notes",
    "",
    "- The report uses `run_summary.json` as the primary source and backfills from `metrics.csv` only if needed.",
    "- The markdown contains the full table plus a per-setting analysis section.",
    "- `best_val_*` and `final_val_*` are treated strictly as the best/final validation points observed in the run.",
    "- `replaced_layers` is sourced from the JSON file written by `train_lora.py` at training time.",
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
