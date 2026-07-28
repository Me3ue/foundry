#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RFD3_BIN="${RFD3_BIN:-rfd3}"
CKPT_PATH="${CKPT_PATH:-$(pwd)/models/rfd3/rfd3_latest.ckpt}"
INPUT_DIR="rfdiffusion3_inputs/unified_templates"
OUT_ROOT="${OUT_ROOT:-logs/inference_outs}"
LOG_ROOT="${LOG_ROOT:-logs/rfdiffusion3_batch}"
TMP_JSON_DIR="$LOG_ROOT/generated_json"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
DUMP_TRAJECTORIES="${DUMP_TRAJECTORIES:-False}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$TMP_JSON_DIR"

SUCCESS_LIST="$LOG_ROOT/success.txt"
FAIL_LIST="$LOG_ROOT/fail.txt"
: > "$SUCCESS_LIST"
: > "$FAIL_LIST"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_target_json() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, re, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    obj = json.loads(p.read_text())
except Exception:
    print("0")
    raise SystemExit(0)
if not isinstance(obj, dict) or len(obj) != 1:
    print("0")
    raise SystemExit(0)
key = next(iter(obj))
print("1" if re.fullmatch(r"[0-9A-Za-z]{4}", key) else "0")
PY
}

json_input_field() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
obj = json.loads(p.read_text())
key = next(iter(obj))
spec = obj[key]
print(spec.get("input", ""))
PY
}

precheck_json() {
  local json_file="$1"
  local label="$2"
  local input_pdb
  input_pdb="$(json_input_field "$json_file")"
  if [ -z "$input_pdb" ]; then
    log "precheck fail $label missing input field"
    return 1
  fi
  if [ ! -f "$input_pdb" ]; then
    log "precheck fail $label missing pdb: $input_pdb"
    return 1
  fi
  if [ ! -f "$CKPT_PATH" ]; then
    log "precheck fail missing checkpoint: $CKPT_PATH"
    return 1
  fi
  return 0
}

make_variant_json() {
  local base_json="$1"
  local variant="$2"
  local out_json="$3"
  "$PYTHON_BIN" - "$base_json" "$variant" "$out_json" <<'PY'
import json, sys
from pathlib import Path

def contig_from_ranges(chain, ranges):
    return ",".join([f"{chain}{s}" if s == e else f"{chain}{s}-{e}" for s, e in ranges])

base = Path(sys.argv[1])
variant = sys.argv[2]
out = Path(sys.argv[3])
obj = json.loads(base.read_text())
key = next(iter(obj))
spec = obj[key]
extra = spec.get("extra", {}) or {}
chain = extra.get("selected_chain")
fixed_ranges = extra.get("fixed_ranges", [])
diffuse_ranges = extra.get("diffuse_ranges", [])
symmetry_enabled = bool(extra.get("symmetry_enabled"))

if not chain or not fixed_ranges or not diffuse_ranges:
    raise SystemExit("template missing fixed/diffuse metadata")

fixed_atoms = {f"{chain}{resi}": "BKBN" for start, end in fixed_ranges for resi in range(int(start), int(end) + 1)}
diffuse_contig = contig_from_ranges(chain, diffuse_ranges)

if variant == "baseline":
    spec["select_fixed_atoms"] = fixed_atoms
    spec["select_unfixed_sequence"] = diffuse_contig
    spec["partial_t"] = 1.0
    spec.pop("symmetry", None)
elif variant == "fixed":
    spec["select_fixed_atoms"] = fixed_atoms
    spec["select_unfixed_sequence"] = diffuse_contig
    spec["partial_t"] = 1.0
    spec.pop("symmetry", None)
elif variant == "partial":
    spec["select_fixed_atoms"] = fixed_atoms
    spec["select_unfixed_sequence"] = diffuse_contig
    spec["partial_t"] = 1.0
    spec.pop("symmetry", None)
elif variant == "symmetry":
    spec["select_fixed_atoms"] = fixed_atoms
    spec["select_unfixed_sequence"] = diffuse_contig
    spec["partial_t"] = 1.0
    spec["__skip__"] = "symmetry_requires_sym_transform"
    spec.pop("symmetry", None)
else:
    raise SystemExit(f"unknown variant: {variant}")

out.write_text(json.dumps({key: spec}, indent=2))
PY
}

run_one() {
  local pdb_id="$1"
  local variant="$2"
  local input_json="$3"
  local out_dir="$OUT_ROOT/$pdb_id/$variant"
  local run_log="$LOG_ROOT/${pdb_id}_${variant}.log"

  if ! precheck_json "$input_json" "$pdb_id/$variant"; then
    echo "$pdb_id/$variant precheck_failed" >> "$FAIL_LIST"
    return 0
  fi

  local skip_marker
  skip_marker="$($PYTHON_BIN - "$input_json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
obj = json.loads(p.read_text())
key = next(iter(obj))
print(obj[key].get("__skip__", ""))
PY
)"
  if [ -n "$skip_marker" ]; then
    log "skip $pdb_id/$variant $skip_marker"
    echo "$pdb_id/$variant skipped_$skip_marker" >> "$FAIL_LIST"
    return 0
  fi

  mkdir -p "$out_dir"

  local cmd=()
  cmd+=("$RFD3_BIN" design)
  cmd+=("out_dir=$out_dir")
  cmd+=("inputs=$input_json")
  cmd+=("ckpt_path=$CKPT_PATH")
  if [ "$variant" = "symmetry" ]; then
    cmd+=("inference_sampler.kind=symmetry")
  fi
  cmd+=("skip_existing=False")
  cmd+=("dump_trajectories=$DUMP_TRAJECTORIES")
  cmd+=("prevalidate_inputs=True")

  log "run $pdb_id/$variant -> $out_dir"
  log "cmd: ${cmd[*]}"
  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') $pdb_id/$variant ==="
    echo "json=$input_json"
    echo "ckpt=$CKPT_PATH"
    echo "cmd=${cmd[*]}"
    echo "config: symmetry=$([ "$variant" = "symmetry" ] && echo yes || echo no) dump_trajectories=$DUMP_TRAJECTORIES"
    echo
    "${cmd[@]}"
  } >"$run_log" 2>&1
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    log "ok $pdb_id/$variant"
    echo "$pdb_id/$variant ok" >> "$SUCCESS_LIST"
  else
    log "fail $pdb_id/$variant (see $run_log)"
    echo "$pdb_id/$variant exit_$exit_code" >> "$FAIL_LIST"
  fi
  return 0
}

main() {
  if [ ! -d "$INPUT_DIR" ]; then
    log "input templates not found at $INPUT_DIR"
    log "run: python make_rfdiffusion3_unified_template.py"
    exit 1
  fi

  local found=0
  local precheck_fail=0
  local target_jsons=()

  shopt -s nullglob
  for json_file in "$INPUT_DIR"/*.json; do
    if [ "$(basename "$json_file")" = "summary.json" ]; then
      continue
    fi
    if ! is_target_json "$json_file"; then
      continue
    fi
    target_jsons+=("$json_file")
    found=1
  done
  shopt -u nullglob

  if [ "$found" -eq 0 ]; then
    log "no unified template jsons found in $INPUT_DIR"
    exit 1
  fi

  rm -f "$TMP_JSON_DIR"/*.json 2>/dev/null || true

  for json_file in "${target_jsons[@]}"; do
    local pdb_id
    pdb_id="$(basename "$json_file" .json)"
    for variant in baseline fixed partial symmetry; do
      local variant_json="$TMP_JSON_DIR/${pdb_id}_${variant}.json"
      make_variant_json "$json_file" "$variant" "$variant_json"
      if ! precheck_json "$variant_json" "$pdb_id/$variant"; then
        precheck_fail=1
      fi
    done
  done

  if [ "$PRECHECK_ONLY" = "1" ]; then
    if [ "$precheck_fail" -eq 0 ]; then
      log "precheck passed for all generated variants"
      exit 0
    fi
    log "precheck failed; fix inputs before running"
    exit 1
  fi

  for json_file in "$TMP_JSON_DIR"/*.json; do
    [ -e "$json_file" ] || continue
    local stem
    stem="$(basename "$json_file" .json)"
    local pdb_id="${stem%_*}"
    local variant="${stem##*_}"
    run_one "$pdb_id" "$variant" "$json_file"
  done

  log "batch complete"
  log "success list: $SUCCESS_LIST"
  log "fail list: $FAIL_LIST"
}

main "$@"
