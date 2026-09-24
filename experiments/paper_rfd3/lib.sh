#!/usr/bin/env bash
# =============================================================================
# 公共 bash 工具：日志、计时、命令构造、多卡调度封装、环境自检
# 用法： source env.sh && source lib.sh
# =============================================================================

C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
C_RED=$'\033[31m';  C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_CYN=$'\033[36m'

log()  { printf '%s[%s]%s %s\n'  "$C_CYN" "$(date +%H:%M:%S)" "$C_RESET" "$*" >&2; }
ok()   { printf '%s  ok%s %s\n'  "$C_GRN" "$C_RESET" "$*" >&2; }
warn() { printf '%s  !%s  %s\n'  "$C_YEL" "$C_RESET" "$*" >&2; }
die()  { printf '%s  x%s  %s\n'  "$C_RED" "$C_RESET" "$*" >&2; exit 1; }
hdr()  { printf '\n%s== %s ==%s\n' "$C_BOLD" "$*" "$C_RESET" >&2; }

# --- 前置条件 ---------------------------------------------------------------
require_cmd()  { command -v "$1" >/dev/null 2>&1 || die "找不到命令 '$1'。检查 RC_ENV_BIN=$RC_ENV_BIN"; }
require_file() { [[ -f "$1" ]] || die "缺少文件: $1"; }

# --- GPU 信息 ---------------------------------------------------------------
gpu_table() {
  local exe; exe="$(command -v nvidia-smi || true)"
  [[ -n "$exe" ]] || return 0
  local out
  out="$("$exe" --query-gpu=index,name,memory.free,memory.total,utilization.gpu \
         --format=csv,noheader 2>/dev/null || true)"
  # 驱动异常时 nvidia-smi 会把错误写到 stdout，这里识别并跳过
  if [[ -z "$out" || "$out" == *"has failed"* || "$out" == *"couldn't communicate"* ]]; then
    return 0
  fi
  printf '%s\n' "$out" | awk -F', *' '
    { printf "    GPU %-2s %-22s 空闲 %6s / %6s MiB  利用率 %s\n", $1, $2, $3, $4, $5 }'
}

# 空闲显存 >= GPU_POOL_MIN_FREE 的卡数
gpu_count_usable() {
  local exe; exe="$(command -v nvidia-smi || true)"
  [[ -n "$exe" ]] || { echo 0; return; }
  local out
  out="$("$exe" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$out" || "$out" == *"has failed"* ]]; then echo 0; return; fi
  printf '%s\n' "$out" |
    awk -v thr="$GPU_POOL_MIN_FREE" '$1+0 >= thr {n++} END {print n+0}'
}

# 有没有可用的 GPU（驱动正常）
gpu_available() {
  local exe; exe="$(command -v nvidia-smi || true)"
  [[ -n "$exe" ]] || return 1
  "$exe" -L >/dev/null 2>&1
}

check_env() {
  hdr "环境自检"
  [[ -x "$PY" ]] || die "Python 不可用: $PY"
  ok "python     = $PY"
  local c
  for c in rfd3 mpnn rf3; do
    if [[ -x "$RC_ENV_BIN/$c" ]]; then ok "$(printf '%-10s' "$c") = $RC_ENV_BIN/$c"
    else warn "$c 未找到（相关步骤会被跳过）"; fi
  done

  hdr "硬件"
  ok "CPU 逻辑核 = $NPROC     CPU 并行度 N_WORKERS = $N_WORKERS"
  if command -v free >/dev/null 2>&1; then
    ok "内存 $(free -h | awk '/^Mem:/ {print $2" 总 / "$7" 可用"}')"
  fi
  if gpu_available; then
    gpu_table
    ok "满足显存门槛（${GPU_POOL_MIN_FREE} MiB）的卡: $(gpu_count_usable) 张"
    if [[ "$MAX_PARALLEL_GPUS" == "auto" ]]; then
      ok "并行度 = auto（按上表决定）"
    else
      ok "并行度 = $MAX_PARALLEL_GPUS"
    fi
    if [[ -n "$GPUS" ]]; then ok "限定使用 GPUS=$GPUS"; fi
  else
    warn "nvidia-smi 不可用（驱动/容器直通问题）—— 只能串行，或你不在 GPU 节点上"
  fi

  hdr "路径与权重"
  ok "仓库   = $FOUNDRY_ROOT"
  ok "产物   = $OUT"
  case "$OUT" in
    /dev/shm/*|/tmp/*) warn "产物目录在 tmpfs 上，上千个 CIF 会吃内存，建议换成真实磁盘" ;;
  esac
  if [[ -f "$RFD3_CKPT" ]]; then ok "RFD3 权重 = $RFD3_CKPT"
  else warn "RFD3 权重缺失: $RFD3_CKPT"; fi
  ok "MPNN 权重目录 = $MPNN_CKPT_DIR"
  ok "规模   = $SCALE  (骨架数=$N_BACKBONES, 序列数=$MPNN_SEQS, seeds=$SEEDS)"
  ok "采样   = η=$STEP_SCALE γ0=$GAMMA_0 steps=$NUM_TIMESTEPS batch=$DIFFUSION_BATCH_SIZE low_memory=$LOW_MEMORY"
}

# --- 命令构造 ---------------------------------------------------------------
# 只*打印*一条 rfd3 design 命令（不执行），供 run_on_gpus 组装任务清单。
# 用法： rfd3_design_cmd <out_dir> <inputs.json> <骨架数> [额外 hydra 覆盖...]
rfd3_design_cmd() {
  local out_dir="$1" spec="$2" n_bb="$3"; shift 3
  local bs="$DIFFUSION_BATCH_SIZE"
  local nb=$(( (n_bb + bs - 1) / bs ))
  local extra=""
  if (( $# )); then printf -v extra ' %q' "$@"; fi
  printf '%s' "\"$RC_ENV_BIN/rfd3\" design \
out_dir=$(printf '%q' "$out_dir") \
inputs=$(printf '%q' "$spec") \
ckpt_path=$(printf '%q' "$RFD3_CKPT") \
diffusion_batch_size=$bs \
n_batches=$nb \
skip_existing=True \
prevalidate_inputs=True \
inference_sampler.num_timesteps=$NUM_TIMESTEPS \
inference_sampler.step_scale=$STEP_SCALE \
inference_sampler.gamma_0=$GAMMA_0 \
inference_sampler.n_recycle=$N_RECYCLE \
low_memory_mode=$LOW_MEMORY${extra}"
}

# 直接执行的版本（单条件、不需要多卡时用）
rfd3_design() {
  local out_dir="$1" spec="$2" n_bb="$3"; shift 3
  local bs="$DIFFUSION_BATCH_SIZE"
  local nb=$(( (n_bb + bs - 1) / bs ))
  mkdir -p "$out_dir"
  log "rfd3 design  ${n_bb} 骨架 = ${nb} batch x ${bs}  ->  $out_dir"
  eval "$(rfd3_design_cmd "$out_dir" "$spec" "$n_bb" "$@")"
}

# --- 多卡并行调度 -----------------------------------------------------------
# 从 stdin 读任务（每行一条完整 shell 命令），按显存情况铺到多张 GPU 上。
# 用法： build_jobs | run_on_gpus            # 并行度自动
#        build_jobs | run_on_gpus 4          # 最多 4 个任务同时跑
run_on_gpus() {
  local parallel="${1:-auto}"
  # 固定文件名，覆盖写入即可（避免依赖 rm，某些环境里有删除保护钩子）
  local jobs_file="$LOGS_DIR/gpu_pool/jobs_current.txt"
  mkdir -p "$LOGS_DIR/gpu_pool"
  cat > "$jobs_file"
  local n; n="$(grep -cvE '^\s*(#|$)' "$jobs_file" || true)"
  if [[ "${n:-0}" -eq 0 ]]; then
    warn "任务清单为空，跳过 ${FUNCNAME[1]:-}"; return 0
  fi
  log "提交 $n 个任务到 GPU 池（并行度=$parallel）"
  local -a args=(--jobs-file "$jobs_file" --min-free-mem "$GPU_POOL_MIN_FREE"
                 --job-mem "$GPU_JOB_MEM")
  [[ "$parallel" != "auto" ]] && args+=(--parallel "$parallel")
  "$PY" "$PAPER_ROOT/lib/gpu_pool.py" "${args[@]}"
}

# --- 序列设计 / 折叠的批量封装 ---------------------------------------------
mpnn_design() {
  local mtype="$1" struct="$2" out_dir="$3" n_seqs="$4" fixed="${5:-None}"
  local ckpt="$PROTEINMPNN_CKPT"
  [[ "$mtype" == "ligand_mpnn" ]] && ckpt="$LIGANDMPNN_CKPT"
  mkdir -p "$out_dir"
  local args=(--model_type "$mtype" --checkpoint_path "$ckpt" --is_legacy_weights True
              --structure_path "$struct" --out_directory "$out_dir"
              --write_fasta True --write_structures True
              --batch_size "$n_seqs" --number_of_batches 1)
  [[ "$fixed" != "None" ]] && args+=(--fixed_residues "$fixed")
  "$RC_ENV_BIN/mpnn" "${args[@]}"
}

fold_batch() {
  local spec="$1" out_dir="$2"
  case "$FOLD_BACKEND" in
    rf3)  mkdir -p "$out_dir"
          "$RC_ENV_BIN/rf3" fold inputs="$spec" out_dir="$out_dir" ckpt_path="$RF3_CKPT" ;;
    af3|chai) log "后端 $FOLD_BACKEND：输入已生成，请交给对应环境运行（见 COMMANDS.md）" ;;
    none) warn "FOLD_BACKEND=none，跳过结构预测" ;;
    *)    die "未知 FOLD_BACKEND=$FOLD_BACKEND" ;;
  esac
}

# --- 把一个多条件的规格拆成"每条件一个文件" ---------------------------------
# 输出 "<key>\t<path>" 行，方便直接喂给 while read 循环再铺到多张卡上。
split_spec() {
  local spec="$1" out_dir="$2"
  mkdir -p "$out_dir"
  "$PY" - "$spec" "$out_dir" <<'PY'
import json, sys, pathlib
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
for key, value in spec.items():
    p = out / f"{key}.json"
    p.write_text(json.dumps({key: value}, indent=2) + "\n")
    print(f"{key}\t{p}")
PY
}

# --- 结果登记 ---------------------------------------------------------------
# 判断 key 是否匹配给定的前缀列表；不传前缀 = 全部匹配。
# 用法： key_matches "$key" $DNA_TARGETS || continue
key_matches() {
  local key="$1"; shift
  (( $# == 0 )) && return 0
  local p
  for p in "$@"; do [[ "$key" == "$p"* ]] && return 0; done
  return 1
}

record_run() {
  local exp="$1" condition="$2" designs="$3" note="${4:-}"
  local f="$METRICS_DIR/runs.csv"
  [[ -f "$f" ]] || echo "timestamp,experiment,condition,n_backbones,scale,step_scale,gamma_0,timesteps,note" > "$f"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -Iseconds)" "$exp" "$condition" "$designs" "$SCALE" \
    "$STEP_SCALE" "$GAMMA_0" "$NUM_TIMESTEPS" "$note" >> "$f"
}

try_run() {
  local label="$1"; shift
  if "$@"; then ok "$label"; else warn "$label 失败（已跳过，汇总里能看到）"; return 1; fi
}
