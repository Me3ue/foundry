#!/usr/bin/env bash
# =============================================================================
# 公共 bash 工具：日志、计时、命令封装、环境自检
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
require_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || die "找不到命令 '$c'。请检查 RC_ENV_BIN=$RC_ENV_BIN"
}
require_file() { [[ -f "$1" ]] || die "缺少文件: $1"; }

check_env() {
  hdr "环境自检"
  [[ -x "$PY" ]] || die "Python 不可用: $PY"
  ok "python  = $PY"
  for c in rfd3 mpnn; do
    if [[ -x "$RC_ENV_BIN/$c" ]]; then ok "$c     = $RC_ENV_BIN/$c"; else warn "$c 未找到（相关实验会被跳过）"; fi
  done
  if [[ "$FOLD_BACKEND" == "rf3" ]]; then
    [[ -x "$RC_ENV_BIN/rf3" ]] && ok "rf3     = $RC_ENV_BIN/rf3" || warn "rf3 未找到"
  fi
  ok "SCALE=$SCALE  N_BACKBONES=$N_BACKBONES  MPNN_SEQS=$MPNN_SEQS  SEEDS=$SEEDS"
  warn "RFD3 权重: $RFD3_CKPT $( [[ -f $RFD3_CKPT ]] && echo '(存在)' || echo '(缺失!)' )"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || warn "nvidia-smi 不可用"
}

# --- 便捷封装 ---------------------------------------------------------------
# 生成 N 个骨架：拆成 n_batches 次，每次 diffusion_batch_size 个样本
# 用法： rfd3_design <out_dir> <inputs.json> <backbones> [额外 hydra 覆盖...]
rfd3_design() {
  local out_dir="$1" spec="$2" n_bb="$3"; shift 3
  local bs="$DIFFUSION_BATCH_SIZE"
  local nb=$(( (n_bb + bs - 1) / bs ))
  mkdir -p "$out_dir"
  log "rfd3 design  ${n_bb} 骨架 = ${nb} batch x ${bs}  ->  $out_dir"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  "$RC_ENV_BIN/rfd3" design \
      out_dir="$out_dir" \
      inputs="$spec" \
      ckpt_path="$RFD3_CKPT" \
      diffusion_batch_size="$bs" \
      n_batches="$nb" \
      skip_existing=True \
      prevalidate_inputs=True \
      inference_sampler.num_timesteps="$NUM_TIMESTEPS" \
      inference_sampler.step_scale="$STEP_SCALE" \
      inference_sampler.gamma_0="$GAMMA_0" \
      inference_sampler.n_recycle="$N_RECYCLE" \
      low_memory_mode="$LOW_MEMORY" \
      "$@"
}

# MPNN 序列设计（蛋白-蛋白 / DNA 用 ProteinMPNN，小分子 / 酶用 LigandMPNN）
# 用法： mpnn_design <model_type> <structure> <out_dir> <n_seqs> [固定残基 JSON]
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

# 结构预测（自洽性评估）
# 用法： fold_batch <inputs.json|dir> <out_dir>
fold_batch() {
  local spec="$1" out_dir="$2"
  case "$FOLD_BACKEND" in
    rf3)  mkdir -p "$out_dir"
          "$RC_ENV_BIN/rf3" fold inputs="$spec" out_dir="$out_dir" ckpt_path="$RF3_CKPT" ;;
    af3)  log "AF3 后端：请把 $spec 交给你的 AlphaFold3 运行环境（见 README §评估后端）"; return 0 ;;
    chai) log "Chai-1 后端：请把 $spec 交给 chai-1 运行环境（见 README §评估后端）"; return 0 ;;
    none) warn "FOLD_BACKEND=none，跳过结构预测" ; return 0 ;;
    *)    die "未知 FOLD_BACKEND=$FOLD_BACKEND" ;;
  esac
}

# --- 结果登记 ---------------------------------------------------------------
# 每个实验跑完后写一行到 out/metrics/runs.csv，供 summarize_all.py 汇总
record_run() {
  local exp="$1" condition="$2" designs="$3" note="${4:-}"
  local f="$METRICS_DIR/runs.csv"
  [[ -f "$f" ]] || echo "timestamp,experiment,condition,n_backbones,scale,step_scale,gamma_0,timesteps,note" > "$f"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -Iseconds)" "$exp" "$condition" "$designs" "$SCALE" \
    "$STEP_SCALE" "$GAMMA_0" "$NUM_TIMESTEPS" "$note" >> "$f"
}

# 安全运行：失败只警告不中断整套流程（长跑必备）
try_run() {
  local label="$1"; shift
  if "$@"; then ok "$label"; else warn "$label 失败（已跳过，稍后可在汇总中看到）"; return 1; fi
}
