#!/usr/bin/env bash
# =============================================================================
# 05_probe_vram.sh —— 实测出"刚好不 OOM 的最大 batch"，把显存吃满
#
# 为什么需要它
# ------------
# 单卡跑复现时，"节省时间"最大的杠杆就是 batch：扩散采样的每条样本互相独立，
# batch 只影响吞吐、不影响采样分布。但具体能开多大取决于卡型、序列长度、
# crop 原子数 —— 手册里的数字（论文默认 8）是为稳妥写的，A6000 49 GB 通常
# 能开到好几倍。
#
# 本脚本做的事：
#   1. 推理侧：递进试 diffusion_batch_size，每次真跑一次 design，用 nvidia-smi
#      轮询记录**峰值显存**，找到不 OOM 的最大值；
#   2. 训练侧：递进试 (diffusion_batch_size_train, crop_size, max_atoms_in_crop)
#      组合，跑 1 个 epoch 的少数样本（验证集缩到 2 个样本以省时间），同样记峰值；
#   3. 把结果写进 $OUT/vram_profile.json —— env.sh 下次会自动读回来，
#      所有实验脚本无需改任何参数就用了最优配置。
#
# 用法
# ----
#   source ./env.sh && source ./lib.sh
#   ./05_probe_vram.sh                 # 全量探测（约 20-40 分钟）
#   ./05_probe_vram.sh --infer-only    # 只探推理（约 10 分钟）
#   ./05_probe_vram.sh --train-only    # 只探训练
#   ./05_probe_vram.sh --dry-run       # 只打印将执行什么，不真跑
#   GPU_ID=3 ./05_probe_vram.sh        # 指定卡
#
# 产出
# ----
#   $OUT/vram_profile.json      推荐配置（env.sh 自动加载）
#   $OUT/reports/vram_probe.md  人读的探测报告（每个点的峰值显存）
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh
# env.sh / lib.sh 里带 set -Eeuo pipefail；探测脚本要容忍单个点失败（OOM 是预期结果），
# 所以这里显式关掉 -e，保留 -u 与 pipefail。
set +e

# ------------------------------------------------------------------ 参数 ---
INFER_ONLY=0
TRAIN_ONLY=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --infer-only) INFER_ONLY=1 ;;
    --train-only) TRAIN_ONLY=1 ;;
    --dry-run)    DRY_RUN=1 ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) die "未知参数: $arg（用 --help 看用法）" ;;
  esac
done

# 用哪张卡：默认取 GPUS 的第一张，否则 0
GPU_ID="${GPU_ID:-${GPUS%%,*}}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 探测用的 batch 序列。到第一个 OOM 就停，所以可以放胆列得高一些。
INFER_BATCHES="${INFER_BATCHES:-8 16 24 32 48 64 96}"
# 训练侧网格： bs:crop:atoms
TRAIN_GRID="${TRAIN_GRID:-8:256:1920 16:256:1920 16:384:3840 24:384:3840 32:512:5760}"
# 单个探测点的最长等待（秒），防止某个点卡死
PROBE_TIMEOUT="${PROBE_TIMEOUT:-1800}"
# 显存余量：推荐值取"成功的最大档"再留这么多 MiB，避免与别人的任务相撞
VRAM_HEADROOM_MB="${VRAM_HEADROOM_MB:-3000}"

PROBE_ROOT="$OUT/_vram_probe"
mkdir -p "$PROBE_ROOT" "$REPORTS_DIR"

# 推理探测用的 spec：优先用无条件单体（最短、最快）。没有就先准备。
PROBE_SPEC="${PROBE_SPEC:-$INPUTS_DIR/specs/exp1_uncond.json}"
PROBE_LENGTH="${PROBE_LENGTH:-200}"   # 用论文上限长度探测 => 结果对所有实验都安全

# ------------------------------------------------------------ 显存采样 ---
# ⚠️ nvidia-smi 在驱动通信失败时会把错误写到 **stdout**（不是 stderr），
#    所以必须校验拿到的是纯数字，否则 "NVIDIA-SMI has failed..." 会被当成
#    显存值去做算术，直接炸掉脚本。
_nvsmi_num() {  # $1=gpu index  $2=query field
  local exe out
  exe="$(command -v nvidia-smi || true)"
  [[ -n "$exe" ]] || { echo ""; return 0; }
  out="$("$exe" --id="$1" --query-gpu="$2" --format=csv,noheader,nounits 2>/dev/null |
         head -1 | tr -d ' ')" || true
  [[ "$out" =~ ^[0-9]+$ ]] || { echo ""; return 0; }
  echo "$out"
}

_gpu_used_mb()  { _nvsmi_num "$1" memory.used; }
_gpu_total_mb() { _nvsmi_num "$1" memory.total; }
_gpu_alive()    { [[ -n "$(_gpu_used_mb "$1")" ]]; }

# 运行命令并记录 GPU 峰值占用（相对启动前的基线）。
# 用法： run_peak <结果变量名> <日志文件> <命令...>
run_peak() {
  local __peakvar="$1" logf="$2"; shift 2
  local base peak=0 cur rc pid

  base="$(_gpu_used_mb "$GPU_ID")"; [[ -n "$base" ]] || base=0
  timeout -k 30 "$PROBE_TIMEOUT" "$@" >"$logf" 2>&1 &
  pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    cur="$(_gpu_used_mb "$GPU_ID")"
    if [[ -n "$cur" ]] && (( cur > peak )); then peak="$cur"; fi
    sleep 2
  done
  wait "$pid"; rc=$?

  local delta=$(( peak - base )); (( delta < 0 )) && delta=0
  printf -v "$__peakvar" '%s' "$delta"
  return "$rc"
}

_is_oom() {  # $1 = 日志文件
  grep -qiE "CUDA out of memory|OutOfMemoryError|HIP out of memory|cudaErrorMemoryAllocation" "$1" 2>/dev/null
}

_now() { date '+%F %T'; }

# 结果累积：每行 "kind|setting|peak_mb|status"
RESULTS_FILE="$PROBE_ROOT/results.psv"
: > "$RESULTS_FILE"
_record() { printf '%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS_FILE"; }

# ------------------------------------------------------- 0. 前置检查 ---
hdr "显存探测（GPU $GPU_ID）"
if ! _gpu_alive "$GPU_ID"; then
  if (( DRY_RUN )); then
    warn "GPU $GPU_ID 查不到显存信息（dry-run 忽略）—— 没有 nvidia-smi 或驱动通信失败"
  else
    die "GPU $GPU_ID 查不到显存信息：nvidia-smi 不可用或驱动通信失败。换 GPU_ID=<空闲卡> 或先修驱动。"
  fi
fi
_require_or_warn() {  # dry-run 时只警告，方便在任何机器上先看流程
  local what="$1" path="$2"
  if [[ -e "$path" ]]; then return 0; fi
  if (( DRY_RUN )); then warn "缺少 $what: $path（dry-run 忽略）"; return 0; fi
  die "缺少 $what: $path"
}
_require_or_warn "rfd3 可执行文件" "$RC_ENV_BIN/rfd3"
_require_or_warn "RFD3 权重" "$RFD3_CKPT"

if _gpu_alive "$GPU_ID"; then
  _used_mb="$(_gpu_used_mb "$GPU_ID")"
  total_mb="$(_gpu_total_mb "$GPU_ID")"
  ok "GPU $GPU_ID: 已用 ${_used_mb} MiB / 共 ${total_mb} MiB（记录的是相对基线增量）"
  if (( total_mb > 0 && _used_mb * 100 / total_mb > 30 )); then
    warn "这张卡已用 $(( _used_mb * 100 / total_mb ))% 显存 —— 探测结果会偏保守。"
    warn "想更准就等它空出来，或换 GPU_ID=<空闲卡> 重跑。"
  fi
fi

# 推理探测需要的输入 spec
if (( ! TRAIN_ONLY )) && (( ! DRY_RUN )) && [[ ! -f "$PROBE_SPEC" ]]; then
  log "生成无条件单体 spec（$PROBE_SPEC）..."
  "$PY" 01_prepare_inputs.py --only uncond || die "01_prepare_inputs.py 失败"
fi

# ---------------------------------------------------------- 1. 推理探测 ---
INFER_BEST=0
INFER_BEST_PEAK=0
if (( ! TRAIN_ONLY )); then
  hdr "1/2 推理：找最大的 diffusion_batch_size"
  log "探测点: $INFER_BATCHES   （长度 L=$PROBE_LENGTH，n_batches=1，每点真跑一次 design）"
  echo

  for bs in $INFER_BATCHES; do
    out_dir="$PROBE_ROOT/infer_bs${bs}"
    logf="$PROBE_ROOT/infer_bs${bs}.log"
    mkdir -p "$out_dir"

    cmd=("$RC_ENV_BIN/rfd3" design
         "out_dir=$out_dir"
         "inputs=$PROBE_SPEC"
         "ckpt_path=$RFD3_CKPT"
         "diffusion_batch_size=$bs"
         "n_batches=1"
         "skip_existing=False"
         "prevalidate_inputs=True"
         "inference_sampler.num_timesteps=$NUM_TIMESTEPS"
         "inference_sampler.step_scale=$STEP_SCALE"
         "inference_sampler.gamma_0=$GAMMA_0"
         "inference_sampler.n_recycle=$N_RECYCLE"
         "low_memory_mode=$LOW_MEMORY")

    if (( DRY_RUN )); then
      printf '  [dry-run] bs=%-3s  %s\n' "$bs" "${cmd[*]}"
      continue
    fi

    printf '  [%s] bs=%-3s 运行中 ...' "$(_now)" "$bs"
    peak=""
    run_peak peak "$logf" "${cmd[@]}"
    rc=$?

    if (( rc == 0 )); then
      printf '\r  [%s] bs=%-3s 峰值 %6s MiB  ✓\n' "$(_now)" "$bs" "$peak"
      _record infer "bs=$bs" "$peak" "ok"
      INFER_BEST="$bs"; INFER_BEST_PEAK="$peak"
    elif _is_oom "$logf"; then
      printf '\r  [%s] bs=%-3s 峰值 %6s MiB  ✗ OOM —— 到此为止\n' "$(_now)" "$bs" "$peak"
      _record infer "bs=$bs" "$peak" "oom"
      break
    else
      printf '\r  [%s] bs=%-3s 失败（非 OOM，rc=%s）\n' "$(_now)" "$bs" "$rc"
      warn "  看日志: $logf"
      _record infer "bs=$bs" "$peak" "error:rc=$rc"
      break
    fi
  done

  if (( INFER_BEST == 0 )) && (( ! DRY_RUN )); then
    warn "推理探测没有成功点 —— 保持论文默认 diffusion_batch_size=8"
    INFER_BEST=8
  fi
fi

# ---------------------------------------------------------- 2. 训练探测 ---
TRAIN_BEST_BS=0; TRAIN_BEST_CROP=0; TRAIN_BEST_ATOMS=0; TRAIN_BEST_PEAK=0
if (( ! INFER_ONLY )); then
  hdr "2/2 训练：找最大的 (batch, crop_size, max_atoms_in_crop)"
  _require_or_warn "interfaces_df.parquet" "$PDB_PARQUET_DIR/interfaces_df.parquet"
  _require_or_warn "pn_units_df.parquet" "$PDB_PARQUET_DIR/pn_units_df.parquet"
  log "探测点: $TRAIN_GRID   （每点 1 个 epoch × 2 个训练样本 + 2 个验证样本）"
  echo

  for point in $TRAIN_GRID; do
    IFS=':' read -r bs crop atoms <<<"$point"
    log_dir="$PROBE_ROOT/train_bs${bs}_crop${crop}_atoms${atoms}"
    logf="$PROBE_ROOT/train_bs${bs}_crop${crop}_atoms${atoms}.log"
    mkdir -p "$log_dir"

    cmd=(env "PROJECT_ROOT=$FOUNDRY_ROOT" "$PY" models/rfd3/src/rfd3/train_lora.py
         "experiment=nmf_zkp_single_layer_pdb"
         "paths.data.pdb_data_dir=$PDB_MIRROR"
         "paths.data.pdb_parquet_dir=$PDB_PARQUET_DIR"
         "paths.log_dir=$log_dir"
         "logger=csv"
         "seed=42"
         "nmf.enabled=true"
         "nmf.target_keywords=[token_initializer.process_s_init.1]"
         "nmf.match_mode=exact" "nmf.max_replacements=1"
         "nmf.apply_to_token_initializer=true"
         "ckpt_config.path=$RFD3_CKPT"
         "ckpt_config.reset_optimizer=true"
         "save_checkpoints=false"
         "trainer.max_epochs=2"
         "trainer.n_examples_per_epoch=2"
         "trainer.validate_every_n_epochs=1000000000"
         "trainer.checkpoint_every_n_epochs=1000000000"
         "datasets.diffusion_batch_size_train=$bs"
         "datasets.crop_size=$crop"
         "datasets.max_atoms_in_crop=$atoms"
         "datasets.val.pdb_holdout.max_examples=2"
         "dataloader.train.dataloader_params.num_workers=2"
         "dataloader.train.dataloader_params.prefetch_factor=2")

    if (( DRY_RUN )); then
      printf '  [dry-run] bs=%-3s crop=%-4s atoms=%-5s  cd %s && %s ...\n' \
             "$bs" "$crop" "$atoms" "$FOUNDRY_ROOT" "${cmd[*]:0:5}"
      continue
    fi

    printf '  [%s] bs=%-3s crop=%-4s atoms=%-5s 运行中 ...' "$(_now)" "$bs" "$crop" "$atoms"
    peak=""
    ( cd "$FOUNDRY_ROOT" && run_peak peak "$logf" "${cmd[@]}" )
    rc=$?

    if (( rc == 0 )); then
      printf '\r  [%s] bs=%-3s crop=%-4s atoms=%-5s 峰值 %6s MiB  ✓\n' \
             "$(_now)" "$bs" "$crop" "$atoms" "$peak"
      _record train "bs=$bs,crop=$crop,atoms=$atoms" "$peak" "ok"
      TRAIN_BEST_BS="$bs"; TRAIN_BEST_CROP="$crop"
      TRAIN_BEST_ATOMS="$atoms"; TRAIN_BEST_PEAK="$peak"
    elif _is_oom "$logf"; then
      printf '\r  [%s] bs=%-3s crop=%-4s atoms=%-5s 峰值 %6s MiB  ✗ OOM\n' \
             "$(_now)" "$bs" "$crop" "$atoms" "$peak"
      _record train "bs=$bs,crop=$crop,atoms=$atoms" "$peak" "oom"
      break
    else
      printf '\r  [%s] bs=%-3s crop=%-4s atoms=%-5s 失败（非 OOM，rc=%s）\n' \
             "$(_now)" "$bs" "$crop" "$atoms" "$rc"
      warn "  看日志: $logf"
      _record train "bs=$bs,crop=$crop,atoms=$atoms" "$peak" "error:rc=$rc"
      break
    fi
  done

  if (( TRAIN_BEST_BS == 0 )) && (( ! DRY_RUN )); then
    warn "训练探测没有成功点 —— 保持论文默认 (bs=4, crop=256, atoms=1920)"
    TRAIN_BEST_BS=4; TRAIN_BEST_CROP=256; TRAIN_BEST_ATOMS=1920
  fi
fi

if (( DRY_RUN )); then
  hdr "dry-run 结束（没有真跑任何任务）"
  exit 0
fi

# ------------------------------------------------------------ 3. 汇总 ---
hdr "探测结果"

# batch 取 8 的整数倍并留余量（更整齐、也更容易和论文口径对照）
_round_batch() {
  local v="$1"
  (( v < 8 )) && { echo "$v"; return; }
  echo $(( (v / 8) * 8 ))
}

INFER_TOTAL="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
REC_INFER="$INFER_BEST"
if (( INFER_BEST_PEAK > 0 && INFER_TOTAL > 0 )); then
  # 按峰值线性外推：留出 headroom 后的 batch 上限
  usable=$(( INFER_TOTAL - VRAM_HEADROOM_MB ))
  ext=$(( INFER_BEST * usable / INFER_BEST_PEAK ))
  ext="$(_round_batch "$ext")"
  (( ext > INFER_BEST )) && REC_INFER="$ext"
fi
(( REC_INFER > 0 )) || REC_INFER=8

REC_TRAIN_BS="$TRAIN_BEST_BS"
(( REC_TRAIN_BS > 0 )) || REC_TRAIN_BS=4

# JSON：env.sh 会自动读这些键
cat > "$OUT/vram_profile.json" <<JSON
{
  "DIFFUSION_BATCH_SIZE": $REC_INFER,
  "DIFFUSION_BS_TRAIN": $REC_TRAIN_BS,
  "CROP_SIZE": ${TRAIN_BEST_CROP:-256},
  "MAX_ATOMS_IN_CROP": ${TRAIN_BEST_ATOMS:-1920},
  "N_EXAMPLES": ${N_EXAMPLES:-128},
  "NUM_WORKERS": ${NUM_WORKERS:-8},
  "_gpu": "$GPU_ID",
  "_gpu_total_mb": ${INFER_TOTAL:-0},
  "_probed_infer_best": $INFER_BEST,
  "_probed_infer_best_peak_mb": ${INFER_BEST_PEAK:-0},
  "_probed_train_best": "${TRAIN_BEST_BS}x${TRAIN_BEST_CROP}x${TRAIN_BEST_ATOMS}",
  "_probed_train_best_peak_mb": ${TRAIN_BEST_PEAK:-0},
  "_headroom_mb": $VRAM_HEADROOM_MB,
  "_probe_time": "$(_now)"
}
JSON

{
  echo "# 显存探测报告"
  echo
  echo "- GPU: $GPU_ID（共 ${INFER_TOTAL:-?} MiB）"
  echo "- 时间: $(_now)"
  echo "- 推理探测长度: L=$PROBE_LENGTH，n_batches=1"
  echo "- 预留余量: ${VRAM_HEADROOM_MB} MiB"
  echo
  echo "| 类型 | 设置 | 峰值显存 (MiB) | 状态 |"
  echo "|---|---|---|---|"
  while IFS='|' read -r kind setting peak status; do
    [[ -z "$kind" ]] && continue
    echo "| $kind | $setting | $peak | $status |"
  done < "$RESULTS_FILE"
  echo
  echo "## 推荐值（已写入 \$OUT/vram_profile.json）"
  echo
  echo '```json'
  cat "$OUT/vram_profile.json"
  echo '```'
  echo
  echo "> 说明：推理 batch 只影响吞吐，不影响采样分布，可以放心改；"
  echo "> 训练 batch 会改变梯度平均，**改了以后 baseline 与各 NMF/LoRA 变体必须同用同一值**。"
} > "$REPORTS_DIR/vram_probe.md"

ok "推理 diffusion_batch_size : 实测最大 $INFER_BEST（峰值 ${INFER_BEST_PEAK} MiB）-> 推荐 $REC_INFER"
ok "训练 (bs, crop, atoms)   : 实测最大 "${TRAIN_BEST_BS}x${TRAIN_BEST_CROP}x${TRAIN_BEST_ATOMS}"（峰值 ${TRAIN_BEST_PEAK} MiB）"
echo
ok "已写入 $OUT/vram_profile.json   （下次 source env.sh 会自动加载）"
ok "人读报告: $REPORTS_DIR/vram_probe.md"
echo
log "提示：想让这些值生效，重新 source env.sh 即可，例如："
log "  source ./env.sh && source ./lib.sh && check_env"
