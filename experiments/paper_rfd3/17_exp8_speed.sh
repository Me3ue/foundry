#!/usr/bin/env bash
# =============================================================================
# 实验 8 —— 推理速度标定（论文 Fig. 1d）
#
# 论文做法：
#   * 在 **NVIDIA A6000** 上测 RFD1 / RFD2 / RFD3 在不同蛋白长度下的推理时间
#   * 结论：RFD3 比 RFD2 快约 10 倍，比 RFD1 快一个数量级
#
# ✅ 你的机器就是 6×RTX A6000，**与论文 Fig. 1d 的硬件完全一致**，
#    所以这里测出来的绝对秒数可以直接和论文对照。
#
# ⚠️ 测速必须独占 GPU：不能和其他任务抢卡，否则数字没意义。
#    所以本脚本**故意串行执行**，并且会挑当前最空闲的一张卡。
#    跑之前建议先确认没有别的实验在跑（nvidia-smi 看一眼）。
#
# ⚠️ 本仓库只包含 RFD3。要画完整的 RFD1/RFD2 曲线还需另外克隆：
#      RFdiffusion1: https://github.com/RosettaCommons/RFDiffusion
#      RFdiffusion2: https://github.com/RosettaCommons/RFdiffusion2
#
# 运行：
#   LENGTHS="50 100 150 200 250 300" REPEATS=3 ./17_exp8_speed.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

hdr "实验 8：推理速度标定（Fig. 1d）"
LENGTHS="${LENGTHS:-50 100 150 200 250 300}"
REPEATS="${REPEATS:-3}"
STEP_TIMESTEPS="${STEP_TIMESTEPS:-$NUM_TIMESTEPS}"
SPEED_BATCH="${SPEED_BATCH:-1}"          # 单骨架，测纯 per-sample 时间
DEST="$DESIGNS_DIR/exp8_speed"
RESULT="$METRICS_DIR/speed_scaling.csv"
[[ -f "$RESULT" ]] || echo "backend,length,repeat,wallclock_s,timesteps,step_scale,gpu" > "$RESULT"

# 挑一张最空闲的卡独占（用 --query-gpu 排序，取空闲显存最大且利用率最低的）
pick_idle_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
             --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' '{printf "%s %s %s\n", $3+0, -$2, $1}' | sort -n | head -1 | awk '{print $3}'
}
IDLE_GPU="$(pick_idle_gpu)"
[[ -n "$IDLE_GPU" ]] && ok "独占 GPU $IDLE_GPU 做测速（其余任务请勿占用它）"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$IDLE_GPU}"

"$PY" - "$LENGTHS" "$INPUTS_DIR/specs/exp8_speed.json" <<'PY'
import json, sys
lengths = [int(x) for x in sys.argv[1].split()]
json.dump({f"L{L}": {"length": L, "is_non_loopy": True} for L in lengths},
          open(sys.argv[2], "w"), indent=2)
print(f"长度点: {lengths}")
PY

for L in $LENGTHS; do
  sub="$INPUTS_DIR/specs/exp8_speed_L${L}.json"
  "$PY" - "$INPUTS_DIR/specs/exp8_speed.json" "$sub" "L${L}" <<'PY'
import json, sys
json.dump({k: v for k, v in json.load(open(sys.argv[1])).items() if k == sys.argv[3]},
          open(sys.argv[2], "w"), indent=2)
PY
  # 预热一次（避免把 CUDA 上下文初始化时间算进去），不计入数据
  if [[ "$REPEATS" -gt 0 ]]; then
    log "预热：L=$L（结果丢弃）"
    rfd3_design "$DEST/L${L}/warmup" "$sub" 1 \
      "diffusion_batch_size=1" "n_batches=1" \
      "inference_sampler.num_timesteps=$STEP_TIMESTEPS" \
      > "$LOGS_DIR/exp8_speed_L${L}_warmup.log" 2>&1 || true
  fi
  for r in $(seq 1 "$REPEATS"); do
    run_dir="$DEST/L${L}/rep_${r}"
    logfile="$LOGS_DIR/exp8_speed_L${L}_rep${r}.log"
    log "长度=$L 第 $r/$REPEATS 次（单骨架，独占 GPU ${CUDA_VISIBLE_DEVICES:-auto}）"
    start=$(date +%s.%N)
    rfd3_design "$run_dir" "$sub" 1 \
      "diffusion_batch_size=$SPEED_BATCH" "n_batches=1" \
      "inference_sampler.num_timesteps=$STEP_TIMESTEPS" \
      > "$logfile" 2>&1 || warn "长度 $L 第 $r 次失败"
    end=$(date +%s.%N)
    secs=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}')
    printf 'rfd3,%s,%s,%s,%s,%s,%s\n' "$L" "$r" "$secs" \
      "$STEP_TIMESTEPS" "$STEP_SCALE" "${CUDA_VISIBLE_DEVICES:-auto}" >> "$RESULT"
    ok "长度=$L  第 $r 次  ${secs}s"
  done
done

hdr "实验 8 完成"
ok "结果: $RESULT"
"$PY" 91_plot_speed.py || warn "绘图失败（缺数据？）"
