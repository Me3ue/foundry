#!/usr/bin/env bash
# =============================================================================
# 实验 8 —— 推理速度标定（论文 Fig. 1d）
#
# 论文做法：
#   * 在 NVIDIA A6000 上测 RFD1 / RFD2 / RFD3 在不同蛋白长度下的推理时间
#   * 结论：RFD3 比 RFD2 快约 10 倍，比 RFD1 快一个数量级
#
# ⚠️ 本仓库只包含 RFD3。要复现完整曲线需要另外克隆：
#      RFdiffusion1: https://github.com/RosettaCommons/RFdiffusion
#      RFdiffusion2: 见 Ahern et al. 2025 的官方仓库
#    本脚本负责把 RFD3 那一条曲线测出来，并把结果写成 Fig. 1d 的格式。
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
DEST="$DESIGNS_DIR/exp8_speed"
RESULT="$METRICS_DIR/speed_scaling.csv"

[[ -f "$RESULT" ]] || echo "backend,length,repeat,wallclock_s,timesteps,step_scale" > "$RESULT"

"$PY" - "$LENGTHS" "$INPUTS_DIR/specs/exp8_speed.json" <<'PY'
import json, sys
lengths = [int(x) for x in sys.argv[1].split()]
spec = {f"L{L}": {"length": L, "is_non_loopy": True} for L in lengths}
json.dump(spec, open(sys.argv[2], "w"), indent=2)
print(f"长度点: {lengths}")
PY

for L in $LENGTHS; do
  sub="$INPUTS_DIR/specs/exp8_speed_L${L}.json"
  "$PY" - "$INPUTS_DIR/specs/exp8_speed.json" "$sub" "L${L}" <<'PY'
import json, sys
json.dump({k: v for k, v in json.load(open(sys.argv[1])).items() if k == sys.argv[3]},
          open(sys.argv[2], "w"), indent=2)
PY
  for r in $(seq 1 "$REPEATS"); do
    run_dir="$DEST/L${L}/rep_${r}"
    log="$LOGS_DIR/exp8_speed_L${L}_rep${r}.log"
    log "长度=$L 第 $r 次  一步一骨架（测纯推理时间）"
    # 1 个骨架，避免批处理把 per-sample 时间摊薄，贴近论文口径
    start=$(date +%s.%N)
    rfd3_design "$run_dir" "$sub" 1 diffusion_batch_size=1 n_batches=1 \
        inference_sampler.num_timesteps="$STEP_TIMESTEPS" > "$log" 2>&1 \
        || warn "长度 $L 第 $r 次失败"
    end=$(date +%s.%N)
    secs=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}')
    printf 'rfd3,%s,%s,%s,%s,%s\n' "$L" "$r" "$secs" "$STEP_TIMESTEPS" "$STEP_SCALE" >> "$RESULT"
    ok "长度=$L  第 $r 次  ${secs}s"
  done
done

hdr "实验 8 完成"
ok "结果: $RESULT"
ok "画图：python 91_plot_speed.py（或直接把 CSV 丢进任意绘图工具）"
