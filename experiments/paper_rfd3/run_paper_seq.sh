#!/usr/bin/env bash
# =============================================================================
# run_paper_seq.sh —— 单卡顺序跑完论文正文的全部实验
#
# 和 run_all.sh 的区别：把「骨架采样 / 序列设计 / 折叠」拆成**三个独立阶段**，
# 每阶段有完成标记。单卡长跑最怕中断，拆开以后可以：
#   * 先出全部骨架（几小时），确认没问题再去做序列和折叠；
#   * 中途断了从断点继续，不用从头再来；
#   * 折叠太慢时单独分批推。
#
# 一条命令跑完全部（推荐）：
#   SINGLE_GPU=1 SCALE=half PROBE=1 ./run_paper_seq.sh
#
# 分阶段（可反复调用，已完成的阶段会跳过）：
#   SINGLE_GPU=1 SCALE=half ./run_paper_seq.sh              # PHASE=all，一气呵成
#   SINGLE_GPU=1 SCALE=half PHASE=sample ./run_paper_seq.sh # 只采骨架
#   SINGLE_GPU=1 SCALE=half PHASE=seq    ./run_paper_seq.sh # 只做序列设计
#   SINGLE_GPU=1 SCALE=half PHASE=fold   ./run_paper_seq.sh # 只做折叠
#   SINGLE_GPU=1             PHASE=report ./run_paper_seq.sh # 只出指标+报告
#
# 常用开关：
#   SCALE=quick|half|paper   规模（quick 冒烟 / half 有统计意义 / paper 论文原规模）
#   PROBE=1                  先跑 05_probe_vram.sh 把显存吃满（只做一次）
#   RESUME=1                 跳过已完成的阶段（断点续跑）
#   WITH_SEQ=0               不做序列设计（PHASE=all 时生效）
#   WITH_FOLD=0              不做折叠 —— 单卡强烈建议先这样出一批骨架
#   ONLY="2 3"               只跑指定实验
#   GPUS=3                   指定卡；留空 + SINGLE_GPU=1 时自动挑最空闲的
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

PHASE="${PHASE:-all}"
PROBE="${PROBE:-0}"
RESUME="${RESUME:-0}"
ONLY="${ONLY:-1 2 3 4 5 6 7 8 9}"

STAGE_DIR="$OUT/_stages"
mkdir -p "$STAGE_DIR" "$LOGS_DIR"

START=$(date +%s)

# ------------------------------------------------------------ 阶段工具 ---
_stage_done() { [[ -f "$STAGE_DIR/$1.done" ]]; }
_mark_stage()  { date '+%F %T' > "$STAGE_DIR/$1.done"; }
_want() {  # PHASE=all 时四个阶段都要；否则只做指定的一个
  [[ "$PHASE" == "all" || "$PHASE" == "$1" ]]
}
_run_stage() {
  local name="$1"; shift
  if (( RESUME )) && _stage_done "$name"; then
    warn "阶段 $name 已完成（RESUME=1），跳过"
    return 0
  fi
  hdr "阶段：$name"
  "$@" || warn "阶段 $name 有失败项，继续后续阶段（详情见上面的日志）"
  _mark_stage "$name"
}

_experiments() {
  cat <<'EOF'
1|10_exp1_unconditional.sh
2|11_exp2_ppi.sh
3|12_exp3_dna.sh
4|13_exp4_small_molecule.sh
5|14_exp5_enzyme.sh
6|15_exp6_symmetry.sh
7|16_exp7_conditioning.sh
8|17_exp8_speed.sh
9|18_exp9_wetlab_insilico.sh
EOF
}

# 跑 9 个实验；$1=WITH_SEQ  $2=WITH_FOLD
run_experiments() {
  local seq="$1" fold="$2" id script
  while IFS='|' read -r id script; do
    [[ -z "$id" ]] && continue
    if [[ " $ONLY " != *" $id "* ]]; then
      warn "跳过实验 $id（ONLY=$ONLY）"
      continue
    fi
    hdr "实验 $id -> $script   (WITH_SEQ=$seq WITH_FOLD=$fold)"
    if ! WITH_SEQ="$seq" WITH_FOLD="$fold" bash "$script"; then
      warn "实验 $id 失败，继续后续实验"
    fi
  done < <(_experiments)
}

# ------------------------------------------------------------ 各阶段实现 ---
phase_probe() {
  # PROBE=1 强制探测；默认只在缺 profile 时提示，不阻塞（探测要 20-40 分钟，
  # 只想先冒烟跑通链路时不该被迫等）。想让一次命令全包，就加 PROBE=1。
  if (( PROBE )); then
    GPU_ID="${GPU_ID:-${GPUS%%,*}}" ./05_probe_vram.sh
  elif [[ -f "$OUT/vram_profile.json" ]]; then
    ok "已有显存档位 $OUT/vram_profile.json（实测值已生效）"
  else
    warn "还没做过显存探测 —— 本次用手册默认 batch（推理=$DIFFUSION_BATCH_SIZE，"
    warn "训练=$DIFFUSION_BS_TRAIN）。A6000 49 GB 上这远没用满。"
    warn "想把显存吃满、明显缩短墙钟，另开一个终端跑一次（20-40 分钟，只需一次）："
    warn "    cd $(pwd) && GPU_ID=${GPUS%%,*} ./05_probe_vram.sh"
    warn "跑完重跑本脚本即可自动加载；或者本次直接用 PROBE=1 一起跑。"
  fi
}

phase_inputs() {
  hdr "从仓库配置导出论文 benchmark 定义"
  "$PY" 02_extract_repo_benchmarks.py || \
    warn "benchmark 导出失败（PPI 会退回仓库自带的 tutorial 结构）"

  hdr "整理输入结构"
  if [[ -d "$PDB_MIRROR_PATH" ]]; then
    ok "使用本地 PDB 镜像: $PDB_MIRROR_PATH"
    "$PY" 01_prepare_inputs.py --mirror "$PDB_MIRROR_PATH"
  else
    warn "本地镜像不存在（$PDB_MIRROR_PATH）—— 改为从 RCSB 逐文件下载（论文 §3 只需 ~21 个结构）"
    "$PY" 01_prepare_inputs.py --no-mirror
  fi
}

phase_metrics() {
  hdr "几何指标（RMSD / 界面 / RASA / 氢键 / clash，${N_WORKERS} 进程并行）"
  "$PY" 70_metrics_geometry.py --workers "$N_WORKERS" || warn "几何指标计算失败"
}

phase_report() {
  hdr "汇总报告"
  "$PY" 90_summarize.py
  hdr "Fig. 1d 速度曲线"
  "$PY" 91_plot_speed.py || warn "速度数据还没生成（实验 8 未跑或未完成）"
}

# --------------------------------------------------------------- 状态表 ---
status_table() {
  hdr "实验状态"
  printf '  %-34s %9s %9s\n' "实验目录" "骨架数" "denoised"
  printf '  %-34s %9s %9s\n' "----------------------------------" "---------" "---------"
  local d name total denoised
  for d in "$DESIGNS_DIR"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    total="$(find "$d" -name '*.cif.gz' 2>/dev/null | wc -l)"
    denoised="$(find "$d" -name '*_denoised_*.cif.gz' 2>/dev/null | wc -l)"
    printf '  %-34s %9s %9s\n' "$name" "$(( total - denoised ))" "$denoised"
  done
  echo
  hdr "需要你手工补的输入（补上后重跑对应实验即可）"
  local missing=0
  if [[ ! -f "$INPUTS_DIR/targets/ame_cases.json" ]]; then
    warn "实验 5（酶 AME）：缺 $INPUTS_DIR/targets/ame_cases.json —— 41 个活性位点案例"
    warn "    模板已由 01_prepare_inputs.py 生成，按 RFdiffusion2 仓库的 benchmark 补齐"
    missing=1
  fi
  if [[ ! -d "$INPUTS_DIR/motifs" ]] || [[ -z "$(ls -A "$INPUTS_DIR/motifs" 2>/dev/null)" ]]; then
    warn "实验 9（湿实验 in silico）：缺 $INPUTS_DIR/motifs/ —— DBRFD3 / 半胱氨酸水解酶的 motif"
    warn "    缺文件时脚本会用占位文件跑通流程，但指标无意义"
    missing=1
  fi
  if (( ! missing )); then
    ok "论文 §3 所需的输入都在（实验 5 与 9 的补充材料除外，见上）"
  fi
}

# ================================================================= 主流程 ===
check_env

if _want probe; then
  _run_stage probe phase_probe
  # 探测写完 profile 后重载，让后面的阶段用上新值
  source ./env.sh
  source ./lib.sh
fi

if _want inputs; then
  _run_stage inputs phase_inputs
fi

# 三阶段：骨架 -> 序列 -> 折叠。
# PHASE=all 时用"每个实验一次做完"的经典顺序（WITH_SEQ / WITH_FOLD 可关掉后两段）；
# 显式指定 PHASE=sample/seq/fold 时才按阶段拆开（此时其它阶段各自会重跑一遍
# 采样，但 RFD3 带 skip_existing=True，已有骨架会被跳过，只是多一次进程启动开销）。
if [[ "$PHASE" == "all" ]]; then
  _run_stage experiments run_experiments "${WITH_SEQ:-1}" "${WITH_FOLD:-1}"
else
  if _want sample; then _run_stage sample run_experiments 0 0; fi
  if _want seq;    then _run_stage seq    run_experiments 1 0; fi
  if _want fold;   then _run_stage fold   run_experiments 0 1; fi
fi

if _want metrics; then
  _run_stage metrics phase_metrics
fi
if _want report; then
  _run_stage report phase_report
fi

status_table

ELAPSED=$(( $(date +%s) - START ))
hdr "完成，用时 $((ELAPSED/3600))h$(((ELAPSED%3600)/60))m"
ok "论文对照报告 : $REPORTS_DIR/paper_experiments_report.md"
ok "逐条件统计   : $METRICS_DIR/summary_by_condition.csv"
ok "逐设计明细   : $METRICS_DIR/per_design.csv"
ok "实验状态标记 : $STAGE_DIR/（RESUME=1 时用来跳过已完成阶段）"
echo
log "报告里第一张表就是「论文口径指标 | 论文报告数字 | 本次复现」三列对照。"
log "想续跑被中断的流程：加 RESUME=1 重跑同一命令即可。"
