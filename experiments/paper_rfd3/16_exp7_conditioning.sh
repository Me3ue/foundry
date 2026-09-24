#!/usr/bin/env bash
# =============================================================================
# 实验 7 —— 原子级条件控制（论文 Fig. 2d / 2e / 2f）
#
# 三个独立的对照实验，验证 RFD3 的原子级条件是否"听话"：
#
#   (a) Fig. 2d 氢键供体/受体条件
#       给配体或核酸的指定原子标注 hbond donor / acceptor，
#       统计生成结构里真正形成氢键的比例。
#       论文数字（小分子）：26.67%（无条件）→ 32.67%（+hbond 条件）
#                            → 36.67%（再叠加 classifier-free guidance）
#       论文数字（DNA）：11% → 11.3% → 12.5%
#       ⚠️ 需要安装 HBPLUS 并把路径写进 .env 的 HBPLUS_PATH
#
#   (b) Fig. 2e RASA 埋藏条件
#       指定配体原子应埋藏 / 应暴露，统计 400+ 设计的实际 RASA 分布
#
#   (c) Fig. 2f 质心条件
#       用 ori_token 指定生成蛋白相对靶点的质心位置，
#       两组不同质心应聚成两簇
#
# 多卡：8 + 3 + 2 = 13 个独立条件，一次性铺到 6 张 A6000 上。
#
# 运行：
#   ./16_exp7_conditioning.sh
#   ONLY=hbond ./16_exp7_conditioning.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

ONLY="${ONLY:-all}"
DEST="$DESIGNS_DIR/exp7_conditioning"
JOBS="$LOGS_DIR/exp7_jobs.txt"
: > "$JOBS"

DNA_SPEC="$INPUTS_DIR/targets/dna_targets.json"
SM_SPEC="$INPUTS_DIR/targets/sm_ligands.json"
[[ -f "$DNA_SPEC" ]] || die "缺少 $DNA_SPEC，请先运行 python 01_prepare_inputs.py"
[[ -f "$SM_SPEC" ]] || die "缺少 $SM_SPEC，请先运行 python 01_prepare_inputs.py"

# 往任务清单里追加一条，并顺手登记
add_job() {
  local out_dir="$1" spec="$2" condition="$3" note="$4"; shift 4
  rfd3_design_cmd "$out_dir" "$spec" "$N_BACKBONES" "$@" >> "$JOBS"
  printf '\n' >> "$JOBS"
  record_run exp7_conditioning "$condition" "$N_BACKBONES" "$note"
}

# ------------------------------------------------------------- (a) 氢键条件 ---
if [[ "$ONLY" == "all" || "$ONLY" == "hbond" ]]; then
  hdr "实验 7a：氢键供体/受体条件（Fig. 2d）"
  if [[ ! -x "$HBPLUS" ]]; then
    warn "未找到 HBPLUS（$HBPLUS），氢键条件无法启用。"
    warn "安装见 https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/ ，"
    warn "然后在 foundry/.env 里设置 HBPLUS_PATH。"
  else
    "$PY" - "$DNA_SPEC" "$SM_SPEC" "$INPUTS_DIR/specs" <<'PY'
import json, sys, pathlib
dna, sm, out = (pathlib.Path(p) for p in sys.argv[1:4])
dna_targets = json.loads(dna.read_text())
ligands = json.loads(sm.read_text())
sys.path.insert(0, str(pathlib.Path.cwd()))
from lib import common

if dna_targets:
    t = dna_targets[0]
    acc = {t["dna_sel"]: "OP1,OP2,O3',O5'"}   # 磷酸骨架作受体
    spec = {}
    spec.update(common.spec_hbond(t, {}, {}, f"{t['name']}_nohbond"))
    spec.update(common.spec_hbond(t, acc, {}, f"{t['name']}_hbond"))
    common.write_spec(out / "exp7_hbond_dna.json", spec)

if ligands:
    lig = ligands[0]
    atoms, half = lig["atoms"], max(1, len(lig["atoms"]) // 2)
    base = {"input": lig["input"], "ligand": lig["ligand"], "length": lig["length"]}
    spec = {
        f"{lig['name']}_nohbond": {**base,
                                   "select_fixed_atoms": {lig["ligand"]: ""}},
        f"{lig['name']}_hbond": {**base,
                                 "select_fixed_atoms": {lig["ligand"]: ""},
                                 "select_hbond_acceptor": {lig["ligand"]: ",".join(atoms[:half])},
                                 "select_hbond_donor": {lig["ligand"]: ",".join(atoms[half:])}},
    }
    common.write_spec(out / "exp7_hbond_sm.json", spec)
PY

    for cfg in "dna:exp7_hbond_dna.json" "sm:exp7_hbond_sm.json"; do
      name="${cfg%%:*}"; spec="$INPUTS_DIR/specs/${cfg##*:}"
      for cond in nohbond hbond; do
        sub="$INPUTS_DIR/specs/exp7_${name}_${cond}.json"
        "$PY" - "$spec" "$sub" "$cond" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1])); keep = "_" + sys.argv[3]
json.dump({k: v for k, v in spec.items() if k.endswith(keep)},
          open(sys.argv[2], "w"), indent=2)
PY
        add_job "$DEST/hbond/${name}_${cond}_plain/seed_0" "$sub" \
                "hbond_${name}_${cond}_plain" "${name}/${cond}/无CFG"
        add_job "$DEST/hbond/${name}_${cond}_cfg/seed_0" "$sub" \
                "hbond_${name}_${cond}_cfg" "${name}/${cond}/CFG=2" \
                "inference_sampler.use_classifier_free_guidance=True" \
                "inference_sampler.cfg_scale=$CFG_SCALE"
      done
    done
  fi
fi

# -------------------------------------------------------------- (b) RASA ---
if [[ "$ONLY" == "all" || "$ONLY" == "rasa" ]]; then
  hdr "实验 7b：RASA 埋藏条件（Fig. 2e）"
  "$PY" - "$SM_SPEC" "$INPUTS_DIR/specs" <<'PY'
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
from lib import common
ligands = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
if not ligands:
    print("没有配体，跳过"); raise SystemExit(0)
lig = ligands[0]; atoms = lig["atoms"]; half = max(1, len(atoms) // 2)
spec = {}
spec.update(common.spec_rasa_grade(lig, atoms, [], f"{lig['name']}_all_buried"))
spec.update(common.spec_rasa_grade(lig, atoms[:half], atoms[half:], f"{lig['name']}_half"))
spec.update(common.spec_rasa_grade(lig, [], atoms, f"{lig['name']}_all_exposed"))
common.write_spec(out / "exp7_rasa.json", spec)
PY
  for cond in all_buried half all_exposed; do
    sub="$INPUTS_DIR/specs/exp7_rasa_${cond}.json"
    "$PY" - "$INPUTS_DIR/specs/exp7_rasa.json" "$sub" "$cond" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
json.dump({k: v for k, v in spec.items() if k.endswith("_" + sys.argv[3])},
          open(sys.argv[2], "w"), indent=2)
PY
    add_job "$DEST/rasa/${cond}/seed_0" "$sub" "rasa_${cond}" "RASA=${cond}" \
            "inference_sampler.use_classifier_free_guidance=True" \
            "inference_sampler.cfg_scale=$CFG_SCALE"
  done
fi

# ---------------------------------------------------------------- (c) 质心 ---
if [[ "$ONLY" == "all" || "$ONLY" == "com" ]]; then
  hdr "实验 7c：质心条件（Fig. 2f）"
  "$PY" - "$DNA_SPEC" "$INPUTS_DIR/specs" <<'PY'
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
from lib import common
targets = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
if not targets:
    print("没有 DNA 靶点，跳过"); raise SystemExit(0)
t = targets[0]
spec = {}
spec.update(common.spec_center_of_mass(t, [-8, -8, 8], f"{t['name']}_comA"))
spec.update(common.spec_center_of_mass(t, [8, 8, -8], f"{t['name']}_comB"))
common.write_spec(out / "exp7_com.json", spec)
PY
  for cond in comA comB; do
    sub="$INPUTS_DIR/specs/exp7_com_${cond}.json"
    "$PY" - "$INPUTS_DIR/specs/exp7_com.json" "$sub" "$cond" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
json.dump({k: v for k, v in spec.items() if k.endswith("_" + sys.argv[3])},
          open(sys.argv[2], "w"), indent=2)
PY
    add_job "$DEST/com/${cond}/seed_0" "$sub" "com_${cond}" "质心=${cond}"
  done
fi

hdr "实验 7：提交 $(grep -cvE '^\s*(#|$)' "$JOBS") 个条件到 GPU 池"
run_on_gpus "$MAX_PARALLEL_GPUS" < "$JOBS" \
  || warn "部分条件失败（日志见 $LOGS_DIR/gpu_pool/）"

hdr "实验 7 完成"
ok "设计结果: $DEST"
ok "氢键比例 / RASA 分布 / 质心聚簇 由 70_metrics_geometry.py 统计"
