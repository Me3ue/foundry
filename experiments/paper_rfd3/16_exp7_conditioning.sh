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
DNA_SPEC="$INPUTS_DIR/targets/dna_targets.json"
SM_SPEC="$INPUTS_DIR/targets/sm_ligands.json"

[[ -f "$DNA_SPEC" ]] || die "缺少 $DNA_SPEC，请先运行 python 01_prepare_inputs.py"
[[ -f "$SM_SPEC" ]] || die "缺少 $SM_SPEC，请先运行 python 01_prepare_inputs.py"

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
root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
from lib import common

if dna_targets:
    t = dna_targets[0]
    # 骨架/磷酸作为受体，碱基 N6 作为供体 —— 论文里对 DNA 就是这么标注的
    acc = {t["dna_sel"]: "OP1,OP2,O3',O5'"}
    don = {}
    spec = {}
    spec.update(common.spec_hbond(t, {}, {}, f"{t['name']}_nohbond"))
    spec.update(common.spec_hbond(t, acc, don, f"{t['name']}_hbond"))
    common.write_spec(out / "exp7_hbond_dna.json", spec)

if ligands:
    lig = ligands[0]
    atoms = lig["atoms"]
    half = max(1, len(atoms) // 2)
    spec = {}
    base = {"input": lig["input"], "ligand": lig["ligand"], "length": lig["length"]}
    spec[f"{lig['name']}_nohbond"] = {**base, "select_fixed_atoms": {lig["ligand"]: ""}}
    spec[f"{lig['name']}_hbond"] = {
        **base,
        "select_fixed_atoms": {lig["ligand"]: ""},
        "select_hbond_acceptor": {lig["ligand"]: ",".join(atoms[:half])},
        "select_hbond_donor": {lig["ligand"]: ",".join(atoms[half:])},
    }
    common.write_spec(out / "exp7_hbond_sm.json", spec)
PY

    for cfg in "dna:exp7_hbond_dna.json" "sm:exp7_hbond_sm.json"; do
      name="${cfg%%:*}"; spec="$INPUTS_DIR/specs/${cfg##*:}"
      for cond in nohbond hbond; do
        sub="$INPUTS_DIR/specs/exp7_${name}_${cond}.json"
        "$PY" - "$spec" "$sub" "$cond" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1])); keep = sys.argv[3]
json.dump({k: v for k, v in spec.items() if k.endswith("_" + keep)}, open(sys.argv[2], "w"), indent=2)
PY
        for cfg2 in "plain:" "cfg:inference_sampler.use_classifier_free_guidance=True inference_sampler.cfg_scale=2.0"; do
          tag="${cfg2%%:*}"; extra="${cfg2#*:}"
          run_dir="$DEST/hbond/${name}_${cond}_${tag}/seed_0"
          log="$LOGS_DIR/exp7_hbond_${name}_${cond}_${tag}.log"
          log "氢键条件 ${name} / ${cond} / ${tag}"
          # shellcheck disable=SC2086
          ( time rfd3_design "$run_dir" "$sub" "$N_BACKBONES" $extra ) 2>&1 \
              | tee "$log" || warn "运行异常"
          record_run exp7_conditioning "hbond_${name}_${cond}_${tag}" "$N_BACKBONES"
        done
      done
    done
  fi
fi

# -------------------------------------------------------------- (b) RASA ---
if [[ "$ONLY" == "all" || "$ONLY" == "rasa" ]]; then
  hdr "实验 7b：RASA 埋藏条件（Fig. 2e）"
  "$PY" - "$SM_SPEC" "$INPUTS_DIR/specs" <<'PY'
import json, sys, pathlib
root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
from lib import common
ligands = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
if not ligands:
    print("没有配体，跳过"); raise SystemExit(0)
lig = ligands[0]
atoms = lig["atoms"]
half = max(1, len(atoms) // 2)
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
json.dump({k: v for k, v in spec.items() if k.endswith("_" + sys.argv[3])}, open(sys.argv[2], "w"), indent=2)
PY
    run_dir="$DEST/rasa/${cond}/seed_0"
    ( time rfd3_design "$run_dir" "$sub" "$N_BACKBONES" \
        inference_sampler.use_classifier_free_guidance=True \
        inference_sampler.cfg_scale="$CFG_SCALE" ) 2>&1 \
        | tee "$LOGS_DIR/exp7_rasa_${cond}.log" || warn "运行异常"
    record_run exp7_conditioning "rasa_${cond}" "$N_BACKBONES"
  done
fi

# ---------------------------------------------------------------- (c) 质心 ---
if [[ "$ONLY" == "all" || "$ONLY" == "com" ]]; then
  hdr "实验 7c：质心条件（Fig. 2f）"
  "$PY" - "$DNA_SPEC" "$INPUTS_DIR/specs" <<'PY'
import json, sys, pathlib
root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
from lib import common
targets = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
if not targets:
    print("没有 DNA 靶点，跳过"); raise SystemExit(0)
t = targets[0]
spec = {}
# 两组相反的质心初始化，论文里生成的蛋白会分别聚在两个位置
spec.update(common.spec_center_of_mass(t, [-8, -8, 8], f"{t['name']}_comA"))
spec.update(common.spec_center_of_mass(t, [8, 8, -8], f"{t['name']}_comB"))
common.write_spec(out / "exp7_com.json", spec)
PY
  for cond in comA comB; do
    sub="$INPUTS_DIR/specs/exp7_com_${cond}.json"
    "$PY" - "$INPUTS_DIR/specs/exp7_com.json" "$sub" "$cond" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
json.dump({k: v for k, v in spec.items() if k.endswith("_" + sys.argv[3])}, open(sys.argv[2], "w"), indent=2)
PY
    run_dir="$DEST/com/${cond}/seed_0"
    ( time rfd3_design "$run_dir" "$sub" "$N_BACKBONES" ) 2>&1 \
        | tee "$LOGS_DIR/exp7_com_${cond}.log" || warn "运行异常"
    record_run exp7_conditioning "com_${cond}" "$N_BACKBONES"
  done
fi

hdr "实验 7 完成"
ok "设计结果: $DEST"
ok "氢键比例 / RASA 分布 / 质心聚簇 由 70_metrics_geometry.py 统计"
