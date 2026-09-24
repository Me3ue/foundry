#!/usr/bin/env bash
# =============================================================================
# 实验 9 —— 论文 §4 EXPERIMENTS 的计算部分
#              （Fig. 4 / Fig. S7 DNA 结合物；Fig. S8 半胱氨酸水解酶）
#
# 论文里的湿实验（酵母表面展示测 EC50、IVTT 筛选、Michaelis-Menten 动力学）
# 无法用脚本复现，这里只做它们在计算侧的全部步骤：
#
#   (a) DNA 结合物两阶段流程（Fig. 4a / Fig. S7a）
#       第 1 步：针对一个随机生成的靶 DNA 序列的 AF3 预测结构采样设计
#       第 2 步：把与 DNA 接触的 motif 固定，重采样其余骨架以提高多样性
#       论文最终送测 5 个设计，其中 DBRFD3 结合 EC50 = 5.89 ± 2.15 µM
#       靶序列：CGAGAACATAGTCG
#
#   (b) 半胱氨酸水解酶（Fig. 4c-d / Fig. S8）
#       motif：天然半胱氨酸水解酶 Ulp-1（PDB 1EUV）的 Cys-His-Asp 三联体
#              + Gln + 半胱氨酸两侧残基的骨架原子 + 处于第一个四面体中间体
#              几何的底物（4-methylumbelliferyl phenyl acetate, 4MU-PhAc）
#       论文筛了 190 个设计，35 个多轮转化，最好 C6 的 kcat/Km = 3557
#
# ⚠️ 这两部分都需要"论文专用的 motif 文件"，随论文补充材料发布。
#    脚本会在缺少文件时给出明确提示，并用占位文件保证流程可跑通。
#
# 运行：
#   ./18_exp9_wetlab_insilico.sh
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./lib.sh

DEST="$DESIGNS_DIR/exp9_wetlab_insilico"
MOTIF_DIR="$INPUTS_DIR/motifs"
mkdir -p "$MOTIF_DIR"

# ------------------------------------------------- (a) DNA 结合物两阶段 ---
hdr "实验 9a：DNA 结合物两阶段流程（Fig. 4a / Fig. S7a）"
TARGET_SEQ="${DBRFD3_TARGET_SEQ:-CGAGAACATAGTCG}"

if ! "$PY" - "$MOTIF_DIR" "$TARGET_SEQ" "$INPUTS_DIR/specs" <<'PY'
import json, sys, pathlib
root = pathlib.Path.cwd(); sys.path.insert(0, str(root))
from lib import common
motif_dir, target_seq, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])

stage1_pdb = motif_dir / "dbrfd3_stage1_target.pdb"
if not stage1_pdb.exists():
    print(f"[需要人工准备] {stage1_pdb}")
    print("  论文做法：先用 AF3 预测随机靶 DNA 序列的构象，把这个预测结构当作输入。")
    print(f"  目标序列（论文验证的 DBRFD3 靶点）：{target_seq}")
    print("  如果你有 AF3/Chai 环境，可以直接折叠下面这条 DNA 序列：")
    print(f"    {target_seq}")
    # 退化为仓库自带的 1bna，保证后续命令格式可跑
    fallback = common.RFD3_INPUT_PDBS / "1bna.pdb"
    stage1_pdb.write_bytes(fallback.read_bytes())
    print(f"  已用 {fallback} 作为占位输入。")

dna = {
    "name": "dbrfd3",
    "input": str(stage1_pdb),
    "contig": "A1-12,/0,110-130",
    "length": "110-130",
    "dna_sel": "A1-12",
    "ori_token": [-8, -8, 8],
}
stage1 = common.spec_dna([dna], diffused=False)
common.write_spec(out / "exp9_dbrfd3_stage1.json", stage1)
stage2 = common.spec_dna([dna], diffused=True)
common.write_spec(out / "exp9_dbrfd3_stage2.json", stage2)
print("stage1（固定 DNA 采样）与 stage2（固定 motif 重采样）规格已写好")
PY
then
  warn "9a 规格生成失败"
fi

for stage in stage1 stage2; do
  spec="$INPUTS_DIR/specs/exp9_dbrfd3_${stage}.json"
  [[ -f "$spec" ]] || continue
  ( time rfd3_design "$DEST/dbrfd3/$stage" "$spec" "$N_BACKBONES" ) 2>&1 \
      | tee "$LOGS_DIR/exp9_dbrfd3_${stage}.log" || warn "$stage 运行异常"
  record_run exp9_wetlab_insilico "dbrfd3_${stage}" "$N_BACKBONES"
done

# ------------------------------------------- (b) 半胱氨酸水解酶 motif ---
hdr "实验 9b：半胱氨酸水解酶（Fig. 4c-d / Fig. S8）"
CYS_MOTIF="$MOTIF_DIR/cysteine_hydrolase_C6.pdb"
if [[ ! -f "$CYS_MOTIF" ]]; then
  warn "缺少 motif 文件：$CYS_MOTIF"
  warn "论文用的是 Ulp-1（PDB 1EUV）的 Cys-His-Asp 三联体 + Gln + 底物 TI1 几何。"
  warn "准备步骤（在你本地跑一次即可）："
  warn "  1) 下载 1EUV 的组氨酸/半胱氨酸活性位点残基"
  warn "  2) 用 4-methylumbelliferyl phenyl acetate 建模第一个四面体中间体 (TI1)"
  warn "  3) 合并成一个 pdb，催化残基用 unindex，底物作为 ligand"
  warn "参考格式见 models/rfd3/docs/examples/enzyme_design.json"
  warn "跳过 9b 的采样步骤。"
else
  "$PY" - "$CYS_MOTIF" "$INPUTS_DIR/specs" <<'PY'
import json, sys, pathlib
root = pathlib.Path.cwd(); sys.path.insert(0, str(root))
from lib import common
motif, out = sys.argv[1], pathlib.Path(sys.argv[2])
case = {
    "name": "CYS_HYDROLASE_C6",
    "input": motif,
    "ligand": "LIG",                  # 换成你建模底物时用的残基名
    "unindex": "A1,A2,A3,A4",         # Cys / His / Asp / Gln 的残基号
    "length": "150-200",
    "fixed_atoms": {
        "A1": "SG,CB", "A2": "NE2,CD2,CE1", "A3": "OD1,OD2,CG",
        "A4": "OE1,NE2,CD", "LIG": "C1,O1,O2",
    },
}
common.write_spec(out / "exp9_cys_hydrolase.json", common.spec_enzyme([case]))
print("半胱氨酸水解酶规格已写好")
PY
  ( time rfd3_design "$DEST/cys_hydrolase/seed_0" \
      "$INPUTS_DIR/specs/exp9_cys_hydrolase.json" "${N_DESIGNS_CYS:-190}" ) 2>&1 \
      | tee "$LOGS_DIR/exp9_cys_hydrolase.log" || warn "运行异常"
  record_run exp9_wetlab_insilico "cys_hydrolase" "${N_DESIGNS_CYS:-190}"
fi

if [[ "${WITH_SEQ:-1}" == "1" ]]; then
  hdr "实验 9：序列设计"
  "$PY" 50_sequence_design.py --experiment exp9_wetlab_insilico \
        --model ligand_mpnn --n-seqs "$LIGAND_MPNN_SEQS" || warn "序列设计失败"
fi
if [[ "${WITH_FOLD:-1}" == "1" ]]; then
  hdr "实验 9：结构预测（motif 全原子 RMSD 的粗筛）"
  "$PY" 60_fold.py --experiment exp9_wetlab_insilico || warn "折叠失败"
fi

hdr "实验 9 完成"
cat <<'EOT' >&2

  湿实验部分（不在本仓库范围内，附论文口径供对照）：
    * DNA 结合物 —— 合成基因 → 酵母表面展示 → 流式滴定
                     DBRFD3: EC50 = 5.89 ± 2.15 µM (n=3, 无 avidity)
    * 半胱氨酸水解酶 —— 190 个设计做 IVTT 筛选（4MU-PhAc 底物）
                     35 个多轮转化设计；最优 C6: kcat/Km = 3557
EOT
