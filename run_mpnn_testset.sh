#!/usr/bin/env bash
set -euo pipefail

preprocess_structure() {
  local input_path="$1"
  local output_root="outputs/mpnn_testset/_preprocessed"
  mkdir -p "$output_root"

  local rel_name
  rel_name="${input_path//\//__}"
  local output_path="$output_root/$rel_name"

  python - "$input_path" "$output_path" <<'PY'
from pathlib import Path
import sys

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
lines = input_path.read_text().splitlines()

# Keep the first complete occurrence of each residue identifier. The input
# files in this test set contain repeated assembly copies as contiguous PDB
# blocks; removing later residue blocks preserves every atom of the selected
# copy instead of merging atoms from different copies.
residue_keys = set()
current_key = None
keep_current = True
has_duplicate_residue = False
out = []
for line in lines:
    if line.startswith(("ATOM  ", "HETATM")):
        key = (line[21], line[22:26], line[26], line[17:20])
        if key != current_key:
            current_key = key
            if key in residue_keys:
                keep_current = False
                has_duplicate_residue = True
            else:
                residue_keys.add(key)
                keep_current = True
        if keep_current:
            out.append(line)
    else:
        # Preserve headers and metadata. TER records are retained only when
        # they belong to the retained stream.
        if line.startswith("TER"):
            if keep_current:
                out.append(line)
        else:
            out.append(line)

if not has_duplicate_residue:
    output_path.write_text(input_path.read_text())
else:
    output_path.write_text("\n".join(out) + "\n")
PY

  printf '%s\n' "$output_path"
}

run_mpnn() {
  local args=()
  local structure_path=""
  while (($#)); do
    case "$1" in
      --structure_path)
        structure_path="$2"
        args+=("--structure_path")
        args+=("$(preprocess_structure "$structure_path")")
        shift 2
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  mpnn "${args[@]}"
}

run_run_mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/BSD_ASPTE_1_130_0/2z3h_A_rec.pdb \
  --fixed_residues \
  '["A47", "A56", "A82", "A26", "A25", "B126", "A55", "A88", "A86", "A28", "A91", "A54"]' \
  --out_directory \
  outputs/mpnn_testset/test_0 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_0

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/GLMU_STRPN_2_459_0/4aaw_A_rec.pdb \
  --fixed_residues \
  '["A401", "A452", "A453", "A384", "A422", "A409", "A419", "A386", "A387", "A404", "A439"]' \
  --out_directory \
  outputs/mpnn_testset/test_1 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_1

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/GRK4_HUMAN_1_578_0/4yhj_A_rec.pdb \
  --fixed_residues \
  '["A197", "A267", "A265", "A330", "A193"]' \
  --out_directory \
  outputs/mpnn_testset/test_2 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_2

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/GSTP1_HUMAN_2_210_0/14gs_A_rec.pdb \
  --fixed_residues \
  '["A8", "A108", "A104", "A7", "A10", "A13"]' \
  --out_directory \
  outputs/mpnn_testset/test_3 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_3

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/GUX1_HYPJE_18_451_0/2v3r_A_rec.pdb \
  --fixed_residues \
  '["A376", "A367", "A212", "A371", "A145", "A175", "A372", "A228", "A214", "A217"]' \
  --out_directory \
  outputs/mpnn_testset/test_4 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_4

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/HDAC8_HUMAN_1_377_0/4rn0_B_rec.pdb \
  --fixed_residues \
  '["B152", "B208", "B306", "B33", "B274"]' \
  --out_directory \
  outputs/mpnn_testset/test_5 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_5

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/HDHA_ECOLI_1_255_0/1fmc_B_rec.pdb \
  --fixed_residues \
  '["B159", "B99", "B196", "B200", "B151", "B254", "B146", "B101"]' \
  --out_directory \
  outputs/mpnn_testset/test_6 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_6

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/HMD_METJA_1_358_0/3daf_A_rec.pdb \
  --fixed_residues \
  '["A64", "A148", "A14", "A13", "A135", "A10", "A9", "A176", "A63"]' \
  --out_directory \
  outputs/mpnn_testset/test_7 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_7

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CCPR_YEAST_69_361_0/1a2g_A_rec.pdb \
  --fixed_residues \
  '["A191", "A175", "A185", "A187", "A181"]' \
  --out_directory \
  outputs/mpnn_testset/test_8 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_8

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/IPMK_HUMAN_49_416_0/5w2g_A_rec.pdb \
  --fixed_residues \
  '["A133", "A131", "A75", "A385", "A144"]' \
  --out_directory \
  outputs/mpnn_testset/test_9 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_9

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CD38_HUMAN_44_300_0/3dzh_A_rec.pdb \
  --fixed_residues \
  '["A189", "A221", "A222", "A146", "A125", "A186", "A127"]' \
  --out_directory \
  outputs/mpnn_testset/test_10 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_10

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/KS6A3_HUMAN_41_357_0/3g51_A_rec.pdb \
  --fixed_residues \
  '["A216", "A79", "A150", "A78", "A149", "A100", "A198", "A195", "A148", "A210", "A197"]' \
  --out_directory \
  outputs/mpnn_testset/test_11 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_11

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CHOD_BREST_46_552_0/1coy_A_rec.pdb \
  --fixed_residues \
  '["A446", "A344", "A76", "A120", "A218"]' \
  --out_directory \
  outputs/mpnn_testset/test_12 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_12

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/LAT_MYCTU_1_449_0/2jjg_A_rec.pdb \
  --fixed_residues \
  '["A167", "A273", "A129", "A330", "A274", "A300", "A128"]' \
  --out_directory \
  outputs/mpnn_testset/test_13 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_13

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/LMBL1_HUMAN_198_526_0/2rhy_A_rec.pdb \
  --fixed_residues \
  '["A386", "A382", "A355", "A361"]' \
  --out_directory \
  outputs/mpnn_testset/test_14 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_14

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/LMBL1_HUMAN_198_526_0/2pqw_A_rec.pdb \
  --fixed_residues \
  '["A382", "A386", "B21", "A355", "B19", "A361"]' \
  --out_directory \
  outputs/mpnn_testset/test_15 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_15

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/M3K14_HUMAN_321_678_0/4g3d_B_rec.pdb \
  --fixed_residues \
  '["B533", "B414", "B469", "B522", "B479", "B535", "B470", "B429", "B440"]' \
  --out_directory \
  outputs/mpnn_testset/test_16 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_16

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/MENE_BACSU_2_486_0/5bur_A_rec.pdb \
  --fixed_residues \
  '["A286", "A153", "A289", "A285", "A471", "A382", "A367"]' \
  --out_directory \
  outputs/mpnn_testset/test_17 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_17

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NAGZ_VIBCH_1_330_0/3gs6_A_rec.pdb \
  --fixed_residues \
  '["A246", "A161", "A130", "A160", "A62", "A204"]' \
  --out_directory \
  outputs/mpnn_testset/test_18 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_18

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NEP_HUMAN_54_750_0/1r1h_A_rec.pdb \
  --fixed_residues \
  '["A693", "A689", "A563", "A580", "A692", "A711", "A579", "A583", "A110", "A542", "A543", "A106", "A558", "A646", "A717", "A102"]' \
  --out_directory \
  outputs/mpnn_testset/test_19 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_19

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NQO1_HUMAN_2_274_0/1dxo_C_rec.pdb \
  --fixed_residues \
  '["C126", "C68", "C128"]' \
  --out_directory \
  outputs/mpnn_testset/test_20 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_20

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NQO1_HUMAN_2_274_0/1gg5_A_rec.pdb \
  --fixed_residues \
  '["A128", "A232", "A236", "C154", "C161", "A131", "A178", "C105", "A126"]' \
  --out_directory \
  outputs/mpnn_testset/test_21 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_21

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NR1H4_HUMAN_258_486_0/5q0k_A_rec.pdb \
  --fixed_residues \
  '["A473", "A288", "A340", "A373", "A352", "A339", "A298", "A356", "A291", "A458", "A469", "A294", "A295", "A344", "A269", "A361"]' \
  --out_directory \
  outputs/mpnn_testset/test_22 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_22

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/OLIAC_CANSA_1_101_0/5b08_A_rec.pdb \
  --fixed_residues \
  '["A89", "A78", "B49", "A81", "A27", "A23", "A86", "A7", "A5", "A72", "A73", "A92"]' \
  --out_directory \
  outputs/mpnn_testset/test_23 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_23

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PA21B_PIG_23_146_0/2azy_A_rec.pdb \
  --fixed_residues \
  '["A111", "A25", "A41", "A20", "A29", "A21", "A23"]' \
  --out_directory \
  outputs/mpnn_testset/test_24 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_24

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PAK4_HUMAN_291_591_ATP_0/5i0b_A_rec.pdb \
  --fixed_residues \
  '["A447", "A335", "A348", "A397", "A398", "A458", "A444"]' \
  --out_directory \
  outputs/mpnn_testset/test_25 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_25

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PHKG1_RABIT_6_296_ATPsite_0/1phk_A_rec.pdb \
  --fixed_residues \
  '["A103", "A106", "A31", "A48", "A110", "A104", "A151", "A167", "A149", "A153"]' \
  --out_directory \
  outputs/mpnn_testset/test_26 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_26

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PHP_SULSO_1_314_0/4keu_A_rec.pdb \
  --fixed_residues \
  '["A223", "A170", "A97"]' \
  --out_directory \
  outputs/mpnn_testset/test_27 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_27

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/COTA_BACSU_1_513_0/4q8b_B_rec.pdb \
  --fixed_residues \
  '["B386", "B384", "B226"]' \
  --out_directory \
  outputs/mpnn_testset/test_28 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_28

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PLCD1_RAT_134_756_0/1djy_A_rec.pdb \
  --fixed_residues \
  '["A549", "A522", "A438", "A551", "A341", "A390"]' \
  --out_directory \
  outputs/mpnn_testset/test_29 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_29

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PNTM_STRAE_2_398_0/5l1v_A_rec.pdb \
  --fixed_residues \
  '["A283", "A240", "A77", "A74", "A232"]' \
  --out_directory \
  outputs/mpnn_testset/test_30 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_30

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CPXB_BACMB_2_464_0/4zfa_A_rec.pdb \
  --fixed_residues \
  '["A87", "A51", "A437", "A330", "A74", "A75", "A78", "A264", "A82", "A188", "A26", "A263", "A354"]' \
  --out_directory \
  outputs/mpnn_testset/test_31 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_31

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PPIA_HUMAN_1_165_0/2rma_A_rec.pdb \
  --fixed_residues \
  '["A103", "A101", "A55", "A102", "A63", "A107"]' \
  --out_directory \
  outputs/mpnn_testset/test_32 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_32

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/QPCT_HUMAN_33_361_0/2zen_A_rec.pdb \
  --fixed_residues \
  '["A207", "A329", "A325", "A330", "A201", "A248", "A159", "A202"]' \
  --out_directory \
  outputs/mpnn_testset/test_34 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_34

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/RIBB_VIBCH_2_218_0/4p6p_A_rec.pdb \
  --fixed_residues \
  '["B137", "A94", "A154", "A38", "A155", "A151", "A39", "A43", "A175"]' \
  --out_directory \
  outputs/mpnn_testset/test_35 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_35

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/RG1_RAUSE_1_513_0/3u5y_B_rec.pdb \
  --fixed_residues \
  '["B392", "B193", "B200", "B469", "B188", "B476", "B274", "B36", "B477", "B199", "B186"]' \
  --out_directory \
  outputs/mpnn_testset/test_36 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_36

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/ROCO4_DICDI_1009_1292_0/4f1m_A_rec.pdb \
  --fixed_residues \
  '["A1161", "A1156", "A1107", "A1108", "A1106"]' \
  --out_directory \
  outputs/mpnn_testset/test_37 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_37

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/DPO4_SULSO_1_347_0/4tqr_A_rec.pdb \
  --fixed_residues \
  '["A185", "A186"]' \
  --out_directory \
  outputs/mpnn_testset/test_38 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_38

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/SDIA_ECOLI_1_171_0/4lfu_A_rec.pdb \
  --fixed_residues \
  '["A71", "A63", "A59", "A43", "A68", "A80"]' \
  --out_directory \
  outputs/mpnn_testset/test_39 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_39

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/DPP2_HUMAN_27_492_0/3jyh_A_rec.pdb \
  --fixed_residues \
  '["A347", "A337", "A78", "A334", "A336"]' \
  --out_directory \
  outputs/mpnn_testset/test_40 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_40

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/TBK1_HUMAN_1_303_0/4iwq_A_rec.pdb \
  --fixed_residues \
  '["A23", "A36", "A142", "A86", "A96", "A15", "A68", "A89", "A87"]' \
  --out_directory \
  outputs/mpnn_testset/test_41 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_41

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/TRAR_RHIRD_1_234_0/1l3l_A_rec.pdb \
  --fixed_residues \
  '["A53", "A61", "A40", "A129", "A72", "A57", "A70"]' \
  --out_directory \
  outputs/mpnn_testset/test_42 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_42

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/UBE2T_HUMAN_1_156_0/5ngz_A_rec.pdb \
  --fixed_residues \
  '["A146", "A73", "A70", "A71", "A74"]' \
  --out_directory \
  outputs/mpnn_testset/test_43 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_43

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/VAOX_PENSI_1_560_0/1e8h_A_rec.pdb \
  --fixed_residues \
  '["A262", "A104", "A103", "A101", "A175", "A98", "A102", "A545"]' \
  --out_directory \
  outputs/mpnn_testset/test_44 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_44

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/XANLY_BACGL_26_777_0/2e24_A_rec.pdb \
  --fixed_residues \
  '["A313", "A315", "A149", "A255", "A309"]' \
  --out_directory \
  outputs/mpnn_testset/test_45 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_45

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/EFTU1_ECOLI_2_178_0/2hcj_B_rec.pdb \
  --fixed_residues \
  '["A21", "B175", "B174", "A24", "A25", "B136", "B173", "A22", "A23", "B138", "B135", "A26"]' \
  --out_directory \
  outputs/mpnn_testset/test_46 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_46

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/F16P1_HUMAN_1_338_0/3kc1_A_rec.pdb \
  --fixed_residues \
  '["A30", "A24", "A27", "A177", "A28", "A112", "A29", "A113"]' \
  --out_directory \
  outputs/mpnn_testset/test_47 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_47

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/FKB1A_HUMAN_2_108_0/1d7j_A_rec.pdb \
  --fixed_residues \
  '["A82", "A36", "A56", "A54", "A37"]' \
  --out_directory \
  outputs/mpnn_testset/test_48 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_48

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/IDHP_HUMAN_40_452_0/4ja8_B_rec.pdb \
  --fixed_residues \
  '["A164", "A298", "B319", "A319", "A315", "A311", "A297", "B315", "B298", "A160", "B311", "B164", "B160", "A294", "A316", "B316", "A306"]' \
  --out_directory \
  outputs/mpnn_testset/test_49 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_49

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/IMA1_HUMAN_68_497_0/4u5s_A_rec.pdb \
  --fixed_residues \
  '["A399", "A357", "A364"]' \
  --out_directory \
  outputs/mpnn_testset/test_50 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_50

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/MCCF_ECOLX_1_344_0/4iiy_A_rec.pdb \
  --fixed_residues \
  '["A186", "A220", "A221", "A92", "A119", "A247", "A91", "A118", "A277", "A246"]' \
  --out_directory \
  outputs/mpnn_testset/test_51 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_51

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/MURA_ECOLI_1_419_catalytic_0/3v4t_A_rec.pdb \
  --fixed_residues \
  '["A124", "A123", "A125", "A164", "A120", "A163", "A160", "A162", "A121"]' \
  --out_directory \
  outputs/mpnn_testset/test_52 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_52

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NOS1_HUMAN_302_723_0/3tym_A_rec.pdb \
  --fixed_residues \
  '["A706", "A565", "A567", "A570", "A584", "A678", "A588", "A587", "A569", "A592"]' \
  --out_directory \
  outputs/mpnn_testset/test_53 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_53

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NOS1_HUMAN_302_723_0/4d7o_A_rec.pdb \
  --fixed_residues \
  '["A678", "A567", "A565", "A706", "A584", "A587", "A592"]' \
  --out_directory \
  outputs/mpnn_testset/test_54 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_54

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NOS2_HUMAN_78_505_0/3ej8_A_rec.pdb \
  --fixed_residues \
  '["A377", "A372"]' \
  --out_directory \
  outputs/mpnn_testset/test_55 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_55

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NOS3_HUMAN_65_480_0/1rs9_A_rec.pdb \
  --fixed_residues \
  '["A355", "A363", "A358"]' \
  --out_directory \
  outputs/mpnn_testset/test_56 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_56

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NOS3_HUMAN_65_480_0/4kcq_A_rec.pdb \
  --fixed_residues \
  '["A477", "A449", "A338", "A355", "A359", "A336"]' \
  --out_directory \
  outputs/mpnn_testset/test_57 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_57

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/NPD_THEMA_1_246_0/3pdh_A_rec.pdb \
  --fixed_residues \
  '["A48", "A33", "A100", "A101", "A159", "A30"]' \
  --out_directory \
  outputs/mpnn_testset/test_58 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_58

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/ODBB_THET8_1_324_0/1umd_B_rec.pdb \
  --fixed_residues \
  '["B86", "C95", "C146", "C176", "C204", "C94", "B58", "C208", "C207", "C177", "C206", "C96", "C144"]' \
  --out_directory \
  outputs/mpnn_testset/test_59 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_59

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/P2Y12_HUMAN_1_342_0/4pxz_A_rec.pdb \
  --fixed_residues \
  '["A105", "A280", "A175", "A263", "A179", "A187", "A259", "A191", "A256", "A109", "A159", "A93", "A97"]' \
  --out_directory \
  outputs/mpnn_testset/test_60 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_60

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PA2B8_DABRR_1_121_0/2gns_A_rec.pdb \
  --fixed_residues \
  '["A2", "A3", "P2", "P3", "A69"]' \
  --out_directory \
  outputs/mpnn_testset/test_61 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_61

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PAC_ECOLX_27_846_0/1ai4_A_rec.pdb \
  --fixed_residues \
  '["B24", "A146", "A142", "B241", "B69", "B1", "B177", "B22", "B23"]' \
  --out_directory \
  outputs/mpnn_testset/test_62 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_62

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/POL_FOAMV_861_1060_0/5mma_A_rec.pdb \
  --fixed_residues \
  '["A214"]' \
  --out_directory \
  outputs/mpnn_testset/test_63 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_63

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/AROE_THET8_1_263_0/2cy0_A_rec.pdb \
  --fixed_residues \
  '["A14", "A16", "A85", "A60", "A64", "A58", "A100", "A235"]' \
  --out_directory \
  outputs/mpnn_testset/test_64 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_64

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PYRD_TRYCC_2_314_catalytic_0/3w83_B_rec.pdb \
  --fixed_residues \
  '["B67", "B194", "B43"]' \
  --out_directory \
  outputs/mpnn_testset/test_65 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_65

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PYRD_TRYCC_2_314_catalytic_0/2e6d_A_rec.pdb \
  --fixed_residues \
  '["A127", "A194", "A70", "A132", "A130", "A71", "A69", "A43"]' \
  --out_directory \
  outputs/mpnn_testset/test_66 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_66

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/PYRE_BACAN_1_210_0/4rv4_A_rec.pdb \
  --fixed_residues \
  '["A38", "A127", "A125", "A128", "B94", "A73", "A124", "A126", "A74", "A121", "A120"]' \
  --out_directory \
  outputs/mpnn_testset/test_67 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_67

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/SIR3_HUMAN_117_398_0/5d7n_D_rec.pdb \
  --fixed_residues \
  '["D146", "D180", "D230", "D294", "D231", "D248", "D158"]' \
  --out_directory \
  outputs/mpnn_testset/test_68 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_68

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/BAZ2A_HUMAN_1795_1898_0/5mgl_A_rec.pdb \
  --fixed_residues \
  '["A1872", "A1827", "A1879", "A1830", "A1873", "A1817"]' \
  --out_directory \
  outputs/mpnn_testset/test_69 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_69

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/SQHC_ALIAD_1_631_0/1h36_A_rec.pdb \
  --fixed_residues \
  '["A601", "A129", "A365", "A489", "A437", "A170", "A263", "A261", "A607", "A169"]' \
  --out_directory \
  outputs/mpnn_testset/test_70 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_70

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/TIAM1_HUMAN_840_931_0/4gvd_A_rec.pdb \
  --fixed_residues \
  '["A874", "A865", "D1"]' \
  --out_directory \
  outputs/mpnn_testset/test_71 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_71

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/TNKS1_HUMAN_1099_1319_0/4tos_A_rec.pdb \
  --fixed_residues \
  '["A1224", "A1201", "A1188", "A1184", "A1191", "A1228", "A1207", "A1197", "A1187", "A1198", "A1212", "A1213", "A1202", "A1192"]' \
  --out_directory \
  outputs/mpnn_testset/test_72 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_72

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/TNKS2_HUMAN_948_1162_0/5aeh_A_rec.pdb \
  --fixed_residues \
  '["A1048", "A1035", "A1060", "A1038", "A1075", "A1039", "A1050", "A1031", "A1045", "A1071", "A1049", "A1044"]' \
  --out_directory \
  outputs/mpnn_testset/test_73 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_73

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/UPPS_ECOLI_1_253_0/4h3c_A_rec.pdb \
  --fixed_residues \
  '["A141", "A100", "A47", "A69", "A89", "A50", "A93", "A51"]' \
  --out_directory \
  outputs/mpnn_testset/test_74 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_74

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/Y635_MYCTU_1_158_0/4rlu_A_rec.pdb \
  --fixed_residues \
  '["A65", "A142", "A61", "A84", "A125", "A138", "A86", "A89"]' \
  --out_directory \
  outputs/mpnn_testset/test_75 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_75

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/ABL2_HUMAN_274_551_0/4xli_B_rec.pdb \
  --fixed_residues \
  '["B359", "B336", "B363", "B315", "B345", "B364", "B416", "B361", "B428"]' \
  --out_directory \
  outputs/mpnn_testset/test_76 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_76

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/ACE_HUMAN_650_1230_0/3l3n_A_rec.pdb \
  --fixed_residues \
  '["A281", "A383", "A513", "A523", "A353", "A511", "A520"]' \
  --out_directory \
  outputs/mpnn_testset/test_77 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_77

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/BGAT_HUMAN_63_353_0/5tjn_A_rec.pdb \
  --fixed_residues \
  '["A233", "A300", "A245", "A303", "A235"]' \
  --out_directory \
  outputs/mpnn_testset/test_78 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_78

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/AK1BA_HUMAN_1_316_0/5liu_X_rec.pdb \
  --fixed_residues \
  '["X49", "X21", "X123", "X80", "X112", "X116", "X220", "X111", "X131", "X301", "X124"]' \
  --out_directory \
  outputs/mpnn_testset/test_79 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_79

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/AKT1_HUMAN_1_137_0/3o96_A_rec.pdb \
  --fixed_residues \
  '["A80", "A272", "A84", "A210", "A264", "A290", "A270", "A205", "A268"]' \
  --out_directory \
  outputs/mpnn_testset/test_80 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_80

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/BGL07_ORYSJ_25_504_0/4qlk_A_rec.pdb \
  --fixed_residues \
  '["A176", "A130", "A245", "A175", "A433", "A29", "A441", "A315", "A440", "A386", "A178"]' \
  --out_directory \
  outputs/mpnn_testset/test_81 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_81

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/ATS5_HUMAN_262_480_0/3hy9_B_rec.pdb \
  --fixed_residues \
  '["B410", "B406", "B443", "B442", "B414", "B379", "B380", "B411", "B440", "B420", "B441"]' \
  --out_directory \
  outputs/mpnn_testset/test_82 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_82

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/BACE2_HUMAN_76_460_0/4bel_A_rec.pdb \
  --fixed_residues \
  '["A211", "A124", "A87", "A131", "A134", "A246", "A126", "A50", "A245", "A46", "A241", "A48", "A243"]' \
  --out_directory \
  outputs/mpnn_testset/test_83 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_83

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/BAPA_SPHXN_30_402_0/3nfb_A_rec.pdb \
  --fixed_residues \
  '["C128", "C124", "A250", "A133", "A135", "C127", "A137", "A76", "A207", "A287"]' \
  --out_directory \
  outputs/mpnn_testset/test_84 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_84

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/BTRN_BACCI_2_250_0/4m7t_A_rec.pdb \
  --fixed_residues \
  '["A152", "A150", "A64", "A22", "A117", "A91", "A63", "A92", "A90", "A115"]' \
  --out_directory \
  outputs/mpnn_testset/test_85 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_85

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CAT_ECOLX_1_219_0/3u9f_C_rec.pdb \
  --fixed_residues \
  '["C166", "A25", "A31", "C158", "A193", "A29", "C170", "C160", "C144", "C133", "C102", "C146"]' \
  --out_directory \
  outputs/mpnn_testset/test_86 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_86

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CDK6_HUMAN_1_312_0/4aua_A_rec.pdb \
  --fixed_residues \
  '["A152", "A101", "A100", "A104"]' \
  --out_directory \
  outputs/mpnn_testset/test_87 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_87

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CDK6_HUMAN_1_312_0/2f2c_B_rec.pdb \
  --fixed_residues \
  '["B98", "B152", "B100", "B162", "B104", "B43", "B61", "B163", "B101", "B27"]' \
  --out_directory \
  outputs/mpnn_testset/test_88 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_88

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CHIB1_ASPFM_39_433_0/3chc_B_rec.pdb \
  --fixed_residues \
  '["B246", "B177", "B245", "B175", "B243"]' \
  --out_directory \
  outputs/mpnn_testset/test_89 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_89

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CHIB_SERMA_1_499_0/1h0i_A_rec.pdb \
  --fixed_residues \
  '["A97", "A215"]' \
  --out_directory \
  outputs/mpnn_testset/test_91 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_91

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CHIB_SERMA_1_499_0/4z2g_A_rec.pdb \
  --fixed_residues \
  '["A292", "A144", "A142", "A214", "A12", "A98", "A403", "A51", "A97"]' \
  --out_directory \
  outputs/mpnn_testset/test_92 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_92

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/COAA_MYCTU_1_312_0/3af2_A_rec.pdb \
  --fixed_residues \
  '["A179", "A247", "A254", "A100", "A238", "A101", "A104", "A102", "A105", "A103", "A108"]' \
  --out_directory \
  outputs/mpnn_testset/test_93 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_93

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/CONA_CANCT_1_237_0/1jn2_P_rec.pdb \
  --fixed_residues \
  '["P100", "P12", "P228", "P14", "P99", "P208"]' \
  --out_directory \
  outputs/mpnn_testset/test_94 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_94

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/DFPA_LOLVU_2_314_0/3li4_A_rec.pdb \
  --fixed_residues \
  '["A21"]' \
  --out_directory \
  outputs/mpnn_testset/test_95 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_95

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/DHAK_ECOLI_1_356_0/3pnm_A_rec.pdb \
  --fixed_residues \
  '["A53", "A80", "A218", "A78", "A109"]' \
  --out_directory \
  outputs/mpnn_testset/test_96 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_96

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/DIDH_RAT_1_319_0/1afs_A_rec.pdb \
  --fixed_residues \
  '["A227", "A310", "A129", "A54", "A55", "A117"]' \
  --out_directory \
  outputs/mpnn_testset/test_97 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_97

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/DYRK2_HUMAN_145_550_0/4azf_A_rec.pdb \
  --fixed_residues \
  '["A228", "A294", "A163", "A160", "A212", "A229", "A178"]' \
  --out_directory \
  outputs/mpnn_testset/test_98 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_98

mpnn \
  --model_type \
  protein_mpnn \
  --checkpoint_path \
  models/mpnn/proteinmpnn_v_48_020.pt \
  --is_legacy_weights \
  True \
  --structure_path \
  test_set/EXG1_CANAL_41_438_0/2pc8_A_rec.pdb \
  --fixed_residues \
  '["A192", "A363", "A191", "A29", "A292", "A135", "A255", "A27"]' \
  --out_directory \
  outputs/mpnn_testset/test_99 \
  --write_fasta \
  True \
  --write_structures \
  True \
  --name \
  test_99

