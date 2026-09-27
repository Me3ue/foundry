#!/usr/bin/env bash
# 由 04_training_subset.py 生成：把 RFD3 微调指向这个子集。
# 用法：  source env.subset.sh  然后照常跑 run_nmf_zkp_pdb_sweep.sh
export PARQUET="/home/zzj/protein/foundry/experiments/paper_rfd3/out/train_subset_smoke/metadata"
export PDB_MIRROR="/home/zzj/protein/foundry/experiments/paper_rfd3/out/train_subset_smoke/mirror"
export DATA="$PDB_MIRROR"
export N_EXAMPLES="${N_EXAMPLES:-128}"
# CCD 与权重不在这里，仍需单独准备（CCD 约 1.7 GB、ckpt 约 2.7 GB）
export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-/media/zzj/Data/ccd_mirror}"
export CCD_PATH="$CCD_MIRROR_PATH"
