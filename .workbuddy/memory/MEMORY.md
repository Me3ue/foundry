# Foundry / RFdiffusion3 项目长期笔记

## 环境

| 项目 | 值 |
|---|---|
| 仓库根 | `/backup01/zzj/protein/foundry`（Foundry：atomworks + RFD3 + RF3 + MPNN） |
| 运行环境 | conda env `rc` → `/home/zhangzijian/anaconda3/envs/rc/bin/{rfd3,mpnn,rf3,rfd3na,foundry}` |
| RFD3 权重 | `/media/zzj/Data/rfd3_latest.ckpt` |
| HBPLUS | `/home/zhangzijian/protein/HBPLUS/hbplus/hbplus`（已写进 `foundry/.env` 的 `HBPLUS_PATH`） |
| TMalign | `/usr/bin/TMalign` |
| GPU | 单卡约 11.5 GiB；显存紧张，长跑前先 `SAMPLES=1` 试水 |

## 约定

- **CLI 风格**：`rfd3` / `rf3` 用 hydra 风格 `arg=value`；`mpnn` 是 argparse 风格
  `--arg value`。别混。
- **省显存三件套**：`low_memory_mode=True`、`diffusion_batch_size<=4`、
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。总样本数由 `n_batches` 控制，
  减小 batch 不改科学口径。
- **论文口径采样参数**（别再凭感觉调）：η=`step_scale`=1.5、γ0=`gamma_0`=0.6、
  `num_timesteps`=200；小分子 diffused-ligand 另加 CFG=2.0。
- 官方文档给 PPI 的生产推荐 `step_scale=3, gamma_0=0.2` 与论文 benchmark 口径不同。
- RFD3 输出：`<prefix>_<jsonkey>_<batch>_model_<n>.cif.gz`（含 `_denoised_` 版本）。
- RF3 的输出置信度是 AF3 风格键名：`ptm` / `iptm` / `chain_pair_pae_min` /
  `complex_plddt` / `ranking_score`。
- 沙箱（bash 工具）里看不到 `/media/zzj/*`、conda 环境，也无法安装 numpy
  （`No space left on device`）。涉及 GPU 的任务只能生成脚本交给用户跑；
  Python 改动做 `py_compile` 语法检查即可。

## 目录

- `experiments/paper_rfd3/` —— RFdiffusion3 论文正文全部实验的复现脚本与手册（2026-09-24 建）
- `docs/paper/rfd3_paper.txt` —— 论文抽取文本（22 页，含补充图 S1-S9；缺 Supplemental Methods）
- `models/rfd3/scripts/` —— 之前留下的近原生/NMF 相关脚本（`run_near_native_rfd3.py`、
  `run_rfd3_paper_matrix.sh`、`run_nmf_zkp_pdb_sweep.sh` 等），与论文复现是两条线
- `evaluation_manifest.csv`、`rfdiffusion3_inputs/`、`logs/rfd3_paper_matrix/` —— NMF/ZKP 线路的产物
