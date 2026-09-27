# Foundry / RFdiffusion3 项目长期笔记

## 环境

| 项目 | 值 |
|---|---|
| 仓库根 | `/backup01/zzj/protein/foundry`（Foundry：atomworks + RFD3 + RF3 + MPNN） |
| 本机（沙箱可见） | `zzj-laptop`：`/home/zzj/protein/foundry`；`/media/zzj/Data/*` 可读，**`/backup01` 与 `/dev/shm` 数据不可见** |
| 运行环境 | conda env `rc` → `/home/zhangzijian/anaconda3/envs/rc/bin/{rfd3,mpnn,rf3,rfd3na,foundry}`；服务器上是 `/backup01/zzj/rc-cu128/bin/python` |
| 权重与数据（服务器） | `/dev/shm/pdb_metadata_latest/`（`rfd3_latest.ckpt` + 两个 parquet）、`/dev/shm/pdb_mirror`（82 GB / 250,359 cif.gz）、`/dev/shm/ccd_mirror`（1.7 GB） |
| 权重（本机） | `/media/zzj/Data/{rfd3_latest.ckpt, pdb_mirror, pdb_metadata_latest, ccd_mirror}` |
| Path 探测 | `experiments/paper_rfd3/env.sh` 的 `_pick_dir`/`_pick_file` 自动选，**两边共用一份脚本**；加机器就加候选路径 |
| HBPLUS | `/home/zhangzijian/protein/HBPLUS/hbplus/hbplus`（已写进 `foundry/.env` 的 `HBPLUS_PATH`） |
| TMalign | `/usr/bin/TMalign` |
| GPU | **2026-09-27 起只有 1 张 RTX A6000 49 GB 可用（GPU 3）**；早前的 "11.5 GiB" 与 "6×A6000" 记录均已过时 |

## 约定

- **CLI 风格**：`rfd3` / `rf3` 用 hydra 风格 `arg=value`；`mpnn` 是 argparse 风格
  `--arg value`。别混。
- **显存调优（单卡正解）**：`experiments/paper_rfd3/05_probe_vram.sh` 递进实测峰值显存，
  结果写 `$OUT/vram_profile.json`，`env.sh` 自动读回。推理 batch 只影响吞吐、
  **不改采样分布**，可以放心顶到不 OOM；**训练 batch 改了就 baseline 与各变体必须同值**。
  单卡手册见 `experiments/paper_rfd3/SERVER_SINGLE_GPU.md`。
- **省显存三件套**（只在显存确实不够时用）：`low_memory_mode=True`、
  `diffusion_batch_size<=4`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  总样本数由 `n_batches` 控制，减小 batch 不改科学口径。49 GB 卡不需要降级。
- **`env.sh` / `lib.sh` 带 `set -Eeuo pipefail`，会被 `source` 继承** ——
  探测类脚本（要容忍单点 OOM）必须在 source 之后显式 `set +e`。
- **论文口径采样参数**（别再凭感觉调）：η=`step_scale`=1.5、γ0=`gamma_0`=0.6、
  `num_timesteps`=200；小分子 diffused-ligand 另加 CFG=2.0。
- 官方文档给 PPI 的生产推荐 `step_scale=3, gamma_0=0.2` 与论文 benchmark 口径不同。
- RFD3 输出：`<prefix>_<jsonkey>_<batch>_model_<n>.cif.gz`（含 `_denoised_` 版本）。
- RF3 的输出置信度是 AF3 风格键名：`ptm` / `iptm` / `chain_pair_pae_min` /
  `complex_plddt` / `ranking_score`。
- 沙箱（bash 工具）里看不到 `/media/zzj/*`、`/backup01/*`、conda 环境，也无法安装 numpy。
  涉及 GPU 的任务只能生成脚本交给用户跑；Python 改动做 `py_compile` + pyflakes 检查。
- **静态排查工具**：pyflakes 4.0.0 装在 `/home/zzj/.workbuddy/tmp/wb_tools`，
  用 `PYTHONPATH=/home/zzj/.workbuddy/tmp/wb_tools /home/zzj/.workbuddy/binaries/python/versions/3.13.12/bin/python3 -m pyflakes <files>`。
  沙箱 `/tmp` 每次调用重置、`~/.local` 不可写 → 装包必须 `pip install --target` 到持久目录。
- **本仓库历史习惯**：提交信息多为 `new`，且**出现过整行删掉函数定义/变量赋值的事故**
  （`_get_run_summary_path`、`crd_mask_L`）。遇 `NameError`/`undefined name` 先用
  `git show <旧提交>:<file>` 找回原实现，别当成设计问题。
- **零训练步验证评估链路**：`MAX_EPOCHS=1` 时（ckpt 载入后 `current_epoch=1>=1`）训练循环
  整体跳过、0 个 optimizer step，但仍会执行 fit 后的那次显式 holdout 验证。
- NMF sweep 的验证指标列名是 `val/pdb_holdout/lddt.mean_lddt_protein`
  （不是 `val/mean_lddt`）；权威源是 `<out>/val_metrics/validation_output_all_epochs.csv`。
- **GPU 是 6× A6000 48GB + 32 逻辑核**（早前“11.5 GiB”的记录已过时）。
  sweep 脚本用 `PRESET=a6000` / `a6000xl` 调档；**峰值显存 ∝ D · L²**，
  且真正生效的裁剪限制是 `max_atoms_in_crop`（1920 原子 ≈ 137 残基），验证阶段恒用 D=1。
- **GPU 运行环境的解释器是 `/backup01/zzj/rc-cu128/bin/python`**（不是 `(base)`，也不是
  `/home/zzj/anaconda3/envs/rc`）。在 `(base)` 里跑会报 `No module named 'pandas'`——那是
  环境选错，不是缺包。判别：`python -c "import pandas, torch, atomworks"` 能全过才对。
- **改 loss/指标前先跑** `python models/rfd3/scripts/check_loss_and_metric_shapes.py`（18 项，
  只要 torch，不用 GPU/数据集）；沙箱里的 CPU torch 在 `/home/zzj/.workbuddy/tmp/wb_torch`。
- 2026-09-25 修了子集 lDDT 归一化（含 DNA/RNA 的结构上 `mean_lddt_protein` 原来被低估
  约 n_subset/n_total 倍）→ **该日期之前的 lDDT 数值不能和新结果直接比**。
- **sweep 的 job 是单卡顺序执行的**，`JOBS` 顺序 = `baseline`（若 `INCLUDE_BASELINE=1`）
  → `encoder` 6 → `proj` 6 → `head` 2。所以中途 kill 时"已完成哪些 tag"由这个顺序决定：
  跑完 baseline 就杀掉 = `train/` 里只有 `baseline`。
- **拆卡跑的报告要合并**：`bash models/rfd3/scripts/merge_sweep_reports.sh <out> [--discover <root>] [--drop-incomplete] [--exclude 'glob,...'] <dir...>`。
  按 tag 选来源目录（完整版优先于半截版，打平取命令行靠前者）；只搬报告必需的小文件，
  跳过 GB 级 `val_structures/`；随后重跑 summarize + plot 生成 `paper_summary.*` /
  `paper_paired_deltas.csv` / `figures/`。**kill 过的轮次一定加 `--drop-incomplete`**。
- 三个 sweep 的 bash 命令行**完全相同**（`LAYER_SET` 等是环境变量，不进 argv），
  靠 `pgrep -f run_nmf_zkp_pdb_sweep.sh` 无法区分；只有子进程 python 的
  `paths.log_dir=<sweep dir>` 能唯一标识某一轮。
- **产物有两个粒度**：① 逐 job（`<tag>.run.log` 实时 tee、`<tag>.run_summary.json`
  =`<name>.step<N>.run_summary.json`、`val_metrics/validation_output_all_epochs.csv`、
  `<tag>.exit_code`）在该 job 结束就立刻有——所以 encoder 不用等 6 个全跑完；
  ② sweep 级汇总（`comparison_table.*`、`paper_summary.*`、`figures/`、
  `paper_training_cost.*`）只在 `JOBS` 循环**全部**结束后才写，中途 kill 就没有。
- **长跑必看 `/tmp`**：torch 默认 `file_descriptor` 共享策略要在 `TMPDIR/pymp-*/`
  里建 unix socket，而 `get_temp_dir()` 的目录只靠 atexit 清理；DataLoader worker
  每 epoch 被信号杀死 → **每 worker 每 epoch 泄漏一个目录**，小 tmpfs 的 `/tmp`
  几百 epoch 后必满，报 `OSError: [Errno 28] ... '/tmp/pymp-xxxx'`（在
  `queues._feed` 线程里），之后 batch 不再送达、主进程可能永久阻塞。
  对策：sweep 脚本已默认 `TMPDIR=${LOG_ROOT}/_worker_tmp`；长跑加
  `PERSISTENT_WORKERS=1`（真正堵住，worker 不再每 epoch 重生——`fabric.fit()` 只建
  一个 DataLoader，epoch 循环复用）；已经在跑的可救：
  `bash models/rfd3/scripts/cleanup_worker_tmp.sh --keep <NUM_WORKERS>` 先 dry-run。
  ⚠️ `/backup01/zzj/logs/...` 这类路径要留神：`save_checkpoints=false` 时**没有断点，
  重启=从头再来**，所以能救就别重启。
- **跑不完也能出论文级数据**：`python models/rfd3/scripts/report_partial_sweep.py
  --out <OUT_DIR> [--discover <LOG_ROOT>] [<sweep dir>...]`。就地读取各目录、按 tag
  选来源（有 validation CSV 的压过被 kill 的），输出 `partial_report.md` + 4 张图
  + CSV/JSON；耗时缺失时回退解析 `run.log` 的 `Epoch N Summary` 表；**不会给被 kill
  的 job 编 holdout 数值**。数值口径 import 自官方 summarizer（实测逐位相同）。
- **参数量少 ≠ 训练快**：`train_lora.py` 在 NMF 注入前把**整个基座**设为
  `requires_grad=False`（只训因子），而 `nmf_zkp_pdb` baseline **无任何冻结配置**（168M
  全量微调）。逐 step 只有 dW 能省 → 理论上限 1.5×，实际约 1.15–1.25×；encoder 一族 6 个
  job ≈ baseline 墙钟的 5–6 倍。测速用 `TIMING=1`（分段计时）或 `<tag>.run.log` 里的
  `Mean Time per Batch (s)`。
- **镜像里可用的 CPU 工具栈**（沙箱排查用）：`/home/zzj/.workbuddy/tmp/wb_torch`（torch CPU、
  pandas、matplotlib）、`wb_torch2`、`wb_tools`（pyflakes）、`wb_hydra`（hydra-core，
  可 `initialize_config_dir("models/rfd3/configs")` + `compose(...)` 验证 override 合法）。
- **训练数据是"parquet 索引 + 镜像"两件套，`path` 列被忽略**：
  `InterfacesDFParser` / `PNUnitsDFParser` 用模板
  `{base_dir}/{pdb_id[1:3]}/{pdb_id}.cif.gz` 自己拼路径（`path_template` 默认值），
  parquet 里的 `path` 列根本不读。→ 想缩训练集就**同时裁 parquet 的行 + 裁镜像**，
  两边一致即可（工具：`experiments/paper_rfd3/04_training_subset.py`）。
  但 `calculate_weights_for_pdb_dataset_df` 会在子集内重算 cluster size，
  采样权重随之变化 —— baseline 与各变体同子集才可比，绝对 lDDT 不能跨集比。
- **规模数字（2026-09-27 实测）**：`interfaces_df.parquet` 687.5 万行 / 训练 filters 后
  563 万行 / 195,296 pdb_id；`pn_units_df.parquet` 381 万 → 326 万 / 220,496 pdb_id。
  镜像 250,359 cif.gz / 82 GB，平均 337 KB/文件。holdout `evaluation_manifest.csv`
  256 行 = **73 个 unique pdb_id**（14 MB）。→ **论文 §3 全量训练集 ≈ 75 GB ≈ 全库，
  但微调只需按采样量裁一个小集**。
- **`pd.read_parquet` 读全列会 OOM**：`interfaces_df` 60 列 / 687 万行（大量 string 列），
  整表读入会被 SIGKILL（`rc=137`）。按训练配置的 `columns_to_load` 只读 18 列。
- **`nvidia-smi` 驱动故障时把错误写到 stdout**（不是 stderr），`2>/dev/null` 挡不住，
  会被当成显存数值做算术 → bash 报 `rc=9` 神秘退出。取值必须 `[[ $x =~ ^[0-9]+$ ]]`。
- 详细排错手册见用户级 skill `foundry-rfd3-eval-debug`（2026-09-25 建）。

## 目录

- `experiments/paper_rfd3/` —— RFdiffusion3 论文正文全部实验的复现脚本与手册（2026-09-24 建）
- `docs/paper/rfd3_paper.txt` —— 论文抽取文本（22 页，含补充图 S1-S9；缺 Supplemental Methods）
- `models/rfd3/scripts/` —— 之前留下的近原生/NMF 相关脚本（`run_near_native_rfd3.py`、
  `run_rfd3_paper_matrix.sh`、`run_nmf_zkp_pdb_sweep.sh` 等），与论文复现是两条线
- `evaluation_manifest.csv`、`rfdiffusion3_inputs/`、`logs/rfd3_paper_matrix/` —— NMF/ZKP 线路的产物
