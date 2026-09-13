# NMF ZKP PDB Holdout 工作交接

这份文档总结截至当前对话的全部排查、结论和代码改动，方便在新对话里直接继续。

## 1. 目标

用 PDB 时间 holdout 比较 replacement NMF 微调和原 RFD3：

- 训练：`rfd3_train_interface` / `pn_units`
- 评估：固定 temporal holdout `datasets/val/pdb_holdout.yaml`
- 启动命令：

```bash
N_EXAMPLES=128 MAX_EPOCHS=575 INCLUDE_BASELINE=1 INCLUDE_ALL=0 \
  bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh
```

比较指标必须是 holdout `val/pdb_holdout/mean_lddt`，不要用训练 loss / train lDDT 当泛化代理。

sweep 脚本会跑：

- `baseline`（无 NMF，从 `rfd3_latest.ckpt` 微调）
- `zkp_encoder`
- `zkp_proj`
- `zkp_head`

`INCLUDE_ALL=0` 时不跑 `zkp_all`。

## 2. 机器与路径

- 仓库：`/root/protein/foundry`
- 环境：`/opt/conda/envs/rc`
- PDB mirror：`/media/zzj/Data/pdb_mirror`
- parquet / ckpt：`/media/zzj/Data/pdb_metadata_latest`
- 日志根目录：`/root/protein/foundry/logs/train_nmf_zkp_pdb`
- 最近一次失败 sweep：`sweep_nmf_zkp_pdb_2026-09-10_18-57-38`
- 训练预算：`crop_size=256`，`max_atoms_in_crop=1920`，`diffusion_batch_size_train=4`，GPU 约 80GB

**关键坑：** conda `site-packages/rfd3` 和 `foundry` 是旧安装副本。训练/预检必须优先 import 仓库源码：

- `models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh` 已 export `PYTHONPATH`
- `models/rfd3/src/rfd3/train_lora.py` 和 `preflight_nmf_pdb_transforms.py` 在 import 前把仓库 `src` 插到 `sys.path` 最前

新对话里如果改了代码却看不到效果，先确认：

```python
import rfd3.transforms.pipelines as p
print(p.__file__)
# 必须是 /root/protein/foundry/models/rfd3/src/rfd3/transforms/pipelines.py
```

## 3. 问题时间线

### 3.1 验证 `X_noisy_L` NaN（9hql）

报错：

```text
AssertionError: network_input (X_noisy_L) for example_id: {['pdb', 'interfaces']}{9hql}{1}{['A_1', 'B_1']}: Tensor contains NaNs!
```

原因不是“黑名单漏了一个 PDB”，而是预检和验证路径不一致：

- 用户预检用了 `--training-conditions`，旧脚本会把 `is_inference=False`
- 训练路径裁剪后会填补未解析坐标；trainer 训练时还能 `nan_to_num`
- 真正的 `validation_loop` 用 `is_inference=true`：不裁剪、不填补
- 旧预检只看 transform 是否抛异常，不检查 `coord_atom_lvl_to_be_noised + noise` 是否有 NaN
- 所以 9hql 能过预检，却在验证组装 `X_noisy_L` 时炸掉

### 3.2 预检几乎全军覆没

用户按验证路径重跑预检后，前 140/4322 条几乎全是 `X_noisy_L` NaN。这不是 holdout 数据废了，而是 **RFD3 验证管线本身不适合实验 PDB**：

- RF3 会在 `coord_to_be_noised` 上跑 `PlaceUnresolvedToken*`
- RFD3 把同样的填补包在 `TrainingRoute` 里，验证直接跳过
- 实验 PDB 几乎都有 occupancy=0 / 未解析原子

当时结论：不要靠黑名单筛掉几乎全部 PDB，也不要关掉验证改用训练指标。应修验证管线。

### 3.3 训练 `appears multiple times`

给 diffusion 段无条件加了 `PlaceUnresolvedToken*` 后，训练报：

```text
Transform `PlaceUnresolvedTokenAtomsOnRepresentativeAtom` appears multiple times
```

日志特征：

- 有 `CropSpatialLikeAF3`：这是训练，不是验证
- 样本是 `{['pdb', 'pn_units']}{7c4y}`、`{['pdb', 'interfaces']}{4y8l}`：训练集
- atomworks 禁止同一 transform 在 history 里出现两次

正确分流：

| 路径 | 裁剪后填 `coord` | diffusion 再填 `coord_to_be_noised` |
|---|---|---|
| 训练 | 要 | 不要（会重复） |
| 验证 | 不要（验证原本不裁剪） | 要，否则 `X_noisy_L` NaN |

### 3.4 最终验证 CUDA OOM

baseline 572 epoch 训完，崩在：

```text
Running explicit final validation on the completed model.
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 43.20 GiB
```

堆栈在 `token_initializer.init_atoms -> motif_pos_embedder`，即 atom-pair `P_LL` 是 `L×L`。

原因：

- 训练按 1920 atoms 裁剪，80GB 够用
- holdout YAML 虽然写了 `max_atoms_in_crop: ${datasets.max_atoms_in_crop}`，但裁剪被包在 `TrainingRoute` 里，验证时被跳过
- 整条 PDB 复合物进入 pairwise embedding，先申请 21.6 GiB，再申请 43.2 GiB

`Found non-zero occupancy in input, setting occupancy to 1` 只是 occupancy 警告，不是 OOM 原因。

这次 fit 结束后、最终验证前才存最终 ckpt 的旧逻辑，导致验证一崩可能丢掉“训完的最终权重”。周期 ckpt 仍可能存在：`checkpoint_every_n_epochs=10`。

## 4. 最终代码状态

### 4.1 验证/训练管线

`models/rfd3/src/rfd3/transforms/pipelines.py`  
`models/rfd3na/src/rfd3na/transforms/pipelines.py`

- `get_crop_transform()`：若 `max_atoms_in_crop is not None`，验证也裁剪；否则仍 `TrainingRoute(crop)`，设计推理不裁
- `get_diffusion_transforms()`：`PlaceUnresolvedToken*` 包在 `InferenceRoute` 里，只填 `coord_to_be_noised`
- 训练：裁剪后填 `coord`，再 `CopyAnnotation`；diffusion 段是 Identity
- 验证/holdout：裁到 1920 atoms，再填 `coord_to_be_noised`

### 4.2 trainer

`models/rfd3/src/rfd3/trainer/rfd3.py`  
`models/rfd3na/src/rfd3na/trainer/rfd3na.py`

- 验证遇到 `X_noisy_L` / `feats` NaN：skip 该样本，不中断 epoch
- 验证遇到 CUDA OOM：skip，并 `torch.cuda.empty_cache()`
- skip 结果：`{"skip": True, "metrics_output": None, ...}`

### 4.3 回调

- `src/foundry/callbacks/metrics_logging.py`：skip 不写 metrics；空 rank CSV 带表头；全部为空时返回空 DataFrame
- `models/rfd3/src/rfd3/trainer/dump_validation_structures.py` 和 rfd3na 对应文件：skip 不 dump
- `models/rfd3/src/rfd3/callbacks.py` 和 rfd3na 对应文件：空 validation CSV 不崩

### 4.4 训练入口

`models/rfd3/src/rfd3/train_lora.py`

- 优先加载仓库源码
- **fit 结束后、最终验证前先 `save_checkpoint()`**
- 然后才跑 explicit final validation
- sweep 仍要求写出 `validation_output_all_epochs.csv`，否则不算成功比较 run

### 4.5 预检

`models/rfd3/scripts/preflight_nmf_pdb_transforms.py`

- 始终 `is_inference=true`
- `--training-conditions` 不再切到训练路径
- transform 成功后检查 `X_noisy_L` 和 `feats` NaN
- `--write-filter` 把失败 PDB 并入 `pdb_holdout.yaml`

`models/rfd3/configs/datasets/val/pdb_holdout.yaml`

- 已有历史坏 PDB 黑名单（含 `9hql`）
- `max_atoms_in_crop: ${datasets.max_atoms_in_crop}` 现在会真正生效

### 4.6 测试

- `models/rfd3/tests/test_preflight_nmf_pdb_transforms.py`
- `models/rfd3/tests/test_validation_unresolved_coords.py`
- `tests/test_callbacks.py`：skip / 空 CSV

当前环境没有 pytest，可用：

```bash
export PYTHONPATH="/root/protein/foundry/src:/root/protein/foundry/models/rfd3/src:/root/protein/foundry/models/rfd3na/src"
/opt/conda/envs/rc/bin/python -c "from rfd3.transforms.pipelines import get_crop_transform; print('ok')"
```

## 5. 新对话应先做什么

1. 确认仓库源码被实际 import，而不是 conda site-packages。
2. 找已有 checkpoint，避免 572 epoch 从零重训：

```bash
find /root/protein/foundry/logs/train_nmf_zkp_pdb/sweep_nmf_zkp_pdb_2026-09-10_18-57-38 -name '*.ckpt'
```

3. 用修好的验证路径重新开 sweep，或只对已有 ckpt 补 holdout。
4. 成功标志：每个 setting 目录下有

```text
*/val_metrics/validation_output_all_epochs.csv
```

以及 sweep 的 `comparison_table.md` / `comparison_table.csv` 里 `best_val_lddt` 非空。

## 6. 建议命令

完整重跑（如果没有可用 ckpt）：

```bash
N_EXAMPLES=128 MAX_EPOCHS=575 INCLUDE_BASELINE=1 INCLUDE_ALL=0 \
  bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh
```

如果 baseline 已有周期 ckpt，优先把 `ckpt_config.path` 指到最后一次 ckpt，把 `trainer.max_epochs` 设得很小，只补验证，不要从 `rfd3_latest.ckpt` 再训 572 epoch。

可选：确认 holdout transform 现在能过：

```bash
python models/rfd3/scripts/preflight_nmf_pdb_transforms.py \
  --experiment nmf_zkp_pdb \
  --data-dir /media/zzj/Data/pdb_metadata_latest \
  --pdb-mirror /media/zzj/Data/pdb_mirror \
  --output logs/pdb_transform_preflight.csv \
  --write-filter models/rfd3/configs/datasets/val/pdb_holdout.yaml
```

注意：预检现在会走验证裁剪（因为 holdout 设了 `max_atoms_in_crop`）。黑名单只应收集真正坏结构，不应再出现“几乎全部失败”。

## 7. 不要再做的事

- 不要把 `--training-conditions` 理解成“更严格的验证筛选”。它曾经把预检切到训练路径，反而漏掉验证 NaN。
- 不要用不断扩大 `pdb_id not in [...]` 来解决系统性管线 bug。
- 不要关掉 holdout，改用训练 loss 比较 NMF vs 原 RFD3。
- 不要无条件在训练和验证路径都跑 `PlaceUnresolvedToken*`；atomworks 会报重复。
- 不要假设改了 `models/rfd3/src` 就一定生效；必须检查 `rfd3.__file__`。

## 8. 仍未完成 / 新对话待验证

- 修好后的 holdout 还没在 GPU 上完整跑通 256 条。
- `sweep_nmf_zkp_pdb_2026-09-10_18-57-38` 的 baseline 最终验证 OOM，`exit=1`。需要确认周期 ckpt 是否还在。
- NMF 三个 setting 可能还没训完，取决于这次 sweep 是否在 baseline 失败后继续。
- 验证裁剪后，holdout lDDT 比较的是 **同一 atom 预算下的 crop**，不是全复合物。这对 NMF vs baseline 仍然公平，因为两边用同一 holdout transform。论文里应写明 holdout 也 crop 到 1920 atoms。
- 偶发超大 crop / 残余 NaN 会被 skip；比较时应看 coverage，不要假装 256/256 全成功。

## 9. 关键文件清单

- `models/rfd3/src/rfd3/transforms/pipelines.py`
- `models/rfd3na/src/rfd3na/transforms/pipelines.py`
- `models/rfd3/src/rfd3/trainer/rfd3.py`
- `models/rfd3na/src/rfd3na/trainer/rfd3na.py`
- `models/rfd3/src/rfd3/train_lora.py`
- `models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh`
- `models/rfd3/scripts/preflight_nmf_pdb_transforms.py`
- `models/rfd3/configs/datasets/val/pdb_holdout.yaml`
- `models/rfd3/configs/experiment/nmf_zkp_pdb.yaml`
- `src/foundry/callbacks/metrics_logging.py`
- `models/rfd3/tests/test_validation_unresolved_coords.py`
- `models/rfd3/tests/test_preflight_nmf_pdb_transforms.py`

## 10. 本次修复：PDB holdout 缺少 specification

最终验证在 `_build_predicted_atom_array_stack()` 中无条件读取 `example["specification"]`。直接由 PDB 数据集生成的 holdout example 不经过 inference input parser，因此可能没有该字段，导致：

```text
KeyError: 'specification'
```

已修复：

- `models/rfd3/src/rfd3/trainer/rfd3.py`
- `models/rfd3na/src/rfd3na/trainer/rfd3na.py`

两条路径现在通过 `_get_example_specification()` 将缺失或 `None` 规格视为空字典；已有规格对象保持原样。这样不会改变模型输入或指标计算，只影响可选的输出 metadata：`task` 在没有 `example` 字段时不写入，`output_full_json=true` 时写入 `{}`。

验证状态：修改文件已通过 `compileall`、`git diff --check` 和 Cursor lint；当前环境未安装 `pytest`，因此 pytest 回归测试尚未执行。重跑前仍需确认使用仓库源码路径，而不是旧的 site-packages 副本。

## 11. 2026-09-12 pandas 环境修复

本次 sweep 的三个 setting 都在训练入口导入阶段失败，尚未加载 checkpoint 或执行训练。根因是 `/opt/conda/envs/rc/lib/python3.12/site-packages/pandas` 安装不完整，缺少 `pandas.util.version`，并非 NMF 或模型代码错误。

已在 rc 环境执行：

```bash
/opt/conda/envs/rc/bin/python -m pip install --force-reinstall --no-deps pandas==2.3.3
```

现在 pandas 2.3.3 可以正常导入，`foundry.utils.logging`、仓库源码 `foundry` 和 `rfd3` 的导入链也已验证通过。

另外，`run_nmf_zkp_pdb_sweep.sh` 已修改为：

- 未显式设置 `PYTHON` 时优先使用 `/opt/conda/envs/rc/bin/python`；
- 在启动任何 job 前先检查 pandas、foundry、rfd3 导入；
- 任一 job 失败时，sweep 最终以非零退出码结束，而不是打印 `Done` 后返回成功。

## 12. 2026-09-12 src_component 缺失修复

baseline 已完成 572 个 epoch 并成功保存最终 checkpoint，但 final validation 在输出结构 metadata 阶段失败：

```text
AttributeError: 'AtomArray' object has no attribute 'src_component'
```

原因是 PDB holdout 直接加载的 `AtomArray` 不一定带 inference parser 才会设置的 `src_component` annotation。该字段仅用于识别 external source 并补充 `diffused_index_map`，不是模型前向或 holdout lDDT 所需字段。

已在以下文件加入 annotation 存在性判断：

- `models/rfd3/src/rfd3/trainer/rfd3.py`
- `models/rfd3na/src/rfd3na/trainer/rfd3na.py`

缺少 `src_component` 时跳过 external-source mapping，其他预测、清理和 metrics 流程继续执行。修改已通过 `compileall`、`git diff --check`、导入检查和 lint。

之前保存的最终 checkpoint：

```text
/root/protein/foundry/logs/train_nmf_zkp_pdb/sweep_nmf_zkp_pdb_2026-09-12_10-08-08/train/baseline/2026-09-12_10-08_JOB_default/ckpt/epoch-0572.ckpt
```

## 13. 论文级 holdout 数据与 validation 频率

`run_nmf_zkp_pdb_sweep.sh` 现在通过 `trainer.validate_every_n_epochs=1000000000` 关闭周期 validation；每个 setting 只在训练 `fit()` 完成后执行一次 explicit final validation。这样不会每个 epoch 重跑 256 个 holdout batch，训练阶段耗时显著降低；最终每个 setting 仍需承担一次完整 holdout validation，这是能力比较所必需的。

validation metrics 配置新增 `lddt` metric，因此新生成的 CSV 应包含：

```text
lddt.mean_lddt
lddt.mean_lddt_protein
```

旧 baseline CSV（只有 29 列、没有 lDDT）不能用于最终能力比较，必须用修复后的代码对 `epoch-0572.ckpt` 再验证一次。

每次完整 sweep 结束后自动生成：

- `paper_per_example_lddt.csv`：逐 example 的各 setting lDDT；
- `paper_summary.csv`：每个 setting 的 n、coverage、mean、SD、median、bootstrap 95% CI；
- `paper_paired_deltas.csv`：相对 baseline 的逐样本配对 delta 及 bootstrap CI；
- `paper_summary.md`：可直接整理进论文的 Markdown 汇总。
