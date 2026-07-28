# rfd3 微调与最小验证工作总结

本文档总结了本次对话中围绕 `rfd3` 最小推理、最小验证、以及本地微调数据准备所做的所有工作，便于在新的对话中继续推进。

---

## 1. 目标与总体结论

本次工作的目标是：

1. 理解 `rfd3` 项目的训练/推理入口
2. 找到最小可执行的推理命令
3. 定位并修复推理过程中的输入文件、checkpoint、路径等问题
4. 搭建本地可用于微调的最小数据组织方式
5. 生成一个尽量贴近原生项目结构的最小训练数据配置
6. 为后续生成 `interfaces_df.parquet` 做准备

### 总体结论

- `rfd3` 的推理入口是可用的，但需要正确的 checkpoint 和输入结构文件。
- 训练入口 `train.py` 依赖的是原生的 `parquet + parser + transform` 数据结构，而不是简单 JSON/单个 PDB 文件。
- 直接把一个 `ContigJsonDataset` 塞进训练入口并不稳，最终方向应该回到项目原生的 `InterfacesDFParser + interfaces_df.parquet` 结构。
- 当前最关键的缺口不是代码，而是 **本地最小可用的 `interfaces_df.parquet`**。

---

## 2. 项目结构与关键文件

在 `models/rfd3` 下，主要相关文件有：

### 配置文件

- `models/rfd3/configs/train.yaml`
  - 训练总入口配置
  - 默认加载 `model: rfd3_base`、`trainer: rfd3_base`、`datasets: design_base` 等

- `models/rfd3/configs/model/rfd3_base.yaml`
  - 模型配置聚合
  - 默认加载 optimizer / scheduler / sampler / EMA / network

- `models/rfd3/configs/model/components/rfd3_net.yaml`
  - `RFD3` 具体网络结构参数

- `models/rfd3/configs/trainer/rfd3_base.yaml`
  - 训练器配置
  - 包含 loss、metrics、batch 相关参数

- `models/rfd3/configs/datasets/design_base.yaml`
  - 原生训练集总配置
  - 默认引用多个 `train/pdb/...` 子数据集和 validation 配置

- `models/rfd3/configs/datasets/train/pdb/af3_train_interface.yaml`
  - 原生 interface 数据集底层配置
  - 关键字段：`dataset_parser`, `dataset.data`, `columns_to_load`, `filters`

- `models/rfd3/configs/datasets/train/pdb/rfd3_train_interface.yaml`
  - 在 `af3_train_interface` 基础上添加 `filters` 和 crop 参数

- `models/rfd3/configs/datasets/train/pdb/pdb_base.yaml`
  - 基础 PDB 数据集配置
  - 其中 `cif_parser_args.cache_dir` 指向 `paths.data.cif_cache_dir`

- `models/rfd3/configs/paths/default.yaml`
  - 默认输出目录、数据根目录等路径定义
  - 默认 `log_dir` 在 `/net/scratch/${oc.env:USER}/training/logs`

### 代码文件

- `models/rfd3/src/rfd3/train.py`
  - 训练主入口
  - 使用 Hydra 读取配置
  - 通过 `recursively_instantiate_datasets_and_samplers(cfg.datasets.train)` 构建训练数据

- `models/rfd3/src/rfd3/model/RFD3.py`
  - 模型前向逻辑
  - 训练时走 diffusion_module，推理时走 inference_sampler

- `models/rfd3/src/rfd3/trainer/rfd3.py`
  - 训练器实现
  - 负责 training_step / validation_step / loss / metrics

---

## 3. 最小推理命令的尝试与问题定位

### 3.1 最小推理命令

最初尝试的命令类似：

```bash
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src/"
uv run python models/rfd3/src/rfd3/run_inference.py \
  ckpt_path=models/rfd3/rfd3_latest.ckpt \
  out_dir=./na_tutorial_outputs \
  inputs=./models/rfd3/docs/tutorials/na_tutorial_files/rfd3_na_tutorial.json \
  n_batches=2 \
  diffusion_batch_size=3 \
  cleanup_virtual_atoms=True
```

### 3.2 第一个错误：checkpoint 文件无效

报错：

```text
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

结论：

- `models/rfd3/rfd3_latest.ckpt` 是空文件或损坏文件
- 不是推理命令本身的问题，而是 checkpoint 不可用

后续确认：

- 该文件大小为 0 或内容为空
- 因此推理会在 `torch.load` 处直接失败

### 3.3 第二个错误：输入 PDB 文件找不到

之后尝试使用教程输入 JSON 后，报错：

```text
FileNotFoundError: .../models/rfd3/docs/tutorials/na_tutorial_files/input_pdbs/2r5z.pdb
```

定位结果：

- `rfd3_na_tutorial.json` 中写的是：

```json
"input": "./input_pdbs/2r5z.pdb"
```

- 这个路径是相对 JSON 文件所在目录解析的
- 实际所需文件应该位于：

```text
models/rfd3/docs/tutorials/na_tutorial_files/input_pdbs/2r5z.pdb
```

但仓库教程说明里实际放置的文件在：

```text
models/rfd3/docs/input_pdbs/2r5z.pdb
```

### 3.4 修正 JSON 输入路径

将：

```json
"input": "./input_pdbs/2r5z.pdb"
```

改为：

```json
"input": "../../input_pdbs/2r5z.pdb"
```

目的是让 JSON 能正确指向 `models/rfd3/docs/input_pdbs/2r5z.pdb`。

### 3.5 PDB mirror 与 CCD mirror 环境变量

报错日志中还出现：

```text
Environment variable CCD_MIRROR_PATH not set
Environment variable PDB_MIRROR_PATH not set
```

后来明确：

- 这两个变量属于环境变量，不属于 JSON 配置
- 应在 shell 中设置：

```bash
export PDB_MIRROR_PATH=/media/zzj/Data/pdb_mirror
export CCD_MIRROR_PATH=/media/zzj/Data/ccd_mirror
```

它们主要供底层依赖库（如 `atomworks`）使用。

---

## 4. 关于 Hydra / 配置层级问题的多轮修正

### 4.1 初始问题：Hydra search path 警告

运行过程中经常出现：

```text
provider=hydra.searchpath in main, path=configs is not available.
```

这更像是配置搜索路径的 warning，而不是根本错误。

### 4.2 `paths.log_dir` 默认指向不可写目录

原始 `models/rfd3/configs/paths/default.yaml` 中：

```yaml
log_dir: /net/scratch/${oc.env:USER}/training/logs
```

在本地机器上无法写入 `/net`，因此报错：

```text
PermissionError: [Errno 13] Permission denied: '/net'
```

#### 处理方式

- 在实验配置中加入本地可写路径
- 或命令行强制覆盖：

```bash
paths.log_dir=/home/zzj/protein/foundry/logs
```

最终还建议直接修改 `models/rfd3/configs/paths/default.yaml` 或在 experiment 配置中覆盖 `paths.log_dir`。

### 4.3 Hydra 的 `hydra.runtime.output_dir` 问题

在某些 experiment 配置中，`hydra.job_logging.handlers.file.filename` 引用了：

```yaml
${hydra.runtime.output_dir}/experiment.log
```

但在当前加载阶段还没有可解析的 `hydra.runtime.output_dir`，导致：

```text
InterpolationKeyError: Interpolation key 'hydra.runtime.output_dir' not found
```

#### 处理方式

- 去掉 experiment 文件中的 `- /hydra: default`
- 让默认 Hydra 配置不被错误覆盖

### 4.4 Hydra 对 `datasets` / `callbacks` / `dataloader` 的相对路径解析问题

曾经在 `experiment/my_validate_small.yaml` 中写了：

```yaml
- override /datasets: my_validate_small
- override /callbacks: design_callbacks
- dataloader: fast
```

Hydra 报错：

```text
Could not find 'experiment/dataloader/fast'
Could not find 'experiment/callbacks/design_callbacks'
Could not override 'datasets@experiment.datasets'
```

#### 原因

- 这些组被 Hydra 当作 `experiment/...` 的相对路径去解析
- 而实际配置文件位于全局组目录，如：
  - `configs/dataloader/fast.yaml`
  - `configs/callbacks/design_callbacks.yaml`

#### 处理方式

后来尝试用：

```yaml
- /dataloader: fast
- /callbacks: design_callbacks
```

但整体上，这条路一直比较曲折。

---

## 5. 关于最小验证配置 `my_validate_small` 的问题

### 5.1 最初意图

为了先验证最小输入链路，我们尝试构造一个轻量版验证配置：

- 使用 `ContigJsonDataset`
- 指向一个 `minimal_validate.json`
- 用作最小数据验证

### 5.2 最小验证 JSON

曾创建：

`data/rfd3_finetune/metadata/minimal_validate.json`

内容类似：

```json
{
  "100d": {
    "input": "/media/zzj/Data/pdb_mirror/00/100d.cif.gz",
    "contig": "A1-10",
    "length": "10-10",
    "is_non_loopy": true
  }
}
```

### 5.3 为什么这条路线不稳

后来逐步发现：

- `train.py` 不是简单读取 `ContigJsonDataset`
- 训练入口期望的是原生训练数据结构
- `recursively_instantiate_datasets_and_samplers` 会要求数据集节点有 `probability`
- 更深层还要求与原生 `train/pdb/...` 一致的嵌套结构

因此，使用 `ContigJsonDataset` 去硬塞训练入口并不贴合项目原生 schema。

### 5.4 最终认知

`train.py` 预期的是：

- `parquet` 索引文件
- `dataset_parser`
- `dataset`
- `transform`
- 以及通过 `defaults` 组合出来的标准层级

而不是推理任务里的 JSON 输入格式。

---

## 6. 对照原生 YAML 之后的最终判断

对照原生配置后，明确得出结论：

### 原生训练数据的典型结构

以 `rfd3_train_interface.yaml` 为例：

```yaml
defaults:
  - af3_train_interface
  - pdb_base
  - _self_

dataset:
  transform:
    crop_contiguous_probability: 0.0
    crop_spatial_probability: 1.0
    filters:
      - ...
```

而底层 `af3_train_interface.yaml`：

```yaml
dataset:
  dataset_parser:
    _target_: atomworks.ml.datasets.parsers.InterfacesDFParser
    base_dir: ${paths.data.pdb_data_dir}
  dataset:
    name: interface
    data: ${paths.data.pdb_parquet_dir}/interfaces_df.parquet
    filters:
      - ...
    columns_to_load:
      - example_id
      - pdb_id
      - assembly_id
      - deposition_date
      - resolution
      - num_polymer_pn_units
      - method
      - cluster
      - n_prot
      - n_nuc
      - n_ligand
      - n_peptide
      - pn_unit_1_iid
      - pn_unit_2_iid
      - pn_unit_1_non_polymer_res_names
      - pn_unit_2_non_polymer_res_names
      - is_inter_molecule
      - all_pn_unit_iids_after_processing
      - involves_loi
```

### 结论

项目原生训练结构不是 “JSON + ContigJsonDataset”，而是：

- `InterfacesDFParser`
- `interfaces_df.parquet`
- `dataset_parser.base_dir`
- `dataset.data`
- `columns_to_load`
- `filters`
- `transform`

所以要做微调，应回到这个原生 schema。

---

## 7. 生成本地最小训练数据配置

### 7.1 新建配置文件

创建了：

`models/rfd3/configs/datasets/train/pdb/my_finetune_interface.yaml`

它参考了原生 `rfd3_train_interface.yaml`，并做了本地化路径修改。

### 7.2 该配置的核心内容

- 使用原生 `af3_train_interface` 和 `pdb_base`
- 将 `dataset_parser.base_dir` 指向本地 `paths.data.pdb_data_dir`
- 将 `dataset.data` 指向本地 `paths.data.pdb_parquet_dir/interfaces_df.parquet`
- 保留原生接口所需列名和 transform 结构
- 通过 `filters` 减少训练样本量，方便最小实验

### 7.3 对应实验配置

还创建了：

`models/rfd3/configs/experiment/my_finetune.yaml`

用于把训练组改成本地微调数据集。

### 7.4 推荐运行命令

```bash
mkdir -p /home/zzj/protein/foundry/logs

python models/rfd3/src/rfd3/train.py \
  experiment=my_finetune \
  ckpt_path=/home/zzj/protein/foundry/models/rfd3/rfd3_latest.ckpt \
  paths.log_dir=/home/zzj/protein/foundry/logs
```

但这条命令后续是否能成功，仍取决于本地是否已经存在有效的 `interfaces_df.parquet`。

---

## 8. 关于 `interfaces_df.parquet` 的现状与分析

### 8.1 仓库中没有现成的 `interfaces_df.parquet`

已经确认：

- `foundry` 仓库中没有附带可直接使用的 `interfaces_df.parquet`
- 也没有 `pn_units_df.parquet`

但原生 yaml 明确依赖这些文件。

### 8.2 官方可参考的内容

官方可参考的不是 parquet 本体，而是：

- `models/rfd3/configs/datasets/train/pdb/af3_train_interface.yaml`
- `models/rfd3/configs/datasets/train/pdb/rfd3_train_interface.yaml`

这些文件定义了 parquet 的 schema 参考。

### 8.3 最少需要哪些字段

从原生配置整理出最小必须字段：

- `example_id`
- `pdb_id`
- `assembly_id`
- `deposition_date`
- `resolution`
- `num_polymer_pn_units`
- `method`
- `cluster`
- `n_prot`
- `n_nuc`
- `n_ligand`
- `n_peptide`
- `pn_unit_1_iid`
- `pn_unit_2_iid`
- `pn_unit_1_non_polymer_res_names`
- `pn_unit_2_non_polymer_res_names`
- `is_inter_molecule`
- `all_pn_unit_iids_after_processing`
- `involves_loi`

### 8.4 本地最小可用 parquet 的建议

建议先用 1~5 条样本手工构造一个小表，字段值可尽量“保守”：

- `deposition_date`: 例如 `2020-01-01`
- `resolution`: 例如 `2.0`
- `num_polymer_pn_units`: 例如 `2`
- `cluster`: 例如 `test_cluster`
- `is_inter_molecule`: `True`

然后导出到：

`/home/zzj/protein/foundry/data/rfd3_finetune/metadata/interfaces_df.parquet`

若 parser 报某个字段类型不对，再针对性修正。

---

## 9. 目录组织建议

### 9.1 原始 mirror 不要改

你的原始 PDB mirror 保持如下结构即可：

```text
/media/zzj/Data/pdb_mirror/
  00/
    100d.cif.gz
    200d.cif.gz
    200l.cif.gz
  01/
  02/
  ...
```

这是一种标准 mirror 组织形式，不需要扁平化。

### 9.2 建议的本地微调目录

建议在仓库内另建：

```text
/home/zzj/protein/foundry/data/rfd3_finetune/
  metadata/
    interfaces_df.parquet
  structures/
    ...（如果需要额外的本地结构副本）
```

其中：

- `metadata/interfaces_df.parquet`：训练样本索引
- `paths.data.pdb_data_dir`：继续指向 `/media/zzj/Data/pdb_mirror`
- `paths.data.pdb_parquet_dir`：指向本地 `metadata`

---

## 10. 当前已生成/修改的文件清单

### 新增文件

1. `models/rfd3/configs/paths/data/local.yaml`
   - 本地路径配置

2. `models/rfd3/configs/datasets/my_validate_small.yaml`
   - 曾用于最小验证探索
   - 但最终结论是：这条路线不如直接走原生训练 schema 稳

3. `models/rfd3/configs/experiment/my_validate_small.yaml`
   - 曾用于验证配置流程
   - 由于 Hydra 覆盖与 schema 复杂性，后续不建议继续沿此方向迭代

4. `data/rfd3_finetune/metadata/minimal_validate.json`
   - 最小验证 JSON
   - 属于推理验证用途，不是最终训练 schema

5. `models/rfd3/configs/datasets/train/pdb/my_finetune_interface.yaml`
   - 这是当前最重要的新增文件
   - 参考原生 `rfd3_train_interface.yaml`，用于本地微调

6. `models/rfd3/configs/experiment/my_finetune.yaml`
   - 对应微调实验配置

### 被修改过的文件

1. `models/rfd3/docs/tutorials/na_tutorial_files/rfd3_na_tutorial.json`
   - 已修正 `2r5z.pdb` 相对路径

2. `models/rfd3/configs/experiment/my_validate_small.yaml`
   - 多次修改 Hydra / paths / datasets / callbacks / dataloader 的写法
   - 最终认知：这条验证路线不如直接走原生训练配置稳定

3. `models/rfd3/configs/datasets/my_validate_small.yaml`
   - 多次尝试调整 `probability` 和层级结构
   - 最终认知：结构与训练入口 schema 不匹配，建议不再作为主方向

---

## 11. 关键报错与对应原因

### 11.1 checkpoint 读取失败

原因：`rfd3_latest.ckpt` 空文件或损坏。

### 11.2 输入 PDB 找不到

原因：教程 JSON 中相对路径与实际文件位置不一致。

### 11.3 `PDB_MIRROR_PATH` / `CCD_MIRROR_PATH` 未设置

原因：环境变量缺失，不是程序逻辑错误。

### 11.4 `/net/scratch/...` 无写权限

原因：默认 `paths.log_dir` 在本机不可写。

### 11.5 `hydra.runtime.output_dir` 不可解析

原因：Hydra 配置覆盖顺序 / runtime context 不匹配。

### 11.6 `Could not find 'experiment/dataloader/fast'` / `callbacks/design_callbacks`

原因：Hydra 将组名按相对路径解析到了 `experiment/...` 下。

### 11.7 `Expected 'probability' key in dataset configuration`

原因：数据集层级结构不符合 `recursively_instantiate_datasets_and_samplers` 预期。

### 11.8 根本性问题：训练 schema 不匹配

原因：使用了推理式 JSON/Contig 结构去跑训练入口，而项目原生训练要求是 parquet + parser + transform 的组合。

---

## 12. 当前最推荐的后续路线

### 路线 A：走原生训练 schema

这是目前最推荐的方向。

1. 准备一个最小可用的 `interfaces_df.parquet`
2. 让它至少包含原生配置中引用的字段
3. 使用 `models/rfd3/configs/datasets/train/pdb/my_finetune_interface.yaml`
4. 用 `models/rfd3/configs/experiment/my_finetune.yaml` 启动训练

### 路线 B：先做纯推理验证

如果只是想确认输入/推理链路：

1. 继续使用教程 JSON
2. 确保 PDB / checkpoint 都可用
3. 用推理入口验证结构文件能否被正确读取

但这不等于能直接完成微调。

---

## 13. 最后总结

本次工作逐步得出的核心结论是：

1. **推理问题**：主要卡在 checkpoint 和输入文件路径
2. **训练问题**：不是简单的 JSON 配置，而是原生 `parquet + parser + transform` schema
3. **配置问题**：Hydra 的 defaults / override / searchpath 容易引发一连串相对路径错误
4. **微调方向**：应回到项目原生的 `af3_train_interface.yaml` / `rfd3_train_interface.yaml` 结构
5. **关键缺口**：需要先生成一个最小可用的 `interfaces_df.parquet`

下一步最值得做的事情是：

- 写一个脚本，从 `pdb_mirror` 中选少量 PDB ID
- 生成最小版本的 `interfaces_df.parquet`
- 然后用 `my_finetune_interface.yaml` 启动训练

---

## 14. 建议在新对话里直接接续的内容

你可以在新对话里直接告诉模型：

- 继续基于本工作总结
- 重点推进 `interfaces_df.parquet` 的最小生成脚本
- 目标是让 `my_finetune_interface.yaml` 真正跑起来

这样可以避免重复排查 Hydra 和推理阶段的旧问题。
