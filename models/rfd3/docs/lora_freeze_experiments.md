# RFD3 LoRA 冻结实验说明

本文档说明基于 `lora_finetune.yaml` 派生出的几个冻结实验，以及它们**预期会训练哪些层**、**预期会冻结哪些层**。

你可以用它和训练日志里的：

- `LoRA trainable parameter summary`
- `Post-freeze trainable parameter summary`

做对照。

---

## 1. 共同前提

这些实验都基于：

```text
models/rfd3/configs/experiment/lora_finetune.yaml
```

共同设置包括：

- `lora.enabled: true`
- `rank: 8`
- `alpha: 16`
- `dropout: 0.05`
- `apply_to_token_initializer: false`
- `target_keywords`:
  - `to_q`
  - `to_k`
  - `to_v`
  - `to_o`
  - `to_g`
  - `fc1`
  - `fc2`

### 共同含义

1. **先注入 LoRA**
   - 主干中匹配到 `target_keywords` 的线性层会被替换成 `LoRALinear`
   - 原始 `base_layer` 冻结
   - 只训练 `lora_A` / `lora_B`

2. **再按 `parameter_freezing_config` 做额外冻结**
   - 进一步限制哪些参数还能训练

3. **`token_initializer` 默认不注入 LoRA**
   - 因为 `apply_to_token_initializer: false`

---

## 2. 你应该怎么判断“冻结有没有生效”

训练启动后，日志里会打印两类摘要：

### A. LoRA 注入后

```text
LoRA trainable parameter summary
```

这时你应该主要看到：

- `...lora_A.weight`
- `...lora_B.weight`

以及可能少量其他未被冻结的参数。

### B. 冻结策略应用后

```text
Post-freeze trainable parameter summary
```

这时你应该看到：

- 可训练参数更少
- 可训练层更少
- 只剩当前实验允许训练的部分

### 判定标准

- 如果某个模块本应被冻结，却还在 `requires_grad=True` 列表里  
  → 冻结路径没命中
- 如果只剩预期的 `lora_A/lora_B` 或 head 层  
  → 冻结生效

---

## 3. 实验总览

| 实验配置 | 名称 | 冻结范围 | 预期效果 |
|---|---|---|---|
| `lora_freeze_input.yaml` | `lora-freeze-input` | 冻结输入/初始化相关层 | 只保留中后段 LoRA 可训练 |
| `lora_freeze_encoder.yaml` | `lora-freeze-encoder` | 冻结 encoder / transformer / decoder | 保留输入侧和其余 LoRA |
| `lora_freeze_diffusion.yaml` | `lora-freeze-diffusion` | 冻结整个 diffusion 主干 | 训练参数应极少 |
| `lora_head_only.yaml` | `lora-head-only` | 默认全冻，只放开 head | 只训练少量 head 参数 |

---

## 4. 实验 1：`lora_freeze_input`

### 配置文件

```text
models/rfd3/configs/experiment/lora_freeze_input.yaml
```

### 冻结策略

```yaml
freeze_by_default: false
param_policies:
  model.token_initializer.*: true
  model.diffusion_module.process_r.*: true
  model.diffusion_module.process_n.*: true
  model.diffusion_module.process_a.*: true
  model.diffusion_module.process_c.*: true
```

### 预期被冻结的层

- `model.token_initializer.*`
- `model.diffusion_module.process_r.*`
- `model.diffusion_module.process_n.*`
- `model.diffusion_module.process_a.*`
- `model.diffusion_module.process_c.*`

### 预期仍可训练的层

主要是 **diffusion 中后段的 LoRA 参数**，例如：

- `model.diffusion_module.diffusion_token_encoder.*.lora_A.weight`
- `model.diffusion_module.diffusion_token_encoder.*.lora_B.weight`
- `model.diffusion_module.encoder.*.lora_A.weight`
- `model.diffusion_module.encoder.*.lora_B.weight`
- `model.diffusion_module.diffusion_transformer.*.lora_A.weight`
- `model.diffusion_module.diffusion_transformer.*.lora_B.weight`
- `model.diffusion_module.decoder.*.lora_A.weight`
- `model.diffusion_module.decoder.*.lora_B.weight`
- `model.diffusion_module.downcast_*.*.lora_A.weight`
- `model.diffusion_module.downcast_*.*.lora_B.weight`

### 预期不应该出现在可训练列表中的名字

- `token_initializer...`
- `process_r...`
- `process_n...`
- `process_a...`
- `process_c...`

### 适合验证什么

- 低层输入映射冻结后，模型还能不能适配单结构
- 中后段 LoRA 是否足以驱动微调

---

## 5. 实验 2：`lora_freeze_encoder`

### 配置文件

```text
models/rfd3/configs/experiment/lora_freeze_encoder.yaml
```

### 冻结策略

```yaml
freeze_by_default: false
param_policies:
  model.diffusion_module.encoder.*: true
  model.diffusion_module.diffusion_transformer.*: true
  model.diffusion_module.decoder.*: true
```

### 预期被冻结的层

- `model.diffusion_module.encoder.*`
- `model.diffusion_module.diffusion_transformer.*`
- `model.diffusion_module.decoder.*`

### 预期仍可训练的层

主要是 **未被上述路径覆盖的 LoRA 参数**，例如：

- `model.diffusion_module.diffusion_token_encoder.*.lora_A.weight`
- `model.diffusion_module.diffusion_token_encoder.*.lora_B.weight`
- `model.diffusion_module.downcast_c.*.lora_A.weight`
- `model.diffusion_module.downcast_c.*.lora_B.weight`
- `model.diffusion_module.downcast_q.*.lora_A.weight`
- `model.diffusion_module.downcast_q.*.lora_B.weight`

以及可能的其他未覆盖模块中的 LoRA。

### 预期不应该出现在可训练列表中的名字

- `...encoder...lora_A...`
- `...encoder...lora_B...`
- `...diffusion_transformer...lora_A...`
- `...diffusion_transformer...lora_B...`
- `...decoder...lora_A...`
- `...decoder...lora_B...`

### 适合验证什么

- 冻结生成核心 attention / transformer 后，模型能力掉多少
- 只保留 token encoder / downcast 是否还够用

---

## 6. 实验 3：`lora_freeze_diffusion`

### 配置文件

```text
models/rfd3/configs/experiment/lora_freeze_diffusion.yaml
```

### 冻结策略

```yaml
freeze_by_default: false
param_policies:
  model.diffusion_module.*: true
```

### 预期被冻结的层

- 整个 `model.diffusion_module.*`

### 预期仍可训练的层

理论上应该很少，甚至接近 0。

因为：

- LoRA 默认主要注入到 `diffusion_module`
- 而这里又把整个 `diffusion_module` 冻结了

### 预期结果

`Post-freeze trainable parameter summary` 中：

- 可训练参数应显著下降
- 甚至可能几乎没有可训练参数

### 适合验证什么

- 冻结整个 diffusion 后，训练是否基本失效
- 作为“最强冻结基线”做对照

---

## 7. 实验 4：`lora_head_only`

### 配置文件

```text
models/rfd3/configs/experiment/lora_head_only.yaml
```

### 冻结策略

```yaml
freeze_by_default: true
param_policies:
  model.diffusion_module.sequence_head.*: false
  model.diffusion_module.to_r_update.*: false
```

### 预期含义

- 默认全部冻结
- 只有下面这些放开：
  - `model.diffusion_module.sequence_head.*`
  - `model.diffusion_module.to_r_update.*`

### 预期可训练层

主要是：

- `model.diffusion_module.sequence_head.*`
- `model.diffusion_module.to_r_update.*`

### 预期不应出现的可训练层

- 绝大多数 `lora_A` / `lora_B`
- encoder / decoder / transformer 中的适配层

### 适合验证什么

- 只训练 head 是否还能学到一点东西
- 作为“最保守微调”基线

---

## 8. 预期训练参数量怎么理解

由于你当前是：

1. 先注入 LoRA
2. 再冻结

所以“训练参数量”通常会按下面的顺序变化：

### 全模型
很大，例如数亿参数

### LoRA 注入后
明显下降，只剩 `lora_A/lora_B`

### 再应用冻结后
继续下降，只剩当前实验允许训练的子集

### 经验判断

- `lora_freeze_input`：可训练参数中等
- `lora_freeze_encoder`：可训练参数更少
- `lora_freeze_diffusion`：可训练参数最少
- `lora_head_only`：通常也很少，且不一定是 LoRA 参数

---

## 9. 推荐的日志对照方式

每次实验启动后，优先看：

1. `LoRA enabled: trainable params=...`
2. `LoRA trainable parameter summary`
3. `Post-freeze trainable parameter summary`

### 好结果

- 第二步和第三步都能看到明确统计
- 第三步的可训练参数明显更少
- 参数名与本文“预期可训练层”一致

### 坏结果

- 第三步和第一步几乎一样
- 本应冻结的模块仍然出现在可训练列表
- 说明 `param_policies` 路径没有命中真实参数名

---

## 10. 启动命令

### 1）冻结输入层

```bash
python models/rfd3/src/rfd3/train_lora.py \
  experiment=lora_freeze_input \
  paths.data.monomer_distillation_parquet_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.monomer_distillation_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.design_benchmark_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.log_dir=/home/zzj/protein/foundry/logs/train \
  logger=csv
```

### 2）冻结 encoder/transformer/decoder

```bash
python models/rfd3/src/rfd3/train_lora.py \
  experiment=lora_freeze_encoder \
  paths.data.monomer_distillation_parquet_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.monomer_distillation_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.design_benchmark_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.log_dir=/home/zzj/protein/foundry/logs/train \
  logger=csv
```

### 3）冻结整个 diffusion

```bash
python models/rfd3/src/rfd3/train_lora.py \
  experiment=lora_freeze_diffusion \
  paths.data.monomer_distillation_parquet_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.monomer_distillation_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.design_benchmark_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.log_dir=/home/zzj/protein/foundry/logs/train \
  logger=csv
```

### 4）只训练 head

```bash
python models/rfd3/src/rfd3/train_lora.py \
  experiment=lora_head_only \
  paths.data.monomer_distillation_parquet_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.monomer_distillation_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.data.design_benchmark_data_dir=/home/zzj/protein/foundry/single_structure_data \
  paths.log_dir=/home/zzj/protein/foundry/logs/train \
  logger=csv
```

---

## 11. 结果对比建议

每个实验跑完后，重点比较：

- `Total Loss`
- `Mean Lddt Protein`
- `CA RMSD`
- `Seq Recovery`
- 可训练参数数量

### 推荐对照表

| 实验 | 可训练参数量 | 结构质量 | 是否值得保留 |
|---|---|---|---|
| `lora_freeze_input` |  |  |  |
| `lora_freeze_encoder` |  |  |  |
| `lora_freeze_diffusion` |  |  |  |
| `lora_head_only` |  |  |  |

---

## 12. 一句话总结

这些实验的核心不是“再写一套训练逻辑”，而是在同一套 LoRA 基础上：

- 改变冻结范围
- 比较不同冻结策略对单结构微调效果的影响

如果你在日志里看到的可训练参数和本文“预期可训练层”一致，就说明冻结策略生效了。
