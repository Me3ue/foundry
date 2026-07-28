# RFD3 + LoRA 微调方案（基于 `single_structure` 配置）

本文档整理了如何在当前仓库里给 RFD3 做 LoRA 微调的完整方案，并结合你现在正在使用的训练命令，给出一套可以直接落地的修改建议。

---

## 1. 你当前的训练方式

你现在运行的是下面这个命令：

```bash
python models/rfd3/src/rfd3/train.py \
  experiment=single_structure \
  paths.data.monomer_distillation_parquet_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.monomer_distillation_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.design_benchmark_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.log_dir="/home/zzj/protein/foundry/logs/train" \
  logger=csv
```

这说明你现在已经在使用仓库里现成的 `single_structure` 实验配置来做单结构过拟合/微调。

对应配置文件在这里：

```text
models/rfd3/configs/experiment/single_structure.yaml
```

这个配置的核心特点是：

- 只训练单个结构相关的数据
- `ckpt_path: null`，表示从随机初始化权重开始
- 训练集和验证集都指向你指定的单结构数据目录
- 训练节奏很短，方便快速观察模型变化

---



## 2. `single_structure.yaml` 的作用

当前配置文件的主要内容包括：

- 只启用 monomer distillation 数据集
- 关闭 PDB 数据集
- 把训练和验证都导向同一个小数据目录
- 设置非常短的 epoch 和频繁验证

也就是说，它本身非常适合作为：

- 单样本过拟合测试
- 新数据域的快速实验
- LoRA 微调的起点

你现在要做的事情，其实就是在这个配置基础上增加一个“LoRA 开关”。

---



## 3. LoRA 和 PEFT 的直观理解



### 3.1 PEFT 是什么

PEFT 的意思是“参数高效微调”。

传统全参微调会把模型所有参数都更新，这样：

- 显存占用大
- 训练慢
- 容易破坏原有能力

PEFT 的思路是：

- 冻结原模型参数
- 只训练少量新增参数

---



### 3.2 LoRA 是什么

LoRA 是一种常见的 PEFT 方法。

对于一个线性层：

```text
y = xW
```

LoRA 不直接修改 `W`，而是增加一个低秩修正项：

```text
y = xW + xBA
```

其中：

- `W` 不训练
- `A` 和 `B` 是新增的小矩阵
- `rank r` 很小，所以参数量很少

效果是：

- 显存更省
- 训练更快
- 更适合在原模型能力上做适配

---



## 4. RFD3 里 LoRA 最适合加在哪里

从仓库结构来看，RFD3 的主要模型入口是：

- `models/rfd3/src/rfd3/model/RFD3.py`
- `models/rfd3/src/rfd3/model/RFD3_diffusion_module.py`
- `models/rfd3/src/rfd3/model/layers/attention.py`



### 推荐优先级



#### 第一优先级：`diffusion_module`

也就是 `RFD3_diffusion_module.py` 里的主干模块：

- `LocalAtomTransformer`
- `DiffusionTokenEncoder`
- `LocalTokenTransformer`
- `CompactStreamingDecoder`

这些模块里通常包含大量线性层，是 LoRA 最有价值的地方。

#### 第二优先级：`token_initializer`

这个模块也可以加 LoRA，但建议第一版先不要动，先把 diffusion 主干微调跑通。

---



## 5. 推荐的最小改动方案

建议你用下面这个方案：

### 新增一个 LoRA 工具文件

```text
models/rfd3/src/rfd3/peft_lora.py
```



### 修改训练入口

```text
models/rfd3/src/rfd3/train.py
```



### 可选新增一个实验配置

```text
models/rfd3/configs/experiment/lora_finetune.yaml
```

如果你希望改动最少，可以先不新建实验配置，直接在命令行里加 `lora.enabled=true` 之类的覆盖参数。

---



## 6. 新增文件：`models/rfd3/src/rfd3/peft_lora.py`

这个文件负责三件事：

1. 定义 LoRA 版线性层
2. 递归替换模型中指定名字的线性层
3. 冻结原模型参数，只留下 LoRA 参数可训练

下面是可直接使用的实现。

```python
import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear only supports nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 0.0
        self.merge_weights = merge_weights
        self.merged = False

        if r > 0:
            self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
            self.lora_dropout = nn.Dropout(lora_dropout)
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.base_layer(x)
        if self.r > 0 and not self.merged:
            out = out + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return out

    def merge(self):
        if self.r <= 0 or self.merged:
            return
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
        self.merged = True

    def unmerge(self):
        if self.r <= 0 or not self.merged:
            return
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data -= delta_w.to(self.base_layer.weight.dtype)
        self.merged = False


def _should_replace_linear(module_name: str, target_keywords: Iterable[str]) -> bool:
    return any(keyword in module_name for keyword in target_keywords)


def inject_lora_into_model(
    model: nn.Module,
    target_keywords: list[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    freeze_all: bool = True,
) -> nn.Module:
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False

    def _replace(parent: nn.Module, prefix: str = ""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear) and _should_replace_linear(full_name, target_keywords):
                setattr(
                    parent,
                    child_name,
                    LoRALinear(
                        base_layer=child,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    ),
                )
            else:
                _replace(child, full_name)

    _replace(model)

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
```

---



## 7. 这个文件的工作方式



### `LoRALinear`

它把原始 `nn.Linear` 包起来：

- 原始层保留
- 原始层参数冻结
- 新增两个小线性层：`lora_A` 和 `lora_B`
- 训练时只更新这两个小层



### `inject_lora_into_model`

它会递归扫描模型：

- 找到名字包含指定关键词的 `nn.Linear`
- 把这些层替换为 `LoRALinear`

例如你想给这些层加 LoRA：

- `to_q`
- `to_k`
- `to_v`
- `to_o`
- `to_g`
- `fc1`
- `fc2`

那就传这些关键词进去。

---



## 8. 修改训练入口：`models/rfd3/src/rfd3/train.py`

你当前的训练脚本里，在这里会构建模型：

```python
trainer.construct_model()
trainer.construct_optimizer()
trainer.construct_scheduler()
```

LoRA 注入应该放在：

- `construct_model()` 之后
- `construct_optimizer()` 之前

这样 optimizer 才会只接收到 LoRA 参数。

### 需要插入的代码

```python
    trainer.construct_model()

    # ==============================================================
    # LoRA fine-tuning setup
    # ==============================================================
    if cfg.get("lora", None) and cfg.lora.enabled:
        from rfd3.peft_lora import inject_lora_into_model, count_trainable_parameters

        model = trainer.state["model"]

        model.diffusion_module = inject_lora_into_model(
            model.diffusion_module,
            target_keywords=cfg.lora.target_keywords,
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            freeze_all=True,
        )

        if cfg.lora.apply_to_token_initializer:
            model.token_initializer = inject_lora_into_model(
                model.token_initializer,
                target_keywords=cfg.lora.target_keywords,
                r=cfg.lora.rank,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                freeze_all=True,
            )

        trainable, total = count_trainable_parameters(model)
        ranked_logger.info(
            f"LoRA enabled: trainable params={trainable:,} / total params={total:,} "
            f"({100.0 * trainable / total:.4f}%)"
        )

    trainer.construct_optimizer()
    trainer.construct_scheduler()
```

---



## 9. 训练入口改动后的效果

改完之后，训练逻辑会变成：

1. 正常构建 RFD3 模型
2. 加载 checkpoint 或随机初始化权重
3. 把指定线性层替换成 LoRA 版本
4. 冻结主干，只训练 LoRA 参数
5. 构建 optimizer
6. 开始训练

---



## 10. 推荐新增配置项

你可以在 `single_structure.yaml` 里加入下面这个配置段：

```yaml
lora:
  enabled: false
  rank: 8
  alpha: 16
  dropout: 0.05
  apply_to_token_initializer: false
  target_keywords:
    - to_q
    - to_k
    - to_v
    - to_o
    - to_g
    - fc1
    - fc2
```

默认先关闭：

```yaml
enabled: false
```

等你真正想用 LoRA 时，再打开：

```yaml
enabled: true
```

---



## 11. 更推荐的做法：新建一个 LoRA 实验配置

为了不改动原来的 `single_structure.yaml`，建议你新建一个实验配置：

```text
models/rfd3/configs/experiment/lora_finetune.yaml
```

内容可以这样写：

```yaml
# @package _global_

defaults:
  - single_structure
  - _self_

name: single-structure-lora

lora:
  enabled: true
  rank: 8
  alpha: 16
  dropout: 0.05
  apply_to_token_initializer: false
  target_keywords:
    - to_q
    - to_k
    - to_v
    - to_o
    - to_g
    - fc1
    - fc2
```

这样你就可以直接用：

```bash
python models/rfd3/src/rfd3/train.py \
  experiment=lora_finetune \
  paths.data.monomer_distillation_parquet_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.monomer_distillation_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.design_benchmark_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.log_dir="/home/zzj/protein/foundry/logs/train" \
  logger=csv \
  ckpt_path="/home/zzj/protein/foundry/models/rfd3/rfd3_latest.ckpt"
```

如果你想从随机权重开始，就把 `ckpt_path` 去掉，或者保持 `single_structure.yaml` 里的 `ckpt_path: null`。

---



## 12. LoRA 参数怎么选



### 推荐起步配置

```yaml
rank: 8
alpha: 16
dropout: 0.05
```

含义是：

- `rank=8`：参数量比较小，适合先试
- `alpha=16`：常见设置，通常是 `2 * rank`
- `dropout=0.05`：轻微正则化



### 更省显存

```yaml
rank: 4
alpha: 8
dropout: 0.0
```



### 更强一点

```yaml
rank: 16
alpha: 32
dropout: 0.05
```

---



## 13. 先给哪些层加 LoRA 最合适

建议按这个顺序来：

### 第一轮只加

```yaml
- to_q
- to_v
- to_o
```

这是最稳的起点。

### 第二轮再加

```yaml
- to_k
- to_g
```



### 第三轮再加

```yaml
- fc1
- fc2
```

如果你发现 LoRA 提升不明显，再逐步扩展到更多层。

---



## 14. 如何检查 LoRA 是否真的生效

你需要确认两件事：

### 14.1 是否只有 LoRA 参数在训练

你可以在注入后打印：

```python
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name, p.shape)
```

如果配置正确，你应该主要看到：

- `lora_A.weight`
- `lora_B.weight`

而不是大量原始主干权重。

### 14.2 参数量是否明显下降

`count_trainable_parameters(model)` 会输出：

- 可训练参数量
- 总参数量

你应该看到训练参数只占总参数的一小部分。

---



## 15. 训练命令如何改

你现在的命令可以继续作为基础，只需要把实验改成 LoRA 版。

### 方式 A：继续使用 `single_structure`，通过命令行覆盖 LoRA 配置

前提是你已经把 `lora:` 段写进 `single_structure.yaml`。

命令可以是：

```bash
python models/rfd3/src/rfd3/train.py \
  experiment=single_structure \
  lora.enabled=true \
  lora.rank=8 \
  lora.alpha=16 \
  lora.dropout=0.05 \
  lora.apply_to_token_initializer=false \
  lora.target_keywords='[to_q,to_k,to_v,to_o,to_g,fc1,fc2]' \
  paths.data.monomer_distillation_parquet_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.monomer_distillation_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.design_benchmark_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.log_dir="/home/zzj/protein/foundry/logs/train" \
  logger=csv
```



### 方式 B：单独建 `lora_finetune` 实验配置

这样命令更短：

```bash
python models/rfd3/src/rfd3/train_lora.py \
  experiment=lora_finetune \
  paths.data.monomer_distillation_parquet_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.monomer_distillation_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.data.design_benchmark_data_dir="/home/zzj/protein/foundry/single_structure_data" \
  paths.log_dir="/home/zzj/protein/foundry/logs/train" \
  logger=csv
```

---



## 16. 你这个场景下的推荐实践

你现在的目标看起来是：

- 基于 `single_structure` 进行快速微调
- 数据量很小
- 想省显存
- 可能还想保留原模型的通用能力

所以我建议：

### 第一版

- 只改 `diffusion_module`
- 只给 `to_q / to_v / to_o` 加 LoRA
- `rank=8`
- `alpha=16`
- `dropout=0.05`
- 不动 `token_initializer`



### 如果效果不足

再把范围扩大到：

- `to_k`
- `to_g`
- `fc1`
- `fc2`

---



## 17. 关于“参考 single_structure 配置进行修改”的建议

你现在的 `single_structure.yaml` 已经很适合作为基底，所以建议这样做：

- 保留原始数据路径参数
- 保留原始训练节奏参数
- 只额外增加一个 `lora:` 配置段
- 训练时再用一个新的实验配置或命令行开关把 LoRA 打开

这样最干净，也最容易回退。

---



## 18. 推荐的最终目录改动



### 新增

```text
models/rfd3/src/rfd3/peft_lora.py
models/rfd3/configs/experiment/lora_finetune.yaml   # 可选
```



### 修改

```text
models/rfd3/src/rfd3/train.py
models/rfd3/configs/experiment/single_structure.yaml # 可选，加入 lora 配置段
```

---



## 19. 一句话版操作步骤

如果你只想记住最重要的流程，就是：

1. 在 RFD3 训练入口里，模型构建后加 LoRA 注入
2. 冻结原模型参数
3. 只训练 LoRA 参数
4. 先只给 attention 的 `to_q/to_v/to_o` 加 LoRA
5. 用 `single_structure` 配置跑一个小实验验证 loss 能不能下降

---



## 20. 最后给你的建议

如果你是第一次做这个，建议不要一步到位改太多。

最稳的路线是：

- 先做一个最小版 LoRA
- 只针对 `diffusion_module`
- 只改少数几个线性层
- 先跑通训练，再逐步扩大范围

这样最不容易出问题。

如果你愿意，下一步我可以继续帮你把这份文档对应的**实际代码补丁**也写出来，直接按文件给你列成可复制的修改版。