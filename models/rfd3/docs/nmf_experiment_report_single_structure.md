# RFD3 替换式 NMF 实验报告

> 适用场景：汇报、组会、实验复盘、论文前期结果整理。
>
> 数据来源：`logs/train_nmf/sweep_nmf_final_single_structure_2026-08-06_11-18-52/comparison_table.md`、`comparison_table.csv`，以及对应训练日志与源码实现。

---

## 0. 一页式结论

这组实验的核心结论可以先概括为三句话：

1. **替换式 NMF 已经成功嵌入 RFD3 的训练流程，并且能够稳定跑完整个单结构微调/过拟合流程。**
2. **从层类型上看，FFN / transition / projection / head 这类“特征混合型”线性层更适合做 NMF 替换；而 `to_q` / `to_k` / `to_v` 这类直接定义注意力几何的层，不适合作为第一批目标。**
3. **在当前单结构实验中，NMF 让 `mean_lddt` 变高，但 `loss` 也显著变差；因此它不是“整体无条件变好”，而是“结构指标提升与优化目标退化并存”的 mixed 结果。**

更具体地说：

- baseline 的 best `lddt` 为 `0.2600`，best `loss` 为 `0.3912`
- `ffn_core` 的 best `lddt` 提升到 `0.4745`，但 best `loss` 升到 `2.5040`
- `proj_mix` 的 best `lddt` 进一步升到 `0.5812`，best `loss` 升到 `3.3290`
- `head_plus_proj` 的 best `lddt` 最高，为 `0.5910`，但 best `loss` 也最高，为 `3.7016`

所以如果严格按实验结果判断：

- **若目标是结构质量指示器 `lddt`，NMF 是有收益的；**
- **若目标是训练目标 `loss`，当前 NMF 设置是明显有代价的；**
- **若要找“最稳妥”的 NMF 插入位置，`ffn_core` 是当前最保守、代价最小的一档；**
- **若要追求更高的 `lddt`，`proj_mix` / `head_plus_proj` 更激进，但损失更大。**

---

## 1. 实验背景与问题定义

RFD3 是一个用于蛋白设计/结构生成的扩散式模型，核心难点不只在于“能不能生成”，还在于“生成的结构是否稳定、合理、与训练目标一致”。

传统全参数微调会直接改动整个模型，带来几个问题：

- 参数量大，训练成本高；
- 容易破坏预训练模型的已有能力；
- 很难定位“到底是哪类层”对结果贡献最大。

替换式 NMF 的动机是：

- 用**低秩、非负**的方式替换部分线性层；
- 让模型保留“特征混合”的基本功能；
- 把“哪些层适合 NMF、哪些层不适合”这件事做成可控的层级消融实验。

本实验不是简单问“要不要用 NMF”，而是更细地问：

1. **RFD3 的哪些层适合 NMF 替换？**
2. **替换后，模型的结构指标和优化目标分别发生了什么变化？**
3. **如何在现有源码上以最小侵入方式嵌入 NMF？**
4. **在单结构过拟合设置下，NMF 的收益和代价分别是什么？**

---

## 2. RFD3 的结构视角：哪些层“像矩阵分解”，哪些层“不像”

### 2.1 从功能上看，RFD3 的线性层可以粗分为四类

#### A. 特征混合 / FFN / transition 类
这类层主要负责：

- 通道扩展 / 压缩；
- 局部特征重组；
- 把已经聚合过的信息重新映射到新的表征空间。

这类层通常更接近“矩阵分解友好”的场景，因为它们的作用更像“把已有特征做低维组合”。

典型名字包括：

- `transition_*`
- `process_c`
- `process_z`
- `process_s_*`
- `pair_mlp`
- `fc1` / `fc2`
- `process_single_*`

#### B. Projection / mixing 类
这类层通常位于信息流的中间或输出路径上，作用是：

- 将中间表示映射到不同子空间；
- 在多个 token / feature 之间做融合；
- 为后续解码或几何更新提供通道重组。

典型名字包括：

- `process_pll`
- `project_pll`
- `upcast.project`
- `downcast.project`
- `process_n`
- `to_r_update`

#### C. 输出头 / head 类
这类层通常更接近任务输出：

- `sequence_head`
- `to_o`
- 与结构或序列最终预测相关的映射层

它们对最终输出的影响较大，替换后可能显著改变校准和损失。

#### D. 注意力几何类
这类层直接决定 attention score 的形成：

- `to_q`
- `to_k`
- `to_v`

它们往往含有较强的 signed weight 语义，直接进行严格非负约束替换风险很大。

---

### 2.2 为什么 FFN / projection 层更适合 NMF

NMF 本质上假设某个权重矩阵可以被分解成非负低秩因子：

\[
W \approx UV, \quad U \ge 0,\; V \ge 0
\]

从结构上看，这更符合“**特征重组 / 通道混合**”而不是“**相似度打分 / 几何方向性**”的层：

- FFN/transition 层往往是“把已有表示重新混合”；
- projection 层往往是“把语义压缩/映射到另一个空间”；
- head 层则是“输出前的最后映射”。

而 `q/k/v` 类层更像是“定义关系本身”，它们对权重符号和分布更敏感。

因此，在实验设计上，先从 **FFN / transition / projection / head** 开始，是更稳妥的顺序。

---

## 3. NMF 的数学原理：从经典分解到替换式参数化

### 3.1 经典 NMF 的基本形式

给定一个非负矩阵 \(W\in\mathbb{R}_{+}^{m\times n}\)，经典 NMF 希望找到两个非负矩阵：

\[
U\in\mathbb{R}_{+}^{m\times r},\quad V\in\mathbb{R}_{+}^{r\times n}
\]

使得：

\[
W \approx UV
\]

其中：

- \(r\) 是 rank，通常远小于 \(m,n\)；
- 目标函数一般是 Frobenius norm：

\[
\min_{U,V\ge 0} \|W - UV\|_F^2
\]

经典 NMF 的优势是：

- 低秩压缩；
- 部件化表示（parts-based representation）；
- 可解释性较强；
- 适合正值、组合型信号。

---

### 3.2 本实验中的“替换式 NMF”不是离线分解，而是可训练重参数化

本实验并不是先离线求一个固定的 NMF，再塞回模型；而是把一个线性层直接替换成一个低秩非负参数化模块。

对原线性层：

\[
y = xW + b
\]

替换成：

\[
\hat{y} = x\,\alpha\,\mathrm{softplus}(U_{raw})\,\mathrm{softplus}(V_{raw}) + b
\]

其中：

- \(U_{raw}, V_{raw}\) 是可训练参数；
- `softplus` 保证因子非负；
- \(\alpha\) 是缩放系数；
- `eps` 用于数值稳定。

也就是说，本实验中的 NMF 更准确地说是：

> **“严格非负的低秩线性层重参数化”**

而不是教科书里那种单独对矩阵做 ALS / multiplicative update 的离线 NMF。

---

### 3.3 为什么用 `softplus`

代码里并没有直接把参数裁成非负，而是通过：

\[
U = \mathrm{softplus}(U_{raw}) + \epsilon
\]

\[
V = \mathrm{softplus}(V_{raw}) + \epsilon
\]

来实现。

这样做的优点是：

- 梯度连续，优化更平滑；
- 参数天然可训练；
- 不需要显式投影到非负正交域；
- 避免 hard clamp 带来的训练抖动。

缺点是：

- 它不是严格意义上的“传统 NMF 求解器”；
- 解释上更像“非负低秩参数化”而不是“完全求解的矩阵分解”。

---

### 3.4 参数量为什么能降很多

原始线性层参数量大约是：

\[
mn + m\;\text{(bias if any)}
\]

替换后参数量约为：

\[
r(m+n) + m\;\text{(bias if any)}
\]

当 \(r \ll \min(m,n)\) 时，参数量会显著下降。

这也是为什么在结果里你会看到：

- baseline trainable params 约为 1.68e8
- NMF 版本 trainable params 约为 1.65e7

也就是训练参数减少了一个数量级左右。

---

## 4. 源代码层面如何嵌入 NMF

### 4.1 核心实现文件

这次实验的关键实现位于：

- `models/rfd3/src/rfd3/nmf.py`
- `models/rfd3/src/rfd3/train_lora.py`
- `models/rfd3/configs/experiment/nmf_single_structure.yaml`
- `models/rfd3/configs/experiment/nmf_ffn_core_single_structure.yaml`
- `models/rfd3/configs/experiment/nmf_proj_mix_single_structure.yaml`
- `models/rfd3/configs/experiment/nmf_head_plus_proj_single_structure.yaml`

---

### 4.2 `nmf.py` 做了什么

`nmf.py` 主要负责三件事：

1. 定义替换后的线性层 `ReplacementNMFLinear`
2. 递归遍历模型并替换目标模块
3. 记录每个被替换层的参数统计与模块名字

核心逻辑是：

- 找到满足 `target_keywords` 的子模块；
- 只替换 2D weight 的线性层；
- 用 `ReplacementNMFLinear` 包装它；
- 冻结原模型参数，只保留 `u_raw` / `v_raw` 可训练；
- 生成 `NMFReplacementRecord`，便于汇总分析。

其 forward 形式可以理解为：

```text
x ──> softplus(U_raw) @ softplus(V_raw) ──> linear(x, W_nmf)
```

其中 `W_nmf = alpha * U @ V`。

---

### 4.3 `train_lora.py` 如何接入 NMF

虽然文件名里还叫 `train_lora.py`，但这里实际上已经支持 NMF 作为另一种 PEFT / replacement 方案。

训练过程中的关键步骤是：

1. 读取 Hydra 配置；
2. 构造 trainer 与 model；
3. 在 `trainer.fit()` 前，检查 `cfg.nmf.enabled`；
4. 若启用，则调用 `inject_nmf_into_model(...)`；
5. 训练开始前就完成替换，因此优化器只看到新的参数结构；
6. 训练结束后写出 `run_summary.json`。

这样做的好处是：

- 训练代码改动小；
- NMF、LoRA、baseline 可以共享同一训练框架；
- 便于做 sweep 和消融比较；
- 结果天然可归档。

---

### 4.4 配置文件如何控制替换范围

当前实验用的是 `target_keywords` 方式来决定替换哪些层。

#### 4.4.1 `nmf_single_structure.yaml`

推荐的第一版目标：

- `fc1`
- `fc2`
- `to_o`
- `to_g`

它的设计理念是：

- 优先替换最像 FFN 的层；
- 先避开 attention 几何层；
- 做一个风险最小的替换起点。

#### 4.4.2 `nmf_ffn_core_single_structure.yaml`

这一版更偏保守，主要覆盖：

- `transition_post_token`
- `transition_post_atom`
- `transition_1`
- `process_s_init`
- `process_z_init`
- `transition_2`
- `process_c`
- `process_s_trunk`
- `process_single_l`
- `process_single_m`
- `process_z`
- `pair_mlp`

这是当前实验里“最稳”的 NMF 组。

#### 4.4.3 `nmf_proj_mix_single_structure.yaml`

这一版扩大到投影和混合层：

- `process_pll`
- `project_pll`
- `to_r_update`
- `upcast.project`
- `downcast.project`
- `process_n`
- `sequence_head`

它比 `ffn_core` 更激进，覆盖更多中后段混合层和输出相关层。

#### 4.4.4 `nmf_head_plus_proj_single_structure.yaml`

这一版进一步聚焦：

- `process_pll`
- `project_pll`
- `to_r_update`
- `sequence_head`

它是“投影 + head”的紧凑版本。

---

## 5. 实验配置：本次实验到底是怎么跑的

### 5.1 数据设置

本次 sweep 使用的是单结构数据实验，基于 `single_structure.yaml` 的逻辑：

- 训练对象是单个结构样本
- 验证集也指向同一个 `monomer.json`
- 目的是快速观察模型在小数据上是否能稳定过拟合/适配

在实验日志里，对应例子是：

- `uncond_92mer`

这类设置的好处是：

- 训练信号非常密集；
- 不容易被数据集复杂度掩盖；
- 适合判断某种替换策略是否“把模型本身搞坏了”。

---

### 5.2 优化与训练节奏

这次 sweep 的特征包括：

- 加载预训练 checkpoint：`/media/zzj/Data/rfd3_latest.ckpt`
- `reset_optimizer: true`
- `rank: 8`
- `alpha: 1.0`
- `eps: 1e-6`
- `nonnegative_param: softplus`
- 每个 epoch 例子数较少，便于频繁验证
- `validate_every_n_epochs: 1`

这意味着实验更适合看趋势，而不是追求大规模泛化结论。

---

### 5.3 sweep 的四个 setting

本次最终比较的是四种配置：

1. `baseline`
2. `ffn_core`
3. `proj_mix`
4. `head_plus_proj`

其中 baseline 是原始 checkpoint，不启用 NMF；其余三个是不同层组的替换 NMF。

---

## 6. 结果总览

### 6.1 关键结果表

| setting | best_val_lddt | best_val_loss | best_epoch | final_epoch | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.2600 | 0.3912 | 581 | 589 | 参考基线 |
| ffn_core | 0.4745 | 2.5040 | 576 | 589 | mixed，但最保守 |
| proj_mix | 0.5812 | 3.3290 | 587 | 589 | lddt 更高，但 loss 更差 |
| head_plus_proj | 0.5910 | 3.7016 | 589 | 589 | lddt 最高，但 loss 也最高 |

---

### 6.2 相对 baseline 的变化

#### `ffn_core`
- best `lddt`: `+0.2145`，变好
- best `loss`: `+2.1129`，变差
- final `lddt`: `+0.2145`，变好
- final `loss`: `+2.1129`，变差

#### `proj_mix`
- best `lddt`: `+0.3212`，变好
- best `loss`: `+2.9379`，变差
- final `lddt`: `+0.3212`，变好
- final `loss`: `+2.9379`，变差

#### `head_plus_proj`
- best `lddt`: `+0.3310`，变好
- best `loss`: `+3.3104`，变差
- final `lddt`: `+0.3310`，变好
- final `loss`: `+3.3104`，变差

---

### 6.3 严谨解释：为什么不能简单说“变好了”

如果只看 `lddt`，NMF 显然有提升；但如果看训练目标 `loss`，则所有 NMF 版本都显著变差。

因此最严谨的说法是：

> **NMF 在当前设置下改善了结构质量相关指标，但显著破坏了优化目标，因此整体结果是 mixed，而不是单纯的改进。**

这点很重要，因为它说明：

- NMF 没有把模型“完全救活”；
- 它可能改变了某种结构偏好，使 `lddt` 变高；
- 但对损失函数的匹配程度下降了很多。

---

## 7. 层级分析：哪些层更适合 NMF，哪些层不适合

### 7.1 当前实验支持的结论

#### 结论 1：`ffn_core` 是最稳妥的 NMF 起点

理由：

- 只替换最像 MLP / transition 的部分；
- 结构变化最小；
- loss 增幅相对最小；
- 仍然能让 `lddt` 提升。

因此，如果你问“要先在哪些层上试 NMF”，答案是：

> **先试 `ffn_core`，再考虑扩大到更宽的 projection/head 范围。**

---

#### 结论 2：projection / mixing 层可以替换，但更激进

`proj_mix` 和 `head_plus_proj` 说明：

- 这些层也能做非负低秩替换；
- 并且 `lddt` 甚至更高；
- 但 loss 的退化更明显。

这意味着它们更适合：

- 消融研究；
- 机制探索；
- 对“结构分数 vs 优化目标”做权衡分析。

不太适合直接作为“默认最优方案”。

---

#### 结论 3：attention 几何层目前仍然不建议作为第一批替换对象

虽然本轮没有直接实验 `to_q` / `to_k` / `to_v`，但从配置设计和结构功能上都可以推断：

- 它们对权重符号更敏感；
- 它们定义的是注意力几何，而不是简单通道组合；
- 非负约束可能过强，容易损伤 score 分布。

所以，这类层更适合作为后续高风险 ablation，而不是第一轮默认替换。

---

### 7.2 进一步的层级建议

按照“适合 NMF 的优先级”排序，可以总结成：

1. **最适合**：FFN / transition / process 类
2. **较适合**：projection / mixing 类
3. **可试但要谨慎**：output head / sequence head
4. **第一轮不建议**：attention 的 `q/k/v`

这也与实验配置的演化顺序一致：

- 先 `ffn_core`
- 再 `proj_mix`
- 再 `head_plus_proj`

---

## 8. 结果背后的可能原因

### 8.1 为什么 `lddt` 会变高

可能的解释包括：

- 非负低秩约束让模型偏向更平滑、更保守的表征；
- 在单结构过拟合场景中，这种约束有时会减少极端扰动；
- `lddt` 评价的结构相似性可能对这种“稳定化”更敏感。

换句话说，NMF 可能起到了某种正则化效果。

---

### 8.2 为什么 `loss` 会变差

`loss` 升高说明模型对训练目标的拟合变差。可能原因包括：

- 非负约束太强，表达能力不足；
- 低秩近似无法完整复原原权重的符号结构；
- output / projection 层被替换后，模型输出分布被重新标定；
- 训练目标中包含的多项损失对这些替换更敏感。

也就是说，NMF 在这个任务里更像“改变了模型偏好”，而不是“无损压缩”。

---

### 8.3 为什么 `proj_mix` / `head_plus_proj` 的 `lddt` 更高但 `loss` 更差

这通常意味着：

- 更靠近输出端的层替换，直接影响最终生成结果；
- 结构相似性指标可能受益于更强的约束；
- 但输出分布、序列恢复、能量/置信度相关项被破坏得更多。

因此，这些层可以提升某些“看上去更像”的指标，但代价是训练目标的偏离更大。

---

## 9. 如何进一步增强实验结论的可信度

如果后面你想把分析做得更完整，建议在 `train_lora.py` 里进一步补这些指标：

### 9.1 推荐增加的 summary 指标

- `best_epoch_seq_recovery`
- `best_epoch_token_lvl_sequence_loss`
- `best_epoch_lp_norm`
- `best_epoch_valid_t_fraction`
- `final_epoch_seq_recovery`
- `final_epoch_token_lvl_sequence_loss`
- `final_epoch_lp_norm`
- `final_epoch_valid_t_fraction`

这些指标已经在训练日志中出现过，但如果写进 summary，就能让 sweep 报告更完整。

---

### 9.2 为什么这些指标有帮助

因为当前的 `lddt` 和 `loss` 已经能告诉你“结果方向”，但还不足以解释“为什么”：

- `seq_recovery` 能看出序列恢复是否受影响；
- `token_lvl_sequence_loss` 能看出离散序列建模有没有变差；
- `lp_norm` 可以粗略反映参数变化幅度；
- `valid_t_fraction` 可以帮助判断采样/有效时间步的分布是否异常。

如果要做更像论文的机制分析，这些指标很值得补。

---

## 10. 最终结论

### 10.1 关于“哪些层适合 NMF”

基于当前实验，可以给出如下排序：

- **首选层**：`transition`、`process_*`、`pair_mlp`、`fc1` / `fc2`
- **次选层**：`process_pll`、`project_pll`、`process_n`、`to_r_update`
- **可试层**：`sequence_head`、部分输出 head
- **暂不建议第一轮尝试**：`to_q`、`to_k`、`to_v`

---

### 10.2 关于 NMF 的整体效果

在这次 single-structure sweep 中：

- NMF **提高了结构相关指标**；
- 但同时 **显著恶化了 loss**；
- 所以它不能被简单定义为“全面改进”。

更准确的说法是：

> **NMF 改变了模型的行为偏好，使结构指标改善，但优化目标退化；属于有明显权衡的 mixed 结果。**

---

### 10.3 当前最值得保留的结论

如果只保留一句适合汇报的话术，可以说：

> **RFD3 中最适合作为替换式 NMF 首批试验对象的是 FFN / transition / projection 类线性层；这些层在保留一定结构质量的同时，最不容易破坏训练流程。注意力几何层不应作为第一轮替换对象。当前实验表明，NMF 能抬高 `lddt`，但会明显牺牲 `loss`，因此它更适合作为结构偏好调控手段，而不是当前设置下的直接最优方案。**

---

## 11. 复现实验的关键文件

- 训练源码：`models/rfd3/src/rfd3/train_lora.py`
- NMF 实现：`models/rfd3/src/rfd3/nmf.py`
- 基础配置：`models/rfd3/configs/experiment/single_structure.yaml`
- NMF 配置：
  - `nmf_single_structure.yaml`
  - `nmf_ffn_core_single_structure.yaml`
  - `nmf_proj_mix_single_structure.yaml`
  - `nmf_head_plus_proj_single_structure.yaml`
- sweep 报告：`logs/train_nmf/sweep_nmf_final_single_structure_2026-08-06_11-18-52/comparison_table.md`

---

## 12. 一句话摘要

> 本实验表明：在 RFD3 上做替换式 NMF 是可行的，但它更像一种对模型表征风格的重塑，而不是无损压缩；FFN / transition / projection 类层最适合优先尝试，attention 几何层应谨慎处理；当前结果表现为 `lddt` 提升与 `loss` 退化并存的 mixed trade-off。
