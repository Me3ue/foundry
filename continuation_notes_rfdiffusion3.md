# RFdiffusion3 本轮对话工作总结与后续衔接说明

## 1. 本轮工作的目标

本轮工作的核心目标是：

- 对一组蛋白做**轻微扰动**后，再用 RFD3 诱导生成
- 让输入结构在**局部受限、全局保留**的前提下完成设计
- 分析哪些目标跑通、哪些失败，以及失败原因
- 把整个过程整理成可直接用于汇报/后续实验的文档
- 进一步补齐结构质量指标和结果分析

---

## 2. 已完成的主要文件

### 2.1 实验说明与结果报告

已创建并持续补充：

- `rfdiffusion3_perturbation_study.md`

这份文档已经包含：

- 实验目标
- 输入数据组织
- 轻微扰动定义
- 设计约束与 Variant 含义
- 模板生成策略
- 每个蛋白的扰动参数表
- 成功/失败统计
- 结构质量指标说明
- 图示建议
- 结论与后续建议

### 2.2 结构指标批量计算脚本

已创建：

- `compute_rfdiffusion3_structure_metrics.py`

用途：

- 自动读取统一模板
- 自动读取每个蛋白/variant 的输出结构
- 与原始输入结构做对比
- 计算：
  - `RMSD(CA)`
  - `RMSD(fixed core)`
  - `RMSD(diffuse)`
  - `TM-score`
  - `local_stability`
- 输出：
  - `logs/rfdiffusion3_batch/structure_metrics.csv`
  - `logs/rfdiffusion3_batch/structure_metrics.json`
  - `logs/rfdiffusion3_batch/structure_metrics.md`

### 2.3 总批处理脚本相关修复

对以下内容做过多轮修正：

- `run_rfdiffusion3_all.sh`
- `make_rfdiffusion3_unified_template.py`

已经修过的关键点包括：

- 删除错误的 `ccd_mirror_path=None`
- symmetry 路径统一改为 skip（因为缺少 `sym_transform`）
- 大蛋白窗口缩减
- `4JTA / 4UY2` 的更保守 fixed core
- `make_rfdiffusion3_unified_template.py` 的 `window` 变量修复

---

## 3. 现在这套实验的基本逻辑

### 3.1 输入是什么

最终喂给 RFD3 的不是原始复合物，而是：

- 单链 PDB：`rfdiffusion3_inputs/single_chain/<PDB>_<CHAIN>.pdb`

这些单链结构经过：

- 仅保留蛋白链
- 提取有效残基
- 划分 fixed core 与 diffuse 区
- 对大蛋白缩窗
- 设定 `partial_t`

### 3.2 轻微扰动是怎么实现的

不是直接给坐标加噪声，而是通过：

- `select_fixed_atoms`
- `select_unfixed_sequence`
- `partial_t`

来限制模型只在局部进行重建。

### 3.3 设计约束是怎么写的

主要写在两层：

1. **统一模板**
   - `rfdiffusion3_inputs/unified_templates/<PDB>.json`
   - 由 `make_rfdiffusion3_unified_template.py` 生成

2. **Variant 变体 JSON**
   - `logs/rfdiffusion3_batch/generated_json/<PDB>_<variant>.json`
   - 由 `run_rfdiffusion3_all.sh` 里的 `make_variant_json()` 生成

---

## 4. Variant 的含义与写法

### 4.1 Variant 总览

| Variant | 含义 |
|---|---|
| `baseline` | 基础设计条件，偏保守 |
| `fixed` | 强调固定核心，通常最稳 |
| `partial` | 更激进的局部扩散/重建 |
| `symmetry` | 对称设计路径，但当前被 skip |

### 4.2 实际约束写法

在 `run_rfdiffusion3_all.sh` 的 `make_variant_json()` 中：

- `select_fixed_atoms`：用 fixed core 的残基范围构造
- `select_unfixed_sequence`：用 diffuse 区构造连续区段
- `partial_t`：按 variant 再次调整
- `symmetry`：当前会被移除并标记 skip

### 4.3 当前实际逻辑

- `baseline`：把 `partial_t` 压到更保守的范围
- `fixed`：基本保留模板值
- `partial`：把 `partial_t` 抬高到更激进
- `symmetry`：`__skip__ = symmetry_requires_sym_transform`

---

## 5. 这批实验里已经观察到的结果

### 5.1 成功与失败概况

- 成功：24
- 失败：28

其中：

- 13 个 symmetry skip 不算真实失败
- 真正的失败主要分为：
  - OOM
  - 代表原子/虚拟原子断言失败
  - 结构输出缺失

### 5.2 成功的目标

主要成功的蛋白：

- `1ABR`
- `1DM0`
- `1QLX`
- `2AAI`
- `5O3L`
- `5OQV`
- `6A6B`
- `7UMQ`

### 5.3 失败的主要目标

主要失败目标：

- `1MDT`
- `3BTA`
- `4JTA`
- `4UY2`
- `5N0B`

失败原因：

- `1MDT / 3BTA / 5N0B`：CUDA OOM
- `4JTA / 4UY2`：`PadTokensWithVirtualAtoms` 阶段 `CB` 缺失相关错误

---

## 6. 结构质量指标已经算出了什么

### 6.1 `structure_metrics.csv` 中已有字段

- `pdb_id`
- `variant`
- `status`
- `reference`
- `generated`
- `chain_id`
- `fixed_core`
- `diffuse`
- `rmsd_ca`
- `rmsd_fixed_ca`
- `rmsd_diffuse_ca`
- `tm_score`
- `local_stability`
- `note`

### 6.2 指标含义

- `rmsd_ca`：全局 CA RMSD
- `rmsd_fixed_ca`：fixed core 的 CA RMSD
- `rmsd_diffuse_ca`：diffuse 区的 CA RMSD
- `tm_score`：整体 fold 保持程度
- `local_stability`：根据 RMSD + TM-score 推出的经验标签

### 6.3 结果趋势

目前观察到：

- fixed core 大多非常稳定
- diffuse 区通常变化更大
- `baseline` 因为 `partial_t` 被压低，往往更保守
- `partial` 更激进，变化更明显
- 部分样本（如 `4UY2/partial`）几乎高保真回收原结构

---

## 7. 当前你已经确认的重要事实

### 7.1 输入是单链结构

你输入给 RFD3 的最终是单链 PDB，而不是原始多链复合物。

### 7.2 微扰不是坐标加噪

微扰主要通过：

- fixed core / diffuse 划分
- `partial_t`
- 让模型只在局部区域进行受限重建

### 7.3 `baseline` 反而更保守

这是因为当前脚本里：

- `baseline` 会把 `partial_t` 压到 `2.0`
- `fixed` 往往保留模板原值
- `partial` 会抬高到 `4.0` 或更高

所以名字和“直觉”并不完全对应。

---

## 8. 目前最值得继续做的事

如果新对话要继续，可以优先接下面几件事：

1. **把结构指标表写进总报告**
   - 用 `structure_metrics.csv` 的结果补强实验结论

2. **补图**
   - 输入 vs 输出结构对照图
   - fixed core 高亮图
   - diffuse 区高亮图

3. **处理 `structure_metrics.csv` 的展示问题**
   - 如果需要，可以再生成更干净的 summary 表
   - 按蛋白、按 variant 汇总成功率和平均指标

4. **继续分析结果是否符合“轻微扰动生成”目标**
   - 哪些蛋白属于高保真回收
   - 哪些属于中度重建
   - 哪些完全失败或不适配当前管线

---

## 9. 继续对话时可直接使用的上下文

如果你要在新对话里继续，建议直接说明：

- 你正在做的是 RFdiffusion3 的轻微扰动诱导生成实验
- 当前已有：
  - `rfdiffusion3_perturbation_study.md`
  - `compute_rfdiffusion3_structure_metrics.py`
  - `logs/rfdiffusion3_batch/structure_metrics.csv`
- 重点是继续把结果分析、结构指标、图示补齐
- 不再需要重新解释基础流程，直接从现有结果继续往下做即可

---

## 10. 最简短的工作摘要

一句话总结当前工作：

> 已完成对一组蛋白的轻微扰动式 RFD3 设计流程搭建、模板生成、批量运行、失败分析与结构质量评估脚本开发，并整理成可继续扩展的实验报告与结果表。
