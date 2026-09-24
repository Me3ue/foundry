# 仓库自带的 benchmark 定义清单

由 `02_extract_repo_benchmarks.py` 自动生成。

## 验证集配置 -> 需要的数据文件

| 配置文件 | benchmark 名 | 数据文件 | 只取这些 key | 评估频率 |
|---|---|---|---|---|
| `val/bcov_ppi_easy_medium.yaml` | bcov-ppi-easy-medium | `(继承/内联)` |  | 1 |
| `val/design_validation_base.yaml` | ??? | `???` |  | ??? |
| `val/dna_binder_design5.yaml` | dna_binder_design | `<design_benchmark_data_dir>/dna_binder.json` |  | 1 |
| `val/dna_binder_long.yaml` | dna_binder_design | `<design_benchmark_data_dir>/tests/dna.json` | ['7rte_sequence_only', '7rte_with_structure'] | 10 |
| `val/dna_binder_short.yaml` | dna_binder_design | `<design_benchmark_data_dir>/rfd3/tests/test_data/dna.json` | ['7rte_sequence_only', '7rte_with_structure'] | 1 |
| `val/indexed.yaml` | indexed-design | `<design_benchmark_data_dir>/indexed.json` |  | 8 |
| `val/mcsa_41.yaml` | woodys-benchmark | `<design_benchmark_data_dir>/mcsa_41.json` |  | 16 |
| `val/mcsa_41_short_rigid.yaml` | rigid-ligand-enzymes | `<design_benchmark_data_dir>/mcsa_41_short_rigid_new.json` |  | 1 |
| `val/pdb_holdout.yaml` |  | `(继承/内联)` |  |  |
| `val/ppi_inference.yaml` | ppi_inference | `???` |  |  |
| `val/sm_binder_hbonds.yaml` | sm_binder_hbonds-design | `<design_benchmark_data_dir>/sm_binder_hbonds.json` | ['FAD', 'IAI', 'OQO', 'SAM'] | 5 |
| `val/sm_binder_hbonds_short.yaml` | sm_binder_hbonds-design-short | `<design_benchmark_data_dir>/sm_binder_hbonds_sampled.json` | ['FAD_1', 'FAD_2', 'FAD_3', 'IAI_1', 'IAI_2', 'IAI_3'] | 1 |
| `val/unconditional.yaml` | unconditional-design | `<design_benchmark_data_dir>/monomer.json` |  | 1 |
| `val/unconditional_deep.yaml` | unconditional-design-deep | `<design_benchmark_data_dir>/unconditional_deep.json` |  | 8 |
| `val/unindexed.yaml` | unindexed-design | `<design_benchmark_data_dir>/unindexed.json` |  |  |
| `val/val_examples/bcov_ppi_easy_medium_with_ori.yaml` | （内联 PPI 定义） | `内联在 YAML 里` |  |  |
| `val/val_examples/bcov_ppi_easy_medium_with_ori_spoof_helical_bundle.yaml` | （内联 PPI 定义） | `内联在 YAML 里` |  |  |
| `val/val_examples/bcov_ppi_easy_medium_with_ori_varying_lengths.yaml` | （内联 PPI 定义） | `内联在 YAML 里` |  |  |
| `val/val_examples/bpem_ori_hb.yaml` | （内联 PPI 定义） | `内联在 YAML 里` |  |  |

## 从训练过滤器反推出的 holdout 测试集

- 训练时间截断：`deposition_date < 2024-12-16`

- 明确排除的 PDB ID（= 论文的 held-out 靶点）：

  - `7rte` — DNA 结合物靶点
  - `7m5w` — DNA 结合物靶点
  - `7n5u` — DNA 结合物靶点
  - `3di3` — il7ra (3di3)
  - `5o45` — pdl1 (5o45)
  - `1z92` — il2ra (1z92)
  - `2gy5` — tie2 (2gy5)
  - `4zxb` — insulinr (4zxb)

- 来源：models/rfd3/configs/datasets/train/pdb/rfd3_train_interface.yaml、models/rfd3/configs/datasets/train/pdb/rfd3_train_pn_unit.yaml
