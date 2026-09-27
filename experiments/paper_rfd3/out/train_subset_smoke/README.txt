RFD3 微调训练子集
=================

行数   : 128
pdb_id : 328（含验证集）
体积   : 460.8 MB
seed   : 42（固定 seed，重跑结果一致）

目录结构（AtomWorks 按 {base_dir}/{pdb_id[1:3]}/{pdb_id}.cif.gz 查找）：
  metadata/interfaces_df.parquet
  metadata/pn_units_df.parquet
  mirror/{xx}/{pdb_id}.cif.gz
  subset_manifest.csv
  env.subset.sh

服务器上跑：
  source env.subset.sh
  bash models/rfd3/scripts/run_nmf_zkp_pdb_sweep.sh

注意
----
1. 子集内 cluster size 会重算，采样权重随之变化。baseline 与各 NMF 变体
   必须用同一子集，横向对比才有效；绝对 lDDT 不能与完整训练集结果直接比。
2. 验证集（pdb_holdout）用到的结构已一并打入，不要删。
3. 每子集统计：
   - interfaces_df.parquet: 过滤后 5,632,469 行 -> 保留 128 行 / 128 pdb_id / 119 cluster
   - pn_units_df.parquet: 过滤后 3,258,883 行 -> 保留 128 行 / 128 pdb_id / 113 cluster
