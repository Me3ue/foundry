这是从完整 PDB 镜像里抽出来的最小子集，供 RFdiffusion3 论文实验使用。
目录结构遵循 RCSB 约定：{PDB ID 的中间两位}/{PDB ID}.cif.gz

在服务器上把路径指过来即可：
    export PDB_MIRROR_PATH=<这个目录的绝对路径>
    # 或在 hydra 里覆盖： paths.data.pdb_data_dir=<这个目录>

共 3 个结构，334.8 KB。明细见 mirror_subset_manifest.csv。
