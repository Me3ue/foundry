# RFdiffusion3 论文实验复现报告

- 生成时间：2026-09-27T09:56:43
- 设计总数：0
- 实验条件数：0

## 论文正文实验 ↔ 复现结果对照

| 实验 | 论文口径指标 | 论文报告 | 本次复现 |
|---|---|---|---|
| exp1_unconditional | designability@1.5A (AF3, 8 MPNN seqs) | 98% of designs have >=1 sequence folding within 1.5 A | 未运行 |
| exp2_ppi | AF3 pass rate (min PAE<=1.5 & binder pTM>=0.8 & RMSD<2.5A) | RFD3 优于 RFD1 的 4/5 个靶点；聚类(TM 0.6)后 5/5；平均 8.2 vs 1.4 个成功簇 | 未运行 |
| exp3_dna | DNA-aligned protein Cα RMSD < 5 A 通过率 | 单体 8.67%；二体 6.67%（rigid/diffused 两设置平均） | 未运行 |
| exp4_small_molecule | AF3 pass rate (bb RMSD<=1.5 & lig RMSD<=5 & PAE<=1.5 & ipTM>=0.8) | RFD3 在 4/4 配体上优于 RFdiffusionAA（rigid 配体） | 未运行 |
| exp5_enzyme_ame | 每案例 100 骨架 × 8 LigandMPNN 序列；Chai-1 的 5 个 diffusion sample 中至少 1 个满足 motif 全原子 RMSD<1.5Å 且配体与骨架无 clash | 41 个 AME 案例中 37 个优于 RFD2 (90%)；>4 residue islands: 15% vs 4%；协议见 RFdiffusion2 论文 (Nat Methods 2025) 的 AME benchmark 一节 | 未运行 |
| exp7_conditioning | 氢键比例 | 小分子 26.67% -> 32.67% (hbond 条件) -> 36.67% (再加 CFG) | 未运行 |

## 每个条件的详细统计
