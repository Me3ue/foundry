import os
import pandas as pd
print("1")
rows = [
    {
        "example_id": "test_1",
        "pdb_id": "100d",
        "assembly_id": 1,
        "deposition_date": "2020-01-01",
        "resolution": 2.0,
        "num_polymer_pn_units": 2,
        "method": "X-RAY DIFFRACTION",
        "cluster": "test_cluster_1",
        "n_prot": 2,
        "n_nuc": 0,
        "n_ligand": 0,
        "n_peptide": 0,
        "pn_unit_1_iid": 0,
        "pn_unit_2_iid": 1,
        "pn_unit_1_non_polymer_res_names": None,
        "pn_unit_2_non_polymer_res_names": None,
        "is_inter_molecule": True,
        "all_pn_unit_iids_after_processing": [0, 1],
        "involves_loi": False,
    }
]

df = pd.DataFrame(rows)

# 类型整理，避免 parquet 里出现奇怪的 object 推断问题
df["deposition_date"] = pd.to_datetime(df["deposition_date"])

out_dir = "/home/zzj/protein/foundry/data/rfd3_finetune/metadata"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "interfaces_df.parquet")

df.to_parquet(out_path, index=False)
print(f"saved to {out_path}")
print(df.dtypes)