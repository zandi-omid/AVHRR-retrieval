#%% Load the CSV and inspect

import pandas as pd

fn = "../../eval_2019_retrieved_vs_refs_50_poleward_per_orbit.csv"
df = pd.read_csv(fn)

print(df.head())
print(df.describe())

#%% Drop invalid / missing values

df_clean = df.copy()

df_clean["NH_KGE_summary"] = pd.to_numeric(df_clean["NH_KGE_summary"], errors="coerce")
df_clean["SH_KGE_summary"] = pd.to_numeric(df_clean["SH_KGE_summary"], errors="coerce")

#%% 10 worst orbits — NH

worst_nh = (
    df_clean
    .dropna(subset=["NH_KGE_summary"])
    .sort_values("NH_KGE_summary", ascending=True)
    .head(10)
)

print("\n=== 10 WORST NH KGE ORBITS ===")
print(worst_nh[["orbit_name", "NH_KGE_summary"]])

#%% 10 worst orbits — SH

worst_sh = (
    df_clean
    .dropna(subset=["SH_KGE_summary"])
    .sort_values("SH_KGE_summary", ascending=True)
    .head(10)
)

print("\n=== 10 WORST SH KGE ORBITS ===")
print(worst_sh[["orbit_name", "SH_KGE_summary"]])
# %%
