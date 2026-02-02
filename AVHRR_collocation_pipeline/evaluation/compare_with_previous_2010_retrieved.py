import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from hydroeval import kge as KGE

# --------------------------------------------------
# Paths
# --------------------------------------------------
retrieved_dir = Path("/ra1/pubdat/AVHRR_CloudSat_proj/SIMON_CODES/retrieved_maps/2010_pt25_45_poleward")
collocated_dir = Path("/scratch/omidzandi/evaluation/2010_collocated_with_ref_preci")

# --------------------------------------------------
# Helper: extract orbit key
# --------------------------------------------------
def orbit_key_from_retrieved(p):
    return p.name.replace("retrieved_", "").replace(".nc", "")

def orbit_key_from_collocated(p):
    return p.name.replace("_collocated_wgs.nc", "")

# --------------------------------------------------
# Build file maps
# --------------------------------------------------
retrieved_files = {orbit_key_from_retrieved(p): p for p in retrieved_dir.glob("retrieved_*.nc")}
collocated_files = {orbit_key_from_collocated(p): p for p in collocated_dir.glob("*_collocated_wgs.nc")}

common_orbits = sorted(set(retrieved_files) & set(collocated_files))
print(f"Found {len(common_orbits)} matching orbits")

# --------------------------------------------------
# Loop & compute KGE
# --------------------------------------------------
records = []

for orbit in common_orbits:
    ref_path = retrieved_files[orbit]
    sim_path = collocated_files[orbit]

    for hemi in ["NH", "SH"]:
        ds_ref = xr.open_dataset(ref_path, group=hemi)
        ds_sim = xr.open_dataset(sim_path, group=hemi)

        if hemi == "NH":
            ref = ds_ref["retrieved_precip_q80"].sel(y=ds_ref.y >= 45.0)
            sim = ds_sim["retrieved"].sel(y=ds_sim.y >= 45.0)
        else:
            ref = ds_ref["retrieved_precip_q70"].sel(y=ds_ref.y <= -45.0)
            sim = ds_sim["retrieved"].sel(y=ds_sim.y <= -45.0)

        # Align grids safely
        ref, sim = xr.align(ref, sim, join="inner")

        # Flatten & mask NaNs
        ref_vals = ref.values.ravel()
        sim_vals = sim.values.ravel()
        mask = np.isfinite(ref_vals) & np.isfinite(sim_vals)

        if mask.sum() < 50:
            kge_val = np.nan
        else:
            kge_val = KGE(sim_vals[mask], ref_vals[mask])[0]

        records.append({
            "orbit": orbit,
            "hemisphere": hemi,
            "KGE": kge_val
        })

        ds_ref.close()
        ds_sim.close()

# --------------------------------------------------
# Results table
# --------------------------------------------------
df_kge = pd.DataFrame(records)
df_kge