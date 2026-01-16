#!/usr/bin/env python

from pathlib import Path
import xarray as xr

# >>>>>> EDIT THIS TO POINT TO ONE ORBIT YOU KNOW <<<<<<
avh_file = Path(
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019/"
    "clavrx_NSS.GHRR.M1.D19003.S0538.E0636.B3265556.SV.hirs_avhrr_fusion.level2.nc"
)

print(f"Opening: {avh_file}")
ds = xr.open_dataset(avh_file)

print("\n=== DATASET SUMMARY ===")
print(ds)

print("\n=== DIMS ===")
for k, v in ds.dims.items():
    print(f"  {k}: {v}")

print("\n=== COORDS ===")
for k, v in ds.coords.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\n=== FIRST 10 DATA VARS ===")
for i, (k, v) in enumerate(ds.data_vars.items()):
    if i >= 10:
        break
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

ds.close()