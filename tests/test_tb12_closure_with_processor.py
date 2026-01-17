#%%
from pathlib import Path
import numpy as np
import xarray as xr

from AVHRR_collocation_pipeline.retrievers.collocate_and_reproj import AVHRRProcessor

# --- paths ---
raw_l2 = Path(
        "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019/"
        "clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2.nc"
    )
out_l2 = Path(
        "/xdisk/behrangi/omidzandi/retrieved_maps/test/"
        "__clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2__with_retrievals_L2.nc"
    )

# use SAME settings you used in the main pipeline
proc = AVHRRProcessor(
    grid_res=0.25,
    lat_thresh_nh=45.0,
    lat_thresh_sh=-45.0,
    lat_ts_nh=70.0,
    lat_ts_sh=-71.0,
    nodata=-9999.0,
    merra2_meta=None,      # we don't need collocation here
    autosnow_meta=None,
)

AVH_VARS = ["temp_11_0um_nom", "temp_12_0um_nom"]

# ---------- 1) grid TB12 from original orbit ----------
df_raw, x_vec_raw, y_vec_raw = proc.load_orbit_df(str(raw_l2), AVH_VARS)
grid_raw = proc.build_var_grids(df_raw, x_vec_raw, y_vec_raw, ["temp_12_0um_nom"])
tb12_wgs_orig = grid_raw["temp_12_0um_nom"]

# ---------- 2) grid TB12 from L2-with-retrievals ----------
df_out, x_vec_out, y_vec_out = proc.load_orbit_df(str(out_l2), AVH_VARS)
grid_out = proc.build_var_grids(df_out, x_vec_out, y_vec_out, ["temp_12_0um_nom"])
tb12_wgs_from_l2 = grid_out["temp_12_0um_nom"]

# ---------- 3) compare ----------
diff = tb12_wgs_from_l2 - tb12_wgs_orig

print("TB12 closure test via AVHRRProcessor:")
print("  original grid shape:", tb12_wgs_orig.shape)
print("  from L2 grid shape :", tb12_wgs_from_l2.shape)
print("  max |Δ|            :", float(np.nanmax(np.abs(diff))))
print("  mean |Δ|           :", float(np.nanmean(np.abs(diff))))

#%%

import matplotlib.pyplot as plt
import numpy as np

# Assuming you already have:
# tb12_wgs_orig       # from raw orbit gridding
# tb12_wgs_from_l2    # from L2-with-retrievals gridding

diff = tb12_wgs_from_l2 - tb12_wgs_orig

vmin = np.nanmin(tb12_wgs_orig)
vmax = np.nanmax(tb12_wgs_orig)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Original TB12
im0 = axes[0].imshow(
    tb12_wgs_orig,
    origin="upper",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set_title("TB12 original (WGS)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# 2) TB12 rebuilt from L2
im1 = axes[1].imshow(
    tb12_wgs_from_l2,
    origin="upper",
    vmin=vmin,
    vmax=vmax,
)
axes[1].set_title("TB12 from L2 (back + regrid)")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# 3) Difference
im2 = axes[2].imshow(
    diff,
    origin="upper",
    # Tight symmetric range for visual check
    vmin=-0.05,
    vmax=0.05,
)
axes[2].set_title("Difference (L2 - original)")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")

plt.tight_layout()
plt.show()
# or:
# plt.savefig("tb12_closure_test.png", dpi=200)

# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from AVHRR_collocation_pipeline.retrievers.collocate_and_reproj import AVHRRProcessor

# ------------------------------------------------------------------
# Paths & config
# ------------------------------------------------------------------
l2_with_retrievals = Path(
    "/xdisk/behrangi/omidzandi/retrieved_maps/test/"
    "clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2__with_retrievals_L2.nc"
)

GRID_RES     = 0.25
LAT_THRESH_NH = 45.0
LAT_THRESH_SH = -45.0
LAT_TS_NH     = 70.0
LAT_TS_SH     = -71.0

# We only care about the retrieved precip here, but AVHRRProcessor
# expects an avh_vars list – include at least one “real” AVHRR var too.
AVH_VARS = [
    "temp_11_0um_nom",
    "temp_12_0um_nom",
    "retrieved_precip_mean",   # 👈 the one we want to grid
]

# ------------------------------------------------------------------
# 1. Set up processor and read the orbit as a DataFrame
# ------------------------------------------------------------------
processor = AVHRRProcessor(
    grid_res=GRID_RES,
    lat_thresh_nh=LAT_THRESH_NH,
    lat_thresh_sh=LAT_THRESH_SH,
    lat_ts_nh=LAT_TS_NH,
    lat_ts_sh=LAT_TS_SH,
    nodata=-9999.0,
    merra2_meta=None,    # not needed here
    autosnow_meta=None,  # not needed here
)

df_l2, x_vec, y_vec = processor.load_orbit_df(str(l2_with_retrievals), AVH_VARS)

# ------------------------------------------------------------------
# 2. Grid retrieved_precip_mean using the same machinery as TB
# ------------------------------------------------------------------
var_grids = processor.build_var_grids(
    df_l2,
    x_vec,
    y_vec,
    varnames=["retrieved_precip_mean"],
)

precip_grid = var_grids["retrieved_precip_mean"]  # shape (ny, nx)

print(
    "retrieved_precip_mean grid:",
    precip_grid.shape,
    "min=", np.nanmin(precip_grid),
    "max=", np.nanmax(precip_grid),
)

# ------------------------------------------------------------------
# 3. Plot with imshow (same style as TB fields)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
im = plt.imshow(precip_grid, origin="upper")
plt.title("retrieved_precip_mean (gridded via AVHRRProcessor)")
plt.colorbar(im, shrink=0.8, label="precip units")
plt.tight_layout()
plt.show()


# %%
