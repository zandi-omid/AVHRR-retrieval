#%%
from __future__ import annotations

from pathlib import Path
import gc
import time

import torch
import xarray as xr

from AVHRR_collocation_pipeline.readers.AutoSnow_reference import load_AutoSnow_reference
from AVHRR_collocation_pipeline.readers.MERRA2_reference import load_MERRA2_reference

from AVHRR_collocation_pipeline.retrievers.collocate_and_reproj import AVHRRProcessor
from AVHRR_collocation_pipeline.retrievers.retrieve_and_reproj import AVHRRHybridRetriever
from AVHRR_collocation_pipeline.retrievers.reproject import reproject_vars_polar_to_wgs
from AVHRR_collocation_pipeline.retrievers.back_to_L2 import AVHRRBackToL2

from pytorch_retrieve.architectures import load_model

import matplotlib.pyplot as plt
import numpy as np

#%%
# ============================================================
# 0) HARD-CODED SETTINGS (from your config)
# ============================================================

# ---- paths ----
AVHRR_DIR = Path("/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019_subset")
MERRA2_DIR = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/MERRA2/merra2_archive_19800101_20250831"
AUTOSNOW_DIR = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AutoSnow/autosnow_in_geotif"
OUT_DIR = Path("/xdisk/behrangi/omidzandi/retrieved_maps/2019_subset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- grid ----
GRID_RES = 0.25
LAT_THRESH_NH = 45.0
LAT_THRESH_SH = -45.0
LAT_TS_NH = 70.0
LAT_TS_SH = -71.0

# ---- DL ----
CKPT_PATH = "/xdisk/behrangi/omidzandi/DL_Simon_codes/avhrr_retrievals/checkpoints/AVHRR_efficient_net_v2_pt_1_45_poleward_SH_ERA5_multi_node_keep_all_fp32-v1.ckpt"
TILE_SIZE = 1536
OVERLAP = 64

# ---- variables ----
AVH_VARS = ["cloud_probability", "temp_11_0um_nom", "temp_12_0um_nom"]
MERRA2_VARS = ["TQV", "T2M"]
INPUT_VARS = ["cloud_probability", "temp_11_0um_nom", "temp_12_0um_nom", "TQV", "T2M", "AutoSnow"]

# ---- output behavior ----
OUT_GRID = "wgs"   # "wgs" or "polar"
ENABLE_WGS_OUTPUT = False   # if False, we'll still write a file, but only polar groups

WRITE_VARS_NH = ["retrieved_precip_q80"]
WRITE_VARS_SH = ["retrieved_precip_q70"]
RENAME_VARS_NH = {"retrieved_precip_q80": "retrieved"}
RENAME_VARS_SH = {"retrieved_precip_q70": "retrieved"}

#%%
# ============================================================
# 1) Pick ONE orbit file (interactive)
# ============================================================

all_files = sorted(AVHRR_DIR.glob("*.nc"))
print(f"Found {len(all_files)} AVHRR files in {AVHRR_DIR}")
assert len(all_files) > 0, "No AVHRR files found."

# Pick one (edit this line however you like)
avh_file = Path(
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019/"
    "clavrx_NSS.GHRR.M2.D19045.S0556.E0739.B6393940.SV.hirs_avhrr_fusion.level2.nc"
)

assert avh_file.exists(), f"AVHRR file not found: {avh_file}"

orbit_tag = avh_file.stem
print("Selected orbit:", orbit_tag)
print("Path:", avh_file)

# ============================================================
# 2) Device + model
# ============================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = load_model(CKPT_PATH).to(device).eval()

retriever = AVHRRHybridRetriever(
    model=model,
    device=device,
    input_vars=INPUT_VARS,
    tile_size=TILE_SIZE,
    overlap=OVERLAP,
    out_grid_resolution_deg=GRID_RES,
    lat_ts_nh=LAT_TS_NH,
    lat_ts_sh=LAT_TS_SH,
)


# ============================================================
# 3) Load reference metadata ONCE
# ============================================================

print("Loading MERRA2 reference...")
merra2_meta = load_MERRA2_reference(MERRA2_DIR)

print("Loading AutoSnow reference...")
autosnow_meta = load_AutoSnow_reference(AUTOSNOW_DIR)

processor = AVHRRProcessor(
    grid_res=GRID_RES,
    lat_thresh_nh=LAT_THRESH_NH,
    lat_thresh_sh=LAT_THRESH_SH,
    lat_ts_nh=LAT_TS_NH,
    lat_ts_sh=LAT_TS_SH,
    nodata=-9999.0,
    merra2_meta=merra2_meta,
    autosnow_meta=autosnow_meta,
)


# ============================================================
# 4) Stage-1: orbit -> (NH polar ds, SH polar ds) + TB11_WGS
# ============================================================

t0 = time.time()
ds_polar, tb11_wgs, x_vec_global, y_vec_global = processor.process_orbit(
    avh_file=str(avh_file),
    avh_vars=AVH_VARS,
    input_vars=INPUT_VARS,
    merra2_vars=MERRA2_VARS,
)
print(f"Stage-1 done in {(time.time()-t0):.1f}s")

def plot_all_vars(ds, hemisphere="NH"):
    vars_ = list(ds.data_vars)
    n = len(vars_)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False
    )

    for ax, var in zip(axes.flat, vars_):
        arr = ds[var].values
        im = ax.imshow(arr, origin="upper", cmap="turbo")
        ax.set_title(f"{hemisphere}: {var}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # turn off unused axes
    for ax in axes.flat[len(vars_):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

ds_nh_polar = ds_polar["NH"]
ds_sh_polar = ds_polar["SH"]

print("NH polar dataset:", ds_nh_polar)
print("SH polar dataset:", ds_sh_polar)

plot_all_vars(ds_nh_polar, hemisphere="NH")
plot_all_vars(ds_sh_polar, hemisphere="SH")

#%%
# ============================================================
# 5) Stage-2: load inputs + GPU inference
# ============================================================

x_nh, tb12_nh, xvec_nh, yvec_nh = retriever.load_group_inputs_from_dataset(ds_nh_polar)
x_sh, tb12_sh, xvec_sh, yvec_sh = retriever.load_group_inputs_from_dataset(ds_sh_polar)

t0 = time.time()
preds_nh = retriever.gpu_predict_tiled_multiquantile(x_nh)
preds_nh = retriever.clean_precip(preds_nh, min_val=0.0, max_val=50.0, drizzle=0.001)
preds_nh = retriever.mask_preds_with_tb12(preds_nh, tb12_nh)

preds_sh = retriever.gpu_predict_tiled_multiquantile(x_sh)
preds_sh = retriever.clean_precip(preds_sh, min_val=0.0, max_val=50.0, drizzle=0.001)
preds_sh = retriever.mask_preds_with_tb12(preds_sh, tb12_sh)
print(f"Stage-2 done in {(time.time()-t0):.1f}s")

# free large tensors
del x_nh, x_sh
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()


# ============================================================
# 6) Build NH/SH datasets for writing (polar first)
# ============================================================

# full multiquantile
ds_nh = xr.Dataset(
    {
        "retrieved_precip_mean": (("y", "x"), preds_nh["mean"]),
        "retrieved_precip_q70":  (("y", "x"), preds_nh["q70"]),
        "retrieved_precip_q75":  (("y", "x"), preds_nh["q75"]),
        "retrieved_precip_q80":  (("y", "x"), preds_nh["q80"]),
    },
    coords={"x": xvec_nh, "y": yvec_nh},
)

ds_sh = xr.Dataset(
    {
        "retrieved_precip_mean": (("y", "x"), preds_sh["mean"]),
        "retrieved_precip_q70":  (("y", "x"), preds_sh["q70"]),
        "retrieved_precip_q75":  (("y", "x"), preds_sh["q75"]),
        "retrieved_precip_q80":  (("y", "x"), preds_sh["q80"]),
    },
    coords={"x": xvec_sh, "y": yvec_sh},
)

print("Built ds_nh polar vars:", list(ds_nh.data_vars))
print("Built ds_sh polar vars:", list(ds_sh.data_vars))


# ============================================================
# 7) Optional: polar -> WGS reprojection + TB11 masking
# ============================================================

def _filter_and_rename(ds: xr.Dataset, keep: list[str] | None, rename: dict[str, str] | None) -> xr.Dataset:
    rename = rename or {}
    if keep is not None:
        ds = ds[keep]
    # validate collisions
    new_names = list(rename.values())
    if len(new_names) != len(set(new_names)):
        raise ValueError(f"Duplicate output names after renaming: {new_names}")
    return ds.rename(rename)

if OUT_GRID == "wgs":
    # keep only requested vars before reproj to reduce work
    ds_nh = _filter_and_rename(ds_nh, WRITE_VARS_NH, RENAME_VARS_NH)
    ds_sh = _filter_and_rename(ds_sh, WRITE_VARS_SH, RENAME_VARS_SH)

    # reproject (returns WGS datasets)
    ds_nh = reproject_vars_polar_to_wgs(
        {v: ds_nh[v].values for v in ds_nh.data_vars},
        hemisphere="NH",
        x_vec=xvec_nh,
        y_vec=yvec_nh,
        grid_resolution_deg=GRID_RES,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
        nodata=float("nan"),
        tag=f"{orbit_tag}_NH",
    )

    ds_sh = reproject_vars_polar_to_wgs(
        {v: ds_sh[v].values for v in ds_sh.data_vars},
        hemisphere="SH",
        x_vec=xvec_sh,
        y_vec=yvec_sh,
        grid_resolution_deg=GRID_RES,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
        nodata=float("nan"),
        tag=f"{orbit_tag}_SH",
    )

    # apply TB11 swath mask (keeps only where AVHRR observed)
    ds_nh = retriever.mask_ds_with_tb11_wgs(ds_nh, tb11_wgs, x_vec_global, y_vec_global)
    ds_sh = retriever.mask_ds_with_tb11_wgs(ds_sh, tb11_wgs, x_vec_global, y_vec_global)


# ============================================================
# 8) Write NetCDF with NH/SH groups
# ============================================================

out_nc = OUT_DIR / f"{orbit_tag}_retrieved_{OUT_GRID}.nc"
if out_nc.exists():
    out_nc.unlink()

retriever.write_orbit_netcdf(
    out_path=out_nc,
    ds_nh=ds_nh,
    ds_sh=ds_sh,
    var_scales={"retrieved": 0.005},   # if you renamed to "retrieved"
    default_scale=0.005,
)

print("Wrote:", out_nc)


# ============================================================
# 9) Optional: attach back to L2 swath
# ============================================================

DO_ATTACH_TO_L2 = False

if DO_ATTACH_TO_L2:
    retrieved_names = sorted(set(list(ds_nh.data_vars) + list(ds_sh.data_vars)))
    l2_writer = AVHRRBackToL2(retrieved_var_names=retrieved_names)
    out_l2 = out_nc.with_name(f"{orbit_tag}_L2.nc")
    l2_writer.attach_to_orbit(
        raw_orbit_path=str(avh_file),
        ds_nh=ds_nh,
        ds_sh=ds_sh,
        out_path=out_l2,
    )
    print("Wrote L2-attached file:", out_l2)
# %%
