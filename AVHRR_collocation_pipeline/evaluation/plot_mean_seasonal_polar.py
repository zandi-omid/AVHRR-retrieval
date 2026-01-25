#%%
import os
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


def plot_season_group_polar(
    season_file: str,
    *,
    group: str,
    lat_thresh: float = 50.0,
    var_list=("retrieved", "ERA5_tp", "IMERG_preci"),
    name_map=None,
    units: str = "Seasonal Mean Precipitation (mm/hr)",
    cmap: str = "jet",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    vmax_frac_of_max: float = 0.8,
    coast_color: str = "white",
    dpi: int = 250,
    figsize=(6.8, 7.2),
    ncols: int = 1,
    gridline_lon_step: int = 30,
    gridline_lat_step: int = 10,
):
    """
    Plot seasonal mean maps for one hemisphere group (NH/SH) in POLAR stereographic axes.

    Assumptions:
      - The file has groups "NH" and "SH"
      - Coordinates are named x (lon in degrees), y (lat in degrees)
      - Data variables are 2D (y,x)

    Notes:
      - Inputs are lon/lat => pcolormesh transform=PlateCarree().
      - Axes are NorthPolarStereo/SouthPolarStereo.
      - Auto-skips missing variables.
      - Shared colorbar.
      - vmax scaling: vmax = vmax_frac_of_max * max(all variables values) unless vmax is set.
    """
    group = group.upper()
    if group not in ("NH", "SH"):
        raise ValueError("group must be 'NH' or 'SH'")

    if name_map is None:
        name_map = {
            "retrieved": "Retrieved",
            "ERA5_tp": "ERA5",
            "IMERG_preci": "IMERG",
        }

    # ---- open ----
    ds = xr.open_dataset(season_file, group=group)

    # ---- poleward filter + projection ----
    if group == "NH":
        ds = ds.where(ds["y"] >= lat_thresh, drop=True)
        proj = ccrs.NorthPolarStereo()
        extent = [-180, 180, lat_thresh, 90]
        hemi_label = f"NH (≥{lat_thresh}°)"
        lat_locator = range(int(lat_thresh), 91, gridline_lat_step)
    else:
        ds = ds.where(ds["y"] <= -lat_thresh, drop=True)
        proj = ccrs.SouthPolarStereo()
        extent = [-180, 180, -90, -lat_thresh]
        hemi_label = f"SH (≤{-lat_thresh}°)"
        lat_locator = range(-90, int(-lat_thresh) + 1, gridline_lat_step)

    if ds.sizes.get("y", 0) == 0 or ds.sizes.get("x", 0) == 0:
        ds.close()
        raise RuntimeError(f"No data left after filtering for {group} at {lat_thresh}°")

    # ---- pick vars present ----
    present = [v for v in var_list if v in ds.data_vars]
    if not present:
        ds.close()
        raise RuntimeError(f"None of {list(var_list)} found in {season_file} group={group}")

    # ---- shared vmax from fraction of max across all vars ----
    vals = []
    for v in present:
        a = ds[v].values
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)

    if not vals:
        ds.close()
        raise RuntimeError("All selected variables are all-NaN after filtering.")

    vals_all = np.concatenate(vals)
    if vmax is None:
        raw_max = np.nanmax(vals_all)
        vmax = float(raw_max * vmax_frac_of_max) if np.isfinite(raw_max) and raw_max > 0 else 1.0

    # ---- layout ----
    nplots = len(present)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            figsize[0] * ncols,
            max(figsize[1], 3.2 * nrows),
        ),
        subplot_kw={"projection": proj},
        dpi=dpi,
    )

    # normalize axes to 1D list
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    # ---- plot ----
    im = None
    for i, v in enumerate(present):
        ax = axes[i]
        da = ds[v]

        # lon/lat grid => PlateCarree input transform
        im = ax.pcolormesh(
            da["x"], da["y"], da,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # features
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="#bcdff1")
        ax.add_feature(cfeature.COASTLINE, edgecolor=coast_color, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, edgecolor=coast_color, linewidth=0.3)

        # gridlines (no labels in polar stereo; labels get messy)
        gl = ax.gridlines(
            draw_labels=False,
            linestyle="--",
            linewidth=0.5,
            color="gray",
            alpha=0.6,
        )
        gl.xlocator = mticker.FixedLocator(list(range(-180, 181, gridline_lon_step)))
        gl.ylocator = mticker.FixedLocator(list(lat_locator))

        ax.set_title(name_map.get(v, v), fontsize=10, pad=6)

    # hide unused subplots
    for j in range(nplots, len(axes)):
        axes[j].set_visible(False)

    # ---- shared colorbar ----
    cax = fig.add_axes([0.20, 0.06, 0.60, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(units, fontsize=10)

    # title
    plt.suptitle(f"{hemi_label} — {Path(season_file).name}", fontsize=12, y=0.98)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.12, wspace=0.04, hspace=0.10)

    ds.close()
    plt.show()
    return fig

#%%

from pathlib import Path

TARGET_DIR = Path("/scratch/omidzandi/evaluation/seasonal_means_2019")

# If you only want files like "seasonal_mean_*.nc", use that:
season_files = sorted(TARGET_DIR.glob("seasonal_mean_*.nc"))

for f in season_files:
    try:
        plot_season_group_polar(
            str(f),
            group="NH",
            lat_thresh=50,
            var_list=("retrieved", "ERA5_tp", "IMERG_preci"),
            vmax_frac_of_max=0.9,
            ncols=1,  # use 2 if you later have 4 vars etc.
        )
    except Exception as e:
        print(f"[NH skip] {f.name}: {e}")

    try:
        plot_season_group_polar(
            str(f),
            group="SH",
            lat_thresh=50,
            var_list=("retrieved", "ERA5_tp", "IMERG_preci"),
            vmax_frac_of_max=0.9,
            ncols=1,
        )
    except Exception as e:
        print(f"[SH skip] {f.name}: {e}")
# %%
