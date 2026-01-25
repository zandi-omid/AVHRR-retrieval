#%%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
#%%

def plot_season_group(
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
    dpi: int = 200,
    figsize=(12, 5),
):
    """
    Plot seasonal mean maps for one hemisphere group (NH or SH) from a NetCDF
    with groups ("NH","SH"), using a *shared colorbar*.

    - Automatically skips variables not present (e.g., IMERG not yet in file).
    - Adds lat/lon tick labels via Cartopy gridlines.
    - Applies poleward filtering:
        NH: keep y >= lat_thresh
        SH: keep y <= -lat_thresh

    Parameters
    ----------
    season_file : str
        Path to seasonal mean NetCDF (with groups).
    group : {"NH","SH"}
        Hemisphere group to plot.
    lat_thresh : float
        Poleward threshold (50 means >=50 for NH, <=-50 for SH).
    var_list : tuple/list
        Variables to attempt to plot in order.
    name_map : dict or None
        Pretty names for titles. Defaults provided if None.
    vmin/vmax : float or None
        Shared color scaling. If vmax=None, computed from data via quantile.
    vmax_quantile : float
        Quantile for auto vmax if vmax is None.
    """

    group = group.upper()
    if group not in ("NH", "SH"):
        raise ValueError("group must be 'NH' or 'SH'")

    if name_map is None:
        name_map = {"retrieved": "Retrieved", "ERA5_tp": "ERA5", "IMERG_preci": "IMERG"}

    # ---- load group ----
    ds = xr.open_dataset(season_file, group=group)

    # ---- poleward filter ----
    if group == "NH":
        ds = ds.where(ds["y"] >= lat_thresh, drop=True)
        extent = [-180, 180, lat_thresh, 90]
        hemi_label = f"NH (≥{lat_thresh}°)"
    else:
        ds = ds.where(ds["y"] <= -lat_thresh, drop=True)
        extent = [-180, 180, -90, -lat_thresh]
        hemi_label = f"SH (≤{-lat_thresh}°)"

    if ds.sizes.get("y", 0) == 0 or ds.sizes.get("x", 0) == 0:
        ds.close()
        raise RuntimeError(f"No data left after filtering for {group} at {lat_thresh}°")

    # ---- choose variables that exist ----
    present = [v for v in var_list if v in ds.data_vars]
    if not present:
        ds.close()
        raise RuntimeError(f"None of {list(var_list)} found in {season_file} group={group}")

    # ---- shared color scaling ----
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
        vmax = float(np.nanmax(vals_all) * vmax_frac_of_max)

    proj = ccrs.PlateCarree()
    nrows = len(present)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(figsize[0], max(figsize[1], 2.2 * nrows)),
        subplot_kw={"projection": proj},
        dpi=dpi,
    )
    if nrows == 1:
        axes = [axes]

    im = None
    for i, v in enumerate(present):
        ax = axes[i]
        da = ds[v]

        im = ax.pcolormesh(
            da["x"], da["y"], da,
            transform=proj,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.coastlines(color=coast_color, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, edgecolor=coast_color, linewidth=0.35)
        ax.set_facecolor("lightgray")
        ax.set_extent(extent, crs=proj)

        title = name_map.get(v, v)
        ax.set_title(title, fontsize=10)

        # ---- lat/lon labels (gridlines) ----
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.6,
            linestyle="--",
        )
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        if i < nrows - 1:
            gl.bottom_labels = False

    # ---- shared colorbar (same style you used) ----
    cax = fig.add_axes([0.20, 0.06, 0.60, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(units, fontsize=10)

    plt.suptitle(f"{hemi_label} — {Path(season_file).name}", fontsize=12, y=0.98)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, hspace=0)

    ds.close()
    plt.show()
    return fig

#%%
SEASON_DIR = Path("/scratch/omidzandi/evaluation/seasonal_means_2019")

# pick which files to plot
# If you have only one file, pattern can still be "seasonal_mean_*.nc"
season_files = sorted(SEASON_DIR.glob("seasonal_mean_*.nc"))

print(f"Found {len(season_files)} seasonal files")
for f in season_files:
    print("Plotting:", f.name)

    # NH
    plot_season_group(
        str(f),
        group="NH",
        lat_thresh=50,
        var_list=("retrieved", "ERA5_tp", "IMERG_preci"),
        vmax_frac_of_max=0.9
    )

    # SH
    plot_season_group(
        str(f),
        group="SH",
        lat_thresh=50,
        var_list=("retrieved", "ERA5_tp", "IMERG_preci"),
        vmax_frac_of_max=0.9
    )

# %%
