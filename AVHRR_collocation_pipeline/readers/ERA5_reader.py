from __future__ import annotations

import numpy as np
import netCDF4 as nc
import AVHRR_collocation_pipeline.utils as utils


def collocate_ERA5_precip(
    df,
    ERA5_meta_by_year,
    varname: str = "tp",
    out_col: str | None = None,
    scale: float = 1000.0,      # tp: meters -> mm
    time_offset_seconds: int = 0,  # <-- NEW: use +3600 to mimic OLD pipeline
):
    """
    Collocate ERA5 hourly data (one file per year) to df.

    Uses df['scan_hour_unix'] if present (recommended).
    Otherwise falls back to nearest-hour rounding on scan_line_times.

    time_offset_seconds:
        0      -> NEW behavior (nearest-hour)
        +3600  -> OLD behavior (nearest-hour then +1 hour)
    """
    if out_col is None:
        out_col = f"ERA5_{varname}"

    # -----------------------
    # Time (nearest hour unix)
    # -----------------------
    if "scan_hour_unix" in df.columns:
        t_hour = df["scan_hour_unix"].to_numpy().astype("int64")
    else:
        t = df["scan_line_times"].to_numpy().astype("int64")
        t_hour = ((t + 1800) // 3600) * 3600

    # Apply optional offset (OLD pipeline used +1 hour for ERA5/IMERG)
    if time_offset_seconds != 0:
        t_hour = t_hour + np.int64(time_offset_seconds)

    # Year per row (after offset!)
    years = (t_hour.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970).astype(np.int32)

    # -----------------------
    # Spatial indices per row
    # -----------------------
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()

    out = np.full(len(df), np.nan, dtype="float32")

    for yr in np.unique(years):
        meta = ERA5_meta_by_year.get(int(yr))
        if meta is None:
            continue

        m_year = (years == yr)

        ix, iy = utils.index_finder(lon[m_year], lat[m_year], meta["lon"], meta["lat"])
        good_xy = (ix >= 0) & (iy >= 0)
        if not np.any(good_xy):
            continue

        rows_year = np.where(m_year)[0]
        rows_good = rows_year[good_xy]

        # Exact match on hourly unix
        t_sub = t_hour[m_year][good_xy]
        tidx = np.searchsorted(meta["time_unix"], t_sub, side="left")
        ok_t = (
            (tidx >= 0)
            & (tidx < len(meta["time_unix"]))
            & (meta["time_unix"][tidx] == t_sub)
        )
        if not np.any(ok_t):
            continue

        rows = rows_good[ok_t]
        ix2 = ix[good_xy][ok_t]
        iy2 = iy[good_xy][ok_t]
        tidx2 = tidx[ok_t]

        lon_sort = meta["lon_sort_index"]

        with nc.Dataset(meta["file"]) as ds:
            var = ds[varname]  # (time, lat, lon_org)

            for t_unique in np.unique(tidx2):
                mm = (tidx2 == t_unique)
                rr = rows[mm]
                xs = ix2[mm]
                ys = iy2[mm]

                arr = var[int(t_unique), :, :]   # lon still 0..360
                arr = arr[:, lon_sort]           # reorder to -180..180
                vals = arr[ys, xs]

                out[rr] = (vals * scale).astype("float32")

    df2 = df.copy()
    df2[out_col] = out
    return df2