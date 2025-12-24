# AVHRR_collocation_pipeline/readers/AutoSnow_reader.py
"""
AutoSnow Reader and Collocation Tools
-------------------------------------

Fast collocation of AVHRR pixels with AutoSnow daily GeoTIFFs.

- Opens each daily GeoTIFF once per unique scan_date
- Vectorized lon/lat -> row/col using raster transform
- Masks values >= invalid_ge as NaN

Author: Omid Zandi
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import rasterio


__all__ = ["collocate_AutoSnow"]


def _lonlat_to_rowcol(transform, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized lon/lat -> (row, col) for north-up grids (no rotation).
    Works with typical WGS84 regular grids.

    row = (y0 - lat) / yres
    col = (lon - x0) / xres
    """
    # Affine: x = a*col + c, y = e*row + f (with b=d=0 for north-up)
    a = transform.a
    e = transform.e
    c = transform.c
    f = transform.f

    col = (lon - c) / a
    row = (lat - f) / e  # e is usually negative, so this still gives correct row

    return np.floor(row).astype(np.int32), np.floor(col).astype(np.int32)


def collocate_AutoSnow(
    df,
    AutoSnow_meta: Dict[str, Any],
    date_col: str = "scan_date",
    out_col: str = "AutoSnow",
    debug: bool = False,
):
    """
    Collocate AutoSnow daily values onto the AVHRR DataFrame.

    Requires df columns:
      - lon, lat
      - date_col (default "scan_date" as "YYYY-MM-DD")

    Returns df copy with new column out_col (float32).
    """
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not found in df columns. Create it (via utils.add_time_columns) first.")
    if "lon" not in df.columns or "lat" not in df.columns:
        raise KeyError("df must contain 'lon' and 'lat' columns.")

    date_to_file = AutoSnow_meta["date_to_file"]
    transform = AutoSnow_meta["transform"]
    width = int(AutoSnow_meta["width"])
    height = int(AutoSnow_meta["height"])
    invalid_ge = float(AutoSnow_meta.get("invalid_ge", 200.0))

    LON = df["lon"].to_numpy(dtype="float64")
    LAT = df["lat"].to_numpy(dtype="float64")
    DATES = df[date_col].astype(str).to_numpy()

    out = np.full(len(df), np.nan, dtype="float32")

    # row/col for all points once (fast)
    rows, cols = _lonlat_to_rowcol(transform, LON, LAT)
    on_grid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    uniq_dates = np.unique(DATES)

    if debug:
        print("\nAutoSnow debug:")
        print("  df scan_date sample:", DATES[0], type(DATES[0]))
        print("  df scan_date unique (first 5):", uniq_dates[:5])
        print("  meta keys (first 5):", list(date_to_file.keys())[:5])
        print("  on_grid fraction:", float(np.mean(on_grid)))

    for d in uniq_dates:
        tif = date_to_file.get(d)
        if tif is None:
            continue

        mask = (DATES == d) & on_grid
        if not np.any(mask):
            continue

        r = rows[mask]
        c = cols[mask]

        with rasterio.open(tif) as src:
            arr = src.read(1)  # 2D
            vals = arr[r, c].astype("float32")

        # apply invalid mask (your rule: >=200 -> NaN)
        vals = np.where(vals >= invalid_ge, np.nan, vals)
        out[mask] = vals

    df2 = df.copy()
    df2[out_col] = out
    return df2