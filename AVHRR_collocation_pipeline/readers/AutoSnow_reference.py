# AVHRR_collocation_pipeline/readers/AutoSnow_reference.py
"""
AutoSnow reference loading utilities.

Builds:
  - date_to_file: {"YYYY-MM-DD": "/path/to/...YYYYJJJ...tif"}
  - raster geometry needed for fast lon/lat -> row/col indexing:
      transform, width, height, crs

Assumes: one GeoTIFF per day, WGS84 grid.
Typical filename:
  gmasi_snowice_reproc_v003_2022360_0.1deg_wgs.tif
"""

from __future__ import annotations

import os
import re
import datetime as dt
from typing import Dict, Any

import rasterio


_AS_RE = re.compile(
    r"^gmasi_snowice_reproc_v\d+_(\d{4})(\d{3})_0\.1deg_wgs\.tif$"
)


def _date_from_autosnow_filename(fname: str) -> str:
    """
    Extract YYYY + JJJ from filename and return "YYYY-MM-DD".
    """
    m = _AS_RE.match(fname)
    if not m:
        raise ValueError(f"Not a valid AutoSnow filename: {fname}")
    year = int(m.group(1))
    jjj = int(m.group(2))
    d = dt.datetime(year, 1, 1) + dt.timedelta(days=jjj - 1)
    return d.strftime("%Y-%m-%d")


def load_AutoSnow_reference(autosnow_dir: str) -> Dict[str, Any]:
    """
    Scan directory for AutoSnow daily GeoTIFFs and load grid metadata once.
    """
    fnames = [f for f in os.listdir(autosnow_dir) if f.endswith(".tif")]
    fnames = [f for f in fnames if _AS_RE.match(f)]
    if not fnames:
        raise FileNotFoundError(
            f"No AutoSnow GeoTIFFs found in {autosnow_dir} matching pattern:\n"
            f"  gmasi_snowice_reproc_v###_YYYYJJJ_0.1deg_wgs.tif"
        )

    fnames = sorted(fnames)
    paths = [os.path.join(autosnow_dir, f) for f in fnames]

    date_to_file = {_date_from_autosnow_filename(os.path.basename(p)): p for p in paths}

    # Read raster geometry from one sample file
    with rasterio.open(paths[0]) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    return {
        "date_to_file": date_to_file,
        "transform": transform,
        "width": width,
        "height": height,
        "crs": crs,
        # your old logic: values >= 200 are invalid
        "invalid_ge": 200.0,
    }


__all__ = ["load_AutoSnow_reference"]