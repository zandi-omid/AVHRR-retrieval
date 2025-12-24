"""
ERA5 reference loading utilities (0.25° total precipitation).

This module does NOT hardcode any paths.
It ONLY provides functions to load ERA5 metadata:

    * per-year NetCDF file path
    * lon / lat grid (reordered to -180..180)
    * time coordinates (datetime64 + UNIX seconds)
    * ymdh strings ("YYYY-MM-DD HH") for fast matching
    * lon sort index to reorder fields from 0..360 → -180..180

Typical filenames (in one directory):

    Total_precip_2007.nc
    Total_precip_2008.nc
    ...
    Total_precip_2020.nc
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any

import numpy as np
import xarray as xr


def _parse_year_from_filename(path: str) -> int:
    """
    Extract the year from a filename like 'Total_precip_2010.nc'.
    """
    base = os.path.basename(path)
    m = re.search(r"(\d{4})", base)
    if not m:
        raise ValueError(f"Could not find 4-digit year in ERA5 file name: {base}")
    return int(m.group(1))


def _load_single_ERA5_file(path: str) -> Dict[str, Any]:
    """
    Load lon/lat/time metadata from a single ERA5 file.

    Handles both:
        - ERA5 files with 'time' coordinate
        - ERA5 CDS-downloaded files with 'valid_time'
    """
    ds = xr.open_dataset(path)

    # ----- LON reorder 0..360 → -180..180 -----
    lon_org = ds["longitude"].values
    lon_shifted = ((lon_org + 180.0) % 360.0) - 180.0
    lon_sort_index = np.argsort(lon_shifted)
    lon = lon_shifted[lon_sort_index].astype("float32")

    # ----- LAT -----
    lat = ds["latitude"].values.astype("float32")

    # ----- TIME -----
    if "time" in ds:
        time_raw = ds["time"].values
    elif "valid_time" in ds:
        time_raw = ds["valid_time"].values
    else:
        raise KeyError(
            f"No time-like coordinate ('time' or 'valid_time') found in {path}. "
            f"Variables are: {list(ds.variables)}"
        )

    # time_raw -> numpy datetime64 array
    time = time_raw.astype("datetime64[h]")
    time_unix = time.astype("datetime64[s]").astype("int64")

    ds.close()

    return {
        "file": path,
        "lon": lon,
        "lat": lat,
        "time": time,
        "time_unix": time_unix,
        "lon_sort_index": lon_sort_index,
    }


def load_ERA5_reference(era5_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Scan a directory for ERA5 total-precip files and load metadata.

    Parameters
    ----------
    era5_dir : str
        Directory containing files like 'Total_precip_2010.nc'.

    Returns
    -------
    dict[int, dict]
        Mapping: year → metadata dict with keys:
            'file', 'lon', 'lat', 'time', 'time_unix', 'ymdh', 'lon_sort_index'

    Notes
    -----
    * All years are assumed to share the same lon/lat grid.
    * The lon array is already in -180..180 order; the corresponding
      reordering of the data must use 'lon_sort_index'.
    """
    files = [
        os.path.join(era5_dir, f)
        for f in os.listdir(era5_dir)
        if f.startswith("Total_precip_") and f.endswith(".nc")
    ]

    if not files:
        raise FileNotFoundError(
            f"No 'Total_precip_YYYY.nc' files found in ERA5 directory: {era5_dir}"
        )

    meta_by_year: Dict[int, Dict[str, Any]] = {}

    for path in sorted(files):
        year = _parse_year_from_filename(path)
        meta_by_year[year] = _load_single_ERA5_file(path)

    return meta_by_year


__all__ = ["load_ERA5_reference"]