# AVHRR_collocation_pipeline/readers/MERRA2_reference.py
from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional

import numpy as np
from netCDF4 import Dataset


# Accept BOTH styles:
# 1) MERRA2_400.tavg1_2d_slv_Nx.20170826.SUB.nc
# 2) MERRA2_400.tavg1_2d_slv_Nx.20170826.nc4   (older/other archives)
_MERRA2_RE = re.compile(
    r"^MERRA2_(\d{3})\.tavg1_2d_slv_Nx\.(\d{8})(?:\.SUB)?\.(?:nc|nc4)$"
)


def _date_from_MERRA2_filename(fname: str) -> str:
    """
    Examples
    --------
    MERRA2_400.tavg1_2d_slv_Nx.20170826.SUB.nc  -> "2017-08-26"
    MERRA2_400.tavg1_2d_slv_Nx.20170826.nc4     -> "2017-08-26"
    """
    m = _MERRA2_RE.match(fname)
    if not m:
        raise ValueError(f"Not a valid MERRA2 daily slv filename: {fname}")
    ymd = m.group(2)
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _find_MERRA2_files(MERRA2_dir: str) -> List[str]:
    """
    Find matching MERRA2 daily files in a directory (non-recursive).
    """
    files: List[str] = []
    for f in os.listdir(MERRA2_dir):
        if f.endswith((".nc", ".nc4")) and _MERRA2_RE.match(f):
            files.append(os.path.join(MERRA2_dir, f))
    return sorted(files)


def _is_monotonic_increasing(x: np.ndarray) -> bool:
    return bool(np.all(np.diff(x) > 0))


def load_MERRA2_reference(MERRA2_dir: str) -> Dict[str, Any]:
    """
    Build:
      - date_to_file: {"YYYY-MM-DD": "/path/to/file"}
      - lon/lat arrays (kept as-is if already -180..180 increasing)
      - lon_sort_index: None if no reorder needed, else argsort index
      - needs_lat_flip: True if file lat is opposite of AVHRR df convention

    Notes
    -----
    Your archive files (SUB) already look like:
      lon: -180 .. 179.375 (monotonic increasing)
      lat: -90 .. 90 (monotonic increasing)
    So in that case: lon_sort_index=None and needs_lat_flip=False.
    """
    paths = _find_MERRA2_files(MERRA2_dir)

    if not paths:
        raise FileNotFoundError(
            f"No MERRA2 daily files found in {MERRA2_dir} matching pattern:\n"
            f"  MERRA2_###.tavg1_2d_slv_Nx.YYYYMMDD.SUB.nc  (or .nc4)\n"
            f"Example:\n"
            f"  MERRA2_400.tavg1_2d_slv_Nx.20170826.SUB.nc"
        )

    # Map date -> file (if duplicates exist, keep the last one by sort order)
    date_to_file: Dict[str, str] = {}
    for p in paths:
        date_to_file[_date_from_MERRA2_filename(os.path.basename(p))] = p

    # Read lon/lat from a sample file
    with Dataset(paths[0]) as ds:
        lon_org = np.array(ds["lon"][:], dtype="float32")
        lat_org = np.array(ds["lat"][:], dtype="float32")

    # ---- LAT handling: do we need to flip data later? ----
    # We want meta["lat"] to be the coordinate array that matches the data as read.
    # Your index_finder expects coords to match the stored array orientation.
    # If lat is decreasing (90..-90), we keep it decreasing and set needs_lat_flip=False,
    # OR we can flip coords and set needs_lat_flip=True. Here we keep coords as-is and
    # only flip data later if you choose to standardize. The simplest: keep as-is and
    # detect if the file is decreasing (common in some products).
    # needs_lat_flip = False
    # lat = lat_org

    lat = lat_org[::-1].astype("float32")
    needs_lat_flip = True

    # If lat is decreasing (90..-90), index_finder must handle decreasing axes.
    # If yours does NOT, then set needs_lat_flip=True and reverse lat here.
    # Given your earlier MERRA2 inspection showed lat increasing, we’re fine.
    if not _is_monotonic_increasing(lat_org) and _is_monotonic_increasing(lat_org[::-1]):
        # lat is decreasing
        # Option A (recommended if index_finder can't handle decreasing):
        # needs_lat_flip = True
        # lat = lat_org[::-1]
        # Option B (if index_finder handles decreasing): keep it as-is.
        # We'll keep as-is by default.
        lat = lat_org
        needs_lat_flip = False

    # ---- LON handling: reorder ONLY if needed ----
    # Case 1: already -180..180 and increasing -> no reordering needed.
    # Case 2: 0..360 or unsorted -> shift to -180..180 and sort (and store index).
    lon_sort_index: Optional[np.ndarray] = None

    lon_min, lon_max = float(np.nanmin(lon_org)), float(np.nanmax(lon_org))
    lon_in_minus180_180 = (lon_min >= -180.0 - 1e-3) and (lon_max <= 180.0 + 1e-3)

    if lon_in_minus180_180 and _is_monotonic_increasing(lon_org):
        lon = lon_org
        lon_sort_index = None
    else:
        lon_shifted = ((lon_org + 180.0) % 360.0) - 180.0
        lon_sort_index = np.argsort(lon_shifted)
        lon = lon_shifted[lon_sort_index].astype("float32")

    return {
        "date_to_file": date_to_file,
        "lon": lon,
        "lat": lat.astype("float32"),
        "lon_sort_index": lon_sort_index,  # None means "do NOT reorder data"
        "needs_lat_flip": needs_lat_flip,
    }


__all__ = ["load_MERRA2_reference"]