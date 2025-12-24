"""
IMERG reference loading utilities.

This module does NOT hardcode any directory paths.
It ONLY provides functions that load IMERG metadata:

    * file list
    * datetimes
    * UNIX timestamps
    * lon/lat grid
"""

from __future__ import annotations
import os
import numpy as np
import datetime
import h5py


# ===============================================================
# 1. List IMERG files
# ===============================================================
def list_IMERG_files(IMERG_dir: str) -> np.ndarray:
    files = [
        os.path.join(IMERG_dir, f)
        for f in os.listdir(IMERG_dir)
        if f.endswith(".HDF5") and f.startswith("3B-HHR")
    ]

    if len(files) == 0:
        raise FileNotFoundError(f"No IMERG HDF5 found in: {IMERG_dir}")

    return np.array(sorted(files))


# ===============================================================
# 2. Parse datetime from filename
# ===============================================================
def IMERG_datetime_from_filename(fname: str) -> datetime.datetime:
    base = os.path.basename(fname)
    parts = base.split(".")

    # Find `20100702-S083000-E085959`
    date_part = next(p for p in parts if "-S" in p)

    ymd, start = date_part.split("-S")
    y, m, d = int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])

    hh = int(start[0:2])
    mm = int(start[2:4])
    ss = int(start[4:6])

    return datetime.datetime(y, m, d, hh, mm, ss)


# ===============================================================
# 3. Load lon/lat grid (same for all files)
# ===============================================================
def load_IMERG_grid(sample_file: str):
    with h5py.File(sample_file, "r") as h5:
        lon_raw = h5["/Grid"]["lon"][:] - 0.05
        lat_raw = h5["/Grid"]["lat"][:][::-1] + 0.05

    # Clean rounding to avoid floats like 179.79999
    lon = np.round(lon_raw.astype(float), 1)
    lat = np.round(lat_raw.astype(float), 1)

    return lon, lat

# ===============================================================
# 4. Full loading function (used by test scripts)
# ===============================================================
def load_IMERG_reference(IMERG_dir: str):
    """
    Load and return all IMERG reference metadata:

        * file list
        * datetimes
        * UNIX timestamps
        * lon/lat grid
    """
    files = list_IMERG_files(IMERG_dir)

    dt_list = [IMERG_datetime_from_filename(f) for f in files]
    dt_array = np.array(dt_list)
    dt_stamps = np.array([dt.timestamp() for dt in dt_list])

    lon, lat = load_IMERG_grid(files[0])

    return files, dt_array, dt_stamps, lon, lat