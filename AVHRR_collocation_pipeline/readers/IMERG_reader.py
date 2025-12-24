"""
IMERG Reader and Collocation Tools
----------------------------------

Fast collocation of AVHRR pixels with IMERG half-hourly precipitation.

Caller passes IMERG metadata:

    IMERG_meta = load_IMERG_reference(IMERG_DIR)

Author: Omid Zandi
"""

from __future__ import annotations

import numpy as np
import h5py
import AVHRR_collocation_pipeline.utils as utils


def collocate_IMERG_precip(df, IMERG_meta):
    """
    Collocate AVHRR pixels with IMERG precipitation.

    df must contain lon/lat and scan_line_times.
    If df has scan_halfhour_unix (recommended), it will be used.
    """
    (
        IMERG_FILES,
        IMERG_DT,
        IMERG_DT_STAMPS,
        IMERG_LON,
        IMERG_LAT,
    ) = IMERG_meta

    # ---------------------------------------------------------
    # 1) Time -> IMERG file index
    # Prefer shared, precomputed half-hour timestamps if present
    # ---------------------------------------------------------
    if "scan_halfhour_unix" in df.columns:
        scan_times = df["scan_halfhour_unix"].to_numpy().astype("int64")
        # IMERG_DT_STAMPS are start times of half-hour files -> exact match desired
        sel_idx = np.searchsorted(IMERG_DT_STAMPS, scan_times, side="right") - 1
    else:
        scan_times = df["scan_line_times"].to_numpy().astype("int64")
        sel_idx = np.digitize(scan_times, IMERG_DT_STAMPS) - 1

    if np.any(sel_idx < 0) or np.any(sel_idx >= len(IMERG_FILES)):
        raise ValueError("Some scan_line_times fall outside IMERG coverage.")

    uniq_indices = np.unique(sel_idx)

    # ---------------------------------------------------------
    # 2) Spatial mapping
    # ---------------------------------------------------------
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    idx, idy = utils.index_finder(lon, lat, IMERG_LON, IMERG_LAT)
    good = (idx >= 0) & (idy >= 0)

    IMERG_out = np.full(len(df), np.nan, dtype="float32")

    # ---------------------------------------------------------
    # 3) Load only needed IMERG files
    # ---------------------------------------------------------
    for t_idx in uniq_indices:
        mask_t = (sel_idx == t_idx) & good
        if not np.any(mask_t):
            continue

        IMERG_file = IMERG_FILES[int(t_idx)]

        with h5py.File(IMERG_file, "r") as h5:
            arr = h5["Grid/precipitation"][0].transpose()
            arr = np.flip(arr, axis=0)
            arr = np.where(arr == -9999.9, np.nan, arr)

        IMERG_out[mask_t] = arr[idy[mask_t], idx[mask_t]]

    df2 = df.copy()
    df2["IMERG_preci"] = IMERG_out
    return df2