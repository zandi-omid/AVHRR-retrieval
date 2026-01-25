#!/usr/bin/env python3
"""
evaluate_retrieval_vs_refs_categorical.py

Categorical metrics (POD, FAR, SR, CSI, Bias, HSS + TP/FP/FN/TN)
for retrieved AVHRR precipitation vs one or more references.

Per orbit file:
  - Read NH and SH groups (WGS84 0.25° grid).
  - Restrict to poleward band:
        NH: y >= LAT_THRESH_N
        SH: y <= LAT_THRESH_S
  - Build references via REF_BUILDERS (flexible).
  - Apply threshold THR to create event masks and compute TP/FP/FN/TN.

Outputs:
  - JSON with aggregated categorical metrics per (hemisphere, sim_var, ref)

Notes:
  - Start with ERA5-only reference.
  - Later enable IMERG and/or hybrid REF by uncommenting in REF_BUILDERS.
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from tqdm import tqdm


# -----------------------------
# CONFIG (edit these)
# -----------------------------
FOLDER = "/scratch/omidzandi/evaluation/2019_collocated_with_ref_preci"
GLOB_PATTERN = "*.nc"
MAX_WORKERS = 24

SIM_VARIABLES = ["retrieved"]

V_ERA5 = "ERA5_tp"
V_IMERG = "IMERG_preci"   # enable later if present

LAT_THRESH_N = 50.0
LAT_THRESH_S = -50.0

# Categorical threshold in SAME units as your fields
# (choose this carefully depending on units of "retrieved" and ERA5_tp in these collocated files)
THR = 0.01

# NH hybrid switch latitude (only used if you enable REF builder)
POLAR_SWITCH = 70.0

SAVE_JSON = "eval_2019_retrieved_vs_refs_categorical_50_poleward.json"
# -----------------------------


# -----------------------------
# Reference builders (flexible)
# -----------------------------
def _get_lat2d(ds_sel, ref_shape):
    lat_vals = ds_sel["y"].values
    return np.broadcast_to(lat_vals[:, None], ref_shape)


def build_ref_era5(grp, ds_sel):
    if V_ERA5 not in ds_sel:
        raise KeyError(f"Missing {V_ERA5}")
    return ds_sel[V_ERA5].values


def build_ref_imerg(grp, ds_sel):
    if V_IMERG not in ds_sel:
        raise KeyError(f"Missing {V_IMERG}")
    return ds_sel[V_IMERG].values


def build_ref_hybrid(grp, ds_sel, lat_split=70.0):
    """
    Example hybrid reference:
      - SH: ERA5 everywhere (within SH poleward band)
      - NH: ERA5 for lat >= lat_split, IMERG for LAT_THRESH_N <= lat < lat_split
    """
    if V_ERA5 not in ds_sel:
        raise KeyError(f"Missing {V_ERA5}")
    if V_IMERG not in ds_sel:
        raise KeyError(f"Missing {V_IMERG}")

    era5 = ds_sel[V_ERA5].values
    imerg = ds_sel[V_IMERG].values
    lat2d = _get_lat2d(ds_sel, era5.shape)

    if grp == "SH":
        return era5

    ref = np.full_like(era5, np.nan)
    mask_era = lat2d >= lat_split
    mask_im = (lat2d < lat_split) & (lat2d >= LAT_THRESH_N)
    ref[mask_era] = era5[mask_era]
    ref[mask_im] = imerg[mask_im]
    return ref


# Enable what you want NOW:
REF_BUILDERS = {
    "ERA5": build_ref_era5,
    # Later, once IMERG exists in the collocated files, enable:
    # "IMERG": build_ref_imerg,
    # "REF": lambda grp, ds_sel: build_ref_hybrid(grp, ds_sel, lat_split=POLAR_SWITCH),
}
REF_NAMES = list(REF_BUILDERS.keys())


# -----------------------------
# Categorical helpers
# -----------------------------
def _counts_for_pair(sim, ref, thr):
    """
    Compute TP/FP/FN/TN for event threshold.
    sim/ref are 2D arrays (or any shape).
    """
    sim = np.asarray(sim)
    ref = np.asarray(ref)

    m = np.isfinite(sim) & np.isfinite(ref)
    if not np.any(m):
        return (0, 0, 0, 0)

    s = sim[m] >= thr
    o = ref[m] >= thr

    tp = int(np.sum(s & o))
    fp = int(np.sum(s & ~o))
    fn = int(np.sum(~s & o))
    tn = int(np.sum(~s & ~o))
    return tp, fp, fn, tn


def _add_counts(a, b):
    return tuple(int(x + y) for x, y in zip(a, b))


def _counts_to_metrics(tp, fp, fn, tn):
    tot = tp + fp + fn + tn

    pod = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (tp + fp) if (tp + fp) else np.nan
    sr = 1 - far if np.isfinite(far) else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan

    # Heidke Skill Score
    # C is expected correct by chance
    C = (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / tot if tot else np.nan
    hss = ((tp + tn) - C) / (tot - C) if tot and np.isfinite(C) else np.nan

    return dict(
        POD=pod, FAR=far, SR=sr, CSI=csi, Bias=bias, HSS=hss,
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
        N=int(tot),
    )


# -----------------------------
# Per-file processing
# -----------------------------
def _process_file(path: str):
    """
    Returns counts for each sim_var, each hemisphere, each ref.
    Output structure:
      out[sim_var][grp][ref_name] = (tp, fp, fn, tn)
    """
    out = {
        sim: {
            "NH": {ref: (0, 0, 0, 0) for ref in REF_NAMES},
            "SH": {ref: (0, 0, 0, 0) for ref in REF_NAMES},
        }
        for sim in SIM_VARIABLES
    }

    orbit_name = Path(path).name

    for grp in ("NH", "SH"):
        try:
            ds = xr.open_dataset(path, group=grp)
        except Exception:
            continue

        if "y" not in ds.coords:
            ds.close()
            continue

        lat = ds["y"].values
        if grp == "NH":
            mask_lat_1d = lat >= LAT_THRESH_N
        else:
            mask_lat_1d = lat <= LAT_THRESH_S

        idx = np.where(mask_lat_1d)[0]
        if idx.size == 0:
            ds.close()
            continue

        ds_sel = ds.isel(y=idx)

        # sim arrays
        sim_vals = {v: ds_sel[v].values for v in SIM_VARIABLES if v in ds_sel.data_vars}
        if not sim_vals:
            ds.close()
            continue

        # references that exist in this file/group
        ref_vals_map = {}
        for ref_name, ref_fn in REF_BUILDERS.items():
            try:
                ref_vals_map[ref_name] = ref_fn(grp, ds_sel)
            except Exception:
                continue

        ds.close()
        if not ref_vals_map:
            continue

        # compute counts
        for sim_var, sim_arr in sim_vals.items():
            for ref_name, ref_arr in ref_vals_map.items():
                out[sim_var][grp][ref_name] = _counts_for_pair(sim_arr, ref_arr, THR)

    return orbit_name, out


# -----------------------------
# Aggregation across orbits
# -----------------------------
def compute_categorical_multi_orbit(folder: str):
    t0 = time.time()
    paths = sorted(Path(folder).glob(GLOB_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No files matching '{GLOB_PATTERN}' in {folder}")

    print(f"🔹 Found {len(paths)} orbit files in: {folder}")
    print(f"🔹 References enabled: {REF_NAMES}")
    print(f"🔹 Sim variables: {SIM_VARIABLES}")
    print(f"🔹 Threshold: {THR}")
    print(f"🔹 Workers: {MAX_WORKERS}")

    # accumulator
    acc = {
        sim: {
            "NH": {ref: (0, 0, 0, 0) for ref in REF_NAMES},
            "SH": {ref: (0, 0, 0, 0) for ref in REF_NAMES},
        }
        for sim in SIM_VARIABLES
    }

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_process_file, str(p)) for p in paths]

        with tqdm(total=len(futures), desc="Processing (categorical)", unit="orbit", ncols=100) as pbar:
            for fut in as_completed(futures):
                try:
                    orbit_name, part = fut.result()
                except Exception as e:
                    print(f"⚠️ worker failed: {e}", file=sys.stderr, flush=True)
                    pbar.update(1)
                    continue

                for sim in SIM_VARIABLES:
                    for hemi in ("NH", "SH"):
                        for ref in REF_NAMES:
                            acc[sim][hemi][ref] = _add_counts(acc[sim][hemi][ref], part[sim][hemi][ref])

                pbar.update(1)

    # convert to metrics
    out = {
        "threshold": THR,
        "poleward_bounds_deg": {"NH": f">={LAT_THRESH_N}", "SH": f"<={LAT_THRESH_S}"},
        "references_enabled": REF_NAMES,
        "categorical_metrics": {
            sim: {
                hemi: {
                    ref: _counts_to_metrics(*acc[sim][hemi][ref])
                    for ref in REF_NAMES
                }
                for hemi in ("NH", "SH")
            }
            for sim in SIM_VARIABLES
        },
    }

    with open(SAVE_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved categorical metrics JSON: {SAVE_JSON}")
    print(f"⏱️ Finished in {(time.time() - t0)/60:.1f} minutes")

    # quick print
    for sim in SIM_VARIABLES:
        for hemi in ("NH", "SH"):
            print(f"\n=== {sim} | {hemi} ===")
            for ref in REF_NAMES:
                m = out["categorical_metrics"][sim][hemi][ref]
                print(f"{ref}: POD={m['POD']:.3f}, FAR={m['FAR']:.3f}, CSI={m['CSI']:.3f}, HSS={m['HSS']:.3f}, Bias={m['Bias']:.3f}")

    return out


if __name__ == "__main__":
    compute_categorical_multi_orbit(FOLDER)