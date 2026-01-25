#!/usr/bin/env python3
"""
evaluate_retrieval_vs_refs.py

Evaluate retrieved AVHRR precipitation maps against one or more references.

Per orbit file:
  - Read NH and SH groups (WGS84, 0.25° grid).
  - Restrict to poleward band (NH: lat >= LAT_THRESH_N, SH: lat <= LAT_THRESH_S).
  - Build one or more references via REF_BUILDERS (flexible).
  - Compute metrics (KGE, CC, RMSE, RE, STD, CRMSE, etc.) for each (hemisphere, sim_var, ref).

Outputs:
  - JSON: weighted metrics aggregated across orbits
  - CSV: per-orbit summary table (e.g., KGE vs a chosen ref)

Notes:
  - Your current files: likely have 'retrieved' + 'ERA5_tp' only.
  - Later: add IMERG (and/or a hybrid REF) by enabling the builders below.
"""

import os
import math
import json
import time
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# -----------------------------
# CONFIG (edit these)
# -----------------------------
FOLDER = "/scratch/omidzandi/evaluation/2019_collocated_with_ref_preci"  # your folder of collocated orbit .nc files
GLOB_PATTERN = "*.nc"  # change if needed
MAX_WORKERS = 24

# Variables in your collocated files
SIM_VARIABLES = ["retrieved"]  # for now you only have one retrieved field

V_ERA5 = "ERA5_tp"
V_IMERG = "IMERG_preci"  # (later, when you add it to collocated files)

# Latitude thresholds for poleward evaluation
LAT_THRESH_N = 50.0   # NH: use data for lat >= 50°
LAT_THRESH_S = -50.0  # SH: use data for lat <= -50°

# KGE variant: "2009" or "2012"
KGE_VERSION = "2012"
EPS = 1e-12

# Output files
SAVE_JSON = "eval_2019_retrieved_vs_refs_50_poleward.json"  # saved in CWD
SAVE_CSV = SAVE_JSON.replace(".json", "_per_orbit.csv")

# Which reference to use for per-orbit quick summary (KGE column)
SUMMARY_REF_NAME = "ERA5"  # must exist in REF_BUILDERS keys
SUMMARY_SIM_VAR = "retrieved"
# -----------------------------


EMPTY_METRICS = {
    "KGE": np.nan,
    "CC": np.nan,
    "RMSE": np.nan,
    "RE": np.nan,
    "STD_X": np.nan,
    "STD_Y": np.nan,
    "STD_RATIO": np.nan,
    "CRMSE": np.nan,
    "n": 0,
}


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
    # "REF": lambda grp, ds_sel: build_ref_hybrid(grp, ds_sel, lat_split=70.0),
}
REF_NAMES = list(REF_BUILDERS.keys())


# -----------------------------
# Metric helpers
# -----------------------------
def _pair_sums(x, y):
    """Compute sufficient statistics for pairs (x, y), ignoring NaNs."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return None
    x, y = x[m], y[m]
    return (
        int(len(x)),
        float(np.sum(x)),
        float(np.sum(y)),
        float(np.sum(x * x)),
        float(np.sum(y * y)),
        float(np.sum(x * y)),
    )


def _finalize_metrics(stats):
    """Compute KGE, CC, RMSE, RE, STD, STD_RATIO, CRMSE from sufficient stats."""
    n, sx, sy, sx2, sy2, sxy = stats
    if n == 0:
        return EMPTY_METRICS.copy()

    n_f = float(n)
    mean_x, mean_y = sx / n_f, sy / n_f

    var_x = max(sx2 / n_f - mean_x**2, 0.0)
    var_y = max(sy2 / n_f - mean_y**2, 0.0)
    std_x, std_y = math.sqrt(var_x), math.sqrt(var_y)

    cov_xy = sxy / n_f - mean_x * mean_y
    rho = cov_xy / (std_x * std_y + EPS)

    beta = mean_x / (mean_y + EPS)

    if KGE_VERSION == "2009":
        alpha = std_x / (std_y + EPS)
    else:  # 2012
        alpha = (std_x / (abs(mean_x) + EPS)) / (std_y / (abs(mean_y) + EPS))

    KGE = 1.0 - math.sqrt((1.0 - rho) ** 2 + (1.0 - alpha) ** 2 + (1.0 - beta) ** 2)

    RMSE = math.sqrt(max(sx2 + sy2 - 2.0 * sxy, 0.0) / n_f)
    RE = (sx - sy) / (sy + EPS)

    CRMSE = math.sqrt(var_x + var_y - 2.0 * std_x * std_y * rho)
    STD_RATIO = std_x / std_y if std_y > 0 else np.nan

    return {
        "KGE": KGE,
        "CC": rho,
        "RMSE": RMSE,
        "RE": RE,
        "STD_X": std_x,
        "STD_Y": std_y,
        "STD_RATIO": STD_RATIO,
        "CRMSE": CRMSE,
        "n": n,
    }


# -----------------------------
# Per-file processing
# -----------------------------
def _process_file(path: str):
    """
    Process a single orbit file:
      - Load NH & SH groups.
      - Restrict by latitude thresholds.
      - Build refs available for this file/group.
      - Compute metrics for each sim_var vs each reference.
    """
    orbit_name = Path(path).name
    results = {"orbit_name": orbit_name, "empty": True}

    try:
        any_data = False

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
                mask_lat = lat >= LAT_THRESH_N
            else:
                mask_lat = lat <= LAT_THRESH_S

            idx = np.where(mask_lat)[0]
            if idx.size == 0:
                ds.close()
                continue

            ds_sel = ds.isel(y=idx)

            # Load sim arrays
            sim_vals = {
                v: ds_sel[v].values
                for v in SIM_VARIABLES
                if v in ds_sel.data_vars
            }

            if not sim_vals:
                ds.close()
                continue

            # Build reference arrays that exist
            ref_vals_map = {}
            for ref_name, ref_fn in REF_BUILDERS.items():
                try:
                    ref_vals_map[ref_name] = ref_fn(grp, ds_sel)
                except Exception:
                    continue

            ds.close()

            if not ref_vals_map:
                continue

            # Compute metrics
            for sim_var, sim_arr in sim_vals.items():
                metrics_by_ref = {}
                for ref_name, ref_arr in ref_vals_map.items():
                    stats = _pair_sums(sim_arr, ref_arr)
                    if stats:
                        mets = _finalize_metrics(stats)
                        mets["n"] = stats[0]
                    else:
                        mets = EMPTY_METRICS.copy()
                    metrics_by_ref[ref_name] = mets

                results[(grp, sim_var)] = metrics_by_ref
                any_data = True

        results["empty"] = not any_data

        # Per-orbit quick summary KGE for chosen (sim, ref) in each hemisphere
        # (kept separate for easy CSV)
        nh_kge = (
            results.get(("NH", SUMMARY_SIM_VAR), {})
            .get(SUMMARY_REF_NAME, {})
            .get("KGE", np.nan)
        )
        sh_kge = (
            results.get(("SH", SUMMARY_SIM_VAR), {})
            .get(SUMMARY_REF_NAME, {})
            .get("KGE", np.nan)
        )
        results["NH_KGE_summary"] = nh_kge
        results["SH_KGE_summary"] = sh_kge

    except Exception as e:
        print(f"⚠️ Error processing {orbit_name}: {e}", file=sys.stderr, flush=True)
        results = {
            "orbit_name": orbit_name,
            "NH_KGE_summary": np.nan,
            "SH_KGE_summary": np.nan,
            "empty": True,
        }

    return results


# -----------------------------
# Multi-orbit aggregation
# -----------------------------
def compute_metrics_multi_orbit(folder: str):
    t0 = time.time()
    paths = sorted(Path(folder).glob(GLOB_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No files matching '{GLOB_PATTERN}' in {folder}")

    print(f"🔹 Found {len(paths)} orbit files in: {folder}")
    print(f"🔹 References enabled: {REF_NAMES}")
    print(f"🔹 Sim variables: {SIM_VARIABLES}")
    print(f"🔹 Workers: {MAX_WORKERS}")

    combined = {
        (grp, sim, ref): []
        for grp in ("NH", "SH")
        for sim in SIM_VARIABLES
        for ref in REF_NAMES
    }

    per_orbit_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(_process_file, str(p)): p for p in paths}

        with tqdm(
            total=len(paths),
            desc="Processing orbits (NH+SH)",
            unit="orbit",
            ncols=100,
        ) as pbar:
            for fut in as_completed(future_map):
                res = fut.result()

                if res.get("empty", False):
                    pbar.update(1)
                    continue

                # Collect per-orbit summary row
                per_orbit_rows.append(
                    {
                        "orbit_name": res.get("orbit_name", "unknown"),
                        "NH_KGE_summary": res.get("NH_KGE_summary", np.nan),
                        "SH_KGE_summary": res.get("SH_KGE_summary", np.nan),
                    }
                )

                # Aggregate detailed metrics
                for key, metrics_by_ref in res.items():
                    if not isinstance(key, tuple) or len(key) != 2:
                        continue
                    grp, sim = key

                    for ref_name, mets in metrics_by_ref.items():
                        if ref_name not in REF_NAMES:
                            continue
                        combined[(grp, sim, ref_name)].append(mets)

                pbar.update(1)

    # Weighted aggregation across orbits
    results = {}
    MET_KEYS = ("KGE", "CC", "RMSE", "RE", "STD_X", "STD_Y", "STD_RATIO", "CRMSE")

    for grp in ("NH", "SH"):
        results[grp] = {}
        for sim in SIM_VARIABLES:
            results[grp][sim] = {}
            for ref in REF_NAMES:
                mets_list = combined[(grp, sim, ref)]
                if not mets_list:
                    results[grp][sim][ref] = {k: np.nan for k in MET_KEYS}
                    continue

                # weighted by n
                weighted = {}
                for k in MET_KEYS:
                    vals = []
                    ns = []
                    for m in mets_list:
                        v = m.get(k, np.nan)
                        n = m.get("n", 0)
                        if np.isfinite(v) and np.isfinite(n) and n > 0:
                            vals.append(v)
                            ns.append(n)
                    if ns:
                        weighted[k] = float(np.nansum(np.array(vals) * np.array(ns)) / np.nansum(ns))
                    else:
                        weighted[k] = np.nan

                results[grp][sim][ref] = weighted

    # Print summary
    for grp in ("NH", "SH"):
        hemi_label = f"≥{LAT_THRESH_N}°" if grp == "NH" else f"≤{LAT_THRESH_S}°"
        for sim in SIM_VARIABLES:
            print(f"\n=== {sim} | {grp} ({hemi_label}) ===")
            for ref in REF_NAMES:
                mets = results[grp][sim][ref]
                print(
                    f"{ref}: "
                    f"KGE={mets['KGE']:.3f}, "
                    f"CC={mets['CC']:.3f}, "
                    f"RMSE={mets['RMSE']:.3f}, "
                    f"RE={mets['RE']:.3f}"
                )

    # Save outputs
    with open(SAVE_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved summary JSON: {SAVE_JSON}")

    if per_orbit_rows:
        df = pd.DataFrame(per_orbit_rows).sort_values("orbit_name")
        df.to_csv(SAVE_CSV, index=False)
        print(f"🧾 Saved per-orbit CSV: {SAVE_CSV}")
    else:
        print("⚠️ No non-empty orbits to save to CSV.")

    print(f"\n⏱️ Finished in {(time.time() - t0) / 60:.1f} minutes")
    return results


if __name__ == "__main__":
    compute_metrics_multi_orbit(FOLDER)