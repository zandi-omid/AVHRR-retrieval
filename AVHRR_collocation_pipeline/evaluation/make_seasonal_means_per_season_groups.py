#!/usr/bin/env python3
"""
Compute seasonal mean NetCDFs (one file per season) from per-orbit collocated WGS files.

Input: orbit NetCDFs with groups "NH" and "SH" (WGS 0.25 grid),
       containing variables like: retrieved, ERA5_tp (later IMERG_preci).

Output: 4 NetCDF files (DJF/MAM/JJA/SON), each with groups NH and SH.
        Each group contains seasonal-mean variables for the chosen VAR_NAMES.

Method: NaN-safe mean using sum/count accumulation per variable per group.

Notes:
- Flexible: add/remove reference variables by editing VAR_NAMES.
- Robust: skips files/groups/vars that are missing.
"""

import os
import re
import gc
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = "/scratch/omidzandi/evaluation/2019_collocated_with_ref_preci"
OUTPUT_DIR = "/scratch/omidzandi/evaluation/seasonal_means_2019"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = ("NH", "SH")

# Put the vars you want in the seasonal file here:
# (today) retrieved + ERA5_tp
# (later) just add "IMERG_preci", "AIRS_preci", etc.
VAR_NAMES = [
    "retrieved",
    "ERA5_tp",
    # "IMERG_preci",
]

# If your filenames contain DYYDDD like D19045 (2019 day 45), this works.
# If not, tell me your filename pattern and we’ll swap this parser.
SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

MAX_WORKERS = 24
ENGINE = "netcdf4"
DECODE_TIMES = False

# Optional compression on output
COMPRESS = True
COMPLEVEL = 4

# =========================
# Helpers
# =========================
def season_from_filename(fname: str) -> str | None:
    """
    Extract season from filename containing pattern like ...D19045...
    meaning year=2019, doy=045.
    """
    m = re.search(r"D(\d{2})(\d{3})", fname)
    if not m:
        return None
    year = 2000 + int(m.group(1))
    doy = int(m.group(2))
    date = np.datetime64(f"{year}-01-01") + np.timedelta64(doy - 1, "D")
    month = int(str(date)[5:7])
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return None


def _process_one_file_for_vars(path: str, var_names: list[str]):
    """
    Worker: open a single orbit file and return per-group per-var (sum, count, template coords).
    Returns:
      {
        "NH": {
           "template": Dataset coords (first time we see it),
           "vars": { var: {"sum": DataArray, "count": DataArray} }
        },
        "SH": {...}
      }
    """
    out = {}
    for grp in GROUPS:
        try:
            ds = xr.open_dataset(path, group=grp, engine=ENGINE, decode_times=DECODE_TIMES)
        except Exception:
            continue

        # Need coords for alignment/output
        if ("y" not in ds.coords) or ("x" not in ds.coords):
            ds.close()
            continue

        grp_out = {"template": ds[[]], "vars": {}}

        for v in var_names:
            if v not in ds:
                continue
            da = ds[v].astype(np.float32)

            mask = np.isfinite(da)
            grp_out["vars"][v] = {
                "sum": da.where(mask, 0.0),
                "count": xr.DataArray(mask.astype(np.int16), coords=da.coords, dims=da.dims),
            }

        ds.close()

        # Only keep group if at least one var found
        if grp_out["vars"]:
            out[grp] = grp_out

    return out


def _init_accumulator():
    # acc[grp][var] -> {"sum": DataArray, "count": DataArray}
    return {grp: {} for grp in GROUPS}


def _accumulate(acc, piece):
    """
    Merge one file's contribution into accumulator.
    Handles alignment safely (exact coords expected).
    """
    for grp, grp_out in piece.items():
        for v, sc in grp_out["vars"].items():
            s_new = sc["sum"]
            c_new = sc["count"]

            if v not in acc[grp]:
                acc[grp][v] = {"sum": s_new, "count": c_new}
            else:
                s_old = acc[grp][v]["sum"]
                c_old = acc[grp][v]["count"]

                # exact coordinate match expected; align just in case
                s_old, s_new = xr.align(s_old, s_new, join="exact")
                c_old, c_new = xr.align(c_old, c_new, join="exact")

                acc[grp][v]["sum"] = (s_old + s_new).astype(np.float32)
                acc[grp][v]["count"] = (c_old + c_new).astype(np.int32)


def _make_encoding(ds: xr.Dataset):
    if not COMPRESS:
        return {}
    enc = {}
    for v in ds.data_vars:
        enc[v] = {"zlib": True, "complevel": COMPLEVEL, "dtype": "float32"}
    return enc


def write_season_file(season: str, acc, out_path: Path, attrs: dict | None = None):
    """
    Write one season file with NH/SH groups, multiple variables each.
    """
    if out_path.exists():
        out_path.unlink()

    for grp in GROUPS:
        if not acc[grp]:
            print(f"[WARN] {season}: no data accumulated for group {grp}. Skipping group.")
            continue

        # Build dataset for this group
        ds_out = xr.Dataset()
        # Use coords from first available var
        first_var = next(iter(acc[grp].keys()))
        ds_out = ds_out.assign_coords(acc[grp][first_var]["sum"].coords)

        for v, sc in acc[grp].items():
            s = sc["sum"]
            c = sc["count"]
            mean = xr.where(c > 0, s / c, np.nan).astype(np.float32)
            ds_out[v] = mean

            ds_out[v].attrs["long_name"] = f"Seasonal mean of {v}"
            ds_out[v].attrs["note"] = "NaN-safe mean via sum/count accumulation"

        ds_out.attrs["season"] = season
        ds_out.attrs["source_dir"] = str(INPUT_DIR)
        ds_out.attrs["created_utc"] = str(np.datetime64("now"))
        if attrs:
            ds_out.attrs.update(attrs)

        mode = "w" if grp == "NH" else "a"
        ds_out.to_netcdf(
            out_path,
            group=grp,
            mode=mode,
            format="NETCDF4",
            encoding=_make_encoding(ds_out),
        )
        ds_out.close()

    print(f"✅ Wrote seasonal file: {out_path}")


# =========================
# Main
# =========================
def main():
    t0 = time.time()

    all_files = sorted(Path(INPUT_DIR).glob("*.nc"))
    if not all_files:
        raise FileNotFoundError(f"No .nc files found in: {INPUT_DIR}")

    files_by_season = {s: [] for s in SEASON_MONTHS}
    skipped = 0
    for f in all_files:
        season = season_from_filename(f.name)
        if season is None:
            skipped += 1
            continue
        files_by_season[season].append(str(f))

    print(f"Found {len(all_files)} files total. Skipped (no DYYDDD tag) = {skipped}")
    for s in files_by_season:
        print(f"  {s}: {len(files_by_season[s])} files")

    for season, files in files_by_season.items():
        if not files:
            print(f"[WARN] No files for season {season}. Skipping.")
            continue

        print(f"\n========== 🌎 Season {season} | {len(files)} files ==========")
        acc = _init_accumulator()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_process_one_file_for_vars, p, VAR_NAMES) for p in files]

            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{season}", ncols=100):
                try:
                    piece = fut.result()
                    if piece:
                        _accumulate(acc, piece)
                except Exception as e:
                    # keep going; one broken file shouldn't kill the season
                    print(f"⚠️ {season}: error in a worker: {e}", flush=True)

        out_path = Path(OUTPUT_DIR) / f"seasonal_mean_{season}.nc"
        write_season_file(
            season,
            acc,
            out_path,
            attrs={
                "lat_threshold_note": "File contains only what exists in inputs; masking handled upstream.",
                "variables": ", ".join(VAR_NAMES),
            },
        )

        # cleanup
        del acc
        gc.collect()

    dt = time.time() - t0
    print(f"\n⏱️ Done in {dt/60:.1f} minutes")


if __name__ == "__main__":
    main()