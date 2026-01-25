#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------- CONFIG -----------------
CSV_FN = "/home/omidzandi/AVHRR_collocation_pipeline/eval_2019_retrieved_vs_refs_50_poleward_per_orbit.csv"

SRC_DIR = Path("/ra1/pubdat/AVHRR_CloudSat_proj/SIMON_CODES/retrieved_archive/2019")
DST_DIR = Path("/ra1/pubdat/AVHRR_CloudSat_proj/SIMON_CODES/retrieved_archive/2019_selected_for_Dave")

TOP_N = 20

# If True: require BOTH NH and SH KGE to be finite, then mean of the two
# If False: take mean over available hemispheres (NH-only or SH-only can rank)
REQUIRE_BOTH_HEMIS = True

DRY_RUN = False # set False to actually copy
# ------------------------------------------


def orbit_to_key(orbit_name: str) -> str:
    """
    Extract a compact key from the orbit_name that should appear in L2 filenames.
    We try to pull: DYYDDD.SHHMM.EHHMM  (e.g., D19045.S0556.E0739)

    This makes matching robust even if orbit_name has extra suffix/prefix.
    """
    m = re.search(r"(D\d{2}\d{3}\.S\d{4}\.E\d{4})", orbit_name)
    if m:
        return m.group(1)
    # fallback: try DYYDDD only
    m = re.search(r"(D\d{2}\d{3})", orbit_name)
    if m:
        return m.group(1)
    return orbit_name


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # --- load csv ---
    df = pd.read_csv(CSV_FN)

    # --- coerce numeric ---
    for c in ["NH_KGE_summary", "SH_KGE_summary"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- compute mean KGE across hemispheres ---
    if REQUIRE_BOTH_HEMIS:
        df2 = df.dropna(subset=["NH_KGE_summary", "SH_KGE_summary"]).copy()
        df2["KGE_mean_NH_SH"] = df2[["NH_KGE_summary", "SH_KGE_summary"]].mean(axis=1)
    else:
        df2 = df.copy()
        df2["KGE_mean_NH_SH"] = df2[["NH_KGE_summary", "SH_KGE_summary"]].mean(axis=1, skipna=True)
        df2 = df2.dropna(subset=["KGE_mean_NH_SH"]).copy()

    # --- top N ---
    top = df2.sort_values("KGE_mean_NH_SH", ascending=False).head(TOP_N).copy()

    print(f"\nTop {TOP_N} by mean(NH,SH) KGE (REQUIRE_BOTH_HEMIS={REQUIRE_BOTH_HEMIS})")
    print(top[["orbit_name", "NH_KGE_summary", "SH_KGE_summary", "KGE_mean_NH_SH"]].to_string(index=False))

    # --- build a fast index of source files ---
    src_files = list(SRC_DIR.glob("*.nc"))
    src_names = [p.name for p in src_files]
    print(f"\nFound {len(src_files)} candidate L2 files in: {SRC_DIR}")

    # --- match and copy ---
    copied = []
    missing = []

    for orbit in top["orbit_name"].tolist():
        key = orbit_to_key(str(orbit))
        # match any filename containing the key
        matches = [p for p in src_files if key in p.name]

        if len(matches) == 0:
            missing.append((orbit, key))
            continue

        # if multiple matches, copy them all (usually 1)
        for p in matches:
            dst = DST_DIR / p.name
            if DRY_RUN:
                print(f"[DRY_RUN] would copy: {p.name} -> {dst}")
            else:
                shutil.copy2(p, dst)
            copied.append(p.name)

    # --- summary ---
    print("\n========== SUMMARY ==========")
    print(f"Would copy / copied: {len(copied)} files")
    print(f"Missing matches for: {len(missing)} orbits")

    if missing:
        print("\nOrbits with no match (orbit_name -> key used):")
        for orbit, key in missing:
            print(f"  {orbit}  ->  {key}")

        # Helpful debug: show a few source filenames if matching failed
        print("\nExample source filenames (first 10):")
        for s in src_names[:10]:
            print(" ", s)

    if not DRY_RUN:
        print(f"\n✅ Files copied into: {DST_DIR}")
    else:
        print("\nSet DRY_RUN = False to actually copy.")


if __name__ == "__main__":
    main()