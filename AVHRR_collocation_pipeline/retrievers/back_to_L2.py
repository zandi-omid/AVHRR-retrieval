from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Optional

import numpy as np
import xarray as xr

from AVHRR_collocation_pipeline.utils import safe_to_netcdf


class AVHRRBackToL2:
    """
    Attach gridded retrievals (NH/SH WGS grids) back onto the original
    AVHRR L2 orbit swath grid.
    """

    def __init__(
        self,
        retrieved_var_names: Sequence[str] = (
            "retrieved_precip_mean",
            "retrieved_precip_q70",
            "retrieved_precip_q75",
            "retrieved_precip_q80",
        ),
        line_dim: str = "scan_lines_along_track_direction",
        pix_dim: str = "pixel_elements_along_scan_direction",
    ) -> None:
        self.retrieved_var_names = list(retrieved_var_names)
        self.line_dim = line_dim
        self.pix_dim = pix_dim

    @staticmethod
    def _merge_hemispheric_grids(
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        var_names: Sequence[str],
    ) -> Dict[str, xr.DataArray]:
        global_vars: Dict[str, xr.DataArray] = {}

        for v in var_names:
            if v not in ds_nh and v not in ds_sh:
                print(f"[WARN] Retrieval variable '{v}' not in ds_nh or ds_sh — skipping.", flush=True)
                continue

            da_nh: Optional[xr.DataArray] = ds_nh[v] if v in ds_nh else None
            da_sh: Optional[xr.DataArray] = ds_sh[v] if v in ds_sh else None

            if da_nh is not None and da_sh is not None:
                global_da = da_nh.combine_first(da_sh)
            elif da_nh is not None:
                global_da = da_nh
            else:
                global_da = da_sh

            global_vars[v] = global_da

        return global_vars

    def attach_to_orbit_ds(
        self,
        raw_orbit_path: Path | str,
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        copy_vars_from_raw: list[str] | None = None,
    ) -> xr.Dataset:
        raw_orbit_path = Path(raw_orbit_path)
        copy_vars_from_raw = copy_vars_from_raw or []

        line_dim = self.line_dim
        pix_dim = self.pix_dim

        # Open safely and eagerly load what we need so no lazy backend handle
        # survives after leaving the context manager.
        with xr.open_dataset(raw_orbit_path, decode_timedelta=True) as ds_raw:
            required = ["latitude", "longitude", "scan_line_time", "temp_11_0um_nom", "temp_12_0um_nom"]
            missing = [v for v in required if v not in ds_raw]
            if missing:
                raise KeyError(f"Raw orbit missing required variables for back_to_L2: {missing}")

            lat_raw = ds_raw["latitude"].load()
            lon_raw = ds_raw["longitude"].load()
            scan_line_time = ds_raw["scan_line_time"].load()
            tb11 = ds_raw["temp_11_0um_nom"].load()
            tb12 = ds_raw["temp_12_0um_nom"].load()

            extra_loaded = {}
            for v in copy_vars_from_raw:
                if v in ds_raw:
                    extra_loaded[v] = ds_raw[v].load()
                else:
                    print(f"[WARN] raw orbit missing '{v}' — limb correction may fail.", flush=True)

        out = xr.Dataset()
        out = out.assign_coords(
            {
                line_dim: lat_raw.coords[line_dim] if line_dim in lat_raw.coords else range(lat_raw.sizes[line_dim]),
                pix_dim: pix_dim if False else (
                    lat_raw.coords[pix_dim] if pix_dim in lat_raw.coords else range(lat_raw.sizes[pix_dim])
                ),
                "latitude": lat_raw,
                "longitude": lon_raw,
            }
        )

        # Fix coord assignment cleanly
        out = xr.Dataset()
        out = out.assign_coords(
            {
                line_dim: lat_raw.coords[line_dim] if line_dim in lat_raw.coords else range(lat_raw.sizes[line_dim]),
                pix_dim: lat_raw.coords[pix_dim] if pix_dim in lat_raw.coords else range(lat_raw.sizes[pix_dim]),
                "latitude": lat_raw,
                "longitude": lon_raw,
            }
        )

        out["scan_line_time"] = scan_line_time
        out["temp_11_0um_nom"] = tb11
        out["temp_12_0um_nom"] = tb12

        for v, da in extra_loaded.items():
            out[v] = da

        global_grids = self._merge_hemispheric_grids(ds_nh, ds_sh, self.retrieved_var_names)

        for v in self.retrieved_var_names:
            if v not in global_grids:
                continue

            da_global = global_grids[v]
            da_on_swath = da_global.interp(y=lat_raw, x=lon_raw, method="nearest")

            rename_map = {}
            if len(da_on_swath.dims) == 2:
                d0, d1 = da_on_swath.dims
                if d0 != line_dim:
                    rename_map[d0] = line_dim
                if d1 != pix_dim:
                    rename_map[d1] = pix_dim
            if rename_map:
                da_on_swath = da_on_swath.rename(rename_map)

            da_on_swath.attrs["coordinates"] = "latitude longitude"
            out[v] = da_on_swath

        return out

    def attach_to_orbit(
        self,
        raw_orbit_path: Path | str,
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        out_path: Path | str,
    ) -> Path:
        out_path = Path(out_path)

        out_ds = self.attach_to_orbit_ds(raw_orbit_path, ds_nh, ds_sh)

        out_ds.attrs.update({
            "title": "AVHRR precipitation retrieval",
            "orbit_tag": Path(raw_orbit_path).stem,
            "created_utc": str(np.datetime64("now")),
            "institution": "University of Arizona",
        })

        for src in (ds_nh, ds_sh):
            for k, v in src.attrs.items():
                out_ds.attrs.setdefault(k, v)

        def _is_precip_var(name: str) -> bool:
            return (
                name == "precipitation"
                or name.startswith("precipitation_")
                or name.startswith("retrieved_precip_")
            )

        for v in list(out_ds.data_vars):
            if not _is_precip_var(v):
                continue
            out_ds[v].attrs.setdefault("units", "mm hr-1")
            out_ds[v].attrs.setdefault("long_name", "Retrieved surface precipitation rate")
            out_ds[v].attrs.setdefault("standard_name", "surface_precipitation_rate")
            out_ds[v].attrs["coordinates"] = "latitude longitude"

        for nm, target in (("x", "longitude"), ("y", "latitude")):
            if nm in out_ds.variables and target in out_ds.variables:
                if out_ds[nm].ndim == out_ds[target].ndim == 2:
                    out_ds = out_ds.drop_vars(nm, errors="ignore")

        if "coordinates" in out_ds.attrs and str(out_ds.attrs["coordinates"]).strip() in ("x y", "y x"):
            out_ds.attrs["coordinates"] = "latitude longitude"

        encoding = {}
        for v in out_ds.data_vars:
            if _is_precip_var(v):
                encoding[v] = {
                    "dtype": "float32",
                    "_FillValue": float("nan"),
                    "zlib": True,
                    "complevel": 4,
                }

        print(f"[DEBUG] Writing L2 file: {out_path}", flush=True)
        safe_to_netcdf(out_ds, out_path, encoding=encoding, format="NETCDF4")
        return out_path