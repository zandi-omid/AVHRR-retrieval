import os
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import xarray as xr

# --- readers ---
from AVHRR_collocation_pipeline.readers.AVHRR_reader import read_AVHRR_orbit_to_df
from AVHRR_collocation_pipeline.readers.IMERG_reader import collocate_IMERG_precip
from AVHRR_collocation_pipeline.readers.ERA5_reader import collocate_ERA5_precip
from AVHRR_collocation_pipeline.readers.MERRA2_reader import collocate_MERRA2
from AVHRR_collocation_pipeline.readers.AutoSnow_reader import collocate_AutoSnow

# --- reprojection ---
from AVHRR_collocation_pipeline.reproject import reproject_vars_wgs_to_polar

# --- utils ---
import AVHRR_collocation_pipeline.utils as utils


class AVHRRProcessor:
    """
    Stage-1 processor for AVHRR:

      raw AVHRR orbit (WGS) 
        -> collocate refs (IMERG, ERA5, MERRA2, AutoSnow)
        -> build WGS grids
        -> reproject WGS grids to polar stereographic (NH/SH)
        -> (optionally) save polar NetCDF

    This prepares inputs for the DL retrieval stage.
    """

    def __init__(
        self,
        *,
        grid_res: float,
        lat_thresh_nh: float = 55.0,
        lat_thresh_sh: float = -55.0,
        lat_ts_nh: float = 70.0,
        lat_ts_sh: float = -71.0,
        nodata: float = -9999.0,
        imerg_meta=None,
        era5_meta_by_year=None,
        merra2_meta=None,
        autosnow_meta=None,
    ):
        self.grid_res = grid_res
        self.lat_thresh_nh = lat_thresh_nh
        self.lat_thresh_sh = lat_thresh_sh
        self.lat_ts_nh = lat_ts_nh
        self.lat_ts_sh = lat_ts_sh
        self.nodata = nodata

        # references loaded outside
        self.imerg_meta = imerg_meta
        self.era5_meta_by_year = era5_meta_by_year
        self.merra2_meta = merra2_meta
        self.autosnow_meta = autosnow_meta

    # --------------------------------------------------------
    # 1) Read orbit to DataFrame
    # --------------------------------------------------------
    def load_orbit_df(self, avh_file: str, avh_vars: List[str]):
        """
        Wrapper around read_AVHRR_orbit_to_df + add_time_columns.
        Also builds the regular WGS grid vectors.
        """
        x_vec, y_vec, x, y = utils.build_test_grid(self.grid_res)

        df = read_AVHRR_orbit_to_df(
            avh_file,
            x_vec,
            y_vec,
            x,
            y,
            avh_vars=avh_vars,
            lat_thresh_N_hemisphere=self.lat_thresh_nh,
            lat_thresh_S_hemisphere=self.lat_thresh_sh,
        )

        if df is None or df.empty:
            print(f"[WARN] AVHRR reader returned empty for {avh_file}")
            return None, None, None

        df = utils.add_time_columns(df)
        return df, x_vec, y_vec

    # --------------------------------------------------------
    # 2) Collocate reference datasets
    # --------------------------------------------------------
    def collocate_all_refs(
        self,
        df,
        *,
        merra2_vars: List[str],
        need_imerg: bool,
        need_era5: bool,
    ):
        """
        Dynamically collocate IMERG / ERA5 / MERRA2 / AutoSnow based on flags
        and available metadata.
        AutoSnow + MERRA2 are always needed because they are DL inputs.
        """
        # IMERG
        if need_imerg and (self.imerg_meta is not None):
            df = collocate_IMERG_precip(df, self.imerg_meta)

        # ERA5
        if need_era5 and (self.era5_meta_by_year is not None):
            df = collocate_ERA5_precip(df, self.era5_meta_by_year, varname="tp")

        # MERRA2 (always if meta is provided, since TQV/T2M are DL features)
        if self.merra2_meta is not None and merra2_vars:
            df = collocate_MERRA2(df, self.merra2_meta, MERRA2_vars=merra2_vars)

        # AutoSnow (always if meta is provided, since it's a DL feature)
        if self.autosnow_meta is not None:
            df = collocate_AutoSnow(df, self.autosnow_meta, date_col="scan_date", out_col="AutoSnow")

        return df

    # --------------------------------------------------------
    # 3) Build 2D WGS grids for selected variables
    # --------------------------------------------------------
    def build_var_grids(self, df, x_vec, y_vec, varnames: List[str]) -> Dict[str, np.ndarray]:
        var_grids: Dict[str, np.ndarray] = {}
        for v in varnames:
            if v not in df.columns:
                print(f"[WARN] '{v}' not found in df — skipping grid")
                continue
            var_grids[v] = utils.df2grid(df, v, x_vec, y_vec).astype("float32")
        return var_grids

    # --------------------------------------------------------
    # 4) Reproject WGS grids → polar stereo (NH + SH)
    # --------------------------------------------------------
    def wgs_to_polar(self, var_grids: Dict[str, np.ndarray], orbit_tag: str, x_vec, y_vec):
        return reproject_vars_wgs_to_polar(
            var_grids,
            orbit_tag=orbit_tag,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution=self.grid_res,
            lat_thresh_nh=self.lat_thresh_nh,
            lat_thresh_sh=self.lat_thresh_sh,
            lat_ts_nh=self.lat_ts_nh,
            lat_ts_sh=self.lat_ts_sh,
            nodata=self.nodata,
        )

    # --------------------------------------------------------
    # 5) One-step pipeline for a single orbit
    # --------------------------------------------------------
    def process_orbit(
        self,
        avh_file: str,
        *,
        avh_vars: List[str],
        input_vars: List[str],
        merra2_vars: List[str],
        need_imerg: bool,
        need_era5: bool,
        extra_eval_vars: Optional[List[str]] = None,
        save_polar_dir: Optional[str] = None,
    ):
        """
        Full Stage-1 pipeline for a single AVHRR orbit.

        Parameters
        ----------
        avh_file : str
            Path to raw AVHRR orbit (WGS).
        avh_vars : list[str]
            AVHRR variables to read from orbit (e.g. ["cloud_probability", "temp_11_0um_nom", "temp_12_0um_nom"])
        input_vars : list[str]
            Variables that the DL model will need on grid (e.g. 
            ["cloud_probability", "temp_11_0um_nom", "temp_12_0um_nom", "TQV", "T2M", "AutoSnow"])
        merra2_vars : list[str]
            MERRA2 variable names (e.g. ["TQV", "T2M"])
        need_imerg, need_era5 : bool
            Whether to collocate IMERG/ERA5 (useful for evaluation later).
        extra_eval_vars : list[str] or None
            Additional variables to grid/reproject for evaluation, e.g. ["IMERG_preci","ERA5_tp"].
        save_polar_dir : str or None
            If provided, saves NH/SH polar NetCDFs into this folder via utils.save_polar_netcdf.

        Returns
        -------
        polar : dict
            {
              "NH": {varname: (("y","x"), array), ...},
              "SH": {varname: (("y","x"), array), ...},
            }
        """
        # 1) Read AVHRR orbit
        df, x_vec, y_vec = self.load_orbit_df(avh_file, avh_vars)
        if df is None:
            return None

        # 2) Collocate references
        df = self.collocate_all_refs(
            df,
            merra2_vars=merra2_vars,
            need_imerg=need_imerg,
            need_era5=need_era5,
        )

        orbit_tag = Path(avh_file).stem

        # 3) Decide which variables to put on WGS grid
        #    (DL inputs + optional evaluation vars)
        all_grid_vars = list(dict.fromkeys(
            list(input_vars) + (extra_eval_vars or [])
        ))

        var_grids = self.build_var_grids(df, x_vec, y_vec, all_grid_vars)

        # 4) Reproject all grid variables to polar stereo (NH/SH)
        polar = self.wgs_to_polar(var_grids, orbit_tag, x_vec, y_vec)

        # 5) Optionally save polar NetCDFs
        if save_polar_dir:
            os.makedirs(save_polar_dir, exist_ok=True)
            utils.save_polar_netcdf(
                polar,
                out_dir=save_polar_dir,
                orbit_tag=orbit_tag,
            )

        return polar