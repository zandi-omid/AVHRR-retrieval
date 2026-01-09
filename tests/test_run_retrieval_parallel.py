from __future__ import annotations

from pathlib import Path
import os
import gc
import time
import socket
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import torch
import xarray as xr

from AVHRR_collocation_pipeline.readers.AutoSnow_reference import load_AutoSnow_reference
from AVHRR_collocation_pipeline.readers.MERRA2_reference import load_MERRA2_reference

from AVHRR_collocation_pipeline.retrievers.collocate_and_reproj import AVHRRProcessor
from AVHRR_collocation_pipeline.retrievers.retrieve_and_reproj import AVHRRHybridRetriever
from AVHRR_collocation_pipeline.retrievers.reproject import reproject_vars_polar_to_wgs

from pytorch_retrieve.architectures import load_model

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
AVHRR_FOLDERS = [
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019",
]

BASE_OUT = Path("/xdisk/behrangi/omidzandi/retrieved_maps/test_ret_parallel")
BASE_OUT.mkdir(parents=True, exist_ok=True)

GRID_RES = 0.25
LAT_THRESH_NH = 45.0
LAT_THRESH_SH = -45.0
LAT_TS_NH = 70.0
LAT_TS_SH = -71.0

MERRA2_DIR = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/MERRA2/merra2_archive_19800101_20250831"
AUTOSNOW_DIR = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AutoSnow/autosnow_in_geotif"
MERRA2_VARS = ["TQV", "T2M"]

# DL inputs (must match what your model expects, in this order)
INPUT_VARS = [
    "cloud_probability",
    "temp_11_0um_nom",
    "temp_12_0um_nom",
    "TQV",
    "T2M",
    "AutoSnow",
]

AVH_VARS = ["cloud_probability", "temp_11_0um_nom", "temp_12_0um_nom"]

# retrieval output format:
#   - "polar": write retrieved fields on polar grid
#   - "wgs":   reproject retrieved fields to lat/lon
OUT_GRID = "wgs"

CKPT_PATH = (
    "/xdisk/behrangi/omidzandi/DL_Simon_codes/avhrr_retrievals/"
    "checkpoints/AVHRR_efficient_net_v2_pt_1_45_poleward_SH_ERA5_multi_node_keep_all_fp32-v1.ckpt"
)


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def list_avhrr_files(folders: list[str]) -> list[Path]:
    files = []
    for f in folders:
        p = Path(f)
        files.extend(sorted(p.glob("*.nc")))
    return files


def extract_orbit_tag(avh_file: Path) -> str:
    return avh_file.stem


# ------------------------------------------------------------
# CPU finalization: reprojection + TB11-mask + NetCDF write
# runs inside CPU thread pool
# ------------------------------------------------------------
def cpu_finalize_orbit(
    *,
    orbit_tag: str,
    out_nc: Path,
    OUT_GRID: str,
    preds_nh: dict,
    preds_sh: dict,
    xvec_nh,
    yvec_nh,
    xvec_sh,
    yvec_sh,
    tb11_wgs,
    x_vec_global,
    y_vec_global,
    retriever: AVHRRHybridRetriever,
) -> None:
    """
    CPU-only stage:
      - polar -> WGS reprojection (if OUT_GRID == 'wgs')
      - optional TB11 WGS masking
      - write NetCDF with NH / SH groups
    """
    try:
        if out_nc.exists():
            out_nc.unlink()

        # ----------------- build NH/SH datasets -----------------
        if OUT_GRID == "polar":
            ds_nh = xr.Dataset(
                {
                    "retrieved_precip_mean": (("y", "x"), preds_nh["mean"]),
                    "retrieved_precip_q70":  (("y", "x"), preds_nh["q70"]),
                    "retrieved_precip_q75":  (("y", "x"), preds_nh["q75"]),
                    "retrieved_precip_q80":  (("y", "x"), preds_nh["q80"]),
                },
                coords={"x": xvec_nh, "y": yvec_nh},
            )
            ds_sh = xr.Dataset(
                {
                    "retrieved_precip_mean": (("y", "x"), preds_sh["mean"]),
                    "retrieved_precip_q70":  (("y", "x"), preds_sh["q70"]),
                    "retrieved_precip_q75":  (("y", "x"), preds_sh["q75"]),
                    "retrieved_precip_q80":  (("y", "x"), preds_sh["q80"]),
                },
                coords={"x": xvec_sh, "y": yvec_sh},
            )

        elif OUT_GRID == "wgs":
            # --- polar -> WGS via central utility ---
            var_arrays_nh = {
                "retrieved_precip_mean": preds_nh["mean"],
                "retrieved_precip_q70":  preds_nh["q70"],
                "retrieved_precip_q75":  preds_nh["q75"],
                "retrieved_precip_q80":  preds_nh["q80"],
            }
            var_arrays_sh = {
                "retrieved_precip_mean": preds_sh["mean"],
                "retrieved_precip_q70":  preds_sh["q70"],
                "retrieved_precip_q75":  preds_sh["q75"],
                "retrieved_precip_q80":  preds_sh["q80"],
            }

            ds_nh = reproject_vars_polar_to_wgs(
                var_arrays_nh,
                hemisphere="NH",
                x_vec=xvec_nh,
                y_vec=yvec_nh,
                grid_resolution_deg=GRID_RES,
                lat_ts_nh=LAT_TS_NH,
                lat_ts_sh=LAT_TS_SH,
                nodata=float("nan"),
                tag=f"{orbit_tag}_NH",
            )
            ds_sh = reproject_vars_polar_to_wgs(
                var_arrays_sh,
                hemisphere="SH",
                x_vec=xvec_sh,
                y_vec=yvec_sh,
                grid_resolution_deg=GRID_RES,
                lat_ts_nh=LAT_TS_NH,
                lat_ts_sh=LAT_TS_SH,
                nodata=float("nan"),
                tag=f"{orbit_tag}_SH",
            )

            # --- mask extra pixels using global TB11 WGS swath ---
            ds_nh = retriever.mask_ds_with_tb11_wgs(ds_nh, tb11_wgs, x_vec_global, y_vec_global)
            ds_sh = retriever.mask_ds_with_tb11_wgs(ds_sh, tb11_wgs, x_vec_global, y_vec_global)

        else:
            raise ValueError("OUT_GRID must be 'polar' or 'wgs'")

        # ----------------- write NetCDF with NH/SH groups -----------------
        retriever.write_orbit_netcdf(
            out_path=out_nc,
            ds_nh=ds_nh,
            ds_sh=ds_sh,
            var_scales={
                "retrieved_precip_mean": 0.005,
                "retrieved_precip_q70":  0.005,
                "retrieved_precip_q75":  0.005,
                "retrieved_precip_q80":  0.005,
            },
            default_scale=0.005,
        )

    except Exception as e:
        print(f"❌ [CPU] Error in finalize for {orbit_tag}: {e}", flush=True)
        if out_nc.exists():
            try:
                out_nc.unlink()
            except Exception:
                pass
    finally:
        gc.collect()


# ------------------------------------------------------------
# Per-orbit GPU part (Stage-1 + Stage-2)
# ------------------------------------------------------------
def gpu_stage_for_orbit(
    avh_file: Path,
    processor: AVHRRProcessor,
    retriever: AVHRRHybridRetriever,
    merra2_meta,
    autosnow_meta,
):
    """
    Does:
      - Stage-1: collocate + WGS->polar for this orbit (NH/SH)
      - Stage-2: GPU tiled retrieval for NH/SH

    Returns:
      dict with:
        orbit_tag, preds_nh, preds_sh,
        xvec_nh, yvec_nh, xvec_sh, yvec_sh,
        tb11_wgs, x_vec_global, y_vec_global
    """
    orbit_tag = extract_orbit_tag(avh_file)

    # --- Stage-1: collocate + WGS->polar --- #
    ds_polar, tb11_wgs, x_vec_global, y_vec_global = processor.process_orbit(
        avh_file=avh_file,
        avh_vars=AVH_VARS,
        input_vars=INPUT_VARS,
        merra2_vars=MERRA2_VARS,
        out_polar_nc=BASE_OUT / f"_{orbit_tag}__polar_inputs.nc",
    )

    ds_nh_polar = ds_polar["NH"]
    ds_sh_polar = ds_polar["SH"]

    # --- Load NH/SH inputs from polar datasets --- #
    x_nh, tb12_nh, xvec_nh, yvec_nh = retriever.load_group_inputs_from_dataset(ds_nh_polar)
    x_sh, tb12_sh, xvec_sh, yvec_sh = retriever.load_group_inputs_from_dataset(ds_sh_polar)

    # --- GPU inference --- #
    preds_nh = retriever.gpu_predict_tiled_multiquantile(x_nh)
    preds_nh = retriever.clean_precip(preds_nh, min_val=0.0, max_val=50.0, drizzle=0.001)
    preds_nh = retriever.mask_preds_with_tb12(preds_nh, tb12_nh)

    preds_sh = retriever.gpu_predict_tiled_multiquantile(x_sh)
    preds_sh = retriever.clean_precip(preds_sh, min_val=0.0, max_val=50.0, drizzle=0.001)
    preds_sh = retriever.mask_preds_with_tb12(preds_sh, tb12_sh)

    # Free big arrays early (ds_polar will go out of scope when this returns)
    del x_nh, x_sh, ds_nh_polar, ds_sh_polar
    gc.collect()

    return {
        "orbit_tag": orbit_tag,
        "preds_nh": preds_nh,
        "preds_sh": preds_sh,
        "xvec_nh": xvec_nh,
        "yvec_nh": yvec_nh,
        "xvec_sh": xvec_sh,
        "yvec_sh": yvec_sh,
        "tb11_wgs": tb11_wgs,
        "x_vec_global": x_vec_global,
        "y_vec_global": y_vec_global,
    }


# ------------------------------------------------------------
# MAIN: multi-rank + GPU + CPU threadpool
# ------------------------------------------------------------
def main():
    print("=== AVHRR end-to-end retrieval (multi-rank + GPU + CPU threads) ===")

    # ---------- list all AVHRR orbits ---------- #
    all_files = list_avhrr_files(AVHRR_FOLDERS)
    if not all_files:
        print("No AVHRR .nc files found.")
        return

    # ---------- multi-rank splitting (SLURM) ---------- #
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size  = int(os.environ.get("SLURM_NTASKS", 1))
    node_name   = socket.gethostname()

    my_files = all_files[global_rank::world_size]

    print(
        f"[Rank {global_rank}/{world_size}] node={node_name}, "
        f"total_files={len(all_files)}, my_files={len(my_files)}",
        flush=True,
    )

    if not my_files:
        print(f"[Rank {global_rank}] No files assigned, exiting.")
        return

    # ---------- device / model ---------- #
    n_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", global_rank % max(1, max(n_gpus, 1))))

    if torch.cuda.is_available() and n_gpus > 0:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(1)

    print(f"[Rank {global_rank}] Using device={device}, GPUs on node={n_gpus}", flush=True)

    model = load_model(CKPT_PATH).to(device).eval()
    retriever = AVHRRHybridRetriever(
        model=model,
        device=device,
        input_vars=INPUT_VARS,
        tile_size=1536,
        overlap=64,
        out_grid_resolution_deg=GRID_RES,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
    )

    # ---------- references (shared across orbits on this rank) ---------- #
    print(f"[Rank {global_rank}] Loading references (MERRA2, AutoSnow)...", flush=True)
    merra2_meta = load_MERRA2_reference(MERRA2_DIR)
    autosnow_meta = load_AutoSnow_reference(AUTOSNOW_DIR)

    processor = AVHRRProcessor(
        grid_res=GRID_RES,
        lat_thresh_nh=LAT_THRESH_NH,
        lat_thresh_sh=LAT_THRESH_SH,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
        nodata=-9999.0,
        merra2_meta=merra2_meta,
        autosnow_meta=autosnow_meta,
    )

    # ---------- CPU thread pool for finalization ---------- #
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    env_reproj = os.environ.get("REPROJ_THREADS", "").strip()
    if env_reproj:
        reproj_threads = int(env_reproj)
    else:
        reproj_threads = max(1, cpus_per_task - 1)

    max_pending = int(os.environ.get("MAX_PENDING_REPROJ", "4"))

    print(
        f"[Rank {global_rank}] Using {reproj_threads} CPU threads + MAX_PENDING_REPROJ={max_pending}",
        flush=True,
    )

    writer_pool = ThreadPoolExecutor(max_workers=reproj_threads)
    futures = []

    start_time = time.time()

    for avh_file in my_files:
        orbit_tag = extract_orbit_tag(avh_file)
        out_nc = BASE_OUT / f"{orbit_tag}_retrieved_{OUT_GRID}.nc"

        print(f"[Rank {global_rank}] >>> Orbit {orbit_tag}", flush=True)

        try:
            # ---------- GPU part (Stage-1 + Stage-2) ---------- #
            gpu_out = gpu_stage_for_orbit(
                avh_file=avh_file,
                processor=processor,
                retriever=retriever,
                merra2_meta=merra2_meta,
                autosnow_meta=autosnow_meta,
            )

            # ---------- CPU part: submit to threadpool ---------- #
            fut = writer_pool.submit(
                cpu_finalize_orbit,
                orbit_tag=gpu_out["orbit_tag"],
                out_nc=out_nc,
                OUT_GRID=OUT_GRID,
                preds_nh=gpu_out["preds_nh"],
                preds_sh=gpu_out["preds_sh"],
                xvec_nh=gpu_out["xvec_nh"],
                yvec_nh=gpu_out["yvec_nh"],
                xvec_sh=gpu_out["xvec_sh"],
                yvec_sh=gpu_out["yvec_sh"],
                tb11_wgs=gpu_out["tb11_wgs"],
                x_vec_global=gpu_out["x_vec_global"],
                y_vec_global=gpu_out["y_vec_global"],
                retriever=retriever,
            )
            futures.append(fut)

            # Drop references from main thread (thread has its own copies)
            del gpu_out
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # throttle number of pending CPU jobs
            if len(futures) >= max_pending:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                futures = list(not_done)

        except Exception as e:
            print(f"[Rank {global_rank}] ❌ Error on orbit {orbit_tag}: {e}", flush=True)
            if out_nc.exists():
                try:
                    out_nc.unlink()
                except Exception:
                    pass

    # ---------- wait for CPU jobs ---------- #
    if futures:
        wait(futures)
    writer_pool.shutdown(wait=True)

    # cleanup
    del model, retriever, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dt = time.time() - start_time
    print(f"[Rank {global_rank}] Done all orbits in {dt/3600:.2f} hours.", flush=True)


if __name__ == "__main__":
    main()