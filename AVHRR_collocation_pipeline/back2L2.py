from __future__ import annotations

from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import xarray as xr
import pandas as pd
import toml
from tqdm import tqdm

from AVHRR_collocation_pipeline.retrievers.back_to_L2 import AVHRRBackToL2


def infer_orbit_tag_from_retrieved(retrieved_file: Path) -> str:
    name = retrieved_file.name
    suffix = "_retrieved_wgs.nc"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected retrieved filename format: {name}")
    return name[:-len(suffix)]


def find_raw_orbit(raw_base_dirs: list[str], orbit_tag: str) -> Path:
    target = f"{orbit_tag}.nc"

    for base in raw_base_dirs:
        base_path = Path(base)

        direct = base_path / target
        if direct.exists():
            return direct

        matches = list(base_path.rglob(target))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Multiple raw orbit matches found for {target}: {matches}")

    raise FileNotFoundError(f"Could not find raw orbit for {target}")


def open_retrieved_groups(retrieved_file: Path) -> tuple[xr.Dataset, xr.Dataset]:
    ds_nh = xr.open_dataset(retrieved_file, group="NH").load()
    ds_sh = xr.open_dataset(retrieved_file, group="SH").load()
    return ds_nh, ds_sh


def process_one_file(retrieved_file_str: str, raw_base_dirs: list[str], out_dir_str: str) -> dict:
    retrieved_file = Path(retrieved_file_str)
    out_dir = Path(out_dir_str)

    ds_nh = None
    ds_sh = None

    try:
        orbit_tag = infer_orbit_tag_from_retrieved(retrieved_file)
        raw_orbit_path = find_raw_orbit(raw_base_dirs, orbit_tag)

        ds_nh, ds_sh = open_retrieved_groups(retrieved_file)

        retrieved_names = sorted(set(list(ds_nh.data_vars) + list(ds_sh.data_vars)))

        l2_writer = AVHRRBackToL2(
            retrieved_var_names=retrieved_names,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        l2_out_path = out_dir / f"{orbit_tag}_L2.nc"

        print(f"[DEBUG] Attaching retrievals back to L2: {l2_out_path}", flush=True)

        l2_writer.attach_to_orbit(
            raw_orbit_path=raw_orbit_path,
            ds_nh=ds_nh,
            ds_sh=ds_sh,
            out_path=l2_out_path,
        )

        return {
            "retrieved_file": str(retrieved_file),
            "orbit_tag": orbit_tag,
            "raw_orbit_path": str(raw_orbit_path),
            "output_l2_path": str(l2_out_path),
            "status": "ok",
            "error_type": "",
            "error_message": "",
        }

    except Exception as e:
        return {
            "retrieved_file": str(retrieved_file),
            "orbit_tag": retrieved_file.name.replace("_retrieved_wgs.nc", ""),
            "raw_orbit_path": "",
            "output_l2_path": "",
            "status": "fail",
            "error_type": type(e).__name__,
            "error_message": str(e).replace("\n", " | "),
        }

    finally:
        try:
            if ds_nh is not None:
                ds_nh.close()
        except Exception:
            pass
        try:
            if ds_sh is not None:
                ds_sh.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = toml.load(args.config)

    retrieved_dir = Path(cfg["paths"]["retrieved_dir"])
    raw_base_dirs = list(cfg["paths"]["raw_orbit_dirs"])
    out_dir = Path(cfg["paths"]["out_dir"])
    log_csv = Path(cfg["paths"]["log_csv"])
    n_workers = int(cfg.get("run", {}).get("n_workers", 1))

    files = sorted(retrieved_dir.glob("*_retrieved_wgs.nc"))
    if not files:
        raise SystemExit(f"No retrieved WGS files found in {retrieved_dir}")

    print(f"[INFO] Found {len(files)} retrieved files", flush=True)
    print(f"[INFO] Using n_workers={n_workers}", flush=True)

    rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(process_one_file, str(f), raw_base_dirs, str(out_dir))
            for f in files
        ]

        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc="Back-to-L2",
            unit="file",
            dynamic_ncols=True,
        ):
            row = fut.result()
            rows.append(row)

            if row["status"] == "ok":
                print(f"[OK]   {row['orbit_tag']}", flush=True)
            else:
                print(f"[FAIL] {row['orbit_tag']} | {row['error_type']} | {row['error_message']}", flush=True)

    df = pd.DataFrame(rows)
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_csv, index=False)

    n_ok = (df["status"] == "ok").sum()
    n_fail = (df["status"] == "fail").sum()

    print("\n[SUMMARY]", flush=True)
    print(f"Total:  {len(df)}", flush=True)
    print(f"OK:     {n_ok}", flush=True)
    print(f"Failed: {n_fail}", flush=True)
    print(f"Log:    {log_csv}", flush=True)


if __name__ == "__main__":
    main()