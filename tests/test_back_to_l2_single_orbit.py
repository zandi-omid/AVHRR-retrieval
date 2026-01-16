#%%
from pathlib import Path
import xarray as xr

from AVHRR_collocation_pipeline.retrievers.back_to_L2 import AVHRRBackToL2

def main():
    # 1) Paths
    raw_orbit = Path(
        "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019/"
        "clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2.nc"
    )

    retrieved_nc = Path(
        "/xdisk/behrangi/omidzandi/retrieved_maps/2019/"
        "clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2_retrieved_wgs.nc"
    )

    out_nc = Path(
        "/xdisk/behrangi/omidzandi/retrieved_maps/test/"
        "__clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2__with_retrievals_L2.nc"
    )

    print("Opening retrieved NH/SH grids...")
    ds_nh = xr.open_dataset(retrieved_nc, group="NH")
    ds_sh = xr.open_dataset(retrieved_nc, group="SH")

    print("Attaching retrievals back to L2 swath...")
    attacher = AVHRRBackToL2(
        retrieved_var_names=[
            "retrieved_precip_mean",
            "retrieved_precip_q70",
            "retrieved_precip_q75",
            "retrieved_precip_q80",
        ],
        precip_scale=0.005,
        tb_scale=0.01,
    )

    out_path = attacher.attach_to_orbit(
        raw_orbit_path=raw_orbit,
        ds_nh=ds_nh,
        ds_sh=ds_sh,
        out_path=out_nc,
    )

    print(f"✅ Wrote L2-style file with retrievals: {out_path}")

    # Quick sanity checks
    ds = xr.open_dataset(out_path, decode_timedelta=True)
    print(ds)
    print("Shapes:")
    print("  latitude:", ds["latitude"].shape)
    print("  retrieved_precip_mean:", ds["retrieved_precip_mean"].shape)

    # Check NaN fraction over whole swath
    frac_nan = float(ds["retrieved_precip_mean"].isnull().mean())
    print(f"NaN fraction in retrieved_precip_mean: {frac_nan:.3f}")

    ds.close()


if __name__ == "__main__":
    main()

#%%

from pathlib import Path
import xarray as xr

raw_orbit = Path(
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019/"
    "clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2.nc"
)

l2_with_retr = Path(
    "/xdisk/behrangi/omidzandi/retrieved_maps/test/"
    "__clavrx_NSS.GHRR.M1.D19001.S0716.E0807.B3262828.MM.hirs_avhrr_fusion.level2__with_retrievals_L2.nc"
)

ds_raw = xr.open_dataset(raw_orbit)
ds_out = xr.open_dataset(l2_with_retr)

for name in ["temp_11_0um_nom", "temp_12_0um_nom"]:
    if name in ds_raw:
        print(f"RAW {name}:    min={float(ds_raw[name].min())}, max={float(ds_raw[name].max())}")
    else:
        print(f"RAW missing {name}")

    if name in ds_out:
        print(f"OUT {name}:    min={float(ds_out[name].min())}, max={float(ds_out[name].max())}")
        print(f"OUT {name}:    nan_frac={float(ds_out[name].isnull().mean())}")
    else:
        print(f"OUT missing {name}")

    print("-" * 60)
# %%
