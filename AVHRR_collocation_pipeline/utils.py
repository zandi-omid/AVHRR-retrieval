#%% import required packages
import numpy as np
from glob import glob
import os 
import pandas as pd
from pyhdf.SD import SD, SDC
from pyhdf import HDF, VS, V
import h5py
from PIL import Image
from netCDF4 import Dataset
import datetime
import rasterio
from rasterio.transform import Affine, from_origin
import xarray as xr
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from typing import Tuple
import random
from typing import Union, List





#%% functions
def dir_files(directory, extension):
    
    
    """"
    Returns an array of files in the specified directory that have the defined
    extension.
    
    :param directory: Directory of interest with directory separator at the end
    :param extension: File extension of interest
    :return: A one dimenstional array of files
    """

    files = np.array(glob(directory + '*.{}'. format(extension)))
    files.sort()
    return files

def AutoSnow_datetime(AutoSnow_file):
    
    """"
    Returns datetime index of the introduced AutoSnow file.
    
    :param AutoSnow_file: AutoSnow file
    :return: Datetime index of the AutoSnow file
    """
    
    filename = os.path.basename(AutoSnow_file)
    dt = pd.to_datetime(filename[21:29], format='%Y%m%d')
    return dt

def AutoSnow_HDF_SDreader(AutoSnow_file, variable):
    
    """"
    Returns variable data from the introduced AutoSnow file.
    
    :param AutoSnow_file: AutoSnow file
    :return: Data for variable of interest
    """
    
    sd = SD(AutoSnow_file, SDC.READ)
    sds_obj = sd.select(variable)
    data = sds_obj.get()
    sd.end()
    return data

def CloudSat_datetime(CloudSat_file):
    
    """"
    Returns datetime index of the introduced CloudSat file.
    
    :param CloudSat_file: CloudSat file
    :return: Datetime index of the CloudSat file
    """
    
    filename = os.path.basename(CloudSat_file)
    dt = pd.to_datetime(filename[:13], format='%Y%j%H%M%S')
    return dt


def CloudSat_datetime_(CloudSat_file):
    
    """"
    Returns datetime index of the introduced CloudSat file.
    
    :param CloudSat_file: CloudSat file
    :return: Datetime index of the CloudSat file
    """
    
    filename = os.path.basename(CloudSat_file)
    dt = datetime.datetime.strptime(filename[:13], '%Y%j%H%M%S')
    return dt

def CloudSat_HDF_SDreader(CloudSat_file, variable):
    
    """"
    Returns variable data from the introduced CloudSat file.
    
    :param CloudSat_file: CloudSat file
    :return: Data for variable of interest
    """
    
    sd = SD(CloudSat_file, SDC.READ)
    sds_obj = sd.select(variable)
    data = sds_obj.get()
    fill_value = sds_obj.attributes()['_FillValue']
    # data = data.astype(float) # This line might be probably needed. I add it.
    data[data == fill_value] = np.nan
    sd.end()
    return data






def IMERG_file_datetime(IMERG_file):
    
    """"
    Returns datetime index of the introduced IMERG file.
    
    :param IMERG_file: IMERG file
    :return: Datetime index of the IMERG file
    """
    
    filename = os.path.basename(IMERG_file)
    dt = pd.to_datetime(filename.split('.')[4][:8], format='%Y%m%d')
    return dt

def tif_to_array(tif_file):
    
    """"
    Returns tif file as a numpy array.
    
    :param tif_file: Tif file
    :return: Equivalent numpy array
    """
    
    image = Image.open(tif_file)
    array = np.array(image)
    return array

def IMERG_NCreader(IMERG_file, variable):
    
    """"
    Returns variable data from the introduced IMERG file.
    
    :param IMERG_file: IMERG file
    :return: Data for variable of interest
    """
    
    with Dataset(IMERG_file) as nc:
        data = nc[variable][:]
        data.data[data.mask]=np.nan
        fill_value = nc[variable].getncattr('_FillValue')
        data.data[data == fill_value] = np.nan
    return data.data.squeeze()



def MERRA2_datetime(MERRA2_file):
    
    """"
    Returns datetime index of the MERRA2 file.
    
    :param MERRA2_file: MERRA2 file
    :return: Datetime index of the MERRA2 file
    """
    
    filename = os.path.basename(MERRA2_file)
    dt = pd.to_datetime(filename[27:35], format='%Y%m%d').date()
    return dt

def MERRA2_datetime_(MERRA2_file):
    
    """"
    Returns datetime index of the MERRA2 file.
    
    :param MERRA2_file: MERRA2 file
    :return: Datetime index of the MERRA2 file
    """
    
    dt = os.path.basename(MERRA2_file)[27:35]
    return dt

def MERRA2_HDF_SDreader(MERRA2_file, field):
    
    """"
    Returns filed data from the MERRA2 file.
    
    :param MERRA2_file: MERRA2 file
    :return: Data for the filed of interest
    """
    
    sd = SD(MERRA2_file, SDC.READ)
    dset = sd.select(field)
    data = dset.get()
    attrs = dset.attributes(full=1)
    if 'missing_value' in attrs:
        fval = attrs['missing_value'][0]
        data[data==fval] = np.nan
    sd.end()
    return data

def AVHRR_datetime(AVHRR_file):
    basename = os.path.basename(AVHRR_file)
    st_str = basename[13:18] + basename[20:24]
    et_str = basename[13:18] + basename[26:30]
    st = pd.to_datetime(st_str, format='%y%j%H%M')
    et = pd.to_datetime(et_str, format='%y%j%H%M')
    if et < st:
        #files which ending time is few minutes after 00:00 pm
        et = et + pd.Timedelta(1, 'day')
    return st, et

def AVHRR_datetime_NC_files(AVHRR_file):

    name_splited = os.path.basename(AVHRR_file).split('.')

    yr_DOY = name_splited[3][1:]
    st_time = name_splited[4][1:]
    end_time = name_splited[5][1:]
    yr = datetime.datetime.strptime(yr_DOY, '%y%j').year

    st_dt = datetime.datetime.strptime(yr_DOY + st_time, '%y%j%H%M')

    end_dt = datetime.datetime.strptime(yr_DOY + end_time, '%y%j%H%M')

    if end_dt < st_dt:
        #files which ending time is few moments after 00:00 pm
        end_dt = end_dt + datetime.timedelta(days=1)

    return st_dt, end_dt, yr

def gridder(ref_grid, grid_resolution, lon, lat, variable):

    """"
    Returns grided data from 1-D arrays of latitudes, longitudes, and measured variable
    
    :param resolution: the final resolution of the grided data
    :param lon, lat: the longitude and latitude of measured variable (1-D array)
    :param variable: the value of the measured variable (1-D array)
    :return: grided dataset over the whole world with desired resolution
    """

    idx = np.digitize(lon, np.arange(-180, 180.1, grid_resolution).round(1)) - 1
    idy = np.digitize(lat, np.arange(-90, 90.1, grid_resolution).round(1)) - 1
    df = pd.DataFrame({'x':idx, 'y':idy, 'z':variable})
    df = df[(df.x != -1) | (df.y != -1)].copy()
    mean_df = df.groupby(['x', 'y']).mean()
    ref_grid[mean_df.index.get_level_values(1), mean_df.index.get_level_values(0)] = mean_df['z']
    
    return ref_grid


def reduce_mem_usage(df):
    # start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if (col == 'scan_line_times') | (col == 'profile_time') | (col == 'lon') | (col == 'lat') | (col == 'IMERG_preci'):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    # end_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def trunc(values, decs=1):
    return np.trunc(values*10**decs)/(10**decs)


def lat_lon_vectorts_from_tiff(tiff_file):

    """"
    Returns latitude and longitude corresponding to left edge of each pixels of the tiff file.
    
    :param tiff_file: the path of the desired tiff file.
    :return: latitude and longitude vectors.
    """

    with rasterio.open(tiff_file) as src:
        array = src.read(1)
        lat = np.arange(src.transform[5], round(src.transform[5] + (array.shape[0]) * src.transform[4], 2), src.transform[4])
        lon = np.arange(src.transform[2], round(src.transform[2] + (array.shape[1]) * src.transform[0], 2), src.transform[0])
        return lat, lon, array
    

def lat_lon_vectors_from_tiff_rasterio_func(tiff_file):
    """
    Returns 1D latitude and longitude vectors representing the pixel **edges**
    of the raster grid.

    :param tiff_file: Path to the GeoTIFF file.
    :return: (lat_vector, lon_vector), each with length = array.shape + 1
    """
    with rasterio.open(tiff_file) as src:
        array = src.read(1)
        height, width = array.shape
        transform = src.transform

        dx = transform.a  # pixel width
        dy = transform.e  # pixel height (usually negative)

        # x0 = transform.c - dx / 2  # left edge of first column
        # y0 = transform.f - dy / 2  # top edge of first row

        lon = transform.c + dx * np.arange(width + 1)
        lat = transform.f + dy * np.arange(height + 1)

        return lat, lon, array


def index_finder(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_bins: np.ndarray,
    lat_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map input longitude/latitude values to grid-cell indices
    defined by target longitude and latitude bin edges/vectors.

    Parameters
    ----------
    lon : np.ndarray
        1D or flattened array of longitudes (any order).
    lat : np.ndarray
        1D or flattened array of latitudes (any order).
    lon_bins : np.ndarray
        1D array of grid longitude centers *in ascending order*
        (e.g., -180, -179.5, ...).
    lat_bins : np.ndarray
        1D array of grid latitude centers *in descending order*
        (e.g., 90, 89.5, ...).

    Returns
    -------
    idx : np.ndarray
        X-index for each lon element (−1 where outside grid).
    idy : np.ndarray
        Y-index for each lat element (−1 where outside grid).

    Notes
    -----
    - Uses np.digitize() for fast binning.
    - Any point outside the grid returns index -1.
    - This function is consistent with the AVHRR reader and df2grid.
    """

    # Ensure numpy arrays
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Digitize longitudes
    idx = np.digitize(lon, lon_bins) - 1

    # Latitude digitization requires right=True because bins decrease
    idy = np.digitize(lat, lat_bins, right=True) - 1

    # Mask out-of-range values
    idx[(idx < 0) | (idx >= len(lon_bins))] = -1
    idy[(idy < 0) | (idy >= len(lat_bins))] = -1

    return idx, idy


def reference_coordinate_maker(grid_resolution):

    """"
    Returns the reference coordiante matrices (and vectors). It should be noted that the coordinate of the left and upper edge of
    each longitude and latitude pixels are returned, respectively. 

    :param grid_resolution: the desired spatial resolution of the grid.
    """""
    
    x_vec = np.arange(-180, 180, grid_resolution).round(2)
    y_vec = np.arange(90, -90, -grid_resolution).round(2)
    x, y = np.meshgrid(x_vec, y_vec)

    return x_vec, y_vec, x, y


def KGE_fn(sim, obs):
    beta  = np.mean(sim)/np.mean(obs)
    alpha = np.std(sim)/np.std(obs)
    rho   = np.corrcoef(sim, obs)[0,1]
    KGE   = 1 - np.sqrt((1-alpha)**2 + (1-beta)**2 + (1-rho)**2)
    KGEss = (KGE - (1-np.sqrt(2)))/np.sqrt(2)
    return [KGEss, KGE, alpha, beta, rho]

def MSE_Fn(sim,obs):
    MSE = np.mean((sim-obs)**2)
    return MSE




def IMERG_datetime_NC_files(IMERG_file):

    filename = os.path.basename(IMERG_file)
    dt = datetime.datetime.strptime(filename.split('.')[4][:8], '%Y%m%d')

    return dt



def IMERG_half_hourly_datetime(file):

    name_splited = os.path.basename(file).split('.')

    time = datetime.datetime.strptime(name_splited[4].split('-')[0] + name_splited[4].split('-')[1][1:], '%Y%m%d%H%M%S')

    return time


def IMERG_half_hourly_reader(file):

    with h5py.File(file, "r") as h5:
        preci = np.flip(h5['/Grid']['precipitation'][:][0].transpose(), axis = 0)
        # preci_16 = np.where(preci == -9999.9, np.nan, preci).astype(np.float16)
        preci = np.where(preci == -9999.9, np.nan, preci)
        # preci = np.where(preci == -9999.9, np.nan, preci).astype(np.float64)

    return preci




def AVHRR_start_end_time(AVHRR_file):

    name_splited = os.path.basename(AVHRR_file).split('.')
    dt = int(name_splited[4][1:] + name_splited[5][1:])

    return dt


def df2grid(
    df,
    var_name: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
) -> np.ndarray:
    """
    Convert a DataFrame with lon/lat and variable into a 2D grid,
    using utils.index_finder for consistency with the main AVHRR pipeline.
    """

    # Extract lon/lat and variable as arrays
    lon = df["lon"].values
    lat = df["lat"].values
    var = df[var_name].values

    # Use the SAME grid mapper as the AVHRR reader
    idx, idy = index_finder(lon, lat, x_vec, y_vec)

    # Only keep points that fall on the grid
    valid = (idx >= 0) & (idy >= 0) & (~np.isnan(var))

    grid = np.full((len(y_vec), len(x_vec)), np.nan, dtype=float)
    grid[idy[valid], idx[valid]] = var[valid]

    return grid



def save_grid_to_tiff(
    grid: np.ndarray,
    outfile: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    nodata: float = np.nan,
) -> None:
    """
    Save a 2D numpy grid (lat/lon aligned) to a GeoTIFF file.

    Parameters
    ----------
    grid : np.ndarray
        2D array (lat-major) of values.
    outfile : str
        Path to output .tif file.
    x_vec : np.ndarray
        1D longitude centers (ascending).
    y_vec : np.ndarray
        1D latitude centers (descending).
    nodata : float, optional
        Value to write for missing data (default: NaN).

    Notes
    -----
    - CRS = WGS84 (EPSG:4326)
    - Uses regular-grid transform derived from x_vec/y_vec.
    """

    # -------------------------
    # Validate array dimensions
    # -------------------------
    if grid.ndim != 2:
        raise ValueError("Grid must be a 2D numpy array")

    if len(x_vec) != grid.shape[1]:
        raise ValueError(f"x_vec length {len(x_vec)} does not match grid width {grid.shape[1]}")

    if len(y_vec) != grid.shape[0]:
        raise ValueError(f"y_vec length {len(y_vec)} does not match grid height {grid.shape[0]}")

    # -------------------------
    # Build raster transform
    # -------------------------
    resolution_x = x_vec[1] - x_vec[0]
    resolution_y = y_vec[0] - y_vec[1]

    transform = from_origin(
        west=x_vec.min(),
        north=y_vec.max(),
        xsize=resolution_x,
        ysize=abs(resolution_y),
    )

    # -------------------------
    # Ensure directory exists
    # -------------------------
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # -------------------------
    # Write GeoTIFF
    # -------------------------
    metadata = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
    }

    grid_to_write = grid.astype(np.float32)

    with rasterio.open(outfile, "w", **metadata) as dst:
        dst.write(grid_to_write, 1)

def resample_to_0_1_degree(data, lat, lon, time):
    """
    Resamples 3D data (time, lat, lon) to a 0.1-degree resolution using bilinear interpolation.

    :param data: The 3D data array with dimensions (time, lat, lon)
    :param lat: The latitude array (1D)
    :param lon: The longitude array (1D)
    :param time: The time array (1D)
    :return: Resampled data, resampled latitudes, resampled longitudes
    """

    # Define the target 0.1-degree resolution grid
    # target_lat = np.arange(90, -90.1, -0.1)
    # target_lon = np.arange(-180, 180.1, 0.1)

    target_lat = np.arange(90, -90, -0.1)
    target_lon = np.arange(-180, 180, 0.1)

    # Create the xarray Dataset with the full 3D data
    ds = xr.Dataset(
        {
            "data": (["time", "lat", "lon"], data)
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
    )

    # Resample the data using bilinear interpolation
    resampled_data = ds.interp(lat=target_lat, lon=target_lon, method="linear")["data"]

    array = resampled_data.values

    # Identify the first and last non-NaN rows and columns
    non_nan_rows = np.where(~np.isnan(array).all(axis=(0, 2)))[0]  # Check along lat and lon dimensions
    non_nan_cols = np.where(~np.isnan(array).all(axis=(0, 1)))[0]  # Check along time and lat dimensions

    # Replace NaN rows by copying the last valid row across all time slices
    if len(non_nan_rows) > 0:
        first_valid_row = non_nan_rows[0]
        array[:, :first_valid_row, :] = array[:, first_valid_row:first_valid_row + 1, :]  # Fill rows above the first valid
        last_valid_row = non_nan_rows[-1]
        array[:, last_valid_row + 1:, :] = array[:, last_valid_row:last_valid_row + 1, :]  # Fill rows below the last valid

    # Replace NaN columns by copying the last valid column across all time slices
    if len(non_nan_cols) > 0:
        first_valid_col = non_nan_cols[0]
        array[:, :, :first_valid_col] = array[:, :, first_valid_col:first_valid_col + 1]  # Fill columns left of the first valid
        last_valid_col = non_nan_cols[-1]
        array[:, :, last_valid_col + 1:] = array[:, :, last_valid_col:last_valid_col + 1]  # Fill columns right of the last valid

    return array, target_lat, target_lon

def resample_to_0_1_degree_NC(ds, lat, lon, time):
    """
    Resamples 3D data (time, lat, lon) to a 0.1-degree resolution using bilinear interpolation.

    :param data: The 3D data array with dimensions (time, lat, lon)
    :param lat: The latitude array (1D)
    :param lon: The longitude array (1D)
    :param time: The time array (1D)
    :return: Resampled data, resampled latitudes, resampled longitudes
    """

    # Define the target 0.1-degree resolution grid
    # target_lat = np.arange(90, -90.1, -0.1)
    # target_lon = np.arange(-180, 180.1, 0.1)

    target_lat = np.arange(90, -90, -0.1)
    target_lon = np.arange(-180, 180, 0.1)

    # Resample the data using bilinear interpolation
    resampled_data = ds.interp(lat=target_lat, lon=target_lon, method="linear")["data"]

    array = resampled_data.values

    # Identify the first and last non-NaN rows and columns
    non_nan_rows = np.where(~np.isnan(array).all(axis=(0, 2)))[0]  # Check along lat and lon dimensions
    non_nan_cols = np.where(~np.isnan(array).all(axis=(0, 1)))[0]  # Check along time and lat dimensions

    # Replace NaN rows by copying the last valid row across all time slices
    if len(non_nan_rows) > 0:
        first_valid_row = non_nan_rows[0]
        array[:, :first_valid_row, :] = array[:, first_valid_row:first_valid_row + 1, :]  # Fill rows above the first valid
        last_valid_row = non_nan_rows[-1]
        array[:, last_valid_row + 1:, :] = array[:, last_valid_row:last_valid_row + 1, :]  # Fill rows below the last valid

    # Replace NaN columns by copying the last valid column across all time slices
    if len(non_nan_cols) > 0:
        first_valid_col = non_nan_cols[0]
        array[:, :, :first_valid_col] = array[:, :, first_valid_col:first_valid_col + 1]  # Fill columns left of the first valid
        last_valid_col = non_nan_cols[-1]
        array[:, :, last_valid_col + 1:] = array[:, :, last_valid_col:last_valid_col + 1]  # Fill columns right of the last valid

    return array, target_lat, target_lon




def pick_random_nc_file(folders: Union[str, List[str]]) -> str:
    """
    Return a random NetCDF (.nc) file from one or more directories.

    Parameters
    ----------
    folders : str or list[str]
        One folder (as a string) or multiple folders (as list of strings).

    Returns
    -------
    str
        Path to a randomly selected .nc file.

    Raises
    ------
    FileNotFoundError
        If no .nc files exist in any provided directory.
    TypeError
        If folders is not a string or list of strings.
    """

    # Normalize input to list
    if isinstance(folders, str):
        folder_list = [folders]
    elif isinstance(folders, list):
        folder_list = folders
    else:
        raise TypeError(
            "folders must be a string or a list of strings, "
            f"received type={type(folders).__name__}"
        )

    all_files: List[str] = []

    for folder in folder_list:
        if not os.path.isdir(folder):
            print(f"WARNING: Directory does not exist → {folder}")
            continue

        nc_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".nc")
        ]

        if len(nc_files) == 0:
            print(f"WARNING: No .nc files found in → {folder}")

        all_files.extend(nc_files)

    if len(all_files) == 0:
        raise FileNotFoundError(
            "No .nc files found in the provided folder(s): "
            f"{folder_list}"
        )

    return random.choice(all_files)


def build_test_grid(resolution: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a global lat/lon grid for testing.

    Parameters
    ----------
    resolution : float, optional
        Grid spacing in degrees. Default is 0.5°.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        x_vec : 1D array of longitudes
        y_vec : 1D array of latitudes
        x : 2D longitude grid
        y : 2D latitude grid
    """
    x_vec = np.arange(-180, 180, resolution)
    y_vec = np.arange(90, -90, -resolution)
    x, y = np.meshgrid(x_vec, y_vec)
    return x_vec, y_vec, x, y

import os
from typing import Optional


def build_tiff_name(
    avh_file: str,
    var_name: str,
    out_dir: str,
) -> str:
    """
    Build a standardized output filename for saving a gridded variable
    from an AVHRR orbit or the collocated dataset as a GeoTIFF file.

    The naming convention is:
        <orbit_basename>__<variable_name>.tif

    Parameters
    ----------
    avh_file : str
        Path to the original AVHRR orbit NetCDF file.
    var_name : str
        Name of the variable being exported (e.g., "temp_12_0um_nom").
    out_dir : str
        Directory where the TIFF file will be saved.

    Returns
    -------
    str
        Full path to the output TIFF file.

    Notes
    -----
    - The `.nc` extension of the input file is removed.
    """

    base = os.path.basename(avh_file)          # "clavrx_NSS....nc"
    base_no_ext = os.path.splitext(base)[0]    # "clavrx_NSS...."

    fname = f"{var_name}_{base_no_ext}.tif"   # standardized name

    return os.path.join(out_dir, fname)



def add_time_columns(
    df: pd.DataFrame,
    time_col: str = "scan_line_times",
    add_nearest_hour: bool = True,
    add_nearest_halfhour: bool = True,
) -> pd.DataFrame:
    """
    Add standardized time columns to df once, for use by ALL collocators.

    Requires:
      df[time_col] = UNIX seconds (int/float)

    Adds:
      - scan_dt               datetime64[s]
      - scan_date             str "YYYY-MM-DD"
      - scan_hour             int16 0..23
      - scan_hour_unix        int64 UNIX sec (nearest hour)      [optional]
      - scan_halfhour_unix    int64 UNIX sec (nearest 30-min)    [optional]
    """
    out = df.copy()

    t = out[time_col].to_numpy().astype("int64")
    scan_dt = t.astype("datetime64[s]")          # seconds resolution

    # date + hour (vectorized)
    day = scan_dt.astype("datetime64[D]")
    out["scan_dt"] = scan_dt
    out["scan_date"] = day.astype(str)

    hour = (scan_dt.astype("datetime64[h]") - day).astype("timedelta64[h]").astype(int)
    out["scan_hour"] = hour.astype(np.int16)

    if add_nearest_hour:
        # nearest hour: floor((t + 1800)/3600)*3600
        out["scan_hour_unix"] = ((t + 1800) // 3600 * 3600).astype("int64")

    if add_nearest_halfhour:
        # nearest 30 min: floor((t + 900)/1800)*1800
        out["scan_halfhour_unix"] = ((t + 900) // 1800 * 1800).astype("int64")

    return out