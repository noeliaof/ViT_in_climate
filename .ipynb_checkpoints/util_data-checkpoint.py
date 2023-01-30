import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import dask
import datetime
#from dotenv import dotenv_values

def rename_dimensions_variables(ds):
    """Rename dimensions and attributes of the given dataset to homogenize data."""
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    return ds


def temporal_slice(ds, start, end):
    """Slice along the temporal dimension."""
    ds = ds.sel(time=slice(start, end))

    if 'time_bnds' in ds.variables:
        ds = ds.drop('time_bnds')

    return ds


def spatial_slice(ds, lon_bnds, lat_bnds):
    """Slice along the spatial dimension."""
    if lon_bnds != None:
        ds = ds.sel(lon=slice(min(lon_bnds), max(lon_bnds)))

    if lat_bnds != None:
        if ds.lat[0].values < ds.lat[1].values:
            ds = ds.sel(lat=slice(min(lat_bnds), max(lat_bnds)))
        else:
            ds = ds.sel(lat=slice(max(lat_bnds), min(lat_bnds)))

    return ds


def get_nc_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """Extract netCDF data for the given file(s) pattern/path."""
    print('Extracting data for the period {} - {}'.format(start, end))
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = rename_dimensions_variables(ds)
    ds = temporal_slice(ds, start, end)
    ds = spatial_slice(ds, lon_bnds, lat_bnds)

    return ds


def get_era5_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """Extract ERA5 data for the given file(s) pattern/path."""
    
    return get_nc_data(files, start, end, lon_bnds, lat_bnds)