#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:38:55 2024

@author: af
"""
import xarray as xr
import xesmf as xe
from pathlib import Path
from o_func import opsys; start_path = Path(opsys())
# Paths to the model datasets
delft_grid = start_path / Path('modelling_DATA/kent_estuary_project/6.Final2/models/kent_1.0.0_UM_wind/kent_31_merged_map.nc')
mo_grid = start_path / Path('Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_T.nc')



# Load the Delft3D FM dataset (unstructured grid)
delft_ds = xr.open_dataset(delft_grid)
delft_ds = delft_ds.rename({'mesh2d_face_x': 'lon', 'mesh2d_face_y': 'lat'})

# Load the Met Office dataset (structured grid)
mo_ds = xr.open_dataset(mo_grid, engine = 'h5netcdf')

# Extract the surface height variable
delft_surf_height = delft_ds['mesh2d_s1']  # Assuming 'mesh2d_s1' is surface height
mo_surf_height = mo_ds['sossheig']  # Met Office surface height

# Prepare source and target grids for regridding
# Delft3D (unstructured grid)
delft_lat = delft_ds['lat']  # Assuming 'mesh2d_node_y' is latitude
delft_lon = delft_ds['lon']  # Assuming 'mesh2d_node_x' is longitude

# Met Office grid (structured grid)
mo_lat = mo_ds['nav_lat']
mo_lon = mo_ds['nav_lon']

# Create the regridder using xESMF
regridder = xe.Regridder(delft_ds, mo_ds, method='nearest_s2d', periodic=False)


# Perform the regridding operation
regridded_surf_height = regridder(delft_surf_height)

# Save the regridded result to a new NetCDF file
regridded_surf_height.to_netcdf("regridded_surface_height.nc")

print("Regridding complete. The result is saved to 'regridded_surface_height.nc'.")