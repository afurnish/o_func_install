#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:11:48 2024
@author: af
"""
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from o_func import opsys; start_path = Path(opsys())
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#%% Data sorting
full_path = start_path / Path('Original_Data/UKC3/mersea_testing/mersea_UKC3_NEMO_only')
full_path = start_path / Path('Original_Data/UKC3/mersea_testing/mersea_UKC3_NEMO_only_cropped_depth_to_100m')

est_path = start_path / Path('GitHub/CMCC_EBM/Multi_Estuary/estuary_definition_files')
nemo_path = start_path / 'GitHub/CMCC_EBM/Multi_Estuary/nemo_definition_files'

files_dict = {'T': [], 'U': [], 'V': []}
est_files_dict = {}
nemo_files_dict = {}


for tuv in files_dict.keys():
    for file in full_path.glob(f'*{tuv}*.nc'):
        files_dict[tuv].append(file)

# Read estuary definition files
for est_file in est_path.glob('estuary_in_*'):
    if est_file.is_file():
        # Extract the name from the filename
        est_name = est_file.stem.split('_in_')[-1]
        with est_file.open() as f:
            data = f.read().splitlines()
            # Store the actual data, excluding metadata comments
            est_files_dict[est_name] = {
                "data": list(map(float, data[:6])),  # Assume the first 6 lines are data
                "metadata": "\n".join(data[6:])      # Store the metadata comments
            }

# # Read NEMO definition files
# for nemo_file in nemo_path.glob('ocean_in_*'):
#     if nemo_file.is_file():
#         # Extract the name from the filename
#         nemo_name = nemo_file.stem.split('_in_')[-1]
#         with nemo_file.open() as f:
#             data = f.read().splitlines()
#             # Store the actual data, excluding metadata comments
#             nemo_files_dict[nemo_name] = {
#                 "data": list(map(int, data[:17])),   # Assume the first 17 lines are data
#                 "metadata": "\n".join(data[17:])     # Store the metadata comments
#             }

#%% Data Loading. 

ds = xr.open_mfdataset(files_dict['T'])
variables_with_deptht = {var: ds[var].dims for var in ds.data_vars if 'deptht' in ds[var].dims}
salinity = ds['vosaline']

estuary_coordinates = {}
# Read NEMO files to extract estuary coordinates and data
for nemo_file in nemo_path.glob('*'):
    with open(nemo_file, 'r') as f:
        lines = f.readlines()
        # Extract the estuary name from the filename
        estuary_name = Path(nemo_file).stem.split('_in_')[-1]
        # Extract nx and ny values
        nx = int(lines[0].strip())  # First line
        ny = int(lines[1].strip())  # Second line
        # Store in the dictionary
        nemo_files_dict[estuary_name] = {
            "nx": nx,
            "ny": ny,
            "data": list(map(int, lines[:17])),  # Assume first 17 lines are data
            "metadata": "\n".join(lines[17:])   # Store the metadata comments
        }

# Extract estuary coordinates from nemo_files_dict
# estuary_coordinates = {estuary: (data['ny'], data['nx']) for estuary, data in nemo_files_dict.items()}


#%%
# Extract estuary coordinates from nemo_files_dict
estuary_coordinates = {
    estuary: (data["data"][1], data["data"][0])  # (ny, nx)
    for estuary, data in nemo_files_dict.items()
}
# Plotting salinity profiles and temporal variations
for estuary, (ny, nx) in estuary_coordinates.items():
    # Extract the salinity data for the given coordinates
    salinity_profile = salinity.sel(y=ny, x=nx)

    # Plot vertical salinity profile (averaged over time if needed)
    avg_salinity_profile = salinity_profile.mean(dim='time_counter')
    plt.figure(figsize=(10, 5))
    # plt.plot(avg_salinity_profile, avg_salinity_profile.deptht)
    for i in range(salinity_profile.sizes['time_counter']):
        plt.plot(salinity_profile.isel(time_counter=i), salinity_profile.deptht, label=f'Time {i+1}')

    plt.gca().invert_yaxis()
    plt.title(f'Salinity Profile - {estuary}')
    plt.xlabel('Salinity (psu)')
    plt.ylabel('Depth (m)')
    # plt.ylim([avg_salinity_profile.deptht[10],avg_salinity_profile.deptht[0]])
    plt.grid()
    plt.show()

    # Plot surface and bottom salinity over time
    surface_salinity = salinity_profile.isel(deptht=0)
    bottom_salinity = salinity_profile.isel(deptht=-1)

    plt.figure(figsize=(10, 5))
    plt.plot(surface_salinity.time_counter, surface_salinity, label='Surface')
    plt.plot(bottom_salinity.time_counter, bottom_salinity, label='Bottom')
    plt.title(f'Temporal Salinity Variation - {estuary}')
    plt.xlabel('Time')
    plt.ylabel('Salinity (psu)')
    plt.legend()
    plt.grid()
    plt.show()

#%% Map plot 
y_max = ds.dims['y']
x_max = ds.dims['x']
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add the coastline of the UK and other features
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Plot estuary locations
for estuary, (ny, nx) in estuary_coordinates.items():
    lon = ds['nav_lon'][ny, nx].values
    print(lon)
    print(nx, ny)
    lat = ds['nav_lat'][ny, nx].values
    print(lat)
    print(nx, ny)
    ax.plot(lon, lat, 'ro', markersize=5, label=estuary)
    ax.text(lon + 0.05, lat, estuary, fontsize=9, verticalalignment='bottom', horizontalalignment='left')
    ax.text(lon - 0.05, lat - 0.05, f'({nx}, {ny})', fontsize=7, color='blue', verticalalignment='top', horizontalalignment='left')
# Optionally add a legend or annotations for estuaries
plt.legend(loc='upper right')
# Set the extent to focus on the UK region

ax.set_extent([-11, 3, 49, 61], crs=ccrs.PlateCarree())

# Setting grid lines and ticks for nx, ny
x_ticks = range(0, x_max, 100)
y_ticks = range(0, y_max, 100)

# Calculate corresponding longitude and latitude for these ticks
x_lons = [ds['nav_lon'][0, x] for x in x_ticks]
y_lats = [ds['nav_lat'][y, 0] for y in y_ticks]

# Set the ticks on the axes
ax.set_xticks(x_lons, crs=ccrs.PlateCarree())
ax.set_xticklabels(x_ticks)
ax.set_yticks(y_lats, crs=ccrs.PlateCarree())
ax.set_yticklabels(y_ticks)

# Draw grid lines
ax.gridlines(xlocs=x_lons, ylocs=y_lats, draw_labels=False, linestyle='--', linewidth=0.5)

# Set titles and labels
plt.title('Estuary Locations with UK Coastline and Grid Indices')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show plot
plt.show()
