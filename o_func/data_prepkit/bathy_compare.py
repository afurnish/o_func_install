#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script is set up to compare the bathymetry between the UKC4 model 
and my own domain, ideally I need to use a bathmetry that is identical but with 
additional rivers being run. 

I could also make a grid with my rivers chopped off and add in the river discgarge 
forcing for accurate comparison. 

Created on Tue Mar  5 12:32:42 2024
@author: af
"""
from os.path import join
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from scipy.interpolate import griddata


from o_func import opsys; start_path = opsys()


#%% XYZ data that is the input for the delft model
original_forcing_for_model_xyz = join(start_path, 'modelling_DATA/kent_estuary_project/7.met_office/models/PRIMEA_riv_nawind_oa_1l_original/PRIMEA_riv_nawind_oa_1l.dsproj_data/bed_level_deepened_channel(testing).xyz')
orig_data_load = np.loadtxt(original_forcing_for_model_xyz)
# Separate the columns for clarity
lon = orig_data_load[:, 0];lat = orig_data_load[:, 1];values = orig_data_load[:, 2]

grid_lon = np.linspace(lon.min(), lon.max(), num=400)  # Adjust num for desired resolution
grid_lat = np.linspace(lat.min(), lat.max(), num=400)  # Adjust num for desired resolution
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
# Interpolate unstructured (lon, lat) points to the grid
grid_values = griddata((lon, lat), values, (grid_lon, grid_lat), method='cubic')
orig_xy_dataarray = xr.DataArray(grid_values, coords=[('lat', grid_lat[:,0]), ('lon', grid_lon[0,:])], dims=['lat', 'lon'])


#%% Loading in processed grid in original primea format
prim_orig_bathy = join(start_path, 'modelling_DATA/kent_estuary_project/6.Final2/models/kent_1.3.7_testing_4_days_UM_run/kent_31_merged_map.nc')
load_prim_orig_bathy = xr.open_dataset(prim_orig_bathy)
prim_output_bathy = load_prim_orig_bathy.mesh2d_node_z
prim_output_bathy_lon = load_prim_orig_bathy.mesh2d_node_z.mesh2d_node_x
prim_output_bathy_lat = load_prim_orig_bathy.mesh2d_node_z.mesh2d_node_y
grid_lon = np.linspace(prim_output_bathy_lon.min(), prim_output_bathy_lon.max(), num=1000)  # Adjust num for desired resolution
grid_lat = np.linspace(prim_output_bathy_lat.min(), prim_output_bathy_lat.max(), num=1000)  # Adjust num for desired resolution
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
# Interpolate unstructured (lon, lat) points to the grid
grid_values = griddata((prim_output_bathy_lon, prim_output_bathy_lat), prim_output_bathy, (grid_lon, grid_lat), method='cubic')
prim_output_dataarray = xr.DataArray(grid_values, coords=[('lat', grid_lat[:,0]), ('lon', grid_lon[0,:])], dims=['lat', 'lon'])

## Adding in a waterdepth from middle of array  - sh to get bathy again
wd = load_prim_orig_bathy.mesh2d_waterdepth[200,:] * -1
sh = load_prim_orig_bathy.mesh2d_s1[200,:]

# minimum waterdepth is 0 # takes too long to do it often
# max waterdepth is 63.846397 therefore data must be made negative 

calc_bathy = wd - sh
wdlon = wd.mesh2d_face_x
wdlat = wd.mesh2d_face_y
grid_lon = np.linspace(wdlon.min(), wdlon.max(), num=1000)  # Adjust num for desired resolution
grid_lat = np.linspace(wdlat.min(), wdlat.max(), num=1000)  # Adjust num for desired resolution
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
grid_valuescalc_bathy = griddata((wdlon, wdlat), calc_bathy, (grid_lon, grid_lat), method='cubic')
calc_bathy_output_dataarray = xr.DataArray(grid_valuescalc_bathy, coords=[('lat', grid_lat[:,0]), ('lon', grid_lon[0,:])], dims=['lat', 'lon'])


#%% UKC4 data directkt from monsoon and prim regirdded dataset
ukc4_bathy_path = join(start_path,'Original_Data/UKC3/ukc4_bathy/amm15.bathydepth.hook.nc')
prim_bathy_path = join(start_path,'modelling_DATA/kent_estuary_project/7.met_office/models/PRIMEA_riv_nawind_oa_1l_original/kent_regrid.nc')

ukc4 = xr.open_dataset(ukc4_bathy_path)
ukc4_bathy = ukc4.Bathymetry*-1
prim  = xr.open_dataset(prim_bathy_path)
prim_bathy = prim.prim_bathymetry[0,:,:]

#%% Performing analysis on the two
def create_kdtree(latitudes, longitudes):
    # Combine lat and lon into a single array of points for KDTree
    points = np.vstack([latitudes.ravel(), longitudes.ravel()]).T
    return cKDTree(points)

# Create KDTree for ukc4_bathy
ukc4_tree = create_kdtree(ukc4_bathy.lat.values, ukc4_bathy.lon.values)

# Flatten prim_bathy coordinates and find nearest ukc4_bathy points
prim_points = np.vstack([prim_bathy.nav_lat.values.ravel(), prim_bathy.nav_lon.values.ravel()]).T
distances, indices = ukc4_tree.query(prim_points)

# Assuming a perfect match isn't required and the closest is good enough
# Extract the corresponding ukc4_bathy data
# Note: This involves reshaping indices to match prim_bathy's shape to ensure we can correctly map values
matched_ukc4_indices = np.unravel_index(indices, ukc4_bathy.lat.shape)
# matched_ukc4_bathy = ukc4_bathy.isel(lat=xr.DataArray(matched_ukc4_indices[0], dims="point"), lon=xr.DataArray(matched_ukc4_indices[1], dims="point")).reshape(prim_bathy.shape)

matched_ukc4_values = ukc4_bathy.values.ravel()[indices]
matched_ukc4_bathy = xr.DataArray(matched_ukc4_values.reshape(prim_bathy.nav_lat.shape),
                                  coords={'y': prim_bathy.coords['nav_lat'], 'x': prim_bathy.coords['nav_lon']},
                             dims=['y', 'x'])
difference = matched_ukc4_bathy - prim_bathy

# fig, ax = plt.subplots()
# plt.pcolor(difference)
# plt.colorbar()
#%% Calculating grid cell size 
def calculate_grid_cell_size(lat, lon):
    # Assuming lat and lon are 2D arrays of latitudes and longitudes
    avg_lat = (np.abs(lat[:-1, :-1] - lat[1:, 1:]).mean())
    # print(avg_lat)
    lat_diff =  avg_lat * 111
    lon_diff = (np.abs(lon[:-1, :-1] - lon[1:, 1:]).mean()) * np.cos(np.radians(53)) * 111.32
    return lat_diff, lon_diff

# Example for prim_bathy


lat_diff_prim, lon_diff_prim = calculate_grid_cell_size(prim_bathy.nav_lat.values, prim_bathy.nav_lon.values)
lat_diff_diff, lon_diff_diff = calculate_grid_cell_size(difference.nav_lat.values, difference.nav_lon.values)
lat_diff_ukc4, lon_diff_ukc4 = calculate_grid_cell_size(ukc4_bathy.lat.values, ukc4_bathy.lon.values)
lat_diff_xy, lon_diff_xy = np.diff(orig_xy_dataarray.lat.values).mean()*111, np.diff(orig_xy_dataarray.lon.values).mean() * np.cos(np.radians(53)) * 111.32


#%% Plotting the grids
vmin = min(matched_ukc4_bathy.min(), prim_bathy.min(), difference.min())
vmax = max(matched_ukc4_bathy.max(), prim_bathy.max(), difference.max())

# Adjust the subplot configuration to 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
axs = axs.ravel()  # Flatten the axis array to easily index it

# Assuming vmin, vmax, and cmap are predefined
cmap = cmo.matter

# Plot matched_ukc4_bathy
cax1 = axs[0].pcolormesh(matched_ukc4_bathy.x, matched_ukc4_bathy.y, matched_ukc4_bathy, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[0].set_title('UKC4 Bathy from monsoon')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')

# Plot prim_bathy
cax2 = axs[1].pcolormesh(prim_bathy.nav_lon, prim_bathy.nav_lat, prim_bathy, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[1].set_title('PRIM regrid Bathy')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')

# Plot the difference
cax3 = axs[2].pcolormesh(difference.nav_lon, difference.nav_lat, difference, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[2].set_title('Difference in ukc4/prim')
axs[2].set_xlabel('Longitude')
axs[2].set_ylabel('Latitude')

# Plot the new data
cax4 = axs[3].pcolormesh(orig_xy_dataarray.lon, orig_xy_dataarray.lat, orig_xy_dataarray, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[3].set_title('XYZ Data prim input')
axs[3].set_xlabel('Longitude')
axs[3].set_ylabel('Latitude')

# Plot prim original output bathy
cax5 = axs[4].pcolormesh(prim_output_dataarray.lon, prim_output_dataarray.lat, prim_output_dataarray, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[4].set_title('Prim output bathy')
axs[4].set_xlabel('Longitude')
axs[4].set_ylabel('Latitude')

# Plot calc_bathy_output_dataarray
cax6 = axs[5].pcolormesh(calc_bathy_output_dataarray.lon, calc_bathy_output_dataarray.lat, calc_bathy_output_dataarray, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[5].set_title('Calc Bathy Output')
axs[5].set_xlabel('Longitude')
axs[5].set_ylabel('Latitude')



# For demonstration, let's assume they are stored in variables like this:
cell_size_matched = f"{lat_diff_ukc4:.2f}km by {lon_diff_ukc4:.2f}km"  # Replace with actual calculation
cell_size_prim    = f"{lat_diff_prim:.2f}km by {lon_diff_prim:.2f}km"  # Using calculated value as example
cell_size_diff    = f"{lat_diff_diff:.2f}km by {lon_diff_diff:.2f}km"  # Placeholder, replace as needed
cell_size_xyz     = f"{lat_diff_xy:.2f}km by {lon_diff_xy:.2f}km"  # Replace with actual calculation for your xr_data

# Adding annotations with cell size under each subplot
# axs[0].text(0.5, -0.1, f'Cell Size: {cell_size_matched}', ha='center', va='top', transform=axs[0].transAxes)
# axs[1].text(0.5, -0.1, f'Cell Size: {cell_size_prim}', ha='center', va='top', transform=axs[1].transAxes)
# axs[2].text(0.5, -0.1, f'Cell Size: {cell_size_diff}', ha='center', va='top', transform=axs[2].transAxes)
# axs[3].text(0.5, -0.1, f'Cell Size: {cell_size_xyz}', ha='center', va='top', transform=axs[3].transAxes)


# Add a colorbar for the new plot (or adjust to share one colorbar if preferred)
fig.colorbar(cax6, ax=axs, orientation='vertical', fraction=0.005, pad=0.02, aspect=50, label='Bathymetry')

# fig, ax = plt.subplots(); plt.scatter(wd.mesh2d_face_x, wd.mesh2d_face_y, c = wd-sh, s = 1)
# fig2, axs2 = plt.subplots()
# plt.scatter(orig_data_load[:,0], orig_data_load[:,1], c = orig_data_load[:,2])