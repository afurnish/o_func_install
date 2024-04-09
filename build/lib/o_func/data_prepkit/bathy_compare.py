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
lon_orig = orig_data_load[:, 0];lat_orig = orig_data_load[:, 1];values_orig = orig_data_load[:, 2]

grid_lon = np.linspace(lon_orig.min(), lon_orig.max(), num=400)  # Adjust num for desired resolution
grid_lat = np.linspace(lat_orig.min(), lat_orig.max(), num=400)  # Adjust num for desired resolution
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
# Interpolate unstructured (lon, lat) points to the grid
grid_values = griddata((lon_orig, lat_orig), values_orig, (grid_lon, grid_lat), method='cubic')
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



#%% Need to interpolate what I have done onto a grid. 

# Extract the target grid's X and Y coordinates
target_x = matched_ukc4_bathy.x
target_y = matched_ukc4_bathy.y

# Interpolate your original data onto the target grid
# You might need to flatten the target_x and target_y if they're 2D arrays
interpolated_values = griddata(
    (lon_orig, lat_orig),  # Original points
    values_orig,  # Original values
    (target_x, target_y),  # Target grid points
    method='cubic'  # Interpolation method
)
interpolated_values_array = xr.DataArray(interpolated_values,
                              dims=['y', 'x'],
                              coords={'y': (('y', 'x'), target_y.values),
                                      'x': (('y', 'x'), target_x.values)})

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

# # Plot the new data
# cax4 = axs[3].pcolormesh(orig_xy_dataarray.lon, orig_xy_dataarray.lat, orig_xy_dataarray, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
# axs[3].set_title('XYZ Data prim input')
# axs[3].set_xlabel('Longitude')
# axs[3].set_ylabel('Latitude') 
# Plot the new data THIS has plotted the original bathy I forced primea with onto the same grid as the bathy for the UKC4 map. 
cax4 = axs[3].pcolormesh(matched_ukc4_bathy.x, matched_ukc4_bathy.y, interpolated_values, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
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

#%%
# fig, ax = plt.subplots(); plt.scatter(wd.mesh2d_face_x, wd.mesh2d_face_y, c = wd-sh, s = 1)
# fig2, axs2 = plt.subplots()
# plt.scatter(orig_data_load[:,0], orig_data_load[:,1], c = orig_data_load[:,2])

# Set up the figure and axes for 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Assume vmin, vmax, and cmap are predefined for consistent color scaling across plots
cmap = cmo.delta
vmin = -10  # Example minimum value
vmax = 10   # Example maximum value

# Plot UKC4 Bathy (assuming matched_ukc4_bathy is an xarray DataArray)
cax1 = axs[0].pcolormesh(matched_ukc4_bathy.x, matched_ukc4_bathy.y, matched_ukc4_bathy, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[0].set_title('UKC4 Bathy from Monsoon')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')

# Plot original PRIMEA (assuming prim_bathy is an xarray DataArray)
cax2 = axs[1].pcolormesh(matched_ukc4_bathy.x, matched_ukc4_bathy.y, interpolated_values, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[1].set_title('XYZ Data prim input onto UKC4 grid')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')

# Calculate and plot the difference (assuming you have two compatible datasets for subtraction)
difference = matched_ukc4_bathy - interpolated_values  # Adjust according to actual data structures
cax3 = axs[2].pcolormesh(difference.x, difference.y, difference, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[2].set_title('Difference (UKC4 - PRIMEA)')
axs[2].set_xlabel('Longitude')
axs[2].set_ylabel('Latitude')

# Adding colorbars to each plot for clarity
fig.colorbar(cax1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04, label='Depth (m)')
fig.colorbar(cax2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04, label='Depth (m)')
fig.colorbar(cax3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04, label='Depth Difference (m)')

# Explanation regarding negative numbers
# Assuming that in your dataset, more negative numbers represent greater depths
plt.figtext(0.5, 0, 'In the difference plot, more positive values indicate areas where PRIMEA is deeper than UKC4.', ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

#%%
import geopandas as gpd

# Adjust the path as necessary
# shapefile_path = join(start_path,'modelling_DATA','kent_estuary_project','land_boundary','QGIS_Shapefiles','UK_WEST_POLYGON_NEGATIVE.shp')

shapefile_path = join(start_path,'modelling_DATA','kent_estuary_project','bathymetry','new_bathymetry_ukc4','NEGATIVE_POLYGON_WIDENED_RIVERS.shp')
land_boundary = gpd.read_file(shapefile_path)

from shapely.geometry import Point

# Assuming 'difference' is your DataArray with 'x' and 'y' coordinates

# Convert grid points to shapely Points
points = [Point(x, y) for x, y in zip(np.ravel(difference.x), np.ravel(difference.y))]

# Check if each point is within any polygon
mask = np.array([land_boundary.contains(point).any() for point in points])

# Reshape the mask to match the DataArray's shape
mask_reshaped = mask.reshape(difference.shape)

difference_masked = difference.where(~mask_reshaped, other=np.nan)
primea_masked = interpolated_values_array.where(~mask_reshaped, other=np.nan)

#%%
# fig, ax = plt.subplots(); plt.scatter(wd.mesh2d_face_x, wd.mesh2d_face_y, c = wd-sh, s = 1)
# fig2, axs2 = plt.subplots()
# plt.scatter(orig_data_load[:,0], orig_data_load[:,1], c = orig_data_load[:,2])

# Set up the figure and axes for 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(18, 8))

# Assume vmin, vmax, and cmap are predefined for consistent color scaling across plots
cmap = cmo.delta
vmin = -10  # Example minimum value
vmax = 10   # Example maximum value

matched_ukc4_bathy_nan = matched_ukc4_bathy
matched_ukc4_bathy_nan.values[matched_ukc4_bathy_nan.values == 0] = np.nan

# Plot UKC4 Bathy (assuming matched_ukc4_bathy is an xarray DataArray)
cax1 = axs[0].pcolormesh(matched_ukc4_bathy_nan.x, matched_ukc4_bathy_nan.y, matched_ukc4_bathy_nan, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[0].set_title('UKC4 Bathymetry')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')

# Plot original PRIMEA (assuming prim_bathy is an xarray DataArray)
cax2 = axs[1].pcolormesh(primea_masked.x, primea_masked.y, primea_masked, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[1].set_title('PRIMEA forced Bathymetry')
axs[1].set_xlabel('Longitude')
# axs[1].set_ylabel('Latitude')

# Calculate and plot the difference (assuming you have two compatible datasets for subtraction)
# difference = matched_ukc4_bathy - interpolated_values  # Adjust according to actual data structures

cax3 = axs[2].pcolormesh(difference_masked.x, difference_masked.y, difference_masked, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
axs[2].set_title('Difference (UKC4 - PRIMEA)')
axs[2].set_xlabel('Longitude')
# axs[2].set_ylabel('Latitude')

# Adding colorbars to each plot for clarity
# fig.colorbar(cax1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04, label='Depth (m)')
# fig.colorbar(cax2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04, label='Depth (m)')
fig.colorbar(cax3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04, label='Depth Difference (m)')

# Explanation regarding negative numbers
# Assuming that in your dataset, more negative numbers represent greater depths
plt.figtext(0.5, 0.02, 'In the difference plot, more positive values indicate areas where PRIMEA is deeper than UKC4.', ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
# plt.subplots_adjust(bottom=2)  # Adjust bottom margin


# If we cut out the data from the UKC4 bathymetry and place it over the existing bathymetry collected we end up with this map here. 
primea_updated = primea_masked.where(np.isnan(matched_ukc4_bathy_nan), matched_ukc4_bathy_nan)

# to the front of primea updated we need to add 5 columns of junk. 
# were gonna use a bigger dataset, and expand the other dataset out by some. 
# This is what you need if you were to slice manually
# sliced_ukc4_bathy = ukc4_bathy[597:687+1,758:795+1]
# its now (91, 43)

place_to_store_new_bathy = join(start_path,'Original_Data','UKC3','ukc4_bathy','UKC4_bathy_with_extra_rivers.nc')
xyz_file_path = join(start_path,'Original_Data','UKC3','ukc4_bathy','UKC4_bathy_with_extra_rivers.xyz')

primea_updated.to_netcdf(place_to_store_new_bathy)

#%% This is part of a plan to add extra data to the area. 
primea_updated_expanded_front = primea_masked.pad({'x': (5, 0)}, constant_values=np.nan)
primea_updated_expanded_front = primea_updated_expanded_front.rename({'x': 'lon', 'y': 'lat'})

sliced_ukc4_bathy = ukc4_bathy[597:687+1,758-5:795+1] # pulls in extra data from the left. 
sliced_ukc4_bathy_nan = sliced_ukc4_bathy
sliced_ukc4_bathy_nan.values[sliced_ukc4_bathy_nan.values == 0] = np.nan

primea_updated_expanded_front = primea_updated_expanded_front.assign_coords(lat=sliced_ukc4_bathy.coords['lat'], lon=sliced_ukc4_bathy.coords['lon'])
primea_updated_front_add_on = primea_updated_expanded_front.where(np.isnan(sliced_ukc4_bathy), sliced_ukc4_bathy)

# Flatten the arrays to create a 2D table
lat_flat = primea_updated_front_add_on.lat.values.flatten()
lon_flat = primea_updated_front_add_on.lon.values.flatten()
z_flat = primea_updated_front_add_on.values.flatten()
import pandas as pd
# Create a DataFrame
df = pd.DataFrame({
    'Longitude': lon_flat,
    'Latitude': lat_flat,
    'Height': z_flat
})
df = df.dropna(subset=['Height'])

df['Longitude'] = df['Longitude'].map(lambda x: f'{x:.7f}')
df['Latitude'] = df['Latitude'].map(lambda x: f'{x:.7f}')
df.to_csv(xyz_file_path, sep='\t', index=False, header=False)
# fig, ax = plt.subplots()
# plt.pcolormesh(sliced_ukc4_bathy)
# fig2, ax2 = plt.subplots()
# plt.pcolormesh(matched_ukc4_bathy)