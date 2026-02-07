#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file should find the nearest rivers in the river climatology of UKC4 datasets
and extract them into suitable variables so that they may be used with running 
simulations that are proportional to the met office. 

Created on Thu Mar  7 12:15:29 2024

@author: af
"""
extra_detail_plots = 'y'
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from  os.path import join 
from o_func import opsys, DirGen; start_path = opsys('PNC')
import pandas as pd
import geopandas as gpd
import o_func.utilities as util
import subprocess
import pkg_resources
import platform
import glob
from pathlib import Path

fig_path = Path(start_path) / 'modelling_DATA/kent_estuary_project/river_boundary_conditions/figures'

#b20_mean_discharge = means[:51]

data = xr.open_dataset(join(start_path, 'Original_Data','UKC3','river_climatology','rivers','AMM15_River_Climatology.nc'))
# data = xr.open_dataset(join(start_path, 'Original_Data','UKC3','river_climatology','rivers','AMM15_River_Climatology_v2.nc'))

runoff = np.array(data.rorunoff[0,:,:])
non_zero_mask = np.where(runoff != 0.0)
row_indices, col_indices = non_zero_mask

# lons = []
# lats = []

# for i in range(len(row_indices)):
#     lons.append(data.lon[row_indices[i], col_indices[i]].item())
#     lats.append(data.lat[row_indices[i], col_indices[i]].item())
# Extract the corresponding latitude and longitude for the non-zero points
lons = [data.lon[row, col].item() for row, col in zip(row_indices, col_indices)]
lats = [data.lat[row, col].item() for row, col in zip(row_indices, col_indices)]

# Define the UK's latitude and longitude bounds for the first plot

#%% Self Contained cell for plotting the UK Map
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
uk_lon_min, uk_lon_max = -11, 2.5
uk_lat_min, uk_lat_max = 49, 61
fig = plt.figure(figsize=(7, 10), dpi = 150)
proj = ccrs.Mercator(central_longitude=-4)
ax = plt.axes(projection=proj)
point_indices = np.arange(0, len(lons))
ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='dotted', edgecolor='black')
ax.scatter(lons, lats, transform=ccrs.PlateCarree(), marker='o', color='blue', label='AMM15 River Climatology\nDischarge Locations')
# for lontest, lattest, indtext in zip(lons, lats, point_indices):
#     ax.text(lontest - 0.02, lattest + 0.01, f'{indtext}', ha='center', va='bottom', fontsize=10, color='blue')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# uk_extent_lon = np.linspace(-10, 2, 13)
# uk_extent_lat = np.linspace(48, 62, 15)
uk_extent_lon = np.linspace(uk_lon_min, uk_lon_max, 16)
uk_extent_lat = np.linspace(uk_lat_min, uk_lat_max, 20)
ax.set_extent([uk_lon_min, uk_lon_max, uk_lat_min, uk_lat_max])

# SET TO WIDE UK EXTENT
# Define whole-number tick positions
xticks = np.arange(np.ceil(uk_lon_min), np.floor(uk_lon_max) + 1, 1)
yticks = np.arange(np.ceil(uk_lat_min), np.floor(uk_lat_max) + 1, 1)

# Apply ticks in geographic coords
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

# Format tick labels as plain numbers (no degree/E/W/N/S)
ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f',
                                                degree_symbol='',
                                                direction_label=False))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f',
                                               degree_symbol='',
                                               direction_label=False))

# Make tick marks small and neat
ax.tick_params(axis='both', which='major', length=4, width=0.8, direction='out')

# Axis labels
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig(fig_path / 'river_discharge_map_uk.png', dpi = 300)

#%%
# Define the UK's latitude and longitude bounds for the second plot
uk_lon_min, uk_lon_max = -3.65, -2.75
uk_lat_min, uk_lat_max = 53.20, 54.52

# how did I get this information? Good question Aaron, I still don't know. 
# This was probably plotted each one sequentically and you read the numvers, 
latitudes = data['lat'].values
longitudes = data['lon'].values

riv_dict = {129: 'Esk',     # These are all indexed from 1, not 0. 1 less from each. 
            124: 'Leven',
            125: 'Kent',
            120: 'Lune',
            118: 'Wyre',
            116: 'Ribble',
            112: 'Alt',
            111: 'Mersey',
            101: 'Dee',
            103: 'Clywd'
            }
riv_dict = {i - 1: name for i, name in riv_dict.items()}
additional_coords = {
    'Dee': (-3.118638742308569, 53.24982016910892),
    'Duddon': (-3.230547161208941, 54.25887801158542),
    'Kent': (-2.811861321397053, 54.25064484652686),
    'Leven': (-3.052120073154467, 54.23186185336646),
    'Lune': (-2.840884669179119, 54.03655050082423),
    'Mersey': (-2.768434835109615, 53.34491510325321),
    'Ribble': (-2.811633371553361, 53.74817881546817),
    'Wyre': (-2.955520867395822, 53.85663354235163)
}
# Initialize a dictionary to store the (nx, ny) indices for each river
river_grid_indices = {}

# Function to find the nearest grid point index
def find_matching_index(latitudes, longitudes, target_lat, target_lon):
    # Calculate the absolute differences
    abs_diff_lat = np.abs(latitudes - target_lat)
    abs_diff_lon = np.abs(longitudes - target_lon)
    # Sum of absolute differences across lat and lon dimensions
    total_diff = abs_diff_lat + abs_diff_lon
    # Find the index of the minimum difference
    min_diff_idx = np.unravel_index(np.argmin(total_diff), total_diff.shape)
    return min_diff_idx

# Loop over each river to find the corresponding grid indices
for idx, (river_name) in riv_dict.items():
    print(idx)
    # Retrieve the corresponding latitude and longitude for the river
    river_lat = lats[idx]
    print(river_lat)
    river_lon = lons[idx]
    print(river_lon)
    
    # Find the matching grid index for the given coordinates
    ny, nx = find_matching_index(latitudes, longitudes, river_lat, river_lon)
    
    # Store the indices in the dictionary
    river_grid_indices[river_name] = (ny, nx)

# Display the grid indices for each river
for river_name, (ny, nx) in river_grid_indices.items():
    print(f'River: {river_name}, Grid Indices: (ny: {ny}, nx: {nx})')
#%% Production of index 
# Function to find the closest grid point
river_names = []
rows = []
cols = []

# Iterate through each river in the dictionary
for row_idx, river_name in riv_dict.items():
    river_names.append(river_name)
    rows.append(row_indices[row_idx])
    cols.append(col_indices[row_idx])
    
    

# Create the DataFrame
river_index_df = pd.DataFrame({
    'River': river_names,
    'Row Index': rows,
    'Col Index': cols
})
#%% 
save_dict = {}
shapefile_path = start_path + "modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_KENT_EPSG_4326_clipped_med_domain.shp"
gdf = gpd.read_file(shapefile_path)

fig = plt.figure(figsize=(10, 12), dpi = 150)
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='red')
# ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='dotted', edgecolor='black')
gdf.plot(ax = ax, color = 'black', linewidth=0.5)
ax.scatter(lons, lats, marker='o', color='blue', label='AMM15 river climatology discharge')

save_dict['lons'] = lons
save_dict['lats'] = lats
save_dict['row_indicies'] = row_indices
save_dict['col_indices'] = col_indices


ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
uk_extent_lon = np.linspace(-3.65, -2.70, 6)
uk_extent_lat = np.linspace(53.20, 54.52, 7)
ax.set_xticks(uk_extent_lon, crs=ccrs.PlateCarree())
ax.set_yticks(uk_extent_lat, crs=ccrs.PlateCarree())
# Label each point with a number from 1 to n
# Label each point with a number from 1 to n and the corresponding river name
# Label each point with a number from 1 to n and the corresponding river name

save_dict['Rivers']  = {}
# for i, (lon, lat) in enumerate(zip(lons, lats), start=1):
#     if i in riv_dict:
#         ax.text(lon - 0.02, lat + 0.01, f'{riv_dict[i]}', ha='center', va='bottom', fontsize=10, color='blue')
# Plotting the river discharge points and their respective indices and names
for i, (lon, lat) in enumerate(zip(lons, lats)):
    # Check if the index corresponds to a river in the dictionary
    if i in riv_dict:
        # Get the river name from the dictionary
        river_name = riv_dict[i]
        # Get the corresponding row and column indices
        row_idx = row_indices[i]
        col_idx = col_indices[i]
        # Plot the river name and optionally the row and column indices
        ax.text(lon - 0.02, lat + 0.01, f'{river_name}', ha='center', va='bottom', fontsize=10, color='blue')
        
        save_dict['Rivers'][river_name] = {}
        save_dict['Rivers'][river_name]['lon'] = lon - 0.02
        save_dict['Rivers'][river_name]['lat'] = lat + 0.01
    
        
        if extra_detail_plots == 'y':
            ax.text(lon - 0.02, lat - 0.02, f'({row_idx}, {col_idx})', ha='center', va='top', fontsize=8, color='red')

import pickle
with open('river_gauges_save_data2.pkl', 'wb') as f:
    pickle.dump(save_dict, f)
first_river_plotted = False
for river, (lon, lat) in additional_coords.items():
    if not first_river_plotted:
        ax.scatter(lon, lat, marker='^', color='red', label='15-min measured river gauge data')
        first_river_plotted = True
    else:
        ax.scatter(lon, lat, marker='^', color='red')
    ax.text(lon + 0.02, lat + 0.01, river, ha='center', va='bottom', fontsize=10, color='red')

#% DO you want to add transects onto this figure ?
transect_paths = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/analysis/QGIS_shapefiles/points_along_estuary_1km_spacing.csv'
transect_data = pd.read_csv(transect_paths)

ax.scatter(transect_data.X, transect_data.Y, c = 'green', marker = '+', s = 1, label = 'Estuarine Transect')


ax.set_extent([uk_lon_min, uk_lon_max + 0.25, uk_lat_min, uk_lat_max])
ax.set_aspect(aspect=0.75) 

plt.tight_layout()
plt.legend()
plt.show()

#%% 
from bng_latlon import OSGB36toWGS84

# Example easting and northing for BNG reference "SJ391983"
easting = 391000  # SJ391983 -> easting
northing = 983000  # SJ391983 -> northing

# Convert to latitude and longitude
latitude, longitude = OSGB36toWGS84(easting, northing)
print(f"{longitude},{latitude}")

import math

def gridDistance(ref1, ref2):
    # Convert to fully numeric references
    p1 = gridrefNumeric(ref1)
    p2 = gridrefNumeric(ref2)

    # Get E/N distances between ref1 & ref2
    deltaE = p2[0] - p1[0]
    deltaN = p2[1] - p1[1]

    # Use Pythagoras' theorem to calculate the distance between the points
    dist = math.sqrt(deltaE ** 2 + deltaN ** 2)

    return round(dist / 1000, 2)  # Return result in km, 2 decimals

def gridBearing(ref1, ref2):
    # Convert to fully numeric references
    p1 = gridrefNumeric(ref1)
    p2 = gridrefNumeric(ref2)

    # Get E/N distances between ref1 & ref2
    deltaE = p2[0] - p1[0]
    deltaN = p2[1] - p1[1]

    # Calculate bearing using arctan and convert from radians to degrees
    deg = (90 - (math.atan2(deltaN, deltaE) * 180 / math.pi) + 360) % 360

    return round(deg)  # Return result in degrees, no decimals

def gridrefNumeric(gridref):
    # Convert letter references to numeric values
    gridref = gridref.upper()
    letE = ord(gridref[0]) - ord('A')
    letN = ord(gridref[1]) - ord('A')

    # Adjust letters after 'I' since 'I' is not used
    if letE > 7: letE -= 1
    if letN > 7: letN -= 1

    # Convert grid letters into 100km-square indexes from false origin (grid square SV)
    e = ((letE + 3) % 5) * 5 + (letN % 5)
    n = 19 - (letE // 5) * 5 - (letN // 5)

    # Remove grid letters and get the numeric part of the reference
    gridref = gridref[2:].replace(" ", "")
    e = int(str(e) + gridref[:len(gridref)//2])
    n = int(str(n) + gridref[len(gridref)//2:])

    # Normalize to a 1m grid
    if len(gridref) == 6:
        e *= 100
        n *= 100
    elif len(gridref) == 8:
        e *= 10
        n *= 10
    # 10-digit references are already in 1m resolution

    return [e, n]

# Example usage
print(gridDistance("SU387148", "SU38714856"))  # Distance in km
print(gridBearing("SU387148", "SU38714856"))  # Bearing in degrees

numeric_coords = gridrefNumeric("SU387148")
#%%

riv_gauge = {
    'Alt': {'grid': 'SJ391983', 'name': 'Kirkby', 'gauge_number': 69032},
    'Esk': {'grid': 'SD131977', 'name': 'Cropple How', 'gauge_number': 74007},
    'Clywd': {'grid': 'SJ069709', 'name': 'Pont-y-Cambwll', 'gauge_number': 66001},
}
# Function to convert BNG grid references to latitude and longitude

def update_riv_gauge_with_latlon(riv_gauge):
    for river, details in riv_gauge.items():
        # Get the OS grid reference
        grid_ref = details['grid']

        # Convert the OS grid reference to easting and northing
        numeric_coords = gridrefNumeric(grid_ref)
        easting, northing = numeric_coords

        # Convert easting and northing to latitude and longitude
        latitude, longitude = OSGB36toWGS84(easting, northing)

        # Add latitude and longitude to the dictionary
        details['lat'] = latitude
        details['lon'] = longitude
        print(f"{river}: Longitude = {latitude}, {longitude}")

# Update the river gauge dictionary
update_riv_gauge_with_latlon(riv_gauge)

# Collect the new lon, lat, and river names for plotting
river_lons = [details['lon'] for details in riv_gauge.values()]
river_lats = [details['lat'] for details in riv_gauge.values()]
river_names = [key for key  in riv_gauge]

#%%
extra_detail_plots = 'n'
ffig = plt.figure(figsize=(10, 12), dpi = 150)
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='red')
# ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='dotted', edgecolor='black')
gdf.plot(ax = ax, color = 'black', linewidth=0.5)
ax.scatter(lons, lats, marker='o', color='blue', label='AMM15 river climatology discharge')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
uk_extent_lon = np.linspace(-3.65, -2.70, 6)
uk_extent_lat = np.linspace(53.20, 54.52, 7)
ax.set_xticks(uk_extent_lon, crs=ccrs.PlateCarree())
ax.set_yticks(uk_extent_lat, crs=ccrs.PlateCarree())
# Label each point with a number from 1 to n
# Label each point with a number from 1 to n and the corresponding river name
# Label each point with a number from 1 to n and the corresponding river name


# for i, (lon, lat) in enumerate(zip(lons, lats), start=1):
#     if i in riv_dict:
#         ax.text(lon - 0.02, lat + 0.01, f'{riv_dict[i]}', ha='center', va='bottom', fontsize=10, color='blue')
# Plotting the river discharge points and their respective indices and names
for i, (lon, lat) in enumerate(zip(lons, lats)):
    # Check if the index corresponds to a river in the dictionary
    if i in riv_dict:
        # Get the river name from the dictionary
        river_name = riv_dict[i]
        # Get the corresponding row and column indices
        row_idx = row_indices[i]
        col_idx = col_indices[i]
        # Plot the river name and optionally the row and column indices
        ax.text(lon - 0.02, lat + 0.01, f'{river_name}', ha='center', va='bottom', fontsize=10, color='blue')
        if extra_detail_plots == 'y':
            ax.text(lon - 0.02, lat - 0.02, f'({row_idx}, {col_idx})', ha='center', va='top', fontsize=8, color='red')

    
first_river_plotted = False
for river, (lon, lat) in additional_coords.items():
    if not first_river_plotted:
        ax.scatter(lon, lat, marker='^', color='red', label='15-min measured river gauge data')
        first_river_plotted = True
    else:
        ax.scatter(lon, lat, marker='^', color='red')
    ax.text(lon + 0.02, lat + 0.01, river, ha='center', va='bottom', fontsize=10, color='red')

#Plot the esk alt and clywd
ax.scatter(river_lons, river_lats, marker='s', color='purple', label='Extra River Gauge Locations')
# Label each river gauge point with its name
for lon, lat, name in zip(river_lons, river_lats, river_names):
    ax.text(lon + 0.02, lat + 0.01, name, ha='center', va='bottom', fontsize=10, color='purple')
#% DO you want to add transects onto this figure ?
transect_paths = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/analysis/QGIS_shapefiles/points_along_estuary_1km_spacing.csv'
transect_data = pd.read_csv(transect_paths)

ax.scatter(transect_data.X, transect_data.Y, c = 'green', marker = '+', s = 1, label = 'Estuarine 1km\nspaced Transects')


ax.set_extent([uk_lon_min, uk_lon_max + 0.25, uk_lat_min, uk_lat_max])
ax.set_aspect(aspect=0.75) 

plt.tight_layout()
plt.legend()

plt.savefig('estuaries_map.png', dpi = 300)


#%% Set up a nice mercator version 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

# Inset bounds (lon/lat)
lon_min, lon_max = -3.61, -2.57
lat_min, lat_max = 53.17, 54.52

# --- Make the panel less skinny: pad the lon range and use a wider figure ---
pad_lon = 0   # add ~0.2° on each side
pad_lat = 0
extent = [lon_min, lon_max, lat_min, lat_max]

proj_map  = ccrs.Mercator(central_longitude=-3)
proj_data = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(5, 7), dpi=300,
                       subplot_kw={'projection': proj_map},
                       constrained_layout=True)

# --- Ensure coastline GeoDataFrame is in lon/lat or supply transform ---
# If your gdf is NOT EPSG:4326, reproject it first:
try:
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf_ll = gdf.to_crs(4326)
    else:
        gdf_ll = gdf
except Exception:
    gdf_ll = gdf  # fallback if CRS unknown; assume lon/lat

# Plot coastline (edge only), **with transform** so Cartopy knows coordinates are lon/lat
gdf_ll.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.6,
            transform=proj_data, zorder=2)

# Plot points (also declare data CRS)
ax.scatter(lons, lats, transform=proj_data, s=22, color='blue',
           label='AMM15 river climatology\nforcing locations', zorder=3)

for i, (lon, lat) in enumerate(zip(lons, lats)):
    # Check if the index corresponds to a river in the dictionary
    if i in riv_dict:
        # Get the river name from the dictionary
        river_name = riv_dict[i]
        # Get the corresponding row and column indices
        row_idx = row_indices[i]
        col_idx = col_indices[i]
        # Plot the river name and optionally the row and column indices
        ax.text(lon - 0.02, lat + 0.01, f'{river_name}', ha='center', va='bottom', fontsize=8, color='blue', transform=proj_data)


# 15 min  Measured Guage Data
first_river_plotted = False
for river, (lon, lat) in additional_coords.items():
    if not first_river_plotted:
        ax.scatter(lon, lat, transform=proj_data,  marker='^', color='red', label='NRFA 15-min river gauge\nforcing locations')
        first_river_plotted = True
    else:
        ax.scatter(lon, lat, transform=proj_data, marker='^', color='red')
    ax.text(lon + 0.02, lat + 0.01, river, ha='center', va='bottom', fontsize=8, color='red', transform=proj_data)


# Handle the extra river gauges
ax.scatter(river_lons, river_lats, transform=proj_data, marker='s', color='purple', label='Unused River Gauges')
# Label each river gauge point with its name
for lon, lat, name in zip(river_lons, river_lats, river_names):
    ax.text(lon + 0.02, lat + 0.01, name, ha='center', va='bottom', fontsize=8, color='purple', transform=proj_data)

# Handle the transect data
ax.scatter(transect_data.X, transect_data.Y, transform=proj_data, c = 'green', marker = '+', s = 1, label = 'Estuarine Transects\n(1 km spaced)')

# Geographic extent (tell Cartopy this is lon/lat)

# Quarter-degree ticks
xticks = np.arange(np.floor(extent[0]*4)/4, np.ceil(extent[1]*4)/4 + 0.25, 0.25)
yticks = np.arange(np.floor(extent[2]*4)/4, np.ceil(extent[3]*4)/4 + 0.25, 0.25)

ax.set_xticks(xticks, crs=proj_data)
ax.set_yticks(yticks, crs=proj_data)

# Plain numeric labels (no °, no E/W/N/S)
ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f',
                                                degree_symbol='', direction_label=False))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f',
                                               degree_symbol='', direction_label=False))
ax.tick_params(axis='both', which='major', length=4, width=0.9, direction='out')

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_aspect('auto')
ax.legend(loc='center left', fontsize = 8)
ax.set_extent(extent, crs=proj_data)

plt.savefig(fig_path / 'estuaries_map.png', dpi=300, bbox_inches='tight')

#%% Make a dataframe of the new river climatology data 
# Initialize a DataFrame to store river names and their time series data
# You may want to adjust the structure based on your exact requirements
# Initialize a list to store data dictionaries
data_list = []

# Iterate through the river dictionary
for key, river_name in riv_dict.items():
    # Adjust the logic here to correctly map the keys to your data points
    # This is a placeholder logic and needs to be adjusted
    if key in range(len(row_indices)):
        y_index, x_index = row_indices[key-101], col_indices[key-101]  # Adjusted indexing logic
        time_series_data = data.rorunoff[:, y_index, x_index].values
        data_list.append({'River Name': river_name, 'Time Series Data': time_series_data})

# Convert the list of dictionaries to DataFrame
rivers_df = pd.DataFrame(data_list)

# Saving the DataFrame to CSV
output_river_path = join(start_path, 'modelling_DATA/kent_estuary_project/river_boundary_conditions')
# rivers_df.to_csv(join(output_river_path, 'River_Climatology_Time_Series_Updated.csv'), index=False)
num_days = 366
# Using a placeholder year, let's use 2020 for simplicity since it's a leap year, ensuring 366 days
date_range = pd.date_range(start='2020-01-01', end='2020-12-31')

# Format the dates to only show month and day as strings like "MM-DD"
formatted_dates = date_range.strftime('%m-%d')

# Reindex the DataFrame with these formatted dates
time_series_df = pd.DataFrame(index=np.arange(num_days))

# Iterate through each river in the original DataFrame
for index, row in rivers_df.iterrows():
    # Ensure the time series data length matches your index length; trim or pad if necessary
    data_length = len(row['Time Series Data'])
    if data_length > num_days:
        # If the data length exceeds the number of days, trim it
        trimmed_data = row['Time Series Data'][:num_days]
    else:
        # If the data length is shorter, pad it with NaNs or another placeholder
        trimmed_data = np.pad(row['Time Series Data'], (0, num_days - data_length), 'constant', constant_values=np.nan)

    # Create a Series from the river's time series data with the new numerical index
    series = pd.Series(trimmed_data, index=np.arange(num_days), name=row['River Name'])
    # Join this series as a new column in the time_series_df DataFrame
    time_series_df = time_series_df.join(series)
time_series_df.index = formatted_dates
time_series_df.index.name = 'month_day'

time_series_df.to_csv(join(output_river_path, 'River_Climatology_Time_Series.csv'), index=True)
# time_series_df.to_csv(join(output_river_path, 'River_Climatology_Time_Series_v2.csv'), index=True)

# Now 'time_series_df' is structured with rivers as columns, and the index is numerical from 0 to 365

path_to_bc_file = join(start_path, 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/amm15_river_climatology')
# Iterate over the columns in the DataFrame
from datetime import datetime
start_date = datetime(2013, 10, 30)
# Define a function to calculate seconds since the start date
def seconds_since_start(date, start_date):
    delta = date - start_date
    return int(delta.total_seconds())



for river_name in time_series_df.columns:
    # Extract the data for the current river
    river_data = time_series_df[river_name]
    
    # Define the filename using the river name and the .bc extension
    filename = f"{river_name}.bc"
    # filename = f"{river_name}_v2.bc"
    # Save the river data to a file
    # Assuming you want to save it as a CSV for example. Adjust the path as needed.
    river_data.to_csv(join(path_to_bc_file, filename), header=False)
    
    # If you prefer to save it in a different format or with specific formatting,
    # you may need to adjust the saving method accordingly.
    
    
def convert_clim_to_discharge_units(array):
    length = 1500
    width = 1500
    density = 1000
    area = length * width    
    flow_rate = (array * area) / density
    return flow_rate


def adjust_dates_around_cutoff(dataframe, cutoff_day, target_year):
    """
    Adjusts the dates in the DataFrame by reordering dates around a cutoff day and assigning
    years such that all dates from cutoff_day onwards are in target_year and before are in the next year.
    Handles removal of February 29 appropriately.

    Parameters:
        dataframe (pd.DataFrame): DataFrame with a 'month_day' index as strings 'MM-DD'.
        cutoff_day (str): 'MM-DD' format, pivot for splitting the year.
        target_year (int): Year to assign to dates on or after cutoff_day.

    Returns:
        pd.DataFrame: DataFrame with adjusted datetime index.
    """
    # Check for leap year based on the presence of '02-29'
    is_current_leap = '02-29' in dataframe.index

    # Determine next year based on target year leap status
    target_year_is_leap = (target_year % 4 == 0 and (target_year % 100 != 0 or target_year % 400 == 0))
    next_year = target_year + 1 if target_year_is_leap else target_year

    # Remove '02-29' if present and target year is not a leap year
    if is_current_leap and not target_year_is_leap:
        dataframe = dataframe.drop('02-29')
    
    # Reindex to ensure a sequential day numbering from 1 to 365 (or 366 in a leap year)
    reindexed_df = dataframe.reset_index(drop=True)
    if '02-29' in dataframe.index:  # Adjust the index day number accordingly
        reindexed_df.index = reindexed_df.index.where(reindexed_df.index < dataframe.index.get_loc('02-29'), reindexed_df.index - 1)

    # Split the data around the cutoff day
    cutoff_day_index = pd.to_datetime(cutoff_day, format='%m-%d').dayofyear
    after_cutoff = reindexed_df.loc[cutoff_day_index:]
    before_cutoff = reindexed_df.loc[:cutoff_day_index - 1]

    # Concatenate and sort by index
    new_dataframe = pd.concat([after_cutoff, before_cutoff])
    new_dataframe.index = pd.date_range(f'{cutoff_day}-{target_year}', periods=len(new_dataframe), freq='D')
    
    return new_dataframe

def generate_bc_files(dataframe, start_date, path):
    """
    Generate .bc files for discharge and salinity based on the DataFrame.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the data.
        start_date (str): Start date in 'YYYY-MM-DD' format.

    Returns:
        None
    """
    # Iterate over each column (estuary) in the DataFrame
    for column in dataframe.columns:
        # Create file names with estuary name
        for each_side in ['0001', '0002']:
            
            discharge_file = os.path.join(path , f"{column}_Discharge.bc")
            # discharge_file = os.path.join(path , f"{column}_Discharge_v2.bc")
            salinity_file = os.path.join(path ,f"{column}_{each_side}_Salinity.bc")
    
            
            # Open files for writing
            with open(discharge_file, 'w') as discharge, open(salinity_file, 'w') as salinity:
                # Write headers for discharge file
                discharge.write("[forcing]\n")
                discharge.write(f"Name                            = {column}_0001\n")
                discharge.write("Function                        = timeseries\n")
                discharge.write("Time-interpolation              = linear\n")
                discharge.write("Quantity                        = time\n")
                discharge.write(f"Unit                            = seconds since {start_date} 00:00:00\n")
                discharge.write("Quantity                        = dischargebnd\n")
                discharge.write("Unit                            = m³/s\n")
                
                # Write headers for salinity file
                salinity.write("[forcing]\n")
                salinity.write(f"Name                            = {column}_{each_side}\n")
                salinity.write("Function                        = timeseries\n")
                salinity.write("Time-interpolation              = linear\n")
                salinity.write("Vertical position type          = single\n")
                salinity.write("Vertical interpolation          = linear\n")
                salinity.write("Quantity                        = time\n")
                salinity.write(f"Unit                            = seconds since {start_date} 00:00:00\n")
                salinity.write("Quantity                        = salinitybnd\n")
                salinity.write("Unit                            = ppt\n")
                salinity.write("Vertical position               = 1\n")
    
                # Iterate over data for the current estuary
                for idx, value in dataframe[column].items():
                    # Convert date to seconds since start_date
                    date_seconds = (pd.to_datetime(idx) - pd.to_datetime(start_date)).total_seconds()
                    # Write discharge data
                    discharge.write(f"{date_seconds}   {value}\n")
                    # Write salinity data (all zeros)
                    salinity.write(f"{date_seconds}   0\n")

def file_stitcher(input_file_path, output_file_path):
    for names in ['Discharge.bc', 'Salinity.bc']:
        

        data_paths = sorted(glob.glob(os.path.join(input_file_path, f"*{names}" )))
        # print(os.path.join(user_dict['csv_path'],f'*_{comps[:-3]}_*.csv'))
        # print('dp',data_paths)
        
        bash_script_path = pkg_resources.resource_filename('o_func', 'data/bash/merge_csv.sh')
        output_filedir = os.path.join(output_file_path, names)
        with open( output_filedir , 'w') as f:
            f.write('')
        if platform.system() == "Windows":
            subprocess.call([r"C:/Program Files/Git/bin/bash.exe", bash_script_path, output_filedir] + data_paths)
        else: # for mac or linux
            subprocess.call([r"bash", bash_script_path, output_filedir] + data_paths)
    
    for filename in os.listdir(input_file_path):
        # Check if the file has a .csv extension
        if filename.endswith(".bc"):
            # Construct the full file path
            file_path = os.path.join(input_file_path, filename)
            # Delete the file
            if os.path.exists(file_path):
                os.remove(file_path)


def add_river_data(bc_paths):
    discharge_rivers_df = time_series_df.apply(convert_clim_to_discharge_units)
    discharge_rivers_df.to_csv(join(output_river_path, 'River_Climatology_Discharge_Time_Series.csv'), index=True)
    # discharge_rivers_df.to_csv(join(output_river_path, 'River_Climatology_Discharge_Time_Series_v2.csv'), index=True)

    discharge_rivers_df_year = adjust_dates_around_cutoff(discharge_rivers_df, '06-06', 2013)
    exclude= ['Esk', 'Alt', 'Clywd']
    prim_dataframe = discharge_rivers_df_year.drop(columns=exclude, errors='ignore')
    prim_dataframe = prim_dataframe.reindex(sorted(prim_dataframe.columns), axis=1)
    discharge_rivers_df_year = discharge_rivers_df_year.reindex(sorted(discharge_rivers_df_year.columns), axis=1) # all rivers
    #plt.figure();plt.plot(discharge_rivers_df_year['Esk'])
    for filepath in bc_paths[1]:
        print(filepath) # inside each filepath, all estuary forcings will be placed.
        riv_in_primea_path = util.md([filepath, 'rivers_daily_in_orig_primea'])
        all_riv_path = util.md([filepath, 'rivers_daily_all_no_duddon'])
        riv_dump_csv = util.md([filepath, 'rivers_daily_dump_csv'])
        
        #all rivers
        generate_bc_files(discharge_rivers_df_year, "2013-10-30", riv_dump_csv)
        file_stitcher(riv_dump_csv, all_riv_path)

        #prim_rivers
        generate_bc_files(prim_dataframe, "2013-10-30", riv_dump_csv)
        file_stitcher(riv_dump_csv, riv_in_primea_path)
        
    return discharge_rivers_df_year

# === SAVE A CATALOG OF CLIMATOLOGY FORCING CELLS FOR OUR ESTUARIES ===
from pathlib import Path

def save_climatology_estuary_catalog(
    out_csv: Path,
    riv_dict: dict,
    lons: list,
    lats: list,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    subset: list = None,
):
    """
    Write a CSV with one row per estuary:
      estuary, clim_row, clim_col, clim_lon, clim_lat
    riv_dict keys are integer indices into lons/lats/row_indices/col_indices (0-based).
    """
    rows = []
    for idx, name in riv_dict.items():
        if subset and name not in subset:
            continue
        # guard against any out-of-range/NaN
        if not (0 <= idx < len(lons)):
            print(f"[WARN] {name}: index {idx} out of range for climatology arrays")
            continue
        r = int(row_indices[idx]); c = int(col_indices[idx])
        lon = float(lons[idx]);    lat = float(lats[idx])
        rows.append({
            "estuary": name,
            "clim_row": r,
            "clim_col": c,
            "clim_lon": lon,
            "clim_lat": lat,
        })

    df = pd.DataFrame(rows).sort_values("estuary")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote climatology estuary catalog -> {out_csv}")

# Choose where to write it
catalog_out = Path(start_path) / "GitHub/EBM_WORKSHOP/output_data/climatology_estuary_cells.csv"

# If you only want your 8 estuaries:
subset_names = ["Dee","Duddon","Kent","Leven","Lune","Mersey","Ribble","Wyre"]

# IMPORTANT: fix a tiny loop bug above (if present) — use `for idx, river_name in riv_dict.items():`
# Call the writer:
save_climatology_estuary_catalog(
    catalog_out, riv_dict, lons, lats, row_indices, col_indices, subset=subset_names
)

if __name__ == '__main__':
    pass
    import glob
    main_path = join(start_path, r'modelling_DATA','kent_estuary_project',r'7.met_office')
    make_paths = DirGen(main_path)
    fn = glob.glob(join(main_path,'models','*'))[0]
    sub_path = make_paths.dir_outputs(os.path.split(fn)[1]) # Dealing with this model run. 
    bc_paths = make_paths.bc_outputs()
    
    discharge_rivers_df_year = add_river_data(bc_paths)
    
    discharge_rivers_df_year.Ribble.to_csv('river_data.csv')
    discharge_rivers_df_year.to_csv('all_rivers.csv')
    # discharge_rivers_df_year.Ribble.to_csv('river_data_v2.csv')
    # discharge_rivers_df_year.to_csv('all_rivers_v2.csv')
    
    
#%% Possibly usefull old junk code
# nov11_dec_mean_discharge = means[305:]
# jan_feb20_sum_discharge = sums[:51]
# nov11_dec_sum_discharge = sums[305:]

# nov11_dec_range = np.linspace(305,365,365-(305-1))
# jan_feb20_range = np.linspace(1,51,51)
# new_range = np.concatenate((nov11_dec_range,jan_feb20_range))

# new_data_mean_discharge = np.concatenate((nov11_dec_mean_discharge,jan_feb20_mean_discharge))
# new_data_sum_discharge = np.concatenate((nov11_dec_sum_discharge,jan_feb20_sum_discharge))

# plt.plot(new_data_sum_discharge)
# plt.ylim([0,25])

# ### Cartopy plotting 
# #%% Create a Cartopy plot
# fig = plt.figure(figsize=(10, 8))
# ax = plt.axes(projection=ccrs.PlateCarree())

# # Plot the valid locations using scatter plot
# plt.scatter(valid_longitudes, valid_latitudes, s=5, color='blue', label='Valid Data')

# # You can add more customization to your plot here, like adding coastlines, gridlines, etc.

# plt.title('Valid Data Locations')
# plt.legend()
# plt.show()das as gpd
# import matplotlib
# matplotlib.use('TkAgg')






# #%% calculate mean discharges 
# means = np.nanmean(data.rorunoff, axis=(1,2))                                  
# sums = np.nansum(data.rorunoff, axis=(1,2))
# #plt.plot(data.time_counter, means)
# #plt.plot(range(len(data.time_counter)), means)

# #rearannging data 
# jan_feb20_mean_discharge = means[:51]
# nov11_dec_mean_discharge = means[305:]
# jan_feb20_sum_discharge = sums[:51]
# nov11_dec_sum_discharge = sums[305:]

# nov11_dec_range = np.linspace(305,365,365-(305-1))
# jan_feb20_range = np.linspace(1,51,51)
# new_range = np.concatenate((nov11_dec_range,jan_feb20_range))

# new_data_mean_discharge = np.concatenate((nov11_dec_mean_discharge,jan_feb20_mean_discharge))
# new_data_sum_discharge = np.concatenate((nov11_dec_sum_discharge,jan_feb20_sum_discharge))

# plt.plot(new_data_sum_discharge)
# plt.ylim([0,25])

# ### Cartopy plotting 
# #%% Create a Cartopy plot
# fig = plt.figure(figsize=(10, 8))
# ax = plt.axes(projection=ccrs.PlateCarree())

# # Plot the valid locations using scatter plot
# plt.scatter(valid_longitudes, valid_latitudes, s=5, color='blue', label='Valid Data')

# # You can add more customization to your plot here, like adding coastlines, gridlines, etc.

# plt.title('Valid Data Locations')
# plt.legend()
# plt.show()




# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = gl.right_labels = False  # Updated attributes again for the second plot
# gl.xformatter = cticker.LongitudeFormatter()
# gl.yformatter = cticker.LatitudeFormatter()
# gl.xlabel_style = {'size': 12, 'color': 'black'}
# gl.ylabel_style = {'size': 12, 'color': 'black'}




# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = gl.right_labels = False  # Updated attributes
# gl.xformatter = cticker.LongitudeFormatter()
# gl.yformatter = cticker.LatitudeFormatter()
# gl.xlabel_style = {'size': 12, 'color': 'black'}
# gl.ylabel_style = {'size': 12, 'color': 'black'}
