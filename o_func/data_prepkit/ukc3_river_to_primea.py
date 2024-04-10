#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file should find the nearest rivers in the river climatology of UKC4 datasets
and extract them into suitable variables so that they may be used with running 
simulations that are proportional to the met office. 

Created on Thu Mar  7 12:15:29 2024

@author: af
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from  os.path import join 
from o_func import opsys; start_path = opsys()
import pandas as pd
import geopandas as gpd
# import matplotlib
# matplotlib.use('TkAgg')

data = xr.open_dataset(join(start_path, 'Original_Data','UKC3','river_climatology','rivers','AMM15_River_Climatology.nc'))

runoff = np.array(data.rorunoff[0,:,:])
non_zero_mask = np.where(runoff != 0.0)
row_indices, col_indices = non_zero_mask

lons = []
lats = []

for i in range(len(row_indices)):
    lons.append(data.lon[row_indices[i], col_indices[i]].item())
    lats.append(data.lat[row_indices[i], col_indices[i]].item())

# Define the UK's latitude and longitude bounds for the first plot
uk_lon_min, uk_lon_max = -10.5, 2
uk_lat_min, uk_lat_max = 49, 61.5

fig = plt.figure(figsize=(10, 8), dpi = 150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([uk_lon_min, uk_lon_max, uk_lat_min, uk_lat_max])
ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='red')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='dotted', edgecolor='black')
ax.scatter(lons, lats, marker='o', color='blue', label='River Climatology Discharge')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
uk_extent_lon = np.linspace(-10, 2, 13)
uk_extent_lat = np.linspace(48, 62, 15)

ax.set_xticks(uk_extent_lon, crs=ccrs.PlateCarree())
ax.set_yticks(uk_extent_lat, crs=ccrs.PlateCarree())
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = gl.right_labels = False  # Updated attributes
# gl.xformatter = cticker.LongitudeFormatter()
# gl.yformatter = cticker.LatitudeFormatter()
# gl.xlabel_style = {'size': 12, 'color': 'black'}
# gl.ylabel_style = {'size': 12, 'color': 'black'}

plt.legend()

# Define the UK's latitude and longitude bounds for the second plot
uk_lon_min, uk_lon_max = -3.65, -2.75
uk_lat_min, uk_lat_max = 53.20, 54.52

riv_dict = {129: 'Est',
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

shapefile_path = "/Volumes/PN/modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_KENT_EPSG_4326_clipped_med_domain.shp"
gdf = gpd.read_file(shapefile_path)

fig = plt.figure(figsize=(10, 12), dpi = 150)
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
for i, (lon, lat) in enumerate(zip(lons, lats), start=1):
    if i in riv_dict:
        ax.text(lon - 0.02, lat + 0.01, f'{riv_dict[i]}', ha='center', va='bottom', fontsize=10, color='blue')
        # ax.text(lon + 0.02, lat - 0.05, f'{i}\n{riv_dict[i]}', ha='center', va='bottom', fontsize=10, color='green')
    
    # else:
    #     ax.text(lon + 0.02, lat - 0.02, str(i), ha='center', va='bottom', fontsize=10, color='black')

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
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = gl.right_labels = False  # Updated attributes again for the second plot
# gl.xformatter = cticker.LongitudeFormatter()
# gl.yformatter = cticker.LatitudeFormatter()
# gl.xlabel_style = {'size': 12, 'color': 'black'}
# gl.ylabel_style = {'size': 12, 'color': 'black'}
plt.tight_layout()
plt.legend()
plt.show()



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
    
    # Save the river data to a file
    # Assuming you want to save it as a CSV for example. Adjust the path as needed.
    river_data.to_csv(join(path_to_bc_file, filename), header=False)
    
    # If you prefer to save it in a different format or with specific formatting,
    # you may need to adjust the saving method accordingly.