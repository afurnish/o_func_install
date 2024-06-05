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
from o_func import opsys, DirGen; start_path = opsys()
import pandas as pd
import geopandas as gpd
import o_func.utilities as util
import subprocess
import pkg_resources
import platform


#b20_mean_discharge = means[:51]

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

plt.legend()

# Define the UK's latitude and longitude bounds for the second plot
uk_lon_min, uk_lon_max = -3.65, -2.75
uk_lat_min, uk_lat_max = 53.20, 54.52

riv_dict = {129: 'Esk',
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

shapefile_path = start_path + "modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_KENT_EPSG_4326_clipped_med_domain.shp"
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

plt.tight_layout()
plt.legend()
plt.show()




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
if __name__ == '__main__':
    import glob
    # We are adding in a function here to save the data to a modelling file path. 
    main_path = join(start_path, r'modelling_DATA','kent_estuary_project',r'7.met_office')
    make_paths = DirGen(main_path)
    fn = glob.glob(join(main_path,'models','*'))[0]
    sub_path = make_paths.dir_outputs(os.path.split(fn)[1]) # Dealing with this model run. 
    bc_paths = make_paths.bc_outputs()
    
    add_river_data(bc_paths)
    
    
    
    
    
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
