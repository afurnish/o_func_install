#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The aim of this data would be to generate a scatter plot of measured salinity values alongside 
predicted salinity values for the PRIMEA model as well as the Met Office UKC4 model. This provides 
sufficient salinity validation for my model albeit sporadic. It shoud be noted that for teh salinity 
to spin up sufficiently much of this data could be lost. 

However we will ensure to compare every single salinity point across the model to ensure homogeneity. 


Created on Tue Oct 15 10:59:31 2024
@author: af
"""
import re
from o_func import opsyst
from o_func import uk_bounds
from pathlib import Path
import pandas as pd
start_path = Path(opsyst('PN'))
data = start_path / Path('Original_Data/salinity')

for file in data.glob('*.csv'):
    if not file.name.startswith('.'):  # Ignore hidden files
        df = pd.read_csv(file)
        
        
# Step 1: Drop rows where 'Time' is NaN or cannot be converted properly
df = df.dropna(subset=['Time'])

#%%
# Step 2: Function to clean and standardize the 'Time' column
def clean_time(time):
    try:
        time_str = str(time).strip()  # Strip any leading/trailing spaces
        
        # Step 1: Remove all non-numeric characters
        time_str = re.sub(r'\D', '', time_str)  # Remove all characters that are not digits
        
        # Step 2: Ensure the string is at least 3 or 4 digits long
        if len(time_str) == 3:  # E.g., '905' -> '09:05'
            time_str = time_str.zfill(4)  # Pad with leading zeros
        elif len(time_str) != 4:
            return None  # If it's not 3 or 4 digits, it's invalid

        # Step 3: Split into hours and minutes
        hours = time_str[:2]
        minutes = time_str[2:]

        # Step 4: Validate the time (hours should be between 00-23, minutes between 00-59)
        if int(hours) < 0 or int(hours) > 23:
            return None
        if int(minutes) < 0 or int(minutes) > 59:
            return None

        # Return in HH:MM format
        return f"{hours}:{minutes}"
    
    except:
        return None
# Step 3: Apply the cleaning function to the 'Time' column
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df['Time_cleaned'] = df['Time'].apply(clean_time)

#%%
df = df.dropna(subset=['Time_cleaned'])
df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time_cleaned'] )

# Define the date range
start_date = '2013-10-30'
end_date = '2014-02-28'

# Filter the DataFrame for rows where the 'Region' is 'North West'
north_west_df = df[df['Region'] == 'North West']

# Further filter for entries between the defined date range
filtered_df = north_west_df[(north_west_df['Date'] >= start_date) & (north_west_df['Date'] <= end_date)]

lon, lat = uk_bounds()

# Define the bounding box coordinates
lon_min, lon_max = min(lon), max(lon)
lat_min, lat_max = min(lat), max(lat)

# Further filter based on latitude and longitude being within the bounding box
filtered_df_within_box = filtered_df[
    (filtered_df['Lon'] >= lon_min) & (filtered_df['Lon'] <= lon_max) &
    (filtered_df['Lat'] >= lat_min) & (filtered_df['Lat'] <= lat_max)
]


#%% Plot the map 
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # Assuming 'filtered_df_within_box' contains the filtered data

# # Create a figure and an axis with a specific projection
# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes(projection=ccrs.PlateCarree())

# # Set the extent of the map to your bounding box
# ax.set_extent([-3.65, -2.75, 53.2, 54.52], crs=ccrs.PlateCarree())

# # Add features to the map (coastlines, borders, land, and ocean)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS)
# ax.add_feature(cfeature.LAND, edgecolor='black')
# ax.add_feature(cfeature.OCEAN)

# # Plot each data point from the DataFrame
# ax.scatter(
#     filtered_df_within_box['Lon'], 
#     filtered_df_within_box['Lat'], 
#     color='red', 
#     s=50, 
#     label='Data Points',
#     transform=ccrs.PlateCarree()
# )

# # Add gridlines for better readability
# ax.gridlines(draw_labels=True)

# # Add a title and legend
# plt.title('Filtered Data Points on Map')
# plt.legend()

# # Show the plot
# plt.show()

# #%% Number point count 
# location_counts = filtered_df_within_box.groupby(['Lat', 'Lon']).size().reset_index(name='Count')

# # Create a figure and an axis with a specific projection
# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes(projection=ccrs.PlateCarree())

# # Set the extent of the map to your bounding box
# ax.set_extent([-3.65, -2.75, 53.2, 54.52], crs=ccrs.PlateCarree())

# # Add features to the map (coastlines, borders, land, and ocean)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS)
# ax.add_feature(cfeature.LAND, edgecolor='black')
# ax.add_feature(cfeature.OCEAN)

# # Plot each location with its count
# for idx, row in location_counts.iterrows():
#     ax.text(
#         row['Lon'], row['Lat'], 
#         str(int(row['Count'])), 
#         color='blue', 
#         fontsize=8, 
#         ha='center', 
#         va='center',
#         transform=ccrs.PlateCarree()
#     )

# # Add gridlines for better readability
# ax.gridlines(draw_labels=True)

# # Add a title
# plt.title('Number of Data Points at Each Location')

# # Show the plot
# plt.show()
#%% Plot the map with adjustments
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Assuming 'filtered_df_within_box' contains the filtered data

# Create a figure and an axis with a specific projection
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the extent of the map to your bounding box
ax.set_extent([-3.65, -2.75, 53.2, 54.52], crs=ccrs.PlateCarree())

# Add higher resolution coastlines (10m)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                edgecolor='black', facecolor='none'))

# Add other features (borders, land, and ocean)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)

# Plot each data point from the DataFrame
ax.scatter(
    filtered_df_within_box['Lon'], 
    filtered_df_within_box['Lat'], 
    color='red', 
    s=50, 
    label='Data Points',
    transform=ccrs.PlateCarree()
)

# Add gridlines for better readability
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Manually add x and y labels
ax.text(-3.2, 53.2, 'Longitude', va='center', ha='center', fontsize=12, transform=ccrs.PlateCarree())
ax.text(-3.65, 53.85, 'Latitude', va='center', ha='center', rotation='vertical', fontsize=12, transform=ccrs.PlateCarree())

# Show the plot without the title (since it will be part of a figure)
plt.show()

#%% Number point count with adjustments
location_counts = filtered_df_within_box.groupby(['Lat', 'Lon']).size().reset_index(name='Count')

# Create a figure and an axis with a specific projection
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the extent of the map to your bounding box
ax.set_extent([-3.65, -2.75, 53.2, 54.52], crs=ccrs.PlateCarree())

# Add higher resolution coastlines (10m)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                edgecolor='black', facecolor='none'))

# Add other features (borders, land, and ocean)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)

# Plot each location with its count
for idx, row in location_counts.iterrows():
    ax.text(
        row['Lon'], row['Lat'], 
        str(int(row['Count'])), 
        color='blue', 
        fontsize=8, 
        ha='center', 
        va='center',
        transform=ccrs.PlateCarree()
    )

# Add gridlines for better readability
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Manually add x and y labels
ax.text(-3.2, 53.1, 'Longitude', va='center', ha='center', fontsize=12, transform=ccrs.PlateCarree())
ax.text(-3.82, 53.85, 'Latitude', va='center', ha='center', rotation='vertical', fontsize=12, transform=ccrs.PlateCarree())

# Show the plot without the title (since it will be part of a figure)
plt.show()
