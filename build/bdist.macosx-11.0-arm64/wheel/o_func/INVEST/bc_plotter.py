#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:58:08 2024

@author: af
"""

import xarray as xr
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load data
data = xr.open_dataset('/Users/af/Downloads/kent_regrid.nc')
ribble_data = pd.read_csv('/Volumes/PN/modelling_DATA/kent_estuary_project/river_boundary_conditions/original_river_data/processed/copy_Ribble_Samlesbury_2003-2020.csv')

# Load the third dataset from the file
forcing_file = '/Users/af/Downloads/Discharge.bc'

with open(forcing_file, 'r') as file:
    lines = file.readlines()

# Parse the third dataset
start_time = datetime(2013, 10, 31, 0, 0, 0)
time_seconds = []
discharge_values = []

for line in lines:
    if line.strip() and not line.startswith('#') and '=' not in line and not line.startswith('['):
        parts = line.split()
        time_seconds.append(int(parts[0]))
        discharge_values.append(float(parts[1]))

forcing_data = pd.DataFrame({
    'time': [start_time + timedelta(seconds=s) for s in time_seconds],
    'discharge': discharge_values
})

# Extract 2D arrays of longitude and latitude
navlon = data.nav_lon.values  # 2D array of longitude
navlat = data.nav_lat.values  # 2D array of latitude

# Flatten the arrays
lon_flat = navlon.flatten()
lat_flat = navlat.flatten()

# Combine into a DataFrame
lat_lon_df = pd.DataFrame({'x': lon_flat, 'y': lat_flat})

# Example point to search for nearest neighbor
points_to_search = lat_lon_df[['x', 'y']].values

# Convert points to radians for Haversine calculation
lon_lat_array_rad = np.radians(points_to_search)  # DataFrame of x and y points to search
ball_tree = BallTree(lon_lat_array_rad, metric='haversine')

# Example latitude and longitude for querying
latlon = {'x': [-3.0565639], 'y': [53.715130]}
lat_londf = pd.DataFrame(latlon)

# Convert the points to radians
points_rad = np.radians(lat_londf)

# Query the BallTree for nearest neighbor
distances, indices = ball_tree.query(points_rad, k=1)

ribble_loc = lat_lon_df.iloc[indices[0][0]]
print("Distances:", distances)
print("Indices:", indices)

# Find the index of the nearest neighbor in the original 2D array
nearest_index_flat = indices[0][0]

# Convert the flat index back to 2D index
nearest_index_2d = np.unravel_index(nearest_index_flat, navlon.shape)

# Extract the time series of tide data using the 2D index
tide_time_series = data.prim_surface_height[:, nearest_index_2d[0], nearest_index_2d[1]].values

# Extract the time coordinate for plotting
time = data.time_primea.values

# Convert the time to a pandas DatetimeIndex
time_index = pd.to_datetime(time)

# Create a pandas Series for easier plotting
tide_series = pd.Series(tide_time_series, index=time_index)

# Prepare ribble_data for plotting
ribble_data.columns = ['time', 'discharge']
ribble_data['time'] = pd.to_datetime(ribble_data['time'], format='%d/%m/%Y %H:%M:%S')
ribble_data.set_index('time', inplace=True)

# Set the x-axis limits
start_date = '2014-02-01'
end_date = '2014-03-01'

# Filter ribble_data and forcing_data to the same date range as tide_series
ribble_data_filtered = ribble_data[start_date:end_date]
forcing_data_filtered = forcing_data[(forcing_data['time'] >= start_date) & (forcing_data['time'] <= end_date)]

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the tide time series
line1, = ax1.plot(tide_series, label='Tide Level', color='blue')
ax1.set_xlabel('Time')
ax1.set_ylabel('Tide Level', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))

# Create a second y-axis
ax2 = ax1.twinx()
line2, = ax2.plot(ribble_data_filtered.index, ribble_data_filtered['discharge'], label='Discharge (Ribble)', color='red')
ax2.set_ylabel('Discharge (m^3/s)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Create a third y-axis
ax3 = ax1.twinx()
line3, = ax3.plot(forcing_data_filtered['time'], forcing_data_filtered['discharge'], label='Discharge (Forcing)', color='green')
ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
ax3.set_ylabel('Discharge (flood event, 35 m^3/s)', color='green')
ax3.tick_params(axis='y', labelcolor='green')

plt.xticks(rotation=45)

# Add a grid
ax1.grid(True)

# Combine legends
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1.15))

# Add legends
fig.tight_layout()

plt.title('Time Series of Tide Level and Discharge')
plt.show()
