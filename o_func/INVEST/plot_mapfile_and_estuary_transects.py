#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:11:13 2025

@author: af
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

humber_map = r'/media/af/Elements/Original_Data/map_files/map_files/Humber/JAN map/kent_31_merged_map.nc'
tyne_map = r'/media/af/Elements/Original_Data/map_files/map_files/Tyne/JANFEB_map/kent_31_merged_map.nc'

humber_transect = r'/media/af/Elements/Original_Data/transects/raw_50_m_transects_only/50m_HUMBER_transect.csv'
tees_transect = r'/media/af/Elements/Original_Data/transects/raw_50_m_transects_only/50m_TEES_transect.csv'

# run = 'humber' 
run = 'tyne'

if run == 'humber':
    trans = humber_transect
    maps = humber_map
elif run == 'tyne':
    trans = tees_transect
    maps = tyne_map


load_csv = pd.read_csv(trans)

data = xr.open_dataset(maps, chunks = 100)

x = data.mesh2d_face_x
y = data.mesh2d_face_y

sh = data.mesh2d_s1
print(sh.shape)

#%% Convert lat lon 
from pyproj import Transformer
import numpy as np

# Your WGS84 lat/lon data (to overlay)
lons = load_csv.x# or from your dataset
lats = load_csv.y

# Create a transformer from WGS84 → OSGB36
transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

# Convert
x_27700, y_27700 = transformer.transform(lons, lats)

#%% 
# Open with chunking (optional but helps prevent overload)
data = xr.open_dataset(maps, chunks={'time': 40})

# Grab data (raw, no conversions)
x = data.mesh2d_face_x.values
y = data.mesh2d_face_y.values
z = data.mesh2d_s1.isel(time=30).values  # just 1st timestep

# Plot with smallest possible scatter markers
plt.scatter(x, y, c=z, cmap='viridis', s=0.5, marker='.', linewidths=0)
# plt.axis('off')  # remove axes for speed
plt.tight_layout()
plt.colorbar()
plt.show()

plt.scatter(x_27700, y_27700, s = 1)


def plot_map_time(data):
    
    # Grab data (raw, no conversions)
    x = data.mesh2d_face_x.values
    y = data.mesh2d_face_y.values
    z = data.mesh2d_s1.isel(time=30).values  # just 1st timestep
    
    # Plot with smallest possible scatter markers
    plt.scatter(x, y, c=z, cmap='viridis', s=0.5, marker='.', linewidths=0)
    # plt.axis('off')  # remove axes for speed
    plt.tight_layout()
    plt.colorbar()
    plt.show()