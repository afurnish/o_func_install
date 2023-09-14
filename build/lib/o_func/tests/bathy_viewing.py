# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:00:15 2023

@author: aafur
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import xarray as xr
from scipy.interpolate import griddata


path = 'F:/modelling_DATA/kent_estuary_project/6.Final2/models/01_kent_2.0.0_no_wind/2.0.0_wind_testing_4_months.dsproj_data/bed_level_deepened_channel(testing).xyz'

data = []

# Open the file and read line by line
with open(path, 'r') as file:
    for line in file:
        # Split each line into three values and convert them to float
        values = line.split()
        values = [float(val) for val in values]
        data.append(values)
data_array = np.array(data)

column1 = data_array[:, 0]
column2 = data_array[:, 1]
column3 = data_array[:, 2]


# Set the desired color range limits
color_min = -40
color_max = 20

# Create a colormap that maps values between color_min and color_max
cmap = plt.cm.get_cmap('viridis')
norm = plt.Normalize(vmin=color_min, vmax=color_max)

fig, ax = plt.subplots()
plt.scatter(column1, column2, c = column3, s = 1,cmap=cmap, norm=norm)


# Now to do the interpolated bathymetry

bathy_path = glob.glob(r'F:\modelling_DATA\kent_estuary_project\6.Final2\models\01_kent_2.0.0_no_wind\2.0.0_wind_testing_4_months.dsproj_data\FlowFM\*.nc')
bd = xr.open_dataset(bathy_path[0], engine='scipy')
# Bathymetry data at node coordinates
node_x = bd.mesh2d_node_z.mesh2d_node_x.values
node_y = bd.mesh2d_node_z.mesh2d_node_y.values
bathymetry_node = bd.mesh2d_node_z.values

# Face coordinates
face_x = bd.mesh2d_face_x.values
face_y = bd.mesh2d_face_y.values

# Reshape face coordinates for griddata input
face_coords = np.column_stack((face_x, face_y))

# Interpolate bathymetry data to face coordinates
bathymetry_face = griddata((node_x, node_y), bathymetry_node, face_coords, method='linear')

fig2, ax2 = plt.subplots()
plt.scatter(face_x, face_y, c = bathymetry_face, s = 1,cmap=cmap, norm=norm)


