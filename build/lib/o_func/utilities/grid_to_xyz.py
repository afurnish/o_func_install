# # -*- coding: utf-8 -*-
# """
# Created on Wed Jul 24 12:02:57 2024

# @author: aafur
# """

# import netCDF4 as nc
# import numpy as np
# import pandas as pd

# # Load the NetCDF file
# file_path = 'SW_domain_FULL_net.nc'
# dataset = nc.Dataset(file_path)

# # Inspect the contents of the NetCDF file
# print(dataset)

# # Extract face x and y coordinates
# face_x = dataset.variables['mesh2d_face_x'][:]
# face_y = dataset.variables['mesh2d_face_y'][:]

# # Combine into XYZ format (assuming z=0 for 2D data)
# xyz_data = np.vstack((face_x, face_y, np.zeros_like(face_x))).T

# # Create a DataFrame for easy manipulation and export
# df_xyz = pd.DataFrame(xyz_data, columns=['x', 'y', 'z'])

# # Save to XYZ file
# output_file = 'face_coordinates.xyz'
# df_xyz.to_csv(output_file, index=False, header=False, sep=' ')

# output_file

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import os 
os.chdir(r'C:\Users\aafur\Downloads')
# Load the NetCDF file using xarray
file_path = 'SW_domain_FULL_net.nc'
dataset = xr.open_dataset(file_path)

# Extract face x and y coordinates
face_x = dataset['mesh2d_face_x'].values
face_y = dataset['mesh2d_face_y'].values

# Combine into XYZ format (assuming z=0 for 2D data)
xyz_data = np.vstack((face_x, face_y, np.zeros_like(face_x))).T

# Create a DataFrame for easy manipulation
df_xyz = pd.DataFrame(xyz_data, columns=['x', 'y', 'z'])

# Save to XYZ file
xyz_output_file = 'face_coordinates.xyz'
df_xyz.to_csv(xyz_output_file, index=False, header=False, sep=' ')

# Create a GeoDataFrame with British National Grid CRS (EPSG:27700)
geometry = [Point(xy) for xy in zip(df_xyz['x'], df_xyz['y'])]
gdf_xyz = gpd.GeoDataFrame(df_xyz, geometry=geometry, crs='EPSG:27700')
gdf_xyz.drop(columns='z', inplace=True)  # Drop 'z' column to simplify DBF

# Save to DBF file
dbf_output_file = 'face_coordinates.dbf'
gdf_xyz.to_file(dbf_output_file, driver='ESRI Shapefile')

xyz_output_file, dbf_output_file

