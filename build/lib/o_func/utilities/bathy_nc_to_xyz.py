#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:13:14 2024

@author: af
"""

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
from os.path import join
import glob

from o_func import opsys; start_path = opsys()


bathy_paths = join(start_path,'modelling_DATA','kent_estuary_project','bathymetry','final_bathies')

# Function to extract bathymetry data and save to XYZ and DBF files
def process_bathy_file(file_path):
    # Load the NetCDF file
    dataset = xr.open_dataset(file_path)

    # Check which variable names are available for bathymetry data
    if 'NetNode_z' in dataset.variables:
        bathymetry_data = dataset['NetNode_z'].values
        longitude = dataset['NetNode_x'].values
        latitude = dataset['NetNode_y'].values
    elif 'mesh2d_node_z' in dataset.variables:
        bathymetry_data = dataset['mesh2d_node_z'].values
        longitude = dataset['mesh2d_node_x'].values
        latitude = dataset['mesh2d_node_y'].values
    else:
        raise ValueError("Bathymetry data not found in the NetCDF file.")

    # Combine into XYZ format
    xyz_data = np.vstack((longitude, latitude, bathymetry_data)).T

    # Create a DataFrame for easy manipulation
    df_xyz = pd.DataFrame(xyz_data, columns=['x', 'y', 'z'])

    # Save to XYZ file
    xyz_output_file = os.path.splitext(file_path)[0] + '_bathy.xyz'
    df_xyz.to_csv(xyz_output_file, index=False, header=False, sep=' ')

    # Create a GeoDataFrame with British National Grid CRS (EPSG:27700)
    geometry = [Point(xy) for xy in zip(df_xyz['x'], df_xyz['y'])]
    gdf_xyz = gpd.GeoDataFrame(df_xyz, geometry=geometry, crs='EPSG:4326')
    # gdf_xyz.drop(columns='z', inplace=True)  # Drop 'z' column to simplify DBF why we dont want this. 

    # Save to DBF file
    dbf_output_file = os.path.splitext(file_path)[0] + '_bathy.dbf'
    gdf_xyz.to_file(dbf_output_file, driver='ESRI Shapefile')

    return xyz_output_file, dbf_output_file

# Process all NetCDF files in the specified directory
for file_path in glob.glob(os.path.join(bathy_paths, '*.nc')):
    try:
        xyz_file, dbf_file = process_bathy_file(file_path)
        print(f"Processed {file_path}:")
        print(f"  XYZ file: {xyz_file}")
        print(f"  DBF file: {dbf_file}")
    except ValueError as e:
        print(f"Error processing {file_path}: {e}")
