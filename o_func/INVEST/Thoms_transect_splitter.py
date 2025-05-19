#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:39:50 2025

@author: af
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import time
start_time = time.time()
# Define input folder for CSV files
# input_folder = r'/Volumes/PN/1temp_ELEMENTS_drive_files/Thom_sharing/Aaron/Aaron transect analysis/50m_transects'
input_folder = r'/Volumes/Elements/Original_Data/transects/raw_50_m_transects_only/'
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

csv_files = [csv_files[0]]
'''
0 Blacklwater - North Thames
1 Colne - North Thames
2 Dart - Exe
3 Exe - Exe
4 Humber - Humber
5 Mersey - LivBay
6 Ribble - LivBay
7 Southampton - Portsmouth
8 TawTorridge - Severn
9 Tees - Tyne

'''
print(csv_files)
# Define output folder
# output_base_folder = r'/Volumes/PN/1temp_ELEMENTS_drive_files/Thom_sharing/Aaron/Transects/50m_m2s2_storage'
output_base_folder = r'/Volumes/PNC/GitHub/EBM_WORKSHOP/Transects/'
# Define paths to the NetCDF files

from pathlib import Path
path = Path(r'/Volumes/Elements')
nc_paths = [
    # path / r'Original_Data/map_files/map_files/Exe/Jan_map/kent_31_merged_map.nc', # Exe
    # path / r'Original_Data/map_files/map_files/Humber/JAN map/kent_31_merged_map.nc', # Humber
    # path / r'Original_Data/map_files/map_files/Liv_Bay/kent_31_merged_map.nc' , # Liv Bay
    # path / r'Original_Data/map_files/map_files/Portsmouth/Jan map/kent_31_merged_map.nc', # Portsmouth
    # path / r'Original_Data/map_files/map_files/Severn/kent_31_merged_mapJANFEB.nc', # Severn
    # path / r'Original_Data/map_files/map_files/Thames North/JAN map/kent_31_merged_map.nc', # Thames North
    path / r'Original_Data/map_files/map_files/Thames South/Jan map/kent_31_merged_map.nc', # Thames South
    # path / r'Original_Data/map_files/map_files/Tyne/JANFEB_map/kent_31_merged_map.nc', # Tyne
    
    # r'/Volumes/PN/1temp_ELEMENTS_drive_files/Thom_sharing/Aaron/Aaron transect analysis/nc map files/50m_flow_ASTRO_hyd_map.nc'
]

# Loop through each NetCDF file
for nc_path in nc_paths:
    # Open NetCDF file using xarray
    ds = xr.open_dataset(nc_path)

    # Extract x and y coordinates from the NetCDF file
    x = ds['mesh2d_face_x'].values
    y = ds['mesh2d_face_y'].values

    # Extract time variable and convert to datetime
    time = pd.to_datetime(ds['time'].values[1:])  # Skip the first time step

    # Define variables to extract
    variables_to_extract = ['mesh2d_s1', 'mesh2d_sa1', 'mesh2d_q1', 'mesh2d_ucx', 'mesh2d_ucy', 'mesh2d_waterdepth']
    variable_names = ['TIDE', 'SAL', 'DIS', 'velX', 'velY', 'DEPTH']

    # Loop through each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(input_folder, csv_file)
        csv_data = pd.read_csv(csv_path)

        # Extract x and y columns from the CSV
        csv_x = csv_data['x'].values
        csv_y = csv_data['y'].values

        # Loop through each variable
        for var_name, output_name in zip(variables_to_extract, variable_names):
            # Extract variable data from NetCDF
            variable_data = ds[var_name].values[:, 1:]  # Skip the first time step

            # Allocate matrix for extracted values
            extracted_values = np.full((len(csv_data), len(time)), np.nan)

            # Find nearest indices for each point in the CSV
            for i, (px, py) in enumerate(zip(csv_x, csv_y)):
                distances = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                nearest_index = np.argmin(distances)
                extracted_values[i, :] = variable_data[1:, nearest_index]

            # Transpose extracted_values to match time-series format
            extracted_values = extracted_values.T

            # Create a DataFrame for the extracted data
            time_str = time.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime as string
            extracted_data = pd.DataFrame(extracted_values, columns=[f'Point_{i+1}' for i in range(extracted_values.shape[1])])
            extracted_data.insert(0, 'Datetime', time_str)

            # Define the output folder and save
            output_folder = os.path.join(output_base_folder, output_name)
            os.makedirs(output_folder, exist_ok=True)

            output_file = os.path.join(output_folder, f"{os.path.splitext(csv_file)[0]}_{output_name}.csv")
            extracted_data.to_csv(output_file, index=False)

            print(f"Saved data for {output_name} to {output_file}")

    # Close NetCDF dataset
    ds.close()

print('Finished in (seconds):', time.time() - start_time)