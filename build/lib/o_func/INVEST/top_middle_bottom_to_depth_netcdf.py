#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:26:10 2024

@author: af
"""
import xarray as xr
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gc
from o_func import opsys

start_path = Path(opsys())

full_path = start_path / 'Original_Data/UKC3/og/shelftmb'
end_path = start_path / 'Original_Data/UKC3/og/shelftmb_combined_to_3_layers_for_tmb'

# Ensure the output directory exists
end_path.mkdir(parents=True, exist_ok=True)

# Define the list of suffixes and corresponding depth values
suffixes = ['_top', '_mid', '_bot']
depth_values = [0, 1, 2]

# Dictionary to hold categorized files
files_dict = {'T': [], 'U': [], 'V': []}

# Categorize files based on 'T', 'U', 'V'
for tuv in files_dict.keys():
    for file in full_path.glob(f'*201311*{tuv}*.nc'):
        files_dict[tuv].append(file)

def create_combined_structure(data):
    # Create a new dataset to store the combined data
    combined_data = xr.Dataset(coords={
        'y': data.coords['y'],
        'x': data.coords['x'],
        'time_counter': data.coords['time_counter'],
        'deptht': ('deptht', depth_values),
        'nav_lat': data.coords['nav_lat'],
        'nav_lon': data.coords['nav_lon']
    })
    return combined_data

def process_and_combine_data(data, combined_data):
    # Loop through the data variables to find top, mid, bot variables
    for var in data.data_vars:
        for i, suffix in enumerate(suffixes):
            if var.endswith(suffix):
                base_name = var.rsplit(suffix, 1)[0]
                combined_var_name = base_name

                if combined_var_name not in combined_data:
                    # Initialize the data array with NaNs
                    combined_data[combined_var_name] = xr.DataArray(
                        np.full((len(depth_values), data.sizes['time_counter'], data.sizes['y'], data.sizes['x']), np.nan),
                        dims=('deptht', 'time_counter', 'y', 'x'),
                        coords={
                            'deptht': ('deptht', depth_values),
                            'time_counter': data.coords['time_counter'],
                            'y': data.coords['y'],
                            'x': data.coords['x'],
                        }
                    )

                # Assign the data to the appropriate depth layer
                combined_data[combined_var_name][i, :, :, :] = data[var].values

    # Include other variables that are not split into top/mid/bot layers
    for var in data.data_vars:
        if not any(var.endswith(suffix) for suffix in suffixes):
            combined_data[var] = data[var]
    return combined_data

def process_file(file_path):
    try:
        # Open the dataset
        data = xr.open_dataset(file_path)
        
        # Create the combined data structure
        combined_data = create_combined_structure(data)

        # Process and combine the data
        combined_data = process_and_combine_data(data, combined_data)

        # Define the output file path
        output_filename = file_path.name.replace('.nc', '_combined.nc')
        output_filepath = end_path / output_filename

        # Save the new dataset to a file with compression
        combined_data.to_netcdf(output_filepath, encoding={var: {'zlib': True} for var in combined_data.data_vars})
        print(f"Processed {file_path} -> {output_filepath}")
        
        # Clean up memory
        del data
        del combined_data
        gc.collect()  # Force garbage collection

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Using ThreadPoolExecutor to parallelize file processing
with ThreadPoolExecutor(max_workers=6) as executor:
    for category, files in files_dict.items():
        executor.map(process_file, files)

print("All files processed.")
