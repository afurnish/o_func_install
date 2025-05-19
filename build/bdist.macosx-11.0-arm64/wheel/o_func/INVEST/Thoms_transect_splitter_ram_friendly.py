#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized for memory usage during NetCDF transect extraction
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import time
from pathlib import Path

start_time = time.time()

# INPUTS
input_folder = r'/Volumes/Elements/Original_Data/transects/raw_50_m_transects_only/'
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
csv_files = [csv_files[9]]  # Just one file for testing
print('Running:', csv_files)


'''
0 Blacklwater - South Thames
1 Colne - South Thames
2 Dart - Exe
3 Exe - Exe
4 Humber - Humber
5 Mersey - LivBay
6 Ribble - LivBay
7 Southampton - Portsmouth
8 TawTorridge - Severn
9 Tees - Tyne

'''

output_base_folder = r'/Volumes/PNC/GitHub/EBM_WORKSHOP/Transects/'

path = Path(r'/Volumes/Elements')
nc_paths = [
    # path / r'Original_Data/map_files/map_files/Exe/Jan_map/kent_31_merged_map.nc', # Exe
    # path / r'Original_Data/map_files/map_files/Humber/JAN map/kent_31_merged_map.nc', # Humber
    # path / r'Original_Data/map_files/map_files/Liv_Bay/kent_31_merged_map.nc' , # Liv Bay
    # path / r'Original_Data/map_files/map_files/Portsmouth/Jan map/kent_31_merged_map.nc', # Portsmouth
    # path / r'Original_Data/map_files/map_files/Severn/kent_31_merged_mapJANFEB.nc', # Severn
    # path / r'Original_Data/map_files/map_files/Thames North/JAN map/kent_31_merged_map.nc', # Thames North
    # path / r'Original_Data/map_files/map_files/Thames South/Jan map/kent_31_merged_map.nc', # Thames South
    path / r'Original_Data/map_files/map_files/Tyne/JANFEB_map/kent_31_merged_map.nc', # Tyne
    
    # r'/Volumes/PN/1temp_ELEMENTS_drive_files/Thom_sharing/Aaron/Aaron transect analysis/nc map files/50m_flow_ASTRO_hyd_map.nc'
]

print('In the domain of:', nc_paths[0].parts[6])


# VARS
# variables_to_extract = ['mesh2d_s1', 'mesh2d_sa1', 'mesh2d_q1', 'mesh2d_ucx', 'mesh2d_ucy', 'mesh2d_waterdepth']
# variable_names = ['TIDE', 'SAL', 'DIS', 'velX', 'velY', 'DEPTH']
variables_to_extract = ['time']
variable_names = ['TIME']
chunk_size = 100  # adjust to manage memory usage

for nc_path in nc_paths:
    print(f"Opening NetCDF: {nc_path}")
    ds = xr.open_dataset(nc_path, chunks={"time": chunk_size, "mesh2d_face": 50000})

    x = ds['mesh2d_face_x'].load().values
    y = ds['mesh2d_face_y'].load().values
    time_index = pd.to_datetime(ds['time'].isel(time=slice(1, None)).values)  # skip t=0

    for csv_file in csv_files:
        csv_path = os.path.join(input_folder, csv_file)
        csv_data = pd.read_csv(csv_path)
        csv_x = csv_data['x'].values
        csv_y = csv_data['y'].values

        # Precompute nearest nodes once
        nearest_indices = [
            np.argmin(np.hypot(x - px, y - py)) for px, py in zip(csv_x, csv_y)
        ]

        for var_name, output_name in zip(variables_to_extract, variable_names):
            print(f"Processing variable: {var_name}")

            # Output setup
            output_folder = os.path.join(output_base_folder, output_name)
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{os.path.splitext(csv_file)[0]}_{output_name}.csv")

            # Prepare column headers
            headers = ['Datetime'] + [f'Point_{i+1}' for i in range(len(nearest_indices))]
            with open(output_file, 'w') as f:
                f.write(','.join(headers) + '\n')

            # Loop in chunks over time
            total_timesteps = ds.sizes['time'] - 1  # skip first
            for t0 in range(1, total_timesteps, chunk_size):
                t1 = min(t0 + chunk_size, total_timesteps)
                data_chunk = ds[var_name].isel(time=slice(t0, t1)).load()  # lazy load
                time_chunk = pd.to_datetime(ds['time'].isel(time=slice(t0, t1)).values)

                # Build rows to append
                rows = []
                for ti in range(t1 - t0):
                    row = [time_chunk[ti].strftime('%Y-%m-%d %H:%M:%S')]
                    for idx in nearest_indices:
                        row.append(str(data_chunk[ti, idx].item()))
                    rows.append(','.join(row))

                with open(output_file, 'a') as f:
                    f.write('\n'.join(rows) + '\n')

            print(f"Saved data for {output_name} to {output_file}")

    ds.close()

print('Finished in (seconds):', time.time() - start_time)
