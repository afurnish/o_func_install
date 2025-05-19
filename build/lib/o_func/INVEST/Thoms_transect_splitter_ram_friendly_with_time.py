#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized for memory usage during NetCDF transect extraction, including special handling for the 'time' variable.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import time
from pathlib import Path
from sklearn.neighbors import BallTree
from pyproj import Transformer

start_time = time.time()

# D_path = Path('/Volumes/Elements/')
# U_path = Path(r'/Volumes/PNC/')
# D_path = Path('D:/')
# U_path = Path(r'U:/')
D_path = Path('/media/af/Elements')
U_path = Path(r'/media/af/PNC')


# INPUTS
input_folder = D_path / r'Original_Data/transects/raw_50_m_transects_only/'
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

output_base_folder = U_path / r'GitHub/EBM_WORKSHOP/Transects/'

nc_paths = [
    # D_path / r'Original_Data/map_files/map_files/Exe/Jan_map/kent_31_merged_map.nc',
    # D_path / r'Original_Data/map_files/map_files/Humber/JAN map/kent_31_merged_map.nc',
    # D_path / r'Original_Data/map_files/map_files/Liv_Bay/kent_31_merged_map.nc',
    # D_path / r'Original_Data/map_files/map_files/Portsmouth/Jan map/kent_31_merged_map.nc',
    # D_path / r'Original_Data/map_files/map_files/Severn/kent_31_merged_mapJANFEB.nc',
    # D_path / r'Original_Data/map_files/map_files/Thames North/JAN map/kent_31_merged_map.nc',
    # D_path / r'Original_Data/map_files/map_files/Thames South/Jan map/kent_31_merged_map.nc',
    D_path / r'Original_Data/map_files/map_files/Tyne/JANFEB_map/kent_31_merged_map.nc',  # Example: Tyne
]

print('In the domain of:', nc_paths[0].parts[6])

# VARIABLES
variables_to_extract = ['mesh2d_s1', 'mesh2d_sa1', 'mesh2d_q1', 'mesh2d_ucx', 'mesh2d_ucy', 'mesh2d_waterdepth', 'time']
variable_names = ['TIDE', 'SAL', 'DIS', 'velX', 'velY', 'DEPTH', 'TIME']
# variables_to_extract = ['mesh2d_s1']  # Use spatial vars for real runs
# variable_names = ['TIDE']
chunk_size = 100  # adjust for memory control

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

    
        # Precompute nearest mesh face once
        # Convert degrees to radians for BallTree
        if 'projected_coordinate_system' in ds:
            crs_var = ds['projected_coordinate_system']
            transformer2 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
            x_lon, y_lat = transformer2.transform(x, y)
            print("Map Data is in a BNG")
        elif 'wgs84' in ds:
            crs_var = ds['wgs84']
            x_lon, y_lat = x, y
            print("Map Data is in WGS84")
        else:
            raise ValueError("No recognized CRS variable found in dataset.")

        
        mesh_radians = np.radians(np.column_stack((y_lat, x_lon)))  # lat, lon
        csv_radians = np.radians(np.column_stack((csv_y, csv_x)))

        tree = BallTree(mesh_radians, metric='haversine')
        distances, nearest_indices = tree.query(csv_radians, k=1)
        # distances_km = distances[:, 0] * 6371.0  # Earth radius in km



        for var_name, output_name in zip(variables_to_extract, variable_names):
            print(f"Processing variable: {var_name}")

            output_folder = os.path.join(output_base_folder, output_name)
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{os.path.splitext(csv_file)[0]}_{output_name}.csv")

            if var_name == 'time':
                # Special case for 'time' variable (no spatial dim)
                time_vals = pd.to_datetime(ds['time'].isel(time=slice(1, None)).values)

                with open(output_file, 'w') as f:
                    f.write('Datetime,All_Points\n')
                    for t in time_vals:
                        t_str = t.strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"{t_str},{t_str}\n")

                print(f"Saved time data to {output_file}")

            else:
                # Regular case for spatial variables
                headers = ['Datetime'] + [f'Point_{i+1}' for i in range(len(nearest_indices))]
                with open(output_file, 'w') as f:
                    f.write(','.join(headers) + '\n')

                total_timesteps = ds.sizes['time'] - 1  # skip first
                for t0 in range(1, total_timesteps, chunk_size):
                    t1 = min(t0 + chunk_size, total_timesteps)
                    data_chunk = ds[var_name].isel(time=slice(t0, t1)).load()
                    time_chunk = pd.to_datetime(ds['time'].isel(time=slice(t0, t1)).values)

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

#%% Plot the finished points
from pyproj import Transformer
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Reproject transect points from EPSG:27700 (OSGB36) to EPSG:4326 (WGS84)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
csv_lon, csv_lat = transformer.transform(csv_x, csv_y)

transformer2 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# # KDTREE METHOD
# # Create tree using the NetCDF face coordinates (already in EPSG:4326)
# points_map = np.column_stack((x, y))
# tree = cKDTree(points_map)
# # Query nearest mesh face for each transect point
# points_csv = np.column_stack((csv_lon, csv_lat))
# distances, nearest_indices = tree.query(points_csv, k=1)



# #%% OLD METHOD
# nearest_indices = [
#             np.argmin(np.hypot(x - px, y - py))
#             for px, py in zip(csv_x, csv_y)
#         ]

if 'projected_coordinate_system' in ds:
    crs_var = ds['projected_coordinate_system']
    xplot, yplot = x, y
    x_lon, y_lat = transformer2.transform(x, y)

elif 'wgs84' in ds:
    xplot, yplot = transformer.transform(x, y)
    x_lon, y_lat = x, y

else:
    raise ValueError("No recognized CRS variable found in dataset.")

#% BALLTREE METHOD 

from sklearn.neighbors import BallTree
# Convert degrees to radians for BallTree
mesh_radians = np.radians(np.column_stack((y_lat, x_lon)))  # lat, lon
csv_radians = np.radians(np.column_stack((csv_y, csv_x)))

tree = BallTree(mesh_radians, metric='haversine')
distances, nearest_indices = tree.query(csv_radians, k=1)
distances_km = distances[:, 0] * 6371.0  # Earth radius in km

# Sanity check: plot
plt.figure(figsize=(10, 6))
plt.scatter(xplot, yplot, s=1, label="Mesh Faces", alpha=0.3)
plt.plot(csv_lon, csv_lat, 'r.-', label='Transect Points (Reprojected)')
plt.scatter(xplot[nearest_indices], yplot[nearest_indices], c='black', s=10, label='Matched Mesh Points')
plt.title('Nearest Index Matching: Transect vs Mesh')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
