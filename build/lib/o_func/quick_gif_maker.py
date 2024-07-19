#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:44:44 2024

@author: af
"""
import xarray as xr
import json

data = xr.open_dataset('/home/af/Documents/kent_31_merged_map.nc')
# Converting the header information to a serializable format

def serialize_attrs(attrs):
    return {key: str(value) for key, value in attrs.items()}

def serialize_coords(coords):
    return {key: str(value.values) for key, value in coords.items()}

header_info_serializable = {}
for var in data.variables:
    header_info_serializable[var] = {
        "dims": data[var].dims,
        "coords": serialize_coords(data[var].coords),
        "attrs": serialize_attrs(data[var].attrs)
    }

# Save header information to a JSON file for readability
output_file_path = '/home/af/Documents/kent_31_merged_map.json'
with open(output_file_path, 'w') as file:
    json.dump(header_info_serializable, file, indent=4)

output_file_path


waterdepth = data.mesh2d_s1
import matplotlib.pyplot as plt
import imageio
plt.scatter(data.mesh2d_face_x, data.mesh2d_face_y, c= waterdepth[2,:]); plt.colorbar()

# Extracting coordinates and water depth data
x_coords = data.mesh2d_face_x.values
y_coords = data.mesh2d_face_y.values
water_depth = data.mesh2d_s1.values

#%% r
# Create a list to store images for the GIF
images = []

# Number of frames and total duration
num_frames = 10
total_duration = 10  # seconds
duration_per_frame = total_duration / num_frames

# Create scatter plots for depth index from 0 to 9
for i in range(10):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x_coords, y_coords, c=water_depth[i, :], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Water Depth')
    plt.title(f'Water Depth at Index {i}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    # Save the plot to a temporary file
    temp_filename = f'/home/af/Documents/water_depth_{i}.png'
    plt.savefig(temp_filename)
    plt.close(fig)
    images.append(imageio.imread(temp_filename))

# Create a GIF
output_gif_path = '/home/af/Documents/water_depth.gif'
imageio.mimsave(output_gif_path, images, duration=500, loop = 0)

output_gif_path
