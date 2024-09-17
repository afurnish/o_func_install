#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:55:06 2024

@author: af
"""

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from pyugrid.ugrid import UGrid
import numpy as np
from o_func import opsys; start_path = Path(opsys())

def path(path):
    new_path = start_path / Path(path)
    return new_path

# Load the original dataset
file_path_original = path("modelling_DATA/kent_estuary_project/grid/netcdf_delft_grids/original_model_grid.nc")  # Replace with the path to the original NetCDF file
ds_original = xr.open_dataset(file_path_original)

# Load the new dataset
file_path_new = path("modelling_DATA/kent_estuary_project/grid/netcdf_delft_grids/extended_rivers_to_combat_saline_intrusion.nc") # Replace with the path to the new NetCDF file
ds_new = xr.open_dataset(file_path_new)

# Extract node coordinates from the original dataset
x_original = ds_original['NetNode_x'].values
y_original = ds_original['NetNode_y'].values

# Extract node coordinates from the new dataset
x_new = ds_new['mesh2d_node_x'].values
y_new = ds_new['mesh2d_node_y'].values

# Determine the points that are unique to the new dataset
original_points = set(zip(x_original, y_original))
new_points = set(zip(x_new, y_new))

# Points unique to the new dataset
new_unique_points = new_points - original_points
x_new_unique, y_new_unique = zip(*new_unique_points) if new_unique_points else ([], [])

# Plotting the original and new unique points
plt.figure(figsize=(12, 12))
plt.scatter(x_original, y_original, s=1, color='gray', label='Original Dataset', alpha=0.5)
plt.scatter(x_new_unique, y_new_unique, s=1, color='blue', label='New Dataset Unique', alpha=0.5)
plt.title('Comparison of Unstructured Grid Points with Unique New Points Highlighted')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()

#%% Plot the urgids
# Determine correct variable names for node coordinates and face nodes
def get_variable_names(ds):
    if 'NetNode_x' in ds.variables and 'NetNode_y' in ds.variables:
        node_x_var = 'NetNode_x'
        node_y_var = 'NetNode_y'
        face_node_var = 'NetElemNode'
    elif 'mesh2d_node_x' in ds.variables and 'mesh2d_node_y' in ds.variables:
        node_x_var = 'mesh2d_node_x'
        node_y_var = 'mesh2d_node_y'
        face_node_var = 'mesh2d_face_nodes'
    else:
        raise ValueError("Unknown dataset format: cannot determine coordinate variable names.")
    return node_x_var, node_y_var, face_node_var


# %% Plotting up the unstructured grids. 
i = 0

trilonvert = []
trilatvert = []
trifaces = []
squares = []
for ds_path in [file_path_original, file_path_new]:
    i = i + 1 
    
    ds = xr.open_dataset(ds_path)
    node_x_var, node_y_var, face_node_var = get_variable_names(ds)
    x = ds[node_x_var].values
    y = ds[node_y_var].values
    faces = ds[face_node_var].values
    
    ug = UGrid.from_ncfile(ds_path)
    face_nodes = faces[:,0:3] -1
# Extract the vertex coordinates and connectivity information
    vertices = ug.nodes[:,0], ug.nodes[:,1]  # Assumes 'lon' and 'lat' are the variable names for longitude and latitude in your UGRID dataset
    faces = ug.faces  
    filled_faces = faces.filled(-999999)
    valid_quad_mask = (filled_faces[:, 3] != -2147483647) & (filled_faces[:, 3] != -999999)
    square_nodes = filled_faces[valid_quad_mask, :4]
   
    #%% THIS ONE WORKS
    fig, ax = plt.subplots(dpi = 300)
    

    index_rows = np.where(faces.mask[:, 3])[0]
    # Plot the triangles using the vertex coordinates and faces
    ax.triplot(vertices[0], vertices[1], faces.data[index_rows,0:3], linewidth=0.1, color='black')
    
    trilonvert.append(vertices[0])
    trilatvert.append(vertices[1])
    trifaces.append(faces.data[index_rows,0:3])
    
    # Set the plot title and labels
    ax.set_title('UGRID Grid Plot')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    lons = vertices[0]
    lats = vertices[1]
    
    # Show the plot
    plt.show()
    squares.append(square_nodes)
    
    for square in square_nodes:
        ax.plot([lons[square[0]], lons[square[1]], lons[square[2]], lons[square[3]], lons[square[0]]],
        [lats[square[0]], lats[square[1]], lats[square[2]], lats[square[3]], lats[square[0]]], 'k-',linewidth=0.25)
    ax.triplot(vertices[0], vertices[1], faces.data[index_rows,0:3], linewidth=0.1, color='black')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

#%%
fig, ax = plt.subplots(figsize=(4, 12), dpi=300)

# Set a uniform line width for plotting
line_width = 0.2

# Plot the larger grid (new grid) in blue
ax.triplot(trilonvert[1], trilatvert[1], trifaces[1], linewidth=line_width, color='blue', alpha=0.5, label='New Grid')

# Plot the smaller grid (original grid) in black on top
ax.triplot(trilonvert[0], trilatvert[0], trifaces[0], linewidth=line_width, color='black', label='Original Grid')

# Optionally highlight unique squares in the new dataset in red
for square in squares[1]:  # Assuming squares[1] corresponds to the new dataset
    ax.plot([trilonvert[1][square[0]], trilonvert[1][square[1]], trilonvert[1][square[2]], trilonvert[1][square[3]], trilonvert[1][square[0]]],
            [trilatvert[1][square[0]], trilatvert[1][square[1]], trilatvert[1][square[2]], trilatvert[1][square[3]], trilatvert[1][square[0]]], 
            'r-', linewidth=line_width, label='Unique Squares in New Grid')

# Customize the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles[:3], labels[:3], loc='upper right', frameon=False)  # Select only necessary labels and disable frame

# Make the lines in the legend thicker
for line in legend.get_lines():
    line.set_linewidth(2.0)

ax.set_title('Comparison of Unstructured Grids')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

