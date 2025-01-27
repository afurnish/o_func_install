# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:26:25 2025

@author: aafur
"""

from pathlib import Path
import xarray as xr
import rasterio
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import time

# Start timing the script
start_time = time.time()

# File paths
thames_path = Path('C:/Users/aafur/Downloads/Thames/Thames/thames_domain_full_net.nc')
thames_tiff = Path('C:/Users/aafur/Downloads/Thames.tif')
output_xyz = Path('C:/Users/aafur/Downloads/Thames_xyz_output.txt')

# Step 1: Load the NetCDF file
step_start = time.time()
xarray_dataset = xr.open_dataset(thames_path)
x = xarray_dataset.mesh2d_face_x.values
y = xarray_dataset.mesh2d_face_y.values
print(f"Loaded NetCDF file in {time.time() - step_start:.2f} seconds")

# Step 2: Load the TIFF file and prepare raster grid
step_start = time.time()
with rasterio.open(thames_tiff) as src:
    raster = src.read(1)  # Read the first band of the raster
    raster[raster == src.nodata] = np.nan  # Replace no-data values with NaN
    raster_bounds = src.bounds
    raster_transform = src.transform

    # Create a grid of the raster coordinates
    rows, cols = raster.shape
    xs = np.linspace(raster_bounds.left, raster_bounds.right, cols)
    ys = np.linspace(raster_bounds.top, raster_bounds.bottom, rows)
    ys = ys[::-1]  # Reverse to match raster's top-to-bottom layout
    xx, yy = np.meshgrid(xs, ys)
print(f"Prepared raster grid in {time.time() - step_start:.2f} seconds")

# Step 3: Interpolate raster values using cKDTree (Revised Chunking Logic)
step_start = time.time()
print("Building KD-Tree for fast interpolation...")

# Flatten raster grid
raster_points = np.column_stack((xx.ravel(), yy.ravel()))
raster_values = raster.ravel()

# Filter out NaN values from the raster
valid_mask = ~np.isnan(raster_values)
valid_raster_points = raster_points[valid_mask]
valid_raster_values = raster_values[valid_mask]

# Build KD-Tree using only valid raster points
tree = cKDTree(valid_raster_points)

# Initialize output array for z-values
z = np.full_like(x, np.nan)  # Initialize with NaN for diagnostics

# Process in chunks
chunk_size = 10000  # Number of points to process in each chunk
num_chunks = (len(x) + chunk_size - 1) // chunk_size  # Ensure all points are included

print("Starting interpolation with progress tracking...")
for i in tqdm(range(num_chunks), desc="Interpolating", unit="chunks"):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(x))  # Handle last chunk correctly
    current_points = np.column_stack((x[start:end], y[start:end]))
    _, indices = tree.query(current_points)
    z[start:end] = valid_raster_values[indices]  # Populate the corresponding z values

# Verify all points were processed
missed_points = np.sum(np.isnan(z))
if missed_points > 0:
    print(f"Warning: {missed_points} points could not be interpolated.")
else:
    print("All points successfully interpolated.")

print(f"Interpolated raster values in {time.time() - step_start:.2f} seconds")

# Step 4: Combine and save the results
step_start = time.time()
xyz = np.column_stack((x, y, z))  # Combine x, y, z into a single array
np.savetxt(output_xyz, xyz, delimiter=',', header='x,y,z', comments='', fmt='%.6f')
print(f"Saved XYZ file to {output_xyz} in {time.time() - step_start:.2f} seconds")

# Total time
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")

#%% take 2 

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:26:25 2025

@author: aafur
"""

from pathlib import Path
import xarray as xr
import rasterio
import numpy as np
from scipy.interpolate import griddata

# File paths
thames_path = Path('C:/Users/aafur/Downloads/Thames/Thames/thames_domain_full_net.nc')
thames_tiff = Path('C:/Users/aafur/Downloads/Thames.tif')
output_xyz = Path('C:/Users/aafur/Downloads/Thames_xyz_output.txt')

# Load the netCDF file to get the x and y coordinates
xarray_dataset = xr.open_dataset(thames_path)
x = xarray_dataset.mesh2d_face_x.values
y = xarray_dataset.mesh2d_face_y.values

# # Load the TIFF file using rasterio
# with rasterio.open(thames_tiff) as src:
#     raster = src.read(1)  # Read the first band of the raster
#     raster[raster == src.nodata] = np.nan  # Replace no-data values with NaN
#     raster_bounds = src.bounds
#     raster_transform = src.transform

#     # Create a grid of the raster coordinates
#     rows, cols = raster.shape
#     xs = np.linspace(raster_bounds.left, raster_bounds.right, cols)
#     ys = np.linspace(raster_bounds.top, raster_bounds.bottom, rows)
#     ys = ys[::-1]  # Reverse to match raster's top-to-bottom layout
#     xx, yy = np.meshgrid(xs, ys)

# # Flatten the raster grid and values for interpolation
# raster_points = np.column_stack((xx.ravel(), yy.ravel()))
# raster_values = raster.ravel()

# Interpolate raster values onto the (x, y) points
# z = griddata(raster_points, raster_values, (x, y), method='linear')

#pyinterp method
# import pyinterp
# grid = pyinterp.Grid2D(raster_points, raster_values)
# z = grid.interpolate(x, y, method='linear')

#dask_method 
# Load the TIFF file using rasterio
with rasterio.open(thames_tiff) as src:
    raster = src.read(1)  # Read the first band of the raster
    raster[raster == src.nodata] = np.nan  # Replace no-data values with NaN
    raster_bounds = src.bounds

    # Create a grid of the raster coordinates
    rows, cols = raster.shape
    xs = np.linspace(raster_bounds.left, raster_bounds.right, cols)
    ys = np.linspace(raster_bounds.top, raster_bounds.bottom, rows)
    ys = ys[::-1]  # Reverse to match raster's top-to-bottom layout

    # Create an Xarray DataArray for the raster
    raster_da = xr.DataArray(
        raster,
        dims=("y", "x"),
        coords={
            "x": xs,
            "y": ys
        }
    )

# Perform interpolation using Dask and Xarray
interpolated_da = raster_da.interp(
    x=("points", x),  # Target x-coordinates
    y=("points", y),  # Target y-coordinates
    method="linear"   # Interpolation method
)

# Extract the interpolated values
z = interpolated_da.values

# Combine x, y, and z into a single array
xyz = np.column_stack((x, y, z))

# Save the results as an XYZ file
np.savetxt(output_xyz, xyz, delimiter=',', header='x,y,z', comments='', fmt='%.6f')

print(f"XYZ file has been saved to {output_xyz}")

#%% 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colorbar as mcolorbar

# Step 5: Plot the resultant XYZ data
def plot_xyz(x, y, z, output_image_path=None):
    """
    Plots the XYZ data as a scatter or heatmap.

    Args:
        x (array): X coordinates.
        y (array): Y coordinates.
        z (array): Z values (elevation or interpolated values).
        output_image_path (str, optional): File path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=z, cmap="viridis", s=1, marker='o', edgecolor='none', alpha=0.8)
    plt.colorbar(scatter, label="Interpolated Z Values")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title("XYZ Interpolation Result")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save or show the plot
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_image_path}")
    else:
        plt.show()

# Plotting the result
plot_xyz(x, y, z, output_image_path='C:/Users/aafur/Downloads/Thames_plot.png')

#%% 
import matplotlib.pyplot as plt

# Plot the original (x, y) points from the NetCDF
plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=1, c='blue', alpha=0.8, label='NetCDF (x, y) points')
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.title("Original (x, y) Points from NetCDF")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
