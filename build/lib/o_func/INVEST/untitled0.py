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
