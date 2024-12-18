# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:29:14 2024

@author: aafur
"""

import pyarrow
if not hasattr(pyarrow, "__version__"):
    pyarrow.__version__ = "1.0.0"

import pandas as pd
import xarray as xr
from osgeo import gdal

# File paths
tiff_path = r'C:/Users/aafur/Downloads/Ports_fin.tif'
netcdf_path = r'C:/Users/aafur/Downloads/SOUTH_FULL_net (1).nc'
output_xyz = "C:/Users/aafur/Downloads/output_points.xyz"

# Load NetCDF data
data = xr.open_dataset(netcdf_path)

# Extract x and y coordinates
x = data.mesh2d_face_x.values
y = data.mesh2d_face_y.values

# Open the TIFF file
raster = gdal.Open(tiff_path)
band = raster.GetRasterBand(1)
band.SetNoDataValue(999)
geotransform = raster.GetGeoTransform()

# Helper function to convert world coordinates to pixel coordinates
def world_to_pixel(geo_transform, x, y):
    """
    Converts geographic coordinates to raster grid coordinates.
    """
    pixel_x = int((x - geo_transform[0]) / geo_transform[1])
    pixel_y = int((y - geo_transform[3]) / geo_transform[5])
    return pixel_x, pixel_y

# Extract values at the specified points
extracted_data = []
for xi, yi in zip(x, y):
    px, py = world_to_pixel(geotransform, xi, yi)
    if 0 <= px < raster.RasterXSize and 0 <= py < raster.RasterYSize:
        value = band.ReadAsArray(px, py, 1, 1)[0][0]
        extracted_data.append((xi, yi, value))

# Save extracted data to XYZ file
with open(output_xyz, 'w') as f:
    for xi, yi, value in extracted_data:
        f.write(f"{xi} {yi} {value}\n")

print(f"Extracted data saved to {output_xyz}")

#%% 
import pandas as pd
import matplotlib.pyplot as plt

# Path to the output points file
output_xyz = "C:/Users/aafur/Downloads/output_points.xyz"

# Load the points file into a DataFrame
data = pd.read_csv(output_xyz, delim_whitespace=True, header=None, names=["x", "y", "value"])

z_min = -60  # Replace with your desired minimum value
z_max = 30  # Replace with your desired maximum value

# Plot the points
plt.figure(figsize=(10, 8))
plt.scatter(data["x"], data["y"], c=data["value"], cmap="viridis", s=0.5, vmin=z_min, vmax=z_max)
plt.colorbar(label="Value")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Point Values from XYZ File")
plt.grid(True)
plt.show()



