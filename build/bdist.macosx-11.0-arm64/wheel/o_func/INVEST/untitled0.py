# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:08:05 2024

@author: aafur
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pyproj import Transformer
from o_func import opsys; start_path = opsys('Elements')

# File path and data loading
path = start_path + 'Original_Data/transects/Tyne_AMM7_nodes.csv'
data = pd.read_csv(path)
lat = data.Lat
lon = data.Lon

# Transformer for coordinates
to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

# Convert WGS84 to BNG
bng_coordinates = [to_bng.transform(lon[i], lat[i]) for i in range(len(lat))]
bng_df = pd.DataFrame(bng_coordinates, columns=["Easting", "Northing"])

# Plotting function
def plot_coords(use_bng=True):
    """
    Plots data points on a map, switching between BNG and WGS84 projections.
    
    Parameters:
    - use_bng: Boolean, if True plots in BNG (British National Grid), else in WGS84.
    """
    if use_bng:
        proj = ccrs.OSGB()  # BNG projection
        x, y = bng_df["Easting"], bng_df["Northing"]
        title = "Data Points in BNG (British National Grid)"
        grid_crs = None  # No gridlines for BNG
    else:
        proj = ccrs.PlateCarree()  # WGS84 projection
        x, y = lon, lat
        title = "Data Points in WGS84 (Latitude/Longitude)"
        grid_crs = ccrs.PlateCarree()

    # Create map
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=proj)

    # Add coastline and land features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor="black")

    # Plot points
    ax.scatter(x, y, color="red", s=50, label="Data Points", transform=grid_crs or proj)

    # Configure gridlines
    if grid_crs:
        gl = ax.gridlines(draw_labels=True, crs=grid_crs)
        gl.top_labels = False
        gl.right_labels = False
    else:
        # No gridlines for BNG
        gl = ax.gridlines(draw_labels=False)

    # Add title and legend
    plt.title(title)
    plt.legend()
    plt.show()

# Switch between BNG and WGS84
plot_coords(use_bng=False)  # Set to False for WGS84
