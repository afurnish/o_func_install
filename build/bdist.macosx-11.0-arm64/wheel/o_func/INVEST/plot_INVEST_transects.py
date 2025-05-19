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
from o_func import opsys
start_path = opsys('Elements')

# File path and data loading
path = 'D:/Original_Data/transects/Humber_AMM7_nodes.csv'

# File path and data loading
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
    
    return x, y 

# Switch between BNG and WGS84
x, y = plot_coords(use_bng=True)  # Set to False for WGS84

#%% 
import datetime

def create_polygon_file(x, y, output_file):
    """
    Creates a polygon file in the specified Deltares RGFGRID format.

    Args:
        x (pd.Series): Series of X coordinates (Easting).
        y (pd.Series): Series of Y coordinates (Northing).
        output_file (str): Path to the output file.
    """
    # Generate the header
    header = f"""\
* Deltares, RGFGRID Version 7.03.00.77422 (Win64), Nov 30 2022, 15:52:41
* File creation date: {datetime.datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}
*
* Coordinate System = Cartesian
*
L000001
        {len(x)}           2"""

    # Prepare the body
    body = "\n".join(f"   {x_val:.7E}   {y_val:.7E}" for x_val, y_val in zip(x, y))

    # Combine header and body
    content = f"{header}\n{body}"

    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)

# Example usage
# Replace with your actual data
import pandas as pd


output_file = "polygon_file.pol"
create_polygon_file(x, y, output_file)
