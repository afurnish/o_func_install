#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Mon Jan 29 11:51:32 2024
@author: af
"""

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import numpy as np
from o_func import opsys 
from pathlib import Path
start_path = Path(opsys('PN'))
elements_path = Path(opsys('Elements'))

# Function to interpolate points along a line
def interpolate_line(line, spacing=100):
    # Calculate the total length of the line
    line_length = line.length
    # Create distances along the line at specified intervals
    distances = np.arange(0, line_length, spacing)
    # Generate points at each distance
    points = [line.interpolate(distance) for distance in distances]
    return points

# Load the shapefile
shapefile_path = elements_path / "INVEST_modelling/grid_generation/qgis_files/Humber_coast_line_27700.shp"
coastline = gpd.read_file(shapefile_path)

# Combine all LineString geometries into one
combined_line = coastline.geometry.union_all()

# Handle MultiLineString
if isinstance(combined_line, MultiLineString):
    # Merge into a single LineString by concatenating coordinates
    all_coords = []
    for line in combined_line.geoms:  # Use .geoms to iterate over MultiLineString
        all_coords.extend(line.coords)
    combined_line = LineString(all_coords)

# Ensure the combined geometry is a LineString
if not isinstance(combined_line, LineString):
    raise ValueError("The shapefile does not contain valid LineString geometries.")

# Interpolate the combined line into points with 100m spacing
spacing = 100  # Spacing in meters
points = interpolate_line(combined_line, spacing=spacing)

# Generate .pol file content
pol_file_content = [
    "*",
    "* Deltares, RGFGRID Version 7.03.00.77422 (Win64), Nov 30 2022, 15:52:41",
    "* File creation date: 2024-12-02, 17:58:42",
    "*",
    "* Coordinate System = Cartesian",
    "*",
    "L000001",
    f"         {len(points)}           2",
]

# Add points to the .pol file content
for point in points:
    pol_file_content.append(f"   {point.x:.7E}   {point.y:.7E}")

# Save to .pol file
output_path = elements_path / Path("INVEST_modelling/grid_generation/qgis_files/Humber_coast_line_27700.pol")
with open(output_path, "w") as pol_file:
    pol_file.write("\n".join(pol_file_content))

print(f".pol file saved to {output_path}")
