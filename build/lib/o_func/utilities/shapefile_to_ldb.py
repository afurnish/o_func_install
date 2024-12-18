#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Mon Jan 29 11:51:32 2024
@author: af
"""

import geopandas as gpd
import os

def convert_shapefile_to_ldb(input_shapefile, output_file):
    # Read the shapefile using GeoPandas
    gdf = gpd.read_file(input_shapefile)
    
    # Open the output file for writing
    with open(output_file, 'w') as f:
        # Iterate through each geometry (line) in the shapefile
        for idx, row in gdf.iterrows():
            # Get the geometry (line) and its ID
            geometry = row.geometry
            line_id = f"L{idx + 1:05d}"  # Formatting the line ID like L00001, L00002, etc.

            # Prepare the coordinates (x, y) for the line
            coords = list(geometry.coords)
            num_rows = len(coords)

            # Write the line ID and number of points
            f.write(f"{line_id}\n")
            f.write(f"{num_rows}    2\n")  # We have 2 columns, x and y

            # Write each coordinate pair (x, y)
            for coord in coords:
                f.write(f"{coord[0]:.7f}    {coord[1]:.7f}\n")  # Writing x, y coordinates with 7 decimals

            print(f"Line {line_id} written.")

    print(f"Conversion complete. All lines written to {output_file}")



#%% 

if __name__ == "__main__":
    # Input shapefile and output file
    def run_lbd():
        input_shapefile = input("Enter the path to the input shapefile: ")
        output_file = input("Enter the output file name (e.g., output.ldb): ")
    
        # Convert the shapefile
        convert_shapefile_to_ldb(input_shapefile, output_file)
        
 
run_lbd()