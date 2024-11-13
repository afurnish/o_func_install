# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:33:31 2024

@author: aafur
"""

import xarray as xr
import pandas as pd
file = 'C:/Users/aafur/Downloads/SOUTH_FULL_net.nc'
data = xr.open_dataset(file)
x = data.mesh2d_face_x
y = data.mesh2d_face_y

# Create a pandas DataFrame
df = pd.DataFrame({
    'x': x,
    'y': y
})

# Save DataFrame to a CSV file
output_file = 'C:/Users/aafur/Downloads/SOUTH_mesh2d_coordinates.csv'
df.to_csv(output_file, index=False)

print(f"CSV file saved as {output_file}")
