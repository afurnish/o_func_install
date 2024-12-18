#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:17:45 2024

@author: af
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Create some sample data
data = np.random.rand(10, 10) * 100

# Use a different built-in colormap from matplotlib if cmocean is not available
cmap = cm.Blues  # Using a blue-themed colormap similar to ocean colors
norm = mcolors.Normalize(vmin=0, vmax=100)

# Plot the data
fig, ax = plt.subplots()
c = ax.imshow(data, cmap=cmap, norm=norm)

# Create a colorbar with pointy ends and adjust the extendfrac for a sharper point
cbar = fig.colorbar(c, extend='both', extendfrac=0.1, orientation='vertical')

# Display the plot
plt.show()



# Create some sample data
data = np.random.rand(10, 10) * 100

# Use a colormap from matplotlib
cmap = cm.Blues
norm = mcolors.Normalize(vmin=0, vmax=100)

# Plot the data
fig, ax = plt.subplots()
c = ax.imshow(data, cmap=cmap, norm=norm)

# Create a colorbar using the same axis object, placing it next to the plot
cbar = plt.colorbar(c, ax=ax, extend='both', extendfrac=0.1, orientation='vertical')

# Display the plot
plt.show()

#%% 
import numpy as np
import matplotlib.pyplot as plt

# Create a sample 20000x20000 array
data = np.random.rand(20000, 20000)

# Use 'imshow' with optimized settings
plt.imshow(data, cmap='viridis', interpolation='none', aspect='auto')
plt.colorbar()
plt.show()
#%% 
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr

# Create a sample 20000x20000 array
data = np.random.rand(20000, 20000)

# Convert the numpy array to an xarray DataArray
data_xr = xr.DataArray(data)

# Use datashader to rasterize the data
cvs = ds.Canvas(plot_width=20000, plot_height=20000)
agg = cvs.raster(data_xr)

# Convert to an image using datashader
img = tf.shade(agg, cmap='viridis')

# Display the image
img.to_pil().show()
