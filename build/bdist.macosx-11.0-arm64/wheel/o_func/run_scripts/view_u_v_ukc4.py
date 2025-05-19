
# import matplotlib.pyplot as plt
# import xarray as xr
# import numpy as np
# U = xr.open_dataset('/Volumes/PN/Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_U.nc')
# V = xr.open_dataset('/Volumes/PN/Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_V.nc')

# lat = U['nav_lat']
# lon = U['nav_lon']
# meanUsurf = np.mean(U.vozocrtx_top, axis=0)
# meanVsurf = np.mean(V.vomecrty_top, axis=0)

# from scipy.ndimage import uniform_filter

# # Window size for averaging (5x5 grid points)
# window_size = 1

# # Apply uniform filter for spatial averaging
# u_avg = uniform_filter(meanUsurf, size=window_size)
# v_avg = uniform_filter(meanVsurf, size=window_size)
# magnitude_avg = np.sqrt(u_avg**2 + v_avg**2)

# # Re-apply skipping to avoid overcrowding the plot
# # skip_avg = (slice(None, None, 5), slice(None, None, 5))

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 8))
# quiv_avg = ax.quiver(lon, lat, u_avg, v_avg, magnitude_avg, scale=5)
# # quiv_avg = ax.quiver(lon[skip_avg], lat[skip_avg], u_avg[skip_avg], v_avg[skip_avg], magnitude_avg[skip_avg], scale=5)
# ax.set_title('Averaged Surface Velocity Map')
# plt.colorbar(quiv_avg, ax=ax, label='Velocity magnitude (m/s)')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:55:19 2024

@author: af
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.ndimage import generic_filter
from os.path import join

from o_func import opsys; start_path = opsys()
# Open datasets
U = xr.open_dataset(join(start_path ,'Original_Data','UKC3','sliced','oa','shelftmb_cut_to_domain','UKC4ao_1h_20131030_20131030_shelftmb_grid_U.nc'))
V = xr.open_dataset(join(start_path ,'Original_Data','UKC3','sliced','oa','shelftmb_cut_to_domain','UKC4ao_1h_20131030_20131030_shelftmb_grid_V.nc'))

lat = U['nav_lat']
lon = U['nav_lon']
meanUsurf = np.mean(U.vozocrtx_top, axis=0)
meanVsurf = np.mean(V.vomecrty_top, axis=0)

# NaN-aware uniform filter function
def nan_aware_uniform_filter(data, size=3):
    def nan_mean_filter(values):
        return np.nanmean(values)
    return generic_filter(data, nan_mean_filter, size=size, mode='nearest')

# Option to apply NaN-aware filter or not
apply_filter = True  # Set this to False if you don't want to apply the filter

# Apply filtering based on the option
if apply_filter:
    window_size = 1  # Change this to adjust the window size for the filter
    u_avg = nan_aware_uniform_filter(meanUsurf, size=window_size)
    v_avg = nan_aware_uniform_filter(meanVsurf, size=window_size)
else:
    u_avg = meanUsurf
    v_avg = meanVsurf

magnitude_avg = np.sqrt(u_avg**2 + v_avg**2)

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
quiv_avg = ax.quiver(lon, lat, u_avg, v_avg, magnitude_avg, scale=5)
ax.set_title('Averaged Surface Velocity Map ' + str(window_size) + 'x' + str(window_size))
plt.colorbar(quiv_avg, ax=ax, label='Velocity magnitude (m/s)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()
