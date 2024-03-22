#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:55:19 2024

@author: af
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
U = xr.open_dataset('/Volumes/PN/Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_U.nc')
V = xr.open_dataset('/Volumes/PN/Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_V.nc')

lat = U['nav_lat']
lon = U['nav_lon']
meanUsurf = np.mean(U.vozocrtx_top, axis=0)
meanVsurf = np.mean(V.vomecrty_top, axis=0)

from scipy.ndimage import uniform_filter

# Window size for averaging (5x5 grid points)
window_size = 1

# Apply uniform filter for spatial averaging
u_avg = uniform_filter(meanUsurf, size=window_size)
v_avg = uniform_filter(meanVsurf, size=window_size)
magnitude_avg = np.sqrt(u_avg**2 + v_avg**2)

# Re-apply skipping to avoid overcrowding the plot
# skip_avg = (slice(None, None, 5), slice(None, None, 5))

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
quiv_avg = ax.quiver(lon, lat, u_avg, v_avg, magnitude_avg, scale=5)
# quiv_avg = ax.quiver(lon[skip_avg], lat[skip_avg], u_avg[skip_avg], v_avg[skip_avg], magnitude_avg[skip_avg], scale=5)
ax.set_title('Averaged Surface Velocity Map')
plt.colorbar(quiv_avg, ax=ax, label='Velocity magnitude (m/s)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()
