#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:23:40 2024

@author: af
"""

import xarray as xr
import matplotlib.pyplot as plt
filepath = '/Volumes/PN/Original_Data/UKC3/sliced/oa_riv/UKC4ao_1h_20131125_20131125_shelftmb_grid_T.nc' 

data = xr.open_dataset(filepath)

# sh = data.sossheig[0,:,:]
# lon = data.nav_lat
# lat = data.nav_lon

# plt.pcolor(lon,lat, sh)

import xarray as xr
# import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Extract the first time step of sea surface height
sh = data.sossheig[0, :, :]
lon = data.nav_lon
lat = data.nav_lat

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# Plot sea surface height
pcolor = ax.pcolormesh(lon, lat, sh, shading='auto', transform=ccrs.PlateCarree())
# Add coastlines for better geographic context
ax.coastlines()
# Optionally, add grid lines
ax.gridlines(draw_labels=True)
# Add a colorbar
plt.colorbar(pcolor, ax=ax, orientation='vertical', label='Sea Surface Height (m)')
plt.title('Sea Surface Height on the first timestep')
plt.show()
