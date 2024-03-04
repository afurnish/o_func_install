#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:56:11 2024

@author: af
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar



data = r'/Volumes/PN/modelling_DATA/kent_estuary_project/6.Final2/models/02_kent_1.0.0_UM_wind/kent_31_merged_map.nc'

load_data = xr.open_dataset(data, chunks = {'time':'auto'})

wd = load_data.mesh2d_s1

wd_max = []
wd = wd[504:,:]
wd_dask = da.from_array(wd, chunks=(wd.shape[0], 'auto'))

# wd_dask = da.from_array(wd, chunks=(wd.shape[0], 'auto'))
def compute_nanmax_with_progress(array):
    with ProgressBar():
        result = np.nanmax(array, axis=0).compute()
    return result

import time
start = time.time()
wd_max = compute_nanmax_with_progress(wd_dask)
print('Finish')

# for i in range(wd.shape[1]):
#     if i % 1000 == 0:
#         print(i)
#     wd_max.append(np.nanmax(wd[:,i]))
np.save('max_value',np.array(wd_max))

# Plotting it up. 
import geopandas as gpd
UKWEST_loc = r'/Volumes/PN/modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
UKWEST = gpd.read_file(UKWEST_loc)
xxx = wd.mesh2d_face_x
yyy = wd.mesh2d_face_y 

from matplotlib import cm

colormaps = cm.viridis

nan_indices = np.where(np.isnan(wd_max))[0]
xxx = np.delete(xxx, nan_indices)
yyy = np.delete(yyy, nan_indices)
wd_nan_removed = np.delete(wd_max, nan_indices)

#%%
from shapely.geometry import Point, Polygon
import matplotlib.tri as tri

# points_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xxx, yyy)])
# points_within_land = gpd.sjoin(points_gdf, UKWEST, how='inner', op='within')
# mask = np.full(xxx.shape, False, dtype=bool)
# # Set True for points that are within land areas (to be masked)
# mask[points_within_land.index] = True
# # wd_nan_removed_masked = np.where(mask, np.nan, wd_nan_removed)
# fig, ax = plt.subplots()
# triang = tri.Triangulation(xxx, yyy)
# pcm = ax.tricontourf(triang, wd_nan_removed, levels=np.linspace(4, 6, 100), cmap=colormaps, extend='both')


# colorbar = 



fig, ax = plt.subplots(figsize = (3,8),dpi = 300)
pcm = plt.tricontourf(xxx, yyy,
    wd_nan_removed,levels= np.linspace( 4, 6, 100),
    cmap=colormaps,extend='both' )#, transform=ccrs.PlateCarree())
# cbar = plt.colorbar(pcm)
# loc = np.linspace( 4,6,6 )
# cbar.set_ticks(loc)
# cbar.set_label('Maximum Surface Height (m)', labelpad=5)
# UKWEST.plot(ax=ax, color="white") # plot on the masking colour
 # Iterate over the polygons in the GeoDataFrame and plot them
UKWEST.plot(ax=ax, facecolor='white', linewidth=1)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-3.65,-2.75])
ax.set_ylim([53.20,54.52])
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
plt.tight_layout()
fig.patch.set_alpha(0)

plt.savefig('/Users/af/Desktop/max_surface_height.png',  transparent=True, bbox_inches='tight', pad_inches =0)

from PIL import Image

def remove_white_background(image_path, output_path, tolerance=200):
    """
    Remove white or near-white background from an image.
    
    Parameters:
    - image_path: path to the input image.
    - output_path: path where the output image will be saved.
    - tolerance: defines what range of white to consider (0-255). 
                 Lower tolerance means more strict white detection.
    """
    # Load image
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        # Change all white (and near-white) pixels to transparent
        if item[0] > tolerance and item[1] > tolerance and item[2] > tolerance:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(output_path, "PNG")

# Example usage
remove_white_background('/Users/af/Desktop/max_surface_height.png', '/Users/af/Desktop/max_surface_height_no_white.png')