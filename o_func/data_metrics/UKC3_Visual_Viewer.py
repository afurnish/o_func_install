#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UKC3 Visual Viewer and Point Generator for PDF
Updated to remove external package dependency on o_functions
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import glob
import xarray as xr
from o_func import opsys

# ───── CONFIGURATION ───────────────────────────────────────────────

# Set base path
start_path = Path(opsys("PNC"))  # or use Path(opsys('PNC')) if you reinstate opsys

# Paths to shapefiles
kent_poly_path = start_path / "modelling_DATA/kent_estuary_project/5.Final/QGIS/kent_area_poly.shp"
outline_path = start_path / "modelling_DATA/kent_estuary_project/5.Final/QGIS/kent_area_poly_as_lines.shp"
ukc3_example_grid = start_path / "Original_Data/UKC3/oa/shelftmb/UKC4ao_1h_20131030_20131030_shelftmb_grid_T.nc"
figure_path = start_path / 'modelling_DATA/kent_estuary_project/grid/figures'
# ───── FUNCTIONS ──────────────────────────────────────────────────

def UKC3_area(first_file):
    """Load first UKC3 sea surface height file and return lon/lat/z"""
    ds = xr.open_dataset(first_file)
    z = ds["sossheig"].isel(time_counter=0).values
    x = ds["nav_lon"].values
    y = ds["nav_lat"].values
    return x, y, z

# ───── LOAD DATA ───────────────────────────────────────────────────

kent_poly = gpd.read_file(kent_poly_path)
outline = gpd.read_file(outline_path)
x, y, z = UKC3_area(ukc3_example_grid)

num_rows, num_cols = x.shape
masked_data = np.ma.masked_invalid(z * 0)
masked_data[np.isnan(z * 0)] = 1

longitude = x
latitude = y

# %%───── FIGURE 1: Domain Outline ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(30, 15))
outline.plot(ax=ax, color="red")
plt.contourf(x, y, z * 0)
spec_col = 758
plt.plot(x[:, spec_col], y[:, spec_col], c='white')

#%% ───── FIGURE 2: Grid with Data Mask ────────────────────────────────

fsize = 35
adjuster = 6 / 8

fig2, ax2 = plt.subplots(figsize=(30, 15))
plt.contourf(x, y, masked_data, cmap='twilight')

# Grid lines
for col in range(num_cols):
    plt.plot(longitude[:, col], latitude[:, col], color='black', linestyle='-', linewidth=0.1)
for row in range(num_rows):
    plt.plot(longitude[row, :], latitude[row, :], color='black', linestyle='-', linewidth=0.1)

outline.plot(ax=ax2, color="red", label='Area of Interest')
plt.xlabel('Longitude', fontsize=fsize)
plt.ylabel('Latitude', fontsize=fsize)
plt.xticks(size=fsize * adjuster)
plt.yticks(size=fsize * adjuster)
plt.legend(fontsize=fsize, loc='upper right')

# %%───── FIGURE 3: High-res Grid ──────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(50, 50))
fsize = 100
plt.contourf(x, y, masked_data, cmap='twilight')

for col in range(num_cols):
    plt.plot(longitude[:, col], latitude[:, col], color='black', linestyle='-', linewidth=0.1)
for row in range(num_rows):
    plt.plot(longitude[row, :], latitude[row, :], color='black', linestyle='-', linewidth=0.5)

plt.xlabel('Longitude', fontsize=fsize)
plt.ylabel('Latitude', fontsize=fsize)
plt.xticks(size=fsize * adjuster)
plt.yticks(size=fsize * adjuster)
ax3.set_xlim(-5, -2.5)
ax3.set_ylim(53, 54.75)

# %%───── OPTIONAL: Save or Show ───────────────────────────────────────

# plt.savefig("ukc3_grid_plot.png", dpi=300)
plt.show()

numeric_data = masked_data.astype(int)

#%%  Inset Map Plot
# extent = [-3.65, -2.75, 53.20, 54.52]
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# x0, x1 = -5, -2  
# y0, y1 = 53, 55 
x0, x1 = -3.7, -2.5
y0, y1 = 53.1, 54.6


# Truncate colormap (pale green to pale blue range)
terrain = plt.get_cmap('terrain')
custom_cmap = LinearSegmentedColormap.from_list(
    'trunc', terrain(np.linspace(0.15, 0.45, 256))
)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})

for ax in [ax1, ax2]:
    ax.pcolormesh(x, y, numeric_data, shading='auto', cmap=custom_cmap)

    
    if ax == ax1:
        lw = 0.2
    else:
        lw = 0.5
    for col in range(num_cols):
        ax.plot(longitude[:, col], latitude[:, col], color='black', linestyle='-', linewidth=lw, alpha=0.2,)
    for row in range(num_rows):
        ax.plot(longitude[row, :], latitude[row, :], color='black', linestyle='-', linewidth=lw, alpha=0.2,)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    if ax == ax2:
        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])
        outline.plot(ax=ax, color="red", label='Area of Interest')
        ax.legend(loc='upper right',framealpha=1, frameon=True)
        ax.set_title('B')
    else:
        

        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor='black', facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.set_title('A')

# Draw only top-right to top-left, and bottom-right to bottom-left
for (xyA, xyB) in [((x0, y1), (x1, y1)),  # top connection
                   ((x0, y0), (x1, y0))]:  # bottom connection
    con = ConnectionPatch(
        xyA=xyA, coordsA="data", axesA=ax2,
        xyB=xyB, coordsB="data", axesB=ax1,
        color="black", linewidth=1.5
    )
    fig.add_artist(con)
    

plt.savefig(figure_path / 'Area_of_interest_map_with_inset.png', dpi = 300)

#%% Inset only

# extent = [-3.65, -2.75, 53.20, 54.52]
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# x0, x1 = -5, -2  
# y0, y1 = 53, 55 
x0, x1 = -3.7, -2.5
y0, y1 = 53.1, 54.6

fig, ax = plt.subplots(figsize=(5, 7))


ax.pcolormesh(x, y, numeric_data, shading='auto', cmap=custom_cmap)
    
lw = 0.2
for col in range(num_cols):
    ax.plot(longitude[:, col], latitude[:, col], color='black', linestyle='-', linewidth=lw, alpha=0.5,)
for row in range(num_rows):
    ax.plot(longitude[row, :], latitude[row, :], color='black', linestyle='-', linewidth=lw, alpha=0.5,)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
    
ax.set_xlim([x0+0.1, x1])
ax.set_ylim([y0, y1])
outline.plot(ax=ax, color="red", label='Area of Interest', linewidth = 0.5)
ax.legend(loc='upper right',framealpha=1, frameon=True)
plt.tight_layout()

plt.savefig(figure_path / 'area_of_interest_inset_only.png', dpi = 300)


#%% Add on the river locations onto the UKC4 grid

import pickle
saved_river_dict = start_path / 'GitHub/o_func_install/o_func/data_metrics/river_gauges_save_data2.pkl'
with open(saved_river_dict, 'rb') as f:
    loaded_dict = pickle.load(f)
    
row_indices = loaded_dict['row_indicies']
col_indices = loaded_dict['col_indices']

lons = np.array(loaded_dict['lons'])
lats = np.array(loaded_dict['lats'])


# Boolean mask of points inside the bounding box
in_bounds = (lons >= x0) & (lons <= x1) & (lats >= y0) & (lats <= y1)
river_names = [
    'Dee', 'Clwyd', 'Mersey', 'Alt', 'Ribble',
    'Wyre', 'Lune', 'Leven', 'Kent', 'Esk'
]

# Filtered coordinates
filtered_lons = lons[in_bounds]
filtered_lats = lats[in_bounds]
    
fig, ax = plt.subplots(figsize=(5, 7))


ax.pcolormesh(x, y, numeric_data, shading='auto', cmap=custom_cmap)
# ax.scatter(filtered_lons, filtered_lats, marker='+', color='red', s =25)


for i, (lon, lat, name) in enumerate(zip(filtered_lons, filtered_lats, river_names)):
    label = 'AMM15 river climatology discharge' if i == 0 else None
    ax.plot(lon, lat, 'o', color='blue', markersize=5, label=label)
    ax.text(lon + 0.01, lat + 0.01, name, fontsize=9, color='blue', ha='center')


lw = 0.2
for col in range(num_cols):
    ax.plot(longitude[:, col], latitude[:, col], color='black', linestyle='-', linewidth=lw, alpha=0.5,)
for row in range(num_rows):
    ax.plot(longitude[row, :], latitude[row, :], color='black', linestyle='-', linewidth=lw, alpha=0.5,)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
    
ax.set_xlim([x0+0.1, x1-0.3])
ax.set_ylim([y0, y1])
# outline.plot(ax=ax, color="red", label='Area of Interest', linewidth = 0.5)
# ax.legend(loc='upper right',framealpha=1, frameon=True)
plt.tight_layout()
plt.legend(loc = 'upper right')
plt.savefig(figure_path / 'area_of_interest_inset_only_with_climatology_discharge.png', dpi = 300)