#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:15:12 2025

@author: af
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

# ─── 0. Choose ───────────────────────────────────────────────
estuary = 'leven'
timestep = 30

# ─── 1. Load the NetCDF Dataset ───────────────────────────────────────────────
ds = xr.open_dataset(
    "/media/af/Elements1/3d_big_models/scw_run_3d_5layer_climatology_85-59442365/output/kent_31_merged_map.nc",
    chunks={"time": 100}
)

# ─── 2. Load Mersey Transect Points ────────────────────────────────────────────
csv_path = "/media/af/PNC/modelling_DATA/kent_estuary_project/land_boundary/analysis/QGIS_shapefiles/points_along_estuary_1km_spacing.csv"
transect_points = pd.read_csv(csv_path)
transect_points = transect_points[transect_points["est_name"] == estuary]

# ─── 3. Match Points to Mesh Faces Using BallTree ──────────────────────────────
x = ds["mesh2d_face_x"].values
y = ds["mesh2d_face_y"].values
coords = np.vstack((x, y)).T

tree = BallTree(coords, leaf_size=16)
transect_xy = transect_points[["X", "Y"]].to_numpy()
_, indices = tree.query(transect_xy, k=1)
face_indices = indices.flatten()

# ─── 4. Choose Time Step and Extract Variables ─────────────────────────────────
sal = ds["mesh2d_sa1"].isel(time=timestep, mesh2d_nFaces=face_indices).load()
depths_now = ds["mesh2d_waterdepth"].isel(time=timestep, mesh2d_nFaces=face_indices).load()
sigma = ds["mesh2d_layer_sigma"].values  # shape: (nLayers,)

# ─── 5. Compute Actual Depths per Sigma Layer ──────────────────────────────────
# Flip to top-to-bottom order if needed
if sigma[0] < sigma[-1]:
    sigma = sigma[::-1]
    sal = sal[:, ::-1]

# True depths: outer product of water depth and sigma coordinates
z_transect = -np.multiply.outer(depths_now.values, sigma)  # shape: (nPoints, nLayers)

# ─── 6. Plot Salinity Transect ─────────────────────────────────────────────────
distance = transect_points["distance"].values
plot_time = pd.to_datetime(ds.time.values[timestep])

plt.figure(figsize=(12, 6))
ax = plt.gca()

# Set the background of the plot area to grey
ax.set_facecolor("lightgrey")

# Create 2D grids for X, Y, and salinity
X = np.tile(distance, (len(sigma), 1))
Y = z_transect.T
C = sal.T

# Plot the salinity
pc = ax.pcolormesh(X, Y, C, shading="auto", cmap="viridis")

# Add colorbar
plt.colorbar(pc, ax=ax, label="Salinity (PSU)")

# Labels and formatting
ax.invert_yaxis()
ax.set_xlabel("Distance along estuary (m)")
ax.set_ylabel("Depth (m)")
ax.set_title(f"{estuary.capitalize()} Estuary Salinity Transect (Time step {timestep}: {plot_time:%Y-%m-%d %H:%M})")
plt.tight_layout()
plt.show()

#%% Plot on the bathymetry
# Load coordinates and z-values (will be lazy-loaded)
x = ds["mesh2d_node_x"].values
y = ds["mesh2d_node_y"].values
z = ds["mesh2d_node_z"].values  # elevation in meters

# Plot as colored scatter
plt.figure(figsize=(10, 8))
sc = plt.scatter(x, y, c=z, cmap="terrain", s=10, marker='.', edgecolors='none')
plt.colorbar(sc, label="Node Elevation (m)")
plt.xlabel("X (degrees east)")
plt.ylabel("Y (degrees north)")
plt.title("Mesh Node Elevation (mesh2d_node_z)")
plt.axis("equal")
plt.tight_layout()
plt.show()