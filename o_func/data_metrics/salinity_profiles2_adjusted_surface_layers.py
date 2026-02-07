import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

# ─── CONFIG ─────────────────────────────────────────────────────────────
estuary_name = "kent"
timestep = 30
csv_path = "/media/af/PNC/modelling_DATA/kent_estuary_project/land_boundary/analysis/QGIS_shapefiles/points_along_estuary_1km_spacing.csv"
nc_path = "/media/af/Elements1/3d_big_models/scw_run_3d_5layer_climatology_85-59442365/output/kent_31_merged_map.nc"

# ─── LOAD DATA ──────────────────────────────────────────────────────────
ds = xr.open_dataset(nc_path, chunks={"time": 100})
transect = pd.read_csv(csv_path)
transect = transect[transect["est_name"] == estuary_name]

# ─── FIND CLOSEST FACES ─────────────────────────────────────────────────
face_coords = np.stack((ds["mesh2d_face_x"].values, ds["mesh2d_face_y"].values), axis=-1)
tree = BallTree(face_coords)
_, idx = tree.query(transect[["X", "Y"]].values, k=1)
face_idx = idx.flatten()

# ─── EXTRACT DATA ───────────────────────────────────────────────────────
surface = ds["mesh2d_s1"].isel(time=timestep, mesh2d_nFaces=face_idx).values
depth = ds["mesh2d_waterdepth"].isel(time=timestep, mesh2d_nFaces=face_idx).values
bed = surface - depth
salinity = ds["mesh2d_sa1"].isel(time=timestep, mesh2d_nFaces=face_idx).values  # (points, layers)

# ─── CALCULATE Z LEVELS FOR EVENLY SPACED 5-LAYER SIGMA SYSTEM ─────────
n_layers = 5
fractions = np.linspace(0.1, 0.9, n_layers)  # Midpoints of 5 even sigma layers
Z = np.zeros((len(face_idx), n_layers))
for i, frac in enumerate(fractions):
    Z[:, i] = bed + frac * depth

# ─── PLOT TRANSECT ──────────────────────────────────────────────────────
distance = transect["distance"].values
time_stamp = pd.to_datetime(ds.time.values[timestep])

plt.figure(figsize=(12, 6))
pc = plt.pcolormesh(distance, Z.T, salinity.T, shading='auto', cmap='viridis')
plt.xlabel("Distance along estuary (m)")
plt.ylabel("Elevation (m, relative to datum)")
plt.title(f"{estuary_name.title()} Estuary Salinity Transect\nTime step {timestep}: {time_stamp:%Y-%m-%d %H:%M}")
plt.colorbar(pc, label="Salinity (PSU)")
plt.grid(True)
plt.tight_layout()
plt.show()
