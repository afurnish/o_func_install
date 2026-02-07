#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-size heatmap that treats square cells appropriately.

Default metric: equivalent square edge length L_eq = sqrt(area) [m],
so a 1500 m square cell appears ~1500 regardless of number of sides.

Switch METRIC to:
  - "equiv_area"              -> sqrt(area) [m] (recommended; shape-independent)
  - "mean_edge"               -> mean edge length [m]
  - "quad_median_else_mean"   -> quads: median edge; others: mean edge
"""

from o_func import opsys
from pathlib import Path
import numpy as np
import xugrid as xu
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- user paths --------------------
start_path  = Path(opsys('PNC'))
main_path   = start_path / 'modelling_DATA/kent_estuary_project/5.Final/1.friction/2.0.1_wind_testing_4_months_5_second_timestep.dsproj_data'
figure_path = start_path / 'modelling_DATA/kent_estuary_project/bathymetry/figures'
bathy_path  = next((main_path / 'FlowFM').glob('*.nc'), None)

# -------------------- config ------------------------
METRIC = "equiv_area"   # "equiv_area" | "mean_edge" | "quad_median_else_mean"
FIXED_VMIN = None       # e.g., 20   (None = auto by percentiles)
FIXED_VMAX = None       # e.g., 1500 (None = auto by percentiles)
CBAR_WIDTH = "5%"       # colorbar width relative to axes
CBAR_PAD   = 0.1        # gap between plot and colorbar (in inches-ish)
FIGSIZE    = (7, 14)
# Optional view limits (comment out / set to None for autoscale)
X_LIMITS = None         # e.g., (-3.61, -2.75)
Y_LIMITS = None

# ====================================================
# Load grid
uds  = xu.open_dataset(bathy_path)
grid = uds.ugrid.grid
Path(figure_path).mkdir(parents=True, exist_ok=True)

# Node coordinates
x = np.asarray(grid.node_x, dtype=float)
y = np.asarray(grid.node_y, dtype=float)

# Face-node connectivity (ragged, -1 padded)
fnc = np.asarray(grid.face_node_connectivity)
if fnc.shape[0] != grid.n_face:
    fnc = fnc.T
n_faces = grid.n_face

# Detect lon/lat
is_lonlat = (np.nanmin(x) >= -360 and np.nanmax(x) <= 360) and (np.nanmin(y) >= -90 and np.nanmax(y) <= 90)

# ---- project coordinates to metres (once) ----
# Local equirectangular about domain median latitude for stable areas/lengths
if is_lonlat:
    lat0 = float(np.nanmedian(y))
    cos0 = np.cos(np.deg2rad(lat0))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * cos0
    x_m = x * m_per_deg_lon
    y_m = y * m_per_deg_lat
else:
    x_m = x
    y_m = y

# ---- helpers on projected coords ----
def edge_lengths_for_face(node_idx: np.ndarray) -> np.ndarray:
    xv = x_m[node_idx]; yv = y_m[node_idx]
    if xv.size < 3:
        return np.array([], dtype=float)
    xv2 = np.concatenate([xv, xv[:1]])
    yv2 = np.concatenate([yv, yv[:1]])
    dx  = np.diff(xv2); dy = np.diff(yv2)
    return np.sqrt(dx*dx + dy*dy)

def polygon_area_m2(node_idx: np.ndarray) -> float:
    # Shoelace formula on projected metres
    xv = x_m[node_idx]; yv = y_m[node_idx]
    if xv.size < 3:
        return np.nan
    return 0.5 * np.abs(np.dot(xv, np.roll(yv, -1)) - np.dot(yv, np.roll(xv, -1)))

def characteristic_length(node_idx: np.ndarray) -> float:
    if METRIC == "equiv_area":
        A = polygon_area_m2(node_idx)
        return np.sqrt(A) if np.isfinite(A) else np.nan
    elif METRIC == "mean_edge":
        el = edge_lengths_for_face(node_idx)
        return float(np.mean(el)) if el.size else np.nan
    elif METRIC == "quad_median_else_mean":
        el = edge_lengths_for_face(node_idx)
        if el.size == 0:
            return np.nan
        return float(np.median(el)) if node_idx.size == 4 else float(np.mean(el))
    else:
        raise ValueError(f"Unknown METRIC {METRIC!r}")

# ---- build polygons + values ----
polys = []
vals  = np.full(n_faces, np.nan, dtype=float)

for i in range(n_faces):
    face_nodes = fnc[i]
    face_nodes = face_nodes[face_nodes >= 0].astype(int)
    if face_nodes.size >= 3:
        # polygon in original coords for axis labels (lon/lat if present)
        poly = np.column_stack([x[face_nodes], y[face_nodes]])
        polys.append(poly)
        vals[i] = characteristic_length(face_nodes)
    else:
        polys.append(None)

# filter invalid
polys_clean, vals_clean = [], []
for p, v in zip(polys, vals):
    if p is not None and np.isfinite(v):
        polys_clean.append(p)
        vals_clean.append(v)
vals_clean = np.asarray(vals_clean, dtype=float)

# colour limits
if FIXED_VMIN is None or FIXED_VMAX is None:
    vmin = 20
    vmax = 1500
else:
    vmin, vmax = float(FIXED_VMIN), float(FIXED_VMAX)

# ---- plot ----
fig, ax = plt.subplots(figsize=FIGSIZE)
pc = PolyCollection(polys_clean, array=vals_clean, cmap="jet",
                    clim=(vmin, vmax), linewidths=0)
ax.add_collection(pc)
ax.autoscale()

if X_LIMITS is not None:
    ax.set_xlim(X_LIMITS)
if Y_LIMITS is not None:
    ax.set_ylim(Y_LIMITS)

ax.set_aspect("equal")
ax.set_xlabel("Longitude" if is_lonlat else "X [m]")
ax.set_ylabel("Latitude"  if is_lonlat else "Y [m]")

# Colorbar: same height as plot, clean layout, no title
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size=CBAR_WIDTH, pad=CBAR_PAD)
cbar = plt.colorbar(pc, cax=cax)
label = {
    "equiv_area": "Equivalent Edge Length [m]",
    "mean_edge": "Mean edge length [m]",
    "quad_median_else_mean": "Edge length [m] (quads: median; others: mean)",
}[METRIC]
cbar.set_label(label)

# No title -> keep figure clean
# plt.tight_layout()
outname = {
    "equiv_area": "cell_size_heatmap_equiv_length.png",
    "mean_edge": "cell_size_heatmap_mean_edge.png",
    "quad_median_else_mean": "cell_size_heatmap_quadmedian_mean.png",
}[METRIC]
plt.savefig(figure_path / outname, dpi=600, bbox_inches="tight")
