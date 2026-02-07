# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:00:15 2023

@author: aafur
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import xarray as xr
from scipy.interpolate import griddata
from o_func import opsys
from pathlib import Path
start_path = Path(opsys('PNC'))


# path =Path(start_path) /  'modelling_DATA/kent_estuary_project/6.Final2/models/01_kent_2.0.0_no_wind/2.0.0_wind_testing_4_months.dsproj_data/bed_level_deepened_channel(testing).xyz'
main_path = start_path / 'modelling_DATA/kent_estuary_project/5.Final/1.friction/2.0.1_wind_testing_4_months_5_second_timestep.dsproj_data'

path =main_path / 'bed_level_deepened_channel(testing).xyz'

net_path = next((main_path / 'FlowFM').glob('*.nc'), None)

figure_path = start_path / 'modelling_DATA/kent_estuary_project/bathymetry/figures'

data = []

# Open the file and read line by line
with open(path, 'r') as file:
    for line in file:
        # Split each line into three values and convert them to float
        values = line.split()
        values = [float(val) for val in values]
        data.append(values)
data_array = np.array(data)

column1 = data_array[:, 0]
column2 = data_array[:, 1]
column3 = data_array[:, 2]


# Set the desired color range limits
color_min = -40
color_max = 20

# Create a colormap that maps values between color_min and color_max
cmap = plt.cm.get_cmap('viridis')
norm = plt.Normalize(vmin=color_min, vmax=color_max)

fig, ax = plt.subplots()
plt.scatter(column1, column2, c = column3, s = 1,cmap=cmap, norm=norm)


# Now to do the interpolated bathymetry

bathy_path = net_path
bd = xr.open_dataset(bathy_path, engine='scipy')
# Bathymetry data at node coordinates
node_x = bd.mesh2d_node_z.mesh2d_node_x.values
node_y = bd.mesh2d_node_z.mesh2d_node_y.values
bathymetry_node = bd.mesh2d_node_z.values

# Face coordinates
face_x = bd.mesh2d_face_x.values
face_y = bd.mesh2d_face_y.values

# Reshape face coordinates for griddata input
face_coords = np.column_stack((face_x, face_y))

# Interpolate bathymetry data to face coordinates
bathymetry_face = griddata((node_x, node_y), bathymetry_node, face_coords, method='linear')

fig2, ax2 = plt.subplots()
plt.scatter(face_x, face_y, c = bathymetry_face, s = 1,cmap=cmap, norm=norm)

#%% 

import xugrid as xu
import matplotlib.pyplot as plt
import numpy as np

# Open dataset (it automatically parses UGRID metadata)
uds = xu.open_dataset(bathy_path)
elev = uds["mesh2d_node_z"]


# Remove a bad piece of data 
# Your estimated bad location
target_x, target_y = -2.758, 53.34004

# Get node coordinates
x = elev["mesh2d_node_x"].values
y = elev["mesh2d_node_y"].values

# Find nearest node
dist = np.sqrt((x - target_x)**2 + (y - target_y)**2)
bad_node_idx = np.argmin(dist)
print(f"Bad node index: {bad_node_idx}")

# Went it to identify if it is a bad node, went along from bad point and found the bad point here 
bad_node_idx = 35278
good_node_idx = 35275
elev.values[35278] = elev.values[35275]

fig, ax = plt.subplots(figsize=(5, 7))# Inspect what fields are available
# elev.attrs["long_name"] = "Bathymetry"

# Plot bathymetry (typically 'mesh2d_node_z' or similar)
plot = elev.ugrid.plot(ax=ax, cmap="viridis", vmin=-40, vmax=20, )

ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
ax.set_xlim([-3.61, -2.75])
plt.tight_layout()
plt.savefig(figure_path / 'bathymetry_plot.png', dpi = 300)
# plt.close()
#'https://deltares.github.io/xugrid/examples/overlap_regridder.html'


#%%


# Plot only the mesh structure (no data shading, just grid lines)
fig, ax = plt.subplots(figsize=(10, 8))
elev.ugrid.plot.line()
ax.set_aspect("equal")
plt.title("Grid Structure (edges only)")
plt.show()

#%% 

'''
THIS CAN FIND BREAKS IN THE BOUNDARIES OF THE CELLS. 

'''
# ==== Brand-new bathy plot + RED outline of valid cells (no GeoPandas needed) ====

import numpy as np
import xugrid as xu
import matplotlib.pyplot as plt

def _np(a):
    """Return as NumPy array regardless of xarray/xugrid version."""
    return a.values if hasattr(a, "values") else np.asarray(a)

def _grid(ds):
    """Get first UGRID grid, old (.grid) or new (.grids[0]) API."""
    ugc = ds.ugrid
    if hasattr(ugc, "grid"):
        return ugc.grid
    if hasattr(ugc, "grids") and len(ugc.grids) > 0:
        return ugc.grids[0]
    raise RuntimeError("No UGRID grid found (.grid/.grids).")

def _face_nodes_and_fill(ug):
    fna = ug.face_node_connectivity
    arr = _np(fna)
    fill = getattr(fna, "fill_value", None)
    if fill is None:
        fill = getattr(fna, "_FillValue", None)
    if fill is None and hasattr(fna, "encoding"):
        fill = fna.encoding.get("_FillValue", None)
    return arr, (-1 if fill is None else int(fill))

def _faces_list(face_nodes, fill_value):
    # list of 1D arrays with the node ids for each face (unpadded)
    return [row[row != fill_value] for row in face_nodes]

def _boundary_segments(node_xy, faces_list, include_face):
    """Return list of boundary segments (x1,y1,x2,y2) for edges touching exactly one included face."""
    # undirected edge count
    edge_counts = {}
    edges_dir = []
    for f_idx, nodes in enumerate(faces_list):
        if not include_face[f_idx] or len(nodes) < 3:
            continue
        m = len(nodes)
        for k in range(m):
            u = int(nodes[k]); v = int(nodes[(k + 1) % m])
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edges_dir.append((u, v))

    segs = []
    for (u, v) in edges_dir:
        key = (u, v) if u < v else (v, u)
        if edge_counts.get(key, 0) == 1:  # boundary
            x1, y1 = node_xy[u]
            x2, y2 = node_xy[v]
            segs.append((x1, y1, x2, y2))
    return segs

# Open (again) to be safe/clean
uds_outline = xu.open_dataset(bathy_path)
ug = _grid(uds_outline)

# Node coordinates (NumPy)
node_xy = np.column_stack([_np(ug.node_x), _np(ug.node_y)])

# Faces + fill
face_nodes, fill_value = _face_nodes_and_fill(ug)
faces_list = _faces_list(face_nodes, fill_value)

# Valid-face mask: keep faces whose ALL nodes have finite mesh2d_node_z
bathy_da = uds_outline["mesh2d_node_z"]
# If there are extra dims (e.g., time), take the first step for masking
if bathy_da.ndim > 1:
    sel = {d: 0 for d in bathy_da.dims if d != ug.node_dimension}
    bathy_da = bathy_da.isel(**sel)
zn = _np(bathy_da)
include_valid = np.array([np.isfinite(zn[f]).all() for f in faces_list], dtype=bool)

# Build boundary segments for the valid region
segments = _boundary_segments(node_xy, faces_list, include_valid)
print(f"[outline] faces total={len(faces_list)}, valid={include_valid.sum()}, boundary segs={len(segments)}")

# ---- BRAND-NEW PLOT ----
fig, ax = plt.subplots(figsize=(5.5, 7.5))
uds_outline["mesh2d_node_z"].ugrid.plot(ax=ax, cmap="viridis", vmin=-40, vmax=20)

# Draw the red outline (cell-edge exact)
for (x1, y1, x2, y2) in segments:
    ax.plot([x1, x2], [y1, y2], color="red", linewidth=1.8, zorder=10)

ax.set_title("Bathymetry with RED valid-data outline")
ax.set_aspect("equal")  # important for geometry
plt.tight_layout()
plt.show()

#%% 
# --- Build ONE clean outer boundary from your segments and plot it on a new figure ---
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge, unary_union, polygonize
import numpy as np
import xugrid as xu
import matplotlib.pyplot as plt

# 1) Convert segments -> shapely LineStrings (cast np.float64 -> float)
lines = [LineString([(float(x1), float(y1)), (float(x2), float(y2))])
         for (x1, y1, x2, y2) in segments]

# 2) Merge lines; polygonize into candidate polygons
merged = unary_union(lines)              # MultiLineString / LineString
merged = linemerge(merged)               # connect where possible
polys = list(polygonize(merged))         # list of Polygon(s)

# 3) If polygonization yields nothing (tiny gaps), gently close with a small buffer
if not polys:
    # domain-based epsilon; increase x10 if still empty
    xs = [float(x1) for (x1, _, x2, _) in segments] + [float(x2) for (_, _, x2, _) in segments]
    ys = [float(y1) for (_, y1, _, y2) in segments] + [float(y2) for (_, _, _, y2) in segments]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    eps = 1e-4 * max(dx, dy)            # try 1e-5..1e-3 if needed
    filled = unary_union(lines).buffer(eps)   # close hairline gaps
    if isinstance(filled, Polygon):
        polys = [filled]
    elif isinstance(filled, MultiPolygon):
        polys = list(filled.geoms)

# 4) Choose the single largest polygon (outer boundary)
outer = max(polys, key=lambda p: p.area) if polys else None

print(f"[outline] segments={len(segments)} | candidates={len(polys)} | "
      f"outer_area={outer.area if outer else 'None'}")

# 5) New, clean plot: bathy + ONLY the outer ring
uds_plot = xu.open_dataset(bathy_path)
fig, ax = plt.subplots(figsize=(5.8, 7.8))
uds_plot["mesh2d_node_z"].ugrid.plot(ax=ax, cmap="viridis", vmin=-40, vmax=20)

if outer is not None:
    outer = outer.buffer(0)  # topological clean
    xx, yy = outer.exterior.xy
    ax.plot(xx, yy, color="red", linewidth=2.2, zorder=10)
else:
    # fallback: show raw segments so you still see something
    for (x1, y1, x2, y2) in segments:
        ax.plot([float(x1), float(x2)], [float(y1), float(y2)],
                color="red", linewidth=1.6, zorder=10)

ax.set_aspect("equal")
ax.set_title("Bathymetry with single largest outer outline")
plt.tight_layout()
plt.show()
