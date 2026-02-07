import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import netCDF4 as nc

# === File path ===
ncfile = '/Volumes/Elements/backup_scw/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/output/kent_31_merged_map.nc'

# === Load the NetCDF ===
ds = nc.Dataset(ncfile)

#%%
# === Node coordinates ===
x = ds.variables['mesh2d_node_x'][:]
y = ds.variables['mesh2d_node_y'][:]

# === Face-node connectivity (1-based index) ===
faces = ds.variables['mesh2d_face_nodes'][:, :]
faces = np.where(faces == -999, -1, faces - 1)  # make 0-based, mask invalids

# === Face-centered field to color by ===
s1 = ds.variables['mesh2d_s1'][-1, :]  # time=0 surface height
s1 = np.ma.masked_equal(s1, -999)     # apply mask

# === Build face patches ===
patches = []
face_colors = []

for i, face_nodes in enumerate(faces):
    nodes = face_nodes[face_nodes >= 0]
    if len(nodes) < 3:
        continue  # skip degenerate faces
    coords = np.column_stack((x[nodes], y[nodes]))
    patches.append(Polygon(coords, closed=True))
    face_colors.append(s1[i])

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 10))
p = PatchCollection(patches, cmap='viridis', edgecolor='k', linewidth=0.1)
p.set_array(np.ma.masked_array(face_colors, mask=np.ma.getmask(s1)))
p.set_clim(vmin=np.nanmin(s1), vmax=np.nanmax(s1))

ax.add_collection(p)
ax.autoscale()
ax.set_aspect('equal')
plt.colorbar(p, ax=ax, label='Surface Height (m)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title('Delft3D FM Surface Height (mesh2d_s1 @ t=0)')
plt.tight_layout()
plt.show()

#%% 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import netCDF4 as nc

# === Load file ===
ncfile = '/Volumes/Elements/backup_scw/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/output/kent_31_merged_map.nc'
ds = nc.Dataset(ncfile)

# === Node coordinates ===
x = ds.variables['mesh2d_node_x'][:]
y = ds.variables['mesh2d_node_y'][:]

# === Face-node connectivity (1-based to 0-based) ===
faces = ds.variables['mesh2d_face_nodes'][:, :]
faces = np.where(faces == -999, -1, faces - 1)

# === Build face polygons ===
patches = []
for face_nodes in faces:
    nodes = face_nodes[face_nodes >= 0]
    if len(nodes) >= 3:
        coords = np.column_stack((x[nodes], y[nodes]))
        patches.append(Polygon(coords, closed=True))

# === Plot grid ===
fig, ax = plt.subplots(figsize=(5, 7))
p = PatchCollection(patches, facecolor='none', edgecolor='black', linewidth=0.2)
ax.add_collection(p)
ax.autoscale()
ax.set_aspect('equal')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.show()
plt.savefig('PRIMEA_domain_extent.png', dpi = 300)