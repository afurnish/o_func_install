# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Fri Sep  6 15:38:55 2024

# @author: af
# """
# import xarray as xr
# import xesmf as xe
# from pathlib import Path
# from o_func import opsys; start_path = Path(opsys())

# import os
# # Set environment variables for detailed logging
# os.environ['ESMF_LOGLEVEL'] = 'TRACE'
# os.environ['ESMF_LOGFILE'] = './esmf_test_log.log'

# # Paths to the model datasets
# delft_grid = start_path / Path('modelling_DATA/kent_estuary_project/6.Final2/models/kent_1.0.0_UM_wind/kent_31_merged_map.nc')
# mo_grid = start_path / Path('Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_T.nc')



# # Load the Delft3D FM dataset (unstructured grid)
# delft_ds = xr.open_dataset(delft_grid, chunks={'nmesh2d_face': 5000})
# #delft_ds = delft_ds.rename({'mesh2d_face_x': 'lon', 'mesh2d_face_y': 'lat'})

# # Load the Met Office dataset (structured grid)
# mo_ds = xr.open_dataset(mo_grid, chunks={'y': 50, 'x': 50})
# #mo_ds = mo_ds.rename({'nav_lon': 'lon', 'nav_lat': 'lat'})

# # Extract the surface height variable
# delft_surf_height = delft_ds['mesh2d_s1']  # Assuming 'mesh2d_s1' is surface height
# mo_surf_height = mo_ds['sossheig']  # Met Office surface height

# # Prepare source and target grids for regridding
# # Delft3D (unstructured grid)
# delft_lat = delft_ds['mesh2d_face_y']  # Assuming 'mesh2d_node_y' is latitude
# delft_lon = delft_ds['mesh2d_face_x']  # Assuming 'mesh2d_node_x' is longitude

# # Met Office grid (structured grid)
# mo_lat = mo_ds['nav_lat']
# mo_lon = mo_ds['nav_lon']

# delft_ds_small = delft_ds.isel(time=slice(0, 10), nmesh2d_face=slice(0, 50))
# mo_ds_small = mo_ds.isel(time_counter=slice(0, 10), y=slice(0, 50), x=slice(0, 50))
# delft_ds_small = delft_ds_small.rename({'mesh2d_face_x': 'lon', 'mesh2d_face_y': 'lat'})
# mo_ds_small = mo_ds_small.rename({'nav_lon': 'lon', 'nav_lat': 'lat'})

# delft_ds = delft_ds.rename({'mesh2d_face_x': 'lon', 'mesh2d_face_y': 'lat'})
# mo_ds = mo_ds.rename({'nav_lon': 'lon', 'nav_lat': 'lat'})
# # Create the regridder using xESMF
# delft_grid = xr.Dataset({'lon': delft_lon, 'lat': delft_lat}).compute()
# mo_grid = xr.Dataset({'lon': mo_lon, 'lat': mo_lat}).compute()


# # regridder = xe.Regridder(delft_grid, mo_grid, method='bilinear')


# # Save the regridded result to a new NetCDF file
# # regridder.to_netcdf('regrid_weights.nc')
# # print("Regridding complete. The result is saved to 'regridded_surface_height.nc'.")

# import numpy as np
# import esmpy
# # Initialize ESMF Manager
# esmpy.Manager(debug=True)

# # Define source grid (Delft3D) with latitude and longitude
# source_lon = np.array(delft_grid['lon'])
# source_lat = np.array(delft_grid['lat'])

# # Define destination grid (Met Office) with latitude and longitude
# dest_lon = np.array(mo_grid['lon'])
# dest_lat = np.array(mo_grid['lat'])

# # Define the grid shape
# nlon_dest = dest_lon.shape[1]  # Number of longitudes
# nlat_dest = dest_lat.shape[0]  # Number of latitudes

# # Create the destination grid (structured grid)
# dest_grid = esmpy.Grid(np.array([nlat_dest, nlon_dest]), coord_sys=esmpy.CoordSys.SPH_DEG)

# # Add the coordinates to the destination grid
# # The staggerloc = ESMF.StaggerLoc.CENTER_VCENTER corresponds to grid centers (non-staggered)

# # Add longitude and latitude coordinates to the grid
# dest_grid.add_coords(staggerloc=esmpy.StaggerLoc.CENTER)

# # Get the pointers to the longitude and latitude grid arrays
# dest_lon_grid = dest_grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)  # Longitude axis (x)
# dest_lat_grid = dest_grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)  # Latitude axis (y)

# # Assign the longitude and latitude data to the grid
# dest_lon_grid[...] = dest_lon
# dest_lat_grid[...] = dest_lat

# # Now the destination grid has been set up with latitude and longitude coordinates
# print("Destination grid created successfully with lon and lat coordinates.")

# source_mesh = esmpy.Mesh(parametric_dim=2, spatial_dim=2)
# # Add the nodes (longitude and latitude) to the mesh
# source_lon = np.array(delft_grid['lon'])
# source_lat = np.array(delft_grid['lat'])
# node_coords = np.array([source_lon, source_lat]).T

# # Add node IDs (1-based indexing for ESMF)
# node_ids = np.arange(1, len(source_lon) + 1)

# face_nodes = delft_ds.mesh2d_face_nodes.values

# # Create the connectivity array for ESMPy
# connectivity_list = []

# # Loop through each face and handle triangles/squares
# for face in face_nodes:
#     # If it's a triangle, pad with -1 for the 4th vertex
#     if np.isnan(face[3]):
#         connectivity_list.append(np.append(face[:3].astype(int) - 1, -1))  # Add -1 for missing 4th vertex
#     else:
#         connectivity_list.append(face[:4].astype(int) - 1)  # Convert to 0-based indexing for squares

# # Convert the list to a numpy array
# connectivity = np.vstack(connectivity_list)

# # Set the node count and node owners (assuming single-process ownership)
# node_count = len(node_coords)
# node_owners = np.zeros(node_count, dtype=int)  # All nodes owned by process 0

# # Add nodes to the source mesh
# source_mesh.add_nodes(node_count=node_count, node_coords=node_coords, node_ids=node_ids, node_owners=node_owners)
# element_conn = connectivity.flatten()
# # Set element types (3 for triangles, 4 for quads)
# element_types = np.full(len(connectivity), 3)  # Default to triangles
# element_types[np.isfinite(face_nodes[:, 3])] = 4  # Set to quads where the 4th node is not NaN

# # Set element IDs (1-based indexing)
# element_ids = np.arange(1, len(connectivity) + 1)

# # Set element count
# element_count = len(connectivity)

# # Add the elements to the mesh
# source_mesh.add_elements(element_count=element_count, element_conn=element_conn, element_types=element_types, element_ids=element_ids)

# print("Elements added to the source mesh.")

# # Create a simple grid and regrid example
# try:
#     grid = esmpy.Grid(np.array([10, 10]), coord_sys=esmpy.CoordSys.SPH_DEG)
#     print("Grid created successfully")
# except Exception as e:
#     print(f"Error: {e}")

import xarray as xr
import xesmf as xe
from pathlib import Path
import numpy as np
import esmpy
import os

# Set environment variables for detailed logging
os.environ['ESMF_LOGLEVEL'] = 'TRACE'
os.environ['ESMF_LOGFILE'] = './esmf_test_log.log'

# Paths to the model datasets
from o_func import opsys
start_path = Path(opsys())
delft_grid = start_path / Path('modelling_DATA/kent_estuary_project/6.Final2/models/kent_1.0.0_UM_wind/kent_31_merged_map.nc')
mo_grid = start_path / Path('Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_T.nc')

# Load datasets
delft_ds = xr.open_dataset(delft_grid, chunks={'nmesh2d_face': 5000})
mo_ds = xr.open_dataset(mo_grid, chunks={'y': 50, 'x': 50})

# Rename coordinates for easier use
delft_ds = delft_ds.rename({'mesh2d_face_x': 'lon', 'mesh2d_face_y': 'lat'})
mo_ds = mo_ds.rename({'nav_lon': 'lon', 'nav_lat': 'lat'})

# Extract the lat/lon
source_lon = np.array(delft_ds['lon'])
source_lat = np.array(delft_ds['lat'])
dest_lon = np.array(mo_ds['lon'])
dest_lat = np.array(mo_ds['lat'])

# Define the grid shape
nlon_dest = dest_lon.shape[1]  # Number of longitudes
nlat_dest = dest_lat.shape[0]  # Number of latitudes

# Initialize ESMF Manager
esmpy.Manager(debug=True)

# Create destination grid for Met Office structured data
dest_grid = esmpy.Grid(np.array([nlat_dest, nlon_dest]), coord_sys=esmpy.CoordSys.SPH_DEG)

# Add coordinates to the destination grid
dest_grid.add_coords(staggerloc=esmpy.StaggerLoc.CENTER)

# Assign the coordinates
dest_lon_grid = dest_grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)  # Longitude
dest_lat_grid = dest_grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)  # Latitude
dest_lon_grid[...] = dest_lon
dest_lat_grid[...] = dest_lat

# Create source mesh (Delft3D unstructured grid)
source_mesh = esmpy.Mesh(parametric_dim=2, spatial_dim=2)

# Get node coordinates (longitude and latitude)
node_coords = np.array([source_lon, source_lat]).T
node_ids = np.arange(1, len(source_lon) + 1)  # 1-based indexing for ESMF

# Face node connectivity (convert NaNs to -1 for triangular faces)
face_nodes = delft_ds.mesh2d_face_nodes.values
connectivity_list = []

for face in face_nodes:
    if np.isnan(face[3]):  # If triangular, fill with -1
        connectivity_list.append(np.append(face[:3].astype(int) - 1, -1))  # 0-based indexing
    else:
        connectivity_list.append(face[:4].astype(int) - 1)

connectivity = np.vstack(connectivity_list)

# Set element types (3 for triangles, 4 for quads)
element_types = np.full(len(connectivity), 3)  # Default to triangles
element_types[np.isfinite(face_nodes[:, 3])] = 4  # Quads where 4th node is not NaN

# Element count and IDs (1-based indexing)
element_count = len(connectivity)
element_ids = np.arange(1, element_count + 1)

# Add nodes and elements to the source mesh
node_count = len(node_coords)
node_owners = np.zeros(node_count, dtype=int)  # Owned by process 0

source_mesh.add_nodes(node_count=node_count, node_coords=node_coords, node_ids=node_ids, node_owners=node_owners)
source_mesh.add_elements(element_count=element_count, element_conn=connectivity.flatten(), element_types=element_types, element_ids=element_ids)

# At this point, the source mesh and destination grid have been created.
# Now, the next step is creating the regrid object, but I’ve left this up to you in case you'd like to use xESMF or a direct ESMF regrid method.

print("Source mesh and destination grid created successfully.")

