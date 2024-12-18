#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:04:09 2024

@author: af
"""

import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Polygon

# Paths
base_path = "/media/af/Elements/Original_Data/transects"
grid_file = f"{base_path}/kent_grid.nc"
transect_files = {
    "DEE": f"{base_path}/DEE_transect.csv",
    "DUDDON": f"{base_path}/DUDDON_transect.csv",
    "KENT": f"{base_path}/KENT_transect.csv",
    "LEVEN": f"{base_path}/LEVEN_transect.csv",
    "LUNE": f"{base_path}/LUNE_transect.csv",
    "MERSEY": f"{base_path}/MERSEY_transect.csv",
    "RIBBLE": f"{base_path}/RIBBLE_transect.csv",
    "WYRE": f"{base_path}/WYRE_transect.csv",
}

# Load grid data
grid_data = xr.open_dataset(grid_file)

# Load transects
transect_data = {name: pd.read_csv(path) for name, path in transect_files.items()}

# Normalize face node indices to zero-based indexing
face_nodes = grid_data['mesh2d_face_nodes'].values - 1  # Convert 1-based to 0-based indexing
lon_nodes = grid_data['mesh2d_node_x'].values  # Longitude of nodes
lat_nodes = grid_data['mesh2d_node_y'].values  # Latitude of nodes

lon_faces = grid_data['mesh2d_face_x'].values
lat_faces = grid_data['mesh2d_face_y'].values

# Define a function to find nearest grid points for a given transect
def find_nearest_grid_points(transect_points, lon_faces, lat_faces):
    """
    Find the nearest grid points in the mesh for a given transect.
    
    Parameters:
        transect_points (np.ndarray): Array of shape (n, 2) with [lon, lat] for each transect point.
        lon_faces (np.ndarray): Array of longitudes for grid face centroids.
        lat_faces (np.ndarray): Array of latitudes for grid face centroids.
    
    Returns:
        dict: A dictionary containing:
            - indices: Indices of nearest grid points
            - distances: Distances from transect points to nearest grid points
    """
    # Build KDTree using face centroids
    face_centroids = np.vstack((lon_faces, lat_faces)).T
    tree = KDTree(face_centroids)
    
    # Query nearest faces for each transect point
    distances, indices = tree.query(transect_points)
    
    return {
        "indices": indices,
        "distances": distances
    }

#% Test map 

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_transect_with_grid_on_map(transect_points, nearest_indices, lon_faces, lat_faces, xlim=None, ylim=None, title="Transect Verification"):
    """
    Plot transect points and their corresponding nearest grid points on a map with a basic map outline.
    
    Parameters:
        transect_points (np.ndarray): Array of shape (n, 2) with [lon, lat] for transect points.
        nearest_indices (np.ndarray): Indices of nearest grid points in the grid.
        lon_faces (np.ndarray): Longitudes of grid face centroids.
        lat_faces (np.ndarray): Latitudes of grid face centroids.
        xlim (tuple): Longitude limits for the map (min_lon, max_lon).
        ylim (tuple): Latitude limits for the map (min_lat, max_lat).
        title (str): Title for the plot.
    """
    # Extract nearest grid point coordinates
    nearest_lon = lon_faces[nearest_indices]
    nearest_lat = lat_faces[nearest_indices]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.set_extent(xlim + ylim if xlim and ylim else None, crs=ccrs.PlateCarree())
    highres_coastline = cfeature.NaturalEarthFeature(
        category='physical', name='coastline', scale='10m', facecolor='none'
    )
    ax.add_feature(highres_coastline, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    
    # Plot transect points
    ax.scatter(transect_points[:, 0], transect_points[:, 1], c='blue', label="Transect Points", s=50, transform=ccrs.PlateCarree())
    
    # Plot nearest grid points
    ax.scatter(nearest_lon, nearest_lat, c='red', label="Nearest Grid Points", s=50, marker='x', transform=ccrs.PlateCarree())
    
    # Connect transect points to nearest grid points
    for i in range(len(transect_points)):
        ax.plot(
            [transect_points[i, 0], nearest_lon[i]],
            [transect_points[i, 1], nearest_lat[i]],
            c='gray', linestyle='--', linewidth=0.7, transform=ccrs.PlateCarree()
        )
    
    # Map settings
    ax.set_title(title)
    ax.legend(loc='lower left')
    plt.show()
    
def get_face_corner_indices(face_indices, face_nodes):
    """
    Retrieve the corner indices of faces based on face indices.
    
    Parameters:
        face_indices (np.ndarray): Array of indices for the faces.
        face_nodes (np.ndarray): 2D array of face-to-node connectivity.
                                 Each row represents a face, and columns represent its corner nodes.
    
    Returns:
        np.ndarray: 2D array of corner node indices for the specified faces.
    """
    # Use the provided face indices to extract rows from face_nodes
    corner_indices = face_nodes[face_indices]
    return corner_indices

def get_corner_coordinates(corner_indices, lon_nodes, lat_nodes):
    """
    Retrieve longitude and latitude coordinates for corner nodes of each cell.
    
    Parameters:
        corner_indices (np.ndarray): 2D array of corner node indices for each cell.
                                     Rows represent cells, columns represent corner nodes.
        lon_nodes (np.ndarray): Array of longitudes for all nodes.
        lat_nodes (np.ndarray): Array of latitudes for all nodes.
    
    Returns:
        list of dict: List of dictionaries, each containing:
                      - 'cell_index': Index of the cell
                      - 'lon_corners': List of longitudes for the corners
                      - 'lat_corners': List of latitudes for the corners
    """
    cells = []
    
    for i, corners in enumerate(corner_indices):
        # Mask NaN values
        valid_mask = ~np.isnan(corners)
        valid_corners = corners[valid_mask].astype(int)  # Convert to integer indices
        
        # Extract coordinates
        lon_corners = lon_nodes[valid_corners]
        lat_corners = lat_nodes[valid_corners]
        
        # Store results
        cells.append({
            'cell_index': i,
            'lon_corners': lon_corners.tolist(),
            'lat_corners': lat_corners.tolist()
        })
    
    return cells

from shapely.geometry import Polygon

def calculate_cell_areas(corner_coordinates):
    """
    Calculate the area of each cell based on corner coordinates.
    
    Parameters:
        corner_coordinates (list of dict): List of dictionaries, each containing:
                                           - 'cell_index': Index of the cell
                                           - 'lon_corners': List of longitudes for the corners
                                           - 'lat_corners': List of latitudes for the corners
    
    Returns:
        list of dict: List of dictionaries, each containing:
                      - 'cell_index': Index of the cell
                      - 'area': Calculated area of the cell in square meters
    """
    areas = []
    
    for cell in corner_coordinates:
        lons = cell['lon_corners']
        lats = cell['lat_corners']
        
        # Skip cells with fewer than 3 corners (invalid polygon)
        if len(lons) < 3 or len(lats) < 3:
            areas.append({'cell_index': cell['cell_index'], 'area': 0})
            continue
        
        # Create a Polygon with the corner coordinates
        polygon = Polygon(zip(lons, lats))
        
        # Calculate the area (result will be in degrees squared)
        area = polygon.area  # Approximation; adjust for desired units if needed
        
        # Store the result
        areas.append({'cell_index': cell['cell_index'], 'area': area})
    
    return areas

import numpy as np

def haversine_area(lons, lats, radius=6378137):
    """
    Calculate the spherical polygon area using the Haversine formula.
    
    Parameters:
        lons (list): List of longitudes in degrees for the polygon corners.
        lats (list): List of latitudes in degrees for the polygon corners.
        radius (float): Radius of the Earth in meters. Defaults to WGS84 value (6378137m).
    
    Returns:
        float: Area of the polygon in square meters.
    """
    if len(lons) < 3:
        return 0  # Not a valid polygon
    
    # Convert degrees to radians
    lons = np.radians(lons)
    lats = np.radians(lats)
    
    # Calculate spherical excess
    total_angle = 0
    for i in range(len(lons)):
        # Get the current and next points
        lon1, lat1 = lons[i - 1], lats[i - 1]  # Previous point
        lon2, lat2 = lons[i], lats[i]          # Current point
        
        # Compute the angle
        total_angle += np.arctan2(
            np.tan(lat2 / 2 + np.pi / 4) * np.tan(lon2 / 2 + np.pi / 4),
            np.tan(lat1 / 2 + np.pi / 4) * np.tan(lon1 / 2 + np.pi / 4)
        )
    
    # Spherical excess in steradians
    spherical_excess = np.abs(total_angle - (len(lons) - 2) * np.pi)
    
    # Area on the sphere
    return spherical_excess * radius ** 2

def calculate_geodesic_areas(corner_coordinates):
    """
    Calculate the geodesic area of each cell in square meters.
    
    Parameters:
        corner_coordinates (list of dict): List of dictionaries, each containing:
                                           - 'cell_index': Index of the cell
                                           - 'lon_corners': List of longitudes for the corners
                                           - 'lat_corners': List of latitudes for the corners
    
    Returns:
        list of dict: List of dictionaries, each containing:
                      - 'cell_index': Index of the cell
                      - 'area': Calculated area of the cell in square meters
    """
    areas = []
    
    for cell in corner_coordinates:
        lons = cell['lon_corners']
        lats = cell['lat_corners']
        
        # Calculate geodesic area
        area = haversine_area(lons, lats)
        areas.append({'cell_index': cell['cell_index'], 'area': area})
    
    return areas

def haversine_distance(lon1, lat1, lon2, lat2, radius=6378137):
    """
    Calculate the great-circle distance between two points on the Earth using the Haversine formula.
    
    Parameters:
        lon1, lat1 (float): Longitude and latitude of the first point in degrees.
        lon2, lat2 (float): Longitude and latitude of the second point in degrees.
        radius (float): Radius of the Earth in meters. Defaults to WGS84 value (6378137m).
    
    Returns:
        float: Distance between the two points in meters.
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c

def calculate_cell_lengths(corner_coordinates):
    """
    Calculate the edge lengths of each cell based on corner coordinates.
    
    Parameters:
        corner_coordinates (list of dict): List of dictionaries, each containing:
                                           - 'cell_index': Index of the cell
                                           - 'lon_corners': List of longitudes for the corners
                                           - 'lat_corners': List of latitudes for the corners
    
    Returns:
        list of dict: List of dictionaries, each containing:
                      - 'cell_index': Index of the cell
                      - 'edge_lengths': List of edge lengths in meters
    """
    lengths = []
    
    for cell in corner_coordinates:
        lons = cell['lon_corners']
        lats = cell['lat_corners']
        
        # Calculate edge lengths
        edge_lengths = []
        for i in range(len(lons)):
            lon1, lat1 = lons[i], lats[i]
            lon2, lat2 = lons[(i + 1) % len(lons)], lats[(i + 1) % len(lats)]  # Wrap around to first corner
            edge_length = haversine_distance(lon1, lat1, lon2, lat2)
            edge_lengths.append(edge_length)
        
        lengths.append({'cell_index': cell['cell_index'], 'edge_lengths': edge_lengths})
    
    return lengths

def spherical_triangle_area(lon1, lat1, lon2, lat2, lon3, lat3, radius=6378137):
    """
    Calculate the area of a spherical triangle using the spherical excess formula.
    
    Parameters:
        lon1, lat1: Longitude and latitude of the first vertex in degrees.
        lon2, lat2: Longitude and latitude of the second vertex in degrees.
        lon3, lat3: Longitude and latitude of the third vertex in degrees.
        radius (float): Radius of the Earth in meters. Defaults to WGS84 (6378137m).
    
    Returns:
        float: Area of the triangle in square meters.
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2, lon3, lat3 = map(np.radians, [lon1, lat1, lon2, lat2, lon3, lat3])
    
    # Calculate the spherical excess
    def haversine_distance_angle(lon1, lat1, lon2, lat2):
        """Calculate the angle subtended by two points at the center of the sphere."""
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Side lengths (angles subtended at the sphere's center)
    a = haversine_distance_angle(lon2, lat2, lon3, lat3)
    b = haversine_distance_angle(lon1, lat1, lon3, lat3)
    c = haversine_distance_angle(lon1, lat1, lon2, lat2)
    
    # Semi-perimeter of the spherical triangle
    s = (a + b + c) / 2
    
    # Tangent of spherical excess
    tan_e = np.sqrt(np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2))
    excess = 4 * np.arctan(tan_e)
    
    # Area of the spherical triangle
    return excess * radius**2

def calculate_triangle_areas(corner_coordinates):
    """
    Calculate the geodesic area of triangular cells in square meters.
    
    Parameters:
        corner_coordinates (list of dict): List of dictionaries, each containing:
                                           - 'cell_index': Index of the cell
                                           - 'lon_corners': List of longitudes for the corners
                                           - 'lat_corners': List of latitudes for the corners
    
    Returns:
        list of dict: List of dictionaries, each containing:
                      - 'cell_index': Index of the cell
                      - 'area': Calculated area of the cell in square meters
    """
    areas = []
    
    for cell in corner_coordinates:
        if len(cell['lon_corners']) < 3 or len(cell['lat_corners']) < 3:
            areas.append({'cell_index': cell['cell_index'], 'area': 0})
            continue
        
        # Extract the three vertices
        lon1, lon2, lon3 = cell['lon_corners'][:3]
        lat1, lat2, lat3 = cell['lat_corners'][:3]
        
        # Calculate the area
        area = spherical_triangle_area(lon1, lat1, lon2, lat2, lon3, lat3)
        areas.append({'cell_index': cell['cell_index'], 'area': area})
    
    return areas

#%%
# Example: Find nearest points for a single transect (e.g., DEE)
dee_transect = transect_data["DEE"]
dee_points = dee_transect[['x', 'y']].values  # Extract transect points
dee_nearest = find_nearest_grid_points(dee_points, lon_faces, lat_faces)


# Example: Verify for DEE transect
dee_nearest_indices = dee_nearest["indices"]
plot_transect_with_grid_on_map(
    transect_points=dee_points,
    nearest_indices=dee_nearest_indices,
    lon_faces=lon_faces,
    lat_faces=lat_faces,
    xlim=(-3.5, -2.5),  # Adjust longitude limits as needed
    ylim=(53.0, 54.0),  # Adjust latitude limits as needed
    title="DEE Transect Verification with Map"
)

# Example: Retrieve corner indices for the DEE transect nearest faces
dee_corner_indices = get_face_corner_indices(dee_nearest_indices, face_nodes)

# Print an example output for verification
print("Corner indices for DEE transect nearest faces:")
print(dee_corner_indices)


# Example: Retrieve corner coordinates for the DEE transect
dee_corner_coordinates = get_corner_coordinates(dee_corner_indices, lon_nodes, lat_nodes)

# Example output for verification
for cell in dee_corner_coordinates[:3]:  # Print the first 3 cells
    print(f"Cell {cell['cell_index']}:")
    print(f"  Lon Corners: {cell['lon_corners']}")
    print(f"  Lat Corners: {cell['lat_corners']}")
    
    
    
# # Example: Calculate areas for the DEE transect cells
# dee_cell_areas = calculate_cell_areas(dee_corner_coordinates)

# # Print example results for verificationD:/
# for cell_area in dee_cell_areas[:3]:  # Print the first 3 cells
#     print(f"Cell {cell_area['cell_index']} Area: {cell_area['area']} square degrees")
    
# # Example: Calculate areas for the DEE transect cells
# dee_geodesic_areas = calculate_geodesic_areas(dee_corner_coordinates)

# # Print example results for verification
# for cell_area in dee_geodesic_areas[:3]:  # Print the first 3 cells
#     print(f"Cell {cell_area['cell_index']} Area: {cell_area['area']} m²")


# Example: Calculate edge lengths for the DEE transect cells
dee_cell_lengths = calculate_cell_lengths(dee_corner_coordinates)
length_save = []
# Print example results for verification
for cell_length in dee_cell_lengths[:]:  # Print the first 3 cells
    print(f"Cell {cell_length['cell_index']} Edge Lengths: {cell_length['edge_lengths']} meters")
    length_save.append(np.mean(cell_length['edge_lengths']))
length_save = np.array(length_save)
    # Example: Calculate areas for the DEE transect triangular cells
dee_triangle_areas = calculate_triangle_areas(dee_corner_coordinates)

save = []
# Print example results for verification
for cell_area in dee_triangle_areas[:]:  # Print the first 3 cells
    print(f"Cell {cell_area['cell_index']} Area: {cell_area['area']} m²")
    save.append(cell_area['area'])
save = np.array(save)
cell_labels = [f'Cell {i + 1}' for i in range(18)]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the first dataset (Area) on the primary y-axis
ax1.plot(cell_labels, save, label='Area', color='blue', marker='o')
ax1.set_ylabel('Cell Area (m²)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(cell_labels, length_save, label='Length', color='red', marker='x')
ax2.set_ylabel('Cell Length (m)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add labels and title
plt.title('Cell Area and Length Comparison')
ax1.set_xlabel('Cells')

# Add a legend
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

plt.tight_layout()
plt.show()
