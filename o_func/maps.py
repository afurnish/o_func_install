# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:51:06 2023

@author: aafur
"""
from o_functions.start import opsys2;start_path = opsys2()
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import geopandas as gpd
import pandas as pd
import glob


def PRIMEA_loc_grid():
    grid_path = start_path + r'modelling_DATA/kent_estuary_project/5.Final/1.friction/1.3.0.0_testing_grid.dsproj_data/FlowFM/UK_West+Duddon+Raven_+liv+ribble+wyre+ll+ul+leven_kent_1.1_net.nc'
    ds = xr.open_dataset(grid_path)
    
    # Extract grid data from the loaded Dataset
    mesh2d_edge_nodes = ds["mesh2d_edge_nodes"].values
    mesh2d_face_nodes = ds["mesh2d_face_nodes"].values
    
    # Extract node coordinates from edge nodes
    mesh2d_edge_x = ds["mesh2d_edge_x"].values
    mesh2d_edge_y = ds["mesh2d_edge_y"].values
    
    mesh2d_face_x = ds["mesh2d_face_x"].values
    mesh2d_face_y = ds["mesh2d_face_y"].values

    triang = tri.Triangulation(mesh2d_face_x, mesh2d_face_y, triangles=ds.mesh2d_face_nodes.values[:,0:3])

    fig, ax = plt.subplots()
    fig.set_figheight(30)
    fig.set_figwidth(20)
    ax.triplot(mesh2d_face_x,mesh2d_face_y)

    unique_vertices, unique_indices = np.unique(ds.mesh2d_face_nodes.values[:,0:3]-1, axis=0, return_index=True)
    mesh2d_face_x = mesh2d_face_x[unique_indices]
    mesh2d_face_y = mesh2d_face_y[unique_indices]
    fig, ax = plt.subplots()
    fig.set_figheight(30)
    fig.set_figwidth(20)
    ax.triplot(mesh2d_edge_x,mesh2d_edge_y)
    
    
    from pyugrid.ugrid import UGrid
    ug = UGrid.from_ncfile(grid_path)  # Replace with the path to your UGRID dataset file
    
    face_nodes = ds.mesh2d_face_nodes.values[:,0:3] -1
# Extract the vertex coordinates and connectivity information
    vertices = ug.nodes[:,0], ug.nodes[:,1]  # Assumes 'lon' and 'lat' are the variable names for longitude and latitude in your UGRID dataset
    faces = ug.faces  # Subtract 1 to convert from 1-based indexing to 0-based indexing
    
    # Create a matplotlib figure and axis
    #%% THIS ONE WORKS
    fig, ax = plt.subplots()
    fig.set_figheight(90)
    fig.set_figwidth(70)
    
    # Plot the triangles using the vertex coordinates and faces
    ax.triplot(vertices[0], vertices[1], faces.data[:,0:3], linewidth=1.0, color='black')
    
    # Set the plot title and labels
    ax.set_title('UGRID Grid Plot')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Show the plot
    plt.show()
    
#%% Testing to plot only the traingles excluding the squares
    index_rows = np.where(faces.mask[:, 3])[0]


    fig, ax = plt.subplots()
    fig.set_figheight(90)
    fig.set_figwidth(70)
    
    # Plot the triangles using the vertex coordinates and faces
    ax.triplot(vertices[0], vertices[1], faces.data[index_rows,0:3], linewidth=1.0, color='black')
    
    # Set the plot title and labels
    ax.set_title('UGRID Grid Plot')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

#%% testing to plot only the squares
    index_rows2 = np.where(faces.mask[:, 3])[0]
    square_nodes = faces.data[faces.data[:, 3] != -999, :4]
    lons = vertices[0]
    lats = vertices[1]
    fig, ax = plt.subplots()
    fig.set_figheight(90)
    fig.set_figwidth(70)

    for square in square_nodes:
        ax.plot([lons[square[0]], lons[square[1]], lons[square[2]], lons[square[3]], lons[square[0]]],
        [lats[square[0]], lats[square[1]], lats[square[2]], lats[square[3]], lats[square[0]]], 'k-')
    ax.triplot(vertices[0], vertices[1], faces.data[index_rows,0:3], linewidth=1.0, color='black')


#def PRIMEA_loc_grid2():
    

    #Please note that the specific steps and variable names may vary depending on the structure and content of your UGRID dataset. Make sure to consult the documentation of your UGRID dataset and pyugrid library for proper usage.


#def loc_map():
    ''' This is a location map to show everything in one image, the tidal boundary, 
    river boundaries, location and area. 
    

    Returns
    -------
    None.

    '''
#%%
    fsize = 100 #main fontsize
    adjuster = 5/8 #sets label names
    adjuster2 = 1/2 #sets ticks
    labelpad = 5
    #colormaps = cm.viridis
    linecolor = 'red'
    wid = 10

    #set ocean boundaries 
    loc_ocean_bound_csv = pd.read_csv(start_path + r'modelling_DATA/kent_estuary_project/5.Final/1.friction/1.3.0.0_testing_grid.dsproj_data/FlowFM/001_delft_ocean_boundary_UKC3_b601t688_length-87_points.pli', delimiter = ' ', header = 1)  

    #set lon_lats for PRIMEA area
    lon,lat = (-3.65,-2.75),(53.20,54.52)
    
    fig, ax= plt.subplots(figsize=(55, 50))
    uk = gpd.read_file('F:/for_pete/uk_full.shp')
    uk.plot(cmap='Pastel2_r', legend=True, ax=ax)
    ax.set_xlim(-9, 15)
    ax.set_ylim(49.5, 61.25)
    plt.xticks(fontsize=fsize*adjuster2)
    plt.yticks(fontsize=fsize*adjuster2)
    
    ax.plot([lon[0], lon[0], lon[1], lon[1], lon[0]],
    [lat[0], lat[1], lat[1], lat[0], lat[0]],
    color= linecolor, linewidth=5)
    ax.set_xlabel('Longitude', fontsize = fsize)
    ax.set_ylabel('Latitude', fontsize = fsize)
    #ax.set_title('Square Plot for UK (Excluding Ireland)')
    
    ### Inset Map 
    ax_inset = ax.inset_axes([0.55, 0.1, 0.4, 0.8])
    
    for square in square_nodes:
        ax_inset.plot([lons[square[0]], lons[square[1]], lons[square[2]], lons[square[3]], lons[square[0]]],
        [lats[square[0]], lats[square[1]], lats[square[2]], lats[square[3]], lats[square[0]]], 'k-')
    ax_inset.triplot(vertices[0], vertices[1], faces.data[index_rows,0:3], linewidth=1.0, color='black')

    #plot ocean bound 
    ax_inset.scatter(loc_ocean_bound_csv.iloc[:,0],loc_ocean_bound_csv.iloc[:,2], 
                     marker = '^', color = 'green', s = fsize*10, label = 'ocean_bound')
    
    #plot river bounds 
    for i,file in enumerate(glob.glob(start_path + r'modelling_DATA/kent_estuary_project/5.Final/1.friction/SCW_runs/ukc3_files/river_pli/*.pli')):
        river = pd.read_csv(file, delimiter = ' ', header = 1)
        if i == 0:
            ax_inset.plot([river.iloc[0,0],river.iloc[1,0]],[river.iloc[0,2],river.iloc[1,2]], linewidth = fsize/3, color = 'orange', label = 'river_bound')
        else:
            ax_inset.plot([river.iloc[0,0],river.iloc[1,0]],[river.iloc[0,2],river.iloc[1,2]], linewidth = fsize/3, color = 'orange')

    #legend
    ax_inset.legend(fontsize = fsize/2)
    
    #set inset map of PRIMEA
    ax_inset.set_xlim(min(lon), max(lon))
    ax_inset.set_ylim(min(lat), max(lat))
    ax_inset.tick_params(labelsize=fsize*adjuster2)
    ax_inset.set_ylabel('Latitude', fontsize = fsize*adjuster)
    ax_inset.set_xlabel('Longitdue',fontsize = fsize*adjuster)
    #ax_inset.set_xticklabels([],minor = False) #remove xticks
    #ax_inset.set_yticklabels([], minor = False)
    
    ax_inset.spines['bottom'].set_color(linecolor)
    ax_inset.spines['top'].set_color(linecolor)
    ax_inset.spines['right'].set_color(linecolor)
    ax_inset.spines['left'].set_color(linecolor)
    ax_inset.spines['top'].set_linewidth(wid/2)
    ax_inset.spines['bottom'].set_linewidth(wid/2)
    ax_inset.spines['left'].set_linewidth(wid/2)
    ax_inset.spines['right'].set_linewidth(wid/2)
    
    plt.savefig(start_path + r'PhD/Conferences/EGU 2023/presentation_figures/PRIMEA_loc_map.png')

#%%
    fig, ax= plt.subplots(figsize=(55, 50))
    uk = gpd.read_file('F:/for_pete/uk_full.shp')
    uk.plot(cmap='Pastel2_r', legend=True, ax=ax)
    ax.set_xlim(-9, 2.5)
    ax.set_ylim(49.5, 61.25)
    plt.xticks(fontsize=fsize*adjuster2)
    plt.yticks(fontsize=fsize*adjuster2)
    
    ax.plot([lon[0], lon[0], lon[1], lon[1], lon[0]],
    [lat[0], lat[1], lat[1], lat[0], lat[0]],
    color= linecolor, linewidth=5, label = 'Area of\nInterest')
    ax.set_xlabel('Longitude', fontsize = fsize)
    ax.set_ylabel('Latitude', fontsize = fsize) 
    #ax.set_title('Square Plot for UK (Excluding Ireland)')
    plt.legend(loc = 'upper left', fontsize = 100)
  
#%%
    fig, ax= plt.subplots(figsize=(55, 50))
    uk = gpd.read_file('F:/for_pete/uk_full.shp')
    uk.plot(cmap='Pastel2_r', legend=True, ax=ax)
    ax.set_xlim(lon[0],lon[1])
    ax.set_ylim(lat[0], lat[1])
    plt.xticks(fontsize=fsize*adjuster2)
    plt.yticks(fontsize=fsize*adjuster2)
    
    heysham = [-2.92025,54.031833]
    liverpool = [-3.018,53.449694]
    #ax.plot([lon[0], lon[0], lon[1], lon[1], lon[0]],
    #[lat[0], lat[1], lat[1], lat[0], lat[0]],
    #color= linecolor, linewidth=5, label = 'Area of\nInterest')
    ax.set_xlabel('Longitude', fontsize = fsize)
    ax.set_ylabel('Latitude', fontsize = fsize) 
    plt.scatter(heysham[0],heysham[1],marker = '^', color = 'r', s = 1000,label = 'Heysham')
    plt.scatter(liverpool[0],liverpool[1],marker = '^', color = 'b', s = 1000*8,label = 'Liverpool')
    #ax.set_title('Square Plot for UK (Excluding Ireland)')
    plt.legend(loc = 'upper left', fontsize = 100)
#%%    
if __name__ == "__main__":
    PRIMEA_loc_grid()
    
    