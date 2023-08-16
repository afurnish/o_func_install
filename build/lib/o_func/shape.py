#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Resturn shapefiles that can be plotted for ease of use. 

Created on Mon Oct 24 13:01:46 2022
@author: af
"""

def coastline(start_p):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    
    plt.rcParams["figure.figsize"] = [20, 15]
    plt.rcParams['font.size'] = '16'
    plt.rcParams["figure.autolayout"] = True

    # set path for coastal shapefile
    path = r'modelling_DATA/kent_estuary_project/land_boundary/' + \
           r'QGIS_Shapefiles/UK_WEST_KENT_EPSG_4326_clipped_med_domain.shp'
    full_path = start_p + path


    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    UKWEST_coastline = gpd.read_file(full_path)
    
    fig,axes = plt.subplots(figsize=(30,15))
    out = UKWEST_coastline.plot(ax=axes, color="red")
    axes.set_xlim(-3.65,-2.75)
    axes.set_ylim(53.20,54.52)

    return UKWEST_coastline

def UKC3_area(start_p):
    import glob
    import xarray as xr
    import matplotlib.pyplot as plt

    
    path = r'Original Data/UKC3/owa/shelftmb' #path for hourly data
    full_path = start_p + path + '/*'
    
    d = []
    for filename in glob.glob(full_path):
        d.append(filename)
    
    T = xr.open_dataset(d[0])
    fig, ax = plt.subplots(figsize=(30,30))

    z = T.sossheig.values[0,:,:]
    x = T.nav_lon.values
    y = T.nav_lat.values

    plt.contourf(x,y,z)
    
    ax.set_xlim(-3.65,-2.75)
    ax.set_ylim(53.20,54.52)
    
    return x, y, z
    
# haversine distance equation    
    
def distance(lat1, lon1, lat2, lon2):
    from math import cos, asin, sqrt

    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))


# equation of a straight line from points


''' Notes 
Use the distance fucntion to work out best dimensions for displaying 
plots across different areas of space
'''