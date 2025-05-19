#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Naval Research Laboratory Delft3D-FM Simulation Configuration and Input
File Generation

Created on Tue May 23 12:34:57 2023
@author: af
"""

#!/usr/bin/env python # coding: utf-8
# version 0.1 2020/01/08 -- Cody Johnson
"""
# Updates:
# 2021/09/05 - Jay Veeramony
# Added ability to output water level boundary conditions
Issues identified:
1. If the boundary point in Delft3D-FM is a land point in the NCOM/HYCOM output due to resolution issues, this routine throws an error. There needs to be a check for this
"""
#import sys
from pathlib import Path
import argparse
import cartopy.crs as ccrs
#import cartopy.feature as cfeature import cmocean.cm as cmo
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import xarray as xr
import sys
s = ' '

### functions ###
def write_wl_for_pli(bc_fn, gcm, pli, quantity): 
    """
    append or write 2D boundary conditions for water level
    bc_fn = path to write or append boundary condition data
    gcm = general circulation model output which contains boundary points (xr.
    DataArray)
    pli = table of boundary support points (pd.DataFrame)
    quantity = variable to output to the BC files (salinity or water_temp) depth_avg = flag to enable depth averaging
    23
    24 Schoenauer et al.
    """
    with open(bc_fn, "a") as f:
        gcm_refd, ref_date = assign_seconds_from_refdate(gcm) 
        for _, (x_pli, y_pli, pli_point_name) in pli.iterrows():
            if x_pli < 0:
                x_pli_east = x_pli + 360
            else:
                x_pli_east = x_pli
            bc_data = waterlevel_to_pli_point(gcm_refd, x_pli_east, y_pli)
            write_wl_record(f, bc_data, pli_point_name, ref_date)
def write_wl_record(f, bc_data, pli_point_name, ref_date): 
    """
    append or write time series water level boundary conditions
    f = file descriptor for writing boundary condition output
    bc_data = data at with percent bed coords (xr.DataFrame) pli_point_name = name for entry
    ref_date = date used to calculate offset of the record in seconds 
    """
    # write a record
    f.write("[forcing]\n")
    f.write(f"Name" + s * 15  + "= {pli_point_name}\n")
    f.write("Function" + s * 15 + "= timeseries\n")
    f.write("Time-interpolation" + s * 15 + "= linear\n")
    f.write("Quantity" + s * 15 + "= time\n")
    f.write(f"Unit" + s * 15 + "= seconds since {ref_date}\n")
    f.write("Quantity" + s * 15 + "= waterlevelbnd\n")
    f.write("Unit" + s * 15 + "= m\n")


# write data after converting to dataframe and iterating over the rows
    for td, value in bc_data.to_dataframe()['surf_el'].iteritems(): 
        value = f"{value:.03f}"
        f.write(f"{td} {value}\n")
        
    f.write("\n")
def write_t3d_for_pli(bc_fn, gcm, pli, quantity, nlevels, depth_avg): 
    """
    append or write 3D boundary conditions for quantities
    bc_fn = path to write or append boundary condition data
    gcm = general circulation model output which contains boundary points (xr.
    DataArray)
    pli = table of boundary support points (pd.DataFrame)
    quantity = variable to output to the BC files (salinity or water_temp) depth_avg = flag to enable depth averaging
    """
    with open(bc_fn, "a") as f:
    
        gcm_refd, ref_date = assign_seconds_from_refdate(gcm) 
        for _, (x_pli, y_pli, pli_point_name) in pli.iterrows():

            if x_pli < 0:
                x_pli_east = x_pli + 360
            else:
                x_pli_east = x_pli
            if (quantity == "salinity") or (quantity == "water_temp"): 
                bc_data = interpolate_to_pli_point(
                    gcm_refd , quantity , x_pli_east , y_pli , nlevels )
                if depth_avg:
                    write_ts2d_record(f, bc_data, pli_point_name, quantity, ref_date)
                else:
                    write_ts3d_record(f, bc_data, pli_point_name, quantity, ref_date)
            # in the case of velocity both components are interpolated
            elif quantity == "velocity":
                bc_data = interpolate_to_pli_point(gcm_refd, ["water_u", "water_v"], x_pli_east, y_pli, nlevels )
                if depth_avg:
                    write_vector_2d_record(f, bc_data, pli_point_name, quantity,ref_date)
                else:
                    write_ts3d_record(f, bc_data, pli_point_name, quantity, ref_date)


def write_vector_2d_record(f, bc_data, pli_point_name, quantity, ref_date): 
    """
    append or write time series boundary conditions for depth averaged velocity
    f = file descriptor for writing boundary condition output bc_data = data at with percent bed coords (xr.DataFrame) pli_point_name = name for entry
    quantity = variable to output to the BC files
    ref_date = date used to calculate offset of the record in seconds 
    """
# get units for quantity
    if quantity == "velocity":
        vector = "uxuyadvectionvelocitybnd:ux,uy" 
        quantbndx = "ux"
        quantbndy = "uy"
        x_comp = "water_u"
        y_comp = "water_v"
        units = "-"
    else:
        print('quantity should be "velocity"\n') 
        raise ValueError
    # write a record
    f.write("[forcing]\n")
    f.write(f"Name" + s * 15  + "= {pli_point_name}\n")
    f.write("Function" + s * 15 + "= timeseries\n")
    f.write("Time-interpolation" + s * 15 + "= linear\n")
    f.write("Quantity" + s * 15 + "= time\n")
    f.write(f"Unit" + s * 15 + "= seconds since {ref_date}\n")
    f.write(f"Vector" + s * 15 + "= {vector}\n")
    f.write(f"Quantity" + s * 15 + "= {quantbndx}\n")
    f.write(f"Unit" + s * 15 + "= {units}\n")
    f.write(f"Quantity" + s * 15 + "= {quantbndy}\n")
    f.write(f"Unit" + s * 15 + "= {units}\n")



# write data after converting to dataframe and iterating over the rows
    for td, values in (bc_data.to_dataframe()[[x_comp, y_comp]].unstack().iterrows()):
        
        
        
        x_comp_val = values.water_u.mean() 
        y_comp_val = values.water_v.mean()
        #values_str = [ f’{x_comp_val:0.2f}’ ’ ’ f’{y_comp_val:.02f}’]
        f.write(f"{td} {x_comp_val:0.2f} {y_comp_val:.02f}\n") 
        
    f.write("\n")
    
    
def write_ts2d_record(f, bc_data, pli_point_name, quantity, ref_date): 
    """
    append or write time series boundary conditions for depth averaged quantities
    f = file descriptor for writing boundary condition output bc_data = data at with percent bed coords (xr.DataFrame) pli_point_name = name for entry
    quantity = variable to output to the BC files
    ref_date = date used to calculate offset of the record in seconds 
    """
# get units for quantity
    if quantity == "salinity": 
        quantbnd = "salinitybnd" 
        units = "ppt"
    elif quantity == "water_temp": 
        quantbnd = "temperaturebnd" 
        units = "deg_C"
    else:
        print('quantity needs to be either "salinity" or "water_temp"\n') 
        raise ValueError
        
    # write a record
    f.write("[forcing]\n")
    f.write(f"Name" + s * 15  + "= {pli_point_name}\n")
    f.write("Function" + s * 15 + "= td3\n")
    f.write("Quantity" + s * 15 + "= time\n")
    f.write("Unit" + s * 15 + "= seconds since {ref_date}\n")
    f.write("Quantity" + s * 15 + "= {quantbnd}\n")
    f.write("Unit" + s * 15 + "= {units}\n")

    # write data after converting to dataframe and iterating over the rows
    for td, values in bc_data.to_dataframe()[quantity].unstack().iterrows():

        # take mean of values to get depth averaged
        value = values.mean()
# see results of interpolation
        if value > 100.0: 
            print(f"Problem with {quantity} exceeding maximum allowed value: {values.max ():.03f} ppt.")
        elif value < 0.0:
            print(f"Problem with {quantity} becoming negative: {values.max():.03f} ppt.")

            print(f"Negative value for {quantity} has been set to 0.01 {units}.") 
            value = 0.01
            
        value = f"{value:.02f}" 
        f.write(f"{td} {value}\n")
    f.write("\n")
        
def write_vector_3d_record(f, bc_data, pli_point_name, quantity, ref_date): 
    """
    append or write 3D boundary conditions for quantities
    f = file descriptor for writing boundary condition output bc_data = data at with percent bed coords (xr.DataFrame) pli_point_name = name for entry
    quantity = variable to output to the BC files
    ref_date = date used to calculate offset of the record in seconds 
    """
    if quantity == "velocity":
        vector = "uxuyadvectionvelocitybnd:ux,uy" 
        quantbndx = "ux"
        quantbndy = "uy"
        x_comp = "water_u"
        y_comp = "water_v"
        units = "-"
    else:
        print('quantity should be "velocity"\n') 
        raise ValueError
        
    # convert percent from bed into formated string
    pos_spec = [f"{perc:.02f}" for perc in bc_data.perc_from_bed.data] 
    pos_spec_str = " ".join(pos_spec[::-1]) # reverse order for D3D

# write a record
    f.write("[forcing]\n")
    f.write(f"Name" + s * 15  + "= {pli_point_name}\n")
    f.write("Function" + s * 15 + "= td3\n")
    f.write("Time-interpolation" + s * 15 + "= linear\n")
    f.write("Vertical position type" + s * 15 + "= percentage from bed\n")
    f.write("Vertical position specification" + s * 15 + "= {pos_spec_str}\n")
    f.write("Vertical interpolation" + s * 15 + "= linear\n")
    f.write("Quantity" + s * 15 + "= time\n")
    f.write("Unit" + s * 15 + "= seconds since {ref_date}\n")
    f.write("Vector" + s * 15 + "= {vector}\n")

    # loop over number of vertical positions
    for vert_pos in range(1, len(pos_spec) + 1): 
        f.write(f"Quantity" + s*10 + "={quantbndx}\n")
        f.write(f"Unit" + s*10 + "= {units}\n")
        f.write(f"Vertical position" + s*10 + "= {vert_pos}\n")
        
        f.write(f"Quantity" + s*10 + "= {quantbndy}\n")
        f.write(f"Unit" + s*10 + "= {units}\n")
        f.write(f"Vertical position" + s*10 + "= {vert_pos}\n")


    # write data after converting to dataframe and iterating over the rows
    for td, values in (bc_data.to_dataframe()[[x_comp, y_comp]].unstack().iterrows()):
        # get componets as array in order to format for d3d input 
        x_comp_vals = values[x_comp].values[::-1] # reverse order for D3D 
        y_comp_vals = values[y_comp].values[::-1] # reverse order for D3D 
        values = [f"{x_comp_val:.02f} {y_comp_val:.03f}" for x_comp_val, y_comp_val in zip(x_comp_vals, y_comp_vals) ]
        values_str = " ".join(values) 
        f.write(f"{td} {values_str}\n")
    f.write("\n")
    
    
def write_ts3d_record(f, bc_data, pli_point_name, quantity, ref_date): 
    """
    append or write 3D boundary conditions for quantities
    f = file descriptor for writing boundary condition output bc_data = data at with percent bed coords (xr.DataFrame) pli_point_name = name for entry
    quantity = variable to output to the BC files
    ref_date = date used to calculate offset of the record in seconds 
    """
    # get units for quantity
    if quantity == "salinity": 
        quantbnd = "salinitybnd" 
        units = "ppt"
    elif quantity == "water_temp": 
        quantbnd = "temperaturebnd" 
        units = "deg_C"
    else:
        print('quantity needs to be either "salinity" or "water_temp"\n') 
        raise ValueError
        
    # convert percent from bed into formated string
    pos_spec = [f"{perc:.02f}" for perc in bc_data.perc_from_bed.data] 
    pos_spec_str = " ".join(pos_spec[::-1]) # reverse order for D3D

  # write a record

    f.write("[forcing]\n")
    f.write(f"Name" + s * 15  + "= {pli_point_name}\n")
    f.write("Function" + s * 15 + "= td3\n")
    f.write("Time-interpolation" + s * 15 + "= linear\n")
    f.write("Vertical position type" + s * 15 + "= percentage from bed\n")
    f.write("Vertical position specification" + s * 15 + "= {pos_spec_str}\n")
    f.write("Vertical interpolation" + s * 15 + "= linear\n")
    f.write("Quantity" + s * 15 + "= time\n")
    f.write("Unit" + s * 15 + "= seconds since {ref_date}\n")

    # loop over number of vertical positions
    for vert_pos in range(1, len(pos_spec) + 1): 
        f.write(f"Quantity" + s*10 + "= {quantbnd}\n")
        f.write(f"Unit" + s*10 + "= {units}\n")
        f.write(f"Vertical position" + s*10 + "= {vert_pos}\n")
        

    # write data after converting to dataframe and iterating over the rows
    for td, values in bc_data.to_dataframe()[quantity].unstack().iterrows():
        # see results of interpolation
        if values.max() > 100.0: 
            print(f"problem with {quantity} exceeding maximum allowed value: {values.max ():.03f} ppt")
        elif values.min() < 0.0:
            print(f"problem with {quantity} becoming negative: {values.max():.03f} ppt")
            print(f"Negative values for {quantity} has been set to 0.01 {units}.")
            values.where(values > 0.01, 0.01, inplace=True)
            
        values = [f"{value:.02f}" for value in values]
        values_str = " ".join(values[::-1]) # reverse order for D3D
        f.write(f"{td} {values_str}\n")
    f.write("\n")


def assign_seconds_from_refdate(gcm): 
    """
    This func assigns seconds from a user specified ref date as coords. This is how D3D interpolates the boundary conditions in time.
    gcm = model output to add coords to
    """
    ref_date = gcm.time.data[0]
    ref_dt = pd.to_datetime(ref_date)
    ref_date_str = ref_dt.strftime("%Y-%m-%d %H:%M:%S")
    timedeltas = pd.to_datetime(gcm.time.data) - ref_dt
    seconds = timedeltas.days * 24 * 60 * 60 + timedeltas.seconds
    gcm = gcm.assign_coords(coords={"seconds_from_ref": ("time", seconds)}) 
    
    return gcm.swap_dims({"time": "seconds_from_ref"}), ref_date_str

def interpolate_to_pli_point(gcm_refd, quantity, x_pli_east, y_pli, nlevels): 
    """interpolates the quanitites to the sigma depths and pli coords
    values = [f"{value:.02f}" for value in values]
    values_str = " ".join(values[::-1]) # reverse order for D3D f.write(f"{td} {values_str}\n")
    = {quantbnd}\n") = {units}\n")
    = {vert_pos}\n")
    323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346
    347 348 349
    350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377
    
    30 Schoenauer et al.
      gcm_refd = gcm with new time coordinates
    quantity = variable to output to the BC files (salinity or water_temp)
    x_pli_east = longitude of pli point in degrees east from meridian (NCOM convention
    )
    y_pli = latitude 
    """
    # interpolate to pli point and drop data below bed level at nearest gcm_refd point
    bc_data = (gcm_refd[quantity] .interp(lon=x_pli_east, lat=y_pli) .dropna(dim="depth").squeeze())
    # add coordinate for percent from bed. D3D uses this in its bc file format
    gcm_refd_zb = bc_data.depth[-1] # get bed level of gcm_refd point
    if nlevels > 1:
        sigma_lvls = np.arange(nlevels)/(nlevels -1)*100 
        zlvls = sigma_lvls[:]*gcm_refd_zb.data/100 
        bc_data = bc_data.interp(depth=zlvls)
        
    perc_from_bed = 100 * (-1 * bc_data.depth + gcm_refd_zb) / gcm_refd_zb
    bc_data = bc_data.assign_coords(coords={"perc_from_bed": ("depth", perc_from_bed.data)})
    
    return bc_data

def waterlevel_to_pli_point(gcm_refd, x_pli_east, y_pli): 
    """interpolates the quanitites to the sigma depths and pli coords
    gcm_refd = gcm with new time coordinates
    x_pli_east = longitude of pli point in degrees east from meridian (NCOM convention
    )
    y_pli = latitude 
    """
    # interpolate to pli point and drop data below bed level at nearest gcm_refd point
    bc_data = (gcm_refd['surf_el'] .interp(lon=x_pli_east, lat=y_pli) .squeeze())
    
    return bc_data

### main loop ###
if __name__ == "__main__":
### arguments ###
    parser = argparse.ArgumentParser() 
    parser.add_argument("nc",
       help="NCOM NetCDF output containing boundary support points and duration of Delft3D simulation",
       )
    parser.add_argument("quantity", help='NCOM variable. Must be either "salintiy", "water_temp", or "velocity"', )
    parser.add_argument( "--pli-list", nargs="*", type=str, help="list of boundary support point polyline filenames", required=True,dest="pli_list",) 
    parser.add_argument("--nlevels",default=11,type=int,help="Number of vertical levels", dest="nlevels",) 
    parser.add_argument("--bc-filename",help="Optional filename for Delft3D boundary condition filename", type=str,dest="bc_filename",) 
    parser.add_argument("--depth-avg",help="flag to enable depth averaged output", default=False,action="store_true",dest="depth_avg",) 
    parser.add_argument("--plot",help="flag to enable plotting", default=False, action="store_true", dest="plot",)
    args = parser.parse_args()
    gcm = args.nc
    quantity = args.quantity 
    pli_list = args.pli_list 
    depth_avg = args.depth_avg 
    plot = args.plot
    nlevels = args.nlevels
    # validate arguments
    if ((quantity != "salinity")and (quantity != "water_temp") and (quantity != "velocity") and (quantity != "surf_el")):
        print(f'<quantity> was specfied as {quantity}, but should be either "salinity","water_temp", "velocity" or "surf_el".')
        raise ValueError
        # open gcm NetCDF output as Xarray dataset
        try:
            gcm = xr.open_dataset(Path(gcm), drop_variables="tau")
        except FileNotFoundError as e:
            print("f<NCOM output> should be path to NCOM NetCDF output") 
            raise e
            # Some netcdf files have "latitude" and "longitude" instead of "lat" and "lon" # In such cases, rename the indices
        try:
            gcm = gcm.rename({'latitude':'lat', 'longitude':'lon'}) 
        except ValueError:
            pass
    # set default boundary condition filename depending on quantity
    bc_fn = args.bc_filename 
    if bc_fn == None:
        if quantity == "salinity": 
            bc_fn = Path("Salinity.bc")
        elif quantity == "water_temp": 
            bc_fn = Path("Temperature.bc")
        elif quantity == "velocity": 
            bc_fn = Path("Velocity.bc")
        elif quantity == "surf_el":
            bc_fn = Path("Waterlevel.bc")
            
    # Overwrite the file on the first instance (i.e., clean up whatever is in the file )
    # There is probably a more elegant way to do this?
    with open(bc_fn, 'w') as f: 
        f.write("")
        
    # pli files opened as Pandas DataFrames
    pli_points = []
    for pli_fn in pli_list:
        print(f"Reading in file: {pli_fn}") 
        pli = pd.read_csv(pli_fn, sep="\s+", skiprows=2, header=None, names=["x", "y", "point_id"] )
        # Check if there are points in the pli file outside the lat/lon extent in the
        # input file. If the points are outside the domain in the provided netcdf file
        # this will either throw Nan’s (for water level BCs) or cause interpolation
        # errors (for the other boundary conditions)
        pli_lat, pli_lon = pli["y"].values, pli["x"].values+360
        latmin, latmax = min(gcm.coords["lat"].values), max(gcm.coords["lat"].values) 
        lonmin, lonmax = min(gcm.coords["lon"].values), max(gcm.coords["lon"].values) 
        if (any(pli_lon < lonmin) or any(pli_lon > lonmax) or any(pli_lat < latmin) or any(pli_lat > latmax)):
            if (all(pli_lon < lonmin) or all(pli_lon > lonmax) or all(pli_lat < latmin) or all(pli_lat > latmax)):
                print("All points in the pli file are outside the domain in the netcdf file")
                print("Please fix before continuing.")
                print("Exiting routine...") 
            else:
                print("Some points in the pli file are outside the domain in the netcdf file")
                print("Please remove these from the pli before continuing.") 
                print("Exiting routine...")
                for i in range(pli_lon.size):
                    if (pli_lon[i] < lonmin or pli_lon[i] > lonmax or pli_lat[i] < latmin or pli_lat[i] > latmax):
                        print(f"{pli_lon[i]-360:0.05f}, {pli_lat[i]:.05f} ")
            sys.exit()

                    
        # Move forward
        if quantity == "surf_el":
            write_wl_for_pli(bc_fn, gcm, pli, quantity) 
        else:
            write_t3d_for_pli(bc_fn, gcm, pli, quantity, nlevels, depth_avg=depth_avg)
        # add points to list for visualization
        pli_points.append(pli)
        
    # concat pli points
    pli_points = pd.concat(pli_points)
    
    ### visualization ###
    # color map depending on quantity 
    if plot:
        matplotlib.use('Agg')
        if quantity == "salinity":
            cmap = cmo.haline
        elif quantity == "water_temp":
            cmap = cmo.thermal
        elif quantity == "velocity":
            cmap = "jet"
        elif quantity == "surf_el":
            cmap = "jet"
            
        # setup orthographic projection for geographic data
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(16, 9)) 
        ax.gridlines(draw_labels=True) 
        ax.coastlines(resolution='50m')
        
        # plot initial quantity at surface
        if (quantity == "salinity") or (quantity == "water_temp"): 
            gcm[quantity].isel(time=0, depth=0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap )
        elif quantity == "velocity":
            tmp = gcm.isel(time=0, depth=0)
            tmp["magnitude"] = np.sqrt(tmp["water_u"] ** 2 + tmp["water_v"] ** 2)
            
            tmp["magnitude"].plot( ax=ax,transform=ccrs.PlateCarree(),cmap=cmap,cbar_kwargs={"label": "velocity magnitude [m/s]"},)
        elif quantity == "surf_el":
            gcm[quantity].isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap)
            
        # add coastline for reference
        # ax.add_feature(cfeature.COASTLINE, edgecolor="0.3")
        # boundary condition support points
        pli_points.plot.scatter("x", "y", marker="x", color="k", ax=ax, transform=ccrs.PlateCarree())
        fig.savefig("point_output_locations.png", bbox_inches="tight")           
            








# 34 Schoenauer et al.
#   tmp["magnitude"].plot( ax=ax,
# transform=ccrs.PlateCarree(),
# cmap=cmap,
# cbar_kwargs={"label": "velocity magnitude [m/s]"},
# )
# elif quantity == "surf_el":
# gcm[quantity].isel(time=0).plot(
# ax=ax, transform=ccrs.PlateCarree(), cmap=cmap
# )
# # add coastline for reference
# # ax.add_feature(cfeature.COASTLINE, edgecolor="0.3")
# # boundary condition support points
# pli_points.plot.scatter(
# "x", "y", marker="x", color="k", ax=ax, transform=ccrs.PlateCarree()
# )
# fig.savefig("point_output_locations.png", bbox_inches="tight")
# 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618
# 1 2 3 4 5 6 7 8 9
# 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
# A.2 coamps2meteo.py
# The following section contains the coamps2meteo.py Python code used to create meteorological forcing files, <.amp>, <.amu>, and <.amv>.
#  # -*- coding: utf-8 -*- """
# Given a region, start date and end date creates meteorological forcing files from COAMPS
# Usage:
# $ python coamps2meteo.py --region COAMPSREGION --start-date start_date --end-date end_date --datadir COAMPS_DATADIR
# date format: YYYYMMDD
# Examples:
# """
# # # # # # #
# Created By: Jay Veeramony Version: 1.1
# Created on: Unknown
# Last modified: 2022-07-22
# Various modifications to bring code in line with python best practice Combined routines
# import argparse import logging import os
# $ python coamps2meteo.py --region eqam --start-date 20190101 --end-date 20190107 --datadir /u/NOGAPS/COAMPSg
# $ python coamps2meteo.py -r eqam -s 20190101 -e 20190107 -d /u/NOGAPS/COAMPSg

# Delft3D-FM simulation configuration and input file generation 35
#   import time import sys import shutil
# from datetime import date , datetime , timedelta from pathlib import Path
# from subprocess import call
# import numpy as np
# from scipy.io import loadmat
# def list_inputfiles(coamps_region, sdtg, edtg, ddir, component): """
# Creates list of input files that are read from the COAMPS database
# Args:
# coamps_region (str): Model region from which data is needed
# sdtg (str): Start date for data retrieval (YYYYMMDD format)
# edtg (str): End date for data retrieval (YYYYMMDD format)
# ddir (str): Data directory
# component: Meteorological component to retrieve, valid values are
# "pres", "wnd_ucmp", "wnd_vcmp"
# Returns:
# List of files that contain the data. These files could be gzipped.
# """
# sdtg = datetime.strptime(sdtg, ’%Y%m%d’) edtg = datetime.strptime(edtg, ’%Y%m%d’)
# ddir = Path(ddir).resolve()/coamps_region
# dirs = list(ddir.glob(’20*’)) print("Type of dirs:", type(dirs)) print("Size of dirs:", len(dirs))
# # check if directory is empty (likely a typhoon mounting error)
# if not dirs:
# print(’dirs is empty’)
# sys.exit(’directory empty or nonexistant. Check mount of typhoon’)
# # initial file
# fileslist = list(dirs[0].glob(coamps_region+’_’+component+’.*’))
# # rest of files
# for idir in range(1, len(dirs)): fileslist.extend(dirs[idir].glob(coamps_region+’_’+component+’.*’))
# file_dates = list()
# for filename in fileslist:
# file_dates.append(str(filename).split(’.’)[1])
# #
# # sort files and dates, necessary bc zipped files #
# file_dates = np.asarray(file_dates , dtype=’int’)
# 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86

# 36 Schoenauer et al.
#   idx = np.argsort(file_dates) file_dates = file_dates[idx]
# fileslist = np.array(fileslist) fileslist = fileslist[idx]
# tstart = int(sdtg.strftime(’%Y%m%d%H’)) tend = int(edtg.strftime(’%Y%m%d%H’))
# idx = (file_dates >= tstart)*(file_dates <= tend) return fileslist[idx]
# def append_met_data(met_file, ref_time, datafile, nlon, nlat): """
# Appends the data to the meteorological file
# Args:
# met_file: file being written
# ref_time: Reference time
# datafile: file from list of files between start/end date nlon, nlat: size of data array (obtained from model info)
# """
# print(’reading file: ’, datafile) if str(datafile).endswith(’gz’):
# print(’zipped file to be copied locally/unzipped’) zipfile = os.path.basename(files[0]) shutil.copyfile(files[0], ’./’+zipfile) call([’gunzip’, zipfile])
# # reassign datafile to just unzipped file
# cols = zipfile.split(’.’)
# datafile = ’./’+cols[0]+’.’+cols[1]
# file_time = cols[1]
# time_array = datetime.strptime(file_time ,
# + \
# ’%Y%m%d%H’) data = read_coamps(datafile, nlon, nlat, 12)
# with
# open(metfile , ’a’) as fid:
# for i in range(len(time_array)):
# np.arange(12) * timedelta(hours=1)
# os.remove(datafile) else:
# tc = time.mktime(time_array[i].timetuple()) - \ time.mktime(ref_time.timetuple())
# logging.info(’TIME = %.3f hours since %s # %s’, tc/3600, ref_time.strftime(’%Y%m%d%H’),
# time_array[i].strftime(’%Y%m%d%H’)) fid.write(’TIME = {0} hours since {1} # {2}’.
# format(tc/3600, ref_time, time_array[i])) fid.write(’\n’)
# for m in reversed(range(data.shape[1])): for j in data[i, m, :]:
# fid.write(’%.2f ’ % j) fid.write(’\n’)
# 87 88 89 90 91 92 93 94 95 96 97 98 99
# 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143

# Delft3D-FM simulation configuration and input file generation 37
#   file_time = str(datafile).split(’.’)[-1]
# time_array = datetime.strptime(file_time , ’%Y%m%d%H’) + \
# np.arange(12) * timedelta(hours=1)
# data = read_coamps(datafile, nlon, nlat, 12)
# with
# open(met_file , ’a’) as fid:
# for i in range(len(time_array)):
# tc = time.mktime(time_array[i].timetuple()) - \ time.mktime(ref_time.timetuple())
# logging.info(’TIME = %.3f hours since %s # %s’, tc/3600, ref_time.strftime(’%Y%m%d%H’),
# time_array[i].strftime(’%Y%m%d%H’)) fid.write(’TIME = {0} hours since {1} # {2}’.
# format(tc/3600, ref_time, time_array[i])) fid.write(’\n’)
# for m in reversed(range(data.shape[1])): for j in data[i, m, :]:
# fid.write(’%.2f ’ % j) fid.write(’\n’)
# def get_modelinfo(area_name): ’’’
# Given the name of an area with met data, extract information about the grid such as its location, grid size and grid resolution
# Args:
# area_name(str): COAMPS model region
# Returns:
# sdir (str): Sub-directory name to check for data. ssdir (str) : Sub-sub directory name
# nx (int) : Number of longitude points on the grid. ny (int) : Number of latitude points on the grid. minlat (float) : Origin - latitude.
# minlon (float) : Origin - longitude.
# maxlat (float) : max extent of latitude.
# maxlon (float) : max extent of longitude.
# ’’’
# path = os.path.dirname(os.path.abspath(__file__))
# area = loadmat(path+’/modelinfo.mat’) # load mat-file
# mdata = area[’modelinfo’] # variable in mat file
# mdtype = mdata.dtype # dtypes of structures are "unsized objects"
# # * SciPy reads in structures as structured NumPy arrays of dtype object * # The size of the array is the size of the structure array, not the number # elements in any particular field. The shape defaults to 2-dimensional. * # For convenience make a dictionary of the data using the names from dtype
# ndata = {n: mdata[0][n] for n in mdtype.names}
# # When reading in the Matlab mat file, the different integers and floats # have varying dtypes, presumably because Matlab uses the lowest size
# # needed to store the data to save space. Make them all uniform size here # for convenience
# ndata[’nx’] = ndata[’nx’].astype(’int’)
# 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200

# 38 Schoenauer et al.
#   ndata[’ny’] = ndata[’ny’].astype(’int’) ndata[’minlat’] = ndata[’minlat’].astype(’float64’) ndata[’minlon’] = ndata[’minlon’].astype(’float64’) ndata[’maxlat’] = ndata[’maxlat’].astype(’float64’) ndata[’maxlon’] = ndata[’maxlon’].astype(’float64’)
# # Get line number where data is to be retrieved from
# iloc = np.squeeze(np.where(ndata[’model1’] == area_name))
# sdir = str(np.squeeze(ndata[’model1’][iloc])) ssdir = str(np.squeeze(ndata[’model2’][iloc])) # nx = ndata[’nx’][iloc]
# # ny = ndata[’ny’][iloc]
# # minlat = ndata[’minlat’][iloc]
# # minlon = ndata[’minlon’][iloc]
# # maxlat = ndata[’maxlat’][iloc]
# # maxlon = ndata[’maxlon’][iloc]
# # return sdir, ssdir, nx, ny, minlat, minlon, maxlat, maxlon area_info = {’sdir’: sdir,
# ’ssdir’: ssdir,
# ’nx’: ndata[’nx’][iloc],
# ’ny’: ndata[’ny’][iloc], ’minlat’: ndata[’minlat’][iloc], ’minlon’: ndata[’minlon’][iloc], ’maxlat’: ndata[’maxlat’][iloc], ’maxlon’: ndata[’maxlon’][iloc]}
# return area_info
# def read_coamps(data_file, nlon, nlat, ntime): """
# Given a binary file with meteorology data in /u/NOGAPS, reads it into a data array Parameters
# Args:
# data_file (str): File containing binary data. nlon (int) : Number of longitude locations. nlat (int) : Number of latitude locations. ntime (int) : Number of taus.
# Returns: data
# """
# (float) : Matrix of size (nlon, nlat, ntime) containing the data.
# np.zeros(ntime) np.zeros(ntime)
# header =
# footer =
# data = np.zeros((nlon, nlat, ntime))
# with
# open(data_file , ’rb’) as fid: for j in range(ntime):
# header[j] = np.fromfile(fid, dtype=’>u4’, count=1) local_data = np.fromfile(fid, dtype=’>f4’, count=nlon*nlat) footer[j] = np.fromfile(fid, dtype=’>u4’, count=1)
# # Reshape the matrix in fortran order
# local_data = np.reshape(local_data, (nlon, nlat), order=’F’) data[:, :, j] = local_data
# 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257

# Delft3D-FM simulation configuration and input file generation 39
#   if header[0] != footer[0]:
# print(’Header ’ + str(header[0]))
# print(’Footer ’ + str(footer[0]))
# print(’File seems to be corrupt or read order is wrong’)
# # To keep it same as the openearth tools structure, permute # the array to have time loop on the first dimension and
# # longitude on the third dimension
# data = data.transpose(2, 1, 0)
# return data
# def write_met_header(met_var, met_file, nlon, nlat, minlon, maxlon, minlat, maxlat):
# """
# Writes header for meteorological file
# """
# dlon = (maxlon - minlon)/(nlon - 1) dlat = (maxlat - minlat)/(nlat - 1)
# with
# open(met_file , ’w’) as fid: fid.write(’### START OF HEADER\n’)
# fid.write(’FileVersion fid.write(’filetype fid.write(’n_cols fid.write(’n_rows fid.write(’grid_unit fid.write(’x_llcorner fid.write(’y_llcorner fid.write(’dx fid.write(’dy fid.write(’n_quantity
# if met_var == ’pres’: fid.write(’quantity1 fid.write(’unit nodata_value = 101300.0
# elif met_var == ’wnd_ucmp’: fid.write(’quantity1 fid.write(’unit nodata_value = 0.0
# elif met_var == ’wnd_vcmp’: fid.write(’quantity1 fid.write(’unit nodata_value = 0.0
# = 1.03\n’)
# = meteo_on_equidistant_grid\n’) = %5d \n’ % nlon)
# = %5d \n’ % nlat)
# = ’ + ’deg’ + ’\n’)
# else:
# sys.exit(’Incorrect variable names’)
# fid.write(’NODATA_value = %.2f \n’ % (nodata_value)) fid.write(’### END OF HEADER\n’)
# """
# Start of the main routing """
# = %5.2f = %5.2f = %5.2f = %5.2f = 1
# \n’ % minlon) \n’ % minlat) \n’ % (dlon)) \n’ % (dlat)) \n’)
# =’ + ’air_pressure’ + ’\n’)
# =’ + ’Pa’ + ’\n’)
# =’ + ’x_wind’ + ’\n’)
# =’ + ’m/s’ + ’\n’)
# =’ + ’y_wind’ + ’\n’)
# =’ + ’m/s’ + ’\n’)
# 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314

# 40 Schoenauer et al.
#   if __name__ == "__main__":
# valid_choices = [’arctic’, ’cencoos_n3’, ’e_pac_comp’, ’europe2’,
# ’hawaii_n3’, ’mbay_n4’, ’n_ind2’, ’southwest_asia’, ’useast’, ’cen_amer’, ’centam’, ’eqam’, ’europe3’, ’indo’, ’mideast’, ’nwatl’, ’southwest_asia2’, ’was3’, ’cen_america’, ’centio’, ’eqam_comp’, ’fukushima’, ’mbay_n1’, ’MREA04’, ’nwpac’, ’southwest_asia2_n3’, ’w_atl’, ’cencoos_n1’, ’eafr’, ’euro’, ’hawaii’, ’mbay_n2’, ’nepac’, ’socal’, ’southwest_asia3’, ’w_pac’, ’cencoos_n2 ’, ’e_pac’, ’europe’, ’hawaii_comp’, ’mbay_n3’, ’n_ind’, ’somalia’, ’test2’, ’w_pac2’]
# parser = argparse.ArgumentParser() parser.add_argument("-r", "--region",
# type=str,
# help="COAMPS region of interest", choices=valid_choices , required=True)
# parser.add_argument("-s", "--start-date", type=str,
# help="data start date YYYYMMDD", required=True,
# dest="str_date")
# parser.add_argument("-e", "--end-date", type=str,
# help="data end date YYYMMDD", required=True, dest="end_date")
# parser.add_argument("-d", "--datadir", type=str,
# help="DATADIR = data directory", default=’/u/NOGAPS/COAMPSg’, required=True,
# dest="datadir")
# args = parser.parse_args() region = args.region str_date = args.str_date end_date = args.end_date if str_date > end_date:
# sys.exit(’input error: {} occurs before {}’ .format(end_date, str_date))
# datadir = args.datadir
# logfilename = (’coamps_’+region+’_’+str_date+’_’+end_date+’_’
# + date.today().strftime(’%Y%h%d’)+’.log’)
# logging.basicConfig(filename=logfilename, level=logging.INFO, format=’%(levelname)s:%(message)s’)
# logging.info(’Command line arguments: %s %s %s’, sys.argv[1], sys.argv[2], sys.argv[3])
# coampsdata = get_modelinfo(region) nx = coampsdata[’nx’]
# ny = coampsdata[’ny’]
# minlon = coampsdata[’minlon’] maxlon = coampsdata[’maxlon’]
# 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371

# Delft3D-FM simulation configuration and input file generation 41
 
# minlat = coampsdata[’minlat’] maxlat = coampsdata[’maxlat’]
# # note that ’pres’ is only used for getting dates
# files = list_inputfiles(region, str_date, end_date, datadir, ’pres’) if len(files) > 0:
# print(’Available range of files:’) print(str(files[0]).split(’.’)[1]) print(str(files[-1]).split(’.’)[1])
# else:
# sys.exit(’No files found, check dates and region’)
# pars = [’pres’, ’wnd_ucmp’, ’wnd_vcmp’]
# for var in pars:
# if var == ’pres’:
# metfile = Path(’./coamps_’+str_date+’_’+end_date+’_’+region+’.amp’) write_met_header(var, metfile, nx, ny, minlon, maxlon,
# minlat, maxlat)
# files = list_inputfiles(region, str_date, end_date, datadir, var)
# # fileTime = files[0].split(’.’)[1]
# fileTime = str(files[0]).split(’.’)[1]
# refTime = datetime.strptime(fileTime, ’%Y%m%d%H’) for f in files:
# logging.info(’****Reading coamps file %s****’, f)
# append_met_data(metfile, refTime, f, nx, ny) elif var == ’wnd_ucmp’:
# metfile = Path(’./coamps_’+str_date+’_’+end_date+’_’+region+’.amu’) write_met_header(var, metfile, nx, ny, minlon, maxlon,
# minlat, maxlat)
# files = list_inputfiles(region, str_date, end_date, datadir, var)
# fileTime = str(files[0]).split(’.’)[1]
# refTime = datetime.strptime(fileTime, ’%Y%m%d%H’) for f in files:
# logging.info(’****Reading coamps file %s****’, f)
# append_met_data(metfile, refTime, f, nx, ny) elif var == ’wnd_vcmp’:
# metfile = Path(’./coamps_’+str_date+’_’+end_date+’_’+region+’.amv’) write_met_header(var, metfile, nx, ny, minlon, maxlon,
# minlat, maxlat)
# files = list_inputfiles(region, str_date, end_date, datadir, var)
# fileTime = str(files[0]).split(’.’)[1]
# refTime = datetime.strptime(fileTime, ’%Y%m%d%H’) for f in files:
# logging.info(’****Reading coamps file %s****’, f) append_met_data(metfile, refTime, f, nx, ny)
# else:
# sys.exit(’Incorrect variable names’)