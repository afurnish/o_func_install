#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Turn a dataarray into a netcdf file

read this:
    https://stackoverflow.com/questions/60173355/xarray-create-dataset-from-list-of-lat-lon-points-not-square

Created on Thu Nov 10 12:42:58 2022
@author: af
"""
test = r'/Volumes/PD/GitHub/python-oceanography/Delft 3D FM Suite 2019/functions'
import xarray as xr
import numpy as np

def nc_make():
    data = xr.DataArray(np.random.randn(2, 3), dims=("Lat", "Lon"), coords={"x": [10, 20]})


    # path is the loaction and name of the new netcdf file
    val = data
    return val
    
   
a = nc_make()
print(a)