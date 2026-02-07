#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:20:53 2025

@author: af
"""
import xarray as xr
import numpy as np
file = '/Volumes/PNC/Original_Data/UKC3/river_climatology/rivers/AMM15_River_Climatology.nc'

ds = xr.open_dataset(file)

# Take the first time slice (face)
slice0 = ds['rorunoff'].isel(time_counter=0)

# Build mask: not NaN and not zero
mask = (~np.isnan(slice0)) & (slice0 != 0)

# Count how many cells satisfy this condition
count = int(mask.sum().values)

print(f"Number of river cells: {count}")
