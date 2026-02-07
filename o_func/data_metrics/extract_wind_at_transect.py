#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:46:48 2025

@author: af
"""

import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
from tqdm import tqdm
from datetime import datetime

# Set this to your directory of wind NetCDFs
wind_dir = "/Volumes/PNC/Original_Data/UKC3/wind/oa"

# List of transect points (lon, lat) in regular coordinates
transect_points={'duddon': (np.float64(-3.35795363024909), np.float64(54.1461186699668)), 'leven': (np.float64(-3.05658044461004), np.float64(54.1012777635345)), 'kent': (np.float64(-2.90926596746237), np.float64(54.114940652221)), 'lune': (np.float64(-2.90855212288223), np.float64(53.9645952901524)), 'wyre': (np.float64(-3.00816005813706), np.float64(53.9715436830371)), 'ribble': (np.float64(-3.10929308003562), np.float64(53.714986494408)), 'mersey': (np.float64(-3.17102565366434), np.float64(53.5105320547315)), 'dee': (np.float64(-3.34869891274123), np.float64(53.4376968343351))}

# Load file paths (assumes files are named like UKC4ao_YYYYMMDD_wind.nc)
wind_files = sorted(glob.glob(os.path.join(wind_dir, "*.nc")))

# Load grid info from the first file
with xr.open_dataset(wind_files[0]) as ds:
    rotated_pole = ccrs.RotatedPole(
        pole_longitude=ds.rotated_latitude_longitude.grid_north_pole_longitude,
        pole_latitude=ds.rotated_latitude_longitude.grid_north_pole_latitude
    )
    target_crs = ccrs.PlateCarree()

    # Coordinates
    rlons = ds.grid_longitude.values
    rlats = ds.grid_latitude.values
    rlon2d, rlat2d = np.meshgrid(rlons, rlats)
    lonlat = target_crs.transform_points(rotated_pole, rlon2d, rlat2d)
    lon2d = lonlat[..., 0]
    lat2d = lonlat[..., 1]

    # Flatten for index matching
    flat_lons = lon2d.flatten()
    flat_lats = lat2d.flatten()

# Precompute grid indexes for each transect point
point_grid_idxs = []
for name, (pt_lon, pt_lat) in transect_points.items():
    dists = np.hypot(flat_lons - pt_lon, flat_lats - pt_lat)
    min_idx = np.argmin(dists)
    y_idx, x_idx = np.unravel_index(min_idx, lon2d.shape)
    point_grid_idxs.append((name, y_idx, x_idx))

# Extract and write data
for name, y_idx, x_idx in tqdm(point_grid_idxs, desc="Extracting wind data"):
    all_times = []
    all_u = []
    all_v = []

    for f in tqdm(wind_files, leave=False):
        try:
            with xr.open_dataset(f) as ds:
                time = ds["time"].values
                u = ds["x_wind"][:, y_idx, x_idx].values
                v = ds["y_wind"][:, y_idx, x_idx].values

                all_times.extend(time)
                all_u.extend(u)
                all_v.extend(v)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

    # Process and save
    times = pd.to_datetime(all_times)
    u_arr = np.array(all_u)
    v_arr = np.array(all_v)
    speed = np.sqrt(u_arr**2 + v_arr**2)
    direction = (180 + np.degrees(np.arctan2(u_arr, v_arr))) % 360  # wind FROM direction

    df_out = pd.DataFrame({
        "datetime": times,
        "u (m/s)": u_arr,
        "v (m/s)": v_arr,
        "speed (m/s)": speed,
        "direction (deg)": direction,
    })
    df_out.to_csv(f"wind_data_{name}.csv", index=False)
