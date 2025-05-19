# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:59:06 2024

@author: af
"""

import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from o_func import opsys

# Define paths
start_path = Path(opsys())
path = start_path / Path('Original_Data/UKC3/og/shelftmb_combined_to_3_layers_for_tmb')
river_path = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/original_river_data/processed')
writemaps = 'y'
side = 'easy'

# Load data
def load_data():
    Ufiles = []
    Vfiles = []
    Tfiles = []
    for file in path.glob('*.nc'):
        if str(file).endswith('T.nc'):
            Tfiles.append(file)
        elif str(file).endswith('U.nc'):
            Ufiles.append(file)
        else:
            Vfiles.append(file)
    
    Tdata = xr.open_mfdataset(Tfiles)
    Udata = xr.open_mfdataset(Ufiles)
    Vdata = xr.open_mfdataset(Vfiles)
    
    return Tdata, Udata, Vdata

Tdata, Udata, Vdata = load_data()

# Adjust time counters
vomecrty = Vdata.vomecrty
vomecrty['time_counter'] = vomecrty['time_counter'] - np.timedelta64(30, 'm')
vozocrtx = Udata.vozocrtx
vozocrtx['time_counter'] = vozocrtx['time_counter'] - np.timedelta64(30, 'm')
salinity = Tdata.vosaline 
salinity['time_counter'] = salinity['time_counter'] - np.timedelta64(30, 'm')
sh = Tdata.sossheig
sh['time_counter'] = sh['time_counter'] - np.timedelta64(30, 'm')

# Determine time range
salinity_start_time = pd.to_datetime(salinity.time_counter[0].values)
salinity_stop_time = pd.to_datetime(salinity.time_counter[-1].values)

# Estuary Dictionary
estuary_data = {
    'Dee'   : {'latlon': {'lat': 34.0522, 'lon': -118.2437}, 'xy': {'x': 779, 'y': 597}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 141},
    'Leven' : {'latlon': {'lat': 36.7783, 'lon': -119.4179}, 'xy': {'x': 783, 'y': 666}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 4},
    'Ribble': {'latlon': {'lat': 40.7128, 'lon': -74.0060},  'xy': {'x': 788, 'y': 631}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 84},
    'Lune'  : {'latlon': {'lat': 40.7128, 'lon': -74.0060},  'xy': {'x': 790, 'y': 651}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 17},
    'Mersey': {'latlon': {'lat': 40.7128, 'lon': -74.0060},  'xy': {'x': 782, 'y': 612}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 158},
    'Wyre'  : {'latlon': {'lat': 40.7128, 'lon': -74.0060},  'xy': {'x': 785, 'y': 647}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 180},
    'Kent'  : {'latlon': {'lat': 40.7128, 'lon': -74.0060},  'xy': {'x': 790, 'y': 666}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 48},
    'Duddon': {'latlon': {'lat': 40.7128, 'lon': -74.0060},  'xy': {'x': 776, 'y': 668}, 'width_at_mouth': 14000, 'height_at_mouth': 5, 'h_l': 5/2, 'angle': 63},
}

# Load river data
def load_river_data(start_time, stop_time):
    full_time_index = pd.date_range(start=start_time, end=stop_time, freq='H')

    for i, file in enumerate(river_path.glob('*.csv')):
        river_name = file.name.split('_')[1]
        print(f'Processing river data for ...{river_name}')
        df = pd.read_csv(file, header=0, parse_dates=[0], dayfirst=True)
        df.columns = ['Date', 'Discharge']
        df['Date'] = pd.to_datetime(df['Date'])
        mask = (df['Date'] >= start_time) & (df['Date'] <= stop_time)
        filtered_data = df.loc[mask]
        filtered_data.set_index('Date', inplace=True)
        filtered_data = filtered_data.reindex(full_time_index)
        filtered_data['Discharge'].interpolate(method='linear', inplace=True)
        filtered_data['Discharge'].ffill(inplace=True)
        filtered_data['Discharge'].bfill(inplace=True)

        if river_name in estuary_data:
            estuary_data[river_name]['time'] = filtered_data.index.to_numpy()
            estuary_data[river_name]['discharge'] = filtered_data['Discharge'].to_numpy()
        else:
            print(f"River name {river_name} not found in dictionary.")
            estuary_data[river_name] = {'time': filtered_data.index.to_numpy(), 'discharge': filtered_data['Discharge'].to_numpy()}

    return estuary_data
    
start_time = salinity_start_time
stop_time = salinity_stop_time
estuary_data = load_river_data(start_time, stop_time)

def extract_with_buffer(data, x, y, rofi_buffer):
    # Ensure the buffer does not exceed data boundaries
    x_start = max(x - rofi_buffer, 0)
    x_end = min(x + rofi_buffer + 1, data.sizes['x'])
    y_start = max(y - rofi_buffer, 0)
    y_end = min(y + rofi_buffer + 1, data.sizes['y'])
    
    new_data = data.sel(x=slice(x_start, x_end-1,1), y=slice(y_start, y_end-1,1))
                   #.sel(x=slice(x_start, x_end-1,1), y=slice(y_start, y_end-1,1))
    # we apply the minus 1 at the end to correct the slicing, plotted the centre cell etc as bright to identify. 
    # Extract the data with buffer zone
    return new_data

# Extract data and attach to estuary_data
def extract_and_attach_data(estuary):
    print(f'Extract for ...{estuary}')
    est = estuary_data[estuary]
    rofi_buffer = 8

    depth_coords = np.array([0, est['height_at_mouth']/2, est['height_at_mouth']]) if vomecrty.deptht.size == 3 else None
    x, y = est['xy']['x'], est['xy']['y']

    est['mll'] = est['height_at_mouth'] - (est['h_l'] / 2)
    est['mul'] = est['h_l'] / 2

    vomy = extract_with_buffer(data=vomecrty, x=x, y=y, rofi_buffer=rofi_buffer)
    vomx = extract_with_buffer(data=vozocrtx, x=x, y=y, rofi_buffer=rofi_buffer)

    est['vel_magnitude'] = np.sqrt(vomy**2 + vomx**2)
    est['vozocrtx'] = vomx.mean(dim=['deptht', 'y', 'x'], skipna=True)
    est['vomecrty'] = vomy.mean(dim=['deptht', 'y', 'x'], skipna=True)

    theta = np.arctan2(np.array(est['vomecrty']), np.array(est['vozocrtx']))
    theta_degrees = np.degrees(theta)
    relative_angle = theta_degrees - est['angle']

    est['salinity'] = extract_with_buffer(data=salinity, x=x, y=y, rofi_buffer=rofi_buffer)
    est['sh'] = extract_with_buffer(data=sh, x=x, y=y, rofi_buffer=rofi_buffer)

    depth_diff = np.abs(depth_coords[:, np.newaxis, np.newaxis, np.newaxis] - est['mll']) if depth_coords is not None else None
    depth_weights = (1 / (depth_diff + 1e-10)) if depth_diff is not None else None
    depth_weights /= depth_weights.sum(axis=0, keepdims=True) if depth_weights is not None else None
    depth_weights = np.broadcast_to(depth_weights, (3, 720, 17, 17)) if depth_weights is not None else None

    weighted_sal = (est['salinity'] * depth_weights).sum(dim='deptht') if depth_weights is not None else est['salinity']
    weighted_vel = (est['vel_magnitude'] * depth_weights).sum(dim='deptht') if depth_weights is not None else est['vel_magnitude']

    est['S_l'] = weighted_sal.mean(dim=['y', 'x'], skipna=True)
    avg_weighted_vel = weighted_vel.mean(dim=['y', 'x'], skipna=True)

    signed_magnitude = np.where(
        (relative_angle >= -90) & (relative_angle <= 90),
        avg_weighted_vel,
        -avg_weighted_vel
    )

    est['Q_l'] = signed_magnitude
    est['S_col'] = est['salinity'].mean(dim=['deptht', 'y', 'x'], skipna=True)

    estuary_data[estuary] = est

def parallel_extract_and_attach():
    with ProcessPoolExecutor() as executor:
        executor.map(extract_and_attach_data, estuary_data.keys())

parallel_extract_and_attach()