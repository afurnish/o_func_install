#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Estuary Box Model program 

designed to run from 3 layer 3d simulations but can also generate 
synthetic timeseries of your choosing to run other simulations. 
Created on Wed Aug 14 13:59:06 2024
@author: af
"""
import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from datetime import datetime
import os
from o_func import opsys; start_path = Path(opsys())

def time64(time_str):
    if ' ' not in time_str:
        time_str += ' 00:00'
    try:
        valid_date = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
    except ValueError:
        raise ValueError(f"Invalid date format: {time_str}")
    return np.datetime64(time_str)

path = start_path / Path('Original_Data/UKC3/og/shelftmb_combined_to_3_layers_for_tmb')
river_path = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/original_river_data/processed')
savepath = start_path / Path('modelling_DATA/EBM_PRIMEA/EBM_python/figures')
storage_location = start_path / Path('modelling_DATA/EBM_PRIMEA/EBM_python/simulation_results')
os.makedirs(storage_location, exist_ok=True)
writemaps = 'n'
side = 'east'
include_tidal = True

artificial_river = 'y' # implement real or not real data
artificial_tide = 'n'
if artificial_tide == 'y':
    start_time = time64('2013-11-01')
    stop_time  = time64('2013-11-30')
# If not real data, what do you want to implement for all estuaries.
discharge = 100
tide_cons = ['M2', 'S2']

#%%  Estuary Dictionary
# These coords are pulled from the river location discharge points, 
# primarily for caqlculating tidal predictions as needed. 
correct_coords = {
    'Dee': (-3.118638742308569, 53.24982016910892),
    'Duddon': (-3.230547161208941, 54.25887801158542),
    'Kent': (-2.811861321397053, 54.25064484652686),
    'Leven': (-3.052120073154467, 54.23186185336646),
    'Lune': (-2.840884669179119, 54.03655050082423),
    'Mersey': (-2.768434835109615, 53.34491510325321),
    'Ribble': (-2.811633371553361, 53.74817881546817),
    'Wyre': (-2.955520867395822, 53.85663354235163)
}



estuary_data = {
    'Dee': { 
        'latlon'         : {'lat': 34.0522, 'lon': -118.2437},
        'xy'             : {'x': 779, 'y': 597},
        'length'         : 10000, # Estuary Length (m)
        'width_at_mouth' : 8000,  # Estuary Width (m)
        'height_at_mouth': 4.75,     # Estuary Height (m)
        'h_l'            : 4.75/2,
        'angle'          : 141,
    },
    'Leven': {
        'latlon'         : {'lat': 36.7783, 'lon': -119.4179},
        'xy'             : {'x': 783, 'y': 666},
        'length'         : 16150, # Estuary Length (m)
        'width_at_mouth' : 4860,  # Estuary Width (m)
        'height_at_mouth': 5,      # Estuary Height (m)
        'h_l'            : 5/2,
        'angle'          : 4,
    },
    'Ribble': {
        'latlon'         : {'lat': 40.7128, 'lon': -74.0060},
        'xy'             : {'x': 788, 'y': 631},
        'length'         : 13500, # Estuary Length (m)
        'width_at_mouth' : 10000,  # Estuary Width (m)
        'height_at_mouth': 7,      # Estuary Height (m)
        'h_l'            : 7/2,
        'angle'          : 84,
    },
    'Lune': {
        'latlon'         : {'lat': 40.7128, 'lon': -74.0060},
        'xy'             : {'x': 790, 'y': 651},
        'length'         : 10280, # Estuary Length (m)
        'width_at_mouth' : 540,  # Estuary Width (m)
        'height_at_mouth': 0.1,      # Estuary Height (m)
        'h_l'            : 0.1/2,
        'angle'          : 17,
    },
    'Mersey': {
        'latlon'         : {'lat': 40.7128, 'lon': -74.0060},
        'xy'             : {'x': 782, 'y': 612},
        'length'         : 16090, # Estuary Length (m)
        'width_at_mouth' : 1770,  # Estuary Width (m)
        'height_at_mouth': 6.25,      # Estuary Height (m)
        'h_l'            : 6.25/2,
        'angle'          : 158,
    },
    'Wyre': {
        'latlon'         : {'lat': 40.7128, 'lon': -74.0060},
        'xy'             : {'x': 785, 'y': 647},
        'length'         : 16090, # Estuary Length (m)
        'width_at_mouth' : 530,   # Estuary Width (m)
        'height_at_mouth': 0.6,   # Estuary Height (m)
        'h_l'            : 0.6/2,
        'angle'          : 180,
    },
    'Kent': {
        'latlon'         : {'lat': 40.7128, 'lon': -74.0060},
        'xy'             : {'x': 790, 'y': 666},
        'length'         : 12000, # Estuary Length (m)
        'width_at_mouth' : 10500,  # Estuary Width (m)
        'height_at_mouth': 5,      # Estuary Height (m)
        'h_l'            : 5/2,
        'angle'          : 48,
    },
    'Duddon': {
        'latlon'         : {'lat': 40.7128, 'lon': -74.0060},
        'xy'             : {'x': 776, 'y': 668},
        'length'         : 12000,  # Estuary Length (m)
        'width_at_mouth' : 4900,  # Estuary Width (m)
        'height_at_mouth': 3,      # Estuary Height (m) a guess 
        'h_l'            : 3/2,
        'angle'          : 63,
    }
}

for estuary_name, coords in correct_coords.items():
    if estuary_name in estuary_data:
        lon, lat = coords
        estuary_data[estuary_name]['latlon'] = {'lat': lat, 'lon': lon}

#%% EBM 
def ensure_first_dim_is_one(array):
    if array.shape[0] != 1:
        return array.T if array.shape[1] == 1 else array[np.newaxis, :]
    return array

# def calculate_segment_means(salinity_series, discharge_series, time_series):
#     segment_salinity_means = []
#     segment_discharge_means = []
#     segment_end_times = []
    
#     start_idx = 0

#     while start_idx < len(salinity_series):
#         # Find the next NaN or the end of the series
#         end_idx = start_idx
#         while end_idx < len(salinity_series) and not np.isnan(salinity_series[end_idx]):
#             end_idx += 1

#         # Calculate means for the segment between start_idx and end_idx
#         if end_idx > start_idx:
#             segment_salinity_mean = salinity_series[start_idx:end_idx].mean()
#             segment_discharge_mean = discharge_series[start_idx:end_idx].mean()
            
#             segment_salinity_means.append(segment_salinity_mean)
#             segment_discharge_means.append(segment_discharge_mean)
            
#             # Store the end time of the segment
#             segment_end_times.append(time_series[end_idx - 1])
        
#         # Move to the next segment
#         start_idx = end_idx + 1

#     return segment_salinity_means, segment_discharge_means, segment_end_times

def calculate_segment_means(salinity_series, discharge_series, time_series):
    segment_salinity_means = []
    segment_discharge_means = []
    segment_end_times = []
    
    start_idx = 0

    while start_idx < len(salinity_series):
        # Find the next NaN or the end of the series
        end_idx = start_idx
        while end_idx < len(salinity_series) and not np.isnan(salinity_series[end_idx]):
            end_idx += 1

        # Calculate means for the segment between start_idx and end_idx
        if end_idx > start_idx:
            segment_salinity_mean = salinity_series[start_idx:end_idx].mean()
            segment_discharge_mean = discharge_series[start_idx:end_idx].mean()
            
            segment_salinity_means.append(segment_salinity_mean)
            segment_discharge_means.append(segment_discharge_mean)
            
            # Store the end time of the segment
            segment_end_times.append(time_series[end_idx - 1])
        
        # Move to the next segment
        start_idx = end_idx + 1

    return segment_salinity_means, segment_discharge_means, segment_end_times

def ebm(W_m, h, Q_r, Q_m, S_l, Q_l, S_oc, length, time_array):
    """ EBM Parameters
    ----------
    W_m : float
        Estuary width (m).
    h : float
        Estuary height (m).
    Q_r : float
        Upper layer river volume flux at head.
    Q_l : float
        Lower layer volume flux (m**3/s, positive if entering). Timeseries hourly
    Q_tide : float
        Tidal volume flux.
    S_l : float
        Lower layer ocean salinity. ()
    Ro_s : float
        Lower layer ocean water density.
    S_oc : float
        Ocean salinity at mouth.
    L_e : float
        Constant estuary length (m), deprecated field.
    Lx : float
        Time-dependent estuary length (km).
    Fr : float
        River flow Froude number.
    Fr_box : float
        Estuary Froude number.
    Eta : float
        Estuary self-similar character.
    vel_tide : float
        Barotropic tidal zonal velocity (m/s, positive if eastward).
    k_x : float
        Horizontal mixing coefficient for tracers (m**2/s).
    Q_m : float
        Mean river discharge.
    Q_u : float
        Upper layer volume flux (m**3/s, positive).
    S_u : float
        Salinity at estuary upper layer (ppt).
    ur : float
        River flow velocity.
    C_k : float
         Box coefficient into parameters for a Python function.

     Returns
    -------
    Q_u : float
        Upper layer volume flux (m**3/s, positive).
    S_u : float
        Salinity at estuary upper layer (ppt).
    ur : float
        River flow velocity.
    C_k : float
        Box coefficient into parameters for a Python function.
    """
    # S_r = 0
    # area = W_m * h
    # volume = W_m * h * length
    # g = 9.81
    # #!!-----------------CMCC model x estuary box----------------------
    # #!!---------------------------------------------------------------
    # #!CMCC EBM v1: set k_x following MacCready 2010 (ref. Banas et al 2004)
    # #!   k_x = W_m*vel_tide*0.035
    # #! print*,'k_x',k_x

    # ur = Q_r/((h/2)*W_m)
    # vel_tide = Q_l / area
    
    # #print('ur',ur)
    # Fr=Q_r/((h/2)*W_m*((h/2)*g)**0.5)
    # #print('Fr',Fr)
    # #print('S_l',S_l)
    # Ro_s = 1000*(1+((7.7*(1E-4)) * S_l))
    # #print('Ro_s',Ro_s)
    
    # #Fr_box = vel_tide/((h*g)**0.5)
    # # We have to handle volume flux into velocity. 
    # Fr_box = vel_tide / ( (h*g)**0.5 )
    # #print('Fr_box',Fr_box)
    # #print('vel_tide',vel_tide)
    # Eta = ((Fr_box)**2)*(W_m/h)
    
    # # Python logic to handle this part of the equation
    # mask = Q_r > Q_m
    # C_k = np.zeros_like(Q_r)
    # #!!! --C_k dimensional equation--!!!
    
    # '''
    # Testing C_k calinration 
    
    # attempt to use absolute values for vel tide in equation, 
    
    # np.exp(-2000 * Eta[~mask] this multiplied by 200 gives the problem curve. 
    # plotted S-U against the bad C_k and the function from above and they all line up nicely. 
           
    # '''
    
    
    # # Qr > Qm : Apply the first equation where the condition is true
    # C_k[mask] = ((Ro_s[mask] / 1000)**20) * (((vel_tide[mask]) / ur[mask])**-0.5) * np.exp(-2000 * Eta[mask]) # plotted everything from here agsint ck and Su and nothing lines up
    # # Else: Apply the second equation where the condition is false (Q_r <= Q_m)
    # C_k[~mask] =  ((Ro_s[~mask] / 1000)**20) * ((vel_tide[~mask]) / ur[~mask])**0.1 * np.exp(-2000 * Eta[~mask])
    # #print('C_k',C_k)
    #     #C_k = 100
        
        

    # #!!!--Compute horiz. diff coeff--!!!
    # k_x = W_m*vel_tide*C_k
    # #print('k_x',k_x)

    # #!!!--volume flux conservation equation--!!!
    # Q_u = Q_r + Q_l #+(h*vel_tide*W_m) # Q_r + Q_l 

    # #print('Q_u',Q_u)

    # #!!!--Lx dimensional equation--!!!
    # #Lx=h*0.019*(Fr**-0.673)*((Ro_s/1000)**108.92)*((Q_tide/Q_r)**0.075)*((Q_l/Q_r)**-0.0098)
    # Lx=h*0.019*(Fr**-0.673)*((Ro_s/1000)**108.92)*( (Q_l/Q_r)**-0.0098)

    # #Lx=20 !uom: km

    # #!!!--salt flux conservation equation--!!!
    
    # # old equation 
    # #S_u=(S_l*Q_l+(S_oc)+k_x*h*W_m*(S_oc/(Lx*1000)) )/(Q_r+Q_l+h*vel_tide*W_m)

    # # New section to handle flushing times
    # S_u = np.full_like(Q_l, np.nan)
    # S_ebb = np.full_like(Q_l, np.nan)
    
    
    # previous_S_u = np.nan  # To store the previous flood tide salinity
    
    # # Loop through each time step
    # for i in range(len(Q_l)):
    #     if Q_l[i] > 0:  # Inflow (flood tide)
    #         S_u[i] = (S_l[i] * Q_l[i] + S_oc[i] + k_x[i] * h * W_m * (S_oc[i] / (Lx[i] * 1000))) / (Q_r[i] + Q_l[i] ) # DOes this bit -> need to be here #+ h * vel_tide[i] * W_m)
    #         previous_S_u = S_u[i]  # Store the flood tide salinity for the next ebb tide
    #     elif Q_l[i] < 0:  # Outflow (ebb tide)
    #         if not np.isnan(previous_S_u):  # Check if there is a previous flood tide salinity
    #             S_ebb[i] = (previous_S_u * abs(Q_l[i]) + S_r * Q_r[i]) / (abs(Q_l[i]) + Q_r[i])

    # # Mask Q_l where S_ebb is NaN to create Q_outflow
    # Q_outflow = np.where(~np.isnan(S_ebb), np.abs(Q_l), np.nan)
    # # Convert arrays to Pandas Series for easier handling of segments
    # S_ebb_series = pd.Series(S_ebb)
    # S_u_series = pd.Series(S_u)
    # Q_outflow_series = pd.Series(Q_outflow)
    # Q_inflow_series = pd.Series(np.where(Q_l > 0, Q_l, np.nan))  # For inflow (flood tide)
    
    # time = S_l.time_counter
  
    # # Calculate segment means for inflow and outflow
    # salinity_in_means, discharge_in_means, time_series_in = calculate_segment_means(S_u_series, Q_inflow_series, time)
    # salinity_out_means, discharge_out_means, time_series_out = calculate_segment_means(S_ebb_series, Q_outflow_series, time)

    # Flushing_Time = []
    # Flushing_Time_Phase = [] # records at the end of the out segment. 
    # # Calculate flushing time for each tidal cycle
    # for sal_in_mean, sal_out_mean, dis_out_mean, end_time in zip(salinity_in_means, salinity_out_means, discharge_out_means, time_series_out):
    #     if not np.isnan(sal_in_mean) and not np.isnan(sal_out_mean) and not np.isnan(dis_out_mean):
    #         ft = volume * (sal_in_mean - sal_out_mean) / (dis_out_mean * sal_in_mean)
    #         Flushing_Time.append(ft / 3600)  # Convert seconds to hours
    #         Flushing_Time_Phase.append(end_time)
            
    # cleaned_flushing_time_phase = [np.datetime64(str(time.values)) if isinstance(time, xr.DataArray) else np.datetime64(time) for time in Flushing_Time_Phase]
  
    # # fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # # # Plot Flushing Time on the primary y-axis
    # # ax1.scatter(cleaned_flushing_time_phase, Flushing_Time, marker='o', color='b', label='Flushing Time')
    # # ax1.set_xlabel('Time (End of Ebb Tide)')
    # # ax1.set_ylabel('Flushing Time (hours)', color='b')
    # # ax1.tick_params(axis='y', labelcolor='b')
    # # ax1.grid(True)
    
    # # # Create a secondary y-axis sharing the same x-axis for Q_u
    # # ax2 = ax1.twinx()
    # # ax2.plot(est['time'], Q_u, color='r', label='Q_u')
    # # ax2.set_ylabel('Q_u (m³/s)', color='r')
    # # ax2.tick_params(axis='y', labelcolor='r')
    
    # # # Create a third y-axis sharing the same x-axis for discharge
    # # ax3 = ax1.twinx()
    
    # # # Offset the third axis to avoid overlap
    # # ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward by 60 points
    # # ax3.plot(est['time'], est['discharge'], color='g', label='Discharge')
    # # ax3.set_ylabel('Discharge (m³/s)', color='g')
    # # ax3.tick_params(axis='y', labelcolor='g')
    
    # # # Add titles and legends
    # # fig.suptitle('Flushing Time, Q_u, and Discharge Over Time')
    # # ax1.legend(loc='upper left')
    # # ax2.legend(loc='upper center')
    # # ax3.legend(loc='upper right')
    # # plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    # # plt.tight_layout()
    # # plt.savefig(savepath / Path(estuary), dpi = 500)
        
    # #! Lower layer salinity * Lower Layer volume flux + Ocean Salinity at mouth. 
    # # Compute the fisher flow number to access whether any of this is suitable or not
    # Fi = ( Q_l /(area/2) )/( Q_r/(area/2) ) # Fi = tidal velocity divided by the river velocity. 
    # #print('Fi')
    # #print('S_u',S_u)
    # #print('Lx',Lx)
    
    # return Q_u, Lx, S_u, Fi, Flushing_Time, cleaned_flushing_time_phase, S_ebb
   
    S_r = 0
    g = 9.81
    area = W_m * h
    volume = W_m * h * length
 
    ur = Q_r / ((h / 2) * W_m)
    vel_tide = Q_l / area
    
    
    Fr = Q_r / ((h / 2) * W_m * ((h / 2) * g)**0.5)
    Ro_s = 1000 * (1 + (7.7e-4 * S_l))
    Fr_box = vel_tide / ((h * g) ** 0.5)
    Eta = (Fr_box ** 2) * (W_m / h)
 
    tolerance = 1e-5  # Small positive value
    mask = Q_r > (Q_m - tolerance)
    
    C_k = np.zeros_like(Q_r)
    
    C_k[mask] = ((Ro_s[mask] / 1000) ** 20) * ((vel_tide[mask] / ur[mask]) ** -0.5) * np.exp(-2000 * Eta[mask])
    C_k[~mask] = 200*((Ro_s[~mask] / 1000) ** 20) * ((vel_tide[~mask] / ur[~mask]) ** 0.1) * np.exp(-2000 * Eta[~mask])
    
    k_x = W_m * vel_tide * C_k
    
    Q_u = Q_r + Q_l # Q_l volume 
    
    Lx = h * 0.019 * (Fr ** -0.673) * ((Ro_s / 1000) ** 108.92) * ((Q_l / Q_r) ** -0.0098)
    
    S_u = np.full_like(Q_l, np.nan)
    S_ebb = np.full_like(Q_l, np.nan)
    
    previous_S_u = np.full((8,), np.nan)  # Array to store the previous flood tide salinity for each location
    
    S_l[np.isfinite(S_l)] = 30
    # Apply constant salinities. say set S_l to 25. 
    # Loop through each time step
    for i in range(len(Q_l)):
        inflow_mask = Q_l[i] > 0
        outflow_mask = Q_l[i] < 0
        
        if np.any(inflow_mask):
            S_u[i, inflow_mask] = (S_l[i, inflow_mask] * Q_l[i, inflow_mask] + S_oc[i, inflow_mask] + 
                                   k_x[i, inflow_mask] * h[:, inflow_mask].flatten() * W_m[:, inflow_mask].flatten() * 
                                   (S_oc[i, inflow_mask] / (Lx[i, inflow_mask] * 1000))) / (Q_r[i, inflow_mask] + Q_l[i, inflow_mask])
            previous_S_u[inflow_mask] = S_u[i, inflow_mask]
 
        if np.any(outflow_mask):
            S_ebb[i, outflow_mask] = (previous_S_u[outflow_mask] * abs(Q_l[i, outflow_mask]) + S_r * Q_r[i, outflow_mask]) / (abs(Q_l[i, outflow_mask]) + Q_r[i, outflow_mask])
 
    # Mask Q_l where S_ebb is NaN to create Q_outflow
    Q_outflow = np.where(~np.isnan(S_ebb), np.abs(Q_l), np.nan)
 
    # Initialize lists to store the results for each estuary
    flushing_time_list = []
    cleaned_flushing_time_phase_list = []
 
    
    est_names = [i for i in estuary_data.keys()]
    dis = str(discharge)

    # Copy nan logic onto this data. 
    S_l_nan_mask = np.isnan(S_u)
    S_l_nans = S_l
    S_l_nans[S_l_nan_mask] = np.nan
    
    Q_r_estuary = Q_r 
    # Loop through each location (estuary) and calculate flushing time
    for loc in range(S_l_nans.shape[1]):
        S_ebb_series = pd.Series(S_ebb[:, loc])
        S_l_series = pd.Series(S_l_nans[:, loc]) # what if this is meant to be S_l
        Q_outflow_series = pd.Series(Q_outflow[:, loc])
        Q_inflow_series = pd.Series(np.where(Q_l[:, loc] > 0, Q_l[:, loc], np.nan))
        Q_r_series = pd.Series(Q_r[:, loc])

        # Calculate segment means for inflow and outflow
        salinity_in_means, discharge_in_means, time_series_in = calculate_segment_means(S_l_series, Q_inflow_series, time_array)
        salinity_out_means, discharge_out_means, time_series_out = calculate_segment_means(S_ebb_series, Q_outflow_series, time_array)
        _, river_discharge_means, _ = calculate_segment_means(S_ebb_series, Q_r_series, time_array)
        dis_out_mean_corrected = [estuary_out + river_out for estuary_out, river_out in zip(discharge_out_means, river_discharge_means)]

        Flushing_Time = []
        Flushing_Time_Phase = []
 
        # Calculate flushing time for each tidal cycle # was dis_out_mean_correct
        for sal_in_mean, sal_out_mean, dis_out_mean, end_time in zip(salinity_in_means, salinity_out_means, river_discharge_means, time_series_out):
            if not np.isnan(sal_in_mean) and not np.isnan(sal_out_mean) and not np.isnan(dis_out_mean):
                ft = (volume[0,loc] * (sal_in_mean - sal_out_mean)) / (dis_out_mean * sal_in_mean)
                Flushing_Time.append(ft / 3600)  # Convert seconds to hours
                Flushing_Time_Phase.append(end_time)
 
                flushing_time_list.append(Flushing_Time)
                cleaned_flushing_time_phase_list.append(Flushing_Time_Phase)
 
    
        
     
    
        new_name = est_names[loc]
        # Save results for this estuary
        filename = f"{new_name}_discharge_{dis}.npz"
        filepath = os.path.join(storage_location, filename)
        np.savez(filepath,
                 sal_in_mean=salinity_in_means,
                 sal_out_mean=salinity_out_means,
                 dis_out_mean=dis_out_mean_corrected,
                 flushing_time=Flushing_Time,
                 flushing_time_phase=Flushing_Time_Phase)
    Fi = (Q_l / (area / 2)) / (Q_r / (area / 2))  # Compute the Fisher flow number element-wise
 
    return Q_u, Lx, S_u, Fi, flushing_time_list, cleaned_flushing_time_phase_list, S_ebb
    
#%% Collecting data paths and loading it into xarray. 
def load_tidal_data():
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
        
    # Load lazaily using dask into xarray format
    Tdata = xr.open_mfdataset(Tfiles, combine='by_coords', engine='h5netcdf', chunks={'time_counter': 100})
    Udata = xr.open_mfdataset(Tfiles, combine='by_coords', engine='h5netcdf', chunks={'time_counter': 100})
    Vdata = xr.open_mfdataset(Tfiles, combine='by_coords', engine='h5netcdf', chunks={'time_counter': 100})

    #% Speed up data for operations by converting everything to Numpy.
    # Example setup: Define the points of interest (replace with your actual points)
    # Doesnt need to access direct data locations at this point in time. 
    # Define constants
    rofi_buffer = 8  # Buffer to put data selection away from edge of land
    sh_buffer = 0    # Buffer for sea height extraction (or similar)
    
    # Example setup: Define the points of interest (replace with your actual points)
    x_coords = np.array([data['xy']['x'] for data in estuary_data.values()])
    y_coords = np.array([data['xy']['y'] for data in estuary_data.values()])
    
    # Create buffer ranges for both buffers
    rofi_range = np.arange(-rofi_buffer, rofi_buffer + 1)
    sh_range = np.arange(-sh_buffer, sh_buffer + 1)
    
    # Create grids for both buffers
    rofi_x_offsets, rofi_y_offsets = np.meshgrid(rofi_range, rofi_range, indexing='ij')
    sh_x_offsets, sh_y_offsets = np.meshgrid(sh_range, sh_range, indexing='ij')
    
    
    # Apply offsets to the x and y coordinates
    rofi_x_indices = (x_coords[:, np.newaxis, np.newaxis] + rofi_x_offsets).clip(0, Udata.sizes['x'] - 1)
    rofi_y_indices = (y_coords[:, np.newaxis, np.newaxis] + rofi_y_offsets).clip(0, Udata.sizes['y'] - 1)
    
    sh_x_indices = (x_coords[:, np.newaxis, np.newaxis] + sh_x_offsets).clip(0, Udata.sizes['x'] - 1)
    sh_y_indices = (y_coords[:, np.newaxis, np.newaxis] + sh_y_offsets).clip(0, Udata.sizes['y'] - 1)
    
    # Open datasets with Dask to handle large files
    Tdata = xr.open_mfdataset(Tfiles, combine='by_coords', engine='h5netcdf', chunks={'time_counter': 100})
    Udata = xr.open_mfdataset(Ufiles, combine='by_coords', engine='h5netcdf', chunks={'time_counter': 100})
    Vdata = xr.open_mfdataset(Vfiles, combine='by_coords', engine='h5netcdf', chunks={'time_counter': 100})
    
    # Define the variables to extract
    T_variables = ['vosaline', 'sossheig']
    U_variables = ['vozocrtx']
    V_variables = ['vomecrty']
    
    # Extract only the variables of interest
    Tdata_subset = Tdata[T_variables]
    Udata_subset = Udata[U_variables]
    Vdata_subset = Vdata[V_variables]
    
    start_time = time.time()
    
    # Use advanced indexing to extract data
    sossheig_buffer = Tdata_subset['sossheig'].isel(x=(('points', 'x', 'y'), sh_x_indices),
                                                    y=(('points', 'x', 'y'), sh_y_indices)).compute().values
    sal_buffer = Tdata_subset['vosaline'].isel(x=(('points', 'x', 'y'), rofi_x_indices),
                                               y=(('points', 'x', 'y'), rofi_y_indices)).compute().values
    velU_buffer = Udata_subset['vozocrtx'].isel(x=(('points', 'x', 'y'), rofi_x_indices),
                                                y=(('points', 'x', 'y'), rofi_y_indices)).compute().values
    velV_buffer = Vdata_subset['vomecrty'].isel(x=(('points', 'x', 'y'), rofi_x_indices),
                                                y=(('points', 'x', 'y'), rofi_y_indices)).compute().values
    t_raw  = Tdata_subset['time_counter'].values
    
    # For some reason the data is flipped, however it doesnt matter but this fixes it. 
    sal_buffer_corrected  = np.flip(np.rot90(sal_buffer,  k=1, axes=(3, 4)), axis=3)
    velU_buffer_corrected = np.flip(np.rot90(velU_buffer, k=1, axes=(3, 4)), axis=3)
    velV_buffer_corrected = np.flip(np.rot90(velV_buffer, k=1, axes=(3, 4)), axis=3)
    
    
    # Compute the magnitude of U and V 
    vel_magnitude =  np.sqrt(velU_buffer_corrected**2 + velV_buffer_corrected**2)
    
    # Take the means of the buffer zones. 
    vozocrtx_mean = np.nanmean(velU_buffer_corrected, axis=(0, 3, 4))
    vomecrty_mean = np.nanmean(velV_buffer_corrected, axis=(0, 3, 4))
    # Calculate the resultant angle of the magnitudes. 
    theta = np.arctan2(vomecrty_mean, vozocrtx_mean)
    theta_degrees = np.degrees(theta)
    angles = [estuary_data[i]['angle'] for i in estuary_data]
    relative_angle = theta_degrees - angles
    
    # Preparing depth averaged calculations. 
    if Tdata_subset.deptht.size == 3:
        ham = [estuary_data[i]['height_at_mouth'] for i in estuary_data]
        depth_coords = np.array([np.array([0, i/2 , i ]) for i in ham])
    wam = [estuary_data[i]['width_at_mouth'] for i in estuary_data]
    h_l = np.array([estuary_data[i]['h_l'] for i in estuary_data])
    mll = np.array(ham) - (h_l /2) # lower layer middle
    mul = h_l / 2 # upper layer middle
    # Calculate depth difference broadcasting properly
    depth_coords_expanded = depth_coords[:, :, np.newaxis, np.newaxis, np.newaxis]  # Shape: (8, 3, 1, 1)
    est_mll_expanded = mll[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]  # Shape: (8, 1, 1, 1)
    
    # Calculate depth differences
    depth_diff = np.abs(depth_coords_expanded - est_mll_expanded)  # Shape: (8, 3, 1, 1)
    depth_weights = 1 / (depth_diff + 1e-10)  # Shape: (8, 3, 1, 1)
    # 3. Normalize the weights along the depth axis
    depth_weights_sum = np.sum(depth_weights, axis=1, keepdims=True)  # Sum along the depth axis
    depth_weights_normalized = depth_weights / depth_weights_sum  # Shape: (8, 3, 1, 1)
    depth_weights_broadcasted = np.broadcast_to(depth_weights_normalized, (8, 3, 720, 17, 17))  # Shape: (8, 3, 720, 17, 17)


    # Broadcast depth_diff to match the full grid shape
    # For a grid of shape (3, 720, 17, 17), we need to expand depth_diff accordingly
    # depth_diff = np.broadcast_to(depth_diff, (8,3, 720, 17, 17))  # Shape: (8, 3, 720, 17, 17)

    masked_salinity = np.where(np.isnan(sal_buffer_corrected), np.nan, sal_buffer_corrected)
    masked_velocity = np.where(np.isnan(vel_magnitude), np.nan, vel_magnitude)
    #!!!
    depth_diff_transposed = np.transpose(depth_weights_broadcasted, (1, 2, 0, 3, 4))


    weighted_sal = np.sum(masked_salinity * depth_diff_transposed, axis=0)
    weighted_vel = np.sum(masked_velocity * depth_diff_transposed, axis=0)
    # Replace zeros with NaNs
    weighted_sal = np.where(weighted_sal != 0, weighted_sal, np.nan)
    weighted_vel = np.where(weighted_vel != 0, weighted_vel, np.nan)
    
    # Calculate the mean along the x and y dimensions
    S_l = np.nanmean(weighted_sal, axis=(-1, -2))  # Mean along the last two axes (x, y)
    avg_weighted_vel = np.nanmean(weighted_vel, axis=(-1, -2))  # Mean along the last two axes (x, y)

      # Assuming `relative_angle` is a numpy array with the same shape as avg_weighted_vel (e.g., (time, locations))
    # `relative_angle` should also be averaged across x and y before this step, similar to avg_weighted_vel
    
    signed_magnitude = np.where(
        (relative_angle >= -90) & (relative_angle <= 90),
        avg_weighted_vel,  # Positive for inflow
        -avg_weighted_vel  # Negative for outflow
    )

    Q_l =  signed_magnitude * h_l * wam
    S_col = np.nanmean(sal_buffer_corrected, axis=(0, 3, 4))
    
    
    access_time = time.time() - start_time
    print(f"Access time with vectorized buffer extraction: {access_time:.2f} seconds")
    
    if writemaps == 'y':
        ## PLOT the estuary buffer locations for salinity 
    #%
        estuary_names = [key for key in estuary_data]  # Extract the names from the dictionary
        # Determine the number of points (based on sal_buffer)
        num_points = sal_buffer_corrected.shape[2]
        # Determine the best subplot grid dimensions
        cols = math.ceil(math.sqrt(num_points))  # Number of columns
        rows = math.ceil(num_points / cols)      # Number of rows
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        # Loop through each point and plot the salinity buffer
        for i in range(num_points):
            row, col = divmod(i, cols)
            ax = axes[row, col] if num_points > 1 else axes
            ax.pcolor(sal_buffer_corrected[0, 0, i, :, :], cmap='viridis')
            ax.set_title(estuary_names[i])  # Set the title based on estuary name
            # Only set the x-label for the bottom plots
            if row == rows - 1:
                ax.set_xlabel('X Index')
            else:
                ax.set_xticklabels([])
            # Only set the y-label for the leftmost plots
            if col == 0:
                ax.set_ylabel('Y Index')
            else:
                ax.set_yticklabels([])
        # Adjust layout to avoid overlap
        plt.tight_layout()
        
    
    #% At this point data variables are handled. Handle time here. 
    
    t = t_raw - np.timedelta64(30, 'm')
    start_time = t[0]
    stop_time = t[-1]
    
    
    
    
    
    
    return start_time, stop_time, Q_l, S_col, S_l, ham, wam 

if artificial_tide == 'n':
    start_time, stop_time, Q_l, S_col, S_l, ham, wam  = load_tidal_data()
elif artificial_tide == 'y':
    pass
    #generate_tidal_data()
#%% Load river
# Could be sped up but for the minute is sufficient. 
def load_river_data(start_time, stop_time, river_path, estuary_data):
    # Create a complete hourly time index from start_time to stop_time
    full_time_index = pd.date_range(start=start_time, end=stop_time, freq='H')

    for i, file in enumerate(river_path.glob('*.csv')):

        river_name = file.name.split('_')[1]
        print(f'Processing river data for ...{river_name}')
        df = pd.read_csv(file, header=0, parse_dates=[0], dayfirst=True)
        df.columns = ['Date', 'Discharge']
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter data within the start and stop time
        mask = (df['Date'] >= start_time) & (df['Date'] <= stop_time)
        filtered_data = df.loc[mask]

        # Set the Date as the index
        filtered_data.set_index('Date', inplace=True)

        # Reindex to ensure all hours are covered
        filtered_data = filtered_data.reindex(full_time_index)

        # Identify missing periods
        missing_periods = filtered_data[filtered_data['Discharge'].isna()]

        if not missing_periods.empty:
            print(f"Handling missing data for {river_name}...")

        # Interpolate missing values linearly
        filtered_data['Discharge'].interpolate(method='linear', inplace=True)

        # Fill any remaining NaN values that might be at the start or end of the period
        filtered_data['Discharge'].ffill(inplace=True)
        filtered_data['Discharge'].bfill(inplace=True)

        # Update the dictionary with the interpolated data
        if river_name in estuary_data:
            estuary_data[river_name]['time'] = filtered_data.index.to_numpy()
            estuary_data[river_name]['discharge'] = filtered_data['Discharge'].to_numpy()
        else:
            print(f"River name {river_name} not found in dictionary.")

    return estuary_data


def artificial_river_data(start_time, stop_time, estuary_data, discharge):
    # Create a complete hourly time index from start_time to stop_time
    full_time_index = pd.date_range(start=start_time, end=stop_time, freq='H')

    # Determine how to handle the discharge parameter based on its type and length
    if isinstance(discharge, (int, float)):
        # If discharge is a single number, create a constant array with that value
        discharge_data = np.full(len(full_time_index), discharge)
    elif isinstance(discharge, list) and len(discharge) == 1:
        # If discharge is a list with one element, create a constant array with that value
        discharge_data = np.full(len(full_time_index), discharge[0])
    elif isinstance(discharge, list) and len(discharge) == len(estuary_data):
        # If discharge is a list with the same length as the number of estuaries,
        # use each value in the list for the corresponding estuary
        for i, river_name in enumerate(estuary_data.keys()):
            estuary_data[river_name]['time'] = full_time_index.to_numpy()
            estuary_data[river_name]['discharge'] = np.full(len(full_time_index), discharge[i])
        return estuary_data
    elif isinstance(discharge, list) and len(discharge) == len(full_time_index):
        # If discharge is a list with the same length as the time index,
        # use the series for each estuary
        discharge_data = np.array(discharge)
    else:
        raise ValueError("Discharge input does not match the required format.")

    # If discharge is a single value or a series, assign it to all estuaries
    for river_name in estuary_data.keys():
        estuary_data[river_name]['time'] = full_time_index.to_numpy()
        estuary_data[river_name]['discharge'] = discharge_data

    return estuary_data

#%%
for discharge in [10,20,30,40,50,75,100,150]:

    # Use the function as needed
    if artificial_river == 'n':
        estuary_data = load_river_data(start_time, stop_time, river_path, estuary_data)
    elif artificial_river == 'y':
        estuary_data = artificial_river_data(start_time, stop_time, estuary_data, discharge)
        
        
    #% Run the EBM
    length =  [estuary_data[i]['length'] for i in estuary_data]
    
    # Assuming the variables are defined as follows:
    W_m = np.array(wam)  # Estuary width (m), shape (8,)
    h = np.array(ham)    # Estuary height (m), shape (8,)
    Q_r = np.array([estuary_data[i]['discharge'] for i in estuary_data.keys()] ) # River discharge, shape (720, 8)
    time_array = np.array([estuary_data[i]['time'] for i in estuary_data.keys()][0] ) # River discharge, shape (720, 8)
    
    Q_m = np.mean(Q_r, axis=1)  # Mean river discharge over time, shape (8,)
    S_l_copy = S_l  # Lower layer salinity, shape (720, 8)
    Q_l_copy = Q_l  # Lower layer volume flux, shape (720, 8)
    S_oc = S_col # Ocean salinity at mouth, shape (720, 8)
    length_copy = np.array(length)  # Estuary length, shape (8,)
    
    # Reshape the (8,) arrays to (8, 1) to allow broadcasting
    W_m = W_m[:, np.newaxis]  # Shape (8, 1)
    h = h[:, np.newaxis]      # Shape (8, 1)
    Q_m = Q_m[:, np.newaxis]  # Shape (8, 1)
    length_copy = length_copy[:, np.newaxis]  # Shape (8, 1)
    
    # Ensure first dimension is 1 for each variable
    W_m = ensure_first_dim_is_one(W_m)  # Shape (1, 8)
    h = ensure_first_dim_is_one(h)      # Shape (1, 8)
    Q_m = ensure_first_dim_is_one(Q_m)  # Shape (1, 8)
    length_copy = ensure_first_dim_is_one(length_copy)  # Shape (1, 8)
    
    if Q_r.shape[0] == 8:
        Q_r = Q_r.T  # Transpose to get shape (720, 8)
    
    
    #%
    Q_u, Lx, S_u, Fi, Flushing_Time, cleaned_flushing_time_phase, S_ebb= ebm(W_m, h, Q_r, Q_m, S_l_copy, Q_l_copy, S_oc, length_copy, time_array)
    #%% EBM extra Functions and plotting
    def plot_flushing_time_and_discharge(cleaned_flushing_time_phase_list, flushing_time_list, Q_u, discharge, time_array, savepath, estuary_names, attempt_label='attempt_2'):
        for loc, estuary in enumerate(estuary_names):
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot Flushing Time on the primary y-axis
            ax1.scatter(cleaned_flushing_time_phase_list[loc], flushing_time_list[loc], marker='o', color='b', label='Flushing Time')
            ax1.set_xlabel('Time (End of Ebb Tide)')
            ax1.set_ylabel('Flushing Time (hours)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True)
            
            # Create a secondary y-axis sharing the same x-axis for Q_u
            ax2 = ax1.twinx()
            ax2.plot(time_array, Q_u[:, loc], color='r', label='Q_u')
            ax2.set_ylabel('Q_u (m³/s)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Create a third y-axis sharing the same x-axis for discharge
            ax3 = ax1.twinx()
            
            # Offset the third axis to avoid overlap
            ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward by 60 points
            ax3.plot(time_array, discharge[:, loc], color='g', label='Discharge')
            ax3.set_ylabel('Discharge (m³/s)', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
            
            # Add titles and legends
            fig.suptitle(f'Flushing Time, Q_u, and Discharge Over Time ({estuary})')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper center')
            ax3.legend(loc='upper right')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
            plt.tight_layout()
    
            # Save the figure with a unique filename
            filename = f"{estuary}_{attempt_label}.png"
            plt.savefig(savepath / Path(filename), dpi=500)
            plt.close(fig)
    
    if artificial_river == 'y':
        al = 'discharge_' + str(discharge)
    else:
        al = "faster_ebm"
    # Assuming the variables are already defined and filled during the EBM processing
    plot_flushing_time_and_discharge(
        cleaned_flushing_time_phase_list=cleaned_flushing_time_phase,
        flushing_time_list=Flushing_Time,
        Q_u=Q_u,
        discharge=Q_r,  # Assuming Q_r is the discharge data
        time_array=time_array,
        savepath=savepath,  # Define the path where you want to save the plots
        estuary_names=list(estuary_data.keys()),  # Use the estuary names from the original data
        attempt_label=al
    )

# %% Plot the numpy saved attributes. 



def plot_estuary_data(storage_location, savepath):
    estuaries = ['Dee', 'Duddon', 'Kent', 'Leven', 'Lune', 'Mersey', 'Ribble', 'Wyre']
    discharges = [10, 20, 30, 40, 50, 75, 100, 150]

    for estuary in estuaries:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for each estuary

        for discharge in discharges:
            # Load the data
            filename = f"{estuary}_discharge_{discharge}.npz"
            filepath = os.path.join(storage_location, filename)
            data = np.load(filepath)

            # Extract data
            sal_in_mean = data['sal_in_mean']
            sal_out_mean = data['sal_out_mean']
            dis_out_mean = data['dis_out_mean']
            flushing_time = data['flushing_time']
            flushing_time_phase = data['flushing_time_phase']

            # Determine the minimum length among all time series
            min_len = min(len(sal_in_mean), len(sal_out_mean), len(dis_out_mean), len(flushing_time_phase))

            # Truncate all arrays to the minimum length
            sal_in_mean = sal_in_mean[:min_len]
            sal_out_mean = sal_out_mean[:min_len]
            dis_out_mean = dis_out_mean[:min_len]
            flushing_time = flushing_time[:min_len]
            flushing_time_phase = flushing_time_phase[:min_len]

            # Convert flushing_time_phase to datetime for plotting
            flushing_time_phase = [np.datetime64(time, 's') for time in flushing_time_phase]

            # Plot Salinity Inflow Mean
            ax1 = axs[0, 0]
            ax1.plot(flushing_time_phase, sal_in_mean, label=f"Discharge {discharge} m³/s")
            ax1.set_title("Salinity Inflow Mean", fontsize=14)
            ax1.set_ylabel("Salinity In (ppt)", fontsize=12)
            ax1.set_xlabel("Time", fontsize=12)
            ax1.tick_params(axis='x', labelrotation=45)
            ax1.legend(fontsize=8)

            # Plot Salinity Outflow Mean
            ax2 = axs[0, 1]
            ax2.plot(flushing_time_phase, sal_out_mean, label=f"Discharge {discharge} m³/s")
            ax2.set_title("Salinity Outflow Mean", fontsize=14)
            ax2.set_ylabel("Salinity Out (ppt)", fontsize=12)
            ax2.set_xlabel("Time", fontsize=12)
            ax2.tick_params(axis='x', labelrotation=45)
            ax2.legend(fontsize=8)

            # Plot Discharge Outflow Mean
            ax3 = axs[1, 0]
            ax3.plot(flushing_time_phase, dis_out_mean, label=f"Discharge {discharge} m³/s")
            ax3.set_title("Discharge Outflow Mean", fontsize=14)
            ax3.set_ylabel("Discharge Out (m³/s)", fontsize=12)
            ax3.set_xlabel("Time", fontsize=12)
            ax3.tick_params(axis='x', labelrotation=45)
            ax3.legend(fontsize=8)

            # Plot Flushing Time
            ax4 = axs[1, 1]
            ax4.plot(flushing_time_phase, flushing_time, label=f"Discharge {discharge} m³/s")
            ax4.set_title("Flushing Time", fontsize=14)
            ax4.set_ylabel("Flushing Time (hours)", fontsize=12)
            ax4.set_xlabel("Time", fontsize=12)
            ax4.tick_params(axis='x', labelrotation=45)
            ax4.legend(fontsize=8)

        plt.tight_layout(pad=3.0)  # Add padding between plots
        
        # Save the figure for the current estuary
        results_path = savepath / f"results_data_sal30_{estuary}.png"
        plt.savefig(results_path, dpi=300)
        plt.close()

# Example usage:
# plot_estuary_data('path_to_folder_containing_npz_files', 'output_filename.png')

plot_estuary_data(storage_location, savepath)
# Example usage:
# plot_estuary_data('path_to_folder_containing_npz_files')



 #%% Run EBM


# #%% Parameters. # Load data from model based on its location.

# def calculate_water_height_change(Q_u, Q_l, Q_r, surface_area, tidal_array):
#     # Calculate Q_net
#     if include_tidal:
        
#         Q_net = Q_u  # Use the total volume flux including tidal effects
#     else:
#         Q_net = Q_r  # Use only the river volume flux
#         #Q_net = Q_u  # calculate total water changes in the box
        
#     timestep = 3600 # number of seconds in timestep. 
    
#     #Q_net_river = Q_r * timestep 
    
#     cumulative_volume_change = (Q_net * timestep) # removed np.cumsum
#     height_change_due_to_flux = cumulative_volume_change / surface_area
#     h_t = tidal_array + height_change_due_to_flux
        
#     return h_t # New water surface height after change in volume. 
 
# def extract_and_attach_data():
#     rofi_buffer = 8 # Buffer to put data selection away from edge of land. 
#     sh_buffer = 0
#     #print('Beginning to extract and attach data to dictionary')
    
#     #time_coords = vomecrty.coords['time_counter']
    
#     for estuary, info in estuary_data.items():
#         print(f'Extract for ...{estuary}')
#         est = estuary_data[estuary]
#         # Case for handling 3 layers. Divide the layers equally, sigma layering. 
#         if Vdata.deptht.size == 3:
#             depth_coords = np.array([0, est['height_at_mouth']/2 ,est['height_at_mouth']])
#         x = info['xy']['x']
#         y = info['xy']['y']
#         # Add in h_l and mll parameters
#         est['mll'] = est['height_at_mouth'] - (est['h_l'] /2) # lower layer middle
#         est['mul'] = est['h_l'] / 2 # upper layer middle

#         # Extract data at the specific (x, y) location
#         # Will extract (depth, time, ((rofibufferY*2) +1), ((rofibufferX*2) +1)
#         vomy = extract_with_buffer(data = vomecrty, x = x, y = y, rofi_buffer = rofi_buffer) # extracts a velocity at this point. 
#         vomx = extract_with_buffer(data = vozocrtx, x = x, y = y, rofi_buffer = rofi_buffer) # extracts a veocity at this point. 
#         est['sh'] = extract_with_buffer(data = sh, x = x, y = y, rofi_buffer = sh_buffer)[:,sh_buffer,sh_buffer] # array [time, 1, 1]
#         # Get magnitudes from the buffer zone before taking a timeseries. 
#         est['vel_magnitude'] = np.sqrt(vomy**2 + vomx**2) # This is still a velocity. 
#         est['vozocrtx'] = vomx.mean(dim=['deptht','y', 'x'], skipna=True)
#         est['vomecrty'] = vomy.mean(dim=['deptht','y', 'x'], skipna=True)
         
#         theta = np.arctan2( np.array(est['vomecrty']) , np.array(est['vozocrtx']))
#         theta_degrees = np.degrees(theta)
#         #max_theta_degrees = np.degrees(max(theta))
#         #min_theta_degrees = np.degrees(min(theta))
#         relative_angle = theta_degrees - est['angle']
        
#         est['salinity'] = extract_with_buffer(data = salinity, x = x, y = y, rofi_buffer = rofi_buffer)
        
#         # Compute depth-based weights
#         depth_diff = np.abs(depth_coords[:, np.newaxis, np.newaxis, np.newaxis] - est['mll'])  # Difference from mll
#         depth_weights = 1 / (depth_diff + 1e-10)  # Avoid division by zero
#         depth_weights /= depth_weights.sum(axis=0, keepdims=True)  # Normalize weights
#         depth_weights = np.broadcast_to(depth_weights, (3, 720, 17, 17))  # Shape: (3, 720, 17, 17)

#         # mask these as the weighting turns nans to zeroes. 
#         masked_salinity = est['salinity'].where(~np.isnan(est['salinity']), other = np.nan)
#         masked_velocity = est['vel_magnitude'].where(~np.isnan(est['vel_magnitude']), other = np.nan)

#         # Apply weights to the data
#         weighted_sal = (masked_salinity * depth_weights).sum(dim='deptht')
#         weighted_vel = (masked_velocity * depth_weights).sum(dim='deptht')
#         weighted_sal = weighted_sal.where(weighted_sal != 0, np.nan)
#         weighted_vel = weighted_vel.where(weighted_vel != 0, np.nan)
        
#         est['S_l'] = weighted_sal.mean(dim=['y', 'x'], skipna=True)
#         avg_weighted_vel = weighted_vel.mean(dim=['y', 'x'], skipna=True)
#         # Handle estuarine direction for what counts as ingoing or outgoing velocities
#         # if side in ['west', 'south']:
#         #     avg_weighted_vel = -1*(avg_weighted_vel)
          
#         signed_magnitude = np.where(
#                 (relative_angle >= -90) & (relative_angle <= 90),
#                 avg_weighted_vel,  # Positive for inflow
#                 -avg_weighted_vel  # Negative for outflow
#             )
#         # Actually calculate a volume rather than a velocity. 
#         est['Q_l'] = signed_magnitude * est['width_at_mouth'] * est['h_l']
#         est['S_col'] = est['salinity'].mean(dim=['deptht','y', 'x'], skipna=True)
  
#         # Verification that weights equal to 1. 
#         #print(np.allclose(depth_weights.sum(axis=0), 1))  # Should be True

#         estuary_data[estuary] = est
#     return estuary_data
# #estuary_data = extract_and_attach_data

# estuary_data = extract_and_attach_data()
# #%% Running the EBM 
# import time
# start = time.time()
# for estuary in estuary_data: 
#     print(f'Running box model for {estuary}')
#     est = estuary_data[estuary]
#     Q_r = est['discharge']      # Surface river discharge    (m3/s)
#     W_m = est['width_at_mouth'] # Width of estuary at mouth  (m)
#     h = est['height_at_mouth']  # Height of estuary at mouth (m)
#     Q_m = np.nanmean(Q_r)       # Mean river discharge       (m3/s)
#     Q_l = est['Q_l']            # Lower layer volume flux from model file (m3/s)
#     S_l = est['S_l']            # Lower layer ocean salinity from model file
#     #vel_tide = 0               # Tide from external tide file, can be ignored. (m/s)
#     #Q_tide = 0                 # Tidal volume flux from external tide file, ignore it 
#     S_oc = est['S_col']                    # Ocean Salinity at mouth
#     est['SA'] = W_m * est['length']
#     length = est['length']
    
#     Q_u, Lx, S_u, Fi, Flushing_Time, cleaned_flushing_time_phase, S_ebb= ebm(W_m, h, Q_r, Q_m, S_l, Q_l, S_oc, length, estuary)
    
#     est['Q_u'] = Q_u 
#     est['Lx'] = Lx 
#     est['S_u'] = S_u 
#     est['Fi'] = np.array(Fi) 
#     est['Fi_mean'] = np.nanmean(abs(est['Fi']))
#     est['Fi_max'] = np.max(abs(est['Fi']))
#     est['Flushing_Time'] = Flushing_Time # Flushing time values
#     est['S_ebb'] = S_ebb
#     est['Flushing_Time_Phase'] = cleaned_flushing_time_phase # times of the fluhsing time values
    
#     # Calculate Surface Height
#     est['sh_box'] = calculate_water_height_change(Q_u = Q_u, Q_l = Q_l, Q_r = Q_r, surface_area = est['SA'], tidal_array = est['sh'])
    
    
# end = str((start - time.time()) / 60)
# print(f'Finished in {end} minutes') 
#%% Make a map of salinity forcing 
# if writemaps == 'y':
#     salinity_array = salinity[0,0,:,:] 
#     plt.figure(figsize=(10, 8))
#     plt.imshow(salinity_array, cmap='viridis', origin='lower')
#     plt.colorbar(label='Salinity')
#     # Plot estuary locations
#     for estuary, info in estuary_data.items():
#         x = info['xy']['x']
#         y = info['xy']['y']
#         print(f"{estuary}: x={x}, y={y}")
#         plt.scatter(x, y, color='red', label=estuary, edgecolor='black')
#         plt.text(x, y, estuary, fontsize=9, ha='right', color='black')
    
    
#     plt.title('Salinity Slice')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')


# Old Stuff not needed 

# # Example of accessing the data
# for estuary, info in estuary_data.items():
#     print(f"Estuary: {estuary}")
#     print(f"  Latitude: {info['latlon']['lat']}")
#     print(f"  Longitude: {info['latlon']['lon']}")
#     print(f"  X Coordinate: {info['xy']['x']}")
#     print(f"  Y Coordinate: {info['xy']['y']}")

"""
Estuary Self-Similar Character and Froude Number Analysis

The concept of "estuary self-similar character" refers to the property of estuarine flows where the flow patterns are invariant under scale transformations. This means that when you zoom in or out, the flow characteristics appear similar, just at a different scale.

In the context of our calculations:

1. **Self-Similarity**: The parameter \(\eta\) is used to characterize the self-similarity of estuarine flows. If \(\eta\) remains constant across different estuarine scales, it indicates that the flow patterns are self-similar. This implies that despite changes in the estuary's physical dimensions, the scaled Froude number can predict similar flow characteristics.

2. **Formula for \(\eta\)**:
   \[
   \eta = \frac{( \text{Fr}_{\text{box}} )^2 \cdot W_m}{h}
   \]
   where \(\text{Fr}_{\text{box}}\) is the Froude number, \(W_m\) is the width of the estuary, and \(h\) is the height.

3. **Practical Interpretation**: If \(\eta\) is consistent for different estuarine systems with varying dimensions, it supports the notion of self-similarity. This means that the flow patterns and behaviors might be similar when scaled appropriately.

4. **Example**:
   - Estuary A: Width = 50 m, Height = 5 m, \(\text{Fr}_{\text{box}} = 2\)
     \[
     \eta_A = \frac{(2)^2 \cdot 50}{5} = 40
     \]
   - Estuary B: Width = 100 m, Height = 10 m, \(\text{Fr}_{\text{box}} = 1\)
     \[None
     \eta_B = \frac{(1)^2 \cdot 100}{10} = 10
     \]

   If \(\eta\) reflects consistent flow behaviors in different estuaries, it indicates self-similarity in the flow patterns.
"""
"""
Box Coefficient (C_k) Calculation

The box coefficient `C_k` is used to characterize the dynamics of estuarine flows based on river discharge, tidal velocity, and estuarine density. It is computed differently depending on whether the river discharge `Q_r` is greater than or less than/equal to the mean river discharge `Q_m`.

1. **Density Influence**:
   - `(Ro_s / 1000)**20`: This term represents the influence of seawater density on the coefficient. The exponent 20 reflects a strong dependence on density, indicating that changes in density significantly impact the coefficient.

2. **Velocity Influence**:
   - `(vel_tide / ur)**-0.5` or `(vel_tide / ur)**0.1`: The velocity ratio terms adjust the coefficient based on tidal velocity compared to a reference velocity. The exponents -0.5 and 0.1 show different sensitivities to this ratio, accounting for varying flow regimes.

3. **Exponential Decay**:
   - `np.exp(-2000 * Eta)`: The large decay factor (2000) indicates a rapid decrease in the influence of the self-similarity parameter `Eta`. This term adjusts the coefficient based on how self-similar the estuarine flow is.

4. **Discharge Conditions**:
   - For high river discharge (`Q_r > Q_m`), the coefficient is highly sensitive to tidal velocity and `Eta`.
   - For low or average river discharge (`Q_r <= Q_m`), the coefficient uses a fixed scaling factor (200) and shows different sensitivities.

These choices are derived from empirical data and model calibration to accurately represent estuarine dynamics under various conditions.
"""

# Testing plotting
# time = estuary_data['Ribble']['time']
# salin = estuary_data['Wyre']['S_u']
# plt.figure();plt.scatter(time, salin, c = 'b'); plt.plot(est['time'], est['discharge'], c = 'r')
# plt.figure();plt.plot(time, salin, c = 'b'); plt.plot(est['time'], est['discharge'], c = 'r')
# plt.figure();plt.plot(time, estuary_data['Ribble']['Q_l'], c = 'r')
# plt.figure();plt.plot(time, estuary_data['Ribble']['vomecrty'], c = 'r');plt.plot(time, estuary_data['Ribble']['vozocrtx'], c = 'r')
# plt.figure(); plt.plot(time, estuary_data['Ribble']['Q_l'], c = 'r') ;plt.plot(time, S_u, c = 'g')

# # Plot phasing of salinity 
# # Create the figure and first axis
# fig, ax1 = plt.subplots()

# # Plot the first dataset on the left y-axis
# ax1.plot(time, estuary_data['Ribble']['Q_l'], color='r')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Q_l (Red)', color='r')
# ax1.tick_params(axis='y', labelcolor='r')

# # Create a second y-axis and plot the second dataset
# ax2 = ax1.twinx()
# ax2.plot(time, S_u.values, color='g')
# ax2.set_ylabel('S_u (Green)', color='g')
# ax2.tick_params(axis='y', labelcolor='g')

# # Show the plot
# plt.title('Q_l and S_u Over Time')
# plt.show()

# # Create the figure and the first axis
# fig, ax1 = plt.subplots()

# # Plot the first dataset on the left y-axis
# ax1.plot(time, estuary_data['Ribble']['Q_l'], color='r')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Q_l (Red)', color='r')
# ax1.tick_params(axis='y', labelcolor='r')

# # Create a second y-axis and plot the second dataset
# ax2 = ax1.twinx()
# ax2.plot(time, estuary_data['Ribble']['S_u'], color='g')
# ax2.set_ylabel('S_u (Green)', color='g')
# ax2.tick_params(axis='y', labelcolor='g')

# # Create a third axis (another twin of ax1)
# ax3 = ax1.twinx()

# # Offset the third axis to avoid overlap
# ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward by 60 points
# ax3.plot(time, estuary_data['Ribble']['discharge'], color='b')
# ax3.set_ylabel('Q_r (Blue)', color='b')
# ax3.tick_params(axis='y', labelcolor='b')

# # Show the plot
# plt.title('Q_l, S_u, and Q_r Over Time')
# plt.show()

# plt.figure(); plt.plot(time, estuary_data['Ribble']['Q_l'], c = 'r') ;plt.plot(time, estuary_data['Ribble']['S_u'], c = 'g')

# plt.figure(); plt.plot(time, estuary_data['Ribble']['S_u'], c = 'r'); plt.plot(time, estuary_data['Ribble']['salinity'].mean(dim=['deptht','y', 'x'], skipna=True), c = 'r')

# #%%% Plot rolling mean averages. 
# # Assuming `estuary_data` is your dictionary and `time` is the corresponding time array (of length 720)
# window_size = 3  # 3 hours, assuming data points represent hourly data

# plt.figure()

# for estuary_name, estuary_dict in estuary_data.items():
#     # Calculate rolling maximum for 'sh'
#     rolling_max_sh = estuary_dict['sh'].rolling(time_counter=window_size, center=True, min_periods=1).max()
    
#     # Calculate rolling maximum for 'sh_box'
#     rolling_max_sh_box = estuary_dict['sh_box'].rolling(time_counter=window_size, center=True, min_periods=1).max()
    
    
#     rolling_diff = rolling_max_sh_box - rolling_max_sh
#     # Save the rolling max results back to the dictionary
#     estuary_dict['rolling_max_sh'] = rolling_max_sh
#     estuary_dict['rolling_max_sh_box'] = rolling_max_sh_box
#     estuary_dict['rolling_max_diff'] = rolling_diff
    
#     # Plot the rolling max for 'sh' and 'sh_box'
    
#     plt.plot(time, rolling_diff, label=f'{estuary_name}')
#     #plt.plot(time, rolling_max_sh_box, label=f'Rolling Max SH_Box ({estuary_name})', color='orange')

        
# plt.xlabel('Time')
# plt.ylabel('Surface Height')
# plt.title('Rolling Max Surface Heights for Estuaries')
# plt.legend()

# #%% 

# # Assuming `estuary_data` is your dictionary and `time` is the corresponding time array (of length 720)
# window_size = 3  # 3 hours, assuming data points represent hourly data

# # Create a new figure for both the time series and histograms
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

# for estuary_name, estuary_dict in estuary_data.items():
#     # Calculate rolling maximum for 'sh'
#     rolling_max_sh = estuary_dict['sh'].rolling(time_counter=window_size, center=True, min_periods=1).max()
    
#     # Calculate rolling maximum for 'sh_box'
#     rolling_max_sh_box = estuary_dict['sh_box'].rolling(time_counter=window_size, center=True, min_periods=1).max()
    
#     # Calculate the difference between rolling maximums
#     rolling_diff = rolling_max_sh_box - rolling_max_sh
    
#     # Save the rolling max results back to the dictionary
#     estuary_dict['rolling_max_sh'] = rolling_max_sh
#     estuary_dict['rolling_max_sh_box'] = rolling_max_sh_box
#     estuary_dict['rolling_max_diff'] = rolling_diff
    
#     # Plot the rolling diff for this estuary on the time series axis
#     ax1.plot(time, rolling_diff, label=f'{estuary_name}')
    
#     # Calculate the histogram for rolling_diff with a fixed number of bins (e.g., 7)
#     hist, bins = np.histogram(rolling_diff, bins=7)
    
#     # Calculate the bin centers
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
#     # Plot the histogram as a bar chart, spaced out along the x-axis
#     ax2.bar(bin_centers + len(estuary_data) * bin_centers.mean(), hist, width=bins[1] - bins[0], alpha=0.7, label=f'{estuary_name}')
    
#     # Add the mean and max lines to the histogram (mean and max markers)
#     mean_diff = rolling_diff.mean().compute()
#     max_diff = rolling_diff.max().compute()
#     ax2.axvline(x=mean_diff + len(estuary_data) * bin_centers.mean(), color='k', linestyle='--')
#     ax2.axvline(x=max_diff + len(estuary_data) * bin_centers.mean(), color='r', linestyle=':')

# # Final touches for the time series plot
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Surface Height Difference')
# ax1.set_title('Rolling Max Surface Heights for Estuaries')
# ax1.legend()

# # Final touches for the histogram plot
# ax2.set_xlabel('Rolling Difference')
# ax2.set_ylabel('Frequency')
# ax2.set_title('Histogram of Rolling Differences for Estuaries')
# ax2.legend()

# # Tight layout and display
# plt.tight_layout()
# plt.show()

# # Optionally, save the figure
# # plt.savefig('combined_rolling_diff_and_histograms.png')

# #%% 


# def inspect_npz_file(filepath):
#     # Load the .npz file
#     data = np.load(filepath)

#     # Extract the arrays from the .npz file
#     sal_in_mean = data['sal_in_mean']
#     sal_out_mean = data['sal_out_mean']
#     dis_out_mean = data['dis_out_mean']
#     flushing_time = data['flushing_time']
#     flushing_time_phase = data['flushing_time_phase']

#     # Print the shapes and first few elements of each array to inspect
#     print(f"File: {os.path.basename(filepath)}")
#     print(f"  sal_in_mean: {len(sal_in_mean)} values")
#     print(f"    First few values: {sal_in_mean[:5]}")
#     print(f"  sal_out_mean: {len(sal_out_mean)} values")
#     print(f"    First few values: {sal_out_mean[:5]}")
#     print(f"  dis_out_mean: {len(dis_out_mean)} values")
#     print(f"    First few values: {dis_out_mean[:5]}")
#     print(f"  flushing_time: {len(flushing_time)} values")
#     print(f"    First few values: {flushing_time[:5]}")
#     print(f"  flushing_time_phase: {len(flushing_time_phase)} values")
#     print(f"    First few values: {flushing_time_phase[:5]}")
#     print("\n")

# # Example usage
# # Replace 'path_to_npz_files' with the directory where your .npz files are stored
# path_to_npz_files = storage_location

# # Loop through some example files
# example_files = ['Dee_discharge_10.npz', 'Duddon_discharge_20.npz', 'Kent_discharge_30.npz']

# for filename in example_files:
#     filepath = os.path.join(path_to_npz_files, filename)
#     inspect_npz_file(filepath)
