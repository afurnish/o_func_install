#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:21:09 2024

@author: af
"""
from scipy.io import loadmat

file = r'/media/af/PN/modelling_DATA/EBM_PRIMEA/EBM_python/20-year-climate-runs/velocity_data/climate_projection_Ribble_2000-2020_Scenario_00000_02.mat'

matdataset = loadmat(file)
vozocrtx_top = matdataset['vozocrtx_top']
vobtcrtx = matdataset['vobtcrtx']


import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Example array for one estuary's Q_l (e.g., time series of flow rates or other tidal cycle indicator)
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def fake_tide():
    # Parameters for generating the synthetic dataset
    total_time_steps = 720  # Total length of the dataset
    base_frequency = 24  # Base frequency for smaller tidal cycles (every 24 hours)
    spring_neap_periodicity = 24 * 14  # Periodicity for the spring-neap cycle (14 days)
    
    # Generate time series
    time_steps = np.arange(total_time_steps)
    
    # Generate the high-frequency oscillation (tidal cycle every 24 hours)
    tidal_cycle = 15000 * np.sin(2 * np.pi * time_steps / base_frequency)
    
    # Generate the spring-neap cycle with a lower frequency
    spring_neap_cycle = 20000 * np.sin(2 * np.pi * time_steps / spring_neap_periodicity)
    
    # Combine the two signals
    Q_l_synthetic = tidal_cycle + spring_neap_cycle
    
    # Add some noise to make the signal more realistic
    noise = 2000 * np.random.randn(total_time_steps)
    Q_l_synthetic += noise
    
    # Break the peaks at the start and tail (undefined peaks)
    Q_l_synthetic[:20] = np.nan
    Q_l_synthetic[-20:] = np.nan
    
    plt.figure(); plt.plot(Q_l_synthetic)
    return Q_l_synthetic


# Example array for one estuary's Q_l (e.g., time series of flow rates or other tidal cycle indicator)
Q_l_timeseries = Q_l[:, 0]  # Assuming you pick one estuary for now (column 0 as an example)
# Q_l_timeseries = fake_tide()
#%%


#%Logic for handling the rest 
# Initialize the masks
# Function to assign spring/neap based on your logic
def generate_spring_neap_mask(Q_l_timeseries):
    
    # Set the approximate periodicity of the spring-neap cycle (14 days, or 336 time steps)
    spring_neap_periodicity = 24 * 14

    # Find the spring peaks using find_peaks with distance between peaks set to periodicity
    spring_peaks, _ = find_peaks(Q_l_timeseries, distance=spring_neap_periodicity)

    # Initialize a list to store the neap trough indices
    neap_troughs = []

    # Loop through each pair of adjacent spring peaks and detect the neap troughs
    for i in range(len(spring_peaks) - 1):
        start = spring_peaks[i] - 10  # Start a little before the spring peak
        end = spring_peaks[i + 1] + 10  # End a little after the next spring peak
        
        # Ensure start and end are within the valid range of data
        start = max(0, start)
        end = min(len(Q_l_timeseries), end)
        
        # Find the neap trough (closest to 0) in the range
        if start < end:
            local_min_index = np.argmin(np.abs(Q_l_timeseries[start:end])) + start
            neap_troughs.append(local_min_index)

    # If the last spring peak doesn't have a subsequent peak (incomplete cycle), handle it separately
    if spring_peaks[-1] < len(Q_l_timeseries) - 1:
        start = spring_peaks[-1] - 10
        end = len(Q_l_timeseries)
        start = max(0, start)
        
        if start < end:
            local_min_index = np.argmin(np.abs(Q_l_timeseries[start:end])) + start
            neap_troughs.append(local_min_index)

    # Convert neap_troughs list to an array for consistency
    neap_troughs = np.array(neap_troughs)
    # Initialize the mask with zeros
    spring_neap_mask = np.full(len(Q_l_timeseries), 0, dtype=int)
    nv = -10000
    sv = 10000
    

    # Identify the first detected point (spring or neap)
    first_spring = spring_peaks[0]
    first_neap = neap_troughs[0] if neap_troughs.size > 0 else np.inf
    initial_point = "spring" if first_spring < first_neap else "neap"
        
    initial_point_value = min(first_spring, first_neap)

    # Compute the average distance as number of points.
    all_points = np.sort(np.concatenate((neap_troughs, spring_peaks)))
    
    if  initial_point == "spring":
        # compute if even
        if len(all_points) % 2 == 0:
            end_point = "neap"
        else:
            end_point = "spring"
    else:
        if len(all_points) % 2 == 0:
            end_point = "spring"
        else:
            end_point = "neap"
    
    mid_points = np.diff(all_points)
    mid_points_halved = (mid_points / 2).astype(int)
    avg_distance = int(mid_points.mean() / 2)
    
    # Handle the points in the middle. 
    new_array = []
    for i in range(len(mid_points_halved)):
        new_array.append(all_points[i])  # Original point
        new_array.append(all_points[i] + mid_points_halved[i])  # Midpoint
    new_array.append(all_points[-1])
    box_points = np.array(new_array)

    num_of_points = len(all_points)
    num_of_bars = 2 * num_of_points - 2
    
    
    # Handle all the values in the middle with modulus function. 
    for i in range(num_of_bars):
        j = i + 1
        i = i-1
        print(j)
        if initial_point == 'spring':
            if ( j // 2 ) % 2 == 0:
                spring_neap_mask[box_points[i+1]:box_points[i+2]] = sv
            else:
                spring_neap_mask[box_points[i+1]:box_points[i+2]] = nv
        else:
            if ( j // 2 ) % 2 == 0:
                spring_neap_mask[box_points[i+1]:box_points[i+2]] = nv
            else:
                spring_neap_mask[box_points[i+1]:box_points[i+2]] = sv
    
    new_start_point = initial_point_value - avg_distance
    new_end_point = all_points[-1] + avg_distance
    
    # Handle beginning
    if  new_start_point < 0:
        if initial_point == 'spring':
            spring_neap_mask[0:initial_point_value] = sv
        else:
            spring_neap_mask[0:initial_point_value] = nv
    else:
        if initial_point == 'spring':
            spring_neap_mask[new_start_point:initial_point_value] = sv
            spring_neap_mask[0:new_start_point] = nv
        else:
            spring_neap_mask[new_start_point:initial_point_value] = nv
            spring_neap_mask[0:new_start_point] = sv
            
    # Handle the end
    if  new_end_point > len(Q_l_timeseries):
        if end_point == 'spring':
            spring_neap_mask[all_points[-1]:] = sv
        else:
            spring_neap_mask[all_points[-1]:] = nv
    else:
        if end_point == 'spring':
            spring_neap_mask[all_points[-1]:new_end_point] = sv
            spring_neap_mask[new_end_point:] = nv
        else:
            spring_neap_mask[all_points[-1]:new_end_point] = nv
            spring_neap_mask[new_end_point:] = sv
            
    
    # Start with the first value. 
    
    return spring_neap_mask



spring_neap_mask = generate_spring_neap_mask(Q_l_timeseries)








# Plot the timeseries and the detected peaks/troughs
plt.figure(figsize=(10, 6))
plt.plot(Q_l_timeseries, label='Q_l Timeseries')

# Plot detected spring peaks
plt.plot(spring_peaks, Q_l_timeseries[spring_peaks], 'ro', label='Spring Peaks')

# Plot detected neap troughs
plt.plot(neap_troughs, Q_l_timeseries[neap_troughs], 'go', label='Neap Troughs')

plt.plot(spring_neap_mask, label = 'springneapmask')
# Add labels and legend
plt.title('Detected Spring and Neap Tides (Handling Incomplete Cycles)')
plt.xlabel('Time Steps')
plt.ylabel('Q_l (Flow Rate or Other Tidal Indicator)')
plt.legend()
plt.show()