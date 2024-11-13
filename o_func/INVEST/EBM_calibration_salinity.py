#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estuary Box Model Calibration Script with Preprocessed Tidal Cycle Averages
"""

from pathlib import Path
import pandas as pd
import numpy as np
import os

# HOMEMADE FUNCTIONS
from o_func import opsyst 

#%% Initial Conditions

# Paths
start_path = Path(opsyst('PN'))                 # Generate PN startpath
start_pathINVEST = Path(opsyst('Elements'))     # Generate INVEST startpath
observed_data_path = start_pathINVEST / Path('Original_Data/INVEST/35ppt_10Q_run_M2/Sal')
mask_file_path = start_path / Path('modelling_DATA/EBM_PRIMEA/EBM_python/mask.npy')
# Mask file is generated every time the EBM is run, so this file should always be run after EBM running

# Define estuaries and date range
estuary_list = ['Dee', 'Leven', 'Ribble', 'Lune', 'Mersey', 'Wyre', 'Kent', 'Duddon']
start_time = np.datetime64('2013-11-10 02:00')
stop_time = np.datetime64('2013-12-07 18:00')
interval_in_hours = int((stop_time - start_time) / np.timedelta64(1, 'h')) + 1
# Load mask and check shape
mask = np.load(mask_file_path)
if mask.shape != (interval_in_hours, len(estuary_list)):
    raise ValueError("Mask shape does not match expected dimensions (720, 8).")

# Function to calculate tidal cycle means using mask
def calculate_tidal_cycle_means(salinity_series, mask_series):
    segment_means = []
    start_idx = 0

    while start_idx < len(salinity_series):
        if  not np.isnan(mask_series[start_idx]):  # Start of a tidal cycle
            end_idx = start_idx
            while end_idx < len(mask_series) and not np.isnan(mask_series[end_idx]):
                end_idx += 1
            
            # Calculate mean for this tidal cycle
            if end_idx > start_idx:
                segment_mean = salinity_series[start_idx:end_idx].mean()
                segment_means.append(segment_mean)
            start_idx = end_idx  # Move to the end of the current tidal cycle
        else:
            start_idx += 1  # Skip NaN values in mask

    return np.array(segment_means)

#%% Step 1: Preprocess Observed Data for Tidal Cycle Averages
observed_cycle_averages = {}
observed_cycle = {}
observed_in = {}
for file in Path(observed_data_path).glob("*.csv"):
    print(file)
    estuary_name = file.stem.split('_')[0].capitalize()
    df = pd.read_csv(file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter data by date range
    df = df[(df['time'] >= start_time) & (df['time'] <= stop_time)]
    observed_salinity = df['wl'].values  # Get salinity values in date range

    # Ensure length consistency with mask
    if len(observed_salinity) != mask.shape[0]:
        raise ValueError(f"Length mismatch for observed salinity in {estuary_name} and mask file.")

    # Calculate tidal cycle averages using the appropriate mask column
    estuary_index = estuary_list.index(estuary_name)
    inverted_mask = np.where(np.isnan(mask), 0, np.nan) # flip the mask to take observed salinities on the output
    tidal_cycle_means = calculate_tidal_cycle_means(observed_salinity, inverted_mask[:, estuary_index])
    tidal_cycle_in_means = calculate_tidal_cycle_means(observed_salinity, mask[:, estuary_index])
    
    observed_cycle_averages[estuary_name] = tidal_cycle_means
    observed_cycle[estuary_name] = observed_salinity
    observed_in[estuary_name] = tidal_cycle_in_means

# The observed_cycle_averages dictionary now contains tidal cycle averaged salinity for each estuary.
# This data is ready for use in the calibration loop.

#%% Step 2: Calibration for Optimal C_k Value
best_simulation_results_path = start_path / Path('modelling_DATA/EBM_PRIMEA/EBM_python/best_simulation_results_as_of_24-09-2024_for_use_in_estuary_multivariate_regression')
best_simulation_results_path = start_path /Path('modelling_DATA/EBM_PRIMEA/EBM_python/simulation_results')

salinity_calibration_results = []

# Iterate through each estuary with discharge = 1
for estuary in estuary_list:
    observed_salinity = observed_cycle_averages.get(estuary, None) # This is the S_out 
    observed_in_sal = observed_in.get(estuary, None) # This is S_in
    if observed_salinity is None:
        print(f"No observed data for {estuary}")
        continue

    # Generate file pattern for C_k simulations with discharge = 1
    file_pattern = f"{estuary}_discharge_10__"

    # Search through files in best simulation results folder
    for filename in os.listdir(best_simulation_results_path):
        if filename.startswith(file_pattern) and filename.endswith(".npz"):
            file_path = os.path.join(best_simulation_results_path, filename)
            
            try:
                C_k_value = float(filename.split('_Ck_value-')[-1].replace('.npz', ''))
            except ValueError:
                print(f"Error parsing C_k value from filename: {filename}")
                continue

            # Load modeled data and calculate mean for each tidal cycle
            data = np.load(file_path)
            S_u_modeled = data['sal_out_mean']
# %%
            S_in = observed_in_sal # data['sal_in_mean']


            # Ensure consistency in number of tidal cycles
            if len(S_u_modeled) != len(observed_salinity):
                print(f"Tidal cycle mismatch for {estuary} with C_k {C_k_value}")
                if len(observed_salinity)< len(S_u_modeled):
                    cut = len(observed_salinity)
                    S_u_modeled = S_u_modeled[:cut]
                else:
                    cut = len(S_u_modeled)
                    observed_salinity = observed_salinity[:cut]

            # Calculate error for each tidal cycle
            salinity_error = np.abs(S_u_modeled - observed_salinity)

            # Append result
            salinity_calibration_results.append({
                'estuary': estuary,
                'C_k_value': C_k_value,
                'S_u_modeled': S_u_modeled,
                'S_u_observed': observed_salinity,
                'salinity_error': salinity_error,
                'S_in': S_in
            })

#%% Compile Results and Find Best Calibration
salinity_calibration_df = pd.DataFrame(salinity_calibration_results)

# Calculate the total error for each C_k value across all tidal cycles
salinity_calibration_df['total_error'] = salinity_calibration_df['salinity_error'].apply(np.sum)
salinity_calibration_df['error_per_cycle'] = salinity_calibration_df['salinity_error'].apply(np.sum) / 58
# Identify the best C_k value (smallest total error) for each estuary
best_calibration = salinity_calibration_df.loc[salinity_calibration_df.groupby('estuary')['total_error'].idxmin()]

# Save the results to CSV
output_path = 'salinity_calibration_best_ck_values.csv'
best_calibration.to_csv(output_path, index=False)

print("Calibration complete. Best C_k values with minimum total error saved to:", output_path)
print(best_calibration)

#%% Governing equations maker. 
import matplotlib.pyplot as plt
# Create a figure with a 2x4 layout
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# Iterate through each estuary and plot in the appropriate subplot
for est_number in range(8):
    ax = axs[est_number // 4, est_number % 4]  # Calculate subplot position
    ax.plot(best_calibration['S_u_observed'].iloc[est_number], label='Delft Out', color = 'cyan')
    ax.plot(best_calibration['S_u_modeled'].iloc[est_number], label='EBM Out', linewidth=5)
    ax.plot(best_calibration['S_in'].iloc[est_number], label='Delft In')
    ax.set_ylabel('Salinity')
    ax.set_xlabel('Time')
    ax.set_title(estuary_list[est_number])  # Title for each estuary
    ax.legend()  # Individual legend for each subplot

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

#%% Lets run some flushing times. 
volumes = [380000000.0,
 176400000,
 630000000,
 392445000,
 11102400,
 177995625.0,
 945000000,
 5116620.0]


flushing_times = []

for i, estuary in enumerate(estuary_list):
    volume = volumes[i]
    S_in = best_calibration['S_in'].iloc[i]
    S_out = best_calibration['S_u_observed'].iloc[i] # S_u_modeled
    print(S_u_modeled.shape)
    print(S_in.shape)
    print(estuary)
    # Calculate flushing times using the provided equation
    flushing_time = (volume * (S_in - S_out)) / (10 * S_in)
    flushing_times.append(flushing_time/3600)
    # flushing_times = abs([i for in in flushing_times)]
# Convert flushing times to a list or array for further use
# flushing_times = np.array(flushing_times)

# Print the flushing times for each estuary
for estuary, ft in zip(estuary_list, flushing_times):
    print(f"Flushing time for {estuary}: {ft}")
    
cal_dataframe = start_path / Path('GitHub/o_func_install/o_func/INVEST/salinity_calibration_best_ck_values.csv')

best_calibration.to_csv(cal_dataframe)
