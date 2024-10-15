#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:31:23 2024

@author: af
"""
#iMPORTED FUNCTIONS
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# HOMEMADE FUNCTIONS
from o_func import opsys, opsys2 

#%% Initial Conditions

#Paths
start_path = Path(opsys())                 # Generate PN startpath
start_pathINVEST = Path(opsys2())  # Generate INVEST startpath
calibration_files = start_pathINVEST / Path('Original_Data/INVEST/35ppt_10Q_run')
Dis = calibration_files / Path('Dis')
Sal = calibration_files / Path('Sal')
Tide = calibration_files / Path('Tide')

#Starting parameters
mapping = 'y'

#%% Loading and plotting data
# Dictionary to store the data for each estuary
data_dict = {}

# Function to capitalize estuary names
def get_estuary_name(file):
    estuary_name = file.stem.split('_')[0].capitalize()
    return estuary_name

# Load the data into the dictionary
for filedirectory, category in zip([Dis, Sal, Tide], ['Dis', 'Sal', 'Tide']):
    for file in filedirectory.glob('*.csv'):
        estuary_name = get_estuary_name(file)
        if estuary_name not in data_dict:
            data_dict[estuary_name] = {'Dis': None, 'Sal': None, 'Tide': None}
        
        # Read the CSV and store it under the appropriate category
        data_dict[estuary_name][category] = pd.read_csv(file)

# Plotting the data in a 3-pane figure
def plot_extracted_data():
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Iterate over each estuary and plot its data
    for estuary, datasets in data_dict.items():
        # Check if all datasets (Dis, Sal, Tide) are available before plotting
        if datasets['Dis'] is not None:
            axes[0].plot(datasets['Dis']['time'], datasets['Dis']['wl'], label=estuary)
        if datasets['Tide'] is not None:
            axes[1].plot(datasets['Tide']['time'], datasets['Tide']['wl'], label=estuary)
        if datasets['Sal'] is not None:
            axes[2].plot(datasets['Sal']['time'], datasets['Sal']['wl'], label=estuary)
    
    # Set labels and titles for the subplots
    axes[0].set_title('Discharge Over Time')
    axes[0].set_ylabel('Discharge')
    
    axes[1].set_title('Tide Over Time')
    axes[1].set_ylabel('Tide')
    
    axes[2].set_title('Salinity Over Time')
    axes[2].set_ylabel('Salinity')
    axes[2].set_xlabel('Time')
    
    # Add legends and grid to all plots
    for ax in axes:
        ax.set_facecolor('white')  # Change background to white
        ax.legend(loc='best')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
if mapping == 'y':
    plot_extracted_data()
    
#%% Salinity calibration
estuary_list = np.array(['Dee', 'Leven', 'Ribble', 'Lune', 'Mersey', 'Wyre', 'Kent', 'Duddon'])
discharge_list = np.array([1, 2, 5, 10, 20, 30, 40, 50])


# Initialize lists to store results
salinity_calibration_results = []

# Path to your data folder
folder_path =start_path / Path( 'modelling_DATA/EBM_PRIMEA/EBM_python/best_simulation_results_as_of_24-09-2024_for_use_in_estuary_multivariate_regression. ')
data_dict = {
    # Sample structure (add your actual data)
    'Dee': {'Dis': [1, 2, 5], 'Sal': np.array([10, 12, 14]), 'Tide': np.array([15, 18, 20])},
    # Add the other estuaries similarly
}

# Calibration logic for salinity
for estuary in estuary_list:
    for discharge in discharge_list:
        file_patterns = [f"{estuary}_discharge_{discharge}__artificial_tide-FES2014", f"{estuary}_discharge_{discharge}__real_tide"]
        file_pattern = file_patterns[0]

        # Find matching filesdancing on my own
        for filename in os.listdir(folder_path):
            if filename.startswith(file_pattern) and filename.endswith(".npz"):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    C_k_value = float(filename.split('_Ck_value-')[-1].replace('.npz', ''))
                except ValueError:
                    print(f"Error parsing C_k value from filename: {filename}")
                    continue

                data = np.load(file_path)

                # Extract variables from the file
                S_u = data['sal_out_mean']  # Modeled upper layer salinity
                S_l = data['sal_in_mean']   # Modeled lower layer salinity (ocean forcing)

                # Observed salinity for this estuary and discharge
                observed_salinity = data_dict[estuary]['Sal'][discharge_list == discharge][0]

                # Calculate error between modeled and observed salinity
                salinity_error = np.abs(S_u - observed_salinity)

                # Save results
                salinity_calibration_results.append({
                    'estuary': estuary,
                    'discharge': discharge,
                    'C_k_value': C_k_value,
                    'S_u_modeled': S_u,
                    'S_u_observed': observed_salinity,
                    'salinity_error': salinity_error,
                })

# Convert results to DataFrame
salinity_calibration_df = pd.DataFrame(salinity_calibration_results)

# Find best C_k value based on the smallest error
best_calibration = salinity_calibration_df.groupby(['estuary', 'discharge']).apply(lambda x: x.loc[x['salinity_error'].idxmin()])

# Save to CSV
best_calibration.to_csv('salinity_calibration.csv')

print(best_calibration)
