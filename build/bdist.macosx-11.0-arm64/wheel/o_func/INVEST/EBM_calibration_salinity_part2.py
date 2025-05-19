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

cal_dataframe = pd.read_csv(start_path / Path('GitHub/o_func_install/o_func/INVEST/salinity_calibration_best_ck_values.csv'))

# Define the dictionary based on the table information
estuary_volumes = {
    "Mersey":    358296615,
    "Kent":        241275544,
    "Leven":      115129827,
    "Duddon":  132010700,
    "Ribble":     61130717,
    "Wyre":      11745203,
    "Lune":      8757500,
    "Dee":        10000 * 8000 * 4.75 # The only one not yet calculated. 
}

# Display the dictionary
print(estuary_volumes)

# Base directory path for the simulation results
base_dir = start_path / Path('modelling_DATA/EBM_PRIMEA/EBM_python/simulation_results')

# List to store dictionaries of data for each file
data_entries = []

# Loop through each file in the base directory
for filename in os.listdir(base_dir):
    print(filename)
    if filename.endswith(".npz"):
        file_path = os.path.join(base_dir, filename)
        
        # Extract estuary name and discharge from the filename
        parts = filename.split('_')
        estuary_name = parts[0]
        
        if estuary_name in estuary_volumes:
            print('parts = ', parts[2])
            discharge_value = int(parts[2].replace("discharge", ""))
            
            # Load the data from the .npz file
            data = np.load(file_path)
            
            # Extract variables
            S_out_mean = data['sal_out_mean'] if 'sal_out_mean' in data else np.nan
            S_in_mean = data['sal_in_mean'] if 'sal_in_mean' in data else np.nan
            C_k_value = data['all_Ck'] if 'all_Ck' in data else np.nan
            eta = data['all_eta'] if 'all_eta' in data else np.nan
            vel_tide = data['all_vel_tide'] if 'all_vel_tide' in data else np.nan
            ur = data['all_ur'] if 'all_ur' in data else np.nan
            Ro_s = data['all_ros'] if 'all_ros' in data else np.nan
            
            # Store data as a dictionary entry
            data_entries.append({
                'estuary': estuary_name,
                'discharge': discharge_value,
                'file_path': file_path,
                'S_out_mean': S_out_mean,
                'S_in_mean': S_in_mean,
                'C_k_value': C_k_value,
                'eta': eta,
                'vel_tide': vel_tide,
                'ur': ur,
                'Ro_s': Ro_s
            })

# Convert the list of dictionaries to a DataFrame for analysis
simulation_df = pd.DataFrame(data_entries)

# Display the first few rows to verify
print(simulation_df.head())

# Save to CSV for further analysis if needed
simulation_df.to_csv('simulation_data_summary.csv', index=False)
