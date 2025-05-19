#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:40:56 2024

@author: af
"""
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

path = r'/Volumes/Elements/INVEST_modelling/EBM_data/THOM_3d_model_CSVs_to_run_with_EBM'

sal_files = []
dis_files = []
for i in glob.glob(path + '/*'):
    if i.endswith('sal.csv'):
        sal_files.append(i)
    else:
        dis_files.append(i)
    
def extract_river_name(file):
    # Split the file name by underscore and take the first part, which is the river name
    return os.path.basename(file).split('_')[0]

sal_rivers = [extract_river_name(f) for f in sal_files]
dis_rivers = [extract_river_name(f) for f in dis_files]

def load_csv_data(file_path):
    return pd.read_csv(file_path, parse_dates=['time'])

# Load all discharge files
dis_data = {extract_river_name(file): load_csv_data(file) for file in dis_files}

# Load all salinity files
sal_data = {extract_river_name(file): load_csv_data(file) for file in sal_files}

# Display the fi

#%%
# Create subplots for salinity and discharge
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot discharge data for each river
for river, data in dis_data.items():
    axs[0].plot(data['time'], data['wl'], label=f'{river} Discharge')

# Customize the discharge plot
axs[0].set_title('Discharge for Various Rivers')
axs[0].set_ylabel('Discharge (wl)')
axs[0].grid(True)
axs[0].legend()

# Plot salinity data for each river
for river, data in sal_data.items():
    axs[1].plot(data['time'], data['wl'], label=f'{river} Salinity')

# Customize the salinity plot
axs[1].set_title('Salinity for Various Rivers')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Salinity (wl)')
axs[1].grid(True)
axs[1].legend()

# Adjust layout for clarity
plt.tight_layout()
plt.savefig('/Volumes/PN/testfig.png', dpi = 400)
# Display the plot
plt.show()