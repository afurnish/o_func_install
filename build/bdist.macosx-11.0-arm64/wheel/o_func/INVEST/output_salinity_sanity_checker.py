# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:45:18 2024

@author: aafur
"""
import pyarrow

# Manually add a __version__ attribute if missing
if not hasattr(pyarrow, "__version__"):
    pyarrow.__version__ = "1.0.0"
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
estuary_names = ['ALL']
base_folder = Path("N:\modelling_DATA\EBM_PRIMEA\EBM_python\simulation_results") # Update with your actual folder path
discharges = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]  # Update if different
estuary_data = {estuary: {} for estuary in estuary_names}




for estuary in estuary_names:
    for discharge in discharges:
        file_path = os.path.join(base_folder, f"ALL_ESTUARIES_discharge_{discharge}__delft35ppt_Ck_value-multivariate_regression.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            print(file_path)
            salinity_out = data['S_u']
            estuary_data[estuary][discharge] = salinity_out
        else:
            print(f"File not found: {file_path}")
            
            
estuary_names = ['Dee', 'Leven', 'Ribble', 'Lune', 'Mersey', 'Wyre', 'Kent', 'Duddon']

print(estuary_data['ALL'][1].shape)


#%%
# Create a figure with 8 subplots (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

# Discharges to iterate over
discharges = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
# Iterate over each estuary for plotting
for idx, estuary in enumerate(['Dee', 'Leven', 'Ribble', 'Lune', 'Mersey', 'Wyre', 'Kent', 'Duddon']):
    ax = axes[idx // 4, idx % 4]  # Select the subplot
    k = 1

    for discharge in discharges:
        k = k + 1
        # Extract salinity data for the current estuary and discharge
        salinity_data = estuary_data['ALL'][discharge][:, idx]
        ax.plot(salinity_data + k, label=f"{discharge} m³/s")
    
    # Customize each subplot
    ax.set_title(estuary)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Salinity")
    ax.legend(title="Discharge", fontsize=8, loc='upper right')
    ax.grid(True)

# Adjust layout and display
plt.tight_layout()