#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" First run EBM then run EBM_salinity_calibration to have all variables to hand
Created on Wed Oct 30 15:33:51 2024

@author: af
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

# Estuary names in order for Q_l_copy and observed_cycle
q_l_estuaries = ['Dee', 'Leven', 'Ribble', 'Lune', 'Mersey', 'Wyre', 'Kent', 'Duddon']
observed_keys = ['Dee', 'Duddon', 'Kent', 'Leven', 'Lune', 'Mersey', 'Ribble', 'Wyre']

# Normalize function for easier phase comparison
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Create a figure with 8 subplots
fig, axes = plt.subplots(4, 2, figsize=(12, 12))
fig.suptitle('Phase Comparison of Discharge (Q_l_copy) and Observed Salinity', fontsize=16)

# Loop through each estuary
for i, estuary in enumerate(q_l_estuaries):
    ax = axes[i // 2, i % 2]  # Determine subplot location
    
    # Normalize Q_l_copy and observed_cycle for the current estuary
    q_l_normalized = normalize(Q_l_copy[:, i])
    observed_normalized = normalize(observed_cycle[observed_keys[i]])

    # Plot both on the same subplot
    ax.plot(q_l_normalized, label='Discharge (Q_l_copy)')
    ax.plot(observed_normalized, label='Observed Salinity', linestyle='--')
    ax.set_title(estuary)
    ax.legend()
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Value')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for title
plt.show()
