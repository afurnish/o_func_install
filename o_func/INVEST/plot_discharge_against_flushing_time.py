# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:45:10 2024

@author: aafur
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from o_func import opsys;  start_path = Path(opsys('PN'))
elements_path = Path(opsys('Elements'))
# Configuration
estuary_names = ['Dee', 'Leven', 'Ribble', 'Lune', 'Mersey', 'Wyre', 'Kent', 'Duddon']
base_folder = start_path / Path("modelling_DATA\EBM_PRIMEA\EBM_python\simulation_results") # Update with your actual folder path
discharges = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]  # Update if different

# Initialize dictionary to store results
estuary_data = {estuary: {} for estuary in estuary_names}

# Load and process files
for estuary in estuary_names:
    for discharge in discharges:
        file_path = os.path.join(base_folder, f"{estuary}_discharge_{discharge}__delft35ppt_Ck_value-multivariate_regression.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            print(file_path)
            flushing_time = data['flushing_time']
            mean_flushing_time = np.max(flushing_time)
            estuary_data[estuary][discharge] = mean_flushing_time
        else:
            print(f"File not found: {file_path}")

load_ebb_CSV = pd.read_csv(elements_path / "Original_Data/thom_table_flushing_times/ebb.csv")

# # Prepare plot
# fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
# axes = axes.flatten()

# for i, estuary in enumerate(estuary_names):
#     ax = axes[i]
#     discharge_values = list(estuary_data[estuary].keys())
#     flushing_times = list(estuary_data[estuary].values())

#     ax.plot(discharge_values, flushing_times, marker='o', linestyle='-', label="Flushing Time")
#     ax.set_title(estuary)
#     ax.set_xlabel("Discharge (m³/s)")
#     ax.set_ylabel("Mean Flushing Time (days)")
#     ax.grid(True)
#     ax.legend()

# # Adjust layout
# plt.tight_layout()
# plt.show()
#%%
# Prepare plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, estuary in enumerate(estuary_names):
    ax = axes[i]
    discharge_values = list(estuary_data[estuary].keys())
    flushing_times = list(estuary_data[estuary].values())
    flushing_times = [i* 0.37 for i in flushing_times]
    # Plot the original flushing times
    ax.plot(discharge_values, flushing_times, marker='o', linestyle='-', label="EBM Max Flushing Time @ 37%")
    print(flushing_times)
    # Add the Ebb data as a second line
    if estuary in load_ebb_CSV['Name'].values:
        ebb_data = load_ebb_CSV[load_ebb_CSV['Name'] == estuary].iloc[0, 1:-1].values  # Extract numeric values for ebb
        ebb_discharges = load_ebb_CSV.columns[1:-1].astype(float)  # Convert discharge columns to numeric
        ax.plot(ebb_discharges, ebb_data /  24, marker='x', linestyle='--', label="Ebb Delft Flushing Time")

    ax.set_title(estuary)
    ax.set_xlabel("Discharge (m³/s)")
    ax.set_ylabel("Flushing Time (days)")
    ax.grid(True)
    ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()

#%% Plot against the actual results 
# Convert static values from hours to days
static_values_hours = {
    "Dee": 148,
    "Duddon": 60,
    "Kent": 161,
    "Leven": 111,
    "Mersey": 422,
    "Lune": 73,
    "Ribble": 72,
    "Wyre": 73,
}
static_values_days = {key: val / 24 for key, val in static_values_hours.items()}

# Extract Delft EBM flushing times at 10 m³/s
ebm_10m3s = {estuary: estuary_data[estuary].get(10, None) * 0.37 for estuary in estuary_names}

# Prepare new plot for comparison
# Prepare new plot for comparison
# Prepare new plot for comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Delft EBM values
ebm_values = []
static_values = []
estuary_labels = []
for estuary in estuary_names:
    ebm_value = ebm_10m3s.get(estuary)
    static_value = static_values_days.get(estuary)
    if ebm_value is not None and static_value is not None:
        ebm_values.append(ebm_value)
        static_values.append(static_value)
        estuary_labels.append(estuary)
        
        # Use square markers for Mersey and Duddon
        if estuary in ["Mersey", "Duddon"]:
            ax.scatter(static_value, ebm_value, marker="s", label=f"{estuary}")
        else:
            ax.scatter(static_value, ebm_value, marker="o", label=f"{estuary}")

# Add 1:1 line for comparison
max_val = max(max(ebm_values), max(static_values))
ax.plot([0, max_val], [0, max_val], linestyle="--", color="gray", label="1:1 Line")

# Configure ticks on every day
day_ticks = np.arange(0, max_val + 1, 1)  # Tick every day
ax.set_xticks(day_ticks)
ax.set_yticks(day_ticks)

# Configure plot
ax.set_xlabel("Delft-M2-Residence-time-10-cumecs (days)")
ax.set_ylabel("EBM-M2-Flushing-time-10-cumecs @ 37%  (days)")
ax.set_xlim([0,18])
ax.set_ylim([0,59])
ax.set_title("Initial residence time comparissons")
ax.legend()
ax.grid(True)

# Show plot
plt.tight_layout()
plt.show()


#%% Stats 
import pandas as pd
import numpy as np

# Prepare data for the table
comparison_data = {
    "Estuary": estuary_names,
    "Delft (days)": [ebm_10m3s[estuary] for estuary in estuary_names],
    "Static (days)": [static_values_days[estuary] for estuary in estuary_names],
}

df_comparison = pd.DataFrame(comparison_data)

# Calculate per-estuary statistics
def calculate_metrics(row):
    delft = row["Delft (days)"]
    static = row["Static (days)"]
    absolute_error = abs(delft - static)
    relative_error = absolute_error / delft  # Relative error as a percentage
    within_20_percent = 1 if relative_error <= 1 else 0  # Check if within 20%
    return pd.Series({
        "Absolute Error (days)": absolute_error,
        "Relative Error (%)": relative_error * 100,
        "Within 40%": within_20_percent,
    })

metrics = df_comparison.apply(calculate_metrics, axis=1)

# Add metrics to the DataFrame
df_comparison = pd.concat([df_comparison, metrics], axis=1)

# Save to CSV
csv_path = "per_estuary_metrics.csv"
df_comparison.to_csv(csv_path, index=False)
print(f"CSV saved to {csv_path}")

# Print DataFrame
print(df_comparison)
