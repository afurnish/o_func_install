#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare v1 and v2 river climatologies — normalised scatter + Dee/Mersey timeseries
@author: af
"""
from o_func import opsys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Setup paths
start_path = Path(opsys('PNC'))
data_path = start_path / "GitHub/o_func_install/o_func/data_prepkit"
fig_path = start_path / 'modelling_DATA/kent_estuary_project/river_boundary_conditions/figures'

# Load CSVs
df_v1 = pd.read_csv(data_path / "all_rivers.csv", parse_dates=['Unnamed: 0'])
df_v2 = pd.read_csv(data_path / "all_rivers_v2.csv", parse_dates=['Unnamed: 0'])

df_v1 = df_v1.rename(columns={'Unnamed: 0': 'datetime'}).set_index('datetime')
df_v2 = df_v2.rename(columns={'Unnamed: 0': 'datetime'}).set_index('datetime')

# Get only river columns common to both
rivers = df_v1.columns.intersection(df_v2.columns)
# Get river names

# Normalise all rivers (0–1 range)
norm_v1 = (df_v1[rivers] - df_v1[rivers].min()) / (df_v1[rivers].max() - df_v1[rivers].min())
norm_v2 = (df_v2[rivers] - df_v2[rivers].min()) / (df_v2[rivers].max() - df_v2[rivers].min())

#%% Functions
import pandas as pd
from datetime import datetime, timedelta

def read_bc_timeseries(filepath, name="Mersey_0001"):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    collecting = False
    base_time = None
    time_discharge = []

    for line in lines:
        line = line.strip()

        if line.startswith("[forcing]"):
            collecting = False  # Reset in case we're between blocks
            continue

        if line.lower().startswith("name") and name in line:
            collecting = True
            continue

        if collecting:
            if line.lower().startswith("unit") and "since" in line:
                # Parse base time
                base_str = line.split("since")[1].strip()
                base_time = datetime.strptime(base_str, "%Y-%m-%d %H:%M:%S")
                continue

            # Only collect lines with two numeric values
            parts = line.split()
            if len(parts) == 2:
                try:
                    t = float(parts[0])
                    q = float(parts[1])
                    time_discharge.append((t, q))
                except ValueError:
                    continue  # Ignore bad lines

    if not time_discharge or base_time is None:
        raise ValueError("Could not find time series or base time in the file.")

    # Convert seconds since base_time to actual datetimes
    datetimes = [base_time + timedelta(seconds=t) for t, _ in time_discharge]
    discharges = [q for _, q in time_discharge]

    df = pd.DataFrame({
        "datetime": datetimes,
        "discharge_m3s": discharges
    })

    return df

df_realmersey = read_bc_timeseries('/Volumes/PNC/modelling_DATA/kent_estuary_project/12.salinity_calibration_laststeps/models/ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv/Discharge.bc', name="Mersey_0001")
df_realdee = read_bc_timeseries('/Volumes/PNC/modelling_DATA/kent_estuary_project/12.salinity_calibration_laststeps/models/ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv/Discharge.bc', name="Dee_0001")

df_realdee['datetime'] = pd.to_datetime(df_realdee['datetime'])
df_realmersey['datetime'] = pd.to_datetime(df_realmersey['datetime'])
#%% === 1st Figure: Normalised v1 vs v2 (2x5 grid) ===

fig, axs = plt.subplots(2, 5, figsize=(14, 7))  # Wider A4-style layout
axs = axs.flatten()

for i, river in enumerate(rivers):
    ax = axs[i]
    ax.scatter(norm_v1[river], norm_v2[river], s=15, marker='s', alpha=0.6)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_title(river, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')  # Square axes

    # Show y-axis ticks and label only on the first column (i == 0 or i == 5)
    if i % 5 == 0:  # Every 5th subplot starting at 0 is in first column
        ax.set_ylabel("v2 Discharge\n[Normalised]", fontsize=8)
    # else:
        # ax.set_yticklabels([])
    
    # Show x-axis ticks and label only on the bottom row (i >= 5)
    if i >= 5:
        ax.set_xlabel("v1 Discharge\n[Normalised]", fontsize=8)
    # else:
        # ax.set_xticklabels([])


    ax.tick_params(labelsize=7)

# Delete unused axes if < 10 plots
for j in range(len(rivers), len(axs)):
    fig.delaxes(axs[j])


# Adjust spacing
# fig.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout()
# Save to file
fig.savefig(fig_path / "normalised_v1_vs_v2_rivers.png", dpi=300, bbox_inches='tight')


#%% === 2nd Figure: Dee and Mersey Timeseries (no grid, 2-day ticks) ===
fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axs[0].plot(df_v1.index, df_v1['Dee'], label='AMM15_River_Climatology_v1', color='navy')
axs[0].plot(df_v2['Dee'], label='AMM15_River_Climatology_v2', color='teal', linestyle='--')
axs[0].plot(df_realdee.datetime,df_realdee['discharge_m3s'], label='15-min NRFA River Discharge', color='darkorange', linestyle='-.')
axs[0].set_title("Dee")
axs[0].set_ylabel("Discharge [m³/s]")
axs[0].legend()
# axs[0].set_ylim([0,20])

axs[1].plot(df_v1.index, df_v1['Mersey'], label='AMM15_River_Climatology_v1', color='navy')
axs[1].plot(df_v2['Mersey'], label='AMM15_River_Climatology_v2', color='teal', linestyle='--')
axs[1].plot(df_realmersey.datetime,df_realmersey['discharge_m3s'], label='15-min NRFA River Discharge', color='darkorange', linestyle='-.')

axs[1].set_title("Mersey")
axs[1].set_ylabel("Discharge [m³/s]")
axs[1].set_xlabel("Time")
axs[1].legend()
# axs[1].set_ylim([0,20])


# Format x-axis: every 2 days
# Format x-axis: monthly ticks

tick_locs = []
tick_labels = []
for month in range(1, 13):
    # Get first index where month matches
    idx = df_v1[df_v1.index.month == month].index[0]
    tick_locs.append(idx)

    # Format as 'Mon YYYY', e.g. 'Jun 2013'
    tick_labels.append(idx.strftime('%b %Y'))
axs[1].set_xticks(tick_locs)
axs[1].set_xticklabels(tick_labels, rotation=45)

# Set x-axis limits: 1st Nov to end of Feb (cross-year)
start_date = pd.to_datetime("2013-11-01")
end_date = pd.to_datetime("2014-02-28")
for ax in axs:
    ax.set_xlim([start_date, end_date])
    ax.grid(False)


# Remove gridlines
for ax in axs:
    ax.grid(False)

fig.tight_layout()
fig.savefig(fig_path / "mersey_dee_timeseries_comparison.png", dpi=300)

#%% === 3rd Section: Statistics (v1 vs v2 vs real discharge) ===
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define output path
stats_path = start_path / "modelling_DATA/kent_estuary_project/river_boundary_conditions"
stats_path.mkdir(parents=True, exist_ok=True)
stats_file = stats_path / "climatology_vs_real_stats.txt"

# Define time period to compare
start_date = pd.to_datetime("2013-11-01")
end_date = pd.to_datetime("2014-02-28")

# Resample real data to daily mean for comparison
df_realdee_daily = df_realdee.set_index('datetime').resample('D').mean()
df_realmersey_daily = df_realmersey.set_index('datetime').resample('D').mean()

# Filter all datasets to the shared time period
df_v1_period = df_v1.loc[start_date:end_date]
df_v2_period = df_v2.loc[start_date:end_date]
df_dee_real = df_realdee_daily.loc[start_date:end_date]
df_mersey_real = df_realmersey_daily.loc[start_date:end_date]

# Function for computing error stats
def compare_series(truth, estimate):
    mae = mean_absolute_error(truth, estimate)
    rmse = mean_squared_error(truth, estimate) ** 0.5
    corr, _ = pearsonr(truth, estimate)
    return mae, rmse, corr

# Gather stats
stats_lines = ["=== Climatology vs Real Discharge Statistics ===\n"]
for river, df_real, v1_series, v2_series in zip(
    ["Dee", "Mersey"],
    [df_dee_real, df_mersey_real],
    [df_v1_period['Dee'], df_v1_period['Mersey']],
    [df_v2_period['Dee'], df_v2_period['Mersey']]
):
    real_vals = df_real['discharge_m3s'].values
    v1_vals = v1_series.values
    v2_vals = v2_series.values

    # Trim to shortest length
    min_len = min(len(real_vals), len(v1_vals), len(v2_vals))
    real_vals = real_vals[:min_len]
    v1_vals = v1_vals[:min_len]
    v2_vals = v2_vals[:min_len]

    stats_lines.append(f"\nRiver: {river}\n")
    
    mae_v1, rmse_v1, corr_v1 = compare_series(real_vals, v1_vals)
    stats_lines.append(f"  v1 vs Real:\n    MAE: {mae_v1:.2f}  RMSE: {rmse_v1:.2f}  Corr: {corr_v1:.2f}")
    
    mae_v2, rmse_v2, corr_v2 = compare_series(real_vals, v2_vals)
    stats_lines.append(f"  v2 vs Real:\n    MAE: {mae_v2:.2f}  RMSE: {rmse_v2:.2f}  Corr: {corr_v2:.2f}")
    
    mae_v1v2, rmse_v1v2, corr_v1v2 = compare_series(v1_vals, v2_vals)
    stats_lines.append(f"  v1 vs v2:\n    MAE: {mae_v1v2:.2f}  RMSE: {rmse_v1v2:.2f}  Corr: {corr_v1v2:.2f}")

# Save stats
with open(stats_file, "w") as f:
    f.write("\n".join(stats_lines))

print(f"✓ Statistics saved to {stats_file}")

#%% Make stats csv 
#%% Make stats CSV from collected statistics (correctly per river)
results = []

# Recalculate and store stats for each river separately
for river, real_df, v1_col, v2_col in [
    ("Dee", df_dee_real, df_v1_period['Dee'], df_v2_period['Dee']),
    ("Mersey", df_mersey_real, df_v1_period['Mersey'], df_v2_period['Mersey'])
]:
    real_vals = real_df['discharge_m3s'].values
    v1_vals = v1_col.values
    v2_vals = v2_col.values

    min_len = min(len(real_vals), len(v1_vals), len(v2_vals))
    real_vals = real_vals[:min_len]
    v1_vals = v1_vals[:min_len]
    v2_vals = v2_vals[:min_len]

    # Store each comparison as a row
    results.extend([
        {
            "River": river,
            "Comparison": "v1 vs Real",
            "MAE": mean_absolute_error(real_vals, v1_vals),
            "RMSE": mean_squared_error(real_vals, v1_vals) ** 0.5,
            "Correlation": pearsonr(real_vals, v1_vals)[0],
        },
        {
            "River": river,
            "Comparison": "v2 vs Real",
            "MAE": mean_absolute_error(real_vals, v2_vals),
            "RMSE": mean_squared_error(real_vals, v2_vals) ** 0.5,
            "Correlation": pearsonr(real_vals, v2_vals)[0],
        },
        {
            "River": river,
            "Comparison": "v1 vs v2",
            "MAE": mean_absolute_error(v1_vals, v2_vals),
            "RMSE": mean_squared_error(v1_vals, v2_vals) ** 0.5,
            "Correlation": pearsonr(v1_vals, v2_vals)[0],
        },
    ])

# Create and format DataFrame
df_stats = pd.DataFrame(results).round(3)
for col in ["MAE", "RMSE", "Correlation"]:
    df_stats[col] = df_stats[col].map("{:.3f}".format)

# Save to CSV
csv_file = stats_path / "climatology_vs_real_stats.csv"
df_stats.to_csv(csv_file, index=False)

print(f"✓ CSV statistics saved to {csv_file}")