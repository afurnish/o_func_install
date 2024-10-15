#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:19:30 2024

@author: af
"""
import numpy as np
import pandas as pd
from o_func import opsys; start_path = opsys()
import matplotlib.pyplot as plt
file = start_path + 'modelling_DATA/EBM_PRIMEA/EBM_python/simulation_results/Ribble_discharge_real_river__AMM7_tide_Ck_value-multivariate_regression.npz'

data = np.load(file)

flushing_time = data['flushing_time']
sal_in = data['sal_in_mean']
sal_out = data['sal_out_mean']
intrusion_len = data['length_of_intrusion']
C_k = data['all_Ck']
years = list(map(str, range(2000, 2021)))
hours_per_year = 8640
datetime_array = []
for year in years:
    y = year
    # Create a datetime range from January 1st for 360 days
    dt_range = pd.date_range(f'{y}-01-01', periods=hours_per_year, freq='h')
    datetime_array.append(dt_range)

dt = np.concatenate(datetime_array)




plt.figure()
plt.plot(dt, intrusion_len)
plt.xlabel('Years through ensemble')
plt.ylabel('Estuarine Intrusion into the Ribble (m)')

#%% 
# Assuming `dt` is the datetime array and `intrusion_len` is the corresponding intrusion data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming `dt` is the datetime array and `intrusion_len` is the corresponding intrusion data
df = pd.DataFrame({'date': dt, 'intrusion_len': intrusion_len})

# Extract the year from the datetime array and add it as a new column
df['year'] = df['date'].dt.year

# Get the list of unique years
years = sorted(df['year'].unique())

# Set up the figure for subplots
fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(20, 6), sharey=True, constrained_layout=True)

# Create a histogram for each year
for i, year in enumerate(years):
    ax = axes[i]
    subset = df[df['year'] == year]['intrusion_len']
    ax.hist(subset, bins=30, alpha=0.7)
    ax.set_title(str(year))
    ax.set_xlim([df['intrusion_len'].min(), df['intrusion_len'].max()])

# Set common labels
fig.text(0.5, 0.04, 'Estuarine Intrusion into the Ribble (m)', ha='center')
fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')

plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df already has a 'year' column and 'intrusion_len' is the data of interest
yearly_data = df.groupby('year')['intrusion_len'].mean()

# Compute a rolling average over a 5-year window for smoothing
rolling_mean = yearly_data.rolling(window=5).mean()

# Plot the original data and the rolling mean
plt.figure(figsize=(10, 6))
plt.plot(yearly_data.index, yearly_data, label='Yearly Intrusion', alpha=0.5)
plt.plot(yearly_data.index, rolling_mean, label='5-Year Rolling Mean', color='red')

# Label the plot
plt.xlabel('Year')
plt.ylabel('Estuarine Intrusion into the Ribble (m)')
plt.title('Trend in Estuarine Intrusion (2000–2020)')
plt.legend()

plt.show()
