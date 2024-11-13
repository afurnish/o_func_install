#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example of histogram plots 


Created on Mon Nov 11 12:58:36 2024
@author: af
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

ribble_file_path = '/media/af/PN/modelling_DATA/EBM_PRIMEA/EBM_python/simulation_results/Ribble_discharge_real_river__artificial_tide-FES2014_t60_tide-M2_S2_Ck_value-multivariate_regression.npz'
all_estuaries_file_path = '/media/af/PN/modelling_DATA/EBM_PRIMEA/EBM_python/simulation_results/ALL_ESTUARIES_discharge_real_river__artificial_tide-FES2014_t60_tide-M2_S2_Ck_value-multivariate_regression.npz'

ribble_data = np.load(ribble_file_path)
timestamps = pd.to_datetime(ribble_data['flushing_time_phase'])
flushing_time_hours = ribble_data['flushing_time']

data_df = pd.DataFrame({
    'Timestamp': timestamps,
    'Flushing Time (hours)': flushing_time_hours
})

data_df['Year'] = data_df['Timestamp'].dt.year
data_df['Season'] = data_df['Timestamp'].dt.month % 12 // 3 + 1
season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
data_df['Season'] = data_df['Season'].map(season_mapping)

seasonal_means = data_df.groupby(['Year', 'Season'])['Flushing Time (hours)'].mean().reset_index()
# Remove negative values from the flushing time data
data_df = data_df[data_df['Flushing Time (hours)'] > 0]

# Timeseries plot with line of best fit
years_numeric = data_df['Timestamp'].dt.year + data_df['Timestamp'].dt.dayofyear / 365.25
slope, intercept = np.polyfit(years_numeric, data_df['Flushing Time (hours)'], 1)
best_fit_line = intercept + slope * years_numeric

#%% Plot the timeseries
plt.figure(figsize=(12, 6))
plt.plot(data_df['Timestamp'], data_df['Flushing Time (hours)'], 'o', label='Flushing Time (Filtered, in Hours)')
plt.plot(data_df['Timestamp'], best_fit_line, 'r-', label='Line of Best Fit')
plt.xlabel('Time')
plt.ylabel('Flushing Time (hours)')
plt.title('Flushing Time (in Hours) with Line of Best Fit')
plt.legend()
plt.grid(True)
plt.show()

#%% Plot the histogram
# Histogram plot grouped by year and season
# Calculate standard deviation for each year and season to use as error bars
error_bars = data_df.groupby(['Year', 'Season'])['Flushing Time (hours)'].std().reset_index()

# Plot histograms with error bars for each season
plt.figure(figsize=(15, 12))
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
for i, season in enumerate(seasons):
    plt.subplot(2, 2, i + 1)
    season_data = data_df[data_df['Season'] == season]
    mean_flushing_time = season_data.groupby('Year')['Flushing Time (hours)'].mean()
    std_flushing_time = season_data.groupby('Year')['Flushing Time (hours)'].std()  # Standard deviation as error bars
    
    # Plot with error bars shaped like stretched "I"s
    plt.bar(mean_flushing_time.index, mean_flushing_time, yerr=std_flushing_time, 
            color='skyblue', edgecolor='black', capsize=5)
    plt.title(f'{season} Flushing Time (Yearly, in Hours) with Error Bars')
    plt.xlabel('Year')
    plt.ylabel('Mean Flushing Time (hours)')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#%% Compute the statistics
# Calculate seasonal trends using linear regression
trend_summary = "Seasonal trend statistics (in hours per year):\n"
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']

for season in seasons:
    season_data = data_df[data_df['Season'] == season]
    if not season_data.empty:
        years_numeric = season_data['Year'] + season_data['Timestamp'].dt.dayofyear / 365.25
        slope, intercept, r_value, p_value, std_err = linregress(years_numeric, season_data['Flushing Time (hours)'])
        trend_summary += f"{season}: Slope = {slope:.2f}, R-squared = {r_value**2:.2f}, p-value = {p_value:.2e}\n"

# Calculate the overall trend
years_numeric_all = data_df['Year'] + data_df['Timestamp'].dt.dayofyear / 365.25
slope_all, intercept_all, r_value_all, p_value_all, std_err_all = linregress(
    years_numeric_all, data_df['Flushing Time (hours)']
)
overall_trend = (
    f"\nOverall trend: Slope = {slope_all:.2f} hours per year, "
    f"R-squared = {r_value_all**2:.2f}, p-value = {p_value_all:.2e}"
)

print(trend_summary + overall_trend)

# Calculate the hourly changes per season for each year
seasonal_hourly_changes = {"Year": sorted(data_df['Year'].unique())}
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']

for season in seasons:
    season_data = data_df[data_df['Season'] == season]
    hourly_changes = []

    # Loop through each unique year and calculate the change from the previous year
    for year in seasonal_hourly_changes["Year"]:
        current_year_data = season_data[season_data['Year'] == year]['Flushing Time (hours)'].mean()
        previous_year_data = season_data[season_data['Year'] == year - 1]['Flushing Time (hours)'].mean()
        
        if not np.isnan(current_year_data) and not np.isnan(previous_year_data):
            hourly_change = current_year_data - previous_year_data
        else:
            hourly_change = np.nan  # Use NaN if data is not available

        hourly_changes.append(hourly_change)

    seasonal_hourly_changes[season] = hourly_changes

# Create a DataFrame from the hourly changes
hourly_changes_df = pd.DataFrame(seasonal_hourly_changes)

# Display the DataFrame
hourly_changes_df.head()

total_change_over_20_years = {}



for season in seasons:

    # Sum the hourly changes across all years, ignoring NaN values

    total_change = hourly_changes_df[season].sum(skipna=True)

    total_change_over_20_years[season] = total_change



# Create a DataFrame to display the total change over 20 years for each season

total_change_df = pd.DataFrame.from_dict(total_change_over_20_years, orient='index', columns=['Total Change (hours)'])

# Calculate the average change per year for each season

average_change_per_year = {season: total_change / 20 for season, total_change in total_change_over_20_years.items()}



# Create a DataFrame to display the average change per year for each season

average_change_df = pd.DataFrame.from_dict(average_change_per_year, orient='index', columns=['Average Change (hours/year)'])

#^error_bars = data_df.groupby(['Year', 'Season'])['Flushing Time (hours)'].std().reset_index()

# Plot histograms with error bars for each season
plt.figure(figsize=(15, 12))
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
for i, season in enumerate(seasons):
    plt.subplot(2, 2, i + 1)
    season_data = data_df[data_df['Season'] == season]
    mean_flushing_time = season_data.groupby('Year')['Flushing Time (hours)'].mean()
    std_flushing_time = season_data.groupby('Year')['Flushing Time (hours)'].std()  # Standard deviation as error bars
    
    # Plot with error bars shaped like stretched "I"s
    plt.bar(mean_flushing_time.index, mean_flushing_time, yerr=std_flushing_time, 
            color='skyblue', edgecolor='black', capsize=5)
    plt.title(f'{season} Flushing Time (Yearly, in Hours) with Error Bars')
    plt.xlabel('Year')
    plt.ylabel('Mean Flushing Time (hours)')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%% 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Calculate the confidence interval for each year and season
confidence_intervals = []
for year in sorted(data_df['Year'].unique()):
    for season in seasons:
        season_data = data_df[(data_df['Year'] == year) & (data_df['Season'] == season)]['Flushing Time (hours)']
        if not season_data.empty:
            mean = season_data.mean()
            sd = season_data.std()
            n = len(season_data)
            # Calculate the 95% confidence interval
            t_value = t.ppf(0.975, df=n-1)  # 0.975 for a two-tailed 95% CI
            margin_of_error = t_value * (sd / np.sqrt(n))
            confidence_intervals.append((year, season, mean, margin_of_error))
        else:
            confidence_intervals.append((year, season, np.nan, np.nan))

# Convert to a DataFrame for plotting
ci_df = pd.DataFrame(confidence_intervals, columns=['Year', 'Season', 'Mean', 'Margin of Error'])

# Plot histograms with confidence intervals for each season
plt.figure(figsize=(15, 12))
for i, season in enumerate(seasons):
    plt.subplot(2, 2, i + 1)
    season_ci = ci_df[ci_df['Season'] == season].dropna()
    plt.bar(season_ci['Year'], season_ci['Mean'], yerr=season_ci['Margin of Error'], 
            color='skyblue', edgecolor='black', capsize=5)
    plt.title(f'{season} Flushing Time (Yearly, in Hours) with 95% Confidence Intervals')
    plt.xlabel('Year')
    plt.ylabel('Mean Flushing Time (hours)')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%% Monthy 
# Calculate the monthly mean flushing times over the 20-year period
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# Calculate the 95% confidence interval for each month
monthly_data = data_df.groupby('Month')['Flushing Time (hours)']
monthly_means = monthly_data.mean()
monthly_stds = monthly_data.std()
sample_sizes = monthly_data.count()

# Calculate the margin of error for 95% confidence interval
t_value = t.ppf(0.975, df=sample_sizes - 1)  # 0.975 for 95% CI, degrees of freedom = sample size - 1
margin_of_error = t_value * (monthly_stds / np.sqrt(sample_sizes))

# Plotting the monthly mean flushing times with 95% confidence intervals
plt.figure(figsize=(10, 6))
plt.bar(monthly_means.index, monthly_means, color='lightblue', edgecolor='black', yerr=margin_of_error, capsize=5)
plt.xlabel('Month')
plt.ylabel('Mean Flushing Time (hours)')
plt.title('Monthly Mean Flushing Times (Over 20 Years) with 95% Confidence Intervals')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


#%% Monthy stats 
import numpy as np
import pandas as pd

# Calculate the change per year for each month
monthly_changes = {"Year": sorted(data_df['Year'].unique())}
for month in range(1, 13):  # Loop through each month from January (1) to December (12)
    changes = []

    for year in monthly_changes["Year"]:
        current_year_data = data_df[(data_df['Year'] == year) & (data_df['Month'] == month)]['Flushing Time (hours)'].mean()
        previous_year_data = data_df[(data_df['Year'] == year - 1) & (data_df['Month'] == month)]['Flushing Time (hours)'].mean()
        
        if not np.isnan(current_year_data) and not np.isnan(previous_year_data):
            change = current_year_data - previous_year_data
        else:
            change = np.nan  # Use NaN if data is not available

        changes.append(change)

    monthly_changes[month] = changes

# Create a DataFrame from the monthly changes
monthly_changes_df = pd.DataFrame(monthly_changes)

# Calculate the total change for each month over the 20 years
total_change_per_month = monthly_changes_df.sum(skipna=True)

# Add the total change as a new row in the DataFrame
monthly_changes_df.loc['Total Change'] = total_change_per_month

# Display the DataFrame
monthly_changes_df.head(15)  # Show the first 15 rows, including the total change row

#%% 
# Calculate the total change for each month over the entire 20-year period
total_monthly_change = monthly_changes_df.iloc[:-1].sum(skipna=True)  # Exclude the 'Total Change' row to sum only yearly changes

# Calculate the average change per month over the 20 years
average_monthly_change = total_monthly_change / 20

# Create a DataFrame to display the total and average change per month
monthly_change_summary_df = pd.DataFrame({
    "Total Change (hours)": total_monthly_change,
    "Average Change (hours/year)": average_monthly_change
})

# Display the DataFrame
monthly_change_summary_df
# Rename the index to use month names in order from January to December
# Remove any extra row, like the "Year" row, from the DataFrame
monthly_change_summary_df = monthly_change_summary_df.iloc[1:13]  # Keep only rows for January to December

# Rename the index to use month names in order from January to December
monthly_change_summary_df.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Display the updated DataFrame
monthly_change_summary_df

# Plotting the total and average change per month on a simple line plot
import matplotlib.pyplot as plt

# Plotting the total and average change per month
plt.figure(figsize=(10, 6))
plt.plot(monthly_change_summary_df.index, monthly_change_summary_df['Total Change (hours)'], 
         marker='o', label='Total Change (hours)', color='blue')
plt.plot(monthly_change_summary_df.index, monthly_change_summary_df['Average Change (hours/year)'], 
         marker='o', label='Average Change (hours/year)', color='orange')

# Adding a horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)

plt.xlabel('Month')
plt.ylabel('Change (hours)')
plt.title('Total and Average Monthly Change Over 20 Years')
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%% 
import matplotlib.pyplot as plt
from scipy.stats import t

# Calculate the seasonal mean flushing times over the 20 years
data_df['Season'] = data_df['Timestamp'].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Autumn
season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
data_df['Season'] = data_df['Season'].map(season_mapping)

seasonal_data = data_df.groupby('Season')['Flushing Time (hours)']
seasonal_means = seasonal_data.mean()
seasonal_stds = seasonal_data.std()
sample_sizes = seasonal_data.count()

# Calculate the 95% confidence interval for each season
t_value = t.ppf(0.975, df=sample_sizes - 1)  # 0.975 for 95% CI, degrees of freedom = sample size - 1
margin_of_error = t_value * (seasonal_stds / np.sqrt(sample_sizes))

# Plotting the seasonal mean flushing times with 95% confidence intervals
plt.figure(figsize=(8, 6))
plt.bar(seasonal_means.index, seasonal_means, color='lightcoral', edgecolor='black', yerr=margin_of_error, capsize=5)
plt.xlabel('Season')
plt.ylabel('Mean Flushing Time (hours)')
plt.title('Seasonal Mean Flushing Times (Over 20 Years) with 95% Confidence Intervals')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

