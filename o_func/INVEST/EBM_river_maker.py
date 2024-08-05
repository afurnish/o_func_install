# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:50:50 2024

@author: aafur
"""

from o_func import opsys
start_path = opsys()
import os
import glob
import pandas as pd
import dask.dataframe as dd

path = os.path.join(start_path, 'modelling_DATA', 'kent_estuary_project', 'river_boundary_conditions', 'original_river_data', 'processed')
output_path = os.path.join('D://', 'INVEST_modelling', 'river_boundary_conditions', 'EBM_conditions')

rivers = sorted(glob.glob(os.path.join(path, 'copy*')))

multiannual_results = []

def process_river_data(river):
    river_name = river.split('_')[-3].lower()
    print(f"Processing river: {river_name}")
    river_data = pd.read_csv(river)
    river_data.columns = ['datetime', 'value']

    def parse_datetime(date_str):
        try:
            return pd.to_datetime(date_str, format='%d/%m/%Y %H:%M:%S')
        except ValueError:
            return pd.to_datetime(date_str, format='%Y-%m-%d')

    river_data['datetime'] = river_data['datetime'].apply(parse_datetime)
    river_data.dropna(subset=['datetime'], inplace=True)
    river_data.set_index('datetime', inplace=True)
    
    # Define the specific date range for extraction
    start_date = '2013-11-01'
    end_date = '2013-11-30'
    
    # Slice the DataFrame for the specific date range
    sliced_data = river_data.loc[start_date:end_date]
    
    # Convert to Dask DataFrame
    dask_sliced_data = dd.from_pandas(sliced_data, npartitions=4)
    
    # Resample to daily frequency and calculate the daily average
    daily_average = dask_sliced_data.resample('D').mean().compute()
    
    # Write the resulting DataFrame to a file
    daily_average['value'].to_csv(os.path.join(output_path, f'discharge_{river_name}.txt'), index=False, header=False)
    
    # Calculate annual average discharge for the full dataset
    dask_river_data = dd.from_pandas(river_data, npartitions=4)
    annual_average_discharge = dask_river_data.resample('Y').mean().compute()
    
    # Convert to DataFrame for accumulating results
    annual_average_discharge_df = annual_average_discharge.reset_index()
    annual_average_discharge_df['year'] = annual_average_discharge_df['datetime'].dt.year
    annual_average_discharge_df.drop(columns='datetime', inplace=True)
    annual_average_discharge_df.rename(columns={'value': 'annual_average_discharge'}, inplace=True)
    annual_average_discharge_df['river'] = river_name
    
    # Append to multiannual results
    multiannual_results.append(annual_average_discharge_df)

# Process each river and accumulate results
for river in rivers:
    process_river_data(river)

# Concatenate all annual results into a single DataFrame
final_annual_average_discharge_df = pd.concat(multiannual_results)

# Calculate the multiannual discharge across all years by averaging the annual averages
multiannual_discharge_df = final_annual_average_discharge_df.groupby('river')['annual_average_discharge'].mean().reset_index()
multiannual_discharge_df.rename(columns={'annual_average_discharge': 'multiannual_discharge'}, inplace=True)

# Write the accumulated results to a single file in the destination directory
final_annual_average_discharge_df.to_csv(os.path.join(output_path, 'annual_average_discharge_all_rivers.csv'), index=False)
multiannual_discharge_df.to_csv(os.path.join(output_path, 'multiannual_discharge_all_rivers.csv'), index=False)

print("Annual Average Discharge:")
print(final_annual_average_discharge_df)
print("\nMultiannual Discharge:")
print(multiannual_discharge_df)
