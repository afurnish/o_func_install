# -*- coding: utf-8 -*-
""" Script to calculate annual mean discharge across estuaries. 


Created on Tue Jan 21 15:24:56 2025
@author: aafur
"""

import os
import pandas as pd
from pathlib import Path

def calculate_average_discharge(folder_path):
    # Dictionary to store the average discharge and date range for each river
    river_discharge = {}
    river_date_range = {}

    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Ensure it's a CSV file
            # Extract river name from the file name
            river_name = file_name.split("_")[1]
            
            # Load the file
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV into a DataFrame
                data = pd.read_csv(file_path, header=None, names=["Date", "Discharge"])
                
                # Parse dates with mixed format inference
                data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
                
                # Check for parsing errors
                if data["Date"].isnull().any():
                    raise ValueError(f"Unrecognized date format in file {file_name}")
                
                data["Year"] = data["Date"].dt.year  # Extract the year
                
                # Calculate average annual discharge
                yearly_discharge = data.groupby("Year")["Discharge"].mean()
                overall_avg_discharge = yearly_discharge.mean()
                
                # Get the first and last date
                start_date = data["Date"].min()
                end_date = data["Date"].max()
                
                # Add the result and date range to the dictionaries
                river_discharge[river_name] = overall_avg_discharge
                river_date_range[river_name] = (start_date, end_date)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

    # Create a DataFrame for the results
    result_df = pd.DataFrame(
        list(river_discharge.items()), columns=["River", "Average Annual Discharge (m³/s)"]
    )
    # Add date range columns to the DataFrame
    result_df["Start Date"] = [river_date_range[river][0] for river in result_df["River"]]
    result_df["End Date"] = [river_date_range[river][1] for river in result_df["River"]]

    # Print the table
    print(result_df.to_string(index=False))


#%%
# Replace 'your_folder_path_here' with the path to your folder containing the files

# Implement this for primea estuaries.
folder_path = Path(r"N:/modelling_DATA/kent_estuary_project/river_boundary_conditions/original_river_data/processed")
calculate_average_discharge(folder_path)

# # now do it for esk, alt, and clywd
# folder_path= Path(r"N:/modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed")
# calculate_average_discharge(folder_path)



