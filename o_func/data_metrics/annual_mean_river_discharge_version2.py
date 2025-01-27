
# -*- coding: utf-8 -*-
""" Script to calculate annual mean discharge across estuaries. 


Created on Tue Jan 21 15:24:56 2025
@author: aafur
"""

import os
import pandas as pd
from pathlib import Path
from dateutil.parser import parse


def infer_date_format(date_string):
    """
    Infer the date format from a single date string.
    """
    try:
        parsed_date = parse(date_string, dayfirst=True, fuzzy=False)
        if " " in date_string:
            return "%Y-%m-%d %H:%M:%S"
        else:
            return "%Y-%m-%d"
    except Exception as e:
        raise ValueError(f"Unable to infer date format: {date_string} ({e})")

def calculate_average_discharge(folder_path):
    # Dictionaries to store results
    river_discharge = {}
    river_date_range = {}

    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Ensure it's a CSV file
            # Extract river name from the file name
            river_name = file_name.split("_")[1]

            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the first row to infer the date format
                sample_data = pd.read_csv(file_path, header=None, nrows=1)
                date_sample = sample_data.iloc[0, 0]
                inferred_format = infer_date_format(date_sample)

                # Load the full file with the inferred date format
                data = pd.read_csv(file_path, header=None, names=["Date", "Discharge"])
                data["Date"] = pd.to_datetime(data["Date"], format=inferred_format)

                # Check for parsing errors
                if data["Date"].isnull().any():
                    raise ValueError(f"Unexpected null dates in file {file_name}")

                # Extract the year and calculate annual mean discharge
                data["Year"] = data["Date"].dt.year
                yearly_discharge = data.groupby("Year")["Discharge"].mean()
                overall_avg_discharge = yearly_discharge.mean()

                # Determine start and end dates
                start_date = data["Date"].min()
                end_date = data["Date"].max()

                # Save results
                river_discharge[river_name] = overall_avg_discharge
                river_date_range[river_name] = (start_date, end_date)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

    # Create results DataFrame
    result_df = pd.DataFrame(
        list(river_discharge.items()), columns=["River", "Average Annual Discharge (m³/s)"]
    )
    # Add start and end dates
    result_df["Start Date"] = [river_date_range[river][0] for river in result_df["River"]]
    result_df["End Date"] = [river_date_range[river][1] for river in result_df["River"]]

    # Print the table
    print(result_df.to_string(index=False))



#%%
# Replace 'your_folder_path_here' with the path to your folder containing the files

# Implement this for primea estuaries.
# folder_path = Path(r"N:/modelling_DATA/kent_estuary_project/river_boundary_conditions/original_river_data/processed")
# calculate_average_discharge(folder_path)

# now do it for esk, alt, and clywd
folder_path= Path(r"N:/modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed")
calculate_average_discharge(folder_path)



