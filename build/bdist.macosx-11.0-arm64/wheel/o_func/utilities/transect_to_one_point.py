#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:06:31 2024

@author: af
"""

def compute_transect_values(variable_df, depth_df, estuary_names):
    '''
    This function can take points across a transect and generate depth/volume averaged points across, 
    which are then returned as an average themselves divided by the depth or volume. 
    
    So to get depth averaged anything which is option 1 if you like. 
    
    We need to take the variable, i.e. salinity then we multiply each salinity point by its correpsonding
    depth which gives us units of psu * m. These are then summed up which gives us psu m^2. This 
    then is divided by the cross sectional surface area of the estuary. 
    
    I believe the simplest method is to sum up the correpsonding depths across the transects and then 
    divide the new salinity per m^2 by its cross sectional surface area to give us depth averaged salinity. 

    Returns
    -------
    None.

    '''
    
    test_regular_mean = 'y'
    
    var_mean = {}
    var_dict = {}
    for estuary_name in estuary_names:
        var = variable_df[estuary_name]
        dep = depth_df[estuary_name]
        if var.shape != dep.shape:
            raise ValueError("DataFrames must have the same shape for element-wise multiplication.")
    
        var_per_metre = var * dep 
        var_per_metre_squared  = var_per_metre.sum(axis=1)
        cross_sectional_area = dep.sum(axis=1)
        variable_weighted_mean = var_per_metre_squared / cross_sectional_area
        var_dict[estuary_name] = variable_weighted_mean
        
        var_mean[estuary_name] = var.mean(axis=1)
    
    if test_regular_mean == 'n':
        return var_dict
    else:
    
        return var_mean
    
    
def read_csv_transects(path, flowrate, variable ):
    """
    Reads CSV files for transects, organizes them by estuary, and stores 
    data in a dictionary of Pandas DataFrames.

    Parameters
    ----------
    path : Path
        Path to the folder containing the transect CSV files.
    model_resolution : str
        Model resolution to filter (e.g., '10m', '20m', etc.).

    Returns
    -------
    estuary_data : dict
        A dictionary where keys are estuary names (lowercase) and values 
        are Pandas DataFrames containing the transect data.
    """
    import pandas as pd
    from pathlib import Path
    
    path = path / flowrate / variable
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")
    
    estuary_data = {}
    for file in path.glob("*.csv"):  
        estuary_name = file.stem.split('_')[1].lower()

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file)
        
        # Store the DataFrame in the dictionary
        estuary_data[estuary_name] = df

        estuary_data[estuary_name].set_index('Datetime', inplace=True)
    return estuary_data
  
def plot_salinity_in_out(discharge_series, salinity_series):
    """
    Plots salinity values for inflow and outflow based on discharge.

    Parameters
    ----------
    discharge_series : pd.Series
        Series with a `Datetime` index representing discharge through time.
    salinity_series : pd.Series
        Series with a `Datetime` index representing salinity through time.
    """
    import matplotlib.pyplot as plt
    # Ensure both Series are aligned by their index
    if not discharge_series.index.equals(salinity_series.index):
        raise ValueError("Discharge and salinity Series must have the same index.")

    # Create masks for inflow and outflow
    inflow_mask = discharge_series > 0  # Positive discharge: inflow
    outflow_mask = discharge_series < 0  # Negative discharge: outflow

    # Map masks to salinity values
    salinity_in = salinity_series.where(inflow_mask)
    salinity_out = salinity_series.where(outflow_mask)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(salinity_in.index, salinity_in, label='Salinity In (Inflow)', color='blue', linewidth=1.5)
    plt.plot(salinity_out.index, salinity_out, label='Salinity Out (Outflow)', color='red', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Salinity')
    plt.title('Salinity In and Out of the Estuary')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def plot_tidal_cycle_averages(discharge_series, salinity_series):
    """
    Groups consecutive inflow and outflow periods based on discharge sign,
    averages the corresponding salinity values, and plots the results.

    Parameters
    ----------
    discharge_series : pd.Series
        Series with a `Datetime` index representing discharge through time.
    salinity_series : pd.Series
        Series with a `Datetime` index representing salinity through time.
    """
    import matplotlib.pyplot as plt

    # Ensure both Series are aligned by their index
    if not discharge_series.index.equals(salinity_series.index):
        raise ValueError("Discharge and salinity Series must have the same index.")

    # Identify groups of consecutive positive or negative values
    sign_groups = (discharge_series > 0).astype(int).diff().ne(0).cumsum()

    # Group salinity values by sign groups and compute mean
    grouped_salinity = salinity_series.groupby(sign_groups).mean()
    grouped_discharge = discharge_series.groupby(sign_groups).mean()  # Optional for reference

    # Separate inflow and outflow based on the mean discharge of each group
    inflow_salinity = grouped_salinity[grouped_discharge > 0]
    outflow_salinity = grouped_salinity[grouped_discharge < 0]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(inflow_salinity.index, inflow_salinity.values, label='Average Salinity In (Inflow)', color='blue', marker='o', linestyle='-', linewidth=1.5)
    plt.plot(outflow_salinity.index, outflow_salinity.values, label='Average Salinity Out (Outflow)', color='red', marker='o', linestyle='-', linewidth=1.5)
    plt.xlabel('Tidal Cycle')
    plt.ylabel('Mean Salinity')
    plt.title('Tidal Cycle Averaged Salinity In and Out of the Estuary')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def plot_tidal_cycle_averages_multi(sal_weighted_discharge, sal_weighted_mean):
    """
    Plots tidal cycle averaged salinity for multiple estuaries in a multi-panel plot.

    Parameters
    ----------
    sal_weighted_discharge : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing salinity-weighted discharge through time.
    sal_weighted_mean : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing salinity through time.
    """
    import matplotlib.pyplot as plt
    
    num_estuaries = len(sal_weighted_discharge)
    fig, axes = plt.subplots(num_estuaries, 1, figsize=(10, 5 * num_estuaries), sharex=True)

    if num_estuaries == 1:
        axes = [axes]  # Make axes iterable even if there's only one panel

    for ax, estuary in zip(axes, sal_weighted_discharge.keys()):
        discharge_series = sal_weighted_discharge[estuary]
        salinity_series = sal_weighted_mean[estuary]

        # Ensure both Series are aligned by their index
        if not discharge_series.index.equals(salinity_series.index):
            raise ValueError(f"Discharge and salinity Series for estuary '{estuary}' must have the same index.")

        # Identify groups of consecutive inflow and outflow periods
        sign_groups = (discharge_series > 0).astype(int).diff().ne(0).cumsum()

        # Group salinity values by sign groups and compute mean
        grouped_salinity = salinity_series.groupby(sign_groups).mean()
        grouped_discharge = discharge_series.groupby(sign_groups).mean()

        # Separate inflow and outflow based on the mean discharge of each group
        inflow_salinity = grouped_salinity[grouped_discharge > 0]
        outflow_salinity = grouped_salinity[grouped_discharge < 0]

        # Plot for this estuary
        ax.plot(
            inflow_salinity.index, inflow_salinity.values, 
            label='Average Salinity In (Inflow)', color='blue', marker='o', linestyle='-', linewidth=1.5
        )
        ax.plot(
            outflow_salinity.index, outflow_salinity.values, 
            label='Average Salinity Out (Outflow)', color='red', marker='o', linestyle='-', linewidth=1.5
        )
        ax.set_title(f"Tidal Cycle Averaged Salinity: {estuary.capitalize()}")
        ax.set_xlabel('Tidal Cycle')
        ax.set_ylabel('Mean Salinity')
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()
    
#%%
if __name__ == '__main__':

    from o_func import opsys
    start_path = opsys('Elements')
    from pathlib import Path
    
    # Define the path to the transects folder
    path = start_path / Path('Original_Data/transects/transects_raw_slices_from_Thom/Transects')

    # Variables for the transect data
    variables = ['DIS', 'SAL', 'TIDE', 'DEPTH', 'velX', 'velY']
    flowrate = '30m'
    # Read the transect data for model resolution '10m'
    discharge = read_csv_transects(path, flowrate, 'DIS')
    salinity      = read_csv_transects(path, flowrate, 'SAL')
    tide            = read_csv_transects(path, flowrate, 'TIDE')
    depth        = read_csv_transects(path, flowrate, 'DEPTH')
    velX           = read_csv_transects(path, flowrate, 'velX')
    velY           = read_csv_transects(path, flowrate, 'velY')
    
    
    # Lets now take say depth averaged salinity of all points. 
    estuary_names = [i for i in discharge.keys()]
    sal_weighted_mean = compute_transect_values(salinity, depth, estuary_names)
    sal_weighted_discharge = compute_transect_values(discharge, depth, estuary_names)
    
    # plot_salinity_in_out(sal_weighted_discharge['dee'], sal_weighted_mean['dee'])
    # plot_tidal_cycle_averages(sal_weighted_discharge['dee'], sal_weighted_mean['dee'])

    plot_tidal_cycle_averages_multi(sal_weighted_discharge, sal_weighted_mean)
