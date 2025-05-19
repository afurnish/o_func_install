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
    
    test_regular_mean = 'n'
    tsd = 50 # transect seperating distance
    var_mean = {}
    var_dict = {}
    for estuary_name in estuary_names:
        var = variable_df[estuary_name]
        dep = depth_df[estuary_name]
        if var.shape != dep.shape:
            raise ValueError("DataFrames must have the same shape for element-wise multiplication.")
    
        var_per_metre = var * tsd * dep 
        var_per_metre_squared  = var_per_metre.sum(axis=1)
        cross_sectional_area = (dep * tsd).sum(axis=1)
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
        inflow_salinity = inflow_salinity[100:]
        outflow_salinity = outflow_salinity[100:]
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
    
def plot_real_discharges(sal_weighted_discharge):
    """
    Plots real discharge values for multiple estuaries in a multi-panel plot.

    Parameters
    ----------
    sal_weighted_discharge : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing salinity-weighted discharge through time.
    """
    import matplotlib.pyplot as plt
    
    num_estuaries = len(sal_weighted_discharge)
    fig, axes = plt.subplots(num_estuaries, 1, figsize=(10, 5 * num_estuaries), sharex=True)

    if num_estuaries == 1:
        axes = [axes]  # Make axes iterable even if there's only one panel

    for ax, estuary in zip(axes, sal_weighted_discharge.keys()):
        discharge_series = sal_weighted_discharge[estuary]

        # Plot real discharge values for this estuary
        ax.plot(
            discharge_series.index, discharge_series.values, 
            label=f'Real Discharge: {estuary.capitalize()}', color='green', marker='o', linestyle='-', linewidth=1.5
        )
        ax.set_title(f"Real Discharges: {estuary.capitalize()}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Discharge')
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()
    
def normalize_series(series):
    """
    Normalizes a Pandas Series to the range [-1, 1].
    
    Parameters
    ----------
    series : pd.Series
        The series to normalize.
        
    Returns
    -------
    pd.Series
        Normalized series.
    """
    return (series - series.min()) / (series.max() - series.min()) * 2 - 1

def plot_discharge_and_salinity(sal_weighted_discharge, raw_salinity):
    """
    Plots normalized discharge and salinity for multiple estuaries in a multi-panel plot.
    
    Parameters
    ----------
    sal_weighted_discharge : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing salinity-weighted discharge through time.
    raw_salinity : dict
        Dictionary where keys are estuary names and values are Pandas Series
        representing raw salinity values through time.
    """
    import matplotlib.pyplot as plt
    
    num_estuaries = len(sal_weighted_discharge)
    fig, axes = plt.subplots(num_estuaries, 1, figsize=(10, 5 * num_estuaries), sharex=True)

    if num_estuaries == 1:
        axes = [axes]  # Make axes iterable even if there's only one panel

    for ax, estuary in zip(axes, sal_weighted_discharge.keys()):
        discharge_series = sal_weighted_discharge[estuary][100:]
        salinity_series = raw_salinity[estuary][100:]

        # Normalize the data
        normalized_discharge = normalize_series(discharge_series)
        normalized_salinity =salinity_series #normalize_series(salinity_series)

        # Plot normalized discharge
        ax.plot(
            discharge_series.index, normalized_discharge.values,
            label='Normalized Discharge', color='green', linestyle='-', linewidth=1.5
        )
        ax.set_ylabel('Normalized Discharge')
        ax.set_title(f"Normalized Discharge and Salinity: {estuary.capitalize()}")

        # Add a twin y-axis for salinity
        ax2 = ax.twinx()
        ax2.plot(
            salinity_series.index, normalized_salinity.values,
            label='Normalized Salinity', color='blue', linestyle='--', linewidth=1.5
        )
        ax2.set_ylabel('Normalized Salinity')

        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.grid()

    plt.tight_layout()
    plt.show()

def csrdat(velocity_x_dict, velocity_y_dict, angle_dict):
    """ calculate_signed_resultant_velocity_from_depth_averaged_data
    Calculates the signed depth-weighted resultant velocity for multiple estuaries.

    Parameters
    ----------
    velocity_x_dict : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing the x-component of velocity.
    velocity_y_dict : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing the y-component of velocity.
    depth_dict : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing the depth at each point.
    angle_dict : dict
        Dictionary where keys are estuary names and values are angles (in degrees)
        representing the primary axis of each estuary.

    Returns
    -------
    dict
        A dictionary where keys are estuary names and values are Pandas Series 
        representing the signed resultant velocity.
    """
    import numpy as np
    import pandas as pd
    
    resultant_velocity_dict = {}

    for estuary in angle_dict.keys():
        # Fetch data for the current estuary
        velocity_x = velocity_x_dict[estuary]
        velocity_y = velocity_y_dict[estuary]
        estuary_angle = angle_dict[estuary]

        # Convert estuary angle to radians
        alpha = np.radians(estuary_angle)

        # Calculate resultant velocity magnitude and angle
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        velocity_angle = np.arctan2(velocity_y, velocity_x)  # Angle of the velocity vector

        # Compute relative angle to estuary axis
        relative_angle = np.degrees(velocity_angle - alpha)
        relative_angle = (relative_angle + 180) % 360 - 180  # Normalize to [-180, 180]

        # Assign positive or negative based on relative angle
        signed_magnitude = np.where(
            (relative_angle >= -90) & (relative_angle <= 90),
            velocity_magnitude,  # Positive for inflow
            -velocity_magnitude  # Negative for outflow
        )

        # Store the signed resultant velocity        
        resultant_velocity_dict[estuary] = pd.Series(signed_magnitude, index=velocity_x.index)

    
    return resultant_velocity_dict

def classify_tide_slope_normalized_dict(tide_height_dict, smoothing_window=15):
    """
    Classifies tide data as 'Flood' (upslope) or 'Ebb' (downslope) for all items in a dictionary
    by normalizing the tide height.

    Parameters
    ----------
    tide_height_dict : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        representing tide height time series.
    smoothing_window : int, optional
        Window size for the rolling mean used to normalize the tide, by default 5.

    Returns
    -------
    dict
        A dictionary where keys are estuary names and values are DataFrames 
        containing 'Tide_Height', 'Normalized_Tide', 'Slope', and 'Phase'.
    """
    import pandas as pd
    
    result_dict = {}

    for estuary, tide_height in tide_height_dict.items():
        # Smooth the tide height with a rolling mean to normalize it
        smoothed_tide = tide_height.rolling(window=smoothing_window, center=True, min_periods=1).mean()
        normalized_tide = tide_height - smoothed_tide

        # Compute the slope of the normalized tide
        slope = normalized_tide.diff()

        # Classify based on the slope
        phase = slope.apply(lambda x: 'Flood' if x > 0 else 'Ebb' if x < 0 else 'Transition')

        # Combine results into a DataFrame
        result = pd.DataFrame({
            'Tide_Height': tide_height,
            'Normalized_Tide': normalized_tide,
            'Slope': slope,
            'Phase': phase
        })

        # Store in the result dictionary
        result_dict[estuary] = result

    return result_dict

def plot_tide_with_phases(result_dict):
    """
    Plots tide data with phases ('Flood', 'Ebb', 'Transition') for all estuaries in one figure.

    Parameters
    ----------
    result_dict : dict
        Dictionary where keys are estuary names and values are DataFrames 
        containing columns 'Tide_Height' and 'Phase'.
    """
    import matplotlib.pyplot as plt
    
    num_estuaries = len(result_dict)
    fig, axes = plt.subplots(num_estuaries, 1, figsize=(10, 5 * num_estuaries), sharex=True)

    # Ensure axes is iterable even if there's only one panel
    if num_estuaries == 1:
        axes = [axes]

    for ax, (estuary, tide_data) in zip(axes, result_dict.items()):
        # Extract phases
        flood = tide_data[tide_data['Phase'] == 'Flood']
        ebb = tide_data[tide_data['Phase'] == 'Ebb']
        transition = tide_data[tide_data['Phase'] == 'Transition']

        # Plot base tide line
        ax.plot(tide_data.index, tide_data['Tide_Height'], color='gray', alpha=0.5, label="Tide Height (Base)")
        # Scatter points for phases
        ax.scatter(flood.index, flood['Tide_Height'], color='red', label='Flood', zorder=5)
        ax.scatter(ebb.index, ebb['Tide_Height'], color='blue', label='Ebb', zorder=5)
        ax.scatter(transition.index, transition['Tide_Height'], color='green', label='Transition', zorder=5)

        # Customize the plot
        ax.set_title(f"Tide with Phases for {estuary}")
        ax.set_ylabel("Tide Height")
        ax.legend()
        ax.grid()

    # Add a shared x-axis label
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

def calculate_salinity_for_phases(result_dict, salinity_dict):
    """
    Calculates mean salinities for each flood and ebb period.

    Parameters
    ----------
    result_dict : dict
        Dictionary where keys are estuary names and values are DataFrames 
        containing tide classification ('Phase') for each time step.
    salinity_dict : dict
        Dictionary where keys are estuary names and values are Pandas Series 
        containing salinity data.

    Returns
    -------
    dict
        A dictionary where keys are estuary names and values are DataFrames
        with columns 'Flood_Salinity' and 'Ebb_Salinity', indexed by group.
    """
    import pandas as pd
    
    salinity_phase_dict = {}

    for estuary, tide_data in result_dict.items():
        # Get salinity data for the current estuary
        salinity = salinity_dict[estuary]
        
        # Ensure indices are aligned
        tide_data = tide_data.loc[salinity.index]

        # Assign unique group IDs to consecutive Flood and Ebb phases
        tide_data['Group'] = (tide_data['Phase'] != tide_data['Phase'].shift()).cumsum()
        
        # Compute mean salinity for each group
        group_means = tide_data.groupby(['Group', 'Phase']).apply(
            lambda g: salinity.loc[g.index].mean()
        ).unstack()

        # Separate Flood and Ebb salinity into separate columns
        flood_salinity = group_means.get('Flood', pd.Series(index=group_means.index))
        ebb_salinity = group_means.get('Ebb', pd.Series(index=group_means.index))

        salinity_phase_dict[estuary] = pd.DataFrame({
            'Flood_Salinity': flood_salinity,
            'Ebb_Salinity': ebb_salinity
        })

    return salinity_phase_dict


def plot_salinity_in_out(salinity_phase_dict):
    """
    Plots mean salinities for flood (inflow) and ebb (outflow) as separate lines.

    Parameters
    ----------
    salinity_phase_dict : dict
        Dictionary where keys are estuary names and values are DataFrames
        with columns 'Flood_Salinity' and 'Ebb_Salinity'.
    """
    import matplotlib.pyplot as plt
    
    num_estuaries = len(salinity_phase_dict)
    fig, axes = plt.subplots(num_estuaries, 1, figsize=(10, 5 * num_estuaries), sharex=True)

    if num_estuaries == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    for ax, (estuary, salinity_data) in zip(axes, salinity_phase_dict.items()):
        salinity_data = salinity_data.iloc[100:]
        # Plot flood and ebb salinities
        ax.plot(
            salinity_data.index, salinity_data['Flood_Salinity'], 
            label='Flood (Salinity In)', color='blue', marker='o'
        )
        ax.plot(
            salinity_data.index, salinity_data['Ebb_Salinity'], 
            label='Ebb (Salinity Out)', color='red', marker='o'
        )

        ax.set_title(f"Salinity In and Out for {estuary.capitalize()}")
        ax.set_ylabel("Mean Salinity")
        ax.legend()
        ax.grid()

    plt.xlabel("Group (Consecutive Tidal Periods)")
    plt.tight_layout()
    plt.show()

#%%
if __name__ == '__main__':

    from o_func import opsys
    start_path = opsys('Elements')
    from pathlib import Path
    angle_dict = {'dee': 141, 'leven': 4, 'ribble': 84, 'lune': 17, 'mersey': 158, 'wyre': 180, 'kent': 48, 'duddon': 63}
    # Define the path to the transects folder
    path = start_path / Path('Original_Data/transects/transects_raw_slices_from_Thom/Transects')

    # Variables for the transect data
    variables = ['DIS', 'SAL', 'TIDE', 'DEPTH', 'velX', 'velY']
    flowrate = '10m'
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
    velX_weighted_mean = compute_transect_values(velX, depth, estuary_names)
    velY_weighted_mean = compute_transect_values(velY, depth, estuary_names)
    tide_weighted_mean = compute_transect_values(tide, depth, estuary_names)
    
    #old
    # plot_salinity_in_out(sal_weighted_discharge['dee'], sal_weighted_mean['dee'])
    # plot_tidal_cycle_averages(sal_weighted_discharge['dee'], sal_weighted_mean['dee'])

    plot_tidal_cycle_averages_multi(sal_weighted_discharge, sal_weighted_mean)
    
    # plot_real_discharges(sal_weighted_discharge)

    # plot_discharge_and_salinity(sal_weighted_discharge, sal_weighted_mean)
    # plot_discharge_and_salinity(velX_weighted_mean, sal_weighted_mean)
    # plot_discharge_and_salinity(velY_weighted_mean, sal_weighted_mean)
    plot_discharge_and_salinity(tide_weighted_mean, sal_weighted_mean)

    resultant_velocity_dict = csrdat(velX_weighted_mean, velY_weighted_mean, angle_dict)

    plot_discharge_and_salinity(resultant_velocity_dict, sal_weighted_mean)

    plot_tidal_cycle_averages_multi(resultant_velocity_dict, sal_weighted_mean)

    #%% Trying tidal method 
    tidal_dict = classify_tide_slope_normalized_dict(tide_weighted_mean)
    # plot_tide_with_phases(tidal_dict)
    
    salinity_phase_dict = calculate_salinity_for_phases(tidal_dict, sal_weighted_mean)
    plot_salinity_in_out(salinity_phase_dict)
