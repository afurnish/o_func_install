#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:05:05 2024

@author: af
"""

import numpy as np
import pandas as pd
import os
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Path to your data folder
folder_path = '/media/af/PN/modelling_DATA/EBM_PRIMEA/EBM_python/best_simulation_results_as_of_24-09-2024_for_use_in_estuary_multivariate_regression. '
new_folder_path = '/media/af/PN/modelling_DATA/EBM_PRIMEA/EBM_python/simulation_results'
# Estuary and discharge lists
estuary_list = np.array(['Dee', 'Leven', 'Ribble', 'Lune', 'Mersey', 'Wyre', 'Kent', 'Duddon'])
discharge_list = np.array([1, 2, 5, 10, 20, 30, 40, 50])

# Estuary volumes (W_m * H_m * L)
wam = np.array([8000, 4860, 10000, 540, 1770, 530, 10500, 4900])
ham = np.array([4.75, 5, 7, 2, 6.25, 0.6, 5, 3])
length = np.array([10000, 16150, 13500, 10280, 16090, 16090, 12000, 12000])
flushtime = 0.37 * 100
# Thoms model value 
database = {
    'estuary': ['Dee', 'Dee', 'Kent', 'Kent', 'Leven', 'Leven', 'Lune', 'Lune', 'Mersey', 'Mersey', 'Ribble', 'Ribble', 'Wyre', 'Wyre', 'Duddon', 'Duddon'],
    'tide_phase': ['spring', 'neap', 'spring', 'neap', 'spring', 'neap', 'spring', 'neap', 'spring', 'neap', 'spring', 'neap', 'spring', 'neap', 'spring', 'neap'],
    'discharge_1': [124, 136, 88, 149, 51, 112, 63, 43, 411, 447, 61, 97, 48, 67, 49, 85]  ,
    'discharge_2': [124, 136, 88, 149, 51, 112, 63, 43, 411, 447, 61, 97, 48, 67, 49, 85],
    'discharge_5': [124, 136, 88, 149, 51, 112, 63, 43, 399, 435, 61, 97, 48, 67, 49, 85],
    'discharge_10': [124, 136, 88, 149, 51, 112, 80, 73, 386, 422, 61, 97, 48, 67, 49, 85],
    'discharge_20': [187, 66, 200, 55, 150, 99, 162, 29, 508, 385, 61, 97, 48, 67, 85, 73],
    'discharge_30': [187, 66, 200, 55, 150, 99, 162, 29, 508, 359, 98, 97, 48, 67, 85, 73],
    'discharge_40': [187, 66, 200, 55, 150, 99, 162, 26, 486, 328, 98, 97, 48, 67, 85, 73],
    'discharge_50': [187, 66, 200, 55, 150, 99, 162, 26, 486, 310, 98, 97, 48, 67, 85, 72]
}
observed_df = pd.DataFrame(database)
# Convert th

# Fixed tidal points for spring and neap tides
spring_datapoint = 38
neap_datapoint = 18
volume = wam * ham * length

#%% 
def extract_tidal_cycles(time_series):
    cycles = []
    current_cycle = []
    for value in time_series:
        if np.isnan(value):
            if current_cycle:  # End of current cycle
                cycles.append(np.mean(current_cycle))  # Store the mean of the current cycle
                current_cycle = []
        else:
            current_cycle.append(value)
    
    # Handle the last cycle
    if current_cycle:
        cycles.append(np.mean(current_cycle))
    
    return np.array(cycles)
#%% 
# Initialize lists to store results
flushing_times = []

# Loop through estuaries and discharges
for estuary in estuary_list:
    for discharge in discharge_list:
        # Construct file name pattern
        file_patterns = [f"{estuary}_discharge_{discharge}__artificial_tide-FES2014", f"{estuary}_discharge_{discharge}__real_tide", f"{estuary}_discharge_{discharge}__real_tide"]
        file_pattern =file_patterns[0]
        # Find matching files
        for filename in os.listdir(folder_path):
            if filename.startswith(file_pattern) and filename.endswith(".npz"):
                file_path = os.path.join(folder_path, filename)
                # print(file_path)
                
                # Extract the C_k value from the filename
                try:
                    C_k_value = float(filename.split('_Ck_value-')[-1].replace('.npz', ''))
                except ValueError:
                    print(f"Error parsing C_k value from filename: {filename}")
                    continue
                data = np.load(file_path)
                
                # Extract variables from the file
                S_u = data['sal_out_mean']  # Salinity outflow upper layer
                S_l = data['sal_in_mean']  # Salinity inflow lower layer
                #data['dis_out_mean']  # Outflow discharge
                
                # Collect the eta, ur, vel_tide and ros data. 
                eta_series = data['all_eta']
                vel_tide_series = data['all_vel_tide']
                ur_series = data['all_ur']
                ros_series = data['all_ros']
                
                eta_cycles = extract_tidal_cycles(eta_series)
                vel_tide_cycles = extract_tidal_cycles(vel_tide_series)
                ur_cycles = extract_tidal_cycles(ur_series)
                Ro_s_cycles = extract_tidal_cycles(ros_series)
                
                min_len = min(len(S_u), len(S_l))
                
                R_outflow = np.full(min_len, discharge)
                
                # Calculate mean salinities and discharges
                S_out_mean = (S_u[min_len:])  # Salinity in
                S_in_mean = (S_l[min_len:])  # Salinity out
                Q_out_mean = (R_outflow)  # Outflow discharge
                
                
                
                # Make sure the dataset is long enough to include datapoints 19 and 35
                if min_len > spring_datapoint and min_len > neap_datapoint:
                    eta_spring = eta_cycles[spring_datapoint]
                    vel_tide_spring = vel_tide_cycles[spring_datapoint]
                    ur_spring = ur_cycles[spring_datapoint]
                    Ro_s_spring = Ro_s_cycles[spring_datapoint]
                    # Extract values at the spring and neap tide points
                    S_out_spring = S_u[spring_datapoint]
                    S_in_spring = S_l[spring_datapoint]
                    Q_out_spring = R_outflow[spring_datapoint]
                    
                    
                    # Neaps 
                    S_out_neap = S_u[neap_datapoint]
                    S_in_neap = S_l[neap_datapoint]
                    Q_out_neap = R_outflow[neap_datapoint]
                    # Extra Neaps
                    eta_neap = eta_cycles[neap_datapoint]
                    vel_tide_neap = vel_tide_cycles[neap_datapoint]
                    ur_neap = ur_cycles[neap_datapoint]
                    Ro_s_neap = Ro_s_cycles[neap_datapoint]
                    # Calculate flushing time at spring tide
                    ft_spring =( (volume[estuary_list == estuary] * (S_in_spring - S_out_spring)) / (Q_out_spring * S_in_spring) ) / 3600
                    
                    # Calculate flushing time at neap tide
                    ft_neap =( (volume[estuary_list == estuary] * (S_in_neap - S_out_neap)) / (Q_out_neap * S_in_neap) ) / 3600
                    
                    # Store the results for both tidal phases
                    flushing_times.append({
                        'estuary': estuary,
                        'discharge': discharge,
                        'C_k_value': C_k_value,
                        'flushing_time': ft_spring,
                        'S_in_mean': S_in_spring,
                        'S_out_mean': S_out_spring,
                        'Q_out_mean': Q_out_spring,
                        'eta': eta_spring,
                        'vel_tide': vel_tide_spring,
                        'ur': ur_spring,
                        'Ro_s': Ro_s_spring,
                        'tidal_phase': 'spring'
                    })
                    
                    flushing_times.append({
                        'estuary': estuary,
                        'discharge': discharge,
                        'C_k_value': C_k_value,
                        'flushing_time': ft_neap,
                        'S_in_mean': S_in_neap,
                        'S_out_mean': S_out_neap,
                        'Q_out_mean': Q_out_neap,
                        'eta': eta_neap,
                        'vel_tide': vel_tide_neap,
                        'ur': ur_neap,
                        'Ro_s': Ro_s_neap,
                        'tidal_phase': 'neap'
                    })

# Convert results to a DataFrame for analysis
flushing_df = pd.DataFrame(flushing_times)

#%%
# Display the results for calibration purposes
calibration_results = []
scaling_factor = 0.37  # Thom's scaling factor for residence time
threshold = 0.3
# Iterate over estuaries
for estuary in observed_df['estuary'].unique():
    for tidal_phase in ['spring', 'neap']:
        # Filter observed and modeled data for this estuary and tidal phase
        observed_sub = observed_df[(observed_df['estuary'] == estuary) & (observed_df['tide_phase'] == tidal_phase)]
        modeled_sub = flushing_df[(flushing_df['estuary'] == estuary) & (flushing_df['tidal_phase'] == tidal_phase)]
        
        # Iterate over the discharge columns in the observed data
        for discharge_col in ['discharge_1', 'discharge_2', 'discharge_5', 'discharge_10', 'discharge_20', 'discharge_30', 'discharge_40', 'discharge_50']:
            discharge_value = int(discharge_col.split('_')[1])  # Get the discharge value (1, 2, 5, etc.)
            
            # Get the observed residence time for this discharge
            observed_residence_time = observed_sub[discharge_col].values[0]
            
            # Filter modeled data for this discharge
            modeled_discharge_sub = modeled_sub[modeled_sub['discharge'] == discharge_value]
            
            # If there is no modeled data for this discharge, skip to the next iteration
            if modeled_discharge_sub.empty:
                continue
            
            # Initialize variables to track the best C_k value and the minimum error
            best_C_k = None
            min_error = np.inf
            best_modeled_flushing_time = None
            best_observed_residence_time = observed_residence_time
            best_S_in_mean = None
            best_S_out_mean = None
            
            # Iterate over all C_k values in the modeled data for this discharge
            for _, row in modeled_discharge_sub.iterrows():
                C_k_value = row['C_k_value']
                modeled_flushing_time = row['flushing_time']
                S_in_mean = row['S_in_mean']  # Inflow salinity
                S_out_mean = row['S_out_mean']  # Outflow salinity
                
                Eta = row['eta']
                Ro_s = row['Ro_s']
                vel_tide = row['vel_tide']
                ur = row['ur']
                
                # Scale the modeled flushing time to match residence time (Thom's model)
                modeled_residence_time = modeled_flushing_time / scaling_factor
                
                # Calculate the error
                error = abs(modeled_residence_time - observed_residence_time)
                
                # Update the best C_k value if this error is smaller
                if error < min_error:
                    min_error = error
                    best_C_k = C_k_value
                    best_modeled_flushing_time = modeled_residence_time  # Store the modeled flushing time
                    best_S_in_mean = S_in_mean  # Store the inflow salinity
                    best_S_out_mean = S_out_mean  # Store the outflow salinity
            
            # Check if modeled flushing time is within ±50% of observed residence time
            lower_bound = observed_residence_time * (1 - threshold)  # 50% below observed value
            upper_bound = observed_residence_time * (1 + threshold)  # 50% above observed value
            within_percent = "yes" if lower_bound <= best_modeled_flushing_time <= upper_bound else "no"
            
            # Store the result for this estuary, discharge, and tidal phase
            calibration_results.append({
                'estuary': estuary,
                'tidal_phase': tidal_phase,
                'discharge': discharge_value,
                'best_C_k': best_C_k,
                'min_error': min_error,
                'observed_residence_time': best_observed_residence_time,  # Store observed time
                'modeled_flushing_time': best_modeled_flushing_time,  # Store modeled time
                'S_in_mean': best_S_in_mean,  # Store inflow salinity
                'S_out_mean': best_S_out_mean,  # Store outflow salinity
                'eta': Eta,
                'Ro_s': Ro_s,
                'vel_tide': vel_tide,
                'ur':ur,
                'within_percent': within_percent  # Yes or No if within 50%
            })

# Convert the calibration results to a DataFrame for analysis
calibration_df = pd.DataFrame(calibration_results)

# Display the calibration results with observed and modeled flushing times
print(calibration_df)   
yes_no_count = calibration_df['within_percent'].value_counts()
calibration_df.to_csv('calibration.csv')
# To get only the count of "yes"
yes_count = yes_no_count['yes']
thres = threshold * 100
print(f"Number of points within {thres}% of modelled : {yes_count} / {len(calibration_df)}")
print(f"Which currently means that about this percentage are correct across all estuaries: {round(yes_count / len(calibration_df )* 100)}")

#%% First Ck calibration to generate equations/ 
# Initialize dictionaries to store separate models for spring and neap tides
spring_estuary_models = {}
neap_estuary_models = {}

# Loop through each estuary and fit separate models for spring and neap
for estuary in calibration_df['estuary'].unique():
    # Filter data for spring and neap
    spring_data = calibration_df[(calibration_df['estuary'] == estuary) & (calibration_df['tidal_phase'] == 'spring')]
    neap_data = calibration_df[(calibration_df['estuary'] == estuary) & (calibration_df['tidal_phase'] == 'neap')]

    # Feature matrix (log(vel_tide/ur) and eta) for spring
    X_spring = np.column_stack([
        np.log(spring_data['vel_tide'] / spring_data['ur']),  # log(vel_tide/ur)
        spring_data['eta']  # Eta
    ])
    y_spring = spring_data['best_C_k'].values  # Target variable for spring

    # Feature matrix (log(vel_tide/ur) and eta) for neap
    X_neap = np.column_stack([
        np.log(neap_data['vel_tide'] / neap_data['ur']),  # log(vel_tide/ur)
        neap_data['eta']  # Eta
    ])
    y_neap = neap_data['best_C_k'].values  # Target variable for neap

    # Fit separate models for spring and neap
    spring_model = LinearRegression()
    neap_model = LinearRegression()

    spring_model.fit(X_spring, y_spring)
    neap_model.fit(X_neap, y_neap)

    # Store the models for each estuary
    spring_estuary_models[estuary] = spring_model
    neap_estuary_models[estuary] = neap_model

# Extract the separate dynamic equations for spring and neap for each estuary
dynamic_spring_neap_equations = {}

for estuary in spring_estuary_models.keys():
    # Spring equation
    spring_coef_log_vel_tide_ur = spring_estuary_models[estuary].coef_[0]  # Coefficient for log(vel_tide/ur)
    spring_coef_eta = spring_estuary_models[estuary].coef_[1]  # Coefficient for Eta
    spring_intercept = spring_estuary_models[estuary].intercept_  # Intercept for spring

    # Neap equation
    neap_coef_log_vel_tide_ur = neap_estuary_models[estuary].coef_[0]  # Coefficient for log(vel_tide/ur)
    neap_coef_eta = neap_estuary_models[estuary].coef_[1]  # Coefficient for Eta
    neap_intercept = neap_estuary_models[estuary].intercept_  # Intercept for neap

    # Store the spring and neap equations for each estuary
    dynamic_spring_neap_equations[estuary] = {
        'spring': f"C_k (spring) = {spring_intercept:.4f} + ({spring_coef_log_vel_tide_ur:.4f} * log(vel_tide / ur)) + ({spring_coef_eta:.4f} * Eta) + ((Ro_s / 1000) ** 20)",
        'neap': f"C_k (neap) = {neap_intercept:.4f} + ({neap_coef_log_vel_tide_ur:.4f} * log(vel_tide / ur)) + ({neap_coef_eta:.4f} * Eta) + ((Ro_s / 1000) ** 20)"
    }

# Convert the equations to a DataFrame for easy display
dynamic_spring_neap_equations_df = pd.DataFrame.from_dict(dynamic_spring_neap_equations, orient='index')

# Save the spring and neap equations to a text file

output_file_spring_neap = 'ck_spring_neap_equations.txt'



with open(output_file_spring_neap, 'w') as f:

    for estuary, equations in dynamic_spring_neap_equations.items():

        f.write(f"{estuary}:\n")

        f.write(f"  Spring: {equations['spring']}\n")

        f.write(f"  Neap: {equations['neap']}\n\n")



# Return the file path for download

output_file_spring_neap

#%% Attempt 2 of CK calibration 
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

#% First Ck calibration to generate equations
# Initialize dictionaries to store separate models for spring and neap tides
spring_estuary_models = {}
neap_estuary_models = {}

# Loop through each estuary and fit separate models for spring and neap
for estuary in calibration_df['estuary'].unique():
    # Filter data for spring and neap
    spring_data = calibration_df[(calibration_df['estuary'] == estuary) & (calibration_df['tidal_phase'] == 'spring')]
    neap_data = calibration_df[(calibration_df['estuary'] == estuary) & (calibration_df['tidal_phase'] == 'neap')]

    # Feature matrix (log(vel_tide/ur), eta) for spring
    X_spring = np.column_stack([
        np.log(spring_data['vel_tide'] / spring_data['ur']),  # log(vel_tide/ur)
        spring_data['eta'],                                   # Eta
        (spring_data['Ro_s'] / 1000) ** 20                    # Ro_s term
    ])
    y_spring = spring_data['best_C_k'].values  # Target variable for spring

    # Feature matrix for neap
    X_neap = np.column_stack([
        np.log(neap_data['vel_tide'] / neap_data['ur']),
        neap_data['eta'],
        (neap_data['Ro_s'] / 1000) ** 20
    ])
    y_neap = neap_data['best_C_k'].values  # Target variable for neap

    # Fit ridge regression for spring and neap with limited regularization
    spring_model = Ridge(alpha=0.1)
    neap_model = Ridge(alpha=0.1)

    spring_model.fit(X_spring, y_spring)
    neap_model.fit(X_neap, y_neap)

    # Store the models for each estuary
    spring_estuary_models[estuary] = spring_model
    neap_estuary_models[estuary] = neap_model

# Extract the simplified equations for spring and neap for each estuary
simplified_spring_neap_equations = {}

for estuary in spring_estuary_models.keys():
    # Get the spring model coefficients
    spring_model = spring_estuary_models[estuary]
    spring_intercept = spring_model.intercept_
    spring_coef_log_vel_tide_ur = spring_model.coef_[0]  # Coefficient for log(vel_tide/ur)
    spring_coef_eta = spring_model.coef_[1]  # Coefficient for Eta
    spring_coef_ros = spring_model.coef_[2]  # Coefficient for Ro_s

    # Get the neap model coefficients
    neap_model = neap_estuary_models[estuary]
    neap_intercept = neap_model.intercept_
    neap_coef_log_vel_tide_ur = neap_model.coef_[0]
    neap_coef_eta = neap_model.coef_[1]
    neap_coef_ros = neap_model.coef_[2]

    # Construct simplified equations
    spring_equation = f"C_k (spring) = {spring_intercept:.4f} + ({spring_coef_log_vel_tide_ur:.4f} * log(vel_tide/ur)) + ({spring_coef_eta:.4f} * Eta) + ({spring_coef_ros:.4f} * (Ro_s / 1000) ** 20)"
    neap_equation = f"C_k (neap) = {neap_intercept:.4f} + ({neap_coef_log_vel_tide_ur:.4f} * log(vel_tide/ur)) + ({neap_coef_eta:.4f} * Eta) + ({neap_coef_ros:.4f} * (Ro_s / 1000) ** 20)"

    # Store the spring and neap equations for each estuary
    simplified_spring_neap_equations[estuary] = {
        'spring': spring_equation,
        'neap': neap_equation
    }

# Convert the equations to a DataFrame for easy display
simplified_spring_neap_equations_df = pd.DataFrame.from_dict(simplified_spring_neap_equations, orient='index')

# Save the spring and neap equations to a text file
output_file_spring_neap = 'simplified_ck_spring_neap_equations.txt'

with open(output_file_spring_neap, 'w') as f:
    for estuary, equations in simplified_spring_neap_equations.items():
        f.write(f"{estuary}:\n")
        f.write(f"  Spring: {equations['spring']}\n")
        f.write(f"  Neap: {equations['neap']}\n\n")


#!!! Directly compare calibartion_df into the equations above

# Load the calibration data

# Define the C_k calculation functions based on the provided equations
def calculate_Ck_spring(vel_tide, ur, eta, Ro_s, coefficients):
    return coefficients['intercept'] + (coefficients['log_vel_tide_ur'] * np.log(vel_tide / ur)) + (coefficients['eta'] * eta) + (coefficients['Ro_s'] * (Ro_s / 1000) ** 20)

def calculate_Ck_neap(vel_tide, ur, eta, Ro_s, coefficients):
    return coefficients['intercept'] + (coefficients['log_vel_tide_ur'] * np.log(vel_tide / ur)) + (coefficients['eta'] * eta) + (coefficients['Ro_s'] * (Ro_s / 1000) ** 20)

# Coefficients for each estuary
Ck_coefficients = {
    'Dee': {
        'spring': {'intercept': -1.1364, 'log_vel_tide_ur': 0.1938, 'eta': 0.0000, 'Ro_s': 0.0000},
        'neap': {'intercept': -2.8948, 'log_vel_tide_ur': 1.0616, 'eta': 0.0000, 'Ro_s': -0.0000}
    },
    'Kent': {
        'spring': {'intercept': -2.1103, 'log_vel_tide_ur': 0.3465, 'eta': -0.0000, 'Ro_s': -0.0000},
        'neap': {'intercept': -3.0994, 'log_vel_tide_ur': 1.0379, 'eta': -0.0000, 'Ro_s': 0.0000}
    },
    'Leven': {
        'spring': {'intercept': -2.4538, 'log_vel_tide_ur': 0.4593, 'eta': 0.0000, 'Ro_s': 0.0000},
        'neap': {'intercept': -3.1792, 'log_vel_tide_ur': 1.3384, 'eta': 0.0000, 'Ro_s': 0.0000}
    },
    'Lune': {
        'spring': {'intercept': -0.1067, 'log_vel_tide_ur': 0.0425, 'eta': 0.0000, 'Ro_s': -0.0000},
        'neap': {'intercept': 0.1317, 'log_vel_tide_ur': 0.3983, 'eta': 0.0000, 'Ro_s': -0.0000}
    },
    'Mersey': {
        'spring': {'intercept': 0.0000, 'log_vel_tide_ur': 0.0000, 'eta': 0.0000, 'Ro_s': 0.0000},
        'neap': {'intercept': -4.3193, 'log_vel_tide_ur': 2.1753, 'eta': 0.0000, 'Ro_s': -0.0000}
    },
    'Ribble': {
        'spring': {'intercept': -0.5352, 'log_vel_tide_ur': 0.0821, 'eta': 0.0000, 'Ro_s': 0.0000},
        'neap': {'intercept': -5.7350, 'log_vel_tide_ur': 1.7151, 'eta': 0.0000, 'Ro_s': -0.0000}
    },
    'Wyre': {
        'spring': {'intercept': 0.0000, 'log_vel_tide_ur': 0.0000, 'eta': 0.0000, 'Ro_s': 0.0000},
        'neap': {'intercept': 0.0269, 'log_vel_tide_ur': 0.0183, 'eta': 0.0000, 'Ro_s': 0.0000}
    },
    'Duddon': {
        'spring': {'intercept': -0.4071, 'log_vel_tide_ur': 0.0821, 'eta': 0.0000, 'Ro_s': 0.0000},
        'neap': {'intercept': -0.4191, 'log_vel_tide_ur': 0.1913, 'eta': 0.0000, 'Ro_s': -0.0000}
    }
}

# Initialize an empty list to store the calculated C_k values
calculated_Ck_values = []

# Loop through the calibration data and calculate the C_k values based on the equations
for index, row in calibration_df.iterrows():
    estuary = row['estuary']
    tidal_phase = row['tidal_phase']
    vel_tide = row['vel_tide']
    ur = row['ur']
    eta = row['eta']
    Ro_s = row['Ro_s']

    # Calculate C_k based on the tidal phase (spring/neap)
    if tidal_phase == 'spring':
        C_k_value = calculate_Ck_spring(vel_tide, ur, eta, Ro_s, Ck_coefficients[estuary]['spring'])
    else:
        C_k_value = calculate_Ck_neap(vel_tide, ur, eta, Ro_s, Ck_coefficients[estuary]['neap'])
    
    # Append the calculated C_k value to the list
    calculated_Ck_values.append(C_k_value)

# Add the calculated C_k values to the dataframe for comparison
calibration_df['calculated_C_k'] = calculated_Ck_values
calibration_df['vt_ur'] = calibration_df['vel_tide'] / calibration_df['ur']
#%%
#!!! Attempt 3 this i believe is the best fit that I have found so far for plugging the equations back into each other. 
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Define the power law function for C_k with different scenarios
def power_law_scenario_1(X, a, b, c, d):
    log_vel_tide_ur, eta, Ro_s = X
    return a * (log_vel_tide_ur ** b) * (eta ** c) * ((Ro_s / 1000) ** d)

def power_law_scenario_2(X, a, b, c, d):
    log_vel_tide_ur, eta, Ro_s = X
    return a * (log_vel_tide_ur ** b) * np.exp(c * eta) * ((Ro_s / 1000) ** d)

# Create a dictionary to store the equations for each estuary-tidal phase
power_law_equations = {}

# Estuaries and tidal phases
estuaries = calibration_df['estuary'].unique()
tidal_phases = ['spring', 'neap']

# Handle NaNs (interpolation or drop them)
calibration_df = calibration_df.dropna(subset=['vel_tide', 'ur', 'eta', 'Ro_s'])

# Calculate log(vel_tide/ur)
calibration_df['log_vt_ur'] = np.log(calibration_df['vel_tide'] / calibration_df['ur'])

# Loop over each estuary and tidal phase
for estuary in estuaries:
    for tidal_phase in tidal_phases:
        # Filter the dataset for the specific estuary and tidal phase
        estuary_data = calibration_df[(calibration_df['estuary'] == estuary) & (calibration_df['tidal_phase'] == tidal_phase)]
        
        # Check if we have enough data for this estuary and tidal phase
        if len(estuary_data) < 2:
            print(f"Skipping {estuary}_{tidal_phase} due to insufficient data")
            continue
        
        # Prepare the data for fitting
        X = np.column_stack((
            estuary_data['log_vt_ur'],
            estuary_data['eta'],
            estuary_data['Ro_s']
        ))
        y = estuary_data['best_C_k'].values

        # First, try scenario 1
        try:
            popt, _ = curve_fit(power_law_scenario_1, X.T, y, maxfev=1000000)
            power_law_equations[f'{estuary}_{tidal_phase}_scenario_1'] = {
                'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3]
            }
        except RuntimeError as e:
            print(f"Scenario 1 failed for {estuary}_{tidal_phase}: {e}")

        # Then, try scenario 2
        try:
            popt, _ = curve_fit(power_law_scenario_2, X.T, y, maxfev=1000000)
            power_law_equations[f'{estuary}_{tidal_phase}_scenario_2'] = {
                'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3]
            }
        except RuntimeError as e:
            print(f"Scenario 2 failed for {estuary}_{tidal_phase}: {e}")

# Output the generated power law equations
for key, values in power_law_equations.items():
    print(f"{key}: {values}")






# Output the generated equations
import numpy as np
import pandas as pd

# # Power-law equations for each estuary and tidal phase
# Save as a set of okay examples
power_law_equations = {
    'Dee_spring': lambda vt_ur, eta, ros: 3.7992 * (vt_ur ** 8.1109) * (eta ** -2.4065) * ((ros / 1000) ** -618.7599),
    'Dee_neap': lambda vt_ur, eta, ros: 0.5143 * (vt_ur ** 3.5133) * (eta ** 1.0295) * ((ros / 1000) ** -87.8392),
    'Kent_spring': lambda vt_ur, eta, ros: 0.0010 * (vt_ur ** 7.0428) * (eta ** -0.9086) * ((ros / 1000) ** -269.9368),
    'Kent_neap': lambda vt_ur, eta, ros: 0.5189 * (vt_ur ** 3.6430) * (eta ** 1.2619) * ((ros / 1000) ** -87.6977),
    'Leven_spring': lambda vt_ur, eta, ros: 0.0000 * (vt_ur ** 6.6108) * (eta ** 0.4549) * ((ros / 1000) ** -92.9577),
    'Leven_neap': lambda vt_ur, eta, ros: 0.4386 * (vt_ur ** 3.6130) * (eta ** 1.1057) * ((ros / 1000) ** -22.0325),
    'Lune_spring': lambda vt_ur, eta, ros: 0.0000 * (vt_ur ** 7.6116) * (eta ** -3.4376) * ((ros / 1000) ** -146.1108),
    'Lune_neap': lambda vt_ur, eta, ros: 0.8818 * (vt_ur ** 1.4879) * (eta ** 1.0534) * ((ros / 1000) ** 7.7606),
    'Mersey_spring': lambda vt_ur, eta, ros: -0.0001 * (vt_ur ** 1.0011) * (eta ** -729.2788) * ((ros / 1000) ** -0.0068),
    'Mersey_neap': lambda vt_ur, eta, ros: 0.1621 * (vt_ur ** 6.6300) * (eta ** 1.2422) * ((ros / 1000) ** -50.0538),
    'Ribble_spring': lambda vt_ur, eta, ros: 142.0402 * (vt_ur ** 1.0058) * (eta ** 55.1473) * ((ros / 1000) ** -12700.2839),
    'Ribble_neap': lambda vt_ur, eta, ros: 0.2361 * (vt_ur ** 4.2946) * (eta ** 1.3365) * ((ros / 1000) ** -38.1270),
    'Wyre_spring': lambda vt_ur, eta, ros: 0.0000 * (vt_ur ** 1.0005) * (eta ** -147.9867) * ((ros / 1000) ** 8.8348),
    'Wyre_neap': lambda vt_ur, eta, ros: 0.0000 * (vt_ur ** 1.0000) * (eta ** 1.0000) * ((ros / 1000) ** 1.0000),
    'Duddon_spring': lambda vt_ur, eta, ros: 0.0000 * (vt_ur ** 7.8632) * (eta ** 0.8786) * ((ros / 1000) ** -56.0561),
    'Duddon_neap': lambda vt_ur, eta, ros: 0.0517 * (vt_ur ** 4.8587) * (eta ** 2.0092) * ((ros / 1000) ** -57.4813)
}






def test_power_law_equations(calibration_df):
    # Initialize an empty list to store the results
    results = []
    missing_keys = []

    # Iterate over each row in the calibration_df
    for _, row in calibration_df.iterrows():
        estuary = row['estuary']
        tidal_phase = row['tidal_phase']
        best_C_k = row['best_C_k']
        vel_tide = row['vel_tide']
        ur = row['ur']
        eta = row['eta']
        Ro_s = row['Ro_s']

        # Calculate log(vel_tide / ur), ensure no division by zero
        if ur != 0:
            log_vt_ur = np.log(vel_tide / ur)
        else:
            log_vt_ur = np.nan  # Handle the case where ur is zero

        # Generate key for power-law equation (e.g., "Dee_spring")
        key = f"{estuary}_{tidal_phase}"

        # Apply the corresponding equation if available
        if key in power_law_equations and not np.isnan(log_vt_ur):
            calculated_C_k = power_law_equations[key](log_vt_ur, eta, Ro_s)
            # Calculate the error between the best C_k and the calculated C_k
            error = np.abs(best_C_k - calculated_C_k)
        else:
            # If the key doesn't exist, or the log_vt_ur is invalid, set the calculated_C_k and error as NaN
            calculated_C_k = np.nan
            error = np.nan
            missing_keys.append(key)  # Log missing keys

        # Append the result to the list
        results.append({
            'estuary': estuary,
            'tidal_phase': tidal_phase,
            'best_C_k': best_C_k,
            'calculated_C_k': calculated_C_k,
            'error': error
        })

    # Convert the results into a DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    
    # Log missing keys
    if missing_keys:
        print(f"Missing keys: {set(missing_keys)}")
    
    # Display the results
    return results_df

# Test the function again with missing key logging
comparison_results_df = test_power_law_equations(calibration_df)

# Display the comparison results
comparison_results_df.to_csv('cal_attempt_3.csv')

# Check for negative values in the 'calculated_C_k' column
negative_ck_values = comparison_results_df[comparison_results_df['calculated_C_k'] < 0]

# Compare 'best_C_k' with 'calculated_C_k'
comparison_results_df['C_k_difference'] = comparison_results_df['best_C_k'] - comparison_results_df['calculated_C_k']

# Get summary statistics for the differences
ck_difference_summary = comparison_results_df['C_k_difference'].describe()

# Show the negative C_k values and summary statistics of C_k differences
negative_ck_values, ck_difference_summary


#%% Check calibrated files from an EBM run with new C_k equations against fluhsing times and best fit. 

# Initialize lists to store results for calibrated comparison
calibrated_flushing_times = []
count= 0
# Loop through estuaries and discharges
for estuary in estuary_list:
    for discharge in discharge_list:
        # Construct file name pattern for the calibrated results
        file_pattern = f"{estuary}_discharge_{discharge}__"
        
        # Find matching files in the new simulation results folder
        for filename in os.listdir(new_folder_path):
            if filename.startswith(file_pattern) and filename.endswith("multivariate_regression.npz"):
                file_path = os.path.join(new_folder_path, filename)
                print(f"Loading file: {file_path}")
                data = np.load(file_path)
                count += 1
                # Extract variables from the file
                S_u = data['sal_out_mean']  # Salinity outflow upper layer
                S_l = data['sal_in_mean']  # Salinity inflow lower layer
                #data['dis_out_mean']  # Outflow discharge
                
                # Collect the eta, ur, vel_tide and ros data. 
                eta_series = data['all_eta']
                vel_tide_series = data['all_vel_tide']
                ur_series = data['all_ur']
                ros_series = data['all_ros']
                ck_series = data['all_Ck']
                
                eta_cycles = extract_tidal_cycles(eta_series)
                vel_tide_cycles = extract_tidal_cycles(vel_tide_series)
                ur_cycles = extract_tidal_cycles(ur_series)
                Ro_s_cycles = extract_tidal_cycles(ros_series)
                C_k_cycles = extract_tidal_cycles(ck_series)
                
                min_len = min(len(S_u), len(S_l))
                
                R_outflow = np.full(min_len, discharge)
                
                # Calculate mean salinities and discharges
                S_out_mean = (S_u[min_len:])  # Salinity in
                S_in_mean = (S_l[min_len:])  # Salinity out
                Q_out_mean = (R_outflow)  # Outflow discharge
                
                
                
                # Make sure the dataset is long enough to include datapoints 19 and 35
                if min_len > spring_datapoint and min_len > neap_datapoint:
                    eta_spring = eta_cycles[spring_datapoint]
                    vel_tide_spring = vel_tide_cycles[spring_datapoint]
                    ur_spring = ur_cycles[spring_datapoint]
                    Ro_s_spring = Ro_s_cycles[spring_datapoint]
                    # Extract values at the spring and neap tide points
                    S_out_spring = S_u[spring_datapoint]
                    S_in_spring = S_l[spring_datapoint]
                    Q_out_spring = R_outflow[spring_datapoint]
                    C_k_spring = C_k_cycles[spring_datapoint]
                    
                    
                    # Neaps 
                    S_out_neap = S_u[neap_datapoint]
                    S_in_neap = S_l[neap_datapoint]
                    Q_out_neap = R_outflow[neap_datapoint]
                    # Extra Neaps
                    eta_neap = eta_cycles[neap_datapoint]
                    vel_tide_neap = vel_tide_cycles[neap_datapoint]
                    ur_neap = ur_cycles[neap_datapoint]
                    Ro_s_neap = Ro_s_cycles[neap_datapoint]
                    C_k_neap = C_k_cycles[neap_datapoint]
                    # Calculate flushing time at spring tide
                    ft_spring =( (volume[estuary_list == estuary] * (S_in_spring - S_out_spring)) / (Q_out_spring * S_in_spring) ) / 3600
                    
                    # Calculate flushing time at neap tide
                    ft_neap =( (volume[estuary_list == estuary] * (S_in_neap - S_out_neap)) / (Q_out_neap * S_in_neap) ) / 3600
                    
                    # Store the results for both tidal phases
                    calibrated_flushing_times.append({
                        'estuary': estuary,
                        'discharge': discharge,
                        'flushing_time': ft_spring,
                        'S_in_mean': S_in_spring,
                        'S_out_mean': S_out_spring,
                        'eta': eta_spring,
                        'vel_tide': vel_tide_spring,
                        'ur': ur_spring,
                        'Ro_s': Ro_s_spring,
                        'C_k':C_k_spring,
                        'tidal_phase': 'spring'
                    })
                    
                    calibrated_flushing_times.append({
                        'estuary': estuary,
                        'discharge': discharge,
                        'flushing_time': ft_neap,
                        'S_in_mean': S_in_neap,
                        'S_out_mean': S_out_neap,
                        'eta': eta_neap,
                        'vel_tide': vel_tide_neap,
                        'ur': ur_neap,
                        'Ro_s': Ro_s_neap,
                        'C_k': C_k_neap,
                        'tidal_phase': 'neap'
                    })

calibrated_flushing_df = pd.DataFrame(calibrated_flushing_times)
calibrated_flushing_df.to_csv('calibrated_flushing_times.csv')

#% Analysis to see what went wrong 
# Define a function to remove brackets and convert to float
def clean_and_convert(value):
    # Check if the value is a string with square brackets
    if isinstance(value, str):
        # Remove the brackets and convert the string to float
        value = value.replace('[', '').replace(']', '')
        return float(value)
    elif isinstance(value, list):
        # If it's a list, extract the first value and return as float
        return float(value[0])
    else:
        return float(value)  # Handle the normal case where it's already a float



comparison_df = pd.merge(
    calibration_df, 
    calibrated_flushing_df, 
    on=['estuary', 'discharge', 'tidal_phase'], 
    suffixes=('_original', '_calculated')
)# Compare C_k and flushing times
comparison_df['modeled_flushing_time'] = comparison_df['modeled_flushing_time'].apply(clean_and_convert)
comparison_df['flushing_time_difference'] = comparison_df['modeled_flushing_time'] - comparison_df['flushing_time']
comparison_df['C_k_difference'] = comparison_df['best_C_k'] - comparison_df['C_k']

#% Second time around of recalibrating Ck

# Correction term: Calculate the average overestimation
correction_term = comparison_df['flushing_time_difference'].mean()

# Apply a correction factor to scale down overestimated C_k values
scaling_factor = 0.9  # Adjust this factor based on observed patterns

# Re-calibrate C_k using a more conservative regression model
adjusted_C_k = comparison_df['best_C_k'] - correction_term
adjusted_C_k *= scaling_factor

# Update the dataframe with the new C_k values
comparison_df['Ck'] = adjusted_C_k

# Re-run the regression with corrected values
for estuary in comparison_df['estuary'].unique():
    estuary_data = comparison_df[comparison_df['estuary'] == estuary]

    # Feature matrix (log(vel_tide / ur) and eta) with adjusted values
    X = np.column_stack([
        np.log(estuary_data['vel_tide_calculated'] / estuary_data['ur_calculated']),
        estuary_data['eta_calculated']
    ])
    y = estuary_data['Ck'].values

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Update the coefficients for each estuary
    intercept = model.intercept_
    coef_vel_tide_ur = model.coef_[0]
    coef_eta = model.coef_[1]

    # Print the adjusted equation for each estuary
    print(f"{estuary}: C_k = {intercept:.4f} + ({coef_vel_tide_ur:.4f} * log(vel_tide / ur)) + ({coef_eta:.4f} * Eta)")
    
#%% Make a table

# Sample estuary data for initial creation
table_structure = {
    "Release": [],
    "Datetime": [],
    "Range (m)": [],
    "1 m3/s": [],
    "2 m3/s": [],
    "5 m3/s": [],
    "10 m3/s": [],
    "20 m3/s": [],
    "30 m3/s": [],
    "40 m3/s": [],
    "50 m3/s": []
}

# Placeholder data for the given estuaries in the image to demonstrate the recreated format
estuaries = ["DEE", "KENT", "LUNE", "RIBBLE", "DUDDON", "LEVEN", "MERSEY", "WYRE"]

# Sample ranges for each estuary as seen in the image
sample_data = {
    "DEE": [136, 66, 124, 187],
    "KENT": [149, 55, 88, 200],
    "LUNE": [112, 43, 63, 187],
    "RIBBLE": [97, 27, 61, 110],
    "DUDDON": [85, 3, 49, 85],
    "LEVEN": [112, 13, 51, 150],
    "MERSEY": [447, 353, 411, 508],
    "WYRE": [136, 67, 61, 224]
}

# Initialize table with estuary data
for estuary in estuaries:
    for tidal_phase in ["Neap", "Mid+", "Spring", "Mid-"]:
        table_structure["Release"].append(tidal_phase)
        table_structure["Datetime"].append("Sample Datetime")
        table_structure["Range (m)"].append(sample_data[estuary][0] if tidal_phase == "Neap" else sample_data[estuary][1])
        table_structure["1 m3/s"].append(sample_data[estuary][2])
        table_structure["2 m3/s"].append(sample_data[estuary][2])
        table_structure["5 m3/s"].append(sample_data[estuary][2])
        table_structure["10 m3/s"].append(sample_data[estuary][2])
        table_structure["20 m3/s"].append(sample_data[estuary][3])
        table_structure["30 m3/s"].append(sample_data[estuary][3])
        table_structure["40 m3/s"].append(sample_data[estuary][3])
        table_structure["50 m3/s"].append(sample_data[estuary][3])

# Now create a DataFrame from this structured dictionary
df_recreated_table = pd.DataFrame(table_structure)
# First, let's filter the calibration data to get the relevant data for spring and neap phases
df_calibration = calibration_df

df_ebm_spring = df_calibration[df_calibration['tidal_phase'] == 'spring']
df_ebm_neap = df_calibration[df_calibration['tidal_phase'] == 'neap']

# Reset the index for easier matching
df_ebm_spring = df_ebm_spring.reset_index(drop=True)
df_ebm_neap = df_ebm_neap.reset_index(drop=True)

# Now, let's add these modeled flushing times to replace the "Mid+" and "Mid-" values in the recreated table
# Assuming "Mid+" corresponds to the EBM spring and "Mid-" to EBM neap

# Replace the "Mid+" rows with EBM spring values from the CSV
df_recreated_table.loc[df_recreated_table['Release'] == "Mid+", '1 m3/s'] = df_ebm_spring['modeled_flushing_time']
df_recreated_table.loc[df_recreated_table['Release'] == "Mid-", '1 m3/s'] = df_ebm_neap['modeled_flushing_time']

# Saving the updated table
updated_table_with_ebm_path = 'updated_estuary_table_with_ebm.csv'
df_recreated_table.to_csv(updated_table_with_ebm_path, index=False)

updated_table_with_ebm_path
