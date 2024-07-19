# -*- coding: utf-8 -*-
"""
Tide plotter that generates a tide at a specific point in time and space. 

Created on Wed Dec  6 10:01:10 2023
@author: aafur
"""

#Install dependencies
from o_func import opsys; start_path = opsys()
from o_func.utilities import near_neigh
from o_func.utilities import uk_bounds_wide
from o_func.utilities import tidconst; tc = tidconst()

from sklearn.neighbors import BallTree
from datetime import datetime, timedelta
from os.path import join as join
import glob
import xarray as xr
import csv
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

import ttide as tt
import matplotlib.dates as mdates

[plt.close() for i in range(30)]
'''
SANINITY CHECK TO SELF

So I have discovered I definately have flipped the forcing data which is instoducing some phase lag. 
This was discovered when on original file at time one most northerly point (top) 87, number was -2.16, 
yet at southerly point 01 number was -2.52. When analysing using phase analysis,
The tide gauge is plotted here at the most southerly point, indexing zero (points file 01) which lines up 
with the northerly point (top) -2.16, which suggests to me that the points to force the ocean boundary were flipped in the NS line. 

The code will be rectified as of the 7th feb 2024. 

'''

#Location of main data in Original Dataset file. 
dataset = 'UK_bounds' # 'world
loc_of_FES2014 = join(start_path, 'Original_Data','FES2014', dataset)

#%% Set up path locations
vel_nor = join(loc_of_FES2014, 'northward_velocity')
vel_eas = join(loc_of_FES2014, 'eastward_velocity')
tide_ocean = join(loc_of_FES2014, 'ocean_tide')
tide_extrapolated = join(loc_of_FES2014, 'ocean_tide_extrapolated')
tide_load = join(loc_of_FES2014, 'load_tide')

#%% Define estuaries class
class est_tide:
    def __init__(self):
        pass

    def est_amp_and_phase_extractor(self,points_to_predic, h_or_v = 'height'):
        self.points_to_predic = points_to_predic
        bounds = uk_bounds_wide()
        '''
        est_predic operates similarly to t_predic but only needs to take 
        a .pli file from delft to function. Operates off all constituents unless
        specified otherwise. 

        Parameters
        ----------
        points_to_predic : .pli file that can be generated locally or from delft 3d fm suite

        Returns
        -------
        A tidal timeseries at each of the points

        '''
        #%% Load rest of nc Data
        if h_or_v == 'height':
            data_path =  tide_ocean
        
        points = points_to_predic
        # print(points.shape)
        x, y = points[0], points[1]
        # import pdb; pdb.set_trace()
        #Return an array of x and y
        #data_array[:,0] X and data_array[:,1] Y
        
        files = []
        [files.append(file) for file in (sorted(glob.glob(join(data_path, '*'))))]
        '''
        files is now a list of tidal constituent nc data files. 
        '''
        
        
        #tc_names = [path.split('/')[-1].split('.')[0].upper().rjust(4) for path in files]
        import os

        tc_names = [os.path.splitext(os.path.split(path)[1])[0].upper().rjust(4) for path in files]
        


        for k, file in enumerate(files):
            print(tc_names[k])
        
            test_data = xr.open_dataset(file)
            if k == 0:
                print(test_data.phase)
                print(test_data.amplitude)
            lon, lat = test_data.lon, test_data.lat
            lons, lats = np.meshgrid(lon, lat)
            if k == 0:
                '''
                This figure as of present config just plots the coastline with the UKC4 dataset overlaid 
                ontop of it. 
                '''
                fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
                c = ax.pcolormesh(lon, lat, test_data.phase, cmap='viridis')
                ax.coastlines(linewidth=1, edgecolor='black')
                
                # Add colorbar
                plt.colorbar(c, ax=ax, label='Data Values')
                plt.show
            
            def map_to_fes():
                lon_2d, lat_2d = np.meshgrid(lon, lat)
                lon_lat_array = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))
                lon_lat_array[:, 0] = (lon_lat_array[:, 0] + 180) % 360 - 180
        
                
                # lon_lat_array = np.column_stack((lon.values, lat.values))
                # print(lon_lat_array)
                lon_lat_array_rad = np.radians(lon_lat_array)
        
                ball_tree = BallTree(lon_lat_array_rad, metric='haversine')
                points_rad = np.radians(np.array(points))
        
                distances, indices = ball_tree.query(points_rad.reshape(1,-1), k=1)
                print(indices)
                # print(indices)
                
                lat_indicies, lon_indicies = np.unravel_index(indices, test_data.phase.shape)
                lon_ind, lat_ind = [], []
                [lon_ind.append( ((lon[i].values) + 180) % 360 - 180) for i in lon_indicies[:,0]]
                [lat_ind.append(lat[i].values) for i in lat_indicies[:,0]]
                 
                return lat_indicies, lon_indicies, lat_ind, lon_ind 
            
            fes_lat_indicies, fes_lon_indicies, fes_lat_ind, fes_lon_ind = map_to_fes()
            # At this point index 0 is still south. 
            plt.plot(fes_lon_ind,fes_lat_ind, '*r')
            # plt.title('Pli points mapped out onto the FES grid')
            
            def map_to_ukc4():
                pass
            '''
            These are now the points that can be used in tidal generation with FES data 
            '''
            # print(type(lon_ind))
            # Convert x and y indices to NumPy arrays
            x_indices = np.array(fes_lon_ind)
            y_indices = np.array(fes_lat_ind)
        
            amp = np.array([])
            pha = np.array([])
    
            for i, l in enumerate(fes_lat_indicies): 
                pha = np.append(pha,test_data.phase[l[0], fes_lon_indicies[i]].values)
                amp = np.append(amp,test_data.amplitude[l[0], fes_lon_indicies[i]].values)
            # print(pha)
            # print(amp)
            
            # Create a dictionary to collect the data
            amp_dic = {tc_names[k]:amp}
            pha_dic = {tc_names[k]:pha}
            
            # Convert the dictionary to a DataFrame
            if k == 0:
                # If it's the first iteration, create the DataFrame
                df_amp = pd.DataFrame(amp_dic)
                df_pha = pd.DataFrame(pha_dic)
            else:
                # If it's not the first iteration, append the new DataFrame to the existing one
                df_amp = pd.concat([df_amp, pd.DataFrame(amp_dic)], axis=1)
                df_pha = pd.concat([df_pha, pd.DataFrame(pha_dic)], axis=1)

        return df_amp, df_pha, tc_names 
        # assign some variable to represent each dataset that is going to get passed through    


    def time_maker(self, time_in, time_out, timestep):
        '''
        This makes timeseries suitable for this suite. 

        Parameters
        ----------
        time_in : Should be string in format of yyyy-mm-dd hh:mm:ss
        time_out : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if isinstance(time_in or time_out, str):
            start = datetime.strptime(time_in, '%Y-%m-%d %H:%M')
            end = datetime.strptime(time_out, '%Y-%m-%d %H:%M')
            total_minutes = int((end - start).total_seconds() / 60)
            timestep_minutes = timestep
            t = [start + timedelta(minutes=i) for i in range(0, total_minutes + 1, timestep_minutes)]
            # print(t)
        return t


    def est_predic(self, t, df_amp, df_pha, tc_names, point_number_index = 0):
        amp_phase = []
        
        point_number = point_number_index # from 0 to 86
        # [amp_phase.append(([df_amp[i][point_number],0, df_pha[i][point_number],0])) for i in tc_names]

        [amp_phase.append(([df_pha[i].iloc[point_number],0, df_amp[i].iloc[point_number],0])) for i in tc_names]
            # eta = tt.t_predic(np.array(t), cons2, FREQ, tidecons2)
        print(amp_phase)
        return amp_phase
        
    def est_compare(self, tidal_timeseries):
        pass

if __name__ == '__main__':
    et = est_tide()
    # data = et.read_pli(points_file) # produces table 
    # For this its generating the points at the ocean boundary for the locations. 
    lat = 53.715130
    lon = -3.0565639
    
    #vorf value 4721 53.7120 -003.0560 05.0256
    df_amp, df_pha, tc_names = et.est_amp_and_phase_extractor([lon,lat], h_or_v = 'height')
    df_amp = df_amp/100
    t = et.time_maker('2013-10-30 00:00', '2014-02-28 00:00', 15)
    amp_phase = et.est_predic(t, df_pha, df_amp, tc_names) # sorts out the amplitude and phasing into correct format
    
    
    #%% Perform the tidal prediction from the newly made amplitude and phase 

    matches = []
    matched_names = []
    matched_freqs = []
    unmatched_names = []

    unmatched_index = []
    for j, pattern in enumerate(tc_names):
        # This should be a dataframe
    # Check for matches in the "Names" column
        mask = tc['Names_FESstyle'].str.contains(pattern)
    
        # If there is a match, store the information
        if mask.any():
            matches.append(pattern.strip())  # Remove leading spaces from the pattern
            matched_names.extend(tc.loc[mask, 'Names'].tolist())
            matched_freqs.extend(tc.loc[mask, 'Freq'].tolist())
        else:
            unmatched_index.append(j)
            unmatched_names.append(pattern.strip())

    # Create a new DataFrame with the matched results
    result_df = pd.DataFrame({'Pattern': matches, 'Names': matched_names, 'Freq': matched_freqs})

    # Print the result DataFrame
    print(result_df)
    # Print the unmatched items and their indexes
    print("\nUnmatched Names:")
    for name in unmatched_names:
        print(name)
    unique_names = result_df['Names'].unique()

    updatedList = [value for i, value in enumerate(amp_phase) if i not in unmatched_index]


    # Format each unique name in the desired two-dimensional style
    CONS = [[name] for name in unique_names]
    
    FREQ = np.array(result_df.Freq)
    CONS2 = [item[0].encode('utf-8') for item in CONS]
    
    eta = tt.t_predic(np.array(t), np.array(CONS2), FREQ, np.array(updatedList).astype(float))
    new_eta = eta + 05.0256 - 4.9 
    
#%% Plot the tide. 
    plt.subplot()
    plt.plot(t, new_eta, color  = 'r', label = 'ODN')
    plt.plot(t, eta, color  = 'g', label = 'MSL')
    
    
    plt.xlim([(t[0] + timedelta(weeks=2)), t[-1]])
    plt.title('Tide at mouth of Ribble')

    plt.axhline(y=3, color='r', linestyle='--', label='3 m')
    plt.axhline(y=4, color='b', linestyle='--', label='4 m')
    plt.axhline(y=5, color='k', linestyle='--', label='5 m')
    plt.xlabel('Time')
    plt.ylabel('Eta (ODN Newlyn')
    plt.legend()
    
    from scipy.signal import find_peaks
    
    import pandas as pd
    df = pd.DataFrame({'Time': t, 'ODN_Newlyn': new_eta})
    df['Time'] = pd.to_datetime(df['Time'])
    # df.to_csv('example_tide_data.csv')
    
    # Identify all the peaks in the 'ODN_Newlyn' data
    # #%% 
    # peaks, _ = find_peaks(df['ODN_Newlyn'])
    
    # # Extract the peak values
    # peak_values = df['ODN_Newlyn'].iloc[peaks]
    # peak_times = df['Time'].iloc[peaks]
    
    # # Create a DataFrame for the peaks
    # peaks_df = pd.DataFrame({'Time': peak_times, 'Peak_Height': peak_values}).reset_index(drop=True)
    
    # # Identify the top 5 highest peaks to determine the spring tides
    # top_spring_peaks = peaks_df.nlargest(5, 'Peak_Height').sort_values(by='Time')
    
    # # Define spring periods (3 days before and 3 days after each spring peak)
    # spring_periods = []
    # for _, row in top_spring_peaks.iterrows():
    #     start_time = row['Time'] - pd.Timedelta(days=3)
    #     end_time = row['Time'] + pd.Timedelta(days=3)
    #     spring_periods.append((start_time, end_time))
    
    # # Identify neap periods (gaps between spring periods)
    # neap_periods = []
    # for i in range(len(spring_periods) - 1):
    #     neap_start = spring_periods[i][1]
    #     neap_end = spring_periods[i + 1][0]
    #     neap_periods.append((neap_start, neap_end))
    
    # # Find the maximum peak within each neap period
    # neap_peaks = []
    # for start_time, end_time in neap_periods:
    #     neap_period = peaks_df[(peaks_df['Time'] > start_time) & (peaks_df['Time'] < end_time)]
    #     if not neap_period.empty:
    #         max_neap_peak = neap_period.loc[neap_period['Peak_Height'].idxmax()]
    #         neap_peaks.append(max_neap_peak)
    
    # # Convert lists to DataFrames
    # spring_peaks_df = pd.DataFrame(top_spring_peaks).reset_index(drop=True)
    # neap_peaks_df = pd.DataFrame(neap_peaks).reset_index(drop=True)
    
    # # Find the maximum neap and spring peaks
    # max_neap_peak = neap_peaks_df['Peak_Height'].max()
    # max_spring_peak = spring_peaks_df['Peak_Height'].max()
    
    # # Find the times corresponding to these maximum peaks
    # max_neap_time = neap_peaks_df.loc[neap_peaks_df['Peak_Height'] == max_neap_peak, 'Time'].values[0]
    # max_spring_time = spring_peaks_df.loc[spring_peaks_df['Peak_Height'] == max_spring_peak, 'Time'].values[0]
    
    # # Output the results
    # max_neap_peak, max_neap_time, max_spring_peak, max_spring_time
    
    # plt.scatter([max_neap_time], [max_neap_peak], c = 'black')
    # plt.scatter(max_spring_time, max_spring_peak, c = 'blue')
    
    # summary_table = pd.DataFrame({
    # 'Tide_Type': ['Neap', 'Spring'],
    # 'Max_Peak_Height': [max_neap_peak, max_spring_peak],
    # 'Time': [max_neap_time, max_spring_time]
    # })
    
        
    # print(summary_table)
    
    #%% 
    peaks, _ = find_peaks(df['ODN_Newlyn'])
    
    # Extract the peak values
    peak_values = df['ODN_Newlyn'].iloc[peaks]
    peak_times = df['Time'].iloc[peaks]
    
    # Create a DataFrame for the peaks
    peaks_df = pd.DataFrame({'Time': peak_times, 'Peak_Height': peak_values}).reset_index(drop=True)
    
    # Identify the top 5 highest peaks to determine the spring tides
    top_spring_peaks = peaks_df.nlargest(5, 'Peak_Height').sort_values(by='Time')
    
    # Define spring periods (3 days before and 3 days after each spring peak)
    spring_periods = []
    for _, row in top_spring_peaks.iterrows():
        start_time = row['Time'] - pd.Timedelta(days=3)
        end_time = row['Time'] + pd.Timedelta(days=3)
        spring_periods.append((start_time, end_time))
    
    # Identify neap periods (gaps between spring periods)
    neap_periods = []
    for i in range(len(spring_periods) - 1):
        neap_start = spring_periods[i][1]
        neap_end = spring_periods[i + 1][0]
        neap_periods.append((neap_start, neap_end))
    
    # Find the maximum peak within each neap period
    neap_peaks = []
    for start_time, end_time in neap_periods:
        neap_period = peaks_df[(peaks_df['Time'] > start_time) & (peaks_df['Time'] < end_time)]
        if not neap_period.empty:
            max_neap_peak = neap_period.loc[neap_period['Peak_Height'].idxmax()]
            neap_peaks.append(max_neap_peak)
    
    # Convert lists to DataFrames
    spring_peaks_df = pd.DataFrame(top_spring_peaks).reset_index(drop=True)
    neap_peaks_df = pd.DataFrame(neap_peaks).reset_index(drop=True)
    
    # Find the maximum neap and spring peaks
    max_neap_peak = neap_peaks_df['Peak_Height'].max()
    max_spring_peak = spring_peaks_df['Peak_Height'].max()
    
    # Find the times corresponding to these maximum peaks
    max_neap_time = neap_peaks_df.loc[neap_peaks_df['Peak_Height'] == max_neap_peak, 'Time'].values[0]
    max_spring_time = spring_peaks_df.loc[spring_peaks_df['Peak_Height'] == max_spring_peak, 'Time'].values[0]
    
    # Output the results
    max_neap_peak, max_neap_time, max_spring_peak, max_spring_time
    
    plt.scatter([max_neap_time], [max_neap_peak], c = 'black')
    plt.scatter(max_spring_time, max_spring_peak, c = 'blue')
    
    summary_table = pd.DataFrame({
    'Tide_Type': ['Neap', 'Spring'],
    'Max_Peak_Height': [max_neap_peak, max_spring_peak],
    'Time': [max_neap_time, max_spring_time]
    })
    
        
    print(summary_table)
    
    
#%% 
        
import pandas as pd
import numpy as np
import os

# Function to create the main DataFrame
def create_time_series_df(start_time, end_time, interval, height_value):
    # Create a date range with the specified start time, end time, and interval
    time_range = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    # Create the DataFrame with the time series and the height column
    df = pd.DataFrame({
        'Time': time_range,
        'Height': height_value
    })
    
    return df

# Function to insert flood events into the main DataFrame
def insert_flood_event(main_df, flood_df, flood_column, event_start_time):
    # Find the index of the maximum value in the flood event column
    peak_index = flood_df[flood_column].idxmax()
    
    # Get the flood event data from the peak index
    flood_event = flood_df[flood_column].dropna().reset_index(drop=True)
    
    # Find the index in the main DataFrame where the event should start
    start_index = main_df.index.get_loc(event_start_time) - peak_index
    
    # Insert the flood event data into the main DataFrame
    for i in range(len(flood_event)):
        if 0 <= start_index + i < len(main_df):
            main_df.iloc[start_index + i, main_df.columns.get_loc('Height')] = flood_event[i]
    
    return main_df

# Function to convert time to seconds since a reference time
def time_to_seconds_since(df, reference_time):
    reference_time = pd.to_datetime(reference_time)
    seconds_since = (df.index - reference_time).total_seconds()
    return seconds_since

# Function to write the data to a .bc file
# Function to write the data to a .bc file
def write_bc_file(df, file_name, reference_time, river_names):
    with open(file_name, 'w') as file:
        for name in river_names:
            file.write("[forcing]\n")
            file.write(f"Name                            = {name}_0001\n")
            file.write("Function                        = timeseries\n")
            file.write("Time-interpolation              = linear\n")
            file.write("Quantity                        = time\n")
            file.write(f"Unit                            = seconds since {reference_time}\n")
            file.write("Quantity                        = dischargebnd\n")
            file.write("Unit                            = m3/s\n")
            
            seconds_since = time_to_seconds_since(df, reference_time)
            
            for time, height in zip(seconds_since, df['Height']):
                file.write(f"{int(time)}   {height}\n")
            
            # Write a blank line after each set of data
            file.write("\n")

# Load and prepare the flood event data
qin_file = os.path.join(r"D:/Original_Data/char_idealised_q_values/Q.xlsx")
df_q = pd.read_excel(qin_file, header=None)
df_q = df_q.transpose() - 15
new_column_names = df_q.iloc[161].values.astype(int)
df_q.columns = new_column_names
df_peak_event = df_q.iloc[123:223].reset_index(drop=True)

# Parameters for creating the main DataFrame
start_time = '2013-10-30 00:00:00'
end_time = '2014-02-28 00:00:00'
interval = '15T'
height_value = 5

# Create the main DataFrame
df_new_riv_flood_event = create_time_series_df(start_time, end_time, interval, height_value)
df_new_riv_flood_event.set_index('Time', inplace=True)

# Set the event start time



event_start_time = pd.Timestamp(max_spring_time)#'2014-03-03T12:15:00')  # Example timestamp, replace with your actual event time

# Insert flood event data into the main DataFrame
river_file_path = r'D:\INVEST_modelling\river_boundary_conditions'
river_names = ['Ribble', 'Dee', 'Mersey', 'Leven', 'Lune', 'Wyre', 'Duddon', 'Kent']

reference_time = ['2013-10-31 00:00:00']
ref_heights = [str(i) for i in [5,4,5]]
for j in range(len(reference_time)):
    for i in [0, 35, 85, 185, 385]:
        if i != 0:
            main_df = insert_flood_event(df_new_riv_flood_event.copy(), df_peak_event, 35, event_start_time)
        else:
            main_df = df_new_riv_flood_event.copy()
            
        # Specify the reference time for the .bc file
        
        
        # Specify the output file name
        file_name = os.path.join(river_file_path, f'Discharge_{i}_height_{ref_heights[j]}.bc')
        
        # Write the data to the .bc file
        write_bc_file(main_df, file_name, reference_time[j], river_names)
        
    # Verify the output
    print(main_df.head())
