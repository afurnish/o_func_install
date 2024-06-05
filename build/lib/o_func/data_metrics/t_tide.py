# -*- coding: utf-8 -*-
"""
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

heysham = join(start_path, 'modelling_DATA','kent_estuary_project','validation','tidal_validation','1.reformatted', 'heysham.csv')


#%% Set up path locations
vel_nor = join(loc_of_FES2014, 'northward_velocity')
vel_eas = join(loc_of_FES2014, 'eastward_velocity')
tide_ocean = join(loc_of_FES2014, 'ocean_tide')
tide_extrapolated = join(loc_of_FES2014, 'ocean_tide_extrapolated')
tide_load = join(loc_of_FES2014, 'load_tide')


test_model_location = join(start_path ,'modelling_DATA','kent_estuary_project','6.Final2','models','02_kent_1.0.0_UM_wind','shortrunSCW_kent_1.0.0_UM_wind')
#  The one below is the upsisde down (actual right way up model)
# test_model_location = join(start_path ,'modelling_DATA','kent_estuary_project','7.met_office','models','upside_down_ocean_bound','runSCW_upside_down_ocean_bound')


water_bc_file = join(test_model_location, 'WaterLevel.bc')
points_file = join(test_model_location, '001_delft_ocean_boundary_UKC3_b601t688_length-87_points.pli')

#%% Define estuaries class
class est_tide:
    def __init__(self):
        pass

    def read_pli(self,points_file):
        '''
        Reads a pli file and returns it as a data array. 
        Parameters
        ----------
        points_file : .pli style file

        Returns
        -------
        Returns an array of lat/lonor x/y

        '''

        # Initialize an empty array
        data_array = np.empty((0, 2), dtype=float)
        # Read data from CSV file
        with open(points_file, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            next(reader)
            # Skip the header line if there is one
            for i, row in enumerate(reader):
                print(row)
                if i != 0:
                    data_array = np.append(data_array, [[float(row[0]), float(row[2])]], axis=0)
        
        # Display the resulting array
        #print(data_array)
        return data_array
    
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
        
        points = self.read_pli(points_to_predic)
        print(points.shape)
        x, y = points[:,0], points[:,1]
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
                points_rad = np.radians(points)
        
                distances, indices = ball_tree.query(points_rad, k=1)
                # print(indices)
                
                lat_indicies, lon_indicies = np.unravel_index(indices, test_data.phase.shape)
                lon_ind, lat_ind = [], []
                [lon_ind.append( ((lon[i].values) + 180) % 360 - 180) for i in lon_indicies[:,0]]
                [lat_ind.append(lat[i].values) for i in lat_indicies[:,0]]
                 
                return lat_indicies, lon_indicies, lat_ind, lon_ind 
            
            fes_lat_indicies, fes_lon_indicies, fes_lat_ind, fes_lon_ind = map_to_fes()
            # At this point index 0 is still south. 
            plt.plot(fes_lon_ind,fes_lat_ind, '*r')
            plt.title('Pli points mapped out onto the FES grid')
            
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
    df_amp, df_pha, tc_names = et.est_amp_and_phase_extractor(points_file, h_or_v = 'height')
    df_amp = df_amp/100
    t = et.time_maker('2013-10-30 00:00', '2013-11-30 00:00', 5)
    amp_phase = et.est_predic(t, df_pha, df_amp, tc_names) # sorts out the amplitude and phasing into correct format
#%%
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
    
    
    
    ''' FREQ stays the same, 
        TIME stays the same
        CONS2 stays the same as its just a list of the names of the constituents. 
        the updated list is a list of phases and amplitudes. 
    '''
    amp_phase = et.est_predic(t, df_pha, df_amp, tc_names, point_number_index = -1) # gets new amps and phases based off location. 
    updatedList = [value for i, value in enumerate(amp_phase) if i not in unmatched_index] # removes unwanted tidal cons
    northeta = tt.t_predic(np.array(t), np.array(CONS2), FREQ, np.array(updatedList).astype(float))

#%% PLOT the lines
    def average_and_report_half_hour(timestamps, data):
        """
        Averages data sampled at irregular intervals to hourly averages and reports these at the half-hour mark.
    
        Parameters:
        - timestamps (list of datetime.datetime): Timestamps for each data point.
        - data (list or np.array): Array of data points corresponding to each timestamp.
    
        Returns:
        - new_times (pd.DatetimeIndex): New timestamps at the half-hour marks after each hour of averaged data.
        - new_data (np.array): Hourly averaged data reported at these half-hour marks.
        """
        # Ensure timestamps is a pandas datetime series for consistent manipulation
        timestamps = pd.to_datetime(timestamps)
        
        # Create a DataFrame from the timestamps and data
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Data': data
        }).set_index('Timestamp')
        
        # Resample to hourly averages. 'H' stands for hourly.
        df_hourly = df.resample('H').mean()
        
        # Shift the resampled data's index by +30 minutes to align the reporting with the half-hour mark
        df_hourly.index = df_hourly.index + pd.Timedelta(minutes=30)
        
        # Extract the new timestamps and their corresponding averaged data
        return df_hourly.index, df_hourly['Data'].values

    newt, neweta = average_and_report_half_hour(t, eta)
    newt, newnortheta = average_and_report_half_hour(t, northeta)
    # Set ticks for every week
    fig = plt.subplots(figsize=(20,7))
    
    '''
    To perform an accurate comparrison as an investigation into the phase relationship
    of the datasets the same temporal analysis must be carried out. UKC4 data is 
    averaged on the hour and then reported on the half hour so any anlysis of this 
    must follow the same protocol. 
    '''
    plt.plot(t,eta, label = 'FES_predicted southerly boundary point (0)')
    plt.plot(t, northeta, label = 'FES_predicted northerly boundary point (86)')
    
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')  # Adjust rotation and alignment as needed

    bot = np.loadtxt('gauge_south.txt')
    top = np.loadtxt('gauge_north.txt')
    # Extract the columns
    botseconds_since_start = bot[:, 0]
    botvalues = bot[:, 1]
    
    topseconds_since_start = top[:, 0]
    topvalues = top[:, 1]
    
    start_date = datetime(2013, 10, 30, 0, 30, 0) # NEEDS TO BE SET FROM WHATEVER DATSET IT CAME FROM, I have set it to OA
    date_column = [start_date + timedelta(seconds=int(seconds)) for seconds in botseconds_since_start]
    
    # Create a Pandas DataFrame
    df = pd.DataFrame({'Datetime': date_column, 'Top': topvalues, 'Bot': botvalues})
    
    plt.plot(df.Datetime, df.Top, 'r', label = 'Northerly Latitude UKC4 forcing')
    plt.plot(df.Datetime, df.Bot, 'g', label = 'Southerly Latitude UKC4 forcing')
    
    
    #plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H %M')) 
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Adjust the format as needed
    plt.xlim(t[400], t[1400])
    plt.xlabel('Hours')
    plt.legend()
#%% Notes 
    '''
    %Y: Year with century as a decimal number (e.g., 2023).
    %y: Year without century as a zero-padded decimal number (e.g., 23 for 2023).
    %m: Month as a zero-padded decimal number (01, 02, ..., 12).
    %b: Abbreviated month name (Jan, Feb, ..., Dec).
    %B: Full month name (January, February, ..., December).
    %d: Day of the month as a zero-padded decimal number (01, 02, ..., 31).
    %a: Abbreviated weekday name (Sun, Mon, ..., Sat).
    %A: Full weekday name (Sunday, Monday, ..., Saturday).
    %H: Hour (00, 01, ..., 23).
    %I: Hour (00, 01, ..., 12).
    %p: AM or PM.
    %M: Minute (00, 01, ..., 59).
    %S: Second (00, 01, ..., 59).
    You can combine these format codes to create custom date formats. For example:
    
    '%Y-%m-%d': 2023-12-31
    '%b %d, %Y': Dec 31, 2023
    '%A, %B %d, %Y': Saturday, December 31, 2023
    '%I:%M %p': 12:30 PM
    '''
    a =1
#%%
et = est_tide()
df_amp, df_pha, tc_names = et.est_amp_and_phase_extractor('tide_gauges.pli', h_or_v = 'height')
df_amp = df_amp/100
t = et.time_maker('2013-10-30 00:00', '2013-11-30 00:00', 5)
amp_phase = et.est_predic(t, df_pha, df_amp, tc_names)
#%%
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

#%%
updatedList = [value for i, value in enumerate(amp_phase) if i not in unmatched_index]


# Format each unique name in the desired two-dimensional style
CONS = [[name] for name in unique_names]

FREQ = np.array(result_df.Freq)
import ttide as tt

CONS2 = [item[0].encode('utf-8') for item in CONS]
#%%

eta = tt.t_predic(np.array(t), np.array(CONS2), FREQ, np.array(updatedList).astype(float))

import matplotlib.dates as mdates

#%% PLOT the lines
# Set ticks for every week
fig = plt.subplots(figsize=(20,7))
plt.plot(t,eta, 'b',label = 'FES_predicted tide gauge')
    
hey = pd.read_csv(heysham, parse_dates=['Date'])

y = hey.Height-1.1
plt.plot(hey.Date - pd.to_timedelta('0 hours'), y, 'r', label = 'Heysham Tide Gauge')

plt.xlim(t[0], t[-1])
plt.ylim(min(eta)-2, max(eta)+2)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H %M')) 
plt.legend()
plt.xlim([t[0], t[2000]])
'''
I think I need to set a timeseries data point that can passed to another function in the suite. 
'''

# tidal_analysis = tt.t_tide(hey.Height, dt = 3600, lat = 54)

'''
001_Heysham_and_Liverpool_Tide_Gauges
    2    2
-2.937363000000000E+000  5.403143800000000E+001 haysham_0001
-3.042241000000000E+000  5.345216300000000E+001 liverpool_0002
'''

if __name__ == '__main__':
    # Generate astronomical forcings for Delft script. 
    output_path = 'WaterLevel.bc'
    et = est_tide()
    df_amp, df_pha, tc_names = et.est_amp_and_phase_extractor(points_file, h_or_v = 'height')
    df_amp = df_amp/100
    t = et.time_maker('2013-10-30 00:00', '2013-11-30 00:00', 5)
    # amp_phase = et.est_predic(t, df_pha, df_amp, tc_names) # sorts out the ampl
    # For this it will put out the relevant information and generate the bc files. 
    
        
    # Fixed parts of the data
    function = "astronomic"
    quantity1 = "astronomic component"
    unit1 = "-"
    quantity2 = "waterlevelbnd amplitude"
    unit2 = "m"
    quantity3 = "waterlevelbnd phase"
    unit3 = "deg"
    
    # Tidal constituent, amplitude, and phase (you can make these variables if they need to change)
    constituents  = ["  M2"]#, "  S2"]
    base_name = "m2_ocean_"

    with open(output_path, 'a') as file:
        file.write('')
    with open(output_path, 'w') as file:
        for i in range(len(df_amp)):
            # Write the header for each forcing block
            file.write("[forcing]\n")
            file.write(f"Name                            = {base_name}{i + 1:04d}\n")
            file.write("Function                        = astronomic\n")
            file.write("Quantity                        = astronomic component\n")
            file.write("Unit                            = -\n")
            file.write("Quantity                        = waterlevelbnd amplitude\n")
            file.write("Unit                            = m\n")
            file.write("Quantity                        = waterlevelbnd phase\n")
            file.write("Unit                            = deg\n")
    
            # Prepare data for all constituents in this section
            data_lines = []
            for const in constituents:
                if const in df_amp.columns and const in df_pha.columns:
                    amplitude = df_amp.at[i, const]
                    phase = df_pha.at[i, const]
                    data_lines.append(f"{const.strip()}  {amplitude:.3f}  {phase:.3f}")
    
            # Write all constituent data at once, properly formatting to avoid unnecessary new lines
            file.write("\n".join(data_lines) + "\n\n")
    
    print("File written successfully.")