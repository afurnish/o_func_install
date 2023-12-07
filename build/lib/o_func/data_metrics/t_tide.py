# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:01:10 2023

@author: aafur
"""

#Install dependencies
from o_func import opsys; start_path = opsys()
from o_func.utilities import near_neigh
from o_func.utilities import uk_bounds_wide

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



#Location of main data in Original Dataset file. 
dataset = 'UK_bounds' # 'world
loc_of_FES2014 = join(start_path, 'Original_Data','FES2014', dataset)

#%% Set up path locations
vel_nor = join(loc_of_FES2014, 'northward_velocity')
vel_eas = join(loc_of_FES2014, 'eastward_velocity')
tide_ocean = join(loc_of_FES2014, 'ocean_tide')
tide_extrapolated = join(loc_of_FES2014, 'ocean_tide_extrapolated')
tide_load = join(loc_of_FES2014, 'load_tide')


test_model_location = join(start_path ,'modelling_DATA','kent_estuary_project','6.Final2','models','02_kent_1.0.0_UM_wind','shortrunSCW_kent_1.0.0_UM_wind')
water_bc_file = join(test_model_location, 'WaterLevel.bc')
points_file = join(test_model_location, '001_delft_ocean_boundary_UKC3_b601t688_length-87_points.pli')
#%% Define estuaries class
class est_tide:
    def __init__(self):
        pass

    def read_pli(self,points_file):
        '''

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
        #Return an array of x and y
        #data_array[:,0] X and data_array[:,1] Y
        
        files = []
        [files.append(file) for file in (sorted(glob.glob(join(data_path, '*'))))]
        tc_names = [path.split('/')[-1].split('.')[0].upper().rjust(4) for path in files]
        
        for k, file in enumerate(files):
            print(tc_names[k])
        
            test_data = xr.open_dataset(file)
            lon, lat = test_data.lon, test_data.lat
            lons, lats = np.meshgrid(lon, lat)
    
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            c = ax.pcolormesh(lon, lat, test_data.phase, cmap='viridis')
            ax.coastlines(linewidth=1, edgecolor='black')
            
            # Add colorbar
            plt.colorbar(c, ax=ax, label='Data Values')
            plt.show
            
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
                    
            plt.plot(lon_ind,lat_ind, '*r')
            '''
            These are now the points that can be used in tidal generation with FES data 
            '''
            print(type(lon_ind))
            # Convert x and y indices to NumPy arrays
            x_indices = np.array(lon_ind)
            y_indices = np.array(lat_ind)
        
            amp = np.array([])
            pha = np.array([])
    
            for i, l in enumerate(lat_indicies): 
                pha = np.append(pha,test_data.phase[l[0], lon_indicies[i]].values)
                amp = np.append(amp,test_data.amplitude[l[0], lon_indicies[i]].values)
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
            print(t)


# #making blank adjustable tide maker
# con = np.char.ljust(np.char.upper(consts), 4)
# cons = con.reshape(-1, 1) # this reshapes them according to original formula 
#%%
# amp = []
# pha = []
# fre = []
# consts_names_rearanged = []

# for i,file in enumerate(sorted(glob.glob(new_folder + r'/*.csv'))):
#     df = pd.read_csv(file)
#     print(i)
#     amplitudes = []
#     phases = []
#     frequencies = []
#     for index, row in df.iterrows():
#         name = format_string(row['TC'])

#         if name in cons:
#             print(name)
#             amplitude = row['Amplitude_(m)']
#             phase = row['Phase_(Deg)']
#             frequency = row['Freq_(deg_hour)']
    
#             amplitudes.append(amplitude)
#             phases.append(phase)
#             frequencies.append(frequency)
#             if i == 0:
#                 consts_names_rearanged.append(name)

#     amp.append(amplitudes)
#     pha.append(phases)
#     fre.append(frequencies)
# print(consts_names_rearanged)

# con2 = np.char.ljust(np.char.upper(consts_names_rearanged), 4)
# cons2 = con2.reshape(-1, 1)

# for i,file in enumerate(sorted(glob.glob(new_folder + r'/*.csv'))):
#     freq = fre[i]
#     FREQ = np.array(freq)
    
#     tidecons = []
#     new_amp = amp[i]
#     new_pha =pha[i]
#     for j,file in enumerate(cons2):
#         tidecons.append(([new_amp[j],0, new_pha[j],0]))
        
#     tidecons2 = np.array(tidecons).astype(float)
#     eta = tt.t_predic(np.array(t), cons2, FREQ, tidecons2)
    
#     fig, ax = plt.subplots(figsize = (7,3))
#     ax.plot(t, eta)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
#     plt.xticks(rotation=45)
#     plt.xlim([t[0], t[0] + timedelta(days=3)])
#     plt.xticks()
#     plt.tight_layout()
#     plt.savefig(new_folder2 + '/3_days.png', dpi = 200)
    
#     fig2, ax2 = plt.subplots(figsize = (7,3))
#     ax2.plot(t, eta)
#     ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     plt.xticks(rotation=45)
#     #plt.xlim([t[0], t[25]])
#     plt.xticks()
#     plt.tight_layout()
#     plt.savefig(new_folder2 + '/0all_days.png', dpi = 200)

#     df2 = pd.DataFrame(eta,t)

    
#     csv_filename = r'/point' + r'_' + str(i+1) + r'.csv'
#     to_write = new_folder2 + '/' + csv_filename
#     df2.to_csv(to_write, header = False)

# #%%MAKE THE BOUNDARY FILE
# bc_file = main_folder_path + '/WaterLevel.bc'
# with open(bc_file, "w") as f:
#     f.write("")
#     f.close()
# #Load data and write
# seconds_since = str(start)

# for i,file in enumerate(sorted(glob.glob(new_folder2+'/*.csv'))):
#     #print(file)
#     surface_height = pd.read_csv(file, names = ['time', 'surface_height'])
#     write_text_block_to_file_H(bc_file,locs_of_boundary.Name[i], str(t[0]))
#     data_to_write = surface_height.surface_height
#     new_time = [i[:-3] for i in surface_height.time] #reformat data, remove seconds off the end
#     converted_timeseries = [int(i) for i in (convert_to_seconds_since_date(new_time, seconds_since))] 

#     with open(bc_file, "a") as f:
#         for j,file in enumerate(data_to_write):
#             #print(file)
#             f.write(str(converted_timeseries[j]) + '    ' + str(file) + '\n')             
#         f.write('\n')
        
        # eta = tt.t_predic(np.array(t), cons2, FREQ, tidecons2)
        
    def est_predic(self):
        pass
        
    def est_compare(self, tidal_timeseries):
        pass

if __name__ == '__main__':
    et = est_tide()
    df_amp, df_pha, tc_names = et.est_amp_and_phase_extractor(points_file, h_or_v = 'height')
    et.time_maker('2013-10-01 00:00', '2013-11-01 00:00', 5)
