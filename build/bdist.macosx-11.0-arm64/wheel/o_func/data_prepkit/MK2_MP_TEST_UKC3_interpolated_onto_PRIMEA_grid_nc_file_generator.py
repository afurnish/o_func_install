# -*- coding: utf-8 -*-
"""
This code originally mapped the ukc3 data points onto the PRIMEA GRID.  

Created on Tue Mar 21 20:01:24 2023
@author: aafur
"""
#%% Import Dependencies
import glob
from sys import platform
from o_func import opsys;start_path = opsys()
import pandas as pd
import xarray as xr
from o_functions.near_neigh import data_flattener
import numpy as np
from scipy.spatial import cKDTree
import sys
import time
from joblib import Parallel, delayed

#%% Hidden files

#%% Choice Function
def winc(i):
    if platform == "win32":
        b = i.replace('\\','/')
    else:
        b = i
    return b

#mother_folder = r'/4.Delft_2019/2.Model_refinement'
mother_folder = r'/5.Final/1.friction'
def path_choice(start_path,mother_folder):
    j = 0
    print('\nSelecting nc data folder and file')
    
    input_message = "\nPick an option:\n"
    input_message += 'Your choice: '
    path = start_path + 'modelling_DATA/kent_estuary_project'+ mother_folder  + '/outputs'
    files = []
    user_input = ''
    j_list = []
    for file in glob.glob(path + '/*'):
        #print(file)
        file = winc(file)

        files.append(file)
        j+=1
        print(str(j) + '.)' + file.split('/')[-1])
        j_list.append(str(j))

    while user_input.lower() not in j_list:
        user_input = input(input_message)
    
    new_path = files[int(user_input)-1] + r'/stats_data'
    
    
    
    name = (files[int(user_input)-1]).split('/')[-1]           
    print('You picked: ' + name + '\n')
    return new_path, name # return the path to nc file rather than the folder


#%% Work with the data
path, name = path_choice(start_path,mother_folder)
ukc3_path = start_path + r'modelling_DATA/kent_estuary_project/5.Final/1.friction/SCW_runs/ukc3_files'
primea = xr.open_dataset(glob.glob(path + '/*PRIMEA.nc')[0].replace('\\','/'))
ukc3 = xr.open_dataset(glob.glob(ukc3_path + r'/*UKC3_data.nc')[0].replace('\\','/'))
'''
I need to map the ukc3 dataset onto the primea dataset
Sync them up time point for timepoint, location for location
perform r squared analysis over the region
then create colour plot to show the results of the r squared analysis
also perform the first r squared analysis but for PRIMEA and UKC3
'''
#prepare data
lon_ukc3 = ukc3.nav_lon.values
lat_ukc3 = ukc3.nav_lat.values
combined = data_flattener(lon_ukc3, lat_ukc3)
sh = primea.surface_height

#set data into format for near_neigh analysis
df_primea = pd.DataFrame(columns = ['x','y'])
df_ukc3 = pd.DataFrame(columns = ['x','y'])

df_primea.x = primea.lon.values
df_primea.y = primea.lat.values
df_ukc3.x = combined[:,1]
df_ukc3.y = combined[:,0]

tree = cKDTree(list(zip(np.ravel(np.dstack([lon_ukc3.ravel()])[0]), np.ravel(np.dstack([lat_ukc3.ravel()])[0]))))
#%% Perform interpolation analysis
z_interpolated_time = np.empty((0, 69866))
p = 2
j = -1


def interpolate(val):
    #store the data vars into variabes
    ukc3_sh = ukc3.sossheig[val].values
    print(ukc3_sh)
    vosaline_top = ukc3.vosaline_top[val].values
    #stack the data
    stacked_sh = np.dstack([ukc3_sh.ravel()])[0]
    stacked_sal = np.dstack([vosaline_top.ravel()])[0]

    z_interpolated_sh = []
    z_interpolated_sal = []
    for i, folder in enumerate(df_primea.x):
        distances, indices = tree.query([df_primea.x[i], df_primea.y[i]], k=4)
        #weights = 1 / distances**p
        #z_interpolated.append(np.sum(weights * np.array(stacked_sh)[indices]) / np.sum(weights))
        z_interpolated_sh.append(np.array(stacked_sh)[indices][0][0])
        z_interpolated_sal.append(np.array(stacked_sal)[indices][0][0])
    z_interpolated_sh = np.array(z_interpolated_sh)
    z_interpolated_sal = np.array(z_interpolated_sal)
    #filename = f"data_{val}.npy" 
    #np.save(filename, z_interpolated)
    print('loop :' + str(i))


    return z_interpolated_sh,z_interpolated_sal


#%% Parallel processing

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())


#loop = ukc3.sossheig.shape[0]
loop = ukc3.sossheig.shape[0]
#loop = 100

#%% Run parallel
start_time= time.time()
if __name__ == '__main__':
    results = Parallel(n_jobs=12)(delayed(interpolate)(x) for x in range(loop))

end_time= time.time()
diff = end_time-start_time
print(diff)

sh_array = [t[0] for t in results]
sal_array = [t[1] for t in results]

combined_sh_array = np.stack(sh_array,axis = 0)
combined_sal_array = np.stack(sal_array,axis = 0)


#%% Generation of nc file
ds = xr.Dataset()
# add the longitude and latitude as variables
ds['lon'] = xr.DataArray(df_primea.x, dims='points')
ds['lat'] = xr.DataArray(df_primea.y, dims='points')
ds['time'] = xr.DataArray(ukc3.time_counter.values[:100], dims='time')


# add the surface_height data as a variable with dimensions time, points
ds['z_interpolated_time'] = xr.DataArray(combined_sh_array, dims=('time', 'var')) #surface height
ds['salinity'] = xr.DataArray(combined_sal_array, dims=('time', 'var'))
# add the time coordinate as a dimension and variable

# save the dataset as a NetCDF file
ds.to_netcdf(ukc3_path + r'/UKC3_og_interpolated_PRIMEA_grid_data3.nc')

# UKC3_interp = xr.open_dataset(path + '/surface_height_UKC3_og_interpolated_PRIMEA_grid.nc')
# ukc3_tim_df = pd.DataFrame(UKC3_interp.time.values, columns = ['ukc3_time'])
