#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:38:12 2024

@author: af
"""

#%% Import dependecies
import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import pandas as pd
from sklearn.neighbors import BallTree
import re
import subprocess
import pkg_resources
import platform
import time
#homemade packages. 
from o_func.utilities.choices import DataChoice
import o_func.utilities as util
from o_func.data_prepkit import primea_bounds_for_ukc4_slicing  # import boundaries of primea model in ukc4 lon and lats
from o_func import DirGen , opsys; start_path = opsys()

#%% Testing flipping function
flip = 'n'

#%% Is it running in parallel or not
p = 'y'

#%% Immutable variables
'''
There are there file types:
 - U - For tangential velocity
 - V - For normal velocity 
 - T - For temperature, water levels, salinity
 - R - River discharge files 
 - UV - V and V velocities combined for more than one level. 
'''

bc = ".bc"  # Assuming bc is a string that holds the boundary condition extension

BoundaryConditionTypes = {
    'NormalVelocity': {
        'filename': 'NormalVelocity' + bc,
        'script_name': 'normalvelocitybnd',
        'units': 'm/s',
        'function': 'timeseries',
        'filetype': 'U'
    },
    'TangentVelocity': {
        'filename': 'TangentVelocity' + bc,
        'script_name': 'tangentialvelocitybnd',
        'units': 'm/s',
        'function': 'timeseries',
        'filetype': 'V'
    },
    'WaterLevel': {
        'filename': 'WaterLevel' + bc,
        'script_name': 'waterlevelbnd',
        'units': 'm',
        'function': 'timeseries',
        'filetype': 'T'
    },
    'Discharge': {
        'filename': 'Discharge' + bc,
        'script_name': 'dischargebnd',
        'units': 'm³/s',
        'function': 'timeseries',
        'filetype': 'R'
    },
    'Salinity': {
        'filename': 'Salinity' + bc,
        'script_name': 'salinitybnd',
        'units': 'ppt',
        'function': 'timeseries',
        'filetype': 'T'
    },
    'Temperature': {
        'filename': 'Temperature' + bc,
        'script_name': 'temperaturebnd',
        'units': 'degrees',
        'function': 'timeseries',
        'filetype': 'T'  # Still assuming 'T' but keeping in mind it might need a unique identifier
    },
    'Velocity': {
        'filename': 'Velocity' + bc,
        'script_name': 'uxuyadvectionvelocitybnd:ux,uy',
        'units': 'm/s',
        'function': 't3d',
        'filetype': 'UV'
    }
}

def string_to_slice(sliced_str):
    parts = sliced_str.split(':')
    start = int(parts[0]) if parts[0] else None  # Convert to int, or None if empty
    end = int(parts[1]) if parts[1] else None  # Convert to int, or None if empty

    
    return slice(start, end)


Filetype_to_index = {'U':1,
                     'UV':string_to_slice('1:'),
                     'R':'NA',
                     'V':2,
                     'T':0,
                     }

# The user dict is a list of variables that functions can ammend to and so connsecutively use. 
user_dict = {}

def ud(): # A quick function to see current contents of the user_dict. 
    print([i for i in user_dict])
    
# %% Useful Functions for script. 
class LayerError(Exception):
    pass

def convert_to_seconds_since_date(timeseries, date_str):
    """ Convert a timeseries of points to seconds since a specific date.
    Args:timeseries (list): The timeseries of points.np.array
        date_str (str): The date to use as the reference, in the format "YYYY-MM-DD HH:MM:SS".
    Returns: list: The timeseries of points converted to seconds since the reference date.
    """
    converted_dates = (np.array(timeseries) - np.datetime64(date_str)).astype('timedelta64[s]') 
    return converted_dates

def find_index(lonval, latval, lonarray, latarray):
    """ 
    
    """
    lon_indices = np.argwhere(lonarray == lonval).flatten().reshape(-1, 2)
    lat_indices = np.argwhere(latarray == latval).flatten().reshape(-1, 2)
    lon_set = {tuple(row) for row in lon_indices}
    lat_set = {tuple(row) for row in lat_indices}
    common_pairs = lon_set.intersection(lat_set)
    common_indices = np.array(list(common_pairs))
    # Sort the common indices
    common_indices.sort(axis=0)
    common_indices_list = common_indices.tolist()[0]
    return common_indices_list

def UKC3_testcase(full_path):
    """ Collect latitude and longitude data from UKC4 datasets.
    """
    first = sorted(glob.glob( join(full_path,'*')))[0]
    T = xr.open_dataset(first, engine = 'netcdf4')
    z = T.sossheig.values[0,:,:]
    x = T.nav_lon.values
    y = T.nav_lat.values
    return x,y,z

def blank_pli(path, dirname):
    pli_lati = user_dict['pli_lati']
    pli_long = user_dict['pli_long']
    f = open(path , "w")
    f.write(dirname)
    f.write('\n    ' + str(len(pli_lati)) + '    ' + str(2))
    f.close()
    #adjustment = (-3.606832728971--3.5984408138453)*0.6
    f = open(path, "a")
    name = []
    for i in range(len(pli_long)):
        num = i+1
        new_form = format(num, '04')
        new_long = float(format(pli_long[i], '.15f'))
        new_lati = float(format(pli_lati[i], '.15f'))
        new_long = (np.format_float_scientific(new_long,precision = 15, exp_digits=3)).replace('e','E')
        new_lati = (np.format_float_scientific(new_lati,precision = 15, exp_digits=3)).replace('e','E')
        
        #subcode to make every line 15 characters after decimal place
        len_long = len(new_long.split('.')[1].split('E')[0])
        len_lati = len(new_lati.split('.')[1].split('E')[0])
        N1,N2 = [],[]
        if len_long != 15:
            N1 = 15 - len_long
            test_string = new_long.split('.')[1].split('E')[0]
            res = test_string.ljust(N1 + len(test_string), '0')
            final = new_long.split('.')[0] + '.' + res + 'E' + new_long.split('E')[1]
            new_long = final
        if len_lati != 15:
            N2 = 15 - len_lati
            test_string = new_lati.split('.')[1].split('E')[0]
            res = test_string.ljust(N2 + len(test_string), '0')
            final = new_lati.split('.')[0] + '.' + res + 'E' + new_lati.split('E')[1]
            new_lati = final
        
        name.append(dirname + '_' + new_form)
        f.write('\n' + new_long + '  ' + new_lati + ' ' + dirname + '_' + new_form)
    f.close()
    
    user_dict['name'] = name
    
def pli_reader(pli_file):
    # Initialize an empty list to hold the extracted columns
    extracted_columns = []
    
    # Open the file and process its contents
    with open(pli_file, 'r') as file:
        lines = file.readlines()  # Read all lines in the file
    
        # Skip the first two lines (headers) using slicing and then iterate over the remaining lines
        for line in lines[2:]:
            parts = line.split()  # Split the line into parts based on whitespace
            if len(parts) >= 2:  # Ensure there are at least two columns
                first_column = float(parts[0])  # Convert the first column to float
                second_column = float(parts[1])  # Convert the second column to float
                extracted_columns.append((first_column, second_column))  # Append as a tuple

    return extracted_columns
#%% Main Scripts
def write_pli_file_names(forcing_data_path, fn):
    """ This is part of a main function to 
    """
    pli_file = glob.glob(join(fn,'run*', '*points.pli'))[0] # reads pli file from first data entry. 
    
    pli_destination = join(os.path.split((os.path.split(fn)[0]))[0], 'files_bc')
    loc_bounds = primea_bounds_for_ukc4_slicing() # Collect min/max bounds from the UKC4 NEMO grid. 
    user_dict['x'],user_dict['y'],z = UKC3_testcase(forcing_data_path)      # Collect a sample of latitude and longitude for manipulation later.
    user_dict
    lower = find_index(loc_bounds['South']['lon'],loc_bounds['South']['lat'],user_dict['x'], user_dict['y']) # Collects the index of the South, 
    upper = find_index(loc_bounds['North']['lon'],loc_bounds['North']['lat'],user_dict['x'], user_dict['y']) # and North points from the NEMO dataset. 

    # This section could be replaced by a dedicated .pli reader rather than making up the points as you go along. 
    # user_dict['pli_long'], user_dict['pli_lati'] = user_dict['x'][lower[0]:upper[0], lower[1]], user_dict['y'][lower[0]:upper[0], lower[1]]
    extracted_columns = pli_reader(pli_file)
    lon = [item[0] for item in extracted_columns]
    lat = [item[1] for item in extracted_columns]
    
    # Convert the columns to numpy arrays
    user_dict['pli_long'] = np.array(lon, dtype=float)
    user_dict['pli_lati'] = pli_lat = np.array(lat, dtype=float)

    
    
    dirname = '001_delft_ocean_boundary_UKC3_b601t688_length-' + str(len(user_dict['pli_long'])) + '_points' # Should be read in from pli. 
    filepath = join(pli_destination, dirname)
    blank_pli(filepath, dirname) # This generates a pli file in the directory of the model you wish to run. 
    user_dict['var_path'] = forcing_data_path
    
def prepare_model_data():
    T_path = join(user_dict['var_path'],'*T.nc')
    U_path = join(user_dict['var_path'],'*U.nc')
    V_path = join(user_dict['var_path'],'*V.nc')
    
    ### Create a large list of all files. 
    all_files = []
    for i in [T_path, U_path, V_path]:
        files = []
        for file in sorted(glob.glob(i)):
            files.append(file)
        all_files.append(files)
        
    df = pd.DataFrame()
    dataset = xr.open_dataset(all_files[0][0])#start_path + r'Original_Data/UKC3/NEMO_shelftmb/UKC4aow_1h_20131130_20131130_shelftmb_grid_T.nc')
    lon = dataset.sossheig.nav_lon.values
    lat = dataset.sossheig.nav_lat.values
    combined_x_y_arrays = np.dstack([lon.ravel(),lat.ravel()])[0]
    df["Lon"], df["Lat"] = combined_x_y_arrays.T
    
    
    df_locs = pd.DataFrame()
    df_locs['x'] = user_dict['pli_long']; df_locs['y'] = user_dict['pli_lati']
    df_points = pd.DataFrame()
    points = np.array(df_locs[['x','y']].copy()).astype('float32')
    df_points["Lon"], df_points["Lat"] = points.T
    
    for column in df[["Lon", "Lat"]]:
        rad = np.deg2rad(df[column].values)
        df[f'{column}_rad'] = rad
    for column in df_points[["Lon", "Lat"]]:
        rad = np.deg2rad(df_points[column].values)
        df_points[f'{column}_rad'] = rad
    
    ball = BallTree(df[["Lon_rad", "Lat_rad"]].values, metric='haversine')
    distances, indices = ball.query(df_points[["Lon_rad", "Lat_rad"]].values, k = 2)
    
    nn = [] #locations of points
    for i in range(len(indices)):
        nearest_neigh = list(divmod(indices[i,0],user_dict['x'].shape[1])) #1458 is length of row, unravel ravel
        nn.append(nearest_neigh)
    
    user_dict['lhs'] = [item[0] for item in nn] # Indexs of the latitude of the 
    # print('lhs = ', lhs)
    rhs = [item[1] for item in nn] # The rhs should all be identical. 
    user_dict['rhs'] = int(np.mean(rhs))
     

    T_store = ['T'] # The purpose of these stores, are to keep component data seperated. 
    U_store = ['U']
    V_store = ['V']
    R_store = ['R']
    UV_store = ['UV']
    
    for item in user_dict['component']:    # Condition to walk through the input variables like WaterLevel
        if item in BoundaryConditionTypes: # Checks that it can match it with the dictionary. 
            # print(item)
            filetype = BoundaryConditionTypes[item]['filetype']
            if filetype == 'T':
                T_store.append(item)
            elif filetype == 'U':
                U_store.append(item)
            elif filetype == 'V':
                V_store.append(item)
            elif filetype == 'R':
                R_store.append(item)
            elif filetype == 'UV':
                UV_store.append(item)
            else:
                pass
            
    for store in [T_store,U_store, V_store, R_store, UV_store]:
        starttime = time.time()
        # File types is a list of T,U,V,R,UV etc. 
        
        if len(store) > 1:
            file_ripper(store)
            if p == 'y':
                indexer = Filetype_to_index[store[0]]
                para_file_rip(all_files[indexer], store[0])
            else:
                non_para_file_rip(all_files[indexer], store[0])
            file_stitcher()
            endtime = time.time()-starttime
            print('Finished in ',endtime,' seconds')
    
   
def file_ripper(store):
    """ Prepares data paths to send to the data extractor 
    """
    datas = store[1:] # 0 index is the variable type such as TUV etc. 
    print(datas)
    var_list = []
    for i in datas:
        var_list.append(os.path.join(user_dict['layer_path'], i))
    user_dict['var_list'] = var_list # This gets changed each iteration and updated. 
        
    print('Paths to send to file cruncher', var_list)
    
def data_extractor(data_from_dict):
    raw_data = []
    for names in user_dict['var_list']:
        ls = user_dict['lhs']
        rs = user_dict['rhs']
        print(names)
        # try to do in pairs of 3.
        print('Names  ', os.path.split(names)[-1])
        if os.path.split(names)[-1] == 'WaterLevel':
            dataset2 = []
            dataset2.append(np.array(data_from_dict.sossheig[:, ls, rs]))
            
            time0 = data_from_dict.sossheig[0,:,:]
            fig40, ax40 = plt.subplots()
            plt.pcolor(time0)
            plt.plot(([rs]* len(ls)), ls)
            plt.text(rs, ls[0], 'First point 0', c = 'r')
            plt.text(rs, ls[-1], 'Last point '+ str(len(ls)), c = 'r')
            dataset2.append('')
            dataset2.append('')
            raw_data.append(dataset2)
            
        if os.path.split(names)[-1] == 'Salinity':
            dataset2 = []
            dataset2.append(np.array(data_from_dict.vosaline_top[:,ls,rs]))
            if user_dict['layer'] > 1:
                dataset2.append(np.array(data_from_dict.vosaline_mid[:,ls,rs]))
                dataset2.append(np.array(data_from_dict.vosaline_bot[:,ls,rs]))
                
            else:
                dataset2.append('')
                dataset2.append('')
            raw_data.append(dataset2)
        if os.path.split(names)[-1] == 'Temperature':
            dataset2 = []
            dataset2.append(np.array(data_from_dict.votemper_top[:,ls,rs]))
            if user_dict['layer'] > 1:
                dataset2.append(np.array(data_from_dict.votemper_mid[:,ls,rs]))
                dataset2.append(np.array(data_from_dict.votemper_bot[:,ls,rs]))
            else:
                
                dataset2.append('')
                dataset2.append('')
            raw_data.append(dataset2)
        if os.path.split(names)[-1] == 'NormalVelocity':
            dataset2 = []
            dataset2.append(np.array(data_from_dict.vozocrtx_top[:,ls,rs]))
            if user_dict['layer'] > 1:
                dataset2.append(np.array(data_from_dict.vozocrtx_mid[:,ls,rs]))
                dataset2.append(np.array(data_from_dict.vozocrtx_bot[:,ls,rs]))
            else:
                dataset2.append('')
                dataset2.append('')
            raw_data.append(dataset2)
        if os.path.split(names)[-1] == 'TangentVelocity':
            dataset2 = []
            dataset2.append(np.array(data_from_dict.vomecrty_top[:,ls,rs]))
            if user_dict['layer'] > 1:
                dataset2.append(np.array(data_from_dict.vomecrty_mid[:,ls,rs]))
                dataset2.append(np.array(data_from_dict.vomecrty_bot[:,ls,rs]))
            else:
                dataset2.append('')
                dataset2.append('')
            raw_data.append(dataset2)
        if os.path.split(names)[-1] == 'Velocity':
            dataset2 = []
            dataset2.append(np.array(data_from_dict.vozocrtx_top[:,ls,rs]))
            dataset2.append(np.array(data_from_dict.vozocrtx_mid[:,ls,rs]))
            dataset2.append(np.array(data_from_dict.vozocrtx_bot[:,ls,rs]))
            dataset2.append(np.array(data_from_dict.vomecrty_top[:,ls,rs]))
            dataset2.append(np.array(data_from_dict.vomecrty_mid[:,ls,rs]))
            dataset2.append(np.array(data_from_dict.vomecrty_bot[:,ls,rs]))
            
            raw_data.append(dataset2)
    return raw_data

def write_header_bc(file_path, component, name, layer = 1, spacing = 'even'):
    spacing = 'ukc4'
    """
    
    Write a text block to a file with the given file path,
    and a customizable 'Name' value.

    Args:
        file_path (str): The file path to write the text block to.
        name (str): The value for the 'Name' line in the text block.
    """
    #Handle exception so that it doesnt write bad files. 
    try:
        items = BoundaryConditionTypes[component]
        scriptname = items.get('script_name', 'Unknown')
        units = items.get('units', 'Unknown')
        function = items.get('function', 'Unknown')
        
            
              # Print all options
    except KeyError:
        print(f"Item '{component}' is not found in options dictionary.")
        # self.view_options()



    ### MAY need to seperate this out depending on how the files are written, will mostly be editing water level and velocity. 
        
    with open(file_path, "w") as f:
        f.write("[forcing]\n")
        f.write(f"Name                            = {name}\n")
        f.write(f"Function                        = {function}\n")
        f.write("Time-interpolation              = linear\n")
        if user_dict['layer'] != 1:
            f.write("Vertical position type          = percentage from bed\n")
            f.write("Vertical position specification = ") 
            if spacing != 'even':
                #f.write(self.spacing + "\n")
                f.write("0 50 100\n")
            else:
                result = [(round(100 / user_dict['layer'] * i)) for i in range(1, user_dict['layer'] + 1)]
                result_str = ' '.join(map(str, result))
                f.write(f"{result_str}\n")
                # Gives you 33 67 100
            f.write("Vertical interpolation          = linear\n")
        f.write("Quantity                        = time\n")
        f.write("Unit                            = seconds since " + user_dict['formatted_str'] + "\n")
        if user_dict['layer'] == 'n':
            num_range = 1
        else:
            num_range = user_dict['layer']
        
        for i in range(num_range):
            f.write(f"Quantity                        = {scriptname}\n")
            f.write(f"Unit                            = {units}\n")
            if num_range > 1:
                num = str(i+1)
                f.write(f"Vertical position               = {num}\n")
        f.close()
        

    print(f"Text block with Name = '{name}' written to {file_path} successfully!")
    

def main_body_data_writer(raw_data, iteration, df):
    # self.old_j = []
    # self.new_j = []
    for boundary_data in range(len(raw_data)):
        
        comp_name = user_dict['var_list'][boundary_data]
        print(comp_name)
        #print('comp_name')
        for j, n in enumerate(user_dict['name']):
            # print(j)
            if flip == 'y':
                newj = (j+1)*-1 # or regular j This code flips the plotting upside down
            else:
                newj = j
            # print((j+1)*-1)
            #print('csv ', self.csv_path)
            filename = join(user_dict['csv_path'], n + f'_{os.path.split(comp_name)[-1]}_' +'.csv')
            #print('filename', filename)
            
            #print('\n Layer value here :' ,self.layer)
            
            if iteration == 0:
                with open(filename, 'w') as f:
                    f.write('') # reset files for fresh data when you rerun
                #layer = 1
                write_header_bc(filename, os.path.split(comp_name)[-1], n, layer = user_dict['layer'])
            if user_dict['layer'] == 1:
                # This bit adds in the columns for the dataframes which means you get 3 layers deep of data. 
                #print('raw_data',raw_data)
                df['data'] = raw_data[boundary_data][0][:,newj]
                # self.old_j.append(raw_data[boundary_data][0][:,j])
                # self.new_j.append(raw_data[boundary_data][0][:,newj])
                
                
            elif user_dict['layer'] == 3: 
                
                df['bottom'] = raw_data[boundary_data][2][:,newj]
                df['middle'] = raw_data[boundary_data][1][:,newj]
                df['top'] = raw_data[boundary_data][0][:,newj]
                
                # So you get Bottom Middle Top or just Top 
            
            #print('dataframe',df)
            df.to_csv(filename, header = False, index = False, sep = ' ', mode = 'a')
        
def para_file_rip(all_files, store):
    '''
    Only works with smaller datasets otherwise dask throws a fit. 
    So this is the one that the program runs on,
    data becomes a collection of all time ukc4 datasets, 
    store in this case is a letter, where all files are a list of files to read. 
    '''
    # Key does not exist, perform the operation
    keyname = 'data_' + store
    if keyname not in user_dict:
        user_dict[keyname] = xr.open_mfdataset(all_files, parallel=True, engine='netcdf4')#, chunks =  {'time_counter':100})
        user_dict['formatted_str'] = user_dict[keyname].time_counter[0].dt.strftime('%Y-%m-%d %H:%M:%S').item()
    
    time = convert_to_seconds_since_date(user_dict[keyname].time_counter,user_dict['formatted_str']) # why is this start time so rigid. 
    df = pd.DataFrame()
    df['time'] = [re.sub(r'[^0-9-]', '', str(i)) for i in time]
    raw_data = data_extractor(user_dict[keyname]) # This is the process that pulls out the data from UKC4. 
    main_body_data_writer(raw_data,0,df)
    

def non_para_file_rip():
    'Havent recoded this bit in yet. Probably not needed unless working with larger filesizes. '
    pass
    
def file_stitcher():
    for names in user_dict['var_list']:
        comps = os.path.split(names)[-1]
        # try to do in pairs of 3.
        # print('Names  ', comps[:-3])
        data_paths = sorted(glob.glob(os.path.join(user_dict['csv_path'],f'*_{comps}_.csv')))
        # print(os.path.join(user_dict['csv_path'],f'*_{comps[:-3]}_*.csv'))
        # print('dp',data_paths)
        
        bash_script_path = pkg_resources.resource_filename('o_func', 'data/bash/merge_csv.sh')
        output_file_path = os.path.join(user_dict['layer_path'],BoundaryConditionTypes[comps]['filename'])
        with open( output_file_path , 'w') as f:
            f.write('')
        if platform.system() == "Windows":
            subprocess.call([r"C:/Program Files/Git/bin/bash.exe", bash_script_path, output_file_path] + data_paths)
        else: # for mac or linux
            subprocess.call([r"bash", bash_script_path, output_file_path] + data_paths)
        

def write_bc_forcing_file(component, bc_paths, layer = 1):
    user_dict['layer'] = layer
    error = 'You need to create the Velocity.bc file which is the 3d tangent/normal velocities'
    if layer > 1 and 'WaterLevel.bc' in component:
        raise LayerError("'WaterLevel.bc' is not allowed when layers are greater than 3.\nAs Surface height is 2D not 3D. ")
    if layer > 1 and 'TangentVelocity.bc' in component:
        raise LayerError("'TangentVelocity.bc' is not allowed when layers are greater than 3.\nAs this is a 2D component.\n" + error)
    if layer > 1 and 'NormalVelocity.bc' in component:
        raise LayerError("'NormalVelocity.bc' is not allowed when layers are greater than 3.\nAs this is a 2D component.\n" + error)
    if layer <= 1 and 'Velocity.bc' in component:
        raise LayerError("'Velocity.bc' is not allowed when layers are less than or equal to 1.\nAs velocity is a 3D component.\nYou need Tangent/Normal Velocity files.")
    if isinstance(layer, int):
        if flip == 'y':
            user_dict['layer_path'] = util.md([bc_paths, str(layer)+ '_flipped'])
        else:
            user_dict['layer_path'] = util.md([bc_paths, str(layer)])
        '''
        Here a directory is made for the new dataset if it doesnt exist already, 
        '''
        
        # print(layer_path) # Prints out the path to which it will dump the CSV files. 
    csv_path = util.md([user_dict['layer_path'],'dump_CSV'])
    user_dict['csv_path'] = csv_path
    user_dict['component'] = component
    prepare_model_data()


#%% Run the program
if __name__ == '__main__':
    main_path = join(start_path, r'modelling_DATA','kent_estuary_project',r'7.met_office')
    make_paths = DirGen(main_path)
    fn = glob.glob(join(main_path,'models','*'))[0]
    sub_path = make_paths.dir_outputs(os.path.split(fn)[1]) # Dealing with this model run. 
    bc_paths = make_paths.bc_outputs()
    
    model_data = 'oa'
    full_path = join(start_path, 'Original_Data' ,'UKC3','sliced',model_data,'shelftmb_cut_to_domain')
    model_data_dict = {'oa' :bc_paths[1][1],
                       'owa':bc_paths[1][2],
                       'ow' :bc_paths[1][3],
                       'og' :bc_paths[1][0],
                            }
    forcing_data_path = join(start_path, 'Original_Data' ,'UKC3','sliced',model_data,'shelftmb_cut_to_domain')

    write_pli_file_names(forcing_data_path, fn)
    
    
    
    write_bc_forcing_file(layer = 1, component = ['WaterLevel','Salinity','Temperature', 'NormalVelocity','TangentVelocity'],
                          bc_paths = model_data_dict[model_data])
    
    
    # Build up the boundary condition files.    
