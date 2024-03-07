#!/usr/bin/env python3
# -*- coding: utf-8 -*-
### WRITES BOUNDARY CONDITIONS AND PLI FILES. 


""" Generate .pli files for use in Delft 3d FM suite to automatically generate 
the boundary condition shapefile out of LAT/LON.

Generates figures:
    - Fig1, 001_illustration_of_ukc3_grid_against_PRIMEA_model_area.png

Outputs:
    - This file will generate a .pli file for use within the Delft 3dfm suite 
    that will alllow the program to generate a tidal/ocean boundary based off of 
    the UKC3 met office grid points.
    - A .png file is also generated to show the area covered by the points in the 
    PRIMEA model domain.

Created on Thu Jan 12 13:22:52 2023
@author: af
"""

''' Notes to self
- Pull in UKC3 grid as an array
- Pull in desired outer coordinates - pli file 
- Nearest neighbour algorithum to stick pli to UKC3 dataset
- Generate new pli file where necessary. 


SANITY CHECK 
Performed on the 7th Feb 2024, you did flip them upside down, checked the file you flipped. 


'''
import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
# from unittest.mock import patch
import pandas as pd
from sklearn.neighbors import BallTree
import re
import subprocess
import pkg_resources
import platform


# import dask

# Set the option to split large chunks


#homemade packages. 
from o_func import opsys; start_path = opsys()
from o_func.utilities.choices import DataChoice
import o_func.utilities as util
from o_func.data_prepkit import primea_bounds_for_ukc4_slicing  # import boundaries of primea model in ukc4 lon and lats

class LayerError(Exception):
    pass

class InMake:
    def __init__(self, model_dir_to_put_files, bc_paths, original_data_path, p = 'n'):
        '''
        model_dir_to_put_files: Should be a function of the dir_gen part of package. 
        '''
        self.p = p
        self.full_path = os.path.join(original_data_path, '*')
        #model_paths
        self.model_path = model_dir_to_put_files
        self.input_path = os.path.join(self.model_path[0], 'inputs')
        
        #pli_paths
        self.loc_pli = os.path.join(start_path,
                                    'modelling_DATA',
                                    'kent_estuary_project',
                                    'tidal_boundary',
                                    'delft_3dfm_inputs'
                                    )
        self.pli_dir = os.path.join(self.loc_pli , 'pli_files')
        #pli_constants
        # self.spec_col = 758# This should really be remade to find these points using lat and lon sections
        # self.upper = 688 #690 fits within the new delft grid with a square edge
        # self.lower = 601 #600
        
        self.loc_bounds = primea_bounds_for_ukc4_slicing()
        
        self.bc_paths = bc_paths
        #original_data_paths
        '''
        KEY TO NOTE
         I have overridden the manual choice so it always runs the og folder for now. 
        '''
        # directory_path3 = os.path.join('Original_Data','UKC3')
        # dc = DataChoice(os.path.join(start_path, directory_path3))
        # Automate input within the function
        # with patch('builtins.input', return_value='4'): # This autoselected the og data.n
        #     self.var_path = dc.var_select()
        
        self.var_path = original_data_path
        #dictionary of boundary file types. 
        self.options = {
            'NormalVelocity.bc': {
                'script_name': 'normalvelocitybnd',
                'units': 'm/s',
                'function':'timeseries',
                'filetype': 'U'
            },
            'TangentVelocity.bc': {
                'script_name': 'tangentialvelocitybnd',
                'units': 'm/s',
                'function':'timeseries',
                'filetype': 'V'
            },
            'WaterLevel.bc': {
                'script_name': 'waterlevelbnd',
                'units': 'm',
                'function':'timeseries',
                'filetype': 'T'
            },
            'Discharge.bc': {
                'script_name': 'dischargebnd',
                'units': 'm³/s',
                'function':'timeseries',
                'filetype': 'R'
            },
            'Salinity.bc': {
                'script_name': 'salinitybnd',
                'units': 'ppt',
                'function':'timeseries',
                'filetype': 'T'
            }
            #guess as to what temperature filetypes would be. 
            ,
            'Temperature.bc': {                     
                'script_name': 'temperaturebnd',
                'units': 'degrees',
                'filetype': 'T'
            }
            ,
            'Velocity.bc': {                     
                'script_name': 'uxuyadvectionvelocitybnd:ux,uy',
                'units': 'm/s',
                'function':'t3d',
                'filetype': 'UV'
            }
        }
    
    @staticmethod
    def find_index(lonval, latval, lonarray, latarray):

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
    @staticmethod
    def convert_to_seconds_since_date(timeseries, date_str):
        """
        Convert a timeseries of points to seconds since a specific date.

        Args:
            timeseries (list): The timeseries of points.np.array
            date_str (str): The date to use as the reference, in the format "YYYY-MM-DD HH:MM:SS".

        Returns:
            list: The timeseries of points converted to seconds since the reference date.
        """
        # Convert the date_str to a datetime object
        
        
        converted_dates = (np.array(timeseries) - np.datetime64(date_str)).astype('timedelta64[s]')
        
        return converted_dates
    
    def view_options(self):
        # Access and print the options dictionary
        print("Options Dictionary:")
        for item_name, item_data in self.options.items():
            script_name = item_data.get('script_name', 'Unknown')
            units = item_data.get('units', 'Unknown')
            print(f"Item: {item_name}, Script Name: {script_name}, Units: {units}")
    
    def UKC3_testcase(self):
        # full_path = os.path.join(start_path, 'Original_Data' ,'UKC3','owa','shelftmb','*')
        first = sorted(glob.glob(self.full_path))[0]
        
        T = xr.open_dataset(first)

        z = T.sossheig.values[0,:,:]
        x = T.nav_lon.values
        y = T.nav_lat.values

        return x, y, z
    
    def write_pli(self):
        
        self.x , self.y, self.z = self.UKC3_testcase()
        
        lower = InMake.find_index(self.loc_bounds['South']['lon'],self.loc_bounds['South']['lat'],self.x, self.y)
        upper = InMake.find_index(self.loc_bounds['North']['lon'],self.loc_bounds['North']['lat'],self.x, self.y)
        
        self.lower = lower[0]
        self.upper = upper[0]
        
        if lower[1] == upper[1]:
            self.spec_col = lower[1]
    
        self.long = self.x[self.lower:self.upper,self.spec_col] # lazy boundary line generator
        self.lati = self.y[self.lower:self.upper,self.spec_col] # lazy boundary line generator
        
        # self.dirname = 'PLI_FILE_delft_ocean_boundary_UKC4_b' + str(self.lower) + 't' + str(self.upper) + '_length-' + str(len(self.long)) + '_points.pli'
        # self.dirname = '001_delft_ocean_boundary_UKC4_b' + str(self.lower) + 't' + str(self.upper) + '_length-' + str(len(self.long)) 
        self.dirname = '001_delft_ocean_boundary_UKC3_b601t688_length-' + str(len(self.long)) + '_points' # need to fix this properly
        path = os.path.join(self.input_path, self.dirname)
        #path = os.path.join(self.pli_dir, dirname)
        f = open(path , "w")
        f.write(self.dirname)
        f.write('\n    ' + str(len(self.lati)) + '    ' + str(2))
        f.close()
        #adjustment = (-3.606832728971--3.5984408138453)*0.6
        f = open(path, "a")
        
        
        self.name = []
        for i in range(len(self.long)):
            num = i+1
            new_form = format(num, '04')
            new_long = float(format(self.long[i], '.15f'))
            new_lati = float(format(self.lati[i], '.15f'))
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
            
            self.name.append(self.dirname + '_' + new_form)
            f.write('\n' + new_long + '  ' + new_lati + ' ' + self.dirname + '_' + new_form)
        f.close()

    def plot_pli(self):
        '''
        Eventually this should be integrated with video plots . 
        '''
        #% Constants 
        fontsize = 15
        data = self.PRIMEA_coastline()
        
        #fig 1
        # - domain of the ukc3 grid cells plotted across the domain of the PRIMEA model
        fig1, ax1 = plt.subplots(figsize=(7.5,15))
        loc_ukc3 = start_path + r'Original Data/UKC3/og/shelftmb/UKO4g_1h_20131030_20131030_shelftmb_grid_T.nc'
        ukc3_df = xr.open_dataset(loc_ukc3)
        nav_lat = ukc3_df.nav_lat.values
        nav_lon = ukc3_df.nav_lon.values
        # testing of seeing what it looks like plotted
        plt.contourf(self.x,self.y,self.z)
        plt.scatter(nav_lon,nav_lat)
        data.plot(ax=ax1, color="red")
        plt.xlim([-3.65,-2.75])
        plt.ylim([53.20,54.52])
        plt.xlabel('Longitude', fontsize = fontsize-30)
        plt.ylabel('Latitude', fontsize = fontsize-30)
        plt.title('An illustration of the UKC3\ngrid across the domain\nof the PRIMEA model', fontsize = fontsize)
        plt.savefig(self.loc_pli + '/figures/001_illustration_of_ukc3_grid_against_PRIMEA_model_area.png', dpi = 400, )

        #% fig 2
        # - plotting the 
        fig2, ax2 = plt.subplots(figsize=(30,15))
        data.plot(ax=ax2, color="red")
        plt.contourf(self.x,self.y,self.z)
        ax2.set_xlim(-3.65,-2.75)
        ax2.set_ylim(53.20,54.52)
        
        y_lims = ([self.lower,self.upper])
        plt.scatter(self.x[y_lims,self.spec_col],self.y[y_lims,self.spec_col],c = 'orange')
        plt.xlabel('Longitude', fontsize = fontsize-30)
        plt.ylabel('Latitude', fontsize = fontsize-30)
        plt.title('Top and Bottom\nboundary limits in the\nPRIMEA model area', fontsize = fontsize)
        plt.savefig(self.loc_pli + '/figures/002_Top_Bottom_bound_limits_PRIMEA_model_area.png', dpi = 400, )

    #def write_bc(self):
        #writes direct boundary files that can be rea in by delft. 
        
    def write_header_bc(self,file_path, component, name, layer = 1, spacing = 'even'):
        self.spacing = spacing
        self.spacing = 'ukc4'
        """
        
        Write a text block to a file with the given file path,
        and a customizable 'Name' value.

        Args:
            file_path (str): The file path to write the text block to.
            name (str): The value for the 'Name' line in the text block.
        """
        #Handle exception so that it doesnt write bad files. 
        try:
            items = self.options[component]
            scriptname = items.get('script_name', 'Unknown')
            units = items.get('units', 'Unknown')
            function = items.get('function', 'Unknown')
            
                
                  # Print all options
        except KeyError:
            print(f"Item '{component}' is not found in options dictionary.")
            self.view_options()
    
    
    
        ### MAY need to seperate this out depending on how the files are written, will mostly be editing water level and velocity. 
            
        with open(file_path, "w") as f:
            f.write("[forcing]\n")
            f.write(f"Name                            = {name}\n")
            f.write(f"Function                        = {function}\n")
            f.write("Time-interpolation              = linear\n")
            if self.layer != 1:
                f.write("Vertical position type          = percentage from bed\n")
                f.write("Vertical position specification = ") 
                if self.spacing != 'even':
                    #f.write(self.spacing + "\n")
                    f.write("0 50 100\n")
                else:
                    result = [(round(100 / self.layer * i)) for i in range(1, self.layer + 1)]
                    result_str = ' '.join(map(str, result))
                    f.write(f"{result_str}\n")
                    # Gives you 33 67 100
                f.write("Vertical interpolation          = linear\n")
            f.write("Quantity                        = time\n")
            f.write("Unit                            = seconds since 2013-10-31 00:00:00\n")
            if self.layer == 'n':
                num_range = 1
            else:
                num_range = self.layer
            
            for i in range(num_range):
                f.write(f"Quantity                        = {scriptname}\n")
                f.write(f"Unit                            = {units}\n")
                if num_range > 1:
                    num = str(i+1)
                    f.write(f"Vertical position               = {num}\n")
            f.close()
            

        print(f"Text block with Name = '{name}' written to {file_path} successfully!")
        
    # def write_bc(self, layer = 'n', spacing = 'even'):
    #     #self.layer = layer
    #     #self.spacing = spacing
    #     with open(os.path.join(self.input_path, 'test_bc.txt'), 'w') as f:
    #         f.write('')
    #         f.close()
    #     print(os.path.join(self.input_path, 'test_bc.txt'))
    #     self.write_header_bc(os.path.join(self.input_path, 'test_bc.txt'), 'velocity_normal')
    def file_ripper(self, store):
        choose = store[0]
        print('Choose', choose)
        if choose == 'T':
            datas = store[1:]
            print(datas)
            self.var_list = []
            for i in datas:
                self.var_list.append(os.path.join(self.layer_path, i))
        elif choose == 'U':
            datas = store[1:]
            self.var_list = []
            for i in datas:
                self.var_list.append(os.path.join(self.layer_path, i))
        elif choose == 'V':
            datas = store[1:]
            self.var_list = []
            for i in datas:
                self.var_list.append(os.path.join(self.layer_path, i))
        elif choose == 'UV':
            datas = store[1:]
            self.var_list = []
            for i in datas:
                self.var_list.append(os.path.join(self.layer_path, i))
            print('var_list',self.var_list)
        
        #self.vel_write = os.path.join(self.layer_path, 'NormalVelocity.bc' )
        
        print('Paths to send to file cruncher', self.var_list)
        
    def data_extract(self, data):
        raw_data = []
        for names in self.var_list:
            print(names)
            # try to do in pairs of 3.
            print('Names  ', os.path.split(names)[-1])
            if os.path.split(names)[-1] == 'WaterLevel.bc':
                dataset2 = []
                dataset2.append(np.array(data.sossheig[:,self.ls,self.new_rs]))
                dataset2.append('')
                dataset2.append('')
                raw_data.append(dataset2)
                
            if os.path.split(names)[-1] == 'Salinity.bc':
                dataset2 = []
                dataset2.append(np.array(data.vosaline_top[:,self.ls,self.new_rs]))
                if self.layer > 1:
                    dataset2.append(np.array(data.vosaline_mid[:,self.ls,self.new_rs]))
                    dataset2.append(np.array(data.vosaline_bot[:,self.ls,self.new_rs]))
                    
                else:
                    dataset2.append('')
                    dataset2.append('')
                raw_data.append(dataset2)
            if os.path.split(names)[-1] == 'Temperature.bc':
                dataset2 = []
                dataset2.append(np.array(data.votemper_top[:,self.ls,self.new_rs]))
                if self.layer > 1:
                    dataset2.append(np.array(data.votemper_mid[:,self.ls,self.new_rs]))
                    dataset2.append(np.array(data.votemper_bot[:,self.ls,self.new_rs]))
                else:
                    
                    dataset2.append('')
                    dataset2.append('')
                raw_data.append(dataset2)
            if os.path.split(names)[-1] == 'NormalVelocity.bc':
                dataset2 = []
                dataset2.append(np.array(data.vozocrtx_top[:,self.ls,self.new_rs]))
                if self.layer > 1:
                    dataset2.append(np.array(data.vozocrtx_mid[:,self.ls,self.new_rs]))
                    dataset2.append(np.array(data.vozocrtx_bot[:,self.ls,self.new_rs]))
                else:
                    dataset2.append('')
                    dataset2.append('')
                raw_data.append(dataset2)
            if os.path.split(names)[-1] == 'TangentVelocity.bc':
                dataset2 = []
                dataset2.append(np.array(data.vomecrty_top[:,self.ls,self.new_rs]))
                if self.layer > 1:
                    dataset2.append(np.array(data.vomecrty_mid[:,self.ls,self.new_rs]))
                    dataset2.append(np.array(data.vomecrty_bot[:,self.ls,self.new_rs]))
                else:
                    dataset2.append('')
                    dataset2.append('')
                raw_data.append(dataset2)
            if os.path.split(names)[-1] == 'Velocity.bc':
                dataset2 = []
                dataset2.append(np.array(data.vozocrtx_top[:,self.ls,self.new_rs]))
                dataset2.append(np.array(data.vozocrtx_mid[:,self.ls,self.new_rs]))
                dataset2.append(np.array(data.vozocrtx_bot[:,self.ls,self.new_rs]))
                dataset2.append(np.array(data.vomecrty_top[:,self.ls,self.new_rs]))
                dataset2.append(np.array(data.vomecrty_mid[:,self.ls,self.new_rs]))
                dataset2.append(np.array(data.vomecrty_bot[:,self.ls,self.new_rs]))
                
                raw_data.append(dataset2)
        return raw_data
    
    def main_body_data_writer(self, raw_data, i, df):
        # self.old_j = []
        # self.new_j = []
        for boundary_data in range(len(raw_data)):
            
            comp_name = self.var_list[boundary_data]
            print(comp_name)
            #print('comp_name')
            for j, n in enumerate(self.name):
                # print(j)
                newj = (j+1)*-1 # or regular j This code flips the plotting upside down
                # print((j+1)*-1)
                #print('csv ', self.csv_path)
                filename = os.path.join(self.csv_path, n + f'_{os.path.split(comp_name[:-3])[-1]}_' +'.csv')
                #print('filename', filename)
                
                #print('\n Layer value here :' ,self.layer)
                
                if i == 0:
                    with open(filename, 'w') as f:
                        f.write('') # reset files for fresh data when you rerun
                    #layer = 1
                    self.write_header_bc(filename, os.path.split(comp_name)[-1], n, layer = self.layer)
                if self.layer == 1:
                    # This bit adds in the columns for the dataframes which means you get 3 layers deep of data. 
                    #print('raw_data',raw_data)
                    df['data'] = raw_data[boundary_data][0][:,newj]
                    # self.old_j.append(raw_data[boundary_data][0][:,j])
                    # self.new_j.append(raw_data[boundary_data][0][:,newj])
                    
                    
                elif self.layer == 3: 
                    
                    df['bottom'] = raw_data[boundary_data][2][:,newj]
                    df['middle'] = raw_data[boundary_data][1][:,newj]
                    df['top'] = raw_data[boundary_data][0][:,newj]
                    
                    # So you get Bottom Middle Top or just Top 
                
                #print('dataframe',df)
                df.to_csv(filename, header = False, index = False, sep = ' ', mode = 'a')
            
    
    def non_para_file_rip(self,dataset):
        print('length of dataset     \n',len(dataset))
        # if len(dataset) == 2:
        #     # dataset2 = dataset[1]
        #     dataset = dataset[0]
        # # data = xr.open_mfdataset(dataset, parallel=True, chunks =  {'time_counter':10, 'nav_lon':100, 'nav_lat':100})
        for i,file in enumerate(dataset):
            print('ripin non para', i)
            if len(dataset) == 2:
                data = xr.open_dataset(file, engine ='netcdf4')
                print('DATASET works here ...\n') # why on earth did I put this in as a conditional statement
            else:
                data = xr.open_dataset(file, engine ='netcdf4')   
            time = self.convert_to_seconds_since_date(data.time_counter,r'2013-10-31 00:00:00')
            df = pd.DataFrame()
            df['time'] = [re.sub(r'[^0-9-]', '', str(i)) for i in time]

            
            raw_data = self.data_extract(data)
            self.main_body_data_writer(raw_data, i, df)
            
            
    def para_file_rip(self,dataset):
        '''
        Only works with smaller datasets otherwise dask throws a fit. 
        '''
        data = xr.open_mfdataset(dataset, parallel=True)#, chunks =  {'time_counter':100})
        # for i,file in enumerate(dataset):
        #     print('ripin ', i)
        #     if len(dataset) == 2:
        #         data = xr.open_dataset(file, engine ='netcdf4')
        #         print('DATASET works here ...\n') # why on earth did I put this in as a conditional statement
        #     else:
        #         data = xr.open_dataset(file, engine ='netcdf4')   
        time = self.convert_to_seconds_since_date(data.time_counter,r'2013-10-30 00:00:00')
        df = pd.DataFrame()
        df['time'] = [re.sub(r'[^0-9-]', '', str(i)) for i in time]
        raw_data = self.data_extract(data)
        self.main_body_data_writer(raw_data,0,df)
        # self.new_j.reverse()
        # result = [elem1 == elem2 for elem1, elem2 in zip(self.old_j, self.new_j)]
        # all_true = np.all(result) # This is the variable that if True declares that the process works. 

    def file_stitcher(self):
        for names in self.var_list:
            comps = os.path.split(names)[-1]
            # try to do in pairs of 3.
            print('Names  ', comps[:-3])
            data_paths = sorted(glob.glob(os.path.join(self.csv_path,f'*_{comps[:-3]}_.csv')))
            print(os.path.join(self.csv_path,f'*_{comps[:-3]}_*.csv'))
            print('dp',data_paths)
            
            bash_script_path = pkg_resources.resource_filename('o_func', 'data/bash/merge_csv.sh')
            output_file_path = os.path.join(self.layer_path,comps)
            with open( output_file_path , 'w') as f:
                f.write('')
            if platform.system() == "Windows":
                subprocess.call([r"C:/Program Files/Git/bin/bash.exe", bash_script_path, output_file_path] + data_paths)
            else: # for mac or linux
                subprocess.call([r"bash", bash_script_path, output_file_path] + data_paths)
            
    def ocean_timeseries(self):
        
        ### FUNCTION here to check if files already exist if so will do nothing. 
        
        T_path = os.path.join(self.var_path,'*T.nc')
        U_path = os.path.join(self.var_path,'*U.nc')
        V_path = os.path.join(self.var_path,'*V.nc')
        
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
        df_locs['x'] = self.long; df_locs['y'] = self.lati
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
            nearest_neigh = list(divmod(indices[i,0],self.x.shape[1])) #1458 is length of row, unravel ravel
            nn.append(nearest_neigh)
        
        self.ls = [item[0] for item in nn]
        rs = [item[1] for item in nn]
        self.new_rs = int(np.mean(rs))
           
        
        ### This function does work but is a little long, try paralleism. 

        T_store = ['T']
        U_store = ['U']
        V_store = ['V']
        R_store = ['R']
        UV_store = ['UV']
        
        for item in self.component:
            print(item)
            if item in self.options:
                filetype = self.options[item]['filetype']
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
        print('T_store', len(T_store))
        print('U_store', len(U_store))
        print('V_store', V_store)
        print('R_store', R_store)
        print('UV_store', len(UV_store))
        ## new loop to run task
        if len(T_store) > 1:
            dataset = all_files[0]
            import time
            starttime = time.time()
            #Parallel(n_jobs=-1)(delayed(self.vid_plotter)(num_iters) for num_iters in range(num_of_figs))
            self.file_ripper(T_store)
            #Parallel(n_jobs=-1)(delayed(self.para_file_rip)(file, i) for i, file in enumerate(dataset))
            if self.p == 'y':
                self.para_file_rip(dataset)
            else:
                self.non_para_file_rip(dataset)
            endtime = time.time()-starttime
            self.file_stitcher()
            print('Finished in ',endtime,' minutes')
           # file_ripper(dataset, T_store)

        if len(U_store) > 1:
            dataset = all_files[1]
            self.file_ripper(U_store)
            if self.p == 'y':
                self.para_file_rip(dataset)
            else:
                self.non_para_file_rip(dataset)
            self.file_stitcher()
            
        if len(V_store) > 1:
            dataset = all_files[2]
            self.file_ripper(V_store)
            if self.p == 'y':
                self.para_file_rip(dataset)
            else:
                self.non_para_file_rip(dataset)
            self.file_stitcher()
        if len(R_store) > 1:
            self.file_ripper()
        if len(UV_store) > 1:
            dataset = all_files[1:]
            self.file_ripper(UV_store)
            if self.p == 'y':
                self.para_file_rip(dataset)
            else:
                self.non_para_file_rip(dataset)
            self.file_stitcher()
            
            
        print ('Finished')
        
        
        
        #data = file_ripper()
        
        # for i in self.component:
        #     file_stitcher(i)    
            
         
            
    def write_boundary_file(self, layer, component):
        self.layer = layer
        self.component = component
        error = 'You need to create the Velocity.bc file which is the 3d tangent/normal velocities'
        if self.layer > 1 and 'WaterLevel.bc' in component:
            # Raise a custom exception to handle the error
            raise LayerError("'WaterLevel.bc' is not allowed when layers are greater than 3.\nAs Surface height is 2D not 3D. ")
        if self.layer > 1 and 'TangentVelocity.bc' in component:
            # Raise a custom exception to handle the error
            
            raise LayerError("'TangentVelocity.bc' is not allowed when layers are greater than 3.\nAs this is a 2D component.\n" + error)
        if self.layer > 1 and 'NormalVelocity.bc' in component:
            # Raise a custom exception to handle the error
            raise LayerError("'NormalVelocity.bc' is not allowed when layers are greater than 3.\nAs this is a 2D component.\n" + error)
        if self.layer <= 1 and 'Velocity.bc' in component:
            # Raise a custom exception to handle the error
            raise LayerError("'Velocity.bc' is not allowed when layers are less than or equal to 1.\nAs velocity is a 3D component.\nYou need Tangent/Normal Velocity files.")

            
    
        if isinstance(self.layer, int):
            self.layer_path = util.md([self.bc_paths, str(self.layer)])
            print(self.layer_path)
        self.csv_path = util.md([self.layer_path,'dump_CSV'])
        
        make_files.ocean_timeseries()
        #component = 'velocity_normal'

if __name__ == '__main__':
    
    from o_func import DirGen , opsys; start_path = opsys()

    # Set example of directory to run the file. 

    #%% Making Directory paths
    main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'7.met_office')
    make_paths = DirGen(main_path)
    sub_path = make_paths.dir_outputs('PRIMEA2D_ogUMk') # Dealing with this model run. 
    bc_paths = make_paths.bc_outputs()
    ### Finishing directory paths

    dc = DataChoice(os.path.join(main_path,'models'))
    fn = dc.dir_select()
    
    # To work this needs to have all of the datasets in it. 
    # full_path = os.path.join(start_path, 'Original_Data' ,'UKC3','owa','shelftmb')
    # og oa oaw ow
    # for k, model_data in enumerate(['og', 'oa', 'owa', 'ow']):
    model_data = 'ow'
    full_path = os.path.join(start_path, 'Original_Data' ,'UKC3','sliced',model_data,'shelftmb_cut_to_domain')
    make_files = InMake(fn, bc_paths[1][3], full_path, p = 'y') #  Pass model path folder into make file folder. 
    
    make_files.write_pli()
    
    #make_files.write_bc(layer = 3)
    ### ALL POSSIBLE OPTIONS
    
    '''
    boundary_files_to_write = ['velocity_normal', 'velocity_tangent', 'water_level', 'discharge', 'salinity']
    
    ,NormalVelocity 'TangentVelocity', 'WaterLevel', 'Salinity']
    '''
    # This line definately works as of 2023-09-18 10:56
    make_files.write_boundary_file(layer = 1, component = ['WaterLevel.bc', 'Salinity.bc','Temperature.bc', 'NormalVelocity.bc','TangentVelocity.bc'])
    # make_files.write_boundary_file(layer = 3, component = ['Velocity.bc'])
    
    #First step is to make the .pli files to which the boundary conditions are made. 
    
        
        
        
        
