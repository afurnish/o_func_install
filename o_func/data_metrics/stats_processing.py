#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will generate statitistics and also process outputs, statistics go live in the stats folder. 


Created on Thu Sep 28 09:15:20 2023
@author: af
"""
import pandas as pd
import os
import xarray as xr
import numpy as np
import sys

from o_func import opsys; start_path = opsys()


class r_stats:
    def __init__(self, dataset):
     self.tide_gauge_path = os.path.join(start_path,
                                         'modelling_DATA',
                                         'kent_estuary_project',
                                         'validation',
                                         'tidal_validation',
                                         r'1.reformatted')
     ukc3_path = os.path.join(start_path,
                  'modelling_DATA',
                  'kent_estuary_project',
                  r'6.Final2',
                  'processed_data',
                  'ukc4_on_PRIMEA_grid')     
                  
                  
     self.data = dataset # primea data path             
                  
     # ukc3_int_path_options = {'default':{'name':'original ukc3 interp dataset',
     #                                     'path': ukc3_path}
     #                             }
     
     # self.ukc3_int_path = ukc3_int_path_options[ukc3_int_path]['path']
     # print(self.ukc3_int_path)
     
    @staticmethod
    def calculate_hourly_means(df):
        """
        Calculate hourly means for a time series DataFrame with timestamps at the middle of each hour.

        Args:
            df (pd.DataFrame): Input DataFrame with DatetimeIndex.

        Returns:
            pd.DataFrame: DataFrame with hourly means and timestamps at the middle of each hour.
        """
        hourly_means = df.resample('H').mean()
        hourly_means.index = hourly_means.index + pd.Timedelta(minutes=30)

        return hourly_means

    def raw_to_useful(self, prim_data,ukc3_data,var,index_only):
        data_to_populate = []
        '''
        Primary purpose is to allign time dimensions for accurate comparison. Data should be sampled. 
        Depending on the variable you may want to interpolate values between but only for plotting not for 
        calculating statistics. 
        
        
        Need to make sure for this function to work you load in the time coords for 
        primea and for ukc3 as well as the variable you wish to use:
            Choices:
                Surface Height = sh
                Salinity       = sal
                
        Here it is really important to claify what time and shape dimension the data is in,
        if the data is sampled in 10 minute intervals this works, however is hourly intervas are
        used everything breaks apart. Ensure the code is robust enough to handle hourly 
        and 10 minute outputs or anything you may decide to throw at it.
        
        # try not to let hourly stuff get to here on the hour as it messes the whole thing up. 
        '''
        #path_to_data = path + '/'+ var + '_mean_half_hour_primea_data.pickle'
        
        
        UKC3_interp = UKC3_interp.z_interpolated_time.values
        
        primea_data = PRIMEA_all_data.surface_height.values
    
        prim_tim_df2 = prim_tim_df[:-1] # just removes last point in case its erronous
        primea_data2 = primea_data[:-1,:] # repeated for the data
        
        # special dataframe to index everything
        
        try:
            # Unpickle the object from the file
            if var == 'sh':
                print('Surface Height Running...')
                with open(path_to_data, 'rb') as file:
                    loaded_data = pickle.load(file)
            elif var == 'sal':
                print('Salinity Running...')
                with open(path_to_data, 'rb') as file:
                    loaded_data = pickle.load(file)
                
            with open(path + '/time_data_half_hour_sampled.pickle', 'rb') as file:
                prim_tim_df4 = pickle.load(file)
            warnings.filterwarnings("ignore")
        except Exception as e:
            print(e)
            print('\nNo file detected, writing file. ETA 10 minutes')

            new_df = []
            warnings.filterwarnings("ignore")
            new_array = []
            for i,data in enumerate(np.transpose(primea_data2)):
                del new_df
                new_df = prim_tim_df2
                new_df['data'] = data
                new_df.index = pd.to_datetime(new_df.primea_time)
                #df = new_df.drop('primea_time', axis=1)

                hourly_means = new_df.resample('1H').mean()
                hourly_means.index += pd.DateOffset(minutes=30)
                new_array.append(hourly_means)
                if i == 0:
                    prim_tim_df4 = hourly_means.index[24:]
                if i % 1000 == 0:
                    print('Run number' + str(i).zfill(3))
            loaded_data = np.transpose(np.array(new_array)[:,:,0])[24:]
        
            warnings.resetwarnings()
            if var == 'sh':
                print('Making Surface Height and Running...')
                with open(path_to_data, 'wb') as file:
                    pickle.dump(loaded_data, file)
            elif var == 'sal':
                print('Making Salinity and Running...')
                with open(path_to_data, 'wb') as file:
                    pickle.dump(loaded_data, file)
            with open(path + '/time_data_half_hour_sampled.pickle', 'wb') as file:
                pickle.dump(np.transpose(prim_tim_df4), file)
        warnings.resetwarnings()
        # # code that goes here sorts the data out into half hour intervals
        # prim_tim_df3 = prim_tim_df2[3:-4]
        # prim_tim_df4 = prim_tim_df3[::6] 

        #ukc3_tim_df = ukc3_time[24:]
        
        common_idx = ukc3_tim_df[ukc3_tim_df['ukc3_time'].isin(prim_tim_df4)].index
        if index_only != 'yes':
            #happy to identify that they are the same. 
            sliced_ukc3_tim_df = ukc3_tim_df.iloc[common_idx]
            sliced_ukc3_sh_df = UKC3_interp_z_interp[common_idx,:]
            sliced_ukc3_tim_df = sliced_ukc3_tim_df
            sliced_ukc3_sh_df2 = sliced_ukc3_sh_df
            
            
            # sh_primea = primea_data2[3:-4,:]
            sh_primea2 = loaded_data#sh_primea[::6,:]
            
            #calculate_hourly_means(sh_primea)
            
            sh_ukc3 = sliced_ukc3_sh_df2
        
            return sh_primea2, sh_ukc3, common_idx, UKC3_interp_z_interp
        else:
            return common_idx
        
    def load_data(self): 
        globbed = sorted(glob.glob(os.path.join( self.tide_gauge_path, '*.csv')))
        self.names = []
        self.df = []
        for i in globbed:
            self.names.append(os.path.split(i)[-1][:-4]  )  
            self.df.append(pd.read_csv(i))
        print(self.df, self.names)
        
        
        #In this case UKC4 was interpolated onto PRIMEA grid, load in that data
        
        #/media/af/PD/modelling_DATA/kent_estuary_project/6.Final2/processed_data/ukc4_on_PRIMEA_grid/sossheig_ogUKC3_.nc
        self.ukc3_dataset = xr.open_dataset(os.path.join(self.ukc3_int_path, 'UKC3_og_interpolated_PRIMEA_grid_data.nc'))
        print(self.ukc3_dataset.z_interpolated_time)
        
        self.primea_dataset = xr.open_dataset(self.prim_data, chunks={'time':100})
        
        
    def run_stats(self):
        self.load_data()
# if __name__ == "__main__":
#     from o_func import DataChoice, DirGen
#     import glob
    
#     #%% Making Directory paths
#     main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
#     make_paths = DirGen(main_path)
#     ### Finishing directory paths
    
#     dc = DataChoice(os.path.join(main_path,'models'))
#     fn = dc.dir_select()
#     sub_path = make_paths.dir_outputs(os.path.split(fn[0])[-1])
#     lp = glob.glob(os.path.join(sub_path, '*.nc'))[0]
#     rs = r_stats(prim_data = lp)
#     rs.load_data()
        
        
#%%   

example_dataset = os.path.join(start_path, 'modelling_DATA','kent_estuary_project',r'6.Final2','models','kent_1.3.7_testing_4_days_UM_run','kent_regrid.nc')
var_dict = {
'surface_height'   : {'TUV':'T',  'UKC4':'sossheig',       'PRIMEA':'mesh2d_s1'},
'surface_salinity' : {'TUV':'T',  'UKC4':'vosaline_top',   'PRIMEA':'mesh2d_sa1'},
'middle_salinity'  : {'TUV':'T',  'UKC4':'',   'PRIMEA':'na'},
'bottom_salinity'  : {'TUV':'T',  'UKC4':'',   'PRIMEA':'na'},
'surface_Uvelocity': {'TUV':'U',  'UKC4':'',   'PRIMEA':'na'},
'middle_Uvelocity' : {'TUV':'U',  'UKC4':'',   'PRIMEA':'na'},
'bottom_Uvelocity' : {'TUV':'U',  'UKC4':'',   'PRIMEA':'na'},
'surface_Vvelocity': {'TUV':'V',  'UKC4':'',   'PRIMEA':'na'},
'middle_Vvelocity' : {'TUV':'V',  'UKC4':'',   'PRIMEA':'na'},
'bottom_Vvelocity' : {'TUV':'V',  'UKC4':'',   'PRIMEA':'na'},
}
class stats:
    def __init__(self, dataset):
     self.tide_gauge_path = os.path.join(start_path,
                                         'modelling_DATA',
                                         'kent_estuary_project',
                                         'validation',
                                         'tidal_validation',
                                         r'1.reformatted')
     self.data = dataset # primea data path             
     
    @staticmethod
    def calculate_hourly_means(df):
        """
        Calculate hourly means for a time series DataFrame with timestamps at the middle of each hour.

        Args:
            df (pd.DataFrame): Input DataFrame with DatetimeIndex.

        Returns:
            pd.DataFrame: DataFrame with hourly means and timestamps at the middle of each hour.
        """
        hourly_means = df.resample('H').mean()
        hourly_means.index = hourly_means.index + pd.Timedelta(minutes=30)

        return hourly_means
    
    @staticmethod
    def prefix(desired_prefix, dictionary):    
        selected_dict = None
        for key, value in dictionary.items():
            if key.startswith(desired_prefix):
                selected_dict = value
                break  # Stop iterating once the first matching prefix is found
        return selected_dict
    #print(prefix('prim', loaded_data[4]['surface_height']))
     
    def load_raw(self): # Need to be able to run it for every dataset
        self.raw_data = xr.open_dataset(self.data)
        # self.time_shifted = self.time_shifter()
        filtered_list_of_dicts = {}
        for dataset_var in [i for i in var_dict]:
            split_datasets = {var: self.raw_data[var] for var in self.raw_data.data_vars if var.endswith(dataset_var)}
            if split_datasets:
                filtered_list_of_dicts[dataset_var] = split_datasets # This is now a dictionary of the data

        self.resampled_prim = self.raw_data.prim_surface_height.resample(time_primea='1H').mean() 
        self.resampled_prim['time_primea'] = self.resampled_prim.time_primea + pd.to_timedelta('30min')
        
        self.ukc4_sliced = self.raw_data.ukc4_surface_height.sel(time_counter=self.resampled_prim['time_primea'])
        
        matching_times = (self.resampled_prim['time_primea'] == self.ukc4_sliced['time_counter']).all()
        

        if matching_times:
            print("The time arrays have identical values.")
        else:
            print("The time arrays do not match.")
            print("Exiting script, take a look at the code...")
            
            sys.exit()
        split_datasets = {var: self.raw_data[var] for var in self.raw_data.data_vars if var.endswith('surface_height')}
        split_datasets = {var: self.raw_data[var] for var in self.raw_data.data_vars if var.endswith('surface_height')}
        return self.raw_data, self.resampled_prim, self.ukc4_sliced, split_datasets, filtered_list_of_dicts
        
if __name__ == '__main__':
    sts = stats(example_dataset)
    loaded_data = sts.load_raw()
    print(loaded_data[3].keys())       