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
import matplotlib.pyplot as plt

from o_func import opsys; start_path = opsys()
from o_func.utilities.near_neigh import near_neigh
#%% Old Example of Data
# class r_stats:
#     def __init__(self, dataset):
#      self.tide_gauge_path = os.path.join(start_path,
#                                          'modelling_DATA',
#                                          'kent_estuary_project',
#                                          'validation',
#                                          'tidal_validation',
#                                          r'1.reformatted')
#      ukc3_path = os.path.join(start_path,
#                   'modelling_DATA',
#                   'kent_estuary_project',
#                   r'6.Final2',
#                   'processed_data',
#                   'ukc4_on_PRIMEA_grid')     
                  
                  
#      self.data = dataset # primea data path             
                  
#      # ukc3_int_path_options = {'default':{'name':'original ukc3 interp dataset',
#      #                                     'path': ukc3_path}
#      #                             }
     
#      # self.ukc3_int_path = ukc3_int_path_options[ukc3_int_path]['path']
#      # print(self.ukc3_int_path)
     
#     @staticmethod
#     def calculate_hourly_means(df):
#         """
#         Calculate hourly means for a time series DataFrame with timestamps at the middle of each hour.

#         Args:
#             df (pd.DataFrame): Input DataFrame with DatetimeIndex.

#         Returns:
#             pd.DataFrame: DataFrame with hourly means and timestamps at the middle of each hour.
#         """
#         hourly_means = df.resample('H').mean()
#         hourly_means.index = hourly_means.index + pd.Timedelta(minutes=30)

#         return hourly_means

#     def raw_to_useful(self, prim_data,ukc3_data,var,index_only):
#         data_to_populate = []
#         '''
#         Primary purpose is to allign time dimensions for accurate comparison. Data should be sampled. 
#         Depending on the variable you may want to interpolate values between but only for plotting not for 
#         calculating statistics. 
        
        
#         Need to make sure for this function to work you load in the time coords for 
#         primea and for ukc3 as well as the variable you wish to use:
#             Choices:
#                 Surface Height = sh
#                 Salinity       = sal
                
#         Here it is really important to claify what time and shape dimension the data is in,
#         if the data is sampled in 10 minute intervals this works, however is hourly intervas are
#         used everything breaks apart. Ensure the code is robust enough to handle hourly 
#         and 10 minute outputs or anything you may decide to throw at it.
        
#         # try not to let hourly stuff get to here on the hour as it messes the whole thing up. 
#         '''
#         #path_to_data = path + '/'+ var + '_mean_half_hour_primea_data.pickle'
        
        
#         UKC3_interp = UKC3_interp.z_interpolated_time.values
        
#         primea_data = PRIMEA_all_data.surface_height.values
    
#         prim_tim_df2 = prim_tim_df[:-1] # just removes last point in case its erronous
#         primea_data2 = primea_data[:-1,:] # repeated for the data
        
#         # special dataframe to index everything
        
#         try:
#             # Unpickle the object from the file
#             if var == 'sh':
#                 print('Surface Height Running...')
#                 with open(path_to_data, 'rb') as file:
#                     loaded_data = pickle.load(file)
#             elif var == 'sal':
#                 print('Salinity Running...')
#                 with open(path_to_data, 'rb') as file:
#                     loaded_data = pickle.load(file)
                
#             with open(path + '/time_data_half_hour_sampled.pickle', 'rb') as file:
#                 prim_tim_df4 = pickle.load(file)
#             warnings.filterwarnings("ignore")
#         except Exception as e:
#             print(e)
#             print('\nNo file detected, writing file. ETA 10 minutes')

#             new_df = []
#             warnings.filterwarnings("ignore")
#             new_array = []
#             for i,data in enumerate(np.transpose(primea_data2)):
#                 del new_df
#                 new_df = prim_tim_df2
#                 new_df['data'] = data
#                 new_df.index = pd.to_datetime(new_df.primea_time)
#                 #df = new_df.drop('primea_time', axis=1)

#                 hourly_means = new_df.resample('1H').mean()
#                 hourly_means.index += pd.DateOffset(minutes=30)
#                 new_array.append(hourly_means)
#                 if i == 0:
#                     prim_tim_df4 = hourly_means.index[24:]
#                 if i % 1000 == 0:
#                     print('Run number' + str(i).zfill(3))
#             loaded_data = np.transpose(np.array(new_array)[:,:,0])[24:]
        
#             warnings.resetwarnings()
#             if var == 'sh':
#                 print('Making Surface Height and Running...')
#                 with open(path_to_data, 'wb') as file:
#                     pickle.dump(loaded_data, file)
#             elif var == 'sal':
#                 print('Making Salinity and Running...')
#                 with open(path_to_data, 'wb') as file:
#                     pickle.dump(loaded_data, file)
#             with open(path + '/time_data_half_hour_sampled.pickle', 'wb') as file:
#                 pickle.dump(np.transpose(prim_tim_df4), file)
#         warnings.resetwarnings()
#         # # code that goes here sorts the data out into half hour intervals
#         # prim_tim_df3 = prim_tim_df2[3:-4]
#         # prim_tim_df4 = prim_tim_df3[::6] 

#         #ukc3_tim_df = ukc3_time[24:]
        
#         common_idx = ukc3_tim_df[ukc3_tim_df['ukc3_time'].isin(prim_tim_df4)].index
#         if index_only != 'yes':
#             #happy to identify that they are the same. 
#             sliced_ukc3_tim_df = ukc3_tim_df.iloc[common_idx]
#             sliced_ukc3_sh_df = UKC3_interp_z_interp[common_idx,:]
#             sliced_ukc3_tim_df = sliced_ukc3_tim_df
#             sliced_ukc3_sh_df2 = sliced_ukc3_sh_df
            
            
#             # sh_primea = primea_data2[3:-4,:]
#             sh_primea2 = loaded_data#sh_primea[::6,:]
            
#             #calculate_hourly_means(sh_primea)
            
#             sh_ukc3 = sliced_ukc3_sh_df2
        
#             return sh_primea2, sh_ukc3, common_idx, UKC3_interp_z_interp
#         else:
#             return common_idx
        
#     def load_data(self): 
#         globbed = sorted(glob.glob(os.path.join( self.tide_gauge_path, '*.csv')))
#         self.names = []
#         self.df = []
#         for i in globbed:
#             self.names.append(os.path.split(i)[-1][:-4]  )  
#             self.df.append(pd.read_csv(i))
#         print(self.df, self.names)
        
        
#         #In this case UKC4 was interpolated onto PRIMEA grid, load in that data
        
#         #/media/af/PD/modelling_DATA/kent_estuary_project/6.Final2/processed_data/ukc4_on_PRIMEA_grid/sossheig_ogUKC3_.nc
#         self.ukc3_dataset = xr.open_dataset(os.path.join(self.ukc3_int_path, 'UKC3_og_interpolated_PRIMEA_grid_data.nc'))
#         print(self.ukc3_dataset.z_interpolated_time)
        
#         self.primea_dataset = xr.open_dataset(self.prim_data, chunks={'time':100})
        
        
#     def run_stats(self):
#         self.load_data()
# # if __name__ == "__main__":
# #     from o_func import DataChoice, DirGen
# #     import glob
    
# #     #%% Making Directory paths
# #     main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
# #     make_paths = DirGen(main_path)
# #     ### Finishing directory paths
    
# #     dc = DataChoice(os.path.join(main_path,'models'))
# #     fn = dc.dir_select()
# #     sub_path = make_paths.dir_outputs(os.path.split(fn[0])[-1])
# #     lp = glob.glob(os.path.join(sub_path, '*.nc'))[0]
# #     rs = r_stats(prim_data = lp)
# #     rs.load_data()
        
        
#%%   
# It is worth saving the original copy of the data for some comparative analysis. 
#example_dataset = os.path.join(start_path, 'modelling_DATA','kent_estuary_project',r'6.Final2','models','kent_1.3.7_testing_4_days_UM_run','kent_regrid.nc')
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
     self.tide_gauge_data = glob.glob(os.path.join(start_path,
                                               'modelling_DATA',
                                               'kent_estuary_project',
                                               'validation',
                                               'tidal_validation',
                                               r'1.reformatted', 'tide*'))
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
    @staticmethod
    def print_dict_keys(dictionary, indent=0):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                print(' ' * indent + f"{key}:")
                stats.print_dict_keys(value, indent + 2)
            else:
                print(' ' * indent + f"{key}")
         
    def load_raw(self): # Need to be able to run it for every dataset
        self.raw_data = xr.open_dataset(self.data)
        # self.time_shifted = self.time_shifter()
        data_dict = {}
        for dataset_var in [i for i in var_dict]:
            prim_datasets = {var: self.raw_data[var] for var in self.raw_data.data_vars if var.startswith('prim')}
            ukc4_datasets = {var: self.raw_data[var] for var in self.raw_data.data_vars if var.startswith('ukc4')}
           
            def rename_dict(dictionary):
                modified_dict = {}
                for old_key, value in dictionary.items():
                    new_key = old_key[5:]  # Remove the first 5 characters
                    modified_dict[new_key] = value
                return modified_dict
            
            if ukc4_datasets: # doesnt matter which one it is
                for (primkey, primvalue), (ukc4key, ukc4value) in zip(prim_datasets.items(), ukc4_datasets.items()):
                    prim_datasets[primkey] = primvalue.resample(time_primea='1H').mean(skipna = True)
                    prim_datasets[primkey]['time_primea'] = prim_datasets[primkey].time_primea + pd.to_timedelta('30min')
                    ukc4_datasets[ukc4key] = ukc4_datasets[ukc4key].sel(time_counter=prim_datasets[primkey].time_primea)
                    self.lon = prim_datasets[primkey]['nav_lon']
                    self.lat = prim_datasets[primkey]['nav_lat']
                    self.time = prim_datasets[primkey]['time_primea']
                    # print(self.overall_time)
                    matching_times = (prim_datasets[primkey]['time_primea'] == ukc4_datasets[ukc4key]['time_counter']).all()
                    if not matching_times:
                        print("The time arrays do not match.")
                        print("Exiting script, take a look at the code...")
                        sys.exit()
                
                
                
                data_dict['ukc4'] = rename_dict(ukc4_datasets) # This is now a dictionary of the data
                data_dict['prim'] = rename_dict(prim_datasets)
                
        self.data_dict = data_dict
        return self.raw_data, data_dict, matching_times
    ####            0                1          2
    
    def load_tide_gauge(self):   
        self.tide_loc_dict = {'Heysham'  :{'x':-2.9311759, 'y':54.0345516},
                              'Liverpool':{'x':-3.0168250, 'y':53.4307320},
                              }
        df_tide_loc = pd.DataFrame(self.tide_loc_dict).T.reset_index()
        df_tide_loc = df_tide_loc.drop(df_tide_loc.columns[0], axis=1)
        df_search_points = pd.DataFrame({'x': self.lon.data.ravel(), 'y': self.lat.data.ravel()})
       # print(self.lat.data.shape)
        tide_dict = {}
        for i in self.tide_gauge_data:
            dataset = pd.read_csv(i, index_col=0, parse_dates=True)
            name = os.path.split(i)[-1][5:-4].capitalize()
            tide_dict[name] = dataset.resample('H').mean()
            tide_dict[name].index = tide_dict[name].index + pd.Timedelta(minutes=30)
            tide_dict[name] = tide_dict[name].loc[(tide_dict[name].index >= self.time[0].data) & (tide_dict[name].index <= self.time[-1].data)]
        self.tide_data_dict = tide_dict
        # print(self.tide_data_dict)
        # print(df_tide_loc)
        #print(self.lon.data.ravel())
        # print(df_search_points)
        dist, indices = near_neigh(df_tide_loc,df_search_points,1)
        
        self.tide_gauge_coords = np.unravel_index(indices, self.lon.shape)
        empty_ind = []
        for k in indices:
            empty_ind.append(divmod(k[0],np.shape(self.lon)[1]))
        df_search_points
        
        return self.tide_gauge_coords, empty_ind
        
    def linear_regression(self, figpath):
        ''' Uses one point in the dataset, like a tide gauge and samples points through time
        '''
        rubbish, tide_gauge_coords = stats.load_tide_gauge(self)
        prim_dict = self.data_dict['prim']
        ukc4_dict = self.data_dict['ukc4']
        common_keys = set(ukc4_dict.keys()) & set(prim_dict.keys())
        extract_prims = []
        extract_ukc4s = []
        for variable_name in common_keys:
            for tide_gauge in tide_gauge_coords:
                # processing the data
                ukc4_data = ukc4_dict[variable_name]
                prim_data = prim_dict[variable_name]
                extract_prims.append(prim_data)
                extract_ukc4s.append(ukc4_data)
                x = tide_gauge[0] # Now the location has been determined you can apply elsewhere. 
                y = tide_gauge[1]
                # Insert location of tide_gauges_here
                primx = prim_data[4:,x,y].data.flatten() # at the testing points [4:,40,20] it is almost identical. 
                ukc4y = ukc4_data[4:,x,y].data.flatten()
                # Plotting up the figures
                plt.figure()
                plt.scatter(primx,ukc4y)    
                
                #print(ukc4_data.time_primea.shape)
                #print(primx.shape)
                # plt.figure()
                # plt.scatter(ukc4_data.time_primea[4:],primx)
                # plt.scatter(ukc4_data.time_primea[4:],ukc4y)
                
                plt.savefig('/home/af/Desktop/'+variable_name+'temp.png', dpi = 300)
                # Statistical processing
                coefficients = np.polyfit(primx, ukc4y, 1)
                regression_line = np.poly1d(coefficients)
                r_squared = np.corrcoef(primx, ukc4y)[0, 1] ** 2 
                # ax[i].plot(subset_x, regression_line(subset_x), label='Regression Line', c = 'b', linewidth = 3)  # plot the regression line
                # ax[i].plot(subset_x, subset_x, label='y=x', c = 'orange', linewidth = 3)  # plot the y=x line for comparison
                print (' other r_squared =' ,r_squared)
                
            # print(coefficients)
        
              # coefficients = np.polyfit(data.flatten(), np.arange(data.shape[0]).repeat(data.shape[1]), 1)
        # return ukc4_data#prim_data
        return extract_prims, extract_ukc4s
        
        
if __name__ == '__main__':
    import time
    from o_func import DataChoice, DirGen
    import glob
    main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
    make_paths = DirGen(main_path)
    ### Finishing directory paths
    
    dc = DataChoice(os.path.join(main_path,'models'))
    #fn = os.path.split(dc.dir_select()[0])[-1]
    #fn = 'kent_1.3.7_testing_4_days_UM_run' # bad model for testing, had issues. 
    fn = 'kent_1.0.0_UM_wind' # bad model for testing, had issues. 

    sub_path, fig_path = make_paths.dir_outputs(fn)
    lp = glob.glob(os.path.join(sub_path, '*.nc'))[0]
    sts = stats(lp)
    load = sts.load_raw()
    stats.print_dict_keys(load[1]) # prints out the dictionaries of data being used. 
    extract_prims, extract_ukc4s = sts.linear_regression(fig_path)
    tide_gauge, ind = sts.load_tide_gauge()
    # SANITY CHECKER
    # for i in range(30):# store linear reression in known figpath
    #     plt.figure()
    #     plt.title('Print out time interpolated ')
    #     print(i)
    #     time.sleep(0.5)
    #     plt.pcolor(extract_prims[0].nav_lon, extract_prims[0].nav_lat,extract_prims[0][i,:,:])
    #     for j in ind:
    #         print(j)
    #         x = j[0]
    #         y = j[1]
    #         plt.scatter(extract_prims[0][:,x,y].nav_lon.values, extract_prims[0][:,x,y].nav_lat.values, color = 'r')
    #         #plt.scatter([-2.9311759,-3.0168250],[54.0345516,53.4307320])
    #     plt.scatter([-2.9311759,-3.0168250],[54.0345516,53.4307320])
        
    #     plt.savefig('/home/af/Desktop/temp.png', dpi = 300)
        
        
    # for i in range(20):# store linear reression in known figpath
    #     plt.title('Print out raw data ')

    #     plt.figure()
    #     print(i)
    #     time.sleep(0.5)
    #     plt.pcolor(extract_prims[0].nav_lon, extract_prims[0].nav_lat,load[0]['prim_surface_height'][i,:,:])
    #     #plt.scatter(extract_prims[:,41,10].nav_lon.values, extract_prims[:,41,10].nav_lat.values, color = 'r')
    #     plt.savefig('/home/af/Desktop/temp.png', dpi = 300)
