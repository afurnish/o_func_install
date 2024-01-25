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
from o_func.utilities.distance import Dist

   
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
        self.bathymetry = self.raw_data['prim_bathymetry'][0,:,:]        
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
        self.df_search_points = df_search_points
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
                if variable_name == 'surface_height':
                    self.ukc4_sh = ukc4_data
                    self.prim_sh = prim_data
                    
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
                # plt.scatter(ukc4_data.time_primea[4:],primx) # 
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
    
    
    def transect(self, fig_path):
        transect_paths = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/analysis/QGIS_shapefiles/points_along_estuary_1km_spacing.csv'
        transect_data = pd.read_csv(transect_paths)
        transect_data = transect_data.rename(columns = {'X':'x','Y':'y'})
        distances, indicies = near_neigh(transect_data,self.df_search_points,1)
        empty_ind = []
        disti = []
        for k in indicies:
            empty_ind.append(divmod(k[0],np.shape(self.lon)[1]))
        unique_estuaries = np.unique(transect_data.id)
        
        prim_time_series_at_locations = []
        ukc4_time_series_at_locations = []
        bathymetry_at_locations = []
        # Iterate over the specified locations and extract time series
        for x, y in empty_ind:
            prim_time_series_at_locations.append(self.prim_sh[4:, x, y])
            ukc4_time_series_at_locations.append(self.ukc4_sh[4:, x, y])
            bathymetry_at_locations.append(self.bathymetry[x, y])
        prim_time_series_at_locations = np.array(prim_time_series_at_locations).T
        ukc4_time_series_at_locations = np.array(ukc4_time_series_at_locations).T
        bathymetry_at_locations = np.array(bathymetry_at_locations).T
        
        maxy = np.max([np.nanmax(prim_time_series_at_locations), np.nanmax(ukc4_time_series_at_locations), np.nanmax(bathymetry_at_locations)])
        miny = np.min([np.nanmin(prim_time_series_at_locations), np.nanmin(ukc4_time_series_at_locations), np.nanmin(bathymetry_at_locations)])

        # run the plotter for each estuary
        def plotter(minmax):
            fig, ax = plt.subplots(unique_estuaries.shape[0])
            fig.set_figheight(15)
            fig.set_figwidth(10)
            for i in unique_estuaries:
                sub_frame = transect_data.loc[transect_data['id'] == i]
                dis = [0]
                sub_frame_primea = prim_time_series_at_locations[:,sub_frame.index]
                sub_frame_ukc3 = ukc4_time_series_at_locations[:,sub_frame.index]
                sub_frame_bathymetry = bathymetry_at_locations[sub_frame.index]
                #print(sub_frame_bathymetry[i].shape)
                if minmax == 'max':
                    min_primea = np.max(sub_frame_primea, axis=0)
                    min_ukc3 =  np.max(sub_frame_ukc3, axis=0)
                elif minmax == 'min':
                    min_primea = np.min(sub_frame_primea, axis=0)
                    min_ukc3 =  np.min(sub_frame_ukc3, axis=0)
                else:
                    min_primea = sub_frame_primea[minmax,:]
                    min_ukc3 =  sub_frame_ukc3[minmax,:]
                for j in range(sub_frame.shape[0]-1):
                    #lat1, lon1, lat2, lon2
                    d = Dist.dist_between_points(sub_frame.y.iloc[j],sub_frame.x.iloc[j], sub_frame.y.iloc[j+1],sub_frame.x.iloc[j+1])
                    dis.append(d)
                new_dist = np.cumsum(dis)
                disti.append(new_dist)
                
                ax[i].plot(new_dist, min_primea, 'r', label = '$\mathrm{UKC4}_{\mathrm{PRIMEA}}$')
                ax[i].plot(new_dist, min_ukc3, 'b', label = '$\mathrm{UKC4}_{\mathrm{ao}}$')
                ax[i].plot(new_dist, sub_frame_bathymetry, 'g', label = 'Bathymetry')
                ax[i].set_title(sub_frame.est_name.iloc[0].capitalize())
                ax[i].set_ylim([miny, maxy])
                fig.supxlabel("Distance transecting estaury Mouth-River (km)")
                fig.supylabel("Height of surface (water or land) (m)")
                if i == 0:
                    fig.legend(loc='upper center', bbox_to_anchor=(0.8, 1.00), ncol=4)
                    
                plt.tight_layout()
            # if isinstance(minmax, str):
            plt.savefig(fig_path + '/timestep_'+ str(minmax) +'_transects_along_estuaries.png', dpi = 300)
            # else:
            #     #plt.subplots_adjust(top=0.5)
            #     #fig.suptitle(str(sliced_ukc3_tim_df.iloc[minmax][0]))
            #     plt.savefig(fig_path + '/timestep_'+ str(minmax) +'_transects_along_estuaries.png', dpi = 300)
             
            return sub_frame_primea
                #print(self.prim_sh)
        sub_frame_primea = plotter(106)
        sub_frame_primea = plotter(100)
        return sub_frame_primea

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
    transect = sts.transect(fig_path)
    
    
    
# EXTRA PLOTTING
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
