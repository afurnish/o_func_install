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
'surface_height'   : {'TUV':'T',  'UKC4':'sossheig',       'PRIMEA':'mesh2d_s1',  'UNITS':'m'  },
'surface_salinity' : {'TUV':'T',  'UKC4':'vosaline_top',   'PRIMEA':'mesh2d_sa1', 'UNITS':'psu' },
'middle_salinity'  : {'TUV':'T',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
'bottom_salinity'  : {'TUV':'T',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
'surface_Uvelocity': {'TUV':'U',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
'middle_Uvelocity' : {'TUV':'U',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
'bottom_Uvelocity' : {'TUV':'U',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
'surface_Vvelocity': {'TUV':'V',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
'middle_Vvelocity' : {'TUV':'V',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
'bottom_Vvelocity' : {'TUV':'V',  'UKC4':'',               'PRIMEA':'na',         'UNITS':'na' },
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
    
    @staticmethod
    def format_string(string):
        # Replace underscores with spaces and capitalize each word
        formatted_string = string.replace('_', ' ').title()
        return formatted_string
    
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
        # import pdb; pdb.set_trace()
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
        
        dist, indices = near_neigh(df_tide_loc,df_search_points,1)
        self.df_search_points = df_search_points
        self.tide_gauge_coords = np.unravel_index(indices, self.lon.shape)
        self.empty_ind = []
        for k in indices:
            self.empty_ind.append(divmod(k[0],np.shape(self.lon)[1]))
        df_search_points
        
        return self.empty_ind # self.tide_gauge_coords,
        
    def linear_regression(self, figpath, data_stats_path):
        ''' Uses one point in the dataset, like a tide gauge and samples points through time
        
            This will also plot out the tidal data
        '''
        with open(os.path.join(data_stats_path, 'r_squared_stats.txt'), "w") as f:
            f.write("{:<20} {:<10} {:<10} {:<10} {:<20}\n".format("Variable", "Tide Gauge", "X", "Y", "R-squared"))

        
        self.tide_save = stats.load_tide_gauge(self)
         
        prim_dict = self.data_dict['prim']
        ukc4_dict = self.data_dict['ukc4']
        self.common_keys = set(ukc4_dict.keys()) & set(prim_dict.keys())
        extract_prims = []
        extract_ukc4s = []
        slicer = 50
        def min_max_scaling(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        
        fig2, ax2 = plt.subplots(1, len(self.tide_save)) # this is the figure for timeseries plots
        
        for variable_name in self.common_keys:
            
            def figs():
                fig, ax = plt.subplots(1, len(self.tide_save), sharey=True, sharex=True) # this is the figure for correlation plots
                fig.set_figheight(3.5)
                fig.set_figwidth(7)
                return fig, ax
            fig, ax = figs()
            ukc4_data = ukc4_dict[variable_name] 
            prim_data = prim_dict[variable_name]
            unit = var_dict[variable_name]['UNITS']
            extract_prims.append(prim_data)
            extract_ukc4s.append(ukc4_data)
            
            
            if variable_name == 'surface_height':  # there is surely a better way to do this
                self.ukc4_sh = ukc4_data # save for later processing
                self.prim_sh = prim_data
    
            # fig, ax = plt.subplots(1, len(self.tide_save), sharey=True, sharex=True)
            
            def make_plot(plottingx = 'prim', plottingy = 'ukc4', corr_tide = 'corr'):
                for i, tide_gauge in enumerate(self.tide_save):
                    
                    tide_gauge_name = [j for j in self.tide_loc_dict.keys()][i]
                    x,y = tide_gauge[0],tide_gauge[1] # Now the location has been determined you can apply elsewhere. 
                    primx = {'PRIMEA Model' : min_max_scaling(prim_data[slicer:,x,y].data.flatten())} # at the testing points [4:,40,20] it is almost identical. 
                    ukc4y = {'UKC4 Model' : min_max_scaling(ukc4_data[slicer:,x,y].data.flatten())}
                    tidex = {'Measured Tide Gauge' : min_max_scaling(self.tide_data_dict[tide_gauge_name].Height[slicer:])}
                    # can rerun this multiple times for surface height only. 
                    
                    def make_subplot(plotx, ploty):
                        xname = [i for i in plotx.keys()][0]
                        yname = [i for i in ploty.keys()][0]
                        coefficients = np.polyfit(plotx[xname], ploty[yname], 1)
                        regression_line = np.poly1d(coefficients)
                        r_squared = np.corrcoef(plotx[xname], ploty[yname])[0, 1] ** 2 
                        
                        # Plotting up the figures    
                        
                        ax[i].scatter(plotx[xname],ploty[yname], s = 1, label = xname + ' vs '+ yname + ' correlation')
                        ax[i].plot(plotx[xname], regression_line(plotx[xname]), label='Regression Line', c = 'b', linewidth = 0.25)  # plot the regression line
                        ax[i].plot(plotx[xname], plotx[xname], label='y=x', c = 'orange', linewidth = 0.25)  # plot the y=x line for comparison
                        
                        fig.text(0.5, 0.01, xname + ' (normalised ' + unit + ')', ha='center', va='bottom', transform=fig.transFigure)
                        fig.text(0.03, 0.5, yname + ' (normalised ' + unit + ')', va='center', rotation='vertical')
                        ax[i].set_title(tide_gauge_name + '\n' + f'(R$^2$ ={r_squared:.2f})')
                        ax[i].set_aspect('equal')   
                        ax[i].legend(loc = 'lower right', fontsize = 4, frameon=False)
                        
                        
                        # a =  (variable_name,' ', tide_gauge_name, ' ',plottingx,' ', plottingy,' other r_squared =' ,r_squared)
                        with open(os.path.join(data_stats_path, 'r_squared_stats.txt'), "a") as f:
                            table_to_return =  (variable_name, tide_gauge_name, plottingx, plottingy, r_squared)
                            print("{:<20} {:<10} {:<10} {:<10} {:<20}".format(*table_to_return))
                            f.write("{:<20} {:<10} {:<10} {:<10} {:<20}".format(*table_to_return))
                            f.write('\n')
                        return xname, yname
                    
                    if plottingx == 'tidegauge' :
                        passx = tidex
                    elif plottingx == 'ukc4':
                        passx = ukc4y
                    elif plottingx == 'prim':
                        passx = primx
                        
                    if plottingy == 'tidegauge' :
                        passy = tidex
                    elif plottingy == 'ukc4':
                        passy = ukc4y
                    elif plottingy == 'prim':
                        passy = primx
                        
                    xname , yname = make_subplot(passx, passy)
                    
                return xname.replace(' ', '_'), yname.replace(' ', '_')
                    # Here run the plotting sctipt 3 times. 
            xname, yname = make_plot()        
            fig.savefig(os.path.join(fig_path,'correlation_' + variable_name + '_' + xname + '_' + yname + '_'+ '.png'), dpi = 300)
            plt.close(fig)
            if variable_name == 'surface_height':
                fig, ax = figs()
                xname, yname = make_plot('prim', 'tidegauge')   
                fig.savefig(os.path.join(fig_path,'correlation_' + variable_name + '_' + xname + '_' + yname + '_'+ '.png'), dpi = 300)
                plt.close(fig)
                fig, ax = figs()
                xname, yname = make_plot('ukc4', 'tidegauge')   
                fig.savefig(os.path.join(fig_path,'correlation_' + variable_name + '_' + xname + '_' + yname + '_'+ '.png'), dpi = 300)
                plt.close(fig)
                
                # ax[i].plot(subset_x, regression_line(subset_x), label='Regression Line', c = 'b', linewidth = 3)  # plot the regression line
                # ax[i].plot(subset_x, subset_x, label='y=x', c = 'orange', linewidth = 3)  # plot the y=x line for comparison
            
                
            # print(coefficients)
        
              # coefficients = np.polyfit(data.flatten(), np.arange(data.shape[0]).repeat(data.shape[1]), 1)
        # return ukc4_data#prim_data
        return extract_prims, extract_ukc4s
    
    def tidal_plots(self, fig_path):
        '''
        I need to plot the number of tide guage locations, 
        which then plots 2 figures, in each figure should be plotted
        the tide gauge, ukc4 and primea data values. 
        
        This can then be done for salinity ensuring to only plot the tide 
        gauge if the data exists. 
        '''
        
        
        prim_dict = self.data_dict['prim']
        ukc4_dict = self.data_dict['ukc4']
        
        fig, ax = plt.subplots(len(self.tide_save), 1)
        for i, variable_name in enumerate(self.common_keys):
            for tide_gauge in self.tide_save:
                x = tide_gauge[0] # Now the location has been determined you can apply elsewhere. 
                y = tide_gauge[1]
                
                ukc4_data = ukc4_dict[variable_name]
                prim_data = prim_dict[variable_name]
                primx = prim_data[50:,x,y].data.flatten() # at the testing points [4:,40,20] it is almost identical. 
                ukc4y = ukc4_data[50:,x,y].data.flatten()
                ax[i].plot(self.time[50:], primx)
                ax[i].plot(self.time[50:], ukc4y)
                
                
                # plt.savefig(os.path.join(fig_path, 'tide_gauge_validation_' + variable_name +'.png'), dpi = 150)
                plt.close
        
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

    def max_compare(self, fig_path):
        prim_dict = self.data_dict['prim']
        ukc4_dict = self.data_dict['ukc4']
        #print(prim_dict)
        
        for i in ['surface_height', 'surface_salinity']:#, 'surface_salinity']:
            prim = prim_dict[i][50:,:,:]
            ukc4 = ukc4_dict[i][50:,:,:]
            
            for j in ['min', 'max']:
                if j == 'min':
                    heightprim = prim.min(dim='time_primea')
                    heightukc4 = ukc4.min(dim='time_primea')
                else:
                    heightprim = prim.max(dim='time_primea')
                    heightukc4 = ukc4.max(dim='time_primea')
            
                height_difference = heightprim - heightukc4
                fig, ax = plt.subplots()
                fig.set_figheight(7)
                fig.set_figwidth(5)
                pcm = ax.pcolor(self.lon, self.lat, height_difference)
                pcm.set_array(height_difference)
                #pcm.set_clim(-1, 1)
                cbar = plt.colorbar(pcm)
                #cbar.set_ticks(np.linspace(-1,1,11))
                cbar.set_label(j + ' Diff between $\mathrm{UKC4}_{\mathrm{PRIMEA}}$ and $\mathrm{UKC4}_{\mathrm{ao}}$ (m)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.tight_layout()
                
                plt.savefig(fig_path + '/' + j + '_'+ i +'_analysis.png', dpi = 300)
                plt.close()
            
        

            
        return heightprim, heightukc4, height_difference
        
        
        
#%%        
        
if __name__ == '__main__':
    from o_func import DataChoice, DirGen
    import glob
    main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
    make_paths = DirGen(main_path)
    ### Finishing directory paths
    
    dc = DataChoice(os.path.join(main_path,'models'))
    #fn = os.path.split(dc.dir_select()[0])[-1]
    #fn = 'kent_1.3.7_testing_4_days_UM_run' # bad model for testing, had issues. 
    fn = 'kent_1.0.0_UM_wind' # bad model for testing, had issues. 

    sub_path, fig_path, data_stats_path = make_paths.dir_outputs(fn)
    lp = glob.glob(os.path.join(sub_path, '*.nc'))[0]
    sts = stats(lp)
    load = sts.load_raw()
    stats.print_dict_keys(load[1]) # prints out the dictionaries of data being used. 
    extract_prims, extract_ukc4s = sts.linear_regression(fig_path, data_stats_path)
    tide_gauge, ind = sts.load_tide_gauge()
    transect = sts.transect(fig_path)
    prim, ukc4, height_diff = sts.max_compare(fig_path)
    tp = sts.tidal_plots(fig_path)
    
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
