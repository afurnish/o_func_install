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
import matplotlib.dates as mdates
import cmocean


from o_func import opsys; start_path = opsys()
from o_func.utilities.near_neigh import near_neigh
from o_func.utilities.distance import Dist
from o_func.utilities.gauges import tide_gauge_loc
   
#example_dataset = os.path.join(start_path, 'modelling_DATA','kent_estuary_project',r'6.Final2','models','kent_1.3.7_testing_4_days_UM_run','kent_regrid.nc')
var_dict = {
'surface_height'   : {'TUV':'T',  'UKC4':'sossheig',       'PRIMEA':'mesh2d_s1',     'UNITS':'m'},
'surface_salinity' : {'TUV':'T',  'UKC4':'vosaline_top',   'PRIMEA':'mesh2d_sa1',    'UNITS':'psu'},
'middle_salinity'  : {'TUV':'T',  'UKC4':'vosaline_mid',   'PRIMEA':'na',            'UNITS':'psu'},
'bottom_salinity'  : {'TUV':'T',  'UKC4':'vosaline_bot',   'PRIMEA':'na',            'UNITS':'psu'},
'surface_temp'     : {'TUV':'T',  'UKC4':'votemper_top',   'PRIMEA':'na',            'UNITS':'\u00B0C'},
'middle_temp'      : {'TUV':'T',  'UKC4':'votemper_mid',   'PRIMEA':'na',            'UNITS':'\u00B0C'},
'bottom_temp'      : {'TUV':'T',  'UKC4':'votemper_bot',   'PRIMEA':'na',            'UNITS':'\u00B0C'},
'surface_Uvelocity': {'TUV':'U',  'UKC4':'vozocrtx_top',   'PRIMEA':'mesh2d_ucx',    'UNITS':'$m\,s^{-1}$'}, # the one with the major issues
'middle_Uvelocity' : {'TUV':'U',  'UKC4':'vozocrtx_mid',   'PRIMEA':'na',            'UNITS':'$m\,s^{-1}$'},
'bottom_Uvelocity' : {'TUV':'U',  'UKC4':'vozocrtx_bot',   'PRIMEA':'na',            'UNITS':'$m\,s^{-1}$'},
'surface_Vvelocity': {'TUV':'V',  'UKC4':'vomecrty_top',   'PRIMEA':'mesh2d_ucy',    'UNITS':'$m\,s^{-1}$'},
'middle_Vvelocity' : {'TUV':'V',  'UKC4':'vomecrty_mid',   'PRIMEA':'na',            'UNITS':'$m\,s^{-1}$'},
'bottom_Vvelocity' : {'TUV':'V',  'UKC4':'vomecrty_bot',   'PRIMEA':'na',            'UNITS':'$m\,s^{-1}$'},
'bathymetry'       : {'TUV':'T',  'UKC4':'NA',             'PRIMEA':'mesh2d_node_z', 'UNITS':'m'},
}
class Stats:
    def __init__(self, dataset, dataset_name):
     self.tide_gauge_path = os.path.join(start_path,
                                         'modelling_DATA',
                                         'kent_estuary_project',
                                         'validation',
                                         'tidal_validation',
                                         r'1.reformatted')
     self.dataset_name = dataset
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
                Stats.print_dict_keys(value, indent + 2)
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
        self.tide_loc_dict = {
                               'Heysham'  :{'x':-2.9594670, 'y':54.0328370},
                               # 'Heysham'  :{'x':-2.9574780, 'y':54.0333366},
                               # 'Liverpool':{'x':-3.1554490, 'y':53.4930250}, # Looks good but too deep on liverpool 
                               # 'Liverpool':{'x':-3.1391550, 'y':53.4622030}, 
                               # 'Liverpool':{'x':-3.1988680, 'y':53.4797420}, 
                                 'Liverpool':{'x':-3.0741720, 'y':53.4634140}, 
                              }
        
        # self.tide_loc_dict = tide_gauge_loc()
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
        
    
    def load_ocean_bound(self):
        pass
        # ocean_bnd = 
    
    def linear_regression(self, figpath, data_stats_path):
        ''' Uses one point in the dataset, like a tide gauge and samples points through time
        
            This will also plot out the tidal data
        '''
        # Start to write outputs to file. 
        with open(os.path.join(data_stats_path, 'r_squared_stats.txt'), "w") as f:
            f.write("{:<20} {:<10} {:<20} {:<15} {:<20}\n".format("Variable", "Tide Gauge", "X", "Y", "R-squared"))
            f.write((89//2)*"-*" + "\n")
        with open(os.path.join(data_stats_path, 'RMSE_stats.txt'), "w") as f:
            f.write("{:<20} {:<10} {:<20} {:<15} {:<20}\n".format("Variable", "Tide Gauge", "X", "Y", "RMSE"))
            f.write((89//2)*"-*" + "\n")
        self.tide_save = Stats.load_tide_gauge(self)
         
        prim_dict = self.data_dict['prim']
        ukc4_dict = self.data_dict['ukc4']
        self.common_keys = set(ukc4_dict.keys()) & set(prim_dict.keys())
        extract_prims = []
        extract_ukc4s = []
        slicer = 50
        
        
        # fig2, ax2 = plt.subplots(1, len(self.tide_save)) # this is the figure for timeseries plots
        
        for variable_name in self.common_keys: # set up common keys to remove bathymetry etc. 
            
            def figs():
                fig, ax = plt.subplots(1, len(self.tide_save), sharey=True, sharex=True) # this is the figure for correlation plots
                fig.set_figheight(3.5)
                fig.set_figwidth(7)
                
                return fig, ax
           
            ukc4_data = ukc4_dict[variable_name] 
            prim_data = prim_dict[variable_name]
            unit = var_dict[variable_name]['UNITS']
            extract_prims.append(prim_data)
            extract_ukc4s.append(ukc4_data)
            self.time_sliced = self.time[slicer:]
            
            if variable_name == 'surface_height':  # there is surely a better way to do this
                self.ukc4_sh = ukc4_data # save for later processing
                self.prim_sh = prim_data
                
            def make_data(): # save the data inside variables for use later for each tide gauge. 
                self.primx = []
                self.ukc4y = []
                self.tidex = []
                for i, tide_gauge in enumerate(self.tide_save):
                    
                    tide_gauge_name = [j for j in self.tide_loc_dict.keys()][i]
                    x,y = tide_gauge[0],tide_gauge[1] # Now the location has been determined you can apply elsewhere. 
                    self.primx.append({'PRIMEA Model' : prim_data[slicer:,x,y].data.flatten()}) # at the testing points [4:,40,20] it is almost identical. 
                    self.ukc4y.append({'UKC4 Model' : ukc4_data[slicer:,x,y].data.flatten()})
                    self.tidex.append({'Measured Tide Gauge' : self.tide_data_dict[tide_gauge_name].Height[slicer:]})
            # fig, ax = plt.subplots(1, len(self.tide_save), sharey=True, sharex=True)
            # 
            def calculate_rmse(actual, predicted):
                """Calculate the Root Mean Square Error between two arrays."""
                differences = actual - predicted  # Element-wise differences
                squared_differences = differences ** 2  # Squared differences
                mean_squared_difference = np.mean(squared_differences)  # Mean of squared differences
                rmse = np.sqrt(mean_squared_difference)  # Square root of the mean
                return rmse
            
            make_data()
            # import pdb; pdb.set_trace()
            def corr_plot(x, y, n = 'y'): 
                '''
            

                Parameters
                ----------
                plotx : Refers to the dataset that should be plotted for the x
                ploty : refers to the dataset that should be plotted for the y

                Returns
                -------
                None.

                '''
                def min_max_scaling(data):
                    return (data - np.min(data)) / (np.max(data) - np.min(data))

                
                fig, ax = figs() # Generate a figure the size of the two tide gauges. 
                for i, tide_gauge in enumerate(self.tide_save):
                    # produces Heysham and Liverpool as strings. 
                    plotx = x[i]     # x dataset pulled out
                    ploty = y[i]     # y dataset for one location held in dictionary of its name. 
                    
                    xname, yname = [i for i in plotx.keys()][0], [i for i in ploty.keys()][0] # get keys
                    if n == 'y':
                        plotx = min_max_scaling(plotx[xname])
                        ploty = min_max_scaling(ploty[yname])
                    coefficients = np.polyfit(plotx, ploty, 1)
                    regression_line = np.poly1d(coefficients)
                    r_squared = np.corrcoef(plotx, ploty)[0, 1] ** 2 
                    
                    # Plotting up the figures    
                    
                    ax[i].scatter(plotx,ploty, s = 1, label = xname + ' vs '+ yname + ' correlation')
                    ax[i].plot(plotx, regression_line(plotx), label='Regression Line', c = 'b', linewidth = 0.25)  # plot the regression line
                    ax[i].plot(plotx, plotx, label='y=x', c = 'orange', linewidth = 0.25)  # plot the y=x line for comparison
                    
                    fig.text(0.5, 0.01, xname + ' (normalised ' + unit + ')', ha='center', va='bottom', transform=fig.transFigure)
                    fig.text(0.03, 0.5, yname + ' (normalised ' + unit + ')', va='center', rotation='vertical')
                    tide_gauge_name = [j for j in self.tide_loc_dict.keys()][i]
                    ax[i].set_title(tide_gauge_name + '\n' + f'(R$^2$ ={r_squared:.2f})')
                    ax[i].set_aspect('equal')   
                    ax[i].legend(loc = 'lower right', fontsize = 4, frameon=False)
                    with open(os.path.join(data_stats_path, 'r_squared_stats.txt'), "a") as f:
                        table_to_return =  (variable_name, tide_gauge_name, xname, yname, r_squared)
                        print("{:<20} {:<10} {:<20} {:<15} {:<20}".format(*table_to_return))
                        f.write("{:<20} {:<10} {:<20} {:<15} {:<20}".format(*table_to_return))
                        f.write('\n')
                # plt.title(self.dataset_name)       
                fig.savefig(os.path.join(fig_path,'correlation_' + variable_name + '_' + xname.replace(' ','_') + '_' + yname.replace(' ','_') + '_'+ '.png'), dpi = 300)
                plt.close(fig)
            
            #plot up ukc4 vs prim for all pairs of variables. 
            # import pdb; pdb.set_trace()
            
            primx = self.primx
            ukc4y = self.ukc4y
            tidex = self.tidex
            corr_plot(primx, ukc4y)
            if variable_name == 'surface_height':
                corr_plot(tidex, primx) # plot primea vs tide gauge. 
                corr_plot(tidex, ukc4y) # plot ukc4 vs tide gauge. 
                         
            def timeseries_plot(list_of_data):
                # len_list = len(list_of_data) # how many plots on one figure. 
                # num_of_figs = len(list_of_data[0]) # how many tide gauge plots in total. 
                
                
                
                
                for i, tide_gauge in enumerate(self.tide_save):
                    fig, ax = plt.subplots(dpi = 300) # this is the figure for correlation plots
                    fig.set_figheight(4) # plotting up tidal signal. 
                    fig.set_figwidth(7)
                    # key list
                    model_keys = [] # length of 3 
                    linetypes = ['-', '--', '--']
                    for kil, model in enumerate(list_of_data):
                        tt= self.time_sliced
                        # if kil == 2: # ukc4 timeshift
                        #     tt = self.time_sliced + np.timedelta64(20, 'm')
                        # if kil == 1: # primea timeshift
                        #     tt = self.time_sliced + np.timedelta64(20, 'm')
                        # print(model)
                        mk = [j for j in model[i].keys()][0]# model is the dataset itself for both heysham and liberpool. 
                        # print(mk)
                        model_keys.append(mk) # should be like PRIMEA Model key etc., 
                        # 
                        surface_height_plot = model[i][mk]
                        lw = [1.5,1,1]
                        col = ['grey', 'blue', 'red']
                        # if kil == 0:
                        #     surface_height_plot = surface_height_plot - 0.5
                        mod_key_new = ['Tide Gauge', r'UKC4$_{\mathrm{PRIMEA}}$', r'UKC4$_{\mathrm{ao}}$']
                        ax.plot(tt,surface_height_plot, label = mod_key_new[kil], linewidth = lw[kil], linestyle = linetypes[kil], color = col[kil]) 
                        
                        # import pdb; pdb.set_trace()    
                    spring_neap = 2*24*14
                    start_at = 50#spring_neap*3
                    day = 2*24
                    fourday = 2*24*4
                    week = 2 * 24 * 7
                    

                    ax.set_xlim([self.time_sliced[start_at], self.time_sliced[start_at + fourday]])
                    tide_gauge_name = [j for j in self.tide_loc_dict.keys()][i]
                    # import pdb; pdb.set_trace()
                    ax.legend(loc = 'lower right', frameon=False)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set tick locations
                    fig.autofmt_xdate()
                    ax.set_ylabel('SSH (m)')
                    # ax.set_xlabel('Time')
                    plt.tight_layout()  
                    # plt.title(self.dataset_name)
                    fig.savefig(os.path.join(fig_path,'timeseries_' + variable_name + '_' + tide_gauge_name + '_week.png'), dpi = 300)
                    plt.close(fig)
                    
            # Run timeseries for both datasest aga
            # import pdb; pdb.set_trace()
            # new_time = self.time_sliced+ np.timedelta64(40, 'm')
            prix = []
            from scipy.interpolate import interp1d

            data = pd.DataFrame({'Timestamp': self.time_sliced, 'Shifted_Time': self.time_sliced + pd.Timedelta(minutes=6)})
            for kio, item in enumerate(self.primx):
                data['Values'] = item['PRIMEA Model']
                full_time_range = pd.date_range(start=data['Timestamp'].min(), end=data['Shifted_Time'].max(), freq='30T')
                shifted_time_range = pd.date_range(start=data['Shifted_Time'].min(), end=data['Shifted_Time'].max(), freq='30T')
                combined_times = full_time_range.union(shifted_time_range).sort_values()
                combined_numeric = combined_times.view(int) / 10**9
                # import pdb;pdb.set_trace()
                interpolate_func = interp1d(combined_numeric, np.interp(combined_numeric, data['Shifted_Time'].view(int) / 10**9, data['Values']), bounds_error=False, fill_value="extrapolate")
                # Now interpolate back to the original half-hour marks
                original_half_hour_marks = pd.date_range(start=data['Timestamp'].min(), end=data['Timestamp'].max(), freq='30T')
                original_half_hour_numeric = original_half_hour_marks.view(int) / 10**9
                interpolated_values = interpolate_func(original_half_hour_numeric)
                result_data = pd.DataFrame({'Timestamp': original_half_hour_marks, 'Interpolated_Value': interpolated_values})
                
                original_frequency_data = result_data.iloc[::2].reset_index(drop=True)
                prix.append({'PRIMEA Model':np.array(original_frequency_data['Interpolated_Value'])})
            data = pd.DataFrame({'Timestamp': self.time_sliced, 'Shifted_Time': self.time_sliced + pd.Timedelta(minutes=6)})
            ukcy = [] 
            for kio, item in enumerate(self.ukc4y):
                # import pdb;pdb.set_trace()
                data['Values'] = item['UKC4 Model']
                full_time_range = pd.date_range(start=data['Timestamp'].min(), end=data['Shifted_Time'].max(), freq='30T')
                shifted_time_range = pd.date_range(start=data['Shifted_Time'].min(), end=data['Shifted_Time'].max(), freq='30T')
                combined_times = full_time_range.union(shifted_time_range).sort_values()
                combined_numeric = combined_times.view(int) / 10**9
                # 
                interpolate_func = interp1d(combined_numeric, np.interp(combined_numeric, data['Shifted_Time'].view(int) / 10**9, data['Values']), bounds_error=False, fill_value="extrapolate")
                # Now interpolate back to the original half-hour marks
                original_half_hour_marks = pd.date_range(start=data['Timestamp'].min(), end=data['Timestamp'].max(), freq='30T')
                original_half_hour_numeric = original_half_hour_marks.view(int) / 10**9
                interpolated_values = interpolate_func(original_half_hour_numeric)
                result_data = pd.DataFrame({'Timestamp': original_half_hour_marks, 'Interpolated_Value': interpolated_values})
                
                original_frequency_data = result_data.iloc[::2].reset_index(drop=True)
                ukcy.append({'UKC4 Model':np.array(original_frequency_data['Interpolated_Value'])})

            tidx = []    
            for kio, item in enumerate(self.tidex):
                new_tide_vaue = item['Measured Tide Gauge'] - 0.5
            #     new_tide = item['Measured Tide Gauge'] - 0.5
                tidx.append({'Measured Tide Gauge': new_tide_vaue})
            
            # timeseries_plot([self.tidex, self.primx, self.ukc4y]) # Will make a plot of all together per location. 
            mylist = [tidx, prix, ukcy]
            timeseries_plot(mylist)
            # print(self.ukc4y)
            tide_gauge_name = [j for j in self.tide_loc_dict.keys()]
            for i, tide_gauge in enumerate(tide_gauge_name):
                # import pdb; pdb.set_trace()
                tide_primea_rmse = calculate_rmse(tidx[i]['Measured Tide Gauge'][50:], prix[i]['PRIMEA Model'][50:])
                tide_ukc3_rmse = calculate_rmse(tidx[i]['Measured Tide Gauge'][50:], ukcy[i]['UKC4 Model'][50:])    
                prim_ukc3_rmse = calculate_rmse(prix[i]['PRIMEA Model'][50:], ukcy[i]['UKC4 Model'][50:])  
                with open(os.path.join(data_stats_path, 'RMSE_stats.txt'), "a") as f:
                    # import pdb; pdb.set_trace()
                    table_to_return =  (variable_name, tide_gauge_name[i], 'tide', 'primea', tide_primea_rmse)
                    # print("{:<20} {:<10} {:<20} {:<15} {:<20}".format(*table_to_return))
                    f.write("{:<20} {:<10} {:<20} {:<15} {:<20}".format(*table_to_return))
                    f.write('\n')
                    table_to_return =  (variable_name, tide_gauge_name[i], 'tide', 'ukc4', tide_ukc3_rmse)
                    f.write("{:<20} {:<10} {:<20} {:<15} {:<20}".format(*table_to_return))
                    f.write('\n')
                    table_to_return =  (variable_name, tide_gauge_name[i], 'prim', 'ukc4', prim_ukc3_rmse)
                    f.write("{:<20} {:<10} {:<20} {:<15} {:<20}".format(*table_to_return))
                    f.write('\n')
                
            
        return extract_prims, extract_ukc4s
    
    # def tidal_plots(self, fig_path):
    #     '''
    #     I need to plot the number of tide guage locations, 
    #     which then plots 2 figures, in each figure should be plotted
    #     the tide gauge, ukc4 and primea data values. 
        
    #     This can then be done for salinity ensuring to only plot the tide 
    #     gauge if the data exists. 
    #     '''
        
        
    #     prim_dict = self.data_dict['prim']
    #     ukc4_dict = self.data_dict['ukc4']
        
    #     fig, ax = plt.subplots(len(self.tide_save), 1)
    #     for i, variable_name in enumerate(self.common_keys):
    #         for tide_gauge in self.tide_save:
    #             x = tide_gauge[0] # Now the location has been determined you can apply elsewhere. 
    #             y = tide_gauge[1]
                
    #             ukc4_data = ukc4_dict[variable_name]
    #             prim_data = prim_dict[variable_name]
    #             primx = prim_data[50:,x,y].data.flatten() # at the testing points [4:,40,20] it is almost identical. 
    #             ukc4y = ukc4_data[50:,x,y].data.flatten()
    #             ax[i].plot(self.time[50:], primx)
    #             ax[i].plot(self.time[50:], ukc4y)
                
                
    #             # plt.savefig(os.path.join(fig_path, 'tide_gauge_validation_' + variable_name +'.png'), dpi = 150)
    #             plt.close
        
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
            # plt.title(self.dataset_name)
            plt.savefig(fig_path + '/timestep_'+ str(minmax) +'_transects_along_estuaries.png', dpi = 300)
            plt.close()
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
                cmap = cmocean.cm.balance
                pcm = ax.pcolor(self.lon, self.lat, height_difference, cmap = cmap)
                
                def sanity_plot(var, saveas):
                    fig3, ax3 = plt.subplots() # this is the figure for correlation plots
                    fig3.set_figheight(7) # plotting up tidal signal. 
                    fig3.set_figwidth(5)
                    pcm = ax3.pcolor(self.lon, self.lat, var)
                    cbar = plt.colorbar(pcm)
                    cbar.set_label(j + ' ' + saveas)
                    # plt.title(self.dataset_name)
                    plt.savefig(fig_path + '/SanityCheck/' + saveas + '_' + j + '_' + i + '.png', dpi = 300)
                    plt.close()
                    
                
                    
                    
                pcm.set_array(height_difference)
                if i == 'surface_height':
                    if j == 'max':
                        pcm.set_clim(-2, 2)
                    else:
                        pcm.set_clim(-7, 7)
                        
                if i == 'surface_salinity':
                    pcm.set_clim(-20, 20)
                #pcm.set_clim(-1, 1)
                cbar = plt.colorbar(pcm)
                #cbar.set_ticks(np.linspace(-1,1,11))
                cbar.set_label(j + ' Diff between $\mathrm{UKC4}_{\mathrm{PRIMEA}}$ and $\mathrm{UKC4}_{\mathrm{ao}}$ (m)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                # plt.title(self.dataset_name)
                plt.tight_layout()
                
                plt.savefig(fig_path + '/diff_' + j + '_'+ i +'_analysis.png', dpi = 300)
                plt.close()
                # sanity_plot(heightprim, 'prim_height')
                # sanity_plot(heightukc4, 'ukc4_height')
                # plt.close()
            
        

            
        return heightprim, heightukc4, height_difference
        
        
        
#%%        
        
if __name__ == '__main__':
    
    for fn in [
                #'bathymetry_testing',
                'oa_nawind_Orig_m0.030_Forcing',
                'oa_nawind_Orig_m0.035_Forcing',
                'oa_nawind_Orig_m0.045_Forcing',
               # 'PRIMEA_riv_nawind_oa_1l_flipped',
               # 'PRIMEA_riv_nawind_oa_1l_original',
               # 'PRIMEA_riv_yawind_oa_1l_flipped',
               # 'PRIMEA_riv_yawind_oa_1l_original',
               # 'kent_1.30_base_from_5.Final',
                ]:
    
        from o_func import DataChoice, DirGen
        import glob
        # main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
        # fn = 'kent_1.0.0_UM_wind' # 
        main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'8.model_calibration')
        # fn = 'kent_1.30_base_from_5.Final' # 
        # fn = 'PRIMEA_riv_nawind_oa_1l_flipped'
        
        make_paths = DirGen(main_path)
        ### Finishing directory paths
        
        dc = DataChoice(os.path.join(main_path,'models'))
        #fn = os.path.split(dc.dir_select()[0])[-1]
        #fn = 'kent_1.3.7_testing_4_days_UM_run' # bad model for testing, had issues. 
        
    
        sub_path, fig_path, data_stats_path = make_paths.dir_outputs(fn)
        lp = os.path.join(sub_path, 'kent_regrid.nc') # this was originally globbed due to 2 nc files 
        sts = Stats(lp, fn)
        load = sts.load_raw()
        Stats.print_dict_keys(load[1]) # prints out the dictionaries of data being used. 
        extract_prims, extract_ukc4s = sts.linear_regression(fig_path, data_stats_path)
        tide_gauge, ind = sts.load_tide_gauge()
        transect = sts.transect(fig_path)
        prim, ukc4, height_diff = sts.max_compare(fig_path)
        # tp = sts.tidal_plots(fig_path)
        
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
# [plt.close() for i in range(400)]
