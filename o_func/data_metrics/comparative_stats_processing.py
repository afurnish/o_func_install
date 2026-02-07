#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This script will generate statitistics and also process outputs, statistics go live in the stats folder. 
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
import cmasher as cmr
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import time as measuretime
from sklearn.neighbors import BallTree
start_measure= measuretime.time()
import pickle


from o_func import opsys; start_path = opsys()
from o_func.utilities.near_neigh import near_neigh
from o_func.utilities.distance import Dist
from o_func.utilities.gauges import tide_gauge_loc
   

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#%% Shifted time
shifted_time_val = 0

#%% 
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

def calculate_rolling_max(data_array, window_size=3):
    """
    Calculate the rolling maximum for a given xarray DataArray over a specified window size.
    
    Parameters:
    - data_array: xarray.DataArray, the data array to process.
    - window_size: int, the size of the rolling window in time steps.
    
    Returns:
    - xarray.DataArray: The rolling maximum values with the same dimensions as the input.
    """
    # Apply rolling window and calculate the max
    # 'min_periods=1' ensures that we get values even if the window is not fully populated (e.g., at edges)
    rolling_max = data_array.rolling(time_primea=window_size, center=True, min_periods=1).max()
    
    return rolling_max

def calculate_rolling_min(data_array, window_size=3):
    
    rolling_min = data_array.rolling(time_primea=window_size, center=True, min_periods=1).min()
    
    return rolling_min

# --- move to top-level (no indentation) ---
def process_point(indexed_uv, sh_prim):
    
    # Within this function, apply some kind of masked array perhaps, and use the nearest 
    # available working storm surge prediction. This will require analysis of all storm surges. 
    import ttide as tt
    index, (y, x) = indexed_uv
    try:
        sh_data = sh_prim[:, y, x].values
        tt_time = sh_prim.time_primea.values
        tt_time_py = pd.to_datetime(tt_time).to_pydatetime()
        base_dnum = pd.to_datetime(tt_time[0]).to_pydatetime()

        lon = sh_prim[:,y,x].nav_lon.values
        lat = sh_prim[:,y,x].nav_lat.values
        tide_analysis = tt.t_tide(sh_data, dt=1, stime=base_dnum, lat=53.5, out_style=None)

        predicted = tt.t_predic(
            t_time=tt_time_py,
            tidecon=tide_analysis['tidecon'],
            names=tide_analysis['nameu'],
            freq=tide_analysis['fu'],
            lat=53.5
        )

        residual = sh_data - predicted
        
        def MSL_adjusted(sh_data):
            msl = np.nanmean(sh_data)  # Mean Sea Level
            sh_demeaned = sh_data - msl
            
            
            tide_analysis = tt.t_tide(sh_demeaned, dt=1, stime=base_dnum, lat=53.5, out_style=None)
            
            sh_predicted_demeaned = tt.t_predic(
                t_time=tt_time_py,
                tidecon=tide_analysis['tidecon'],
                names=tide_analysis['nameu'],
                freq=tide_analysis['fu'],
                lat=53.5
            )
            
            
            predicted_MSL_adjusted = sh_predicted_demeaned + msl

            return predicted_MSL_adjusted

        predicted_MSL_adjusted = MSL_adjusted(sh_data)
        predicted_MSL_adjusted_mean = np.nanmean(predicted_MSL_adjusted)
        if predicted_MSL_adjusted_mean > 0.3 or predicted_MSL_adjusted_mean < -0.3:
            meanSL_mask = False
        else:
            meanSL_mask = True
            
        # Here apply the storm surge correction to adjusted t_tide analysis. 
        
        return {
            'index': index,
            'y': y,
            'x': x,
            'lon':lon,
            'lat':lat,
            'max_residual': float(np.max(residual)),
            'mae': float(np.mean(np.abs(residual))),
            'bias': float(np.mean(residual)),
            'corr': float(np.corrcoef(sh_data, predicted)[0, 1]),
            'predicted': predicted.tolist(),
            'tidecon': tide_analysis['tidecon'].tolist(),
            'names': tide_analysis['nameu'].tolist(),
            'freq': tide_analysis['fu'].tolist(),
            'predicted_MSL_adjusted': predicted_MSL_adjusted.tolist(),
            'predicted_MSL_adjusted_mean':predicted_MSL_adjusted_mean,
            'meanSL_mask':meanSL_mask,
        }
    
    except Exception as e:
        print(f"Point {index} ({y},{x}) failed: {e}")
        return None

# Load the CSV
df_storms = pd.read_csv(Path(start_path) / 'Original_Data/wind_data/storm_list20132014.csv')

# Function to extract the central date of a storm event
def extract_central_date(date_str):
    import re
    from datetime import datetime
    # Match date ranges, e.g., '5th - 6th December'
    match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s*-\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)', date_str)
    if match:
        day1 = int(match.group(1))
        day2 = int(match.group(2))
        month = match.group(3)
        central_day = (day1 + day2) // 2
        year = 2013 if month.lower() in ['december', 'november'] else 2014
        return datetime.strptime(f"{central_day} {month} {year}", "%d %B %Y").date()
    else:
        # Match single day entries, e.g., '3rd January'
        match_single = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)', date_str)
        if match_single:
            day = int(match_single.group(1))
            month = match_single.group(2)
            year = 2013 if month.lower() in ['december', 'november'] else 2014
            return datetime.strptime(f"{day} {month} {year}", "%d %B %Y").date()
        return None

# Apply the function
df_storms['CentralDate'] = df_storms['DateofStorm'].apply(extract_central_date)

# Reorganize the dataframe
df_storms_cleaned = df_storms[['CentralDate', 'StormDesignation', 'LetterDesignation']].set_index('CentralDate')

# Display or export
print(df_storms_cleaned)

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
            df (pd.DataFrame): Input DataFrame with DatetimeIndex.12 ;

        Returns:
            pd.DataFrame: DataFrame with hourly means and timestamps at the middle of each hour.
        """
        hourly_means = df.resample('h').mean()
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
                    prim_datasets[primkey] = primvalue.resample(time_primea='1h').mean(skipna = True)
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
        self.bathymetry = self.raw_data['prim_bathymetry'][0,:,:] * -1 # Must be made negative as values are positive.       
        self.data_dict = data_dict
        return self.raw_data, data_dict, matching_times
    ####            0                1          2
    
    def load_tide_gauge(self): 
        # Original datasets
        #'Heysham'  :{'x':-2.9594670, 'y':54.0328370},
        #'Liverpool':{'x':-3.0741720, 'y':53.4634140}
        self.tide_loc_dict = {
                                # 'Heysham'  :{'x':-2.9594670, 'y':54.0328370},
                                # 'Heysham'  :{'x':-2.9574780, 'y':54.0333366},
                                # 'Liverpool':{'x':-3.1554490, 'y':53.4930250}, # Looks good but too deep on liverpool 
                                # 'Liverpool':{'x':-3.1391550, 'y':53.4622030}, 
                                # 'Liverpool':{'x':-3.1988680, 'y':53.4797420}, 
                                # 'Liverpool':{'x':-3.0741720, 'y':53.4634140},
                                # 'Ribble':{'x': -3.0565639, 'y': 53.715130},
                                # 'Liverpool'  :{'x':-2.9594670, 'y':54.0328370},
                                # test case north west of docks by 2km
                                # 'Heysham'    :{'x':-2.932401, 'y':54.042532},
                                #test case inside the harbour
                                # 'Heysham'    :{'x':-2.922459, 'y':54.032553},
                                
                                # So far this is the best test case, now for liverpool. 
                                # test case south east by 2km 
                                # 'Heysham'    :{'x':-2.935248, 'y':54.020656},
                                
                                #Liverpool test case original
                                # 'Liverpool':{'x':-3.0741720, 'y':53.4634140}
                                #best case study so far. 
                                #test case up the stream by the docks
                                # 'Liverpool':{'x':-3.020363, 'y':53.438447},
                                
                                #best liverpool case for salinity
                                # 'Liverpool'    :{'x':-3.058533, 'y':53.4588182}, # This was the one working for a long old time 
                                'Liverpool'    :{'y':53.47606161335259, 'x':-3.0875790015848263}, # This one may provide better stats! 
                                # liverpool further offshore 
                                # 'Liverpool'    :{'x':-3.076068, 'y':53.462082},
                                
                                # 'Liverpool':{'x':-3.0741720, 'y':53.4634140},
                                #new heysham test case on the ferry line. 
                                'Heysham'    :{'x':-2.946128, 'y':54.022946},
                              }
        
        
        
        
        # self.tide_loc_dict = tide_gauge_loc()
        
        # self.tide_loc_dict = tide_gauge_loc()
        df_tide_loc = pd.DataFrame(self.tide_loc_dict).T.reset_index()
        df_tide_loc = df_tide_loc.drop(df_tide_loc.columns[0], axis=1)
        df_search_points = pd.DataFrame({'x': self.lon.data.ravel(), 'y': self.lat.data.ravel()})
       # print(self.lat.data.shape)
        tide_dict = {}
        for i in self.tide_gauge_data:
            dataset = pd.read_csv(i, index_col=0, parse_dates=True)
            name = os.path.split(i)[-1][5:-4].capitalize()
            tide_dict[name] = dataset.resample('h').mean()
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
                    self.primx.append({'IRENE Model' : prim_data[slicer:,x,y].data.flatten()}) # at the testing points [4:,40,20] it is almost identical. 
                    self.ukc4y.append({'UKC4 Model' : ukc4_data[slicer:,x,y].data.flatten()})
                    if tide_gauge_name != 'Ribble':
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
                    if tide_gauge != 'Ribble':
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
                        
                        fig.text(0.5, 0.01, xname + ' (normalised [' + unit + '])', ha='center', va='bottom', transform=fig.transFigure)
                        fig.text(0.03, 0.5, yname + ' (normalised [' + unit + '])', va='center', rotation='vertical')
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
                
                
                
                #%%
                for i, tide_gauge in enumerate(self.tide_save):
                    #%
                    fig, ax = plt.subplots(dpi = 300) # this is the figure for correlation plots
                    fig.set_figheight(4) # plotting up tidal signal. 
                    fig.set_figwidth(7)
                    # key list
                    model_keys = [] # length of 3 
                    linetypes = ['-', '--', '--']
                    if variable_name != 'surface_height':
                        list_of_data = list_of_data[1:] # remove the tide gauge
                        col = ['red', 'blue']
                        mod_key_new = [r'UKC4$_{\mathrm{PRIMEA}}$', r'UKC4$_{\mathrm{ao}}$']
                    else:
                        mod_key_new = ['Tide Gauge', r'UKC4$_{\mathrm{PRIMEA}}$', r'UKC4$_{\mathrm{ao}}$']
                        col = ['grey', 'red', 'blue']
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
                        lw = [2,1,1]
                        
                        # if kil == 0:
                        #     surface_height_plot = surface_height_plot - 0.5

                        if mod_key_new[kil] ==  r'UKC4$_{\mathrm{PRIMEA}}$':
                            new_name = 'IRENE'
                        elif mod_key_new[kil] == r'UKC4$_{\mathrm{ao}}$':
                            new_name = 'UKC4'
                        else:
                            new_name = mod_key_new[kil]
                        ax.plot(tt,surface_height_plot, label = new_name, linewidth = lw[kil], linestyle = linetypes[kil], color = col[kil]) 
                        high_tide_num = 174 + 13 + 12
                        # ax.scatter(tt[high_tide_num], surface_height_plot[high_tide_num], s = 4)
                        ax.scatter(
                            pd.Timestamp(tt[high_tide_num].item()).to_pydatetime(),  # Convert to Python datetime
                            surface_height_plot.iloc[high_tide_num] if hasattr(surface_height_plot, 'iloc') else surface_height_plot[high_tide_num],  # Dynamic indexing
                            s=4
                        )
                        
                        #%

                        # plot what time of tide the transect comes from
                        # import pdb; pdb.set_trace()    
                    spring_neap = 24*14
                    two_months = 24 * 28
                    start_at = 48 + 48 + 36 + 36#spring_neap*3
                    start_at_two_months = 24 * 30 * 2
                    day = 24
                    fourday = 24*4
                    week = 24 * 7
                    
                    if len(self.time_sliced) < 242:
                        the_starting_point = start_at
                        time_indexed= start_at + 42
                    else:
                        the_starting_point = start_at_two_months
                        time_indexed= start_at_two_months + spring_neap
                        
                    if variable_name == 'surface_height':
                        the_starting_point = 1440 -700
                        time_indexed = the_starting_point + (24*21)
                        
                    ax.set_xlim([self.time_sliced[the_starting_point], self.time_sliced[time_indexed]])
                    tide_gauge_name = [j for j in self.tide_loc_dict.keys()][i]
                    # import pdb; pdb.set_trace()
                    ax.legend(loc = 'upper right', frameon=False)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set tick locations
                    #%%
                    fig.autofmt_xdate()
                    
                    if variable_name == 'surface_height':
                        variable_name_normal = 'Surface Height'
                    elif variable_name == 'salinity':
                        variable_name_normal = 'Salinity'
                    else:
                        variable_name_normal = variable_name
                    ax.set_ylabel( variable_name_normal + ' [' + var_dict[variable_name]['UNITS'] + ']') 
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

            data = pd.DataFrame({'Timestamp': self.time_sliced, 'Shifted_Time': self.time_sliced + pd.Timedelta(minutes=shifted_time_val)})
            for kio, item in enumerate(self.primx):
                # data['Values'] = item['PRIMEA Model']
                # full_time_range = pd.date_range(start=data['Timestamp'].min(), end=data['Shifted_Time'].max(), freq='30min')
                # shifted_time_range = pd.date_range(start=data['Shifted_Time'].min(), end=data['Shifted_Time'].max(), freq='30min')
                # combined_times = full_time_range.union(shifted_time_range).sort_values()
                # combined_numeric = combined_times.view(int) / 10**9
                # # import pdb;pdb.set_trace()
                # interpolate_func = interp1d(combined_numeric, np.interp(combined_numeric, data['Shifted_Time'].view(int) / 10**9, data['Values']), bounds_error=False, fill_value="extrapolate")
                # # Now interpolate back to the original half-hour marks
                # original_half_hour_marks = pd.date_range(start=data['Timestamp'].min(), end=data['Timestamp'].max(), freq='30min')
                # original_half_hour_numeric = original_half_hour_marks.view(int) / 10**9
                # interpolated_values = interpolate_func(original_half_hour_numeric)
                # result_data = pd.DataFrame({'Timestamp': original_half_hour_marks, 'Interpolated_Value': interpolated_values})
                
                # original_frequency_data = result_data.iloc[::2].reset_index(drop=True)
                # prix.append({'PRIMEA Model':np.array(original_frequency_data['Interpolated_Value'])})
                data['Values'] = item['IRENE Model']

                # Create the full and shifted time ranges
                full_time_range = pd.date_range(start=data['Timestamp'].min(), end=data['Shifted_Time'].max(), freq='30min')
                shifted_time_range = pd.date_range(start=data['Shifted_Time'].min(), end=data['Shifted_Time'].max(), freq='30min')
                
                # Combine the time ranges and sort them
                combined_times = full_time_range.union(shifted_time_range).sort_values()
                
                # Convert combined_times to Unix timestamps in seconds
                combined_numeric = combined_times.astype(np.int64) // 10**9
                
                # Convert Shifted_Time to Unix timestamps in seconds
                shifted_time_seconds = data['Shifted_Time'].astype(np.int64) // 10**9
                
                # Create the interpolation function
                interpolate_func = interp1d(
                    combined_numeric,
                    np.interp(combined_numeric, shifted_time_seconds, data['Values']),
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                # Interpolate back to the original half-hour marks
                original_half_hour_marks = pd.date_range(start=data['Timestamp'].min(), end=data['Timestamp'].max(), freq='30min')
                original_half_hour_numeric = original_half_hour_marks.astype(np.int64) // 10**9
                
                # Get the interpolated values
                interpolated_values = interpolate_func(original_half_hour_numeric)
                
                # Create the result DataFrame
                result_data = pd.DataFrame({'Timestamp': original_half_hour_marks, 'Interpolated_Value': interpolated_values})
                
                # Extract original frequency data
                original_frequency_data = result_data.iloc[::2].reset_index(drop=True)
                prix.append({'IRENE Model': np.array(original_frequency_data['Interpolated_Value'])})
            data = pd.DataFrame({'Timestamp': self.time_sliced, 'Shifted_Time': self.time_sliced + pd.Timedelta(minutes=shifted_time_val)})
            ukcy = [] 
            for kio, item in enumerate(self.ukc4y):
                # # import pdb;pdb.set_trace()
                # data['Values'] = item['UKC4 Model']
                # full_time_range = pd.date_range(start=data['Timestamp'].min(), end=data['Shifted_Time'].max(), freq='30min')
                # shifted_time_range = pd.date_range(start=data['Shifted_Time'].min(), end=data['Shifted_Time'].max(), freq='30min')
                # combined_times = full_time_range.union(shifted_time_range).sort_values()
                # combined_numeric = combined_times.view(int) / 10**9
                # # 
                # interpolate_func = interp1d(combined_numeric, np.interp(combined_numeric, data['Shifted_Time'].view(int) / 10**9, data['Values']), bounds_error=False, fill_value="extrapolate")
                # # Now interpolate back to the original half-hour marks
                # original_half_hour_marks = pd.date_range(start=data['Timestamp'].min(), end=data['Timestamp'].max(), freq='30min')
                # original_half_hour_numeric = original_half_hour_marks.view(int) / 10**9
                # interpolated_values = interpolate_func(original_half_hour_numeric)
                # result_data = pd.DataFrame({'Timestamp': original_half_hour_marks, 'Interpolated_Value': interpolated_values})
                
                # original_frequency_data = result_data.iloc[::2].reset_index(drop=True)
                # ukcy.append({'UKC4 Model':np.array(original_frequency_data['Interpolated_Value'])})
                # Assuming 'data' and 'item' are already defined
                data['Values'] = item['UKC4 Model']
                
                # Create the full and shifted time ranges
                full_time_range = pd.date_range(start=data['Timestamp'].min(), end=data['Shifted_Time'].max(), freq='30min')
                shifted_time_range = pd.date_range(start=data['Shifted_Time'].min(), end=data['Shifted_Time'].max(), freq='30min')
                
                # Combine the time ranges and sort them
                combined_times = full_time_range.union(shifted_time_range).sort_values()
                
                # Convert combined_times to Unix timestamps in seconds
                combined_numeric = combined_times.astype(np.int64) // 10**9
                
                # Convert Shifted_Time to Unix timestamps in seconds
                shifted_time_seconds = data['Shifted_Time'].astype(np.int64) // 10**9
                
                # Create the interpolation function
                interpolate_func = interp1d(
                    combined_numeric,
                    np.interp(combined_numeric, shifted_time_seconds, data['Values']),
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                # Interpolate back to the original half-hour marks
                original_half_hour_marks = pd.date_range(start=data['Timestamp'].min(), end=data['Timestamp'].max(), freq='30min')
                original_half_hour_numeric = original_half_hour_marks.astype(np.int64) // 10**9
                
                # Get the interpolated values
                interpolated_values = interpolate_func(original_half_hour_numeric)
                
                # Create the result DataFrame
                result_data = pd.DataFrame({'Timestamp': original_half_hour_marks, 'Interpolated_Value': interpolated_values})
                
                # Extract original frequency data
                original_frequency_data = result_data.iloc[::2].reset_index(drop=True)
                ukcy.append({'UKC4 Model': np.array(original_frequency_data['Interpolated_Value'])})
            tidx = []    
            for kio, item in enumerate(self.tidex):
                new_tide_vaue = item['Measured Tide Gauge'] - 0.5
            #     new_tide = item['Measured Tide Gauge'] - 0.5
                tidx.append({'Measured Tide Gauge': new_tide_vaue})
            
            # timeseries_plot([self.tidex, self.primx, self.ukc4y]) # Will make a plot of all together per location. 
            mylist = [tidx, prix, ukcy]
            timeseries_plot(mylist)
            
            def print_dict_tree(d, indent=0):
                """Recursively print dictionary keys as a tree structure."""
                for key, value in d.items():
                    print('  ' * indent + str(key))
                    if isinstance(value, dict):
                        print_dict_tree(value, indent + 1)
                        
            def tidal_analysis(mylist):
                import ttide as tt
                data = mylist
                
                tide_storage = {}
                
                def filter_tide_by_snr(tide_analysis, snr_limit):
                    """
                    Filter the tide_analysis output to include only constituents with SNR >= snr_limit.
                    
                    Notes on SNR filtering for tidal analysis:

                    Signal-to-Noise Ratio (SNR)
                    – SNR is a measure of how strong a fitted tidal constituent is compared to the uncertainty in its amplitude estimate.
                    – In t_tide, SNR ≈ (amplitude ÷ amplitude_error)².
                    – High SNR (≫ 1) means the constituent is well resolved; low SNR (< 1) means it is dominated by noise.
                    
                    Why filter by SNR?
                    – Low-SNR constituents have poorly constrained amplitude and phase, and large confidence intervals.
                    – Including them in RMSE or bias calculations can skew skill metrics.
                    – Common practice: only keep constituents with SNR ≥ 4 for reconstruction and error statistics.
                    
                    filter_tide_by_snr function
                    – Takes the output dict from ttide.t_tide and a threshold (snr_limit).
                    – Returns a new dict containing only constituents whose snr ≥ snr_limit.
                    – Copies over metadata (nobs, ngood, dt, lat, stime, nodal correction info).
                    
                    Usage example:
                    filtered = filter_tide_by_snr(tide_analysis, snr_limit=4.0)
                    # feed ‘filtered’ into ttide.recon and compute RMSE/bias using only high-SNR tides
                    
                    Key parameters:
                    – dt: sampling interval in hours
                    – stime: start time in MATLAB datenum (to align epochs)
                    – lat: station latitude for nodal corrections
                    
                    Next steps after filtering:
                    – Reconstruct the tide time series with ttide.recon(filtered)
                    – Compare reconstructed vs observed heights to compute amplitude RMSE, phase bias, etc.
                    – Report skill metrics using only the robust (high-SNR) constituents.
                                    
                    Parameters:
                        tide_analysis (dict): The output dict from tt.tide.t_tide.
                        snr_limit (float): Minimum SNR threshold.
                
                    Returns:
                        dict: A new tide_analysis-like dict with only high-SNR constituents.
                    """
                    # Extract arrays
                    snr = tide_analysis['snr']
                    mask = snr >= snr_limit
                
                    # Filter constituent names, frequencies, and tidecon parameters
                    filtered = {
                        'nameu': tide_analysis['nameu'][mask],
                        'fu':    tide_analysis['fu'][mask],
                        'tidecon': tide_analysis['tidecon'][mask, :],
                        'snr':   tide_analysis['snr'][mask],
                        # copy metadata
                        'nobs': tide_analysis['nobs'],
                        'ngood': tide_analysis['ngood'],
                        'dt': tide_analysis['dt'],
                        'lat': tide_analysis.get('lat'),
                        'stime': tide_analysis.get('stime'),
                        'ltype': tide_analysis.get('ltype'),
                        'nodcor': tide_analysis.get('nodcor')
                    }
                    return filtered
            
                def filter_tide_by_names(tide_analysis, keep_names):
                    """
                    Filter the tide_analysis output to include only specified tidal constituents.
                
                    Parameters:
                        tide_analysis (dict): The output dict from ttide.t_tide.
                        keep_names (list of bytes or str): Constituent names to keep, e.g. [b'N2  ', b'M2  ', ...].
                    
                    Returns:
                        dict: A new tide_analysis-like dict with only the specified constituents.
                    """
                    # Normalize keep_names to bytes
                    keep_bytes = [n if isinstance(n, (bytes,)) else n.encode() for n in keep_names]
                    
                    # Extract arrays
                    names = tide_analysis['nameu']
                    mask = np.array([n in keep_bytes for n in names])
                    
                    # Filter constituent names, frequencies, and tidecon parameters
                    filtered = {
                        'nameu': tide_analysis['nameu'][mask],
                        'fu':    tide_analysis['fu'][mask],
                        'tidecon': tide_analysis['tidecon'][mask, :],
                        'snr':   tide_analysis['snr'][mask],
                        # copy metadata
                        'nobs': tide_analysis['nobs'],
                        'ngood': tide_analysis['ngood'],
                        'dt': tide_analysis['dt'],
                        'lat': tide_analysis.get('lat'),
                        'stime': tide_analysis.get('stime'),
                        'ltype': tide_analysis.get('ltype'),
                        'nodcor': tide_analysis.get('nodcor')
                    }
                    return filtered
                
                ## Tide data in order of tide gauge, primea, ukc4
                for i, tide_gauge in enumerate(self.tide_save):
                    # This loop should set the tide gauges. 
                    tide_gauge_name = [j for j in self.tide_loc_dict.keys()][i]
                    tide_storage[tide_gauge_name] = {}
                    print(tide_gauge)
                    # fig, ax = plt.subplots() # this is the figure for correlation plots
                    # fig.set_figheight(4) # plotting up tidal signal. 
                    # fig.set_figwidth(7)
                    # key list
                    model_keys = [] # length of 3 
                    linetypes = ['-', '--', '--']
                    
                    mod_key_new = ['Tide Gauge', r'PRIMEA', r'UKC4']
                    col = ['grey', 'blue', 'red']
                    for kil, model in enumerate(data):
                        # This loop should flick through tide guage, primea and ukc4. 
                        mod_key = mod_key_new[kil]
                        print(mod_key)
                        tide_storage[tide_gauge_name][mod_key] = {}
                        
                        tt_time= self.time_sliced
                        tt_time = tt_time
                        first_ts = pd.to_datetime(tt_time.values[0]).to_pydatetime()
                        base_dnum = first_ts.toordinal()       
                        if mod_key == 'UKC4':
                            t0_offsets = base_dnum 
                        else:
                            t0_offsets = base_dnum + (0.5/24)
                        # This is the 
                        mk = [j for j in model[i].keys()][0]# model is the dataset itself for both heysham and liberpool. 
                        model_keys.append(mk) # should be like PRIMEA Model key etc., 
                        surface_height_plot = model[i][mk]
                        tide_storage[tide_gauge_name][mod_key]['Surface Height'] = surface_height_plot
                        
                        from ttide.t_getconsts import t_getconsts
                        ctime = np.array([])  # Empty array to skip time-based filtering
                        const, sat, shallow = t_getconsts(ctime)
                        all_constituents = const['name']

                        # Set dt to 1 to be 1 hour. That seems to work. 
                        tide_analysis = tt.t_tide(
                            np.array(surface_height_plot), 
                            dt = 1, stime = t0_offsets, lat = 53.5
                        )
                        
                        # tide_analysis = filter_tide_by_snr(tide_analysis, snr_limit=20.0)
                        # print(tide_analysis)
                        # These have been determined from first set, SNR of 20. 
                        keep_list = [b'N2  ', b'M2  ', b'S2  ', b'MN4 ', b'M4  ', b'MS4 ', b'2MN6',
                                         b'M6  ', b'2MS6', b'M8  ']
                        tide_analysis = filter_tide_by_names(tide_analysis, keep_list)
                        
                        amplitude = tide_analysis['tidecon'][:, 0]
                        tide_storage[tide_gauge_name][mod_key]['amp'] = amplitude
                        phase = tide_analysis['tidecon'][:, 2]
                        tide_storage[tide_gauge_name][mod_key]['pha'] = phase
                        names = tide_analysis['nameu'].astype(str)
                        tide_storage[tide_gauge_name][mod_key]['con_names'] = names

                
                def plot_tidal_analysis(tide_storage):
                    RMSE_path = Path(data_stats_path)/Path('RMSE_stats.txt')
                    output_dir = RMSE_path.parent
                    with RMSE_path.open('a') as f:
                        f.write('\n--------------------Amp&Phase({gauge})--------------------\n')
                    def wrap_phase_linear(observed, model):
                        """Wrap model phase to ensure values are close to observed phase while keeping them in [0, 360)."""
                        wrapped_model = []
                        for o, m in zip(observed, model):
                            diff = m - o
                            if diff > 180:
                                m -= 360
                            elif diff < -180:
                                m += 360
                            wrapped_model.append(m)
                        return np.array(wrapped_model)
                    #%%
                    # For the polar plots

                    
                    for gauge in ['Liverpool', 'Heysham']:
                        polar_fig, polar_ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))
                        scatter_fig, scatter_ax = plt.subplots(2, 2, figsize=(12, 12))
                        amp_data = {}
                        phase_diff_data = {}
                        observed_amp_data = {}
                        for j, model in enumerate(['PRIMEA', 'UKC4']):
                            # Extract data for each gauge and model
                            observed_amp = tide_storage[gauge]['Tide Gauge']['amp']
                            observed_pha = tide_storage[gauge]['Tide Gauge']['pha']
                            model_amp = tide_storage[gauge][model]['amp']
                            model_pha = tide_storage[gauge][model]['pha']
                            constituents = tide_storage[gauge]['Tide Gauge']['con_names']
                            
                            # Convert to radians
                            obs_rad = np.deg2rad(observed_pha)
                            mod_rad = np.deg2rad(model_pha)
                            
                            x_obs = np.sin(obs_rad)
                            x_mod = np.sin(mod_rad)



                            # wrapped_model_pha = wrap_phase_linear(observed_pha, model_pha)
                            x = np.arange(len(constituents))  # x-axis positions
                
                
                            diff = (model_pha - observed_pha + 180) % 360 - 180
                            model_adj = (observed_pha + diff)
                            
                            model_mod = model_adj % 360

                            mask2 = model_mod != model_adj
                            
                            affected = np.where(mask2)[0]
                            obs_adjusted = observed_pha.copy()
                            obs_adjusted[mask2] = abs(obs_adjusted[mask2] - 360)


                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                            # Amplitude scatter plot with y=x line
                            if model == 'UKC4':
                                model_name = 'UKC4'
                            elif model == 'PRIMEA':
                                model_name = 'IRENE'
                            else:
                                model_name = model
                            
                            ax1.scatter(observed_amp, model_amp, label=f'{model_name} vs Observed')
                            ax1.plot([min(observed_amp), max(observed_amp)], [min(observed_amp), max(observed_amp)], color='gray', linestyle='--', label='y = x')
                            for i, name in enumerate(constituents):
                                ax1.text(observed_amp[i], model_amp[i], name, fontsize=8, ha='right')
                            ax1.set_xlabel('Observed Amplitude [m]')
                            if model == 'UKC4':
                                mod_name = 'UKC4'
                            elif model == 'PRIMEA':
                                mod_name = 'IRENE'
                            ax1.set_ylabel(f'{mod_name} Amplitude [m]')
                            # ax1.set_title(f'{gauge} - Amplitude Comparison')
                            ax1.set_aspect('equal', 'box')
                            ax2.set_title('A')

                            ax1.legend()
                
                            # Phase scatter plot with y=x line
                            
                            ax2.scatter(observed_pha, model_pha, label=f'{mod_name} vs Observed')
                            ax2.plot([min(observed_pha), max(observed_pha)], [min(observed_pha), max(observed_pha)], color='gray', linestyle='--', label='y = x')
                            for i, name in enumerate(constituents):
                                ax2.text(observed_pha[i], model_pha[i], name, fontsize=8, ha='right')
                            ax2.set_xlabel('Observed Phase (°)')
                            ax2.set_ylabel(f'{mod_name} Phase (°)')
                            ax2.set_title('B')
                            # **Ensure 0–360° axes:**
                            ax2.set_xlim(0, 400)
                            ax2.set_ylim(0, 400)
                                
                            ax2.legend()
                            ax2.set_aspect('equal', 'box')
                
                            plt.tight_layout()
                
                            fig.savefig(os.path.join(fig_path,'tidal_const_analysis_'+ gauge+ '_' + model +'_vs_observed.png'), dpi = 300)
                            plt.close()
                            
                            #%
                
                            # Amplitude scatter plot with y=x line
                            
                            if model == 'UKC4':
                                scatter_ax[0,0].scatter(observed_amp, model_amp, label=f'{model_name} vs Observed')
                                scatter_ax[0,0].plot([min(observed_amp), max(observed_amp)], [min(observed_amp), max(observed_amp)], color='gray', linestyle='--', label='y = x')
                                for i, name in enumerate(constituents):
                                    scatter_ax[0,0].text(observed_amp[i], model_amp[i], name, fontsize=8, ha='right')
                                scatter_ax[0,0].set_xlabel('Observed Amplitude [m]')
                                scatter_ax[0,0].set_ylabel(f'Amplitude [m]')
                                scatter_ax[0,0].set_title('A (UKC4)')
                                # ax1.set_title(f'{gauge} - Amplitude Comparison')
                                # scatter_ax[0,0].set_aspect('equal', 'box')
                                
                                scatter_ax[1,0].scatter(observed_pha, model_pha, label=f'{model_name} vs Observed')
                                scatter_ax[1,0].plot([min(observed_pha), max(observed_pha)], [min(observed_pha), max(observed_pha)], color='gray', linestyle='--', label='y = x')
                                for i, name in enumerate(constituents):
                                    scatter_ax[1,0].text(observed_pha[i], model_pha[i], name, fontsize=8, ha='right')
                                scatter_ax[1,0].set_xlabel('Observed Phase [°]')
                                scatter_ax[1,0].set_ylabel(f'Phase [°]')
                                # ax2.set_title(f'{gauge} - Phase Comparison')
                                # **Ensure 0–360° axes:**
                                scatter_ax[1,0].set_xlim(0, 400)
                                scatter_ax[1,0].set_ylim(0, 400)
                
                                # scatter_ax[1].set_aspect('equal', 'box')
                            if model == 'PRIMEA':
                                scatter_ax[0,1].scatter(observed_amp, model_amp, label=f'{model_name} vs Observed')
                                scatter_ax[0,1].plot([min(observed_amp), max(observed_amp)], [min(observed_amp), max(observed_amp)], color='gray', linestyle='--', label='y = x')
                                for i, name in enumerate(constituents):
                                    scatter_ax[0,1].text(observed_amp[i], model_amp[i], name, fontsize=8, ha='right')
                                scatter_ax[0,1].set_xlabel('Observed Amplitude [m]')
                                # scatter_ax[0,1].set_ylabel(f'Amplitude (m)')
                                scatter_ax[0,1].set_title('B (IRENE)')
                                scatter_ax[1,1].set_title('D (IRENE)')
                                scatter_ax[1,0].set_title('C (UKC4)')
                                # ax1.set_title(f'{gauge} - Amplitude Comparison')
                                # scatter_ax[1,0].set_aspect('equal', 'box')
                                
                                scatter_ax[1,1].scatter(observed_pha, model_pha, label=f'{model_name} vs Observed')
                                scatter_ax[1,1].plot([min(observed_pha), max(observed_pha)], [min(observed_pha), max(observed_pha)], color='gray', linestyle='--', label='y = x')
                                for i, name in enumerate(constituents):
                                   scatter_ax[1,1].text(observed_pha[i], model_pha[i], name, fontsize=8, ha='right')
                                scatter_ax[1,1].set_xlabel('Observed Phase [°]')
                                # scatter_ax[1,1].set_ylabel(f'Phase (degrees)')
                                # ax2.set_title(f'{gauge} - Phase Comparison')
                                # **Ensure 0–360° axes:**
                                scatter_ax[1,1].set_xlim(0, 400)
                                scatter_ax[1,1].set_ylim(0, 400)
                
                                # scatter_ax[1,1].set_aspect('equal', 'box')
                            
                            
                
                            # Phase scatter plot with y=x line
                            
                            
                
                
                            # fig.savefig(os.path.join(fig_path,'tidal_const_analysis_'+ gauge+ '_' + model +'_vs_observed.png'), dpi = 300)
                            # plt.close()
                            #%% 
                            
                            # Convert degrees to radians for plotting
                            observed_phase_rad = np.deg2rad(observed_pha)
                            modelled_phase_rad = np.deg2rad(model_pha)
                            
                            # Calculate phase differences (in degrees)
                            phase_diff_deg = np.rad2deg(np.angle(np.exp(1j * (observed_phase_rad - modelled_phase_rad))))
                            
                            # Normalize phase differences for color mapping
                            norm = mcolors.Normalize(vmin=0, vmax=180)
                            cmap = cm.viridis
                            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                            sm.set_array([])
                            #%
                            # Create polar plot
                            # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
                            polar_ax[j].set_theta_zero_location('N')  # 0° at the top
                            polar_ax[j].set_theta_direction(-1)       # Angles increase clockwise
                            # ax.set_rscale('log')             # Logarithmic radial scale
                    
                            # Assuming 'ax' is your polar axes object
                            r_label_angle = np.deg2rad(38)  # Position at 135 degrees
                            r_label_radius = polar_ax[j].get_rmax() + 1  # Halfway along the radial axis

                            polar_ax[j].text(r_label_angle, r_label_radius, 'Amplitude [m]',
                                             
                            rotation=64, rotation_mode='anchor',
                            ha='center', va='center', color='black',
                            fontsize=12)
                            # Plot lines between observed and modelled points
                            for i in range(len(constituents)):
                                theta = [observed_phase_rad[i], modelled_phase_rad[i]]
                                r = [observed_amp[i], model_amp[i]]
                                color = cmap(norm(abs(phase_diff_deg[i])))
                                polar_ax[j].plot(theta, r, color='darkgreen', linewidth=1)
                            
                            theta_label_radius = polar_ax[j].get_rmax() * 1.1  # Slightly beyond the outermost circle
                            
                            
                            polar_ax[j].set_xlabel('Phase (°)')
                            # Plot observed points with circle markers
                            polar_ax[j].scatter(observed_phase_rad, observed_amp, color='blue', s=25, marker='o', label='Observed')
                            
                            # Plot modelled points with cross markers
                            polar_ax[j].scatter(modelled_phase_rad, model_amp, color='red', s=25, marker='x', label='Modelled')
                            if model == 'UKC4':
                                model_name = 'UKC4'
                            elif model == 'PRIMEA':
                                model_name = 'IRENE'
                            else:
                                model_name = model
                                
                            if j == 0:
                                letter_label = 'A'
                            elif j == 1:
                                letter_label = 'B'
                            else:
                                letter_label = 'jabber'
                            polar_ax[j].set_title(f'{letter_label} ({model_name})')
                            
                            # Identify indices of top 5 observed amplitudes
                            top_indices = np.argsort(observed_amp)[-4:]
                            
                            # Annotate the top 5 constituents
                            for i in top_indices:
                                angle = observed_phase_rad[i]
                                radius = observed_amp[i]
                                label = constituents[i]
                                rotation = np.rad2deg(angle)
                                polar_ax[j].annotate(label,
                                            xy=(angle, radius),
                                            xytext=(5, 5),
                                            textcoords='offset points',
                                            ha='left',
                                            va='bottom',
                                            # rotation=rotation,
                                            rotation_mode='anchor',
                                            fontsize=10,
                                            color='darkgreen')

                            
                            # Add colorbar
                            # cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
                            # cbar.set_label('Phase Difference (degrees)')
                            
                            # plt.close()
                            #%
                            # Calculate phase differences
                            raw_diff = observed_pha - model_pha
                            wrapped_diff = (raw_diff + 180) % 360 - 180
                            phase_diff_deg = np.abs(wrapped_diff)
                            
                            raw_amp_diff = abs(observed_amp - model_amp)
                            # Sort constituents by observed amplitude
                            if model == 'PRIMEA':
                                sorted_indices = np.argsort(observed_amp)[::-1]
                                sorted_constituents = [constituents[i] for i in sorted_indices]
                                observed_amp_sorted = observed_amp[sorted_indices]
                                phase_diff_sorted = phase_diff_deg[sorted_indices]
                                
                                
                            amp_data[model] = model_amp[sorted_indices]
                            phase_diff_data[model] = phase_diff_deg[sorted_indices]
                            observed_amp_data[model] = raw_amp_diff[sorted_indices]
                            # Plotting
                            # fig, ax1 = plt.subplots(figsize=(10, 6))
                            
                            # ax2 = bar_ax[j].twinx()
                            
                            # x = np.arange(len(constituents_sorted))
                            # width = 0.4
                            
                            # # Bar plots
                            # bar_ax[j].bar(x - width/2, observed_amp_sorted, width=width, color='skyblue', label='Amplitude (m)')
                            # ax2.bar(x + width/2, phase_diff_sorted, width=width, color='salmon', label='Phase Difference (°)')
                            
                            # # Labels and titles
                            # bar_ax[j].set_xlabel('Tidal Constituents')
                            # bar_ax[j].set_ylabel('Amplitude (m)', color='skyblue')
                            # ax2.set_ylabel('Phase Difference (°)', color='salmon')
                            # bar_ax[j].set_xticks(x)
                            # bar_ax[j].set_xticklabels(constituents_sorted, rotation=45, ha='right')
                            # # plt.title('Tidal Constituents: Amplitude and Phase Difference')
                            
                            # Legends
                            # fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
                            
                            # plt.tight_layout()
                            # plt.show()
                            
                            
                            weighted_bias = np.sum((model_amp - observed_amp) * observed_amp) / np.sum(observed_amp)
                            weighted_rmse = np.sqrt(np.sum(((model_amp - observed_amp) ** 2) * observed_amp) / np.sum(observed_amp))
                            phase_diff = np.abs((observed_pha - model_pha + 180) % 360 - 180)
                            weighted_phase_error = np.sum(phase_diff * observed_amp) / np.sum(observed_amp)
                            
                            # Compute amplitude bias and RMSE
                            amp_bias = model_amp - observed_amp
                            amp_rmse = np.sqrt((model_amp - observed_amp) ** 2)
                            
                            # Compute phase difference, bias, and RMSE
                            phase_diff = np.abs((observed_pha - model_pha + 180) % 360 - 180)
                            phase_bias = model_pha - observed_pha
                            phase_rmse = np.sqrt((phase_bias) ** 2)
                            
                            # Create DataFrame for per-constituent metrics
                            df = pd.DataFrame({
                                'Constituent': constituents,
                                'Observed Amplitude (m)': observed_amp,
                                'Modelled Amplitude (m)': model_amp,
                                'Amplitude Bias (m)': amp_bias,
                                'Amplitude RMSE (m)': amp_rmse,
                                'Observed Phase (°)': observed_pha,
                                'Modelled Phase (°)': model_pha,
                                'Phase Difference (°)': phase_diff,
                                'Phase Bias (°)': phase_bias,
                                'Phase RMSE (°)': phase_rmse
                            })
                            
                            # Compute weights based on observed amplitude
                            weights = observed_amp / np.sum(observed_amp)
                            
                            # Compute overall weighted metrics
                            overall_amp_bias = np.sum(amp_bias * weights)
                            overall_amp_rmse = np.sqrt(np.sum((amp_bias ** 2) * weights))
                            overall_phase_bias = np.sum(phase_bias * weights)
                            overall_phase_rmse = np.sqrt(np.sum((phase_bias ** 2) * weights))
                            overall_phase_diff = np.sum(phase_diff * weights)
                            
                            # Save per-constituent metrics to CSV
                            df.to_csv(f'{output_dir}/{gauge}_{model}_vs_Observed_tidal_harmonic_metrics.csv', index=False)
                            
                            # Append overall metrics to the CSV
                            with RMSE_path.open('a') as f:
                                f.write('\n')
                                f.write(f'{model}_{gauge}_Weighted Amplitude Bias (m),{overall_amp_bias}\n')
                                f.write(f'{model}_{gauge}_Weighted Amplitude RMSE (m),{overall_amp_rmse}\n')
                                f.write(f'{model}_{gauge}_Weighted Phase Bias (°),{overall_phase_bias}\n')
                                f.write(f'{model}_{gauge}_Weighted Phase RMSE (°),{overall_phase_rmse}\n')
                                f.write(f'{model}_{gauge}_Weighted Phase Difference (°),{overall_phase_diff}\n')
                        
                           
                        # polar_fig.tight_layout()
                        handles, labels = polar_ax[0].get_legend_handles_labels()
                        polar_fig.legend(handles, labels, loc='lower left')
                        polar_fig.savefig(os.path.join(fig_path,'tidal_const_analysis_circle_' + gauge + 'models_vs_observed.png'), dpi = 300)
                        scatter_fig.savefig(os.path.join(fig_path,'tidal_const_analysis_scatterdual_' + gauge + 'models_vs_observed.png'), dpi = 300)
                        #%%
                        x = np.arange(len(sorted_constituents))
                        width = 0.6
                        
                        fig, ax1 = plt.subplots(figsize=(18, 6))
                        ax2 = ax1.twinx()
                        
                        # Define x positions for amplitude and phase
                        x_amp = x * 2     # Even positions
                        x_phase = x * 2 + 1  # Odd positions
                        
                        # Plot amplitude (same x position, different colors + transparency)
                        ax1.bar(x_amp, observed_amp_data['PRIMEA'], width=width, color='skyblue', alpha=0.6, label='IRENE vs Observed Amplitude')
                        ax1.bar(x_amp, observed_amp_data['UKC4'], width=width, color='dodgerblue', alpha=0.6, label='UKC4 vs Observed Amplitude')
                        # ax1.bar(x_amp, observed_amp_data, width=width, color='yellow', alpha=0.3, label='Observed')

                        # Plot phase difference
                        ax2.bar(x_phase, phase_diff_data['PRIMEA'], width=width, color='lightcoral', alpha=0.6, label='IRENE vs Observed Phase')
                        ax2.bar(x_phase, phase_diff_data['UKC4'], width=width, color='firebrick', alpha=0.6, label='UKC4 vs Observed Phase')
                        
                        # X-tick labels centered between amplitude and phase
                        xticks = (x_amp + x_phase) / 2
                        ax1.set_xticks(xticks)
                        ax1.set_xticklabels(sorted_constituents, rotation=45, ha='right')
                        ax1.set_xlabel('Tidal Constituents')
                        
                        # Y labels
                        ax1.set_ylabel('Amplitude [m]', color='navy')
                        ax2.set_ylabel('Phase Difference [°]', color='darkred')
                        
                        # Legend
                        h1, l1 = ax1.get_legend_handles_labels()
                        h2, l2 = ax2.get_legend_handles_labels()
                        fig.legend(h1 + h2, l1 + l2, loc='center', bbox_to_anchor=(0.5, 0.75))
                        
                        # plt.title(f'{gauge} – Overlaid Amplitude & Phase Comparison (PRIMEA vs UKC4)')
                        plt.tight_layout()
                        fig.savefig(os.path.join(fig_path,'tidal_const_analysis_bar_graph_'+ gauge +'_models_vs_observed.png'), dpi = 300)






                        #%%    
                plot_tidal_analysis(tide_storage)
                
            # I think this is the correct place for this to be. 
                         
            # print(self.ukc4y)
            tide_gauge_name = [j for j in self.tide_loc_dict.keys()]
            for i, tide_gauge in enumerate(tide_gauge_name):
                # import pdb; pdb.set_trace()
                tide_primea_rmse = calculate_rmse(tidx[i]['Measured Tide Gauge'][50:], prix[i]['IRENE Model'][50:])
                tide_ukc3_rmse = calculate_rmse(tidx[i]['Measured Tide Gauge'][50:], ukcy[i]['UKC4 Model'][50:])    
                prim_ukc3_rmse = calculate_rmse(prix[i]['IRENE Model'][50:], ukcy[i]['UKC4 Model'][50:])  
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
                                
            if variable_name == 'surface_height':
                tidal_analysis(mylist)
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
        transect_data = transect_data.sort_values(by='est_name', kind='stable').reset_index(drop=True)
        transect_data['id_old'] = transect_data['id']
        transect_data['id'] = pd.factorize(transect_data['est_name'], sort=True)[0]
        self.transect_data = transect_data
        distances, indicies = near_neigh(transect_data,self.df_search_points,1)
        
        self.transect_distances = distances
        self.transect_indicies = indicies
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

        def int_to_letter(n):
            return chr(ord('A') + n)
        # run the plotter for each estuary
        def plotter(minmax):
            fig, ax = plt.subplots(unique_estuaries.shape[0])
            fig.set_figheight(12)
            fig.set_figwidth(8)
            
            for i in unique_estuaries:
                letter_to_add = int_to_letter(i)
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
                
                ax[i].plot(new_dist, min_primea, 'r', label = 'IRENE')
                ax[i].plot(new_dist, min_ukc3, 'b', label = 'UKC4')
                ax[i].plot(new_dist, sub_frame_bathymetry, 'g', label = 'Bathymetry')
                ax[i].set_title(letter_to_add + ' (' +  sub_frame.est_name.iloc[0].capitalize() + ')')
                ax[i].set_ylim([miny, maxy])
                fig.supxlabel("Distance transecting estaury Mouth-River [km]")
                fig.supylabel("Height of surface [m]")
                if i == 0:
                    # fig.legend(loc='upper center', bbox_to_anchor=(0.8, 1.00), ncol=4)
                    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2)
                plt.tight_layout()
            # if isinstance(minmax, str):
            # plt.title(self.dataset_name)
            plt.savefig(fig_path + '/timestep_'+ str(minmax) +'_transects_along_estuaries.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            # else:
            #     #plt.subplots_adjust(top=0.5)
            #     #fig.suptitle(str(sliced_ukc3_tim_df.iloc[minmax][0]))
            #     plt.savefig(fig_path + '/timestep_'+ str(minmax) +'_transects_along_estuaries.png', dpi = 300)
             
            return sub_frame_primea
                #print(self.prim_sh)
        sub_frame_primea = plotter(106)
        sub_frame_primea = plotter(100)
        sub_frame_primea = plotter(174-6)
        for hours_to_plot in range(168,180, 1):
            _ = plotter(hours_to_plot)
        sub_frame_primea = plotter(174)
        sub_frame_primea = plotter('max')
        sub_frame_primea = plotter('min')
        sub_frame_primea = plotter(174+6)
        sub_frame_primea = plotter(174+13+13)# should be a high tide scenario. 
        return sub_frame_primea

    def max_compare(self, fig_path):
        prim_dict = self.data_dict['prim']
        ukc4_dict = self.data_dict['ukc4']
        #print(prim_dict)
        
        for i in ['surface_height', 'surface_salinity']:#, 'surface_salinity']:
            prim = prim_dict[i][50:,:,:]
            ukc4 = ukc4_dict[i][50:,:,:]
            
            ''' This method is flawed, you are better off having a rolling 3 hour window in which a max and min can be calculated. 
            '''
            
            
            for j in ['min', 'max']:
                if j == 'min':
                    if i == 'surface_salinity':
                        heightprim = calculate_rolling_min(prim, window_size=24).mean(dim='time_primea')
                        heightukc4 = calculate_rolling_min(ukc4, window_size=24).mean(dim='time_primea')
                    else:
                        # heightprim = prim.min(dim='time_primea')
                        # heightukc4 = ukc4.min(dim='time_primea')
                        heightprim = calculate_rolling_min(prim).mean(dim='time_primea')
                        heightukc4 = calculate_rolling_min(ukc4).mean(dim='time_primea')
                    
                else:
                    if i == 'surface_salinity':
                        heightprim = calculate_rolling_max(prim, window_size=24).mean(dim='time_primea')
                        heightukc4 = calculate_rolling_max(ukc4, window_size=24).mean(dim='time_primea')
                    else:
                        # heightprim = prim.max(dim='time_primea')
                        # heightukc4 = ukc4.max(dim='time_primea')
                        heightprim = calculate_rolling_max(prim).mean(dim='time_primea')
                        heightukc4 = calculate_rolling_max(ukc4).mean(dim='time_primea')
                   
                # Creating exonetial colour map 
                def exp_scale(data, exponent=3):
                    sign_data = np.sign(data)  # Preserve the sign for diverging scale
                    scaled_data = sign_data * (np.abs(data) ** exponent)
                    return scaled_data
                
                def adjust_colormap(cmap, exponent=1.5):
                    # Sample the colormap
                    cmap_vals = cmap(np.linspace(0, 1, 256))
                
                    # Ensure cmap_vals is correctly shaped (256, 4) for RGBA
                    if cmap_vals.shape[1] != 4:
                        raise ValueError("Colormap values should be in RGBA format")
                
                    # Create new indices for exponential adjustment
                    new_indices = np.linspace(0, 1, 256) ** (1 / exponent)  # Adjust exponent here
                
                    # Interpolate new colormap using adjusted indices
                    new_cmap_vals = np.empty_like(cmap_vals)
                    for i in range(4):  # Interpolate for each color channel (RGBA)
                        new_cmap_vals[:, i] = np.interp(new_indices, np.linspace(0, 1, 256), cmap_vals[:, i])
                    
                    return ListedColormap(new_cmap_vals)

                
                # adjusted_cmap = adjust_colormap(cmocean.cm.balance)

                from scipy.ndimage import distance_transform_edt
                height_difference = heightprim - heightukc4
                height_difference = height_difference[:,1:]
                lon_cut = self.lon[:,1:]
                lat_cut = self.lat[:,1:]
                
                # coast_mask = np.isnan(height_difference)
                
                # distance_from_coast_cells = distance_transform_edt(~coast_mask)
                # distance_from_coast_km = xr.DataArray(
                #     distance_from_coast_cells * 1.5,
                #     dims=height_difference.dims,
                #     coords=height_difference.coords
                # )
                
                # bin_edges_km = [0.0, 1.51, 4.5, 15.0, 30.0, 75.0]
                # bin_labels_km = ["0.0 – 1.5", "1.5 – 4.5", "4.5 – 15.0", "15.0 – 30.0", "30.0 – 75.0"]
                
                # mean_diffs = []
                # std_diffs = []
                
                # for i in range(len(bin_edges_km) - 1):
                #     bin_mask = (distance_from_coast_km >= bin_edges_km[i]) & (distance_from_coast_km < bin_edges_km[i + 1])
                #     valid_data = height_difference.where(bin_mask)
                
                #     # Compute mean/std safely with .item() to get float
                #     mean_val = valid_data.mean(skipna=True).item()
                #     std_val = valid_data.std(skipna=True).item()
                
                #     mean_diffs.append(round(mean_val, 5))
                #     std_diffs.append(round(std_val, 5))
                
                # # Step 7: Build final DataFrame
                # df_nearshore_bins = pd.DataFrame({
                #     "Distance Band (km)": bin_labels_km,
                #     "Mean Difference (m)": mean_diffs,
                #     "Standard Deviation (m)": std_diffs
                # })
                
                # Distance from coast
                # Step 2: Create coastal distance mask
                coast_mask = np.isnan(height_difference)
                distance_from_coast_cells = distance_transform_edt(~coast_mask)
                distance_from_coast_km = xr.DataArray(
                    distance_from_coast_cells * 1.5,
                    dims=height_difference.dims,
                    coords=height_difference.coords
                )
                
                # Step 3: Gradient calculation
                grad = np.abs(np.gradient(height_difference.values, axis=0))
                grad_da = xr.DataArray(grad, coords=height_difference.coords)
                
                # Step 4: Bin definitions
                bin_edges_km = [-0.01, 1.51, 3.0, 6.0, 12.0, 24.0, 48.0]
                bin_labels_km = ["0.0 – 1.5", "1.5 – 3.0", "3.0 – 6.0", "6.0 – 12.0", "12.0 – 24.0", "24.0 – 48.0"]
                
                # Step 5: Loop to compute stats
                mean_diffs, std_diffs, mean_grads, max_diffs = [], [], [], []
                
                for ibin in range(len(bin_edges_km) - 1):
                    bin_mask = (distance_from_coast_km >= bin_edges_km[ibin]) & (distance_from_coast_km < bin_edges_km[ibin + 1])
                    
                    valid_data = height_difference.where(bin_mask)
                    valid_grad = grad_da.where(bin_mask)
                
                    mean_val = valid_data.mean(skipna=True).item()
                    std_val = valid_data.std(skipna=True).item()
                    grad_val = valid_grad.mean(skipna=True).item()
                    max_val = valid_data.max(skipna=True).item()

                    
                    mean_diffs.append(round(mean_val, 5))
                    std_diffs.append(round(std_val, 5))
                    mean_grads.append(round(grad_val, 5))
                    max_diffs.append(round(max_val, 5))

                
                # Step 6: Build DataFrame
                df_nearshore_bins = pd.DataFrame({
                    "Distance Band (km)": bin_labels_km,
                    "Mean Difference (m)": mean_diffs,
                    "Standard Deviation (m)": std_diffs,
                    "Maximum Difference (m)": max_diffs,
                    "Mean Gradient (m/cell)": mean_grads
                })
                df_nearshore_bins.columns = [
                    "DistanceBand", "MeanDiff", "StdevDiff", "MaxDiff", "MeanGrad"
                ]

                
                
                fig, ax = plt.subplots()
                fig.set_figheight(7)
                fig.set_figwidth(5)
                ax.set_facecolor('lightgrey')
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
                
                # pcm = ax.pcolor(self.lon, self.lat, exp_scale(height_difference, 10), cmap=adjusted_cmap, shading='auto')  # Ensure shading='auto' for better color interpolation
                
                
                
                
                if i == 'surface_height':
                    # cmap = cmocean.cm.balance
                    cmap = cmr.fusion_r
                    pcm = ax.pcolor(lon_cut, lat_cut, height_difference, cmap = cmap)
                    pcm.set_array(height_difference)
                    if j == 'max':
                        pcm.set_clim(-1, 1)
                    else:
                        pcm.set_clim(-2, 2)
                        
                if i == 'surface_salinity':
                    cmap = cmr.waterlily
                    pcm = ax.pcolor(lon_cut, lat_cut, height_difference, cmap = cmap)
                    pcm.set_array(height_difference)
                    pcm.set_clim(-20, 20)
                    
                #pcm.set_clim(-1, 1)
                cbar = plt.colorbar(pcm)
                #cbar.set_ticks(np.linspace(-1,1,11))
                
                if j == 'min':
                    jname = 'Minimum '
                elif j== 'max':
                    jname = 'Maximum '
                    
                if i == 'surface_salinity':
                    iname = 'Salinity '
                elif i == 'surface_height':
                    iname = 'Surface Height '
                cbar.set_label(jname + f'$\Delta$IRENE {iname} [' + var_dict[i]['UNITS'] + ']')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.xlim([-3.58, -2.8])
                # plt.title(self.dataset_name)
                plt.tight_layout()
                
                plt.savefig(fig_path + '/diff_' + j + '_'+ i +'_analysis.png', dpi = 300)
                plt.close()
                # sanity_plot(heightprim, 'prim_height')
                # sanity_plot(heightukc4, 'ukc4_height')
                # plt.close()
            
                df_nearshore_bins.to_csv(Path(data_stats_path) / Path('diff_' + j + '_'+ i +'_analysis_as_statistics.csv'), index = False)

            
        return heightprim, heightukc4, height_difference
    
    
    def salinity_validation(self, ukc4sal, primsal):
        
        ukc4_sal_original = ukc4sal
        prim_sal_original = primsal
        '''
        This function matches observed salinity values with modeled values from both UKC4 and PRIMEA datasets.
    
        Parameters
        ----------
        df : pandas DataFrame
            DataFrame containing observational data with columns ['Lat', 'Lon', 'DateTime', 'Salinity'].
        ukc4sal : xarray.DataArray
            UKC4 salinity data with time, latitude, and longitude dimensions.
        primsal : xarray.DataArray
            PRIMEA salinity data with time, latitude, and longitude dimensions.
        lat_name : str, optional
            The name of the latitude variable in the xarray datasets. Default is 'nav_lat'.
        lon_name : str, optional
            The name of the longitude variable in the xarray datasets. Default is 'nav_lon'.
    
        Returns
        -------
        df_result : pandas DataFrame
            DataFrame with matched observed and modeled salinity values.
        '''
        #%% 
        window_hours = 72
        ukc4sal = ukc4_sal_original.rolling(time_primea=window_hours, center=True).max()
        primsal = prim_sal_original.rolling(time_primea=window_hours, center=True).max()
        
        from o_func.data_prepkit import extract_salinities
        from o_func import uk_bounds
        import pandas as pd
        lon, lat = uk_bounds()
        df = extract_salinities(start_path, lon, lat)
        df = df.dropna(subset=['Salinity']) # This is the raw observed dataset filtered to these areas
        
        
        # Quick check of last month only
        # Define the cutoff date
        # cutoff_date = pd.to_datetime('2014-02-01')
        end_cutoff = pd.to_datetime('2014-02-28')
        start_cutoff = pd.to_datetime('2013-01-01')
        # Filter the DataFrame for rows where 'DateTime' is after the cutoff date
        filtered_df = df[(df['DateTime'] > start_cutoff) & (df['DateTime'] < end_cutoff)]

        df = filtered_df
        
        df = df.reset_index(drop=True)
        from sklearn.neighbors import BallTree
        # Prepare coordinates from model data
        model_lats_ukc4 = ukc4sal['nav_lat'].values
        model_lons_ukc4 = ukc4sal['nav_lon'].values
        model_lats_prim = primsal['nav_lat'].values
        model_lons_prim = primsal['nav_lon'].values
    
        # Flatten latitude and longitude arrays to 1D for BallTree
        flat_model_coords_ukc4 = np.column_stack([model_lats_ukc4.ravel(), model_lons_ukc4.ravel()])
        flat_model_coords_prim = np.column_stack([model_lats_prim.ravel(), model_lons_prim.ravel()])
    
        # Build BallTrees for both UKC4 and PRIMEA models
        tree_ukc4 = BallTree(np.deg2rad(flat_model_coords_ukc4), metric='haversine')
        tree_prim = BallTree(np.deg2rad(flat_model_coords_prim), metric='haversine')
    
        # Convert observational lat/lon to radians for BallTree
        obs_coords = np.deg2rad(np.column_stack([df['Lat'].values, df['Lon'].values]))
    
        # Query BallTree for nearest neighbors (locations)
        dist_ukc4, idx_ukc4 = tree_ukc4.query(obs_coords, k=1)
        dist_prim, idx_prim = tree_prim.query(obs_coords, k=1)
        grid_shape = model_lats_ukc4.shape
    
        # Find nearest time indices in the model datasets
        def find_nearest_time(model_times, obs_time):
            time_diffs = np.abs((model_times - np.datetime64(obs_time)).astype('timedelta64[s]'))
            return np.argmin(time_diffs)
    
        # Initialize lists to store model salinity values
        matched_ukc4_salinities = []
        matched_prim_salinities = []
    
        # Loop through each observation
        for i, obs in df.iterrows():
            # print(i)
            # Get the observation time
            obs_time = obs['DateTime']
    
            # Find the nearest time index in the model datasets
            nearest_time_ukc4 = find_nearest_time(ukc4sal['time_primea'].values, obs_time)
            nearest_time_prim = find_nearest_time(primsal['time_primea'].values, obs_time)
    
            # Get the corresponding grid point indices from the BallTree
            flat_idx_ukc4 = idx_ukc4[i][0]  # This is the flat index from BallTree
            flat_idx_prim = idx_prim[i][0]
    
            # Convert the flat index back to the 2D grid
            grid_idx_ukc4 = np.unravel_index(flat_idx_ukc4, grid_shape)
            grid_idx_prim = np.unravel_index(flat_idx_prim, grid_shape)
    
            # Extract the salinity values for UKC4 and PRIMEA at the nearest time and location
            ukc4_salinity = ukc4sal.isel(time_primea=nearest_time_ukc4, y=grid_idx_ukc4[0], x=grid_idx_ukc4[1]).values
            prim_salinity = primsal.isel(time_primea=nearest_time_prim, y=grid_idx_prim[0], x=grid_idx_prim[1]).values
    
            # Append the salinity values to the lists
            matched_ukc4_salinities.append(ukc4_salinity)
            matched_prim_salinities.append(prim_salinity)
    
        # Add the matched salinities to the DataFrame
        df['UKC4_Salinity'] = matched_ukc4_salinities
        df['PRIMEA_Salinity'] = matched_prim_salinities
    
    
        observed_salinities = df['Salinity']
        ukc4_salinities = df['UKC4_Salinity']
        primea_salinities = df['PRIMEA_Salinity']
        #% Plotting Function
        def remove_outliers_iqr(observed, modelled):
            q1, q3 = np.percentile(observed, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        
            mask = (observed >= lower_bound) & (observed <= upper_bound)
            return observed[mask], modelled[mask]
        def remove_nans(observed, modelled):
            observed = np.array(observed, dtype=float)
            modelled = np.array(modelled, dtype=float)
            # Identify non-NaN values in the modelled array
            mask = ~np.isnan(modelled)
            # Apply mask to both arrays
            return observed[mask], modelled[mask]
                
        obs_ukc4, mod_ukc4 = remove_nans(observed_salinities, ukc4_salinities)
        # obs_ukc4, mod_ukc4 = remove_outliers_iqr(obs_ukc4, mod_ukc4)
        
        obs_prim, mod_prim = remove_nans(observed_salinities, primea_salinities)
        # obs_prim, mod_prim = remove_outliers_iqr(obs_prim, mod_prim)

        common_limit = [0, 35]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot PRIMEA vs Observed
        axes[0].scatter(obs_prim, mod_prim, color='blue', label='IRENE vs Observed', alpha=0.5)
        axes[0].plot(common_limit, common_limit, color='red', linestyle='--', label='y=x')  # Reference line
        axes[0].set_title('IRENE vs Observed Salinity')
        axes[0].set_xlabel('Observed Salinity [psu]')
        axes[0].set_ylabel('IRENE Modelled Salinity [psu]')
        axes[0].set_xlim(common_limit)
        axes[0].set_ylim(common_limit)
        axes[0].legend()
        
        # Plot UKC4 vs Observed
        axes[1].scatter(obs_ukc4, mod_ukc4, color='green', label='UKC4 vs Observed', alpha=0.5)
        axes[1].plot(common_limit, common_limit, color='red', linestyle='--', label='y=x')  # Reference line
        axes[1].set_title('UKC4 vs Observed Salinity')
        axes[1].set_xlabel('Observed Salinity [psu]')
        axes[1].set_ylabel('UKC4 Modelled Salinity [psu]')
        axes[1].set_xlim(common_limit)
        axes[1].set_ylim(common_limit)
        axes[1].legend()
        
        salinity_dict_to_save = {}
        salinity_dict_to_save['obs_prim'] = obs_prim
        salinity_dict_to_save['mod_prim'] = mod_prim
        salinity_dict_to_save['obs_ukc4'] = obs_ukc4
        salinity_dict_to_save['mod_ukc4'] = mod_ukc4
        
        data_proc_path = Path(data_stats_path).parent / 'data_proc'
        with open(data_proc_path / 'salinity_validation_data.pkl', 'wb') as file:
            pickle.dump(salinity_dict_to_save, file)
            
        
        # Quantifying fit using RMSE
        rmse_primea = np.sqrt(np.mean((obs_prim - mod_prim) ** 2))
        rmse_ukc4 = np.sqrt(np.mean((obs_ukc4 - mod_ukc4) ** 2))
        plt.savefig(fig_path + '/initial_salinity_validation.png', dpi = 300)
        plt.close()
        # COMPUTE Bins 
        # Combine observed and modelled into DataFrames for binning
        df_prim = pd.DataFrame({'obs': obs_prim, 'mod': mod_prim})
        df_ukc4 = pd.DataFrame({'obs': obs_ukc4, 'mod': mod_ukc4})
        
        # Define salinity bins (adjust as needed)
        bins = list(range(0, 40, 5))  # [0, 5, 10, ..., 35]
        labels = [f"{a}–{b}" for a, b in zip(bins[:-1], bins[1:])]
        
        # Bin the observed values
        df_prim['bin'] = pd.cut(df_prim['obs'], bins=bins, labels=labels)
        df_ukc4['bin'] = pd.cut(df_ukc4['obs'], bins=bins, labels=labels)
        
        # Compute RMSE in each bin
        print("\nPRIMEA RMSE by salinity bin:")
        for label in labels:
            sub = df_prim[df_prim['bin'] == label]
            if not sub.empty:
                rmse_bin = np.sqrt(np.mean((sub['mod'] - sub['obs'])**2))
                print(f"  {label} PSU: RMSE = {rmse_bin:.2f}")
        
        print("\nUKC4 RMSE by salinity bin:")
        for label in labels:
            sub = df_ukc4[df_ukc4['bin'] == label]
            if not sub.empty:
                rmse_bin = np.sqrt(np.mean((sub['mod'] - sub['obs'])**2))
                print(f"  {label} PSU: RMSE = {rmse_bin:.2f}")
                
        # COMPUTE Normal RMSE
        
        results = []
        
        for label in labels:
            sub_p = df_prim[df_prim['bin'] == label]
            sub_u = df_ukc4[df_ukc4['bin'] == label]
        
            def compute_stats(sub):
                if sub.empty:
                    return (np.nan, np.nan, 0)
                rmse = np.sqrt(np.mean((sub['mod'] - sub['obs']) ** 2))
                bias = np.mean(sub['mod'] - sub['obs'])
                count = len(sub)
                return (rmse, bias, count)
        
            rmse_p, bias_p, count_p = compute_stats(sub_p)
            rmse_u, bias_u, count_u = compute_stats(sub_u)
        
            results.append({
                "Bin": label,
                "RMSE_PRIMEA": rmse_p,
                "Bias_PRIMEA": bias_p,
                "N_PRIMEA": count_p,
                "RMSE_UKC4": rmse_u,
                "Bias_UKC4": bias_u,
                "N_UKC4": count_u,
            })
        
        import pandas as pd
        df_stats = pd.DataFrame(results)
        print(df_stats.to_string(index=False))
        
        
        labels = [r['Bin'] for r in results]
        rmse_p_list = [r['RMSE_PRIMEA'] for r in results]
        rmse_u_list = [r['RMSE_UKC4'] for r in results]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 0.2, rmse_p_list, width=0.4, label='IRENE')
        ax.bar(x + 0.2, rmse_u_list, width=0.4, label='UKC4')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Salinity [psu]')
        # ax.set_title('Salinity RMSE by 5 PSU Bins')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_path + '/stats_salinity_validation.png', dpi = 300)
        plt.close()
        rows = []

        RMSE_bin_path = Path(data_stats_path)/Path('Salinity_bin_stats.csv')
        for r in results:
            rmse_p = r["RMSE_PRIMEA"]
            rmse_u = r["RMSE_UKC4"]
        
            # Compute % improvement (positive means PRIMEA is better)
            if np.isnan(rmse_p) or np.isnan(rmse_u):
                improvement_pct = np.nan
                better = "NA"
            else:
                improvement_pct = 100 * (rmse_u - rmse_p) / rmse_u if rmse_u != 0 else np.nan
                better = "PRIMEA" if rmse_p < rmse_u else "UKC4"
        
            rows.append({
                "Bin": r["Bin"],
                "RMSE_PRIMEA": rmse_p,
                "Bias_PRIMEA": r["Bias_PRIMEA"],
                "N_PRIMEA": r["N_PRIMEA"],
                "RMSE_UKC4": rmse_u,
                "Bias_UKC4": r["Bias_UKC4"],
                "N_UKC4": r["N_UKC4"],
                "Better_Model": better,
                "PRIMEA_Improvement_per": improvement_pct
            })
        
        # Convert to DataFrame and save
        df_out = pd.DataFrame(rows)
        df_out.to_csv(RMSE_bin_path, index=False)
        
                
                
        
        print(f'PRIMEA RMSE: {rmse_primea:.2f}, UKC4 RMSE: {rmse_ukc4:.2f}')
        #%%
        RMSE_path = Path(data_stats_path)/Path('RMSE_stats.txt')
        # Format the appended data
        new_data = (
            "\n"
            "--------------------------------------------------------------------------------\n"
            "Salinity Validation to Observed Points:\n"
            f"    PRIMEA RMSE: {rmse_primea:.2f}\n"
            f"    UKC4 RMSE: {rmse_ukc4:.2f}\n"
        )
        
        # Append the new data to the file
        with RMSE_path.open('a') as file:
            file.write(new_data)
        
        return df

        # Example usage
        # df = extract_salinities(start_path, lon, lat)
        #!!! result_df = salinity_validation(df, ukc4sal, primsal)

    
    # Optionally: df_cleaned.to_csv('cleaned_storm_table.csv')
    def salinity_rofi(self, ukc4sal, primsal):
        #%%
        START = np.datetime64("2014-01-01")  # 1 Jan (inclusive)
        
        ukc4_salinity = ukc4sal.sel(time_primea=slice(START, None))
        prim_salinity  = primsal.sel(time_primea=slice(START, None))
                
        # --- ROFI diagnostics for UKC4 vs PRIMEA/IRENE ---
        from pyproj import CRS, Transformer
        from sklearn.neighbors import BallTree
        
        # -------------- USER SETTINGS --------------
        SSTAR = 31.0                 # ROFI threshold (psu)
        MOUTH_LON, MOUTH_LAT = -3.3, 53.5  # estuary mouth lon/lat  (set for your case)
        CENTERLINE_BEARING = 315     # bearing (deg) pointing offshore; 0=N, 90=E, 180=S, 270=W
        CENTERLINE_LEN_KM = 50.0
        CENTERLINE_STEP_KM = 0.5
        EPSG_UTM = 32630             # Pick a suitable UTM zone for your domain (32630 ~ Liverpool Bay)
        SPINUP_TIME = None           # e.g. "2013-11-05" or integer index to drop initial spin-up; or None
        
        # -------------- INPUTS (already loaded) --------------
        # Expect:
        #   ukc4_salinity: DataArray(time_primea, y, x) with coords nav_lon(y,x), nav_lat(y,x)
        #   prim_salinity: DataArray(time_primea, y, x) with same dims/coords
        
        # -------------- HELPERS --------------
        def normalise(da):
            """Return DataArray with dims renamed to (time,y,x), values float32 in psu,
               and 2-D lon/lat attached as coords 'lon','lat' (curvilinear)."""
            da = da.rename({"time_primea": "time"})
            vals = da.astype("float32")
            # Some PRIMEA exports declare '1e-3' units; rescale only if values look tiny
            if np.nanmax(vals.values) < 1.0:
                vals = vals * 1e3
            lon2d = da["nav_lon"].values
            lat2d = da["nav_lat"].values
            vals = vals.assign_coords(lon=(("y","x"), lon2d),
                                      lat=(("y","x"), lat2d))
            return vals
        
        def maybe_drop_spinup(da, spinup):
            if spinup is None:
                return da
            if isinstance(spinup, (int, np.integer)):
                return da.isel(time=slice(spinup, None))
            # else assume datetime-like string
            return da.sel(time=slice(np.datetime64(spinup), None))
        
        # Projection helpers
        crs_ll = CRS.from_epsg(4326)
        crs_utm = CRS.from_epsg(EPSG_UTM)
        tfm = Transformer.from_crs(crs_ll, crs_utm, always_xy=True)
        tfm_back = Transformer.from_crs(crs_utm, crs_ll, always_xy=True)
        
        def to_utm(lon2d, lat2d):
            X, Y = tfm.transform(lon2d, lat2d)
            return np.asarray(X), np.asarray(Y)
        
        def cell_areas_m2(lon2d, lat2d):
            """Approx cell area as dx * dy on projected grid."""
            X, Y = to_utm(lon2d, lat2d)
            dx = np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1))
            dx = np.pad(dx, ((0,0),(0,1)), mode="edge")
            dy = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0))
            dy = np.pad(dy, ((0,1),(0,0)), mode="edge")
            A = dx * dy  # m^2
            return A
        
        def occupancy(da, sstar=SSTAR):
            """Fraction of time with S < S* (ignores NaNs)."""
            return (da < sstar).mean(dim="time", skipna=True)
        
        def build_centerline(lon0, lat0, bearing_deg, length_km, step_km):
            """Straight centerline from mouth in UTM, returned as lon/lat points & cumulative distance (km)."""
            x0, y0 = tfm.transform(lon0, lat0)
            theta = np.deg2rad(90.0 - bearing_deg)  # convert compass bearing to math angle
            n = int(length_km / step_km) + 1
            xs = x0 + (np.arange(n)*step_km*1000.0) * np.cos(theta)
            ys = y0 + (np.arange(n)*step_km*1000.0) * np.sin(theta)
            cl_lon, cl_lat = tfm_back.transform(xs, ys)
            seg = np.hypot(np.diff(xs), np.diff(ys))
            cumd_km = np.concatenate([[0.0], np.cumsum(seg)]) / 1000.0
            return np.asarray(cl_lon), np.asarray(cl_lat), cumd_km
        
        def _balltree_from_lonlat(lon2d, lat2d):
            """Haversine BallTree on curvilinear grid."""
            pts = np.column_stack([np.deg2rad(lat2d.ravel()),
                                   np.deg2rad(lon2d.ravel())])
            return BallTree(pts, metric="haversine")
        
        def nearest_indices_along(lon2d, lat2d, tgt_lon, tgt_lat):
            """Map target lon/lat points to nearest (y,x) indices on curvilinear grid."""
            tree = _balltree_from_lonlat(lon2d, lat2d)
            q = np.column_stack([np.deg2rad(np.asarray(tgt_lat)),
                                 np.deg2rad(np.asarray(tgt_lon))])
            _, ind = tree.query(q, k=1)
            ind = ind.ravel()
            ny, nx = lon2d.shape
            iy, ix = np.divmod(ind, nx)
            return iy.astype(int), ix.astype(int)
        
        def front_distance_ts_nearest(da, cl_lon, cl_lat, cumd_km, sstar=SSTAR):
            """Farthest distance along centerline where S < S* (nearest-cell sampling)."""
            lon2d = da.coords["lon"].values
            lat2d = da.coords["lat"].values
            iy, ix = nearest_indices_along(lon2d, lat2d, cl_lon, cl_lat)
            # time × points slice
            Sline = da.isel(y=("pt", iy), x=("pt", ix))
            below = (Sline < sstar).values  # shape: (time, points)
            # farthest 'True' index from the mouth (end of the line is offshore)
            idx = below.shape[1] - 1 - np.argmax(below[:, ::-1], axis=1)
            hits = below.any(axis=1)
            idx[~hits] = -1
            dist = np.where(idx >= 0, cumd_km[idx], np.nan)
            return xr.DataArray(dist, coords={"time": da.time}, dims=("time",))
        
        def plume_area_ts(da, area_m2, sstar=SSTAR):
            mask = (da < sstar)
            A = xr.DataArray(area_m2, coords={"y": da.y, "x": da.x}, dims=("y","x"))
            km2 = (mask * A).sum(dim=("y","x"), skipna=True) / 1e6
            return km2
        
        # -------------- PIPELINE --------------
        # Normalise inputs
        S1 = normalise(ukc4_salinity)
        S2 = normalise(prim_salinity)
        
        # Optional spin-up removal
        S1 = maybe_drop_spinup(S1, SPINUP_TIME)
        S2 = maybe_drop_spinup(S2, SPINUP_TIME)
        
        # Diagnostics
        P1 = occupancy(S1, SSTAR)   # 0..1
        P2 = occupancy(S2, SSTAR)
        
        cl_lon, cl_lat, cumd = build_centerline(MOUTH_LON, MOUTH_LAT,
                                                CENTERLINE_BEARING,
                                                CENTERLINE_LEN_KM,
                                                CENTERLINE_STEP_KM)
        
        D1 = front_distance_ts_nearest(S1, cl_lon, cl_lat, cumd, SSTAR)
        D2 = front_distance_ts_nearest(S2, cl_lon, cl_lat, cumd, SSTAR)
        
        Agrid = cell_areas_m2(S1.lon.values, S1.lat.values)
        A1 = plume_area_ts(S1, Agrid, SSTAR)
        A2 = plume_area_ts(S2, Agrid, SSTAR)
        
        # -------------- PLOTS --------------
        # (1) Occupancy maps
        fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
        for ax, P, name in [(axes[0], P1, "UKC4"),
                            (axes[1], P2, "PRIMEA/IRENE")]:
            im = ax.pcolormesh(S1.lon, S1.lat, 100.0 * P.where(np.isfinite(P)), shading="nearest")
            ax.plot(MOUTH_LON, MOUTH_LAT, marker="x", ms=6)
            ax.set_title(f"{name}: plume occupancy (S < {SSTAR:.1f} psu)")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            cb = plt.colorbar(im, ax=ax); cb.set_label("% of timesteps")
        
        # (2) Front distance time series
        plt.figure(figsize=(10,4))
        plt.plot(D1.time, D1, label="UKC4")
        plt.plot(D2.time, D2, label="PRIMEA/IRENE")
        plt.ylabel("Front distance along centerline (km)")
        plt.xlabel("Time")
        plt.title(f"ROFI front distance vs time (S < {SSTAR:.1f} psu)")
        plt.legend()
        plt.tight_layout()
        
        # (3) Plume area time series (km^2)
        plt.figure(figsize=(10,4))
        plt.plot(A1.time, A1, label="UKC4")
        plt.plot(A2.time, A2, label="PRIMEA/IRENE")
        plt.ylabel(r"Plume area (km$^2$)  [S < " + f"{SSTAR:.1f}" + " psu]")
        plt.xlabel("Time")
        plt.title("ROFI plume area vs time")
        plt.legend()
        plt.tight_layout()


        # --- Styled occupancy maps to match your max_compare figures ---
        import numpy.ma as ma
        # import cmasher as cmr
        import cmocean as cmo
        
        # Common valid mask so land/NaN handled the same in both
        common = np.isfinite(S1).any("time") & np.isfinite(S2).any("time")
        P1c = P1.where(common)
        P2c = P2.where(common)
        
        # Convert to percentage
        occ1 = (100.0 * P1c).astype("float32")
        occ2 = (100.0 * P2c).astype("float32")
        
        # "Cut off some of the x-axis" (drop first column to clean coastline edge)
        occ1 = occ1.isel(x=slice(1, None))
        occ2 = occ2.isel(x=slice(1, None))
        lon_cut = S1.lon.isel(x=slice(1, None))
        lat_cut = S1.lat.isel(x=slice(1, None))
        
        # Mask to show land/NaN as light grey
        occ1m = ma.masked_invalid(occ1.values)
        occ2m = ma.masked_invalid(occ2.values)
        
        # Use your salinity colormap; set NaNs to light grey
        # Use cmocean haline and set NaNs to light grey
        cmap = cmo.cm.haline.copy()
        cmap.set_bad("lightgrey")

        # Plot settings
        xlim = (-3.58, -2.80)   # your window
        figsize = (5, 7)
        clabel = "% of timesteps [S < {:.1f} psu]".format(SSTAR)
        
        def plot_occ(lon2d, lat2d, data_m, title, save_path):
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor("lightgrey")  # back-fill around pcolormesh
            pcm = ax.pcolormesh(lon2d, lat2d, data_m, shading="nearest", cmap=cmap, vmin=0, vmax=100)
            # YOU CAN plot the location of mouth across estuary transects
            # ax.plot(MOUTH_LON, MOUTH_LAT, marker="x", ms=6)
            cbar = plt.colorbar(pcm, ax=ax)
            cbar.set_label(clabel)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.set_xlim(xlim)
            # ax.set_title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            # plt.close(fig)
        
        plot_occ(lon_cut, lat_cut, occ1m, "UKC4: plume occupancy", Path(fig_path) /  "ukc4_rofi_occupancy.png")
        plot_occ(lon_cut, lat_cut, occ2m, "PRIMEA/IRENE: plume occupancy", Path(fig_path) /  "primea_rofi_occupancy.png")

        #%%
        
    def storm_surge_analysis(self):
        #%%
        import ttide as tt
        from datetime import datetime, timedelta
        import scipy.io as sio
        import subprocess
        from scipy.io import loadmat
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from functools import partial
        from tqdm import tqdm
        import warnings

        storm_storage = Path(fig_path) / "storm_surge"
        storm_storage.mkdir(parents=True, exist_ok=True)
        
        # Function to convert datetime to MATLAB required format. 
        def datetime_to_matlab_datenum(dt_array):
            return np.array([
                dt.toordinal() + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400 + 366
                for dt in dt_array
            ])
   
        # Generate correct time inputs for use within ttide analysis. 
        def time_maker_from_xarray(tt_time, offset_half_hour=False):
            """
            Create T-Tide time inputs from xarray datetime64[ns] array.
        
            Parameters
            ----------
            tt_time : xarray.DataArray or np.ndarray
                1D array of datetime64[ns]
            offset_half_hour : bool
                If True, shifts stime by 0.5/24 to match PRIMEA convention
        
            Returns
            -------
            t_datetime : list of datetime.datetime
                List of datetime objects
            t_rel_days : np.ndarray
                Relative days from start (for t_predic)
            dt_hours : float
                Timestep in hours (for t_tide)
            stime : float
                MATLAB-style datenum using `.toordinal()` only
            """
            t_np = pd.to_datetime(tt_time.values)
        
            if len(t_np) < 2:
                raise ValueError("Need at least two time points")
        
            dt_seconds = (t_np[1] - t_np[0]).total_seconds()
            dt_hours = dt_seconds / 3600.0
        
            start = t_np[0].to_pydatetime()
            t_rel_days = np.array([(t - start).total_seconds() / 86400 for t in t_np])
            stime = start.toordinal() + (0.5 / 24 if offset_half_hour else 0)  # round to midnight or 00:30
        
            return t_np.tolist(), t_rel_days, dt_hours, stime


        # Function recombines the ttide prediction into a workable dataarray. 
        def recombine_ttide_predition(output, sh_prim, label, varname='predicted'):
            """
            Reconstruct a full 3D grid DataArray (time, y, x) from sparse prediction output.
    
            Parameters
            ----------
            output : list of dicts
                Each dict should have keys 'y', 'x', and the variable name (e.g., 'predicted').
            sh_prim : xarray.DataArray
                Reference DataArray with correct dimensions and coordinates.
            label : str
                Label to identify the variable in metadata.
            varname : str, default='predicted'
                The key in the `output` dicts to extract.
        
            Returns
            -------
            xarray.DataArray
                Reconstructed variable as a full-sized DataArray.
            """
            time_dim = 'time_primea'
            nt, ny, nx = sh_prim.sizes[time_dim], sh_prim.sizes['y'], sh_prim.sizes['x']
            full_grid = np.full((nt, ny, nx), np.nan)
        
            for item in output:
                y, x = item['y'], item['x']
                values = np.array(item[varname])  # e.g., predicted tide
                full_grid[:, y, x] = values
        
            return xr.DataArray(
                data=full_grid,
                dims=(time_dim, 'y', 'x'),
                coords={
                    time_dim: sh_prim.coords[time_dim],
                    'y': sh_prim.coords['y'],
                    'x': sh_prim.coords['x'],
                    'lat': (('y', 'x'), sh_prim['nav_lat'].values),
                    'lon': (('y', 'x'), sh_prim['nav_lon'].values)
                },
                attrs={
                    'long_name': f'{label} ({varname})',
                    'units': sh_prim.attrs.get('units', 'unknown')
                },
                name=f'{varname}_{label.lower()}'
            )
        
        
        # Extract raw data here 
        prim_dict = self.data_dict['prim']
        ukc4_dict = self.data_dict['ukc4']

        # Extract surface height component
        sh_prim = prim_dict['surface_height' ].isel(time_primea=slice(24*7, None))
        sh_ukc4 = ukc4_dict['surface_height'].isel(time_primea=slice(24*7, None))
        
        # What is land and what is sea
        valid_indices = np.argwhere(~np.isnan(sh_prim[0].values))
        flat_grid = sh_prim.values  # (time, y, x)

        # Offer warning when erronous data is present. 

        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in cast.*")

        
        # Perform parallel processing of data to extract the storm surge components. 
        wrapped_func = partial(process_point, sh_prim=sh_prim)

        output_prim = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(wrapped_func, (i, idx)) for i, idx in enumerate(valid_indices)]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Points"):
                result = future.result()
                if result is not None:
                    output_prim.append(result)
                    
        # run for ukc4
        wrapped_func = partial(process_point, sh_prim=sh_ukc4)

        output_ukc4 = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(wrapped_func, (i, idx)) for i, idx in enumerate(valid_indices)]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Points"):
                result = future.result()
                if result is not None:
                    output_ukc4.append(result)



        print(output_prim[0].keys())
        ny, nx = sh_prim.sizes['y'], sh_prim.sizes['x']
        max_resid_grid = np.full((ny, nx), np.nan)
        for item in output_prim:
            y, x = item['y'], item['x']
            max_resid_grid[y, x] = item['max_residual']
               
        max_resid_da = recombine_ttide_predition(output = output_prim, sh_prim = sh_prim, label = 'predicted_MSL_adjusted', varname='predicted_MSL_adjusted')
        
        def plot_max_residual():
            max2d = max_resid_da.max(dim='time_primea')
    
            max2d.plot(
                x='lon',
                y='lat',
                cmap='Reds',
                figsize=(10, 6),
                cbar_kwargs={'label': 'Max Storm Surge Residual [m]'}
            )
            # plt.title('Maximum Storm Surge Residual (Observed - Predicted Tide)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.tight_layout()
            plt.show()
        
        
        plot_max_residual()
        
        def plot_mask(output, sh):
            from matplotlib.colors import ListedColormap

            # Create custom colormap with black for False (0) and yellow for True (1)
            cmap = ListedColormap(['black', 'yellow'])

            meanmask_resid_da = recombine_ttide_predition(output = output, sh_prim = sh, label = 'MSL Masked at -0.3 to +0.3 m', varname='meanSL_mask')[0,:,:]
            plt.figure(figsize=[5,7])
            pc = plt.pcolor(meanmask_resid_da.lon, meanmask_resid_da.lat, meanmask_resid_da, cmap=cmap, vmin=0, vmax=1)
            # plt.colorbar(pc, label='Mask (0=False, 1=True)')
            cbar = plt.colorbar(pc, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['False', 'True'])
            cbar.set_label('Mask Value')
            # plt.title('Boolean Mask at Time Step 0')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.xlim([-3.6,-2.8])
            plt.tight_layout()
            plt.savefig(Path(fig_path)/'boolean_storm_surge_mask.png', dpi = 150)
            plt.close()
            return meanmask_resid_da
        
        
        # def build_mask_mapper(mask_da, data_source=None):
        #     from sklearn.neighbors import BallTree
        #     """
        #     Build mapper from valid mask points to invalid ones, optionally filtering by non-NaN data.
            
        #     Parameters
        #     ----------
        #     mask_da : xarray.DataArray
        #         Boolean mask: True = valid (yellow), False = fill (black)
        #     data_source : xarray.DataArray, optional
        #         DataArray of the same shape as mask_da. If provided, only valid & non-NaN values are used.
            
        #     Returns
        #     -------
        #     dict with mapping and coordinates
        #     """
        #     lat = mask_da['lat'].values
        #     lon = mask_da['lon'].values
        #     mask = mask_da.values.astype(bool)
        #     yx_shape = mask.shape
        
        #     flat_mask = mask.flatten()
        #     flat_lat = lat.flatten()
        #     flat_lon = lon.flatten()
        #     coords_rad = np.deg2rad(np.column_stack((flat_lat, flat_lon)))
        
        #     if data_source is not None:
        #         valid_data_mask = ~np.isnan(data_source.values)
        #         flat_data_mask = valid_data_mask.flatten()
        #         flat_mask &= flat_data_mask  # update to require valid data
        #         print("Filtered valid mask points by non-NaN data values.")
        
        #     valid_idx = np.where(flat_mask)[0]
        #     missing_idx = np.where(~flat_mask)[0]
        
        #     valid_coords = coords_rad[valid_idx]
        #     tree = BallTree(valid_coords, metric='haversine')
        #     _, nearest_valid_idx = tree.query(coords_rad[missing_idx], k=1)
        
        #     mapping = valid_idx[nearest_valid_idx[:, 0]]
        
        #     return {
        #         'mapping': mapping,
        #         'valid_idx': valid_idx,
        #         'missing_idx': missing_idx,
        #         'yx_shape': yx_shape
        #     }
        from sklearn.neighbors import BallTree
        
        def build_mask_mapper(mask_da):
            """
            Builds a mapping from False cells to True cells in a mask, prioritizing adjacent True neighbors.
            Falls back to nearest True using BallTree if no adjacent valid cell is available.
        
        
            The mask_da congtains true/false and nans. 
            Parameters
            ----------
            mask_da : xarray.DataArray
                A 2D DataArray with True, False, and NaN:
                - True: valid source cell
                - False: cell to be filled
                - NaN: unusable cell (ignored)
        
            Returns
            -------
            dict
                Dictionary with keys:
                    - 'mapping': index into valid array for each missing cell
                    - 'missing_idx': (N, 2) array of [y, x] for each missing point
                    - 'valid_idx': (M, 2) array of [y, x] for each valid point
                    - 'yx_shape': shape of the original mask
            """
            
            
            mask = mask_da.values
            yx_shape = mask.shape
            lat = mask_da['lat'].values
            lon = mask_da['lon'].values
        
            # Define valid and missing indices
            # Valid is points that are okay
            valid_mask = (mask == True)
            # Fill is the points which contain the poor data
            fill_mask = (mask == False)
        
            # These are the index locations of the good and bad data. 
            valid = np.argwhere(valid_mask)
            missing = np.argwhere(fill_mask)
            
            # Check that there are actually values to map
            if valid.size == 0 or missing.size == 0:
                raise ValueError("No valid or missing cells found in mask.")

            valid_latlon = np.deg2rad(np.column_stack((
                lat[valid[:, 0], valid[:, 1]],
                lon[valid[:, 0], valid[:, 1]]
            )))
            missing_latlon = np.deg2rad(np.column_stack((
                lat[missing[:, 0], missing[:, 1]],
                lon[missing[:, 0], missing[:, 1]]
            )))
            
            # Build BallTree and query nearest valid point for each missing point
            tree = BallTree(valid_latlon, metric='haversine')
            _, nn = tree.query(missing_latlon, k=1)
            mapping = nn[:, 0]  # index into valid array
                    

            return {
                'mapping': mapping,
                'missing_idx': missing,
                'valid_idx': valid,
                'yx_shape': yx_shape
            }


        def apply_mask_mapping(data_da, mapper):
            """
            Applies a precomputed mapping to fill masked values in a dataset.
        
            Parameters
            ----------
            data_da : xarray.DataArray
                2D or 3D array (time, y, x) or (y, x) to fill using the mapper
            mapper : dict
                Dictionary returned from build_mask_mapper()
        
            Returns
            -------
            xarray.DataArray
                A new DataArray with the masked points filled
            """
            mapping = mapper['mapping']
            missing_idx = mapper['missing_idx']
            yx_shape = mapper['yx_shape']
        
            def fill_2d(data):
                if data.shape != yx_shape:
                    raise ValueError(f"Expected shape {yx_shape}, got {data.shape}")
                
                flat = data.flatten()
                filled = flat.copy()
                
                # Convert (y, x) → flat index for source and target
                missing_flat = np.ravel_multi_index(missing_idx.T, yx_shape)
                valid_idx = mapper['valid_idx']
                source_yx = valid_idx[mapping]  # shape (N, 2)
                source_flat = np.ravel_multi_index(source_yx.T, yx_shape)
            
                # Fill the missing cells with the corresponding source values
                filled[missing_flat] = flat[source_flat]
            
                return filled.reshape(yx_shape)
            # Handle 3D input with time dimension
            non_spatial_dims = [dim for dim in data_da.dims if dim not in ('y', 'x')]
            if non_spatial_dims:
                time_dim = non_spatial_dims[0]
                filled_data = np.stack([
                    fill_2d(data_da.isel({time_dim: t}).values)
                    for t in range(data_da.sizes[time_dim])
                ])
                return xr.DataArray(
                    filled_data,
                    dims=(time_dim, 'y', 'x'),
                    coords={
                        time_dim: data_da[time_dim],
                        'lat': data_da['lat'],
                        'lon': data_da['lon']
                    },
                    name=f"{data_da.name}_filled"
                )
            else:
                # 2D case
                filled = fill_2d(data_da.values)
                return xr.DataArray(
                    filled,
                    dims=('y', 'x'),
                    coords={
                        'lat': data_da['lat'],
                        'lon': data_da['lon']
                    },
                    name=f"{data_da.name}_filled"
                )



        def plot_mapping_flow(mask_da, mapper, title='Mapping Flow'):
            """
            Plot lines showing how each masked cell (False) was filled from its nearest valid (True) neighbor.
        
            Parameters
            ----------
            mask_da : xarray.DataArray
                The boolean mask used to generate the mapping (2D: y, x)
            mapper : dict
                The mapping output from build_adjacency_first_mapper_from_mask()
            title : str
                Plot title
            """
          
            lat = mask_da['lat'].values
            lon = mask_da['lon'].values
            mask = mask_da.values
        
            mapping = mapper['mapping']            # shape (N,)
            valid_idx = mapper['valid_idx']        # shape (M, 2) — (y, x) of source points
            missing_idx = mapper['missing_idx']    # shape (N, 2) — (y, x) of targets
            yx_shape = mapper['yx_shape']
        
            # Look up source yx pairs directly from valid_idx using mapping
            source_yx = valid_idx[mapping]         # (N, 2)
            target_yx = missing_idx                # (N, 2)
        
            # Extract lat/lon of source and target
            source_lat = lat[source_yx[:, 0], source_yx[:, 1]]
            source_lon = lon[source_yx[:, 0], source_yx[:, 1]]
            target_lat = lat[target_yx[:, 0], target_yx[:, 1]]
            target_lon = lon[target_yx[:, 0], target_yx[:, 1]]
        
            # Plot the base mask
            plt.figure(figsize=(6, 8))
            cmap = ListedColormap(['black', 'yellow'])
            pc = plt.pcolor(lon, lat, mask, cmap=cmap, vmin=0, vmax=1)
            cbar = plt.colorbar(pc, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['False', 'True'])
            cbar.set_label('Mask Value')
        
            # Draw lines from target to source
            for x0, y0, x1, y1 in zip(target_lon, target_lat, source_lon, source_lat):
                plt.plot([x0, x1], [y0, y1], color='red', linewidth=1.0, alpha=0.6)
        
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(title)
            plt.xlim([np.nanmin(lon), np.nanmax(lon)])
            plt.ylim([np.nanmin(lat), np.nanmax(lat)])
            plt.tight_layout()
            plt.show()
        
        meanmask_resid_da_prim = plot_mask(output_prim, sh_prim)
        mask_mapper_prim = build_mask_mapper(meanmask_resid_da_prim)
        prim_ttide = recombine_ttide_predition(output = output_prim, sh_prim = sh_prim, label = 'Ttide predicted surface heights', varname='predicted')
        remapped_prim_tide = apply_mask_mapping(prim_ttide, mask_mapper_prim)
        # Plot prim mapping flow
        plot_mapping_flow(meanmask_resid_da_prim, mask_mapper_prim, title='Mapping Flow')
        
        # Use the same map points from prim to ukc4
        ukc4_ttide = recombine_ttide_predition(output = output_ukc4, sh_prim = sh_ukc4, label = 'Ttide predicted surface heights', varname='predicted')
        remapped_ukc4_tide = apply_mask_mapping(ukc4_ttide, mask_mapper_prim)
        # Plot prim mapping flow
        plot_mapping_flow(meanmask_resid_da_prim, mask_mapper_prim, title='Mapping Flow')
        
        def plot_tidal_predictions_and_storm_surge(point):
            y, x = point['y'], point['x']
            lon, lat = round(point['lon'].item(),3), round(point['lat'].item(),3)
            print(f"Plotting for grid point: y={y}, x={x}")
            
            # 2. Extract time and data
            observed = sh_prim[:, y, x].values
            predicted = np.array(point['predicted_MSL_adjusted'])
            
            # 3. Time vector (should match tide prediction)
            tt_time = sh_prim.time_primea.values
            tt_time_py = pd.to_datetime(tt_time)
            
            # 4. Compute residual
            residual = observed - predicted
            
            # 5. Plot
            fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            
            axs[0].plot(tt_time_py, observed, label="Observed", color='black')
            axs[0].plot(tt_time_py, predicted, label="Predicted Tide", color='blue')
            axs[0].legend()
            axs[0].set_ylabel("Surface Height [m]")
            axs[0].set_title(f"Point (Lon={lon}, Lat={lat})")
            
            axs[1].plot(tt_time_py, residual, label="Storm Surge Residual", color='red')
            axs[1].axhline(0, linestyle='--', color='k', linewidth=0.5)
            axs[1].legend()
            axs[1].set_ylabel("Storm Surge Residual [m]")
            axs[1].set_xlabel("Time")
            
            plt.tight_layout()
            plt.show()
        #%%    
        # This shows you how to plot the positions of the known storms within the dataset. 

        

        
        def plot_tidal_predictions_and_storm_surge_with_map(point, label = 'prim', saveas='local.png'):
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import cartopy.crs as ccrs
            import geopandas as gpd
            from pathlib import Path
            import pandas as pd
            import numpy as np
            from matplotlib.dates import DayLocator, DateFormatter

        
        
            if label == 'prim':
                observed_label = 'IRENE'
                predicted_label = 'T_Tide'
            elif label == 'ukc4':
                observed_label = 'UKC4'
                predicted_label = 'T_Tide'
            # === Input info ===
            y, x = point['y'], point['x']
            lon, lat = round(point['lon'].item(), 3), round(point['lat'].item(), 3)
            print(f"Plotting for grid point: y={y}, x={x}")
        
            # === Extract tidal data ===
            observed = sh_prim[:, y, x].values
            predicted = np.array(point['predicted_MSL_adjusted'])
            residual = observed - predicted
            tt_time_py = pd.to_datetime(sh_prim.time_primea.values)
        
            # === Figure layout ===
            fig = plt.figure(figsize=(13, 9))
            outer_gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
            left_gs = outer_gs[0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.15)
            ax1 = fig.add_subplot(left_gs[0])
            ax2 = fig.add_subplot(left_gs[1], sharex=ax1)
            ax_map = fig.add_subplot(outer_gs[1], projection=ccrs.PlateCarree())
        
            # === Time Series Plots ===
            ax1.plot(tt_time_py, observed, label=observed_label, color='black', linewidth=0.8)
            ax1.plot(tt_time_py, predicted, label=predicted_label, color='blue', linewidth=0.8)
            ax1.set_ylabel("Surface Height [m]")
            ax1.set_title("A")
            ax1.tick_params(labelbottom=False)

            ax1.legend()
        
            ax2.set_title("B")
            ax2.plot(tt_time_py, residual, label=f"Residual ({observed_label} - {predicted_label})", color='red', linewidth=0.8)

            max_residual = np.nanmax(residual) + 0.2
            min_residual = np.nanmin(residual) + 0.2
            for timer, (date, row) in enumerate(df_storms_cleaned.iterrows()):
                letter = row['LetterDesignation']
                storm = row['StormDesignation']
                
                # Plot the letter at the date
                if timer == 0: 
                    ax2.text(date, max_residual-0.1, letter, fontsize=12, ha='center', va='center', label = 'Storm Designation')
                else:
                    ax2.text(date, max_residual-0.1, letter, fontsize=12, ha='center', va='center')
                ax2.plot([date, date], [max_residual-0.2, 0], linestyle='--', color='k', linewidth=0.5, alpha=0.6)


            ax2.axhline(0, linestyle='--', color='k', linewidth=0.5)
            ax2.set_ylabel("Storm Surge Residual [m]")
            ax2.set_xlabel("Time")
            ax2.set_ylim([min_residual - 0.4,max_residual+0.4 ])
            ax2.xaxis.set_major_locator(DayLocator(interval=3))  # every other day
            ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            tt_time_py_plus15 = tt_time_py[-1] + pd.Timedelta(days=0)

            ax2.set_xlim([tt_time_py[0], tt_time_py_plus15])
            # Create custom legend handles
            # import matplotlib.patches as mpatches
            # legend_handles = [
            #     mpatches.Patch(label=f"{row['LetterDesignation']} – {row['StormDesignation']}")
            #     for _, row in df_storms_cleaned.iterrows()
            # ]
        
            # ax2.legend(handles=legend_handles, title="Storm Events", loc='upper right', bbox_to_anchor=(0.5, -0.25), ncol=3)
            # ax2.legend( loc='upper right')
            # Create legend handles for storm letters
            from matplotlib.lines import Line2D

            storm_legend_handles = [
                Line2D([0], [0], color='k', linestyle='None', marker='',
                       label=f"{row['LetterDesignation']} – {row['StormDesignation']}")
                for _, row in df_storms_cleaned.iterrows()
            ]
            
            # Add standard residual legend + storm legend
            ax2.legend(
                handles=[Line2D([], [], color='red', label=f"Residual ({observed_label} - {predicted_label})")]
                        + storm_legend_handles,
                loc='upper center', bbox_to_anchor=(0.5, -0.35),
                ncol=4, fontsize=9, title="Legend", title_fontsize=10,
                frameon=True
            )
            # === Map Panel (Shapefile Coastline Only) ===
            extent = [-3.65, -2.75, 53.20, 54.52]
            ax_map.set_extent(extent, crs=ccrs.PlateCarree())
        
            # Load and plot shapefile (no fill, just line)
            coast_path = Path(start_path) / 'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_KENT_EPSG_4326_clipped_med_domain.shp'
            coast_gdf = gpd.read_file(coast_path)
            coast_gdf.plot(ax=ax_map, edgecolor='black', linewidth=0.8, facecolor='none', transform=ccrs.PlateCarree())
        
            # Plot point
            ax_map.plot(lon, lat, 'ro', markersize=6, transform=ccrs.PlateCarree(), zorder=3)
        
            ax_map.set_title("C", fontsize=10)
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            ax_map.set_xlabel(f'Lon={lon}, Lat={lat}')

        
            plt.tight_layout()
            plt.savefig(Path(fig_path) / 'storm_surge' / saveas, dpi = 300, bbox_inches='tight', pad_inches=0.3)
            plt.close()
        # Plot a figure of model and ttide with storm surge at any point. 
        # plot_tidal_predictions_and_storm_surge(point = output_prim[1])
        # Plot out a Dee figure
        plot_tidal_predictions_and_storm_surge_with_map(point = output_prim[1], label = 'prim', saveas = 'Dee_storm_surge_at_one_point.png')
        
        
#%%
        self.ttide_prim = output_prim
        self.ttide_ukc4 = output_ukc4
        self.ssh_prim = sh_prim
        self.ssh_ukc4 = sh_ukc4

        def make_max_residual_da(output, sh_prim, label):
            ny, nx = sh_prim.sizes['y'], sh_prim.sizes['x']
            max_resid_grid = np.full((ny, nx), np.nan)
            for item in output:
                y, x = item['y'], item['x']
                max_resid_grid[y, x] = item['max_residual']
                
            return xr.DataArray(
                data=max_resid_grid,
                dims=('y', 'x'),
                coords={
                    'y': sh_prim.coords['y'],
                    'x': sh_prim.coords['x'],
                    'lat': (('y', 'x'), sh_prim['nav_lat'].values),
                    'lon': (('y', 'x'), sh_prim['nav_lon'].values)
                },
                attrs={
                    'long_name': f'Max residual ({label})',
                    'units': 'm'
                },
                name=f'max_residual_{label.lower()}'
            )
        
        def make_residual_da_from_prediction(output, sh_prim, label):
            time_dim = 'time_primea'  # match your coordinate name
            nt, ny, nx = sh_prim.sizes[time_dim], sh_prim.sizes['y'], sh_prim.sizes['x']
            residual_grid = np.full((nt, ny, nx), np.nan)
        
            for item in output:
                y, x = item['y'], item['x']
                predicted = np.array(item['predicted'])  # shape (nt,)
                observed = sh_prim[:, y, x].values       # shape (nt,)
                residual = observed - predicted
                residual_grid[:, y, x] = residual
        
            return xr.DataArray(
                data=residual_grid,
                dims=(time_dim, 'y', 'x'),
                coords={
                    time_dim: sh_prim.coords[time_dim],
                    'y': sh_prim.coords['y'],
                    'x': sh_prim.coords['x'],
                    'lat': (('y', 'x'), sh_prim['nav_lat'].values),
                    'lon': (('y', 'x'), sh_prim['nav_lon'].values)
                },
                attrs={
                    'long_name': f'Sea surface height residuals ({label})',
                    'units': 'm'
                },
                name=f'residuals_{label.lower()}'
            )


        # I think it is safe to say these are the storm surge residuals for this bit.
        resid_prim = make_residual_da_from_prediction(output_prim, sh_prim, 'PRIM')
        resid_ukc4 = make_residual_da_from_prediction(output_ukc4, sh_prim, 'UKC4')


        # Here we will now begin to replace with the other datasets. 
        # self.resid_prim = resid_prim
        # self.resid_ukc4 = resid_ukc4
        
        self.resid_prim = remapped_prim_tide
        self.resid_ukc4 = remapped_ukc4_tide
        # Mean difference
        mean_diff = (resid_prim - resid_ukc4).mean(dim='time_primea')
        # RMSE
        rmse = ((resid_prim - resid_ukc4) ** 2).mean(dim='time_primea') ** 0.5
                
        # --- Create DataArrays ---
        max_resid_prim = make_max_residual_da(output_prim, sh_prim, 'PRIM')
        max_resid_ukc4 = make_max_residual_da(output_ukc4, sh_prim, 'UKC4')
        diff = max_resid_prim - max_resid_ukc4
        diff.name = 'max_residual_difference'
        diff.attrs['long_name'] = 'PRIM - UKC4 Max Residual Difference'
        diff.attrs['units'] = 'm'
        
        # --- Plotting function ---
        def plot_residual_map(da, title, cmap='Reds', vmin=None, vmax=None):
            fig, ax = plt.subplots(figsize=(5, 7))
            ax.set_facecolor('lightgrey')
        
            pcm = da.plot(
                ax=ax,
                x='lon',
                y='lat',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                rasterized=True  # helpful for vector exports
            )
        
            # Add colorbar with consistent formatting
            cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
            cbar.set_label('Storm Surge Residual [m]')
            if vmin is not None and vmax is not None:
                cbar.set_ticks(np.linspace(vmin, vmax, 9))
        
            # Axis formatting
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim([-3.58, -2.8])  # same as your other plot
            # ax.set_title(title)
            
            plt.tight_layout()
            plt.savefig(Path(fig_path) / f'{title}.png', dpi=300)
            plt.close()
        
        # --- Plot all three ---
        global_max = np.nanmax([max_resid_prim.max(), max_resid_ukc4.max()])
        global_min = 0  # residuals are non-negative
        plot_residual_map(max_resid_prim, 'PRIM_Max_Storm_Surge_Residual', vmin=global_min, vmax=global_max)
        plot_residual_map(max_resid_ukc4, 'UKC4_Max_Storm_Surge_Residual', vmin=global_min, vmax=global_max)
        global_max = np.nanmax([diff.max()])
        plot_residual_map(diff, 'Difference_PRIM-UKC4_Max_Residual_Storm_Surge', cmap='bwr', vmin=-global_max, vmax=global_max)
        plot_residual_map(mean_diff, title='Mean_Residual_Difference_PRIMEA-UKC4', cmap='bwr', vmin=-1, vmax=1)
        plot_residual_map(rmse, title='RMSE_Between_PRIMEA_and_UKC4', cmap='viridis', vmin=0, vmax=1)

#%%
    def storm_surge_transects(self):
        resid_prim = self.resid_prim[100:,:,:]
        resid_ukc4 = self.resid_ukc4[100:,:,:]
        transect_data = self.transect_data
        
        distances = self.transect_distances
        indicies = self.transect_indicies
        
        grid_shape = (resid_prim.sizes['y'], resid_prim.sizes['x'])
        yx_pairs = [divmod(i[0], grid_shape[1]) for i in indicies]  # list of (y, x) for each point

        ts_prim = np.array([resid_prim[:, y, x].values for (y, x) in yx_pairs]).T  # shape (time, points)
        ts_ukc4 = np.array([resid_ukc4[:, y, x].values for (y, x) in yx_pairs]).T
        ts_diff = ts_prim - ts_ukc4
        time_vector = pd.to_datetime(resid_prim['time_primea'].values)
        distance_vector = transect_data['distance'].values / 1000  # km
        x_vector = transect_data['x']
        y_vector = transect_data['y']
        distances_km = transect_data['distance'].values / 1000  # shape: (points,)

        # Plotting per estuary
        estuary_ids = transect_data['id'].values
        estuary_names = transect_data['est_name'].values
    
    
        fig_path_storm = Path(fig_path) / 'storm_surge'
        fig_path_storm.mkdir(parents=True, exist_ok=True)
      
        for est_id in np.unique(estuary_ids):
            mask = estuary_ids == est_id
            est_name = estuary_names[mask][0].capitalize()
        
            idx_all = np.array([i[0] for i in indicies])[mask]
            dist_subset = distance_vector[mask]
            x_subset = x_vector[mask]
            y_subset = y_vector[mask]
            prim_subset = ts_prim[:, mask].T
            ukc4_subset = ts_ukc4[:, mask].T
            diff_subset = prim_subset - ukc4_subset
        
            # Deduplicate based on UKC4 grid cell index
            unique_idx, unique_mask = np.unique(idx_all, return_index=True)
            # dist_subset = dist_subset[unique_mask]
            # prim_subset = prim_subset[unique_mask, :]
            # ukc4_subset = ukc4_subset[unique_mask, :]
            # diff_subset = diff_subset[unique_mask, :]
        
            for label, data, cmap, vmin, vmax in [
                ('PRIM', prim_subset, 'Reds', 0, np.nanmax(ts_prim)),
                ('UKC4', ukc4_subset, 'Blues', 0, np.nanmax(ts_ukc4)),
                ('Difference', diff_subset, 'bwr', np.nanmin(diff_subset), np.nanmax(diff_subset)),
            ]:
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                bound = max(abs(data_min), abs(data_max))

                fig, ax = plt.subplots(figsize=(10, 5))
                
                if label == 'Difference':
                    up = +bound
                    down = -bound
                else:
                    up = vmax
                    down = vmin
                pcm = ax.pcolormesh(
                    time_vector,
                    dist_subset,
                    data,
                    shading='auto',
                    cmap=cmap,
                    vmin=down,
                    vmax=up
                )
                
                cbar = fig.colorbar(pcm, ax=ax)
                cbar.set_label('Storm Surge Residual [m]')
                # ax.set_title(f'{label} Hovmöller - {est_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Distance Along Estuary [km]')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                plt.tight_layout()
        
                out_file = fig_path_storm / f"{label}_Hovmoller_{est_name}.png"
                plt.savefig(out_file, dpi=300)
                plt.close()


        rolling_window = 24
        # Loop through estuaries again for rolling-mean plots
        for est_id in np.unique(estuary_ids):
            mask = estuary_ids == est_id
            est_name = estuary_names[mask][0].capitalize()
        
            idx_all = np.array([i[0] for i in indicies])[mask]
            dist_subset = distance_vector[mask]
            prim_subset = ts_prim[:, mask].T
            ukc4_subset = ts_ukc4[:, mask].T
            diff_subset = prim_subset - ukc4_subset
        
            # Deduplicate using UKC4 grid indices
            unique_idx, unique_mask = np.unique(idx_all, return_index=True)
            # dist_subset = dist_subset[unique_mask]
            # prim_subset = prim_subset[unique_mask, :]
            # ukc4_subset = ukc4_subset[unique_mask, :]
            # diff_subset = diff_subset[unique_mask, :]
        
            # Apply 14-day rolling mean (window size in timesteps)
            prim_rolling = pd.DataFrame(prim_subset).rolling(window=rolling_window, axis=1, min_periods=1).mean().values
            ukc4_rolling = pd.DataFrame(ukc4_subset).rolling(window=rolling_window, axis=1, min_periods=1).mean().values
            diff_rolling = prim_rolling - ukc4_rolling
        
            for label, data, cmap, vmin, vmax in [
                ('PRIM', prim_rolling, 'Reds', 0, np.nanmax(ts_prim)),
                ('UKC4', ukc4_rolling, 'Blues', 0, np.nanmax(ts_ukc4)),
                ('Difference', diff_rolling, 'bwr', np.nanmin(diff_rolling), np.nanmax(diff_rolling)),
            ]:
                
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                bound = max(abs(data_min), abs(data_max))

                fig, ax = plt.subplots(figsize=(10, 5))
                
                if label == 'Difference':
                    up = +bound
                    down = -bound
                else:
                    up = vmax
                    down = vmin
              
                pcm = ax.pcolormesh(
                    time_vector,
                    dist_subset,
                    data,
                    shading='auto',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                cbar = fig.colorbar(pcm, ax=ax)
                cbar.set_label('Storm Surge Residual [m]')
                # ax.set_title(f'Rolling Mean ({rolling_window}-hour) - {label} Hovmöller - {est_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Distance Along Estuary [km]')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                plt.tight_layout()
        
                out_file = fig_path_storm / f"rolling_mean_{rolling_window}_{label}_Hovmoller_{est_name}.png"
                plt.savefig(out_file, dpi=300)
                plt.close()
                
        rolling_window = 24
        # Loop through estuaries again for rolling-mean plots
        for est_id in np.unique(estuary_ids):
            mask = estuary_ids == est_id
            est_name = estuary_names[mask][0].capitalize()
        
            idx_all = np.array([i[0] for i in indicies])[mask]
            dist_subset = distance_vector[mask]
            prim_subset = ts_prim[:, mask].T
            ukc4_subset = ts_ukc4[:, mask].T
            diff_subset = prim_subset - ukc4_subset
        
            # Deduplicate using UKC4 grid indices
            unique_idx, unique_mask = np.unique(idx_all, return_index=True)
            # dist_subset = dist_subset[unique_mask]
            # prim_subset = prim_subset[unique_mask, :]
            # ukc4_subset = ukc4_subset[unique_mask, :]
            # diff_subset = diff_subset[unique_mask, :]
        
            # Apply 14-day rolling mean (window size in timesteps)
            prim_rolling = pd.DataFrame(prim_subset).rolling(window=rolling_window, axis=1, min_periods=1).max(skipna=True).values
            ukc4_rolling = pd.DataFrame(ukc4_subset).rolling(window=rolling_window, axis=1, min_periods=1).max(skipna=True).values
            diff_rolling = prim_rolling - ukc4_rolling
        
            for label, data, cmap, vmin, vmax in [
                ('PRIM', prim_rolling, 'Reds', 0, np.nanmax(ts_prim)),
                ('UKC4', ukc4_rolling, 'Blues', 0, np.nanmax(ts_ukc4)),
                ('Difference', diff_rolling, 'bwr', np.nanmin(diff_rolling), np.nanmax(diff_rolling)),
            ]:
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                bound = max(abs(data_min), abs(data_max))

                fig, ax = plt.subplots(figsize=(10, 5))
                
                if label == 'Difference':
                    up = +bound
                    down = -bound
                else:
                    up = vmax
                    down = vmin
                pcm = ax.pcolormesh(
                    time_vector,
                    dist_subset,
                    data,
                    shading='auto',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                cbar = fig.colorbar(pcm, ax=ax)
                cbar.set_label('Storm Surge Residual [m]')
                # ax.set_title(f'Rolling Max ({rolling_window}-hour) - {label} Hovmöller - {est_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Distance Along Estuary [km]')
                
                ax.set_yticks(dist_subset)
                ax.grid(which='major', axis='y', linestyle='--', alpha=0.3)
                
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                plt.tight_layout()
        
                out_file = fig_path_storm / f"rolling_max_{rolling_window}_{label}_Hovmoller_{est_name}.png"
                plt.savefig(out_file, dpi=300)
                plt.close()
                
                
    def skew_surge(self):
        # These skew surge needs to be readjusted because of the shape of the data. 
        
        # ttide_prim = self.ttide_prim
        # ttide_ukc4 = self.ttide_ukc4
        
        # Possibly naming convenstions are the issue here. 
        ttide_prim = self.resid_prim[100:,:,:]
        ttide_ukc4 = self.resid_ukc4[100:,:,:]
        ssh_prim = self.ssh_prim[100:,:,:]
        ssh_ukc4 = self.ssh_ukc4[100:,:,:]
        
        # point_in_space = 100
        # x = ttide_prim[point_in_space]['x']
        # y = ttide_prim[point_in_space]['y']
        x = 5
        y = 5
        
        # Transect stuff 
        resid_prim = self.resid_prim
        transect_data = self.transect_data
        
        indicies = self.transect_indicies
        
        grid_shape = (resid_prim.sizes['y'], resid_prim.sizes['x'])
        yx_pairs = [divmod(i[0], grid_shape[1]) for i in indicies]  # list of (y, x) for each point
        distance_vector = transect_data['distance'].values / 1000  # km

        # Plotting per estuary
        estuary_ids = transect_data['id'].values
        estuary_names = transect_data['est_name'].values
    
    
        fig_path_storm = Path(fig_path) / 'storm_surge'
        fig_path_storm.mkdir(parents=True, exist_ok=True)
        
        def recombine_ttide_predition(output, sh_prim, label, varname='predicted'):
            """
            Reconstruct a full 3D grid DataArray (time, y, x) from sparse prediction output.
    
            Parameters
            ----------
            output : list of dicts
                Each dict should have keys 'y', 'x', and the variable name (e.g., 'predicted').
            sh_prim : xarray.DataArray
                Reference DataArray with correct dimensions and coordinates.
            label : str
                Label to identify the variable in metadata.
            varname : str, default='predicted'
                The key in the `output` dicts to extract.
        
            Returns
            -------
            xarray.DataArray
                Reconstructed variable as a full-sized DataArray.
            """
            time_dim = 'time_primea'
            nt, ny, nx = sh_prim.sizes[time_dim], sh_prim.sizes['y'], sh_prim.sizes['x']
            full_grid = np.full((nt, ny, nx), np.nan)
        
            for item in output:
                y, x = item['y'], item['x']
                values = np.array(item[varname])  # e.g., predicted tide
                full_grid[:, y, x] = values
        
            return xr.DataArray(
                data=full_grid,
                dims=(time_dim, 'y', 'x'),
                coords={
                    time_dim: sh_prim.coords[time_dim],
                    'y': sh_prim.coords['y'],
                    'x': sh_prim.coords['x'],
                    'lat': (('y', 'x'), sh_prim['nav_lat'].values),
                    'lon': (('y', 'x'), sh_prim['nav_lon'].values)
                },
                attrs={
                    'long_name': f'{label} ({varname})',
                    'units': sh_prim.attrs.get('units', 'unknown')
                },
                name=f'{varname}_{label.lower()}'
            )
        
        # test_example_ttide_all = recombine_ttide_predition(ttide_prim, ssh_prim, 'ttide_prim', varname='predicted_MSL_adjusted')
        test_example_ttide = ttide_prim[:,x, y]
        test_example_ssh = ssh_prim[:, x, y]
        time = test_example_ssh.time_primea
        
        
        # plt.figure()
        # plt.plot(time, test_example_ttide, label = 'tTide prediction', linewidth = 1)
        # plt.plot(time, test_example_ssh, label = 'PRIMEA surface height', linewidth = 0.5)
        # # plt.xlim([1100, 1400])
        # plt.legend()
        # # plt.close()
        
        # plt.figure()
        # plt.plot(time, test_example_ssh - test_example_ttide, label = 'residual storm surge')
        # plt.legend()
        # plt.close()
        
        def calc_skew_surge(time, pred, total, window_hours=12):
            """
            Calculate skew surge (max observed - max predicted) per tidal cycle.
        
            Parameters
            ----------
            time : xarray.DataArray or pandas.DatetimeIndex
            pred : np.ndarray or DataArray
            total : np.ndarray or DataArray
            window_hours : float
                Approx length of a tidal cycle (default: 12 hours)
        
            Returns
            -------
            pd.DataFrame
                DataFrame with skew surge and time of each tidal cycle
            """
            time = pd.to_datetime(time.values)
            dt = (time[1] - time[0]).total_seconds() / 3600  # hours
            step = int(window_hours / dt)
        
            skew_times = []
            skew_values = []
        
            for i in range(0, len(time) - step, step):
                pred_peak = np.nanmax(pred[i:i+step])
                total_peak = np.nanmax(total[i:i+step])
                skew = total_peak - pred_peak
        
                skew_times.append(time[i + step // 2])
                skew_values.append(skew)
        
            return pd.DataFrame({'time': skew_times, 'skew_surge': skew_values})
        
        def calc_skew_surge_3d(time, pred_3d, obs_3d, window_hours=12):
            """
            Vectorized skew surge calculation over full 3D arrays.
        
            Parameters
            ----------
            time : array-like of datetime64
                Time array of shape (nt,)
            pred_3d : ndarray
                Predicted tide from T_TIDE, shape (nt, ny, nx)
            obs_3d : ndarray
                Observed surface height, shape (nt, ny, nx)
            window_hours : int
                Tidal cycle duration to sample over (default = 12h)
        
            Returns
            -------
            skew_surge : ndarray
                Skew surge values of shape (n_cycles, ny, nx)
            skew_time : np.ndarray of datetime64
                Central times of tidal cycles
            """
        
            # Time handling
            time = pd.to_datetime(time)
            dt = (time[1] - time[0]).total_seconds() / 3600
            step = int(window_hours / dt)
            nt, ny, nx = pred_3d.shape
            n_cycles = (nt - step) // step
        
            # Output array
            skew_surge = np.full((n_cycles, ny, nx), np.nan)
            skew_time = []
        
            for i in range(n_cycles):
                t_start = i * step
                t_end = t_start + step
                window_pred = pred_3d[t_start:t_end, :, :]
                window_obs = obs_3d[t_start:t_end, :, :]
        
                max_pred = np.nanmax(window_pred, axis=0)
                max_obs = np.nanmax(window_obs, axis=0)
        
                skew = max_obs - max_pred
                skew_surge[i, :, :] = skew
                skew_time.append(time[t_start + step // 2])
        
            return skew_surge, np.array(skew_time)
                
        # Compute skew surge for one example
        df_skew = calc_skew_surge(time, test_example_ttide, test_example_ssh)
        
        # compute skew surge for all points. 
        df_all_skew, df_time = calc_skew_surge_3d(time, ttide_prim, ssh_prim)

        threshold_95 = np.nanpercentile(df_skew['skew_surge'], 95)
        #%%
        #!!!!!!
        # Plot
        max_residual = np.nanmax(df_skew['skew_surge'])
        min_residual = np.nanmin(df_skew['skew_surge'])
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(df_skew['time'], df_skew['skew_surge'], marker='o', linestyle='-', label='Skew surge')
        # plt.axhline(0.3, color='red', linestyle='--', label='Significant surge threshold (0.3 m)')
        plt.axhline(threshold_95, color='red', linestyle='--', label=f'95th percentile [{threshold_95:.2f} m]')


        for timer, (date, row) in enumerate(df_storms_cleaned.iterrows()):
            letter = row['LetterDesignation']
            storm = row['StormDesignation']
            
            # Plot the letter at the date
            if timer == 0: 
                plt.text(date, max_residual+0.12, letter, fontsize=12, ha='center', va='center', label = 'Storm Designation')
            else:
                plt.text(date, max_residual+0.12, letter, fontsize=12, ha='center', va='center')
            plt.plot([date, date], [max_residual-0.01, min_residual-0.1], linestyle='--', color='k', linewidth=1, alpha=0.6)
        plt.title(f"Skew Surge at Point ({x}, {y})")
        plt.ylabel("Skew Surge [m]")
        plt.xlabel("Time")
        plt.ylim([min_residual-0.1, max_residual +0.4])
        plt.grid(True)
        
        # Handle legend
        from matplotlib.lines import Line2D

        # Create handles for storm designation legend entries
        storm_legend_handles = [
            Line2D([0], [0], color='k', linestyle='None', marker='',
                   label=f"{row['LetterDesignation']} – {row['StormDesignation']}")
            for _, row in df_storms_cleaned.iterrows()
        ]
        
        # Combine with standard plot handles
        main_handles = [
            Line2D([], [], color='blue', linestyle='-', label='Skew surge'),
            Line2D([], [], color='red', linestyle='--', label=f'95th percentile [{threshold_95:.2f} m]')
        ]
        
        # Create combined legend
        plt.legend(
            handles=main_handles + storm_legend_handles,
            title='Legend',
            loc='upper center',
            bbox_to_anchor=(0.5, -0.35),
            ncol=5,
            fontsize=9,
            title_fontsize=10,
            frameon=False
        )
        # plt.tight_layout()
        # Set daily ticks and rotate
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))  # one per day
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # or '%d %b'
        plt.xticks(rotation=45, ha='right')  # 45° downward or use rotation=90 for vertical

        plt.show()
        
        # %% For the rest of them 
        # Prepare tide prediction across the domain
        predicted_all = ttide_prim
        
        # Preallocate skew surge array
        time_ref = pd.to_datetime(ssh_prim['time_primea'].values)
        dt_hours = (time_ref[1] - time_ref[0]).total_seconds() / 3600
        window_hours = 12
        step = int(window_hours / dt_hours)
        n_cycles = (len(time_ref) - step) // step
        
        skew_surge_matrix = np.full((len(yx_pairs), n_cycles), np.nan)
        
        # First, dynamically determine n_cycles from the first working point
        # This will make a skew surge per estuary point. 
        first_valid_skew = None
        for i, (y, x) in enumerate(yx_pairs):
            pred_ts = predicted_all[:, y, x].values
            total_ts = ssh_prim[:, y, x].values
            try:
                df_skew = calc_skew_surge(time_ref, pred_ts, total_ts)
                if first_valid_skew is None:
                    first_valid_skew = df_skew
                    n_cycles = len(df_skew)
                    skew_surge_matrix = np.full((len(yx_pairs), n_cycles), np.nan)
                if len(df_skew) != n_cycles:
                    print(f"Skipping point ({y}, {x}) due to mismatched skew surge length: {len(df_skew)} vs expected {n_cycles}")
                    continue
                skew_surge_matrix[i, :] = df_skew['skew_surge'].values
            except Exception as e:
                print(f"Skipping point ({y}, {x}) due to error: {e}")
        

        skew_time_pd = pd.to_datetime(first_valid_skew['time'].values)

        x_ = []
        y_ = []
        log_known_good_bad_est_points = {}
        for est_id in np.unique(estuary_ids):
            mask = estuary_ids == est_id
            est_name = estuary_names[mask][0].capitalize()
            log_known_good_bad_est_points[est_name] = {}
            idx_all = np.array([i[0] for i in indicies])[mask]
            dist_subset = distance_vector[mask]
            skew_subset = skew_surge_matrix[mask, :]  # shape: (points, skew_time)
        
            # Deduplicate
            unique_idx, unique_mask = np.unique(idx_all, return_index=True)
            dist_subset = dist_subset[unique_mask]
            skew_subset = skew_subset[unique_mask, :]
            log_known_good_bad_est_points[est_name]
            
            # Determine the transect points
            estmask = estuary_ids == est_id
            est_indices = np.array([i[0] for i in indicies])[estmask]
            est_pairs = [divmod(i, grid_shape[1]) for i in est_indices]
        
            # Deduplicate based on UKC4 index
            _, unique_mask = np.unique(est_indices, return_index=True)
            est_yx_pairs = [est_pairs[i] for i in unique_mask]
            
            def save_transect_locations():            
                x, y = zip(*est_yx_pairs)
                x_.append(x)
                y_.append(y)
            save_transect_locations()
                
            for indexer, [y,x] in enumerate(est_yx_pairs):
                log_known_good_bad_est_points[est_name][str(indexer)] = {}
                ssh_point = ssh_prim[:, y, x].values
                ttide_point = predicted_all[:, y, x].values
                
                ssh_mean = np.nanmean(ssh_point)
                ttide_mean = np.nanmean(ttide_point)
                log_known_good_bad_est_points[est_name][str(indexer)]['ssh'] = ssh_mean
                log_known_good_bad_est_points[est_name][str(indexer)]['ttide'] = ttide_mean
                
            if est_name == 'Lune':
                # Get yx grid coordinates for this estuary
                lune_mask = estuary_ids == est_id
                lune_indices = np.array([i[0] for i in indicies])[lune_mask]
                lune_yx_pairs = [divmod(i, grid_shape[1]) for i in lune_indices]
            
                # Deduplicate based on UKC4 index
                _, unique_mask = np.unique(lune_indices, return_index=True)
                lune_yx_pairs = [lune_yx_pairs[i] for i in unique_mask]
                            
                i1, i3 = 0, 2
                (y1, x1) = lune_yx_pairs[i1]
                (y3, x3) = lune_yx_pairs[i3]
                # Extract time series at both points
                time = pd.to_datetime(ssh_prim['time_primea'].values)
                observed_1 = ssh_prim[:, y1, x1].values
                observed_3 = ssh_prim[:, y3, x3].values
                predicted_1 = predicted_all[:, y1, x1].values
                predicted_3 = predicted_all[:, y3, x3].values
            
    
            
                
                # Plot observed and predicted at both points
                plt.figure(figsize=(12, 6))
                # plt.plot(time, observed_1, label='Obs (Row 1)', linewidth=0.7)
                # plt.plot(time, predicted_1, label='Tide (Row 1)', linestyle='--')
                plt.plot(time, observed_3, label='Obs (Row 3)', linewidth=0.7)
                plt.plot(time, predicted_3, label='Tide (Row 3)', linestyle='--')
                plt.title("Predicted vs Observed Tide - Lune Estuary (Rows 1 & 3)")
                plt.xlabel("Time")
                plt.ylabel("Surface Height [m]")
                plt.grid(True)
                plt.legend()
                # plt.xlim([time[200], time[550]])
                plt.tight_layout()
                plt.show()
                #%%
        def plot_ssh_vs_ttide(estuary_data, estuary_name):
            """
            Plot observed SSH vs ttide-predicted mean/median values for one estuary.
            
            Parameters
            ----------
            estuary_data : dict
                Dictionary with keys as point numbers and values as {'ssh': ..., 'ttide': ...}
            estuary_name : str
                Name of the estuary (used in plot title)
            """
            # Sort by point index (as string keys)
            point_ids = sorted(estuary_data.keys(), key=lambda x: int(x))
            
            # Extract values
            ssh_vals   = [estuary_data[pt]['ssh'] for pt in point_ids]
            ttide_vals = [estuary_data[pt]['ttide'] for pt in point_ids]
            indices    = [int(pt) + 1 for pt in point_ids]
            
            plt.figure(figsize=(10, 5))
            plt.plot(indices, ssh_vals, marker='o', label='Observed Mean SSH')
            plt.plot(indices, ttide_vals, marker='x', label='T_TIDE Predicted Mean')
            plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            plt.xlabel("Point Index (0 = Estuary Mouth → Inland)")
            plt.ylabel("Mean Sea Surface Height [m]")
            plt.title(f"{estuary_name} – Observed vs Predicted Tidal Means")
            plt.legend()
            plt.ylim([-0.5, 1.5])
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        for estuary, data in log_known_good_bad_est_points.items():
            plot_ssh_vs_ttide(data, estuary)
            
        plt.figure()
        plt.pcolor(ssh_prim[100,:,:])
        for x, y in zip(x_, y_):
            plt.scatter(y, x)
     

        # x_all = [y for sublist in y_ for y in sublist]
        # y_all = [x for sublist in x_ for x in sublist]
        
        # # Convert index coordinates to lon/lat
        # lon_pts = [ssh_prim.nav_lon.values[y, x] for x, y in zip(x_all, y_all)]
        # lat_pts = [ssh_prim.nav_lat.values[y, x] for x, y in zip(x_all, y_all)]

        def hovmoler_skew_surge_plot():
            
             
            distances = self.transect_distances
            indicies = self.transect_indicies
        
            ny, nx = df_all_skew.shape[1], df_all_skew.shape[2]
            grid_shape = (ny, nx)
            yx_pairs = [divmod(i[0], grid_shape[1]) for i in indicies]  # list of (y, x) for each point

            ts_skew = np.array([df_all_skew[:, y, x] for (y, x) in yx_pairs]).T 
            x_vector = transect_data['x']
            y_vector = transect_data['y']
            time_vector = skew_time_pd[:-1]
            
            for est_id in np.unique(estuary_ids):
                mask = estuary_ids == est_id
                est_name = estuary_names[mask][0].capitalize()
                print(est_name)
                idx_all = np.array([i[0] for i in indicies])[mask]
                dist_subset = distance_vector[mask]
                x_subset = x_vector[mask]
                y_subset = y_vector[mask]
                diff_subset = ts_skew[:, mask].T
            
                # Deduplicate based on UKC4 grid cell index
                unique_idx, unique_mask = np.unique(idx_all, return_index=True)
                # dist_subset = dist_subset[unique_mask]
                # prim_subset = prim_subset[unique_mask, :]
                # ukc4_subset = ukc4_subset[unique_mask, :]
                # diff_subset = diff_subset[unique_mask, :]
            
                for label, data, cmap, vmin, vmax in [
                    ('Skew_Surge', diff_subset, 'bwr', np.nanmin(diff_subset), np.nanmax(diff_subset)),
                ]:
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)
                    bound = max(abs(data_min), abs(data_max))
                    vmin, vmax = -bound, +bound
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Identify rows where all values are NaN
                    valid_mask = ~np.all(np.isnan(data), axis=1)
                    
                    # Apply mask to both dist_subset and data
                    dist_subset_cleaned = dist_subset[valid_mask]
                    data_cleaned = data[valid_mask, :]
                    if label == 'Difference':
                        up = +bound
                        down = -bound
                    else:
                        up = vmax
                        down = vmin
                        pcm = ax.pcolormesh(
                        time_vector,
                        dist_subset_cleaned,
                        data_cleaned,
                        shading='auto',
                        cmap=cmap,
                        vmin=down,
                        vmax=up
                    )
                    
                    cbar = fig.colorbar(pcm, ax=ax)
                    cbar.set_label('$\Delta$IRENE Skew Storm Surge Residual [m]')
                    # ax.set_title(f'{label} Hovmöller - {est_name}')
                    # ax.set_title(f'{est_name}')

                    ax.set_xlabel('Time')
                    ax.set_ylabel('Distance Along Estuary [km]')
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    fig.autofmt_xdate()
                    plt.tight_layout()
            
                    out_file = fig_path_storm / f"{label}_Hovmoller_{est_name}.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close()
            
      #%% 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.dates as mdates
        # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import cartopy.crs as ccrs
        import os
        
        def hovmoler_skew_surge_plot_with_wind(self):
            import geopandas as gpd
            import numpy as np
            import pandas as pd
            import xarray as xr
            from pathlib import Path
            store_skew_storm_surge = {}
            data_proc_path = Path(data_stats_path).parent / 'data_proc'

        
            estuary_bounds = {
                'Esk': [-3.45, -3.30, 54.27, 54.40],
                'Duddon': [-3.37, -3.16, 54.12, 54.29],
                'Leven': [-3.15, -2.95, 54.08, 54.30],
                'Kent': [-2.95, -2.75, 54.09, 54.30],
                'Lune': [-2.95, -2.78, 53.93, 54.07],
                'Wyre': [-3.06, -2.80, 53.83, 54.00],
                'Ribble': [-3.15, -2.65, 53.62, 53.80],
                'Alt': [-3.10, -2.90, 53.45, 53.65],
                'Mersey': [-3.20, -2.55, 53.27, 53.55],
                'Dee': [-3.40, -2.95, 53.17, 53.45],
                'Clywd': [-3.50, -3.25, 53.20, 53.35],
            }
        
            self.estuary_bounds = estuary_bounds
        
            ukc3_example_grid = Path(start_path) / "Original_Data/UKC3/oa/shelftmb/UKC4ao_1h_20131030_20131030_shelftmb_grid_T.nc"
            kent_poly_path = Path(start_path) / "modelling_DATA/kent_estuary_project/5.Final/QGIS/kent_area_poly.shp"
            outline_path = Path(start_path) / "modelling_DATA/kent_estuary_project/5.Final/QGIS/kent_area_poly_as_lines.shp"
            def UKC3_area(first_file):
                ds = xr.open_dataset(first_file)
                z = ds["sossheig"].isel(time_counter=0).values
                x = ds["nav_lon"].values
                y = ds["nav_lat"].values
                return x, y, z
        
            x, y, z = UKC3_area(ukc3_example_grid)
            outline = gpd.read_file(outline_path)
        
            distances = self.transect_distances
            indicies = self.transect_indicies
            ny, nx = df_all_skew.shape[1], df_all_skew.shape[2]
            grid_shape = (ny, nx)
            yx_pairs = [divmod(i[0], grid_shape[1]) for i in indicies]
            ts_skew = np.array([df_all_skew[:, y, x] for (y, x) in yx_pairs]).T
            x_vector = transect_data['x']
            y_vector = transect_data['y']
            time_vector = skew_time_pd[:-1]
        
            store_td_invalid = {}
            store_td_valid = {}
            for est_id in np.unique(estuary_ids):
                mask = estuary_ids == est_id
                est_name = estuary_names[mask][0].capitalize()
                est_bounds = estuary_bounds[est_name]
                store_skew_storm_surge[est_name] = {}
                dist_subset = distance_vector[mask]
                diff_subset = ts_skew[:, mask].T
                td = transect_data[transect_data["est_name"].str.lower() == est_name.lower()]
                store_td_invalid[est_name] = {}
                store_td_valid[est_name] = {}
                for label, data, cmap in [('Skew_Surge', diff_subset, 'bwr')]:
                    valid_mask = ~np.all(np.isnan(data), axis=1)
                    dist_subset_cleaned = dist_subset[valid_mask]
                    data_cleaned = data[valid_mask, :]
                    td_valid = td.loc[valid_mask]
                    td_invalid = td[~valid_mask]
        
                    store_td_invalid[est_name] = td_invalid
                    store_td_valid[est_name] = td_valid
                    vmin = -np.nanmax(np.abs(data_cleaned))
                    vmax = -vmin
        #%
                    fig = plt.figure(figsize=(14, 10))
                    gs = gridspec.GridSpec(
                        3, 2,
                        width_ratios=[30, 1],
                        height_ratios=[1, 2, 1.5],
                        hspace=0.1, wspace=0.05
                    )
        
                    # --- Top: Wind Panel ---
                    ax_wind = fig.add_subplot(gs[0, 0])
                    ax_dummy_cb = fig.add_subplot(gs[0, 1])
                    ax_dummy_cb.axis("off")
        
                    wind_csv_path = f"/Volumes/PNC/Original_Data/UKC3/wind/estuary_transect_wind_field/wind_data_{est_name.lower()}.csv"
                    if os.path.exists(wind_csv_path):
                        wind_df = pd.read_csv(wind_csv_path, parse_dates=["datetime"])
                        ax_wind.plot(wind_df["datetime"], wind_df["speed (m/s)"], color='tab:blue', label="Wind Speed (ms$^{-1}$)")
                        ax_wind.set_ylabel("Wind Speed [m/s]", color='tab:blue')
                        ax_wind.tick_params(axis='y', labelcolor='tab:blue')
        
                        ax_dir = ax_wind.twinx()
                        ax_dir.plot(wind_df["datetime"], wind_df["direction (deg)"], color='tab:green', label="Wind Dir (°)", alpha=0.7)
                        ax_dir.set_ylabel("Wind Direction [°]", color='tab:green')
                        ax_dir.tick_params(axis='y', labelcolor='tab:green')
                        ax_dir.set_ylim(0, 360)
                        ax_dir.set_xlim(time_vector[0], time_vector[-1])

        
                        # for deg, label_c in zip([0, 90, 180, 270], ['N', 'E', 'S', 'W']):
                        #     ax_dir.axhline(deg, color='grey', linestyle='--', linewidth=0.5)
                        #     ax_dir.text(wind_df["datetime"].iloc[0], deg + 5, label_c, color='grey', fontsize=8)
        
                        # ax_wind.legend(loc='centre left')
                        # ax_dir.legend(loc='upper right')
                        ax_wind.tick_params(labelbottom=False)
                    else:
                        ax_wind.text(0.5, 0.5, f"No wind data for {est_name}", transform=ax_wind.transAxes, ha='center')
                    ax_wind.set_title('A')
                    # --- Middle: Skew Surge Hovmöller ---
                    ax_skew = fig.add_subplot(gs[1, 0], sharex=ax_wind)
                    pcm = ax_skew.pcolormesh(
                        time_vector, dist_subset_cleaned, data_cleaned,
                        shading='auto', cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    ax_skew.set_ylabel("Distance Along Estuary [km]")
                    ax_skew.set_xlabel("Time")
                    ax_skew.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                    ax_skew.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax_skew.set_title('B')

                    plt.setp(ax_skew.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
                    ax_cbar = fig.add_subplot(gs[1, 1])
                    cbar = fig.colorbar(pcm, cax=ax_cbar)
                    cbar.set_label('$\Delta$IRENE Skew Storm Surge Residual [m]')
        
                    # --- Bottom: Map Inset ---
                    # ax_map = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree())
                    ax_map = fig.add_axes([0.123, 0.01, 0.25, 0.25], projection=ccrs.PlateCarree())  # [left, bottom, width, height] in figure coordinates

                    outline.plot(ax=ax_map, edgecolor='black', facecolor='none', linewidth=0.8)
                    ax_map.scatter(td_valid['x'], td_valid['y'], color='red', s=20, transform=ccrs.PlateCarree())
                    ax_map.scatter(td_invalid['x'], td_invalid['y'], color='green', s=20, transform=ccrs.PlateCarree())
                    ax_map.set_extent(est_bounds, crs=ccrs.PlateCarree())
                    ax_map.set_xticks([]); ax_map.set_yticks([])
                    ax_map.set_title('C')
                    # Set axis labels
                    ax_map.set_xlabel("Longitude", fontsize=10)
                    ax_map.set_ylabel("Latitude", fontsize=10)
                    
                    # Set visible ticks
                    from matplotlib.ticker import MaxNLocator

                    # Set fewer ticks on the inset map
                    ax_map.xaxis.set_major_locator(MaxNLocator(nbins=3))  # max 4 lon ticks
                    ax_map.yaxis.set_major_locator(MaxNLocator(nbins=3))  # max 4 lat ticks

                    
                    # Format tick labels
                    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
                    ax_map.xaxis.set_major_formatter(LongitudeFormatter())
                    ax_map.yaxis.set_major_formatter(LatitudeFormatter())
                    
                    # plt.tight_layout()
                    out_file = fig_path_storm / f"{label}_Hovmoller_{est_name}_with_insets.png"
                    # print(out_file)
                    plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    
                    # Here we will save the storm surge data to make a new type of plot. 
                    store_skew_storm_surge[est_name]['td_valid'] = td_valid
                    store_skew_storm_surge[est_name]['td_invalid'] = td_invalid
                    store_skew_storm_surge[est_name]['wind_df'] = wind_df
                    store_skew_storm_surge[est_name]['time_vector'] = time_vector
                    store_skew_storm_surge[est_name]['dist_subset_cleaned'] = dist_subset_cleaned
                    store_skew_storm_surge[est_name]['data_cleaned'] = data_cleaned
                    store_skew_storm_surge[est_name]['vmin'] = vmin
                    store_skew_storm_surge[est_name]['vmax'] = vmax
                    store_skew_storm_surge[est_name]['cmap'] = cmap
                    store_skew_storm_surge[est_name]['outline'] = outline
                    store_skew_storm_surge[est_name]['est_bounds'] = est_bounds
#%
            self.store_td_invalid = store_td_invalid
            self.store_td_valid = store_td_valid
            # Save up the skew storm surge data 
            print('Storing the skew surge dataframe')
            with open(data_proc_path / 'skew_storm_surge_saved.pkl', 'wb') as file:
                pickle.dump(store_skew_storm_surge, file)
        hovmoler_skew_surge_plot_with_wind(self)
        
            
    #%%
    def salinity_transect_analysis(self):
    #%%
        import cartopy.crs as ccrs
        import geopandas as gpd
        estuary_bounds = self.estuary_bounds
        store_td_invalid = self.store_td_invalid
        store_td_valid = self.store_td_valid
        outline_path = Path(start_path) / "modelling_DATA/kent_estuary_project/5.Final/QGIS/kent_area_poly_as_lines.shp"
        outline = gpd.read_file(outline_path)
        prim_dict_original = self.data_dict['prim']
        ukc4_dict_original = self.data_dict['ukc4']
        transect_data = self.transect_data
        
        transect_distances = self.transect_distances
        transect_indices = self.transect_indicies
       
        save_dir = os.path.join(fig_path, "salinity_transects")
        os.makedirs(save_dir, exist_ok=True)
        estuary_data = {
            'Dee': { 
                'angle'          : 141,
            },
            'Leven': {
                'angle'          : 4,
            },
            'Ribble': {
                'angle'          : 84,
            },
            'Lune': {
                'angle'          : 17,
            },
            'Mersey': {
                'angle'          : 158,
            },
            'Wyre': {
                'angle'          : 180,
            },
            'Kent': {
                'angle'          : 48,
            },
            'Duddon': {
                'angle'          : 63,
            }
        }
        
        # Number of timesteps to skip for spin-up
        skip_steps = 1440  # e.g., 2 months if 1 step = 1 hour
        
        def trim_spinup(data_dict, skip_steps):
            """Return a new dict with the first `skip_steps` timesteps removed for all time-series arrays."""
            trimmed = {}
            for key, arr in data_dict.items():
                try:
                    if arr.ndim >= 1 and arr.shape[0] > skip_steps:
                        trimmed[key] = arr[skip_steps:]
                    else:
                        trimmed[key] = arr  # keep unchanged if not time series
                except AttributeError:
                    trimmed[key] = arr  # not an array, skip
            return trimmed
        
        # Apply trimming
        prim_dict = trim_spinup(prim_dict_original, skip_steps)
        ukc4_dict = trim_spinup(ukc4_dict_original, skip_steps)
        def project_along_estuary(U_east_txy, V_north_txy, angles_deg_per_point):
            """
            U_east_txy, V_north_txy: arrays shape (T, P) after sampling at transect points.
            angles_deg_per_point: shape (P,), degrees CCW from East.
            returns U_along shape (T, P).
            """
            ang = np.deg2rad(np.asarray(angles_deg_per_point))  # (P,)
            phi = np.deg2rad(90.0 - np.asarray(ang))   # shape (P,)
            cosA = np.cos(phi)[None, :]                          # (1, P)
            sinA = np.sin(phi)[None, :]                          # (1, P)
            return U_east_txy * cosA + V_north_txy * sinA
        
        # def check_sign(time, u_along_section_mean, eta_section_mean, name):
        #     dt = np.gradient(time.astype('datetime64[ns]').astype('int64')*1e-9)
        #     deta = np.gradient(eta_section_mean, dt)
        #     flood = deta > 0
        #     sign_ok = np.nanmean(np.sign(u_along_section_mean[flood])) > 0
        #     print(f"{name}: positive during flood? {sign_ok}")
        #     return sign_ok
        # --- sampling at transect points ---
        
        def unravel_indices_for(ds_2d):
            """Return (ny, nx) for a 2D field with dims (y, x) or similar."""
            # Handles DataArray with dims e.g. ('time','y','x') or ('t','y','x')
            ydim = [d for d in ds_2d.dims if d.lower() in ('y', 'eta', 'j')][-1]
            xdim = [d for d in ds_2d.dims if d.lower() in ('x', 'xi', 'i')][-1]
            ny = ds_2d.sizes[ydim]
            nx = ds_2d.sizes[xdim]
            return ny, nx, ydim, xdim
        
        def extract_at_transect(model_dict, transect_indices_flat):
            """
            Returns dict with time, eta(T,P), S(T,P), U(T,P), V(T,P).
            Assumes model_dict keys: 'surface_height','surface_salinity','surface_Uvelocity','surface_Vvelocity'
            """
            Sda = model_dict['surface_salinity']
            Uda = model_dict['surface_Uvelocity']
            Vda = model_dict['surface_Vvelocity']
            Hda = model_dict['surface_height']
        
            time = Sda['time_primea'].values
            ny, nx, ydim, xdim = unravel_indices_for(Sda)
            yi, xi = np.unravel_index(transect_indices_flat, (ny, nx))
        
            # shape: (T, P)
            S = Sda.values[:, yi, xi]
            U = Uda.values[:, yi, xi]
            V = Vda.values[:, yi, xi]
            ETA = Hda.values[:, yi, xi]
            return dict(time=time, S=S, U=U, V=V, ETA=ETA)
        
        def _get_time_coord_from_da(da):
            """Find a reasonable time coordinate on an xarray DataArray."""
            for nm in ('time_primea', 'time', 't'):
                if nm in da.coords:
                    return da[nm].values
            # last resort: look for any 1D datetime-like coord
            for nm, coord in da.coords.items():
                if getattr(coord, "ndim", 0) == 1 and np.issubdtype(coord.dtype, np.datetime64):
                    return coord.values
            raise ValueError("No time coordinate found on DataArray")
        
        def extract_at_transect_3x3(model_dict, transect_indices_flat):
            """
            Extract a 3x3 neighbourhood around each transect cell.
        
            Returns a dict with:
              - time: (T,)
              - S, U, V, ETA: (T, P, 9) arrays ordered by neighbour offsets
                              [(-1,-1), (-1,0), (-1,1),
                               ( 0,-1), ( 0,0), ( 0,1),
                               ( 1,-1), ( 1,0), ( 1,1)]
                Out-of-bounds neighbours are filled with NaNs.
              - neigh_ix: (P, 9, 2) int array of (y,x) indices used per neighbour (NaN rows use -1)
              - center_pos: (P, 2) int array of (y_center, x_center) for each transect point
            """
            Sda = model_dict['surface_salinity']
            Uda = model_dict['surface_Uvelocity']
            Vda = model_dict['surface_Vvelocity']
            Hda = model_dict['surface_height']
        
            time = _get_time_coord_from_da(Sda)
        
            # infer y/x dims
            ydim = [d for d in Sda.dims if d.lower() in ('y', 'eta', 'j')][-1]
            xdim = [d for d in Sda.dims if d.lower() in ('x', 'xi', 'i')][-1]
            ny = Sda.sizes[ydim]
            nx = Sda.sizes[xdim]
        
            # full arrays (T, ny, nx)
            Sfull  = Sda.values
            Ufull  = Uda.values
            Vfull  = Vda.values
            ETAfull= Hda.values
            T = Sfull.shape[0]
        
            # centre indices (P,)
            yi, xi = np.unravel_index(transect_indices_flat, (ny, nx))
            P = yi.size
        
            # neighbour ordering (dy, dx)
            offsets = [(-1,-1), (-1,0), (-1,1),
                       ( 0,-1), ( 0,0), ( 0,1),
                       ( 1,-1), ( 1,0), ( 1,1)]
            K = len(offsets)
        
            # allocate outputs
            S = np.full((T, P, K), np.nan, dtype=float)
            U = np.full((T, P, K), np.nan, dtype=float)
            V = np.full((T, P, K), np.nan, dtype=float)
            ETA = np.full((T, P, K), np.nan, dtype=float)
            neigh_ix = np.full((P, K, 2), -1, dtype=int)  # (y,x) per neighbour
            center_pos = np.column_stack([yi.astype(int), xi.astype(int)])
        
            # fill per transect point
            for p in range(P):
                yc = int(yi[p]); xc = int(xi[p])
                for k, (dy, dx) in enumerate(offsets):
                    yy = yc + dy
                    xx = xc + dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        S[:, p, k]   = Sfull[:, yy, xx]
                        U[:, p, k]   = Ufull[:, yy, xx]
                        V[:, p, k]   = Vfull[:, yy, xx]
                        ETA[:, p, k] = ETAfull[:, yy, xx]
                        neigh_ix[p, k, 0] = yy
                        neigh_ix[p, k, 1] = xx
                    else:
                        # leave NaNs / -1s for out-of-bounds
                        pass
        
            return dict(time=time, S=S, U=U, V=V, ETA=ETA,
                        neigh_ix=neigh_ix, center_pos=center_pos, offsets=np.array(offsets, dtype=int))
        def align_models_on_time(A, B):
            """
            A, B are dicts with keys time, S, U, V, ETA.
            Returns time, (S_A,U_A,V_A,ETA_A), (S_B,U_B,V_B,ETA_B) aligned.
            """
            tA = A['time']; tB = B['time']
            tC, idxA, idxB = np.intersect1d(tA, tB, return_indices=True)
            outA = (A['S'][idxA], A['U'][idxA], A['V'][idxA], A['ETA'][idxA])
            outB = (B['S'][idxB], B['U'][idxB], B['V'][idxB], B['ETA'][idxB])
            return tC, outA, outB
        
        
        # --- tide segmentation and metrics ---
        
        def find_tide_windows(u_along_mean, slack_thresh=1e-6, min_len=8):
            """
            Split the series into tide-by-tide windows from zero-crossings of u_along_mean.
            u_along_mean: (T,) section-mean along velocity (positive flood, negative ebb).
            Returns list of (i0, i1) index pairs (inclusive start, exclusive end).
            """
            u = np.asarray(u_along_mean)
            # treat small magnitudes as slack to stabilize sign detection
            u_eff = u.copy()
            u_eff[np.abs(u_eff) < slack_thresh] = 0.0
            sign = np.sign(u_eff)
            sign[sign == 0] = np.nan  # will be forward-filled
        
            # Forward fill NaNs with last non-nan sign to avoid micro-slacks
            last = 0.0
            for i in range(sign.size):
                if np.isnan(sign[i]):
                    sign[i] = last
                else:
                    last = sign[i]
        
            # zero-crossings where sign changes
            change = np.where(np.diff(sign) != 0)[0] + 1
            # window edges
            edges = np.r_[0, change, u.size]
            windows = []
            for a, b in zip(edges[:-1], edges[1:]):
                if b - a >= min_len:
                    windows.append((a, b))
            return windows
        
        def pumping_uc_prime(u, c):
            """Return <u'c'> over a window (1D arrays)."""
            u_ = u - np.mean(u)
            c_ = c - np.mean(c)
            return float(np.mean(u_ * c_))
        
        def hysteresis_loop_area(x_eta, y_s):
            """
            Shoelace polygon area of the loop in (eta, S) space.
            Assumes data follow the tidal order around the loop.
            Returns signed area (units: [eta]*[S]).
            """
            x = np.asarray(x_eta); y = np.asarray(y_s)
            # close polygon
            x2 = np.r_[x, x[0]]; y2 = np.r_[y, y[0]]
            area = 0.5 * np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1])
            return float(area)
        
        
        # --- per-estuary driver ---
        
        def estuary_masks(transect_df):
            """
            Returns dict: {est_name: np.array(bool) mask over rows} preserving row order.
            """
            masks = {}
            for name in transect_df['est_name'].unique():
                masks[name] = (transect_df['est_name'].values == name)
            return masks
        
        def midpoint_index(distances):
            """Index of the transect midpoint (closest to median distance)."""
            d = np.asarray(distances)
            return int(np.argmin(np.abs(d - np.median(d))))

        def nanmean_with_min_count(A, axis=0, min_count=3):
            """Like nanmean, but returns NaN when valid-count < min_count."""
            A = np.asarray(A)
            with np.errstate(invalid='ignore'):
                m = np.isfinite(A)
                cnt = np.sum(m, axis=axis)
                out = np.nanmean(A, axis=axis)
                if axis == 0:
                    out[cnt < min_count] = np.nan
                else:
                    # put mask back along the axis
                    out = np.where(cnt < min_count, np.nan, out)
            return out
        
        def nearest_valid_column(S_window, candidate_j, search_radius=10, min_valid=0.7):
            """
            Pick the nearest column to candidate_j with >= min_valid fraction finite over the window.
            S_window: (T_w, P_sel)
            """
            T_w, P_sel = S_window.shape
            for r in range(0, search_radius+1):
                for j in (candidate_j - r, candidate_j + r):
                    if 0 <= j < P_sel:
                        frac_ok = np.isfinite(S_window[:, j]).mean()
                        if frac_ok >= min_valid:
                            return j
            # fallback: column with max finite fraction
            ok_fracs = np.array([np.isfinite(S_window[:, j]).mean() for j in range(P_sel)])
            return int(np.nanargmax(ok_fracs))
        
        def d_etadt(eta_series, time):
            """Compute dη/dt in m/s using numpy datetime64 time."""
            t = np.asarray(time).astype('datetime64[ns]').astype('int64') * 1e-9  # seconds
            dt = np.gradient(t)
            return np.gradient(eta_series, dt)
        
        def find_tide_windows_from_deta(deta, slack_thresh=1e-5, min_len=8):
            """
            Windows between sign changes of dη/dt (flood: dη/dt>0, ebb: dη/dt<0).
            Uses small threshold to avoid chattering near slack.
            """
            x = np.array(deta, dtype=float)
            x[np.abs(x) < slack_thresh] = 0.0
            s = np.sign(x)
            # fill zeros with previous sign to stabilize
            last = np.nan
            for i in range(s.size):
                if s[i] == 0:
                    s[i] = last if not np.isnan(last) else 0
                else:
                    last = s[i]
            change = np.where(np.diff(s) != 0)[0] + 1
            edges = np.r_[0, change, s.size]
            windows = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if (b - a) >= min_len]
            return windows
        
        def pumping_uc_prime_safe(u, c, min_len=6):
            """<u'c'> with NaN handling; returns NaN if too few valid points."""
            u = np.asarray(u); c = np.asarray(c)
            ok = np.isfinite(u) & np.isfinite(c)
            if ok.sum() < min_len:
                return np.nan
            u0 = u[ok] - np.nanmean(u[ok])
            c0 = c[ok] - np.nanmean(c[ok])
            return float(np.nanmean(u0 * c0))
        
        def hysteresis_loop_area(x_eta, y_s):
            """Signed polygon area in (η,S) space, NaN-safe (drops NaN pairs)."""
            x = np.asarray(x_eta); y = np.asarray(y_s)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3:
                return np.nan
            x = x[ok]; y = y[ok]
            x2 = np.r_[x, x[0]]; y2 = np.r_[y, y[0]]
            return 0.5 * float(np.sum(x2[:-1]*y2[1:] - x2[1:]*y2[:-1]))
        
        # ---------- patched driver
        
        def select_best_velocity_neighbour(U_TPK, V_TPK, min_valid_frac=0.6):
            """
            Selects the neighbour index (0..K-1) for each transect column that has the
            highest RMS speed (sqrt(U^2 + V^2)), ignoring NaNs and requiring a minimum
            fraction of valid data.
            """
            T, P, K = U_TPK.shape
            sel_idx = np.full(P, 4, dtype=int)  # default to centre cell (index 4 in 3×3)
            for p in range(P):
                best_k, best_rms = 4, -np.inf
                for k in range(K):
                    u = U_TPK[:, p, k]
                    v = V_TPK[:, p, k]
                    ok = np.isfinite(u) & np.isfinite(v)
                    if ok.mean() < min_valid_frac:
                        continue
                    rms = np.sqrt(np.nanmean(u[ok]**2 + v[ok]**2))
                    if rms > best_rms:
                        best_rms = rms
                        best_k = k
                sel_idx[p] = best_k
            return sel_idx

        def compute_estuary_metrics_v2(
            prim_dict, ukc4_dict, transect_df, transect_indices,
            use_angles_from_df=True, estuary_angle_fallback=None,
            sec_min_points=5,  # min columns to accept a section mean at a time
            loop_search_radius=12, loop_min_valid=0.6,
            slack_thresh=1e-5, min_len=8
        ):
            # Extract & align once (as before)
            A = extract_at_transect(prim_dict, transect_indices.flatten())
            B = extract_at_transect(ukc4_dict, transect_indices.flatten())
            
            # Handle velocity seperately to avoid 0m/s cells
            Avel = extract_at_transect_3x3(prim_dict, transect_indices.flatten())
            Bvel = extract_at_transect_3x3(ukc4_dict, transect_indices.flatten())
            # Select best neighbour index for each transect column
            selA = select_best_velocity_neighbour(Avel['U'], Avel['V'])
            selB = select_best_velocity_neighbour(Bvel['U'], Bvel['V'])
            # Replace U and V in A/B with chosen neighbour velocities
            A['U'] = np.take_along_axis(Avel['U'], selA[None, :, None], axis=2).squeeze(-1)
            A['V'] = np.take_along_axis(Avel['V'], selA[None, :, None], axis=2).squeeze(-1)
            
            B['U'] = np.take_along_axis(Bvel['U'], selB[None, :, None], axis=2).squeeze(-1)
            B['V'] = np.take_along_axis(Bvel['V'], selB[None, :, None], axis=2).squeeze(-1)
                    
            time, (S_A, U_A, V_A, ETA_A), (S_B, U_B, V_B, ETA_B) = align_models_on_time(A, B)

            masks_by_est = {}
            for name in transect_df['est_name'].unique():
                masks_by_est[name] = (transect_df['est_name'].values == name)
        
            angles_per_point = transect_df['angle'].values if use_angles_from_df else None
            distances = transect_df['distance'].values
        
            metrics = {}
        
            for est_name, mask in masks_by_est.items():
                idx = np.where(mask)[0]
                if idx.size < 10:
                    continue
        
                # angles
                if use_angles_from_df:
                    angs = angles_per_point[idx]
                else:
                    if estuary_angle_fallback is None or est_name not in estuary_angle_fallback:
                        continue
                    angs = np.full(idx.size, estuary_angle_fallback[est_name]['angle'])
        
                # slice
                S_Ae, UAe, VAe, ETA_Ae = S_A[:, idx], U_A[:, idx], V_A[:, idx], ETA_A[:, idx]
                S_Be, UBe, VBe, ETA_Be = S_B[:, idx], U_B[:, idx], V_B[:, idx], ETA_B[:, idx]
                d_est = distances[idx]
        
                # along-velocity
                UA_al = project_along_estuary(UAe, VAe, angs)
                UB_al = project_along_estuary(UBe, VBe, angs)
        
                # section means with NaN handling
                S_A_sec = nanmean_with_min_count(S_Ae, axis=1, min_count=sec_min_points)
                S_B_sec = nanmean_with_min_count(S_Be, axis=1, min_count=sec_min_points)
                U_A_sec = nanmean_with_min_count(UA_al, axis=1, min_count=sec_min_points)
                U_B_sec = nanmean_with_min_count(UB_al, axis=1, min_count=sec_min_points)
                ETA_sec = nanmean_with_min_count(0.5*(ETA_Ae + ETA_Be), axis=1, min_count=sec_min_points)
        
                # tide windows from dη/dt
                deta = d_etadt(ETA_sec, time)
                tides = find_tide_windows_from_deta(deta, slack_thresh=slack_thresh, min_len=min_len)
                if not tides:
                    metrics[est_name] = {'PRIMEA': {'tides': []}, 'UKC4': {'tides': []}}
                    continue
        
                # choose a representative column near midpoint that's valid per tide
                mid_global = idx[int(np.argmin(np.abs(d_est - np.median(d_est))))]
        
                est_metrics = {'PRIMEA': {'tides': []}, 'UKC4': {'tides': []}}
        
                for (i0, i1) in tides:
                    # pick nearest valid column separately for each model & tide (uses salinity validity)
                    j_local = int(np.where(idx == mid_global)[0][0])
        
                    jA = nearest_valid_column(S_Ae[i0:i1, :], j_local,
                                              search_radius=loop_search_radius, min_valid=loop_min_valid)
                    jB = nearest_valid_column(S_Be[i0:i1, :], j_local,
                                              search_radius=loop_search_radius, min_valid=loop_min_valid)
        
                    # PRIMEA metrics
                    pumpA = pumping_uc_prime_safe(U_A_sec[i0:i1], S_A_sec[i0:i1])
                    areaA = hysteresis_loop_area(ETA_A[:, idx[jA]][i0:i1], S_A[:, idx[jA]][i0:i1])
                    est_metrics['PRIMEA']['tides'].append({'i0': int(i0), 'i1': int(i1),
                                                            'pumping_section_mean': pumpA,
                                                            'loop_area_midpoint': float(areaA)})
        
                    # UKC4 metrics
                    pumpB = pumping_uc_prime_safe(U_B_sec[i0:i1], S_B_sec[i0:i1])
                    areaB = hysteresis_loop_area(ETA_B[:, idx[jB]][i0:i1], S_B[:, idx[jB]][i0:i1])
                    est_metrics['UKC4']['tides'].append({'i0': int(i0), 'i1': int(i1),
                                                          'pumping_section_mean': pumpB,
                                                          'loop_area_midpoint': float(areaB)})
        
                metrics[est_name] = est_metrics
        
            return time, metrics
        
        import matplotlib.dates as mdates

        
        def _edges_from_centers(x):
            x = np.asarray(x, dtype=float)
            if x.size < 2:
                return np.array([x[0]-0.5, x[0]+0.5])
            dx = np.diff(x)
            left  = x[0]  - dx[0]/2
            right = x[-1] + dx[-1]/2
            mids = (x[:-1] + x[1:]) / 2
            return np.r_[left, mids, right]
        

        def _pad_to_square(bounds, pad_frac=0.10):
            """
            Make lon/lat bounds square with a bit of padding so all insets look similar.
            bounds = (lon_min, lon_max, lat_min, lat_max) in degrees.
            """
            lon0, lon1, lat0, lat1 = bounds
            w = lon1 - lon0
            h = lat1 - lat0
            size = max(w, h)
            # expand to square
            cx = 0.5 * (lon0 + lon1)
            cy = 0.5 * (lat0 + lat1)
            half = 0.5 * size
            # add padding on top of that
            half *= (1 + pad_frac)
            return (cx - half, cx + half, cy - half, cy + half)
        
        def hovmoller_salinity_strict(est_name, time, distances, S_A, S_B, idx_points, diff_vlim=5):
            # --- subset + strict NaN filter (unchanged) ---
            from matplotlib import gridspec
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # not strictly needed here
            idx_points = np.asarray(idx_points, dtype=int)
            d  = np.asarray(distances)[idx_points]
            SA = np.asarray(S_A)[:, idx_points]
            SB = np.asarray(S_B)[:, idx_points]
        
            good_cols = np.isfinite(SA).all(axis=0) & np.isfinite(SB).all(axis=0)
            if good_cols.sum() == 0:
                print(f"[{est_name}] No transect points without NaNs; nothing to plot.")
                return
            d, SA, SB = d[good_cols], SA[:, good_cols], SB[:, good_cols]
        
            order = np.argsort(d)
            d, SA, SB = d[order], SA[:, order], SB[:, order]
        
            # --- edges + limits (unchanged) ---
            t_nums  = mdates.date2num(np.asarray(time))
            t_edges = _edges_from_centers(t_nums)
            d_edges = _edges_from_centers(d / 1000.0)  # km
        
            DIFF = SA - SB
            sal_min = float(np.nanmin([SA.min(), SB.min()]))
            sal_max = float(np.nanmax([SA.max(), SB.max()]))
        
            # --- figure: left column for Hovmöller, right column for inset map ---
            fig = plt.figure(figsize=(13.5, 8))
            gs  = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[1.0, 0.42], height_ratios=[1,1,1],
                                    wspace=0.15, hspace=0.2)
        
            # left column axes
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
            ax2 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)
            axes = [ax0, ax1, ax2]
        
            cm = plt.get_cmap('viridis')
        
            im0 = ax0.pcolormesh(t_edges, d_edges, SA.T, cmap=cm, shading='flat',
                                 edgecolors='none', vmin=sal_min, vmax=sal_max)
            ax0.set_title(f"A ({est_name} — IRENE)")
            ax0.set_ylabel("Distance [km]")
            c0 = fig.colorbar(im0, ax=ax0, label="Salinity [psu]", pad=0.012, fraction=0.046)
        
            im1 = ax1.pcolormesh(t_edges, d_edges, SB.T, cmap=cm, shading='flat',
                                 edgecolors='none', vmin=sal_min, vmax=sal_max)
            ax1.set_title(f"B ({est_name} — UKC4)")
            ax1.set_ylabel("Distance [km]")
            c1 = fig.colorbar(im1, ax=ax1, label="Salinity [psu]", pad=0.012, fraction=0.046)
        
            
            max_all_diff = np.nanmax(np.abs(DIFF))
            im2 = ax2.pcolormesh(t_edges, d_edges, DIFF.T, cmap='RdBu_r', shading='flat',
                                 edgecolors='none', vmin=-max_all_diff, vmax=max_all_diff)
           
            ax2.set_title(f"C ({est_name} — ΔIRENE)")
            ax2.set_ylabel("Distance [km]")
            ax2.set_xlabel("Time")
            c2 = fig.colorbar(im2, ax=ax2, label="ΔSalinity [psu]", pad=0.012, fraction=0.046)
        
            # nice dates
            axes[-1].xaxis_date()
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        
            # --- right column: consistent map inset with square padded extent ---
            est_bounds = estuary_bounds[est_name.capitalize()]      # (lon_min, lon_max, lat_min, lat_max)
            td_valid   = store_td_valid[est_name.capitalize()]
            td_invalid = store_td_invalid[est_name.capitalize()]
        
            # map axis occupies the full right column across all rows
            ax_map = fig.add_subplot(gs[:, 1], projection=ccrs.PlateCarree())
            ax_map.set_facecolor('white')
            ax_map.set_title('D')
            im0.set_clim(0, 35)
            im1.set_clim(0, 35)
            
            # outline + points
            outline.plot(ax=ax_map, edgecolor='black', facecolor='none', linewidth=0.8, transform=ccrs.PlateCarree())
            if len(td_valid):
                ax_map.scatter(td_valid['x'], td_valid['y'], s=12, color='red', transform=ccrs.PlateCarree(), zorder=3)
            if len(td_invalid):
                ax_map.scatter(td_invalid['x'], td_invalid['y'], s=12, color='green', transform=ccrs.PlateCarree(), zorder=3)
        
            # enforce square, padded extent so every inset looks consistent
            lon0, lon1, lat0, lat1 = _pad_to_square(est_bounds, pad_frac=0.10)
            ax_map.set_extent((lon0, lon1, lat0, lat1), crs=ccrs.PlateCarree())
        
            # tidy map
            # ax_map.coastlines(resolution='10m', linewidth=0.6)
            ax_map.set_xticks([]); ax_map.set_yticks([])
            # optional frame:
            for spine in ax_map.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
        
        
            # show x ticks ONLY on the bottom panel
            for ax in (ax0, ax1):
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            # --- save ---
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            savename = Path(save_dir) / f"{est_name}_hovmoller_comparison_with_inset.png"
            fig.savefig(savename, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
            print(f"[{est_name}] plotted {d.size} clean transect points [d range {d.min():.0f}–{d.max():.0f} m].")
            print(f"Saved to {savename}")
                           

        def plot_hysteresis_for_estuary(est_name, time,
                                        S_A, U_A, ETA_A,
                                        S_B, U_B, ETA_B,
                                        transect_df, indexer='mid',
                                        min_nonzero_frac=0.5, eps=1e-4,
                                        use_eta_phase_if_no_u=True):
            """
            Picks target cell near 'ocean' | 'mid' | 'river', but if UKC4 velocity is
            zero/masked there, moves to the nearest neighbour along the transect that
            has usable velocity. Optionally falls back to dη/dt for ebb/flood splitting.
            """
        
            import numpy as np
            import matplotlib.pyplot as plt
            from pathlib import Path
        
            # rows for this estuary
            mask = transect_df['est_name'].str.lower().values == est_name.lower()
            if not mask.any():
                print(f"No transect rows for {est_name}")
                return
        
            # sort by distance so neighbours are truly adjacent along-section
            est_rows = np.where(mask)[0]
            dists = transect_df.loc[mask, 'distance'].to_numpy()
            order = np.argsort(dists)
            cols = est_rows[order]               # global column indices into S_*, U_*, ETA_*
            dists_sorted = dists[order]
        
            # choose preferred relative index
            if indexer == 'ocean':
                target_rel = 0
            elif indexer == 'river':
                target_rel = len(cols) - 1
            else:  # 'mid'
                target_rel = int(np.argmin(np.abs(dists_sorted - np.median(dists_sorted))))
        
            # helpers
            def frac_finite(a):  return np.isfinite(a).mean()
            def frac_nonzero(a): return (np.abs(a) > eps).mean()
        
            def good_col(j):
                c = cols[j]
                okA = frac_finite(S_A[:, c]) > 0.8 and frac_finite(U_A[:, c]) > 0.8
                okB = frac_finite(S_B[:, c]) > 0.8 and frac_finite(U_B[:, c]) > 0.8
                nzB = frac_nonzero(U_B[:, c]) >= min_nonzero_frac
                return okA and okB and nzB
        
            def semi_good(j):  # for dη/dt fallback
                c = cols[j]
                okA = frac_finite(S_A[:, c]) > 0.8 and frac_finite(ETA_A[:, c]) > 0.8
                okB = frac_finite(S_B[:, c]) > 0.8 and frac_finite(ETA_B[:, c]) > 0.8
                return okA and okB
        
            # candidate order: target, +1, -1, +2, -2, ...
            cand = [target_rel]
            for k in range(1, len(cols)):
                cand += [target_rel + k, target_rel - k]
            cand = [j for j in cand if 0 <= j < len(cols)]
        
            chosen_rel = next((j for j in cand if good_col(j)), None)
            if chosen_rel is None and use_eta_phase_if_no_u:
                chosen_rel = next((j for j in cand if semi_good(j)), None)
        
            if chosen_rel is None:
                print(f"No usable columns for {est_name}")
                return
        
            c = cols[chosen_rel]
            dist_label = str(int(round(dists_sorted[chosen_rel])))
        
            # extract series
            eta_A, sal_A, vel_A = ETA_A[:, c], S_A[:, c], U_A[:, c]
            eta_B, sal_B, vel_B = ETA_B[:, c], S_B[:, c], U_B[:, c]
        
            # phase labels (prefer velocity; fallback to dη/dt)
            def time_seconds(t):
                t = np.asarray(t)
                try:
                    return (t - t[0]).astype('timedelta64[s]').astype(float)
                except Exception:
                    return np.array([(ti - t[0]).total_seconds() for ti in t], float)
        
            tsec = time_seconds(time)
        
            def phase_from(vel, eta):
                if frac_nonzero(vel) >= min_nonzero_frac and frac_finite(vel) > 0.8:
                    return np.where(vel > 0, 'flood', 'ebb')
                if use_eta_phase_if_no_u:
                    deta = np.gradient(eta, tsec)
                    return np.where(deta > 0, 'flood', 'ebb')
                return np.array(['unk'] * len(eta), dtype=object)
        
            phase_A = phase_from(vel_A, eta_A)
            phase_B = phase_from(vel_B, eta_B)
        
            # plot
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(eta_A[phase_A == 'flood'], sal_A[phase_A == 'flood'],
                       s=15, c='blue', alpha=0.6, label='IRENE flood')
            ax.scatter(eta_A[phase_A == 'ebb'],   sal_A[phase_A == 'ebb'],
                       s=15, c='cyan', alpha=0.6, label='IRENE ebb')
        
            ax.scatter(eta_B[phase_B == 'flood'], sal_B[phase_B == 'flood'],
                       s=15, c='red', alpha=0.6, label='UKC4 flood')
            ax.scatter(eta_B[phase_B == 'ebb'],   sal_B[phase_B == 'ebb'],
                       s=15, c='orange', alpha=0.6, label='UKC4 ebb')
        
            ax.set_xlabel("Surface Height [m]")
            ax.set_ylabel("Salinity [psu]")
            ax.legend()
            plt.tight_layout()
            plt.savefig(Path(save_dir) / f"{est_name}_hysteresis_{indexer}_at_{dist_label}m.png", dpi=300)
            plt.close()
        def pumping_uc_prime(u, c):
            ok = np.isfinite(u) & np.isfinite(c)
            if ok.sum() < 6: 
                return np.nan
            u0 = u[ok] - np.nanmean(u[ok])
            c0 = c[ok] - np.nanmean(c[ok])
            return float(np.nanmean(u0 * c0))
        
        def plot_across_transect_hysteresis(
            est_name, time, distances,
            S_A, Upar_A, S_B, Upar_B,
            idx_points, tide_slice=None, min_valid=30,
            x_axis="vel", ETA_A=None, ETA_B=None ):
                                                    
            """
            Plots hysteresis loops across transect points.
            
            Parameters
            ----------
            x_axis : str
                "vel" → use along-channel velocity (Upar_* arrays) for x-axis
                "sh"  → use surface height (ETA_* arrays) for x-axis
            ETA_* : array-like
                Required if x_axis="sh".
            """
            # Validate x_axis choice
            if x_axis not in ("vel", "sh"):
                raise ValueError("x_axis must be 'vel' or 'sh'")
        
            # Pick x-data arrays
            if x_axis == "vel":
                X_A, X_B = Upar_A, Upar_B
                x_label = r"$u_{\parallel}$ [ms$^{-1}$]"
            else:  # surface height
                if ETA_A is None or ETA_B is None:
                    raise ValueError("ETA_A and ETA_B must be provided if x_axis='sh'")
                X_A, X_B = ETA_A, ETA_B
                x_label = r"Surface Height [m]" 
        
            cols = np.asarray(idx_points, dtype=int)
        
            # Optional: restrict to a time window
            if tide_slice:
                i0, i1 = tide_slice
                S_A, X_A = S_A[i0:i1, :], X_A[i0:i1, :]
                S_B, X_B = S_B[i0:i1, :], X_B[i0:i1, :]
        
            # Build a validity mask per column for each model
            valid_pairs_A = (np.isfinite(S_A) & np.isfinite(X_A)).sum(axis=0)
            valid_pairs_B = (np.isfinite(S_B) & np.isfinite(X_B)).sum(axis=0)
        
            # Keep only cols that meet the threshold in BOTH models
            good_cols = [j for j in cols
                         if valid_pairs_A[j] >= min_valid and valid_pairs_B[j] >= min_valid]
        
            if not good_cols:
                print(f"[{est_name}] No columns with >= {min_valid} valid (x,S) pairs in both models.")
                return
        
            # Make subplots sized to the number of good columns
            n = len(good_cols)
            ncols = min(6, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.0*nrows), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).ravel()
        
            # Consistent limits across panels
            x_all = np.concatenate([X_A[:, good_cols].ravel(), X_B[:, good_cols].ravel()])
            s_all = np.concatenate([S_A[:, good_cols].ravel(), S_B[:, good_cols].ravel()])
            x_all = x_all[np.isfinite(x_all)]
            s_all = s_all[np.isfinite(s_all)]
            if x_all.size and s_all.size:
                xlim = (np.nanpercentile(x_all, 1), np.nanpercentile(x_all, 99))
                ylim = (np.nanpercentile(s_all, 1), np.nanpercentile(s_all, 99))
            else:
                xlim = ylim = None
        
            for k, j in enumerate(good_cols):
                ax = axes[k]
                ax.scatter(X_A[:, j], S_A[:, j], s=8, alpha=0.6, label='IRENE' if k==0 else None)
                ax.scatter(X_B[:, j], S_B[:, j], s=8, alpha=0.6, marker='x', label='UKC4' if k==0 else None)
                ax.set_title(f"d={distances[j]:.0f} [m]")
                if xlim: ax.set_xlim(xlim)
                if ylim: ax.set_ylim(ylim)
                if k % ncols == 0:
                    ax.set_ylabel("Salinity [psu]")
                if k // ncols == nrows - 1:
                    ax.set_xlabel(x_label)
        
            for ax in axes[n:]:
                ax.axis('off')
        
            fig.legend(loc="upper right")
            # fig.suptitle(f"Hysteresis across transect — {est_name}", y=0.98)
            fig.tight_layout()
            plt.savefig(Path(save_dir) / f"{est_name}_hysteresis_transects_{x_axis}.png", dpi=300)
            plt.close()
        def pumping_profile_vs_distance(est_name, distances, 
                                S_A, Upar_A, S_B, Upar_B, idx_points, tide_slice):
        
            # Filter idx_points for the current estuary
            if hasattr(idx_points, "__len__"):
                cols = [i for i in idx_points if transect_data.loc[i, "est_name"].lower() == est_name]
            else:
                raise ValueError("idx_points must be a sequence of indices")
        
            if len(cols) == 0:
                print(f"[{est_name}] No points found, skipping")
                return
        
            i0, i1 = tide_slice
            pA = [pumping_uc_prime(Upar_A[i0:i1, j], S_A[i0:i1, j]) for j in cols]
            pB = [pumping_uc_prime(Upar_B[i0:i1, j], S_B[i0:i1, j]) for j in cols]
        
            plt.figure(figsize=(6,4))
            plt.plot(distances[cols], pA, label="IRENE", marker="o")
            plt.plot(distances[cols], pB, label="UKC4", marker="s")
            plt.axhline(0, lw=1)
            plt.xlabel("Distance along transect [m]")
            plt.ylabel(r"Tidal pumping $\langle u'c' \rangle$ [ms$^{-1}$·psu]")
            # plt.title(f"Pumping profile — {est_name} (t={i0}:{i1})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(save_dir) / f"{est_name}_pumping_profile.png", dpi=300)
            plt.close()

        def plot_salinity_timeseries_transect(est_name, time, distances, S_A, S_B, idx_points, num_stations=3):
            """
            Compare salinity timeseries at selected distances along transect.
            Skips stations where either model has all NaNs.
            """
            # Sort transect points by distance
            sort_idx = np.argsort(distances[idx_points])
            d_sorted = distances[idx_points][sort_idx]
            SA_sorted = S_A[:, idx_points][:, sort_idx]
            SB_sorted = S_B[:, idx_points][:, sort_idx]
            
            # Function to test if a station has any valid data
            def is_valid(si):
                return not (np.all(np.isnan(SA_sorted[:, si])) or np.all(np.isnan(SB_sorted[:, si])))
        
            # Target positions (mouth, mid, head)
            target_indices = np.linspace(0, len(d_sorted)-1, num_stations, dtype=int)
            selected_indices = []
        
            for ti in target_indices:
                # If the target point is invalid, search outward until we find a valid one
                offset = 0
                found = False
                while not found and (ti - offset >= 0 or ti + offset < len(d_sorted)):
                    for candidate in [ti - offset, ti + offset]:
                        if 0 <= candidate < len(d_sorted) and is_valid(candidate):
                            selected_indices.append(candidate)
                            found = True
                            break
                    offset += 1
            
            # Ensure uniqueness and keep in distance order
            selected_indices = sorted(set(selected_indices), key=lambda i: d_sorted[i])
            
            fig, axes = plt.subplots(len(selected_indices), 1, figsize=(12, 2.5*len(selected_indices)), sharex=True)
            if len(selected_indices) == 1:
                axes = [axes]
            
            letters = [chr(ord('A') + i) for i in range(len(selected_indices))]

            for ax, si, letter in zip(axes, selected_indices, letters):
                ax.plot(time, SA_sorted[:, si], label='IRENE', color='tab:blue')
                ax.plot(time, SB_sorted[:, si], label='UKC4', color='tab:orange', alpha=0.7)
                ax.set_ylabel("Salinity [psu]")
                dist_km = float(d_sorted[si]) / 1000.0
                ax.set_title(f"{letter} (Distance along transect = {dist_km:.0f} km)")
                ax.grid(True, alpha=0.3)
            
            axes[0].legend()
            axes[-1].set_xlabel("Time")
            axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))   # weekly
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

            # fig.suptitle(f"Salinity timeseries comparison — {est_name}", y=0.95)
            plt.tight_layout()
            plt.savefig(Path(save_dir) / f"{est_name}_salinity_transects.png", dpi = 300)
            plt.close()
        
        def tidal_pumping():
            
            time, metrics = compute_estuary_metrics_v2(
                prim_dict, ukc4_dict,
                transect_df=transect_data,
                transect_indices=transect_indices,
                use_angles_from_df=True,                 # use your per-point angles
                estuary_angle_fallback=estuary_data      # only used if use_angles_from_df=False
            )
            
            # Example: compare PRIMEA vs UKC4 on the Dee
            dee = metrics.get('dee') or metrics.get('Dee')
            prim_vals = [t['pumping_section_mean'] for t in dee['PRIMEA']['tides']]
            ukc4_vals = [t['pumping_section_mean'] for t in dee['UKC4']['tides']]
            # Now you can compute differences, boxplots, etc.
            
            prim_areas = [t['loop_area_midpoint'] for t in dee['PRIMEA']['tides']]
            ukc4_areas = [t['loop_area_midpoint'] for t in dee['UKC4']['tides']]

            time, (S_A, U_A, V_A, ETA_A), (S_B, U_B, V_B, ETA_B) = align_models_on_time(
                extract_at_transect(prim_dict, transect_indices.flatten()),
                extract_at_transect(ukc4_dict, transect_indices.flatten())
            )
            

            for estuary in [i.lower() for i in estuary_data.keys()]:
                plot_hysteresis_for_estuary(estuary, time, S_A, U_A, ETA_A, S_B, U_B, ETA_B, transect_data)
                plot_hysteresis_for_estuary(estuary, time, S_A, U_A, ETA_A, S_B, U_B, ETA_B, transect_data, indexer = 'ocean')
                plot_hysteresis_for_estuary(estuary, time, S_A, U_A, ETA_A, S_B, U_B, ETA_B, transect_data, indexer = 'river')
            # estuary = 'mersey'
            for estuary in [i.lower() for i in estuary_data.keys()]:
                idx_points = transect_data.loc[transect_data['est_name'] == estuary, :].index.values  # your “old way”
                distances_arr = transect_data['distance'].values
                plot_across_transect_hysteresis(
                    estuary, time, distances_arr,
                    S_A, U_A, S_B, U_B,
                    idx_points, x_axis="vel"
                )
                
                plot_across_transect_hysteresis(
                    estuary, time, distances_arr,
                    S_A, U_A, S_B, U_B,
                    idx_points, x_axis="sh",
                    ETA_A=ETA_A, ETA_B=ETA_B
                )
                            
            # est = "ribble"
            est_hovmoller_saved_data = {}
            
            for est in [i.lower() for i in estuary_data.keys()]:
                idx_points = transect_data.loc[transect_data['est_name'] == est, :].index.values  
                hovmoller_salinity_strict(est, time, transect_data['distance'].values,
                                          S_A, S_B, idx_points, diff_vlim=5)
                est_hovmoller_saved_data[est] = {}
                est_hovmoller_saved_data[est]['time'] = time
                est_hovmoller_saved_data[est]['distance'] = transect_data['distance'].values
                est_hovmoller_saved_data[est]['S_A'] = S_A
                est_hovmoller_saved_data[est]['S_B'] = S_B
                est_hovmoller_saved_data[est]['idx_points'] = idx_points
                est_hovmoller_saved_data[est]['diff_vlim'] = 5
                
            data_proc_path = Path(data_stats_path).parent / 'data_proc'
            with open(data_proc_path / 'salinity_hovmuller_data.pkl', 'wb') as file:
                pickle.dump(est_hovmoller_saved_data, file)
                
            
            # spinup = 1440
            tide_slice = (0, S_A.shape[0])  
            for est_name in [i.lower() for i in estuary_data.keys()]:
                idx_points = transect_data.loc[transect_data['est_name'] == est_name, :].sort_values('distance').index.values

                pumping_profile_vs_distance(est_name,
                    transect_data['distance'].values,
                    S_A, U_A, S_B, U_B,
                    idx_points,  # full set
                    tide_slice)

            # estuary = "ribble"
            for estuary in [i.lower() for i in estuary_data.keys()]:
                idx_points = transect_data.loc[transect_data['est_name'] == estuary, :].sort_values('distance').index.values
                
                plot_salinity_timeseries_transect(
                    est_name=estuary,
                    time=time,
                    distances=transect_data['distance'].values,
                    S_A=S_A,
                    S_B=S_B,
                    idx_points=idx_points,
                    num_stations=3  # mouth, mid, head
                )
        tidal_pumping()
        #%%
def find_dir(file_path, filename='kent_regrid.nc'):
    """
    Find directories within the given file_path that contain the specified filename.
    
    :param file_path: Path to the directory to search within.
    :param filename: Name of the file to look for in each directory.
    :return: List of directory names containing the specified file.
    """
    directories_with_file = [entry for entry in os.listdir(file_path)
                             if os.path.isdir(os.path.join(file_path, entry)) and
                                os.path.isfile(os.path.join(file_path, entry, filename))]
    
    return directories_with_file        
        
        
#%%        
        
if __name__ == '__main__':
  
    # multi_file_path = path = os.path.join(start_path,'modelling_DATA','kent_estuary_project','7.met_office','models')
    # multi_file_path = path = os.path.join(start_path,'modelling_DATA','kent_estuary_project','11.3d_testing','models')
    # multi_file_path = path = os.path.join(start_path,'modelling_DATA','kent_estuary_project','12.salinity_calibration_laststeps','models')
    multi_file_path = path = os.path.join(start_path,'modelling_DATA','kent_estuary_project','13.3D_finals','models')


    list_of_files = find_dir(multi_file_path)
    # list_of_files = list_of_files[-1] # only change the last one for the conference. 
    list_of_files = [  
          #'bathymetry_testing',
          
          
          # 'ao_nawind_AllRivNoDuddonClimatology_m0.035_Forcing',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_95_Discouv', # best one so far
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_95_Discouv_9.5_Viscouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_95_Discouv_0.15_smor',
          # 'ao_yawind_8_rivs_real_flows_m0.035_Forcing',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_115_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_125_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_135_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_135_Discouv_2_Viscouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_160_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_200_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_240_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_260_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_280_Discouv',
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_300_Discouv',
          # 'ao_yawind_orig8RealRiver_m0.035_Forcing_300_Discouv',
          ### 2D layer salinity models
          # 'ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv',
          # 'ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv',
                  # '3d_5layer_climatology_85',
                  # '3d_10_layer',
                  
                  
          ### These are the last models, the most important models. 
          ### 3d layer models. 
          '3d_10_layer_climatology_layer0',
          '3d_10_layer_realriv_layer0',
          '3d_10_layer_climatology_layer1', # Layer fixed so now works
          '3d_10_layer_realriv_layer1',
          '1d_1_layer_climatology_layer0',
          '1d_1_layer_realriv_layer0',
          
         # 'oa_nawind_Orig_m0.035_Forcing_4_months',
      #   'oa_nawind_Orig_m0.030_Forcing',
      #   'oa_nawind_Orig_m0.035_Forcing',
      #   'oa_nawind_Orig_m0.040_Forcing',
      #   'oa_nawind_Orig_m0.045_Forcing',
      #   'oa_nawind_Orig_m0.050_Forcing',
      ]
    #  'PRIMEA_riv_nawind_oa_1l_flipped',
    #  'PRIMEA_riv_nawind_oa_1l_original',
    # # 'PRIMEA_riv_yawind_oa_1l_flipped',
    # # 'PRIMEA_riv_yawind_oa_1l_original',
    # # 'kent_1.30_base_from_5.Final',
    #  ]
    for fn in list_of_files:
        print(fn)
        from o_func import DataChoice, DirGen
        import glob
        # main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
        # fn = 'kent_1.0.0_UM_wind' # 
        main_path = os.path.split(multi_file_path)[0]#os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'8.model_calibration')
        # fn = 'kent_1.30_base_from_5.Final' # ValueError: zero-size array to reduction operation minimum which has no identity

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
        #create a simple dictionary to avoid conflicts later
        ukc4_dict = {dataarray.name: dataarray for dataarray in extract_ukc4s}
        prim_dict = {dataarray.name: dataarray for dataarray in extract_prims}
        #Plot salinity
        
        surface_salinity = sts.salinity_validation(ukc4_dict['ukc4_surface_salinity'],  prim_dict['prim_surface_salinity'])
        
        rofi_regime = sts.salinity_rofi(ukc4_dict['ukc4_surface_salinity'],  prim_dict['prim_surface_salinity'])
        
        # Process the storm surge data and use the nearest points
        sts.storm_surge_analysis()
        # Take that and perform plots of transects. 
        sts.storm_surge_transects()
        sts.skew_surge()
        sts.salinity_transect_analysis()
        # This needs to be set up with a dictionary, so outputs from linear regression need to be in a dictionary. 
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
final_time = measuretime.time()
time_taken = final_time - start_measure
minutes =  str(round(time_taken // 60)).zfill(2)
seconds = str(round(time_taken % 60)).zfill(2)
print('Finished in ', minutes, ':',seconds)
