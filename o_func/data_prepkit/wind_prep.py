""" Generation of boundary conditions

A user manual for some of this stuff
https://sfincs.readthedocs.io/en/latest/input_forcing.html

Created on Tue May 23 14:00:31 2023
@author: af
"""

#%% import dependecies
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#import cartopy.crs as ccrs
from datetime import datetime
import os
import glob
import iris 
import iris.coord_categorisation

import o_func.utilities as util

#%% Custom packages

from o_func.utilities.start import opsys; start_path = opsys()

#%% constants
s = ' '
#naming system for output files
u_name = 'Ucomponent.amu'
v_name = 'Vcomponent.amv'
p_name = 'pressure.amp'

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Generation of wind boundary conditions

For the generation of wind files on an equidistant grid with the following extensions
.amu - The U x velocity field
.amv - The V y velocity field
.amp - The atmospheric pressure

A user manual for some of this stuff
https://sfincs.readthedocs.io/en/latest/input_forcing.html

https://usermanual.wiki/Pdf/DFlowFMUserManual.1771347134/view
pg 181 of the delft 3d fm suite manual

Created on Tue May 23 14:00:31 2023
@author: af

example of equidistant wind files

QUANTITY =windx
FILENAME =windxdir.amu
FILETYPE =4
METHOD   =2
OPERAND  =O

QUANTITY =windy
FILENAME =windydir.amv
FILETYPE =4
METHOD   =2
OPERAND  =O

QUANTITY =atmosphericpressure
FILENAME =pressure.amp
FILETYPE =4
METHOD   =2
OPERAND  =O

example of a boundary forcing file :
    
[boundary]
quantity     = waterlevelbnd
locationfile = right.pli
forcingfile  = simplechannel.bc
"""

#%% constants
s = ' '

#%% Functions
class ItemNotFoundException(Exception):
    def __init__(self, item):
        self.item = item
        super().__init__(f"The item '{item}' was not found in the list.")


class WindWrite():
    def __init__(self):
        self.s = ' '
        
        self.u_name = 'x_wind.amu' # x wind velocity component
        self.v_name = 'y_wind.amv' # y wind velocity component
        self.p_name = 'ssp.amp'    # atmopheric pressure at sea surface component. 
        self.met_files = [self.u_name,self.v_name,self.p_name]
    
        self.data_types = ['UM', 'ERA5']
    #% define class methods
    @staticmethod
    def rand_wind_field(size, amplitude, period):
        t = np.linspace(0, 2*np.pi, size)
        wind_speeds = amplitude * np.sin(2*np.pi / period * t)  # Generate the wind speeds using the sinusoidal function
        return wind_speeds
    @staticmethod
    def generate_atmospheric_pressure(size, amplitude, period, min_pressure, max_pressure):
        t = np.linspace(0, 2*np.pi, size)  # Time parameter for the sinusoidal function
        pressure = ((max_pressure - min_pressure) / 2) * np.sin(2*np.pi / period * t) + ((max_pressure + min_pressure) / 2)
        # Generate the atmospheric pressure within the desired range using the sinusoidal function
        return pressure
    
    @staticmethod
    def to_datetime(date):
        """
        Converts a numpy datetime64 object to a python datetime object 
        Input:
          date - a np.datetime64 object
        Output:
          DATE - a python datetime object
        """
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                     / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)
    
#% Rest of class functions
    
    
    def write_met_header(self, met_file, nlon, nlat, minlon, maxlon, minlat, maxlat, start, output_path):
        """
        Writes header for meteorological file
        """
        dlon = (maxlon - minlon)/(nlon - 1) 
        dlat = (maxlat - minlat)/(nlat - 1)
        #path = start_path + 'modelling_DATA/kent_estuary_project/wind_input/wind_test_forcing/'
        met_var = met_file.split('.')[-1]
        s = self.s
        if met_var == 'amp':
            var_name = 'quantity1' + s*8 + '=    air_pressure\n'
            var_unit = 'unit' + s*13 + '=    Pa\n'
            nodata_value = 101300.0   
        elif met_var == 'amu':
            var_name = 'quantity1' + s*8 + '=    x_wind\n'
            var_unit = 'unit' + s*13 + '=    m s-1\n'
            nodata_value = 0.0   
        elif met_var == 'amv':
            var_name = 'quantity1' + s*8 + '=    y_wind\n'
            var_unit = 'unit' + s*13 + '=    m s-1\n'
            nodata_value = 0.0   
        else:
            sys.exit('Incorrect variable names')
    
    
        with open(output_path + met_file , 'a') as fid:
            fid.write('### START OF HEADER\n')
            fid.write('### This file is writen by Aaron Furnish through Bangor University\n')
            fid.write('### Additional comments\n')
            fid.write('FileVersion' + s*6 + '=    1.03\n')
            fid.write('filetype' + s*9 + '=    meteo_on_equidistant_grid\n')
            fid.write('NODATA_value     =    %.2f \n' % (nodata_value))
            fid.write('n_cols' + s*11 + '=  %5d \n' % nlon) # number of columns
            fid.write('n_rows' + s*11 + '=  %5d \n' % nlat) # number of rows
            fid.write('grid_unit' + s*8 + '=    deg\n') # could be metres (m)
            fid.write('x_llcorner' + s*7 + '=    %5.2f \n' % minlon) 
            fid.write('y_llcorner' + s*7 + '=    %5.2f \n' % minlat)
            fid.write('dx' + s*15 + '=   %5.2f \n' % (dlon))
            fid.write('dy' + s*15 + '=   %5.2f \n' % (dlat))
            fid.write('n_quantity' + s*7 + '=    1\n')
            fid.write(var_name)
            fid.write(var_unit)
            fid.write('### END OF HEADER\n')
            
        fid.close()
        
    def write_wind_external_forcing(self,file_path,names):
        '''
        Parameters
        ----------
        file_path : A string
            Filepath to the .ext external forcing file to manipulate and edit. 
        names : A list of 3 strings, containing the following, name.amp, name.amu, name.amv
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        #constants
        pressure = r'ssp'
        windx = r'windx'
        windy = r'windy'
        
        #Loop
        for file in names:
            f = file.split('.')[-1]
            n = file.split('.')[0]
            if f == r'amu':
                filetype = windx
            elif f == r'amv':
                filetype = windy
            elif f == r'amp':
                filetype = pressure
            with open(file_path,'a') as f:
                f.write(f'QUANTITY ={filetype}')
                f.write(f'FILENAME ={n}.{f}')
                f.write('FILETYPE =4')
                f.write('METHOD   =2')
                f.write('OPERAND  =O')
                f.write('\n') #reset the sequence ready for next external forcing. 
     
    def met_body(self,file_path,vel,start, interval, data_source, output_path):
        '''
        
    
        Parameters
        ----------
        file_path : TYPE
            DESCRIPTION.
        vel : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        interval : interger or float, represnting the interval in hours, if hourly
        it would be 1, if every 30 minutes it would be 0.5.
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        # try:
        #     if self.data_type not in self.data_types:
        #         raise ItemNotFoundException(self.data_type)
        # # Handle the custom exception
        # except ItemNotFoundException as e:
        #     print(e)
        
        
        if data_source[0] == 'ERA5':
            start_index, end_index = data_source[1], data_source[2]
            vel = vel[start_index:end_index]
            print('\nERA5 data must be transposed from lat lon to lon lat\n')
            num_rows, num_cols = vel[0].shape
            print(num_rows, num_cols) # 148,89
        else:
            start_index, end_index = data_source[1], data_source[2]
            vel = vel[start_index:end_index]
            print('\nERA5 data must be transposed from lat lon to lon lat\n')
            #print("Not coded yet, exiting scipt")
            #sys.exit()
            num_rows, num_cols = vel[0].shape
            print(num_rows, num_cols) # 148,89
            # if self.file.endswith('.amu') or self.file.endswith('.amv'):
            #     num_rows = num_rows - 1 # Just makes a quick correction to the number of rows. 
        
        
        #if num_rows == 149:
        #    num_rows = num_rows - 1
        
            
            
        t = 1
        
        
        for i in range(len(vel)):
            if data_source[0] == 'ERA5':
                single_t_vel = vel[i]
                # This iterates over each time value. However, it does not need transposing 
                # as it is already in the correct shape
            else:
                single_t_vel = vel[i]
                #print(vel.shape)
                #print('You havent coded this yet, exiting scipt...')
                #sys.exit()
                
                #single_t_vel = (vel[i]) # possible option for other data
            with open(os.path.join(output_path, file_path), 'a') as file:
            # Iterate over each row
                for row_index in range(num_rows):
                    if row_index == 0:
                        new_time = str(t*i)
                        file.write(f'TIME = {new_time} hours since {start}\n') # set up the time component
        
                    # Write the u values for the row, separated by spaces
                    vel_values = single_t_vel[row_index, :]
                    vel_line = ' '.join(str(value) for value in vel_values)
                    file.write(vel_line + '\n')
                        
    # def era5_plot(self, wind_file):
    #     '''
    #     Parameters
    #     ----------
    #     path : string
    #         Path to ERA5 dataset for generating mock wind fields to test model simulation. 
    
    #     Returns
    #     -------
    #     None.
    
    #     '''
    #     xrdf = xr.open_dataset(wind_file)
    #     lon = xrdf.longitude.values
    #     lat = xrdf.latitude.values
    #     u10 = xrdf.u10.values # shape = time, lat, lon
    #     v10 = xrdf.v10.values
    #     sp = xrdf.sp.values 
    
       
    #     for i in range(1):
    #         u = np.transpose(u10[i])
    #         v = np.transpose(v10[i])
    #         pressure = np.transpose(sp[i])/100
            
    #         WS = np.sqrt(u**2 + v**2)
    #         #WD = np.arctan2(-u, -v) * (180/np.pi)
    #         #projection = ccrs.PlateCarree()
    #         fig, ax = plt.subplots(subplot_kw={'projection': projection})
    #         ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=projection)
    #         lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    #         #barb_scale = 100  # Controls the size of the wind barbs
    #         # ax.barbs(lon_mesh, lat_mesh, u, v, WS, transform=projection, length=3, pivot='middle', 
    #         #          barbcolor='black', flagcolor='r', linewidth=0.5, 
    #         #          sizes=dict(emptybarb=0.25, spacing=0.2), zorder=10, alpha=0.8)
    #         q = ax.quiver(lon_mesh, lat_mesh, u, v, WS, transform=projection, pivot='middle', 
    #                   angles='uv', scale_units='xy', scale=25)
    
    #         # Add map features
    #         ax.coastlines(resolution='10m')
    #         ax.gridlines(draw_labels=True, linestyle='--')
    #         # Create a colorbar
         
    #         cbar = plt.colorbar(q, orientation='vertical')
    #         cbar.set_label('Wind Speed (ms$^{-1}$)')
    #         #cbar.ax.yaxis.set_label_coords(4, 0.5)
    #         cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    
    
    #         # Show the wind barb map
    #         plt.title('Wind Quiver')
    #         plt.savefig('temp/wind_quiver_uk_t=1.pdf', dpi = 300)
            
    #         # Wind pressure
            
    #         lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    #         lon_mesh = np.transpose(lon_mesh)
    #         lat_mesh = np.transpose(lat_mesh)
    #         fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize = (6,4.5))
    #         #fig.tight_layout()
    #         # Set the map boundaries based on latitude and longitude arrays
    #         ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=projection)
            
    #         # Plot filled contours of pressure
    #         # Calculate contour levels dynamically
    #         contour_interval = 20
    #         lowest_value = np.floor(np.min(pressure)/10) * 10
    #         highest_value = np.ceil(np.max(pressure)/10) * 10
    #         contour_levels = np.arange(lowest_value, highest_value + contour_interval, contour_interval)
    #         contour_levels_fill = np.arange(lowest_value, highest_value + contour_interval, contour_interval/4)
    
    #         contourf = ax.contourf(lon_mesh, lat_mesh, pressure, levels=contour_levels_fill, transform=projection, cmap='RdBu_r')
            
    #         # Overlay contour lines
    #         contour_lines = ax.contour(lon_mesh.T, lat_mesh.T, pressure.T, contour_levels, colors='brown', linewidths=1)
    #         plt.clabel(contour_lines, colors = 'k', fmt = '%1.0f', fontsize=5)
    
    #         # Add map features
    #         ax.coastlines(resolution='10m')
    #         ax.gridlines(draw_labels=True, linestyle='--')
            
    #         # Add a colorbar
    #         cbar = plt.colorbar(contourf, orientation='vertical')
    #         cbar.set_label('Pressure (hPa)')
    #         cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    
            
    #         # Show the wind pressure map with colorbar
    #         plt.title('Wind Pressure Map')
    #         plt.savefig('temp/wind_pressure_uk_t=1.pdf', dpi = 300)
    
    def era5_write(self,wind_files, time, output_path):
        self.data_type= self.data_types[1]
        
        
        xrdf = xr.open_dataset(wind_file)
        lon = xrdf.longitude.values
        lat = xrdf.latitude.values
        u10 = xrdf.u10.values # shape = time, lat, lon
        v10 = xrdf.v10.values
        sp = xrdf.sp.values
        t = xrdf.time.values
        
        #time handler for slicing
        start_time_slice = np.datetime64(time[0])
        end_time_slice = np.datetime64(time[1])
        start_index = np.where(t == start_time_slice)[0][0]
        end_index = np.where(t == end_time_slice)[0][0]
        
        # SOrting out start time 
        initial_timestep = start_time_slice#xrdf.time[0].values
        second_timestep = t[start_index+1]#xrdf.time[1].values  set to one after timeslice
        interval = (second_timestep - initial_timestep).astype('timedelta64[h]').astype(float)
        # set start time plus timezone offset
        start = WindWrite.to_datetime(initial_timestep).strftime('%Y-%m-%d %H:%M:%S %z+00:00') 
    
        nlon = len(lon) # set number of columns of lon
        nlat = len(lat) # set number of columns of la
        maxlon = max(lon)
        minlon = min(lon)
        maxlat = max(lat)
        minlat = min(lat)
        
        met_files2 = ['ERA_5_' + i for i in  self.met_files]
        
        data_source = ['ERA5', start_index, end_index]
        
        for file in met_files2:
            with open(output_path + file,'w') as f: # overwrites main file
                f.write("")
            f.close()
            print(file)
            self.write_met_header(file, nlon, nlat, minlon, maxlon, minlat, maxlat,start, output_path)
            if file.endswith('.amu'):
                vel = u10
                self.met_body(file,vel,start, interval,data_source, output_path)
            elif file.endswith('.amv'):
                vel = v10
                self.met_body(file,vel,start, interval,data_source, output_path)
            elif file.endswith('.amp'):
                vel = sp
                self.met_body(file,vel,start, interval,data_source, output_path)
        
    
    def UM_write(self, wind_files, time, output_path, wind_model):
        '''
        Wind_files should be directory to wind files location for UM grid. 
        '''
        nc_path = os.path.join(wind_files,wind_model,'*.nc')
        
        files = []
        for file in sorted(glob.glob(nc_path)):
            files.append(file)
            
        data = xr.open_mfdataset(files[1:])
        print(data)
        
        # x_wind, y_wind have diff coords to pressure by 1. Treat them seperately. 
        cubes = iris.load(file)
        
        pole_lat = cubes[0].coord_system().grid_north_pole_latitude
        pole_lon = cubes[0].coord_system().grid_north_pole_longitude
        
        
        rotated_lons, rotated_lats = iris.analysis.cartography.rotate_pole(
                        lons = np.array([-4, -2]), 
                        lats = np.array([53, 55]),         
                        pole_lon = pole_lon, 
                        pole_lat = pole_lat
                        )
       


        
        
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        
        subset_lat_lon_x_wind = data.x_wind.sel(
                                  grid_longitude=slice(find_nearest(data.x_wind.grid_longitude,rotated_lons[0]+360),
                                                       find_nearest(data.x_wind.grid_longitude,rotated_lons[1]+360)),  # Replace with your desired latitude range
                                  grid_latitude=slice(find_nearest(data.x_wind.grid_latitude,rotated_lats[0]),
                                                      find_nearest(data.x_wind.grid_latitude,rotated_lats[1]))  # Replace with your desired longitude range
                                  )
        
        subset_lat_lon_y_wind = data.y_wind.sel(
                                  grid_longitude=slice(find_nearest(data.y_wind.grid_longitude,rotated_lons[0]+360),
                                                       find_nearest(data.y_wind.grid_longitude,rotated_lons[1]+360)),  # Replace with your desired latitude range
                                  grid_latitude=slice(find_nearest(data.y_wind.grid_latitude,rotated_lats[0]),
                                                      find_nearest(data.y_wind.grid_latitude,rotated_lats[1]))  # Replace with your desired longitude range
                                  )
        
        ### Pressure has to be treated slightly differently. 
        subset_lat_lon_air_pressure = data.air_pressure_at_sea_level.sel(
                                  grid_longitude_0=slice(find_nearest(data.air_pressure_at_sea_level.grid_longitude_0,rotated_lons[0]+360),
                                                       find_nearest(data.air_pressure_at_sea_level.grid_longitude_0,rotated_lons[1]+360)),  # Replace with your desired latitude range
                                  grid_latitude_0=slice(find_nearest(data.air_pressure_at_sea_level.grid_latitude_0,rotated_lats[0]),
                                                      find_nearest(data.air_pressure_at_sea_level.grid_latitude_0,rotated_lats[1]))  # Replace with your desired longitude range
                                  )
        
        start_time_slice = np.datetime64(time[0])
        end_time_slice = np.datetime64(time[1])        
        start_index = np.where(data.time == start_time_slice)[0][0]
        end_index = np.where(data.time == end_time_slice)[0][0]
        
        initial_timestep = start_time_slice#xrdf.time[0].values
        second_timestep = data.time[start_index+1]#xrdf.time[1].values  set to one after timeslice
        interval = (second_timestep - initial_timestep).astype('timedelta64[h]').astype(float)
        # set start time plus timezone offset
        start = WindWrite.to_datetime(initial_timestep).strftime('%Y-%m-%d %H:%M:%S %z+00:00')
        
        
        met_files2 = ['UM_' + i for i in  self.met_files]
        data_source = ['UM', start_index, end_index]
        
        ##### Now the data has been partitioned, it would be a good idea to transform it back. 
        x_wind_xarray_cubes = subset_lat_lon_x_wind.to_iris()
        sp_xarray_cubes = subset_lat_lon_air_pressure.to_iris()
        
        Prot_lon, Prot_lat = iris.analysis.cartography.get_xy_grids(x_wind_xarray_cubes[0]) # get pressure grid.
        Wrot_lon, Wrot_lat = iris.analysis.cartography.get_xy_grids(sp_xarray_cubes[0]) # get pressure grid.
        
        Pnew_lon, Pnew_lat = iris.analysis.cartography.unrotate_pole(
                                Prot_lon, 
                                Prot_lat, 
                                pole_lat = cubes[0].coord_system().grid_north_pole_latitude, 
                                pole_lon = cubes[0].coord_system().grid_north_pole_longitude
                                )
        Wnew_lon, Wnew_lat = iris.analysis.cartography.unrotate_pole(
                                Wrot_lon, 
                                Wrot_lat, 
                                pole_lat = cubes[0].coord_system().grid_north_pole_latitude, 
                                pole_lon = cubes[0].coord_system().grid_north_pole_longitude
                                )
        
        ### For x_wind and y_wind 
        #nlon = len(Wnew_lon[0,:]) # set number of columns of lon
        #nlat = len(Wnew_lat[:,0]) # set number of columns of lat
        # if nlat == 149:
            # nlat == nlat - 1
        maxlon = max(Wnew_lon[0,:])
        minlon = min(Wnew_lon[0,:])
        maxlat = max(Wnew_lat[:,0])
        minlat = min(Wnew_lat[:,0])
        # When handling met office pressure data its on a diff grid 
        # nlat = nlat - 1 # this fixes it for the moment. Also need to change how many rows it does. 
        
        #data.x_wind.shape
        # Out[44]: (2904, 1026, 950)
      
        
        for file in met_files2:
            self.file = file
            
            output_file = os.path.join(output_path, file)
            
            if file.endswith('.amu'):
                with open(output_file,'w') as f: # overwrites main file
                    f.write("")
                f.close()
                print(file)
                
                vel = subset_lat_lon_x_wind.values
                nlon = vel.shape[2]
                nlat = vel.shape[1]
                self.write_met_header(file, nlon, nlat, minlon, maxlon, minlat, maxlat,start, output_path)
                
                
                
                self.met_body(file,vel,start, interval,data_source, output_path)
                print('shape vel ', vel.shape)
            elif file.endswith('.amv'):
                with open(output_file,'w') as f: # overwrites main file
                    f.write("")
                f.close()
                print(file)
                
                
                vel = subset_lat_lon_y_wind.values
                nlon = vel.shape[2]
                nlat = vel.shape[1]
                self.write_met_header(file, nlon, nlat, minlon, maxlon, minlat, maxlat,start, output_path)

                self.met_body(file,vel,start, interval,data_source, output_path)
                print('shape vel ', vel.shape)

            elif file.endswith('.amp'):
                #nlon = len(Pnew_lon[0,:]) # set number of columns of lon
                #nlat = len(Pnew_lat[:,0])
                
                
                print(' len(Pnew_lat[:,0])',  len(Pnew_lat[:,0]))# set number of columns of lat
                maxlon = max(Pnew_lon[0,:])
                minlon = min(Pnew_lon[0,:])
                maxlat = max(Pnew_lat[:,0])
                minlat = min(Pnew_lat[:,0])
                
                
                
                with open(output_file,'w') as f: # overwrites main file
                    f.write("")
                f.close()
                print(file)
                
                vel = subset_lat_lon_air_pressure.values
                nlon = vel.shape[2]
                nlat = vel.shape[1]
                self.write_met_header(file, nlon, nlat, minlon, maxlon, minlat, maxlat,start, output_path)

                self.met_body(file,vel,start, interval,data_source, output_path)
                print('shape vel ', vel.shape)

    
            print('nlon was ', nlon)
            print('nlat was ', nlat)
    #%%
    




#%% Main 
if __name__ == '__main__':
    #%% ERA5
    #variables
    starttime = '2013-10-31 00:00'
    wind_file = start_path + r'modelling_DATA/kent_estuary_project/wind_input/wind_data_uk_era5_2013_2014.nc'
    time = ['2013-10-15','2014-04-01']
    output_path = start_path + r'modelling_DATA/kent_estuary_project/wind_input'
    
    # ww = WindWrite() # begin class of windwriter

    # test case with era5 dataset. 
    # ww.era5_write(wind_file, time, output_path)

    #%% UM 
    
    #path choice
    wind_model = ['oa', 'og', 'owa', 'ow']
    wind_model = 'oa'
    ww = WindWrite() # begin class of windwriter
    time = ['2013-10-31 01:00:00','2014-03-01 00:00:00']
    write_to_path = os.path.join(output_path, wind_model)
    wind_files = os.path.join(start_path, 'Original_Data','UKC3','wind')
    util.md([write_to_path])
    ww.UM_write(wind_files, time, write_to_path, wind_model)
    
    
    # using the um grid dataset. 
    
    
    # start = r'2013-10-31 00:00:00 +00:00'
    # #naming system for output files
    # u_name = 'Ucomponent.amu'
    # v_name = 'Vcomponent.amv'
    # p_name = 'pressure.amp'
    
    # #output_path = r'/Volumes/PD/GitHub/delft_3dfm/delft3dfm/temp/'  ' temporary figure output path
    # output_path = start_path + r'modelling_DATA/kent_estuary_project/wind_input/wind_test_forcing/'
    
    # met_files = (u_name,v_name,p_name)
    # interval = 1 # set interval as 1 hour
    # time = ['2013-10-15','2014-04-01'] #set start time and end_time to control filesize
    # era5_write(wind_file, met_files,time,output_path)
    
    # era5_plot(wind_file)

    # size = 10  # Number of data points
    # amplitude = 5  # Maximum wind speed amplitude (half of the desired range)
    # period = 0.006  # Period of the oscillation (in data points)
    
    # wind_field = rand_wind_field(size, amplitude, period)
    # plt.plot(wind_field)
    
    # size = 100  # Number of data points
    # amplitude = (1050 - 990) / 2  # Half of the desired pressure range
    # period = 24  # Period of the oscillation (in data points)
    # min_pressure = 990  # Minimum atmospheric pressure
    # max_pressure = 1050  # Maximum atmospheric pressure
    
    # atmospheric_pressure = generate_atmospheric_pressure(size, amplitude, period, min_pressure, max_pressure)
    # plt.plot(atmospheric_pressure)




######################################################### SPare stuff from UM writer 
#iris.analysis.cartography.unrotate_pole(rotated_lons, rotated_lats, pole_lon, pole_lat)
#iris.analysis.cartography.rotate_pole(np.array([-4, -2]), np.array([53, 55]), pole_lon, pole_lat)

### New idea, subset it by its rotated coords, so its easy to slice then do what you need. 

# Prot_lon, Prot_lat = iris.analysis.cartography.get_xy_grids(cubes[0]) # get pressure grid.
# Wrot_lon, Wrot_lat = iris.analysis.cartography.get_xy_grids(cubes[1]) # get pressure grid.
# Pnew_lon, Pnew_lat = iris.analysis.cartography.unrotate_pole(
#                         Prot_lon, 
#                         Prot_lat, 
#                         pole_lat = cubes[0].coord_system().grid_north_pole_latitude, 
#                         pole_lon = cubes[0].coord_system().grid_north_pole_longitude
#                         )
# Wnew_lon, Wnew_lat = iris.analysis.cartography.unrotate_pole(
#                         Wrot_lon, 
#                         Wrot_lat, 
#                         pole_lat = cubes[0].coord_system().grid_north_pole_latitude, 
#                         pole_lon = cubes[0].coord_system().grid_north_pole_longitude
#                         )

# W = np.dstack([Wnew_lon.ravel(),Wnew_lat.ravel()])[0] # wind coords lon/lat
# P = np.dstack([Pnew_lon.ravel(),Pnew_lat.ravel()])[0]
# #split the data here into x_wind, y_wind and pressure

# x_wind = [np.ravel(data.x_wind) for i in data.x_wind]

# pressure = data.air_pressure_at_sea_level

# pressure.assign_coords(grid_longitude_0 = P[:,0] )
# x_wind = data.x_wind
# y_wind = data.y_wind
