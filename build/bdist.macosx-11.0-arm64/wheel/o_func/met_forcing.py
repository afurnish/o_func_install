#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from o_functions.start import opsys2; start_path = opsys2()
#%% constants
s = ' '
#naming system for output files
u_name = 'Ucomponent.amu'
v_name = 'Vcomponent.amv'
p_name = 'pressure.amp'

#%% Functions
def write_met_header(met_file, nlon, nlat, minlon, maxlon, minlat, maxlat):
    """
    Writes header for meteorological file
    """
    dlon = (maxlon - minlon)/(nlon - 1) 
    dlat = (maxlat - minlat)/(nlat - 1)
    path = start_path + 'modelling_DATA/kent_estuary_project/wind_input/wind_test_forcing/'
    met_var = met_file.split('.')[-1]

    with open(path + met_file , 'w') as fid:
        fid.write('### START OF HEADER\n')
        fid.write('FileVersion' + s*6 + '= 1.03\n')
        fid.write('filetype' + s*9 + '= meteo_on_equidistant_grid\n')
        fid.write('n_cols' + s*11 + '= %5d \n' % nlon) # number of columns
        fid.write('n_rows' + s*11 + '= %5d \n' % nlat) # number of rows
        fid.write('grid_unit' + s*8 + '= deg\n') # could be metres (m)
        fid.write('x_llcorner' + s*7 + '= %5.2f \n' % minlon) 
        fid.write('y_llcorner' + s*7 + '= %5.2f \n' % minlat)
        fid.write('dx' + s*15 + '= %5.2f \n' % (dlon))
        fid.write('dy' + s*15 + '= %5.2f \n' % (dlat))
        fid.write('n_quantity' + s*7 + '= 1\n')

        if met_var == 'amp':
            fid.write('quantity1' + s*8 + '= air_pressure\n')
            fid.write('unit' + s*13 + '= Pa\n')
            nodata_value = 101300.0   
        elif met_var == 'amu':
            fid.write('quantity1' + s*8 + '= x_wind\n')
            fid.write('unit' + s*13 + '= m/s\n')
            nodata_value = 0.0   
        elif met_var == 'amv':
            fid.write('quantity1' + s*8 + '= y_wind\n')
            fid.write('unit' + s*13 + '= m/s\n')
            nodata_value = 0.0   
            
        else:
            sys.exit('Incorrect variable names')
        
        fid.write('NODATA_value = %.2f \n' % (nodata_value))
        fid.write('TIME = 0 hours since  2013-10-31 00:00:00 +00:000') # set up the time component
        fid.write('### END OF HEADER\n')
    fid.close()
 
#%%
def rand_wind_field(size, amplitude, period):
    t = np.linspace(0, 2*np.pi, size)
    wind_speeds = amplitude * np.sin(2*np.pi / period * t)  # Generate the wind speeds using the sinusoidal function
    return wind_speeds
def generate_atmospheric_pressure(size, amplitude, period, min_pressure, max_pressure):
    t = np.linspace(0, 2*np.pi, size)  # Time parameter for the sinusoidal function
    pressure = ((max_pressure - min_pressure) / 2) * np.sin(2*np.pi / period * t) + ((max_pressure + min_pressure) / 2)
    # Generate the atmospheric pressure within the desired range using the sinusoidal function
    return pressure

size = 10  # Number of data points
amplitude = 5  # Maximum wind speed amplitude (half of the desired range)
period = 0.006  # Period of the oscillation (in data points)

wind_field = rand_wind_field(size, amplitude, period)
plt.plot(wind_field)

size = 100  # Number of data points
amplitude = (1050 - 990) / 2  # Half of the desired pressure range
period = 24  # Period of the oscillation (in data points)
min_pressure = 990  # Minimum atmospheric pressure
max_pressure = 1050  # Maximum atmospheric pressure

atmospheric_pressure = generate_atmospheric_pressure(size, amplitude, period, min_pressure, max_pressure)
plt.plot(atmospheric_pressure)

#%% Main 
if __name__ == '__main__':
    
    nlon = 400
    nlat = 140
    maxlon = -2.5
    minlon = -4
    minlat = 53
    maxlat = 55
    met_files = (u_name,v_name,p_name)
    for file in met_files:
        print(file)
        write_met_header(file, nlon, nlat, minlon, maxlon, minlat, maxlat)
    
#(-3.65,-2.75),(53.20,54.52)