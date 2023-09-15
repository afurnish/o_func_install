#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Running script to prep data. 
'''
#%% Importing Dependecies


import os
import xarray as xr
import glob
import time as t 

from o_func import DataChoice, DirGen ,VideoPlots, opsys; start_path = opsys()
from o_func.data_prepkit import OpenNc

#%% Making Directory paths
main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
make_paths = DirGen(main_path)
sub_path = make_paths.dir_outputs('kent_2.0.0_no_wind')
### Finishing directory paths

dc = DataChoice(os.path.join(main_path,'models'))
fn = dc.dir_select()
#%% Lazy loading dataset
### Large dataset path for testing 
lp = glob.glob(r'F:\modelling_DATA\kent_estuary_project\5.Final\1.friction\SCW_runs\kent_2.0.0_wind\*.nc')[0]

sp = t.time()
#lp = glob.glob(os.path.join(fn[0],'*.nc'))[0]
#Make chunk sizes 10 if it all goes to pot 
main_dataset = xr.open_dataset(lp, 
                   chunks={'time': 'auto', 
                   'mesh2d_face_x_bnd': 'auto',
                   'mesh2d_face_y_bnd': 'auto'
                   }, 
                   engine = 'scipy'
                   )
print('Data Loaded in :', t.time() - sp ,'seconds')

yyy = main_dataset.mesh2d_face_x_bnd.mesh2d_face_y
xxx = main_dataset.mesh2d_face_x_bnd.mesh2d_face_x

time = main_dataset.coords['time']
fs = 20

wd  = main_dataset.mesh2d_waterdepth    #waterdepth              (m)
sh  = main_dataset.mesh2d_s1            #water surface height    (m)
sal = main_dataset.mesh2d_sa1           #salinity               (psu)

# Limits for colourbar, colourbar labels, longitude and latitude. 
wd_bounds = [(-4,5,70),(-4,5,16),(-3.65,-2.75),(53.20,54.52)]
sal_bounds = [(0,40,80),(0,35,8),(-3.65,-2.75),(53.20,54.52)]



# Class to split the dataset into a managable chunk. 
start_slice = OpenNc()
new_data = start_slice.slice_nc(main_dataset)

#%% Vid prep
png_sh_path = make_paths.vid_var_path(var_choice='Surface_Height')
png_sal_path = make_paths.vid_var_path(var_choice='Salinity')


pv = VideoPlots(dataset = new_data.surface_height,
                xxx     = new_data.mesh2d_face_x,
                yyy     = new_data.mesh2d_face_y,
                bounds  = wd_bounds,
                path    = png_sh_path
                )
starrrr = t.time()
make_videos2 = pv.joblib_para(num_of_figs=400, land = 0.07, vel = main_dataset.mesh2d_ucmag) # colour plots
print('Time between sh: ', t.time() - starrrr)
pv.bathy_plot()

pvs = VideoPlots(dataset = new_data.salinity,
                xxx     = new_data.mesh2d_face_x,
                yyy     = new_data.mesh2d_face_y,
                bounds  = sal_bounds,
                path    = png_sal_path
                )
starrrr = t.time()
make_videos2 = pvs.joblib_para(num_of_figs=400) # colour plots
print('Time between sal: ', t.time() - starrrr)





