#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Running script to prep data. 
'''
#%% Importing Dependecies
import os
import xarray as xr
import glob
import time
import pandas as pd

from o_func import DataChoice, DirGen, opsys; start_path = opsys()

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

sp = time.time()
#lp = glob.glob(os.path.join(fn[0],'*.nc'))[0]
#Make chunk sizes 10 if it all goes to pot 
main_dataset = xr.open_dataset(lp, 
                   chunks={'time': 'auto', 
                   'mesh2d_face_x_bnd': 'auto',
                   'mesh2d_face_y_bnd': 'auto'
                   }, 
                   engine = 'scipy'
                   )


yyy = main_dataset.mesh2d_face_x_bnd.mesh2d_face_y
xxx = main_dataset.mesh2d_face_x_bnd.mesh2d_face_x

time = main_dataset.coords['time']
fs = 20

wd  = main_dataset.mesh2d_waterdepth    #waterdepth              (m)
sh  = main_dataset.mesh2d_s1            #water surface height    (m)
sal = main_dataset.mesh2d_sa1           #salinity               (psu)
