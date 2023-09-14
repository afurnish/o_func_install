# -*- coding: utf-8 -*-
""" File to make the various inputs for the set model. 


Created on Thu Sep 14 09:41:24 2023
@author: aafur
"""

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
sub_path = make_paths.dir_outputs('kent_1.0.0_UM_wind') # Dealing with this model run. 
### Finishing directory paths

dc = DataChoice(os.path.join(main_path,'models'))
fn = dc.dir_select()

#%% Need to generate multiple layer files. 