#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:59:06 2024

@author: af
"""
import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from datetime import datetime, timedelta
from o_func import opsys; start_path = Path(opsys())

#%% input parameters
path = start_path / Path('Original_Data/UKC3/og/shelftmb_combined_to_3_layers_for_tmb')
river_path = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/original_river_data/processed')
writemaps = 'y'
side = 'east'
include_tidal = True