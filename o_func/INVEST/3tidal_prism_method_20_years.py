#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:00:43 2025

@author: af
"""

from scipy.io import loadmat
import numpy as np

ribble_data = '/Volumes/PN/1temp_ELEMENTS_drive_files/temp/Ribble_rivers.mat'

matdataset = loadmat(ribble_data)

Q = matdataset['Q_projections']

pres_data = Q['pres']
numerical_data = np.concatenate([np.array(x) for x in pres_data.flat if isinstance(x, np.ndarray)])
extracted_data = numerical_data['raw_series'][0,0]






