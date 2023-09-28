#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:06:47 2023

@author: af
"""

import numpy as np
def data_flattener(x_array,y_array):
    combined_x_y_arrays = np.dstack([y_array.ravel(),x_array.ravel()])[0]
    return combined_x_y_arrays  
