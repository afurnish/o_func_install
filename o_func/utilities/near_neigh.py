#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:06:47 2023

@author: af
"""
from sklearn.neighbors import BallTree

import numpy as np
def data_flattener(x_array,y_array):
    combined_x_y_arrays = np.dstack([y_array.ravel(),x_array.ravel()])[0]
    return combined_x_y_arrays  

def near_neigh(df_loc,df_area,k):
    '''
    x and y must be the first 2 columns, other columns do not matter. 
    Very important that they are both lower case. 

    Parameters
    ----------
    df_loc : TYPE
        DESCRIPTION.
    df_area : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    distances : TYPE
        DESCRIPTION.
    indicies : TYPE
        DESCRIPTION.
        
    Example:
        df_locs = locations to be found
              x          y
        0 -3.016825  53.430732
        1 -3.642720  54.671460
        2 -2.931176  54.034552
        
        df_area = area of locations to be searched
                    x          y
        0     -2.839762  54.204873
        1     -2.840894  54.205005
        2     -2.841358  54.205134
        3     -2.842000  54.205760
        4     -2.839240  54.204919
                ...        ...
        69861 -3.275970  53.487347
        69862 -3.266266  53.497516
        69863 -3.269990  53.492205
        69864 -3.269976  53.489259
        69865 -3.265208  53.494281

    '''
    
    
    for column in df_area.iloc[:, :2]: 
        #print(column)           #a
        rad = np.deg2rad(df_area[column].values)
        df_area[f'{column}_rad'] = rad
    for column in df_loc.iloc[:, :2]:
        #print(column) 
        rad = np.deg2rad(df_loc[column].values) #b
        df_loc[f'{column}_rad'] = rad

    # generate haversine ball tree
    ball = BallTree(df_area[["x_rad", "y_rad"]].values, metric='haversine')
        
    # default was 2
    k = k # number of nearest neigbours to point
    distances, indicies = ball.query(df_loc[["x_rad", "y_rad"]].values, k = k)
    
    return distances, indicies