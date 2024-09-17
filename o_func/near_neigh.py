#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Nearest Neigbour Algorithm with haversine BallTree

To use;
ensure that two dataframes are used where the first column (0) is x (longitude)
ensure that the second column of each dataframe (1) is y (latitude) of points to be found

set in the format where;
points is the df_locs to seach with, i.e. two locations

df_area is the dataframe to be searched

They should both be set into pandas dataframes called like so 

df[["col0_name", "col1_name"]]


Created on Fri Oct 21 11:47:56 2022
@author: af
"""
import time
import scipy
from sklearn.neighbors import BallTree
import numpy as np
from math import cos, asin, sqrt

def near_neigh(df_loc,df_area,k):
    '''
    x and y must be the first 2 columns, other columns do not matter. 
    Very important that they are both lower case. 

    Parameters
    ----------
    df_loc : Points that you wish to be searched through a larger grid
    df_area : The larger grid
    k : How many points do you want to find close to the target point, 1,2,3 etc. 

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

def data_flattener(x_array,y_array):

    combined_x_y_arrays = np.dstack([y_array.ravel(),x_array.ravel()])[0]
    # # Create some dummy data
    # y_array = np.random.random(10000).reshape(100,100)
    # x_array = np.random.random(10000).reshape(100,100)
    
    # points = np.random.random(10000).reshape(2,5000)
    
    # # Shoe-horn existing data for entry into KDTree routines
    # combined_x_y_arrays = np.dstack([y_array.ravel(),x_array.ravel()])[0]
    # points_list = list(points.transpose())
    
    
    # def do_kdtree(combined_x_y_arrays,points):
    #     mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    #     dist, indexes = mytree.query(points)
    #     return dist,indexes
        
    
    # start = time.time()
    # dist,indx = do_kdtree(combined_x_y_arrays,points_list)
    # end = time.time()
    # print ('Completed in: '+ str(end-start))
    return combined_x_y_arrays  

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))