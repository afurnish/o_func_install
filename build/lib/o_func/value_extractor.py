#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Value Extractor Delft

- Script to extract values from hydrodyanmic delft 3d mesh suite 
for use in python or another computer coding software

-This script contains diff fucntions depending on the data type

Created on Sun Nov 13 12:34:59 2022
@author: af
"""
#%% import dependecies and set rc params

import os
import time
import glob
import xarray as xr
import pandas as pd
#from tkinter.filedialog import askdirectory
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from o_functions.shape import coastline, UKC3_area
from o_functions.choices import *
# from tkinter.filedialog import askopenfilename
from o_functions.near_neigh import near_neigh

#%% Functions
def ukc3_extract(start_path,folder_path,ext,var,points,maps,save):
    '''
    folder_path = full folder path including start path that could easily be defined from 
        tkinkter
    ext = T,U,V as 
    var = variable from ukc3 dataset to use, for guide see bottom of this python function
    points = path of point location with at least x and y for column 0,1
    maps = 'y' or 'n' for print location map or not
    save = string containing either 'no' or a series or strings whihc contain the folder 
        path and name of dataset
    '''
    full_path = folder_path + r'/*' + ext + r'.nc'
    name = (folder_path.split('/'))[-2] # gets name of model from folder_path
    # Essentially removes the .nc
    
    file_list = glob.glob(full_path)
    ukc3 = xr.open_dataset(file_list[0])
    
    attribute = getattr(ukc3,var)
    
    lon = attribute.nav_lon.values
    lat = attribute.nav_lat.values
    df = pd.read_csv(points)
    
    df2 = pd.DataFrame()
    combined_x_y_arrays = np.dstack([lon.ravel(),lat.ravel()])[0]
    df2["x"], df2["y"] = combined_x_y_arrays.T

    df.rename(columns = {'X':'x', 'Y':'y'}, inplace = True)
    
    dist, indi = near_neigh(df, df2)
    
    # converts everything back into latitude and longitiude indexing
    nn = [] #locations of points
    for i in range(len(indi)):
        nearest_neigh = divmod(indi[i,0],np.shape(attribute)[2]) #1458 is length of row, unravel ravel
        nn.append(nearest_neigh)

    names_df4 = ['id','Name','lat_index','lon_index']
    df4 = pd.DataFrame(np.nan, index=range(len(df)), columns=names_df4)
    df4['id'] = df['id']
    df4['Name'] = df['est_name']
    
    for i in range(len(nn)):
        df4.iloc[i,2] = nn[i][0] # the down component (y)
        df4.iloc[i,3] = nn[i][1] # the across component (x)

    df4.lat_index = df4.lat_index.astype(int) # ensure indexes are integers
    df4.lon_index = df4.lon_index.astype(int)
    # splitting dataframe by groups
    # grouping by particular dataframe column
    #grouped = df4.groupby(df4.id)
    #df_new = grouped.get_group(0)
    
    names_df3 = ['new_lon','new_lat']
    # this creates a frame of the new points vs old points
    df3 = pd.DataFrame(np.nan, index=range(len(df)), columns=names_df3)
    for column in names_df3:
        for ik in range(len(df)):
            df3.iloc[ik,0] = lon[nn[ik]]
            df3.iloc[ik,1] = lat[nn[ik]]
    df3['old_lon'] = df['x']
    df3['old_lat'] = df['y']
    
    if maps == 'n':
        start = time.time()
        time_names = []
        d = [] # list of filenames
        allArrays = [] # generate empty numpy array for hourly data
        print('Running over ' + str(len(sorted(glob.glob(full_path)))) + ' days' )
        num = 0
        for filename in sorted(file_list): # run through day files
            d.append(filename)
            data = xr.open_dataset(filename) # open each hourly dataset so we can stitch together names
            num = num + 1
            attribute = getattr(data,var)
            #capture each datetime string and append to list
            time_names.append((pd.to_datetime(attribute.time_counter.values)).strftime('%Y-%m-%d_%H:%M'))
            # All hours, one locaton in y and x
            hourly = attribute.values[:,df4.iloc[:,2],df4.iloc[:,3]]
            allArrays.append(hourly)
            near_end = time.time()
            near_total_time = near_end - start
            print("\n"+ 'Loop No:' + str(num)+ ' takes: ' + str(near_total_time) + ' seconds')
    
        end = time.time()
    
        total_time = end - start
        print("\n"+ 'TOTAL TIME WAS ' + str(total_time) + ' seconds')
    
        #%% Putting Data into useable format
        data_array_stacked = np.vstack(allArrays) #stitch together data
        time_array_stacked = np.vstack(time_names) #stitch together time data
        lin_time = time_array_stacked.ravel() # unravel time data from 161 X 24 into 3864
    
        df5 = pd.DataFrame()
        df5['time'] = [word.replace('_',' ') for word in lin_time ]
        df5['time'] = df5['time'].apply(lambda _: datetime.strptime(_,"%Y-%m-%d %H:%M"))
    
        time_check = pd.date_range( df5.iloc[0,0] , df5.iloc[-1,0] , freq="60min").difference(df5.time)
        if time_check.size == 0:
            print('\nNo time values missing')
        else:
            print('\nThere are time values missing \n')
            print(time_check)
    
        trans_data_array = data_array_stacked.transpose()
        
    
        # xx, yy, zz = UKC3_area(start_path)
        # data = coastline(start_path)
        # fig, ax = plt.subplots(figsize=(100,50))
        # plt.pcolor(xx,yy,zz)
        # data.plot(ax=ax, color="red")
        # ax.set_xlim(-3.65,-2.75)
        # ax.set_ylim(53.20,54.52)
        # plt.scatter(df3.old_lon, df3.old_lat, c = 'y') #original points
        # plt.scatter(df3.new_lon, df3.new_lat, c = 'c') #points interped to grid
        # text = list(range(151))
        # for i in text:
        #     text[i] = str(text[i])
        # for i in range(len(df3.new_lon)):
        #     plt.annotate(text[i], (df3.new_lon[i], df3.new_lat[i] + 0.002))
        # plt.scatter(xx,yy, c = 'white', marker='+')
        # plt.xlabel('Longitude')
        # plt.xlabel('Latitude')

        # Maybe just export whats needed to plot the points
     # save data to folder externally
        if save != 'n' or 'N' or 'no' or 'No' or 'NO':
            name_list = df4['Name'].value_counts()
            name_uni = df4['Name'].unique()
    
            seperator = []
            for x in name_uni:
                num = name_list[x]
                seperator.append(num)
            #np.save(save[0] + r'/' + save[1] + '.npy' ,trans_data_array) #e.g. owa
            np.save(save[0] + r'/' + r'data_array.npy' ,trans_data_array)
            np.save(save[0] + r'/' + r'seperator' + '.npy' ,(seperator))
            np.save(save[0] + r'/' + r'name_uni.npy', name_uni)
            np.save(save[0] + r'/' + r'df.npy', df)
            return lon, lat, name, trans_data_array
    else:
        return lon, lat, name, df3
    

def ukc3_extract_map():
    a = 1
    return a
    
def d_3dfm_mapped_ukc3(start_path):
    # print('\nSelect a delft 3dfm .nc output file')
    ASK = delft_file_chooser(start_path)
    # ASK = askopenfilename()
    map_do = xr.open_dataset(ASK)
    yyy = map_do.mesh2d_face_x_bnd.mesh2d_face_y.values #set x loc of data
    xxx = map_do.mesh2d_face_x_bnd.mesh2d_face_x.values #set y loc of data face loc

    sh = map_do.mesh2d_s1.values
    points = points_path(start_path)
    #points = askopenfilename()
    df = pd.read_csv(points)
    df.rename(columns = {'X':'x', 'Y':'y'}, inplace = True)

    # need to remap my mapping data onto the new UKC3 map points
    # what are the UKC3 map points in Lon and Lat
    # Interpolate my model points onto the UKC3 grid point 
    df2 = pd.DataFrame()
    lon = xxx
    lat = yyy
    combined_x_y_arrays = np.dstack([lon.ravel(),lat.ravel()])[0]
    df2["Lon"], df2["Lat"] = combined_x_y_arrays.T
    
    
    df.rename(columns = {'X':'x', 'Y':'y'}, inplace = True)
    df2.rename(columns = {'Lon':'x','Lat':'y'}, inplace = True)
    
    #use of near_neigh function
    dist, indi = near_neigh(df, df2)
    # indi[:,0] this is the variable to index sh to
    
    names_df3 = ['new_lon','new_lat']
    # this creates a frame of the new points vs old points
    df3 = pd.DataFrame(np.nan, index=range(len(df)), columns=names_df3)
    
    for column in names_df3:
        for ik in range(len(df)):
            df3.iloc[ik,0] = xxx[indi[ik,0]]
            df3.iloc[ik,1] = yyy[indi[ik,0]]
           
    
    df3['old_lon'] = df['x']
    df3['old_lat'] = df['y']
    df3['indi'] = indi[:,0]
    
    sh_data_array = []
    for i in df3.indi:
        nw_arra = sh[:,i] 
        sh_data_array.append(nw_arra)
    
    return sh_data_array

    
''' Attributes that cab be chosen for 
ukc3_extract:
    __T_Variables__
    votemper_mid
    votemper_bot
    sossheig = sea surface height
    vosaline_top = salinity surface layer
    vosaline_mid = salinity middle layer
    vosaline_bot = salinity bottom layer
    __U_Variables__

'''    