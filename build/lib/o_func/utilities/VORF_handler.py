# -*- coding: utf-8 -*-
""" Reformat tidal data downloaded from this website - 
https://www.bodc.ac.uk/data/hosted_data_systems/sea_level/uk_tide_gauge_network/processed/

-- Use this to process the tidal data into a sensible format. 

Created on Thu Mar  9 15:03:29 2023
@author: aafur
"""
#%% dependencies 
import pandas as pd
from o_func import opsys;  start_path = opsys()
import glob
import numpy as np
from math import cos, asin, sqrt
from sklearn.neighbors import BallTree

path = r'modelling_DATA/kent_estuary_project/validation/tidal_validation'
main_path = start_path + path
new_folder = r'/1.reformatted'

path2 = r'modelling_DATA/kent_estuary_project/bathymetry/bathy_files'
VORF_csv = start_path + path2 +  r'/original_bathy/VORF_UK.csv'


regex = r'^\s*(\d+\))\s+(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)$'

#%% Haversine Distance Equation

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))

def nearest_neighbour(data, v):
    return min(data, key = lambda p:distance(v['lat'],v['lat'],p['lat'],p['lat']))

#%% Process the data with VORF algorithm 
#we're gonna go straight from CD to MSL rather than going from CD to ODN

from o_func.utilities.gauges import tide_gauge_loc
tg = tide_gauge_loc()
# heysham = [-2.92025,54.031833]
# liverpool = [-3.018,53.449694]
workington = [-3.567167,54.650722]

heysham = [tg['Heysham']['x'],tg['Heysham']['y']]
liverpool = [tg['Liverpool']['x'],tg['Liverpool']['y']]



header_list = ["x", "y", "z"]
df_VORF = pd.read_csv(VORF_csv, delimiter = ',')#,  names=header_list_vorf)

df_tide = pd.DataFrame([workington,heysham,liverpool], columns = ['x','y'])

for column in df_VORF[["x", "y"]]:            #a
    rad = np.deg2rad(df_VORF[column].values)
    df_VORF[f'{column}_rad'] = rad
for column in df_tide[["x", "y"]]:
    rad = np.deg2rad(df_tide[column].values) #b
    df_tide[f'{column}_rad'] = rad

ball = BallTree(df_VORF[["x_rad", "y_rad"]].values, metric='haversine') # generate haversine ball tree
k = 2 # number of nearest neigbours to point
distances, indices = ball.query(df_tide[["x_rad", "y_rad"]].values, k = k)
# generate nearest distances and indices. 

vor_z = df_VORF['z'][indices[0]]    

#%% Output the data

tide_gauges = []
j = -1
for tide_gauge in glob.glob(main_path + '/3.raw_data/*'):
    j += 1
    n = tide_gauge.replace("\\", "/")
    tide_gauges.append(n)
    name = n.split('/')[-1]
    data = []
    tide = pd.DataFrame()
    
    for i in glob.glob(n + '/*.txt'):
        adj = i.replace("\\","/")
        data.append(adj)
        df = pd.read_csv(adj, header = 10, sep = '\\n', names = ['raw_data'], engine = 'python')
        df = pd.DataFrame(df['raw_data'].str.extract(regex))
        df[2] = df[2].str.replace('[^\d\.]+', '', regex = True)
        df[3] = df[3].str.replace('[^\d\.]+', '', regex = True)
        df = df.rename(columns = {0:'index',1:'Date',2:'Height',3:'residuals'})
        df2 = df[['Date']]
        df2.loc[:,'Height'] = df['Height'].astype(float)
        tide = pd.concat([tide,df2], axis = 0)
    vor_z = np.max( df_VORF['z'][indices[j]] )  
    tide.Height = tide.Height - vor_z
    tide.to_csv(main_path + '/1.reformatted/' + name + '.csv', index = False)

    












# tide_gauges = []
# for tide_gauge in glob.glob(main_path + '/3.raw_data/*'):
#     n = tide_gauge.replace("\\", "/")
#     tide_gauges.append(n)
#     name = n.split('/')[-1]
#     data = []
#     tide = pd.DataFrame()
#     if name == 'liverpool':
#         adjustment = liverpool
#     elif name == 'heysham':
#         adjustment = heysham
#     elif name == 'workington':
#         adjustment = workington
#     for i in glob.glob(n + '/*.txt'):
#         adj = i.replace("\\","/")
#         data.append(adj)
#         df = pd.read_csv(adj, header = 10, sep = '\\n', names = ['raw_data'], engine = 'python')
#         df = pd.DataFrame(df['raw_data'].str.extract(regex))
#         df[2] = df[2].str.replace('[^\d\.]+', '', regex = True)
#         df[3] = df[3].str.replace('[^\d\.]+', '', regex = True)
#         df = df.rename(columns = {0:'index',1:'Date',2:'Height',3:'residuals'})
#         df2 = df[['Date','Height']]
#         df2['Height'] = df2['Height'].astype(float) + adjustment
#         tide = pd.concat([tide,df2], axis = 0)
#     tide.to_csv(main_path + '/1.reformatted/' + name + '.csv', index = False)

    


