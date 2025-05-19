#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This is a selection of functions to handle statistics for usage later

Statistics included are:
    - PDF - probability density functions
        Usage: path = path to points to compute the PDF

Created on Wed Nov 23 15:27:29 2022
@author: af
"""
#%% Import dependecies
import math
import glob
import numpy as np
import xarray as xr
import pandas as pd
from o_functions.choices import *
from o_functions.value_extractor import d_3dfm_mapped_ukc3

#%% Constants
pi = math.pi
e = math.e
k = 1

#%% Folder generation 
def fol_gen(path):
    name = r'point_directory_'
    paths = glob.glob(path)

#%% Probability Density Function
def PDF(start_path,pdf_type):
    '''
    Parameters
    ----------
    data_array : is a an array of series taken from a series of points that can be 
    split up with the seperator function.
    seperator : this defines how many lines in the timeseries of points denote 
    how they are seperated

    Returns
    -------
    new_pdf_array : a probability density distribution of the entered data_array
    which can then be plotted. 

    '''
    folder_path = tim_at_point_choice(start_path)
    
    # if folder_path.split('/')[-1] != 'All':
    #     print('You have selected a single UKC3 run')
    #     print('Continues to run code ...')

           
    data_array = np.load(folder_path + r'/data_array.npy', allow_pickle=True)
    seperator = np.load(folder_path + r'/seperator.npy', allow_pickle=True)
    name_uni = np.load(folder_path + r'/name_uni.npy', allow_pickle=True)
    
    pdf_array = []
    for row in data_array:
        m_n = np.mean(row)
        sigma = np.std(row) # standard deviation of each row in array
        if pdf_type == ('normal'):
            pdf = ( (1) / (sigma * (math.sqrt(2*pi))) ) * e ** (-1 * ( row - m_n )**2 / ( 2 * sigma**2 ))
        elif pdf_type == ('weibull'):
            # pdf = []
            # value became row
            pdf = ( (k)/(math.sqrt(2*sigma)) * e ** (-1*(row/(math.sqrt(2*sigma))**k)) )

            # for value in row:
                # if value >= 0:
                    # val_pdf = ( (k)/(math.sqrt(2*sigma)) * e ** (-1*(value/(math.sqrt(2*sigma))**k)) )
                    # pdf.append(val_pdf)
                # elif value < 0:
                    # val_pdf = 0
                    # pdf.append(val_pdf)
                # else:
                    # pdf.append(np.nan)
            # pdf = np.array(pdf)
        pdf_array.append(pdf)
    
    pdf_array = np.vstack(pdf_array)
    
    new_data_array = np.array_split(data_array,np.cumsum(seperator))[0:8]
    # create an array of lists where each item is an estuary
    # each estuary then has a different number of points and each point has a timeseries.
    new_pdf_array = np.array_split(pdf_array,np.cumsum(seperator))[0:8]
        
    sh_data_array = d_3dfm_mapped_ukc3(start_path)  

    pdf_array = []
    for row in sh_data_array:
        m_n = np.mean(row)
        sigma = np.std(row)
        if pdf_type == ('normal'):
            pdf = ( (1) / (sigma * (math.sqrt(2*pi))) ) * e ** (-1 * ( row - m_n )**2 / ( 2 * sigma**2 ))
        elif pdf_type == ('weibull'):
            pdf = []
            pdf = ( (k)/(math.sqrt(2*sigma)) * e ** (-1*(row/(math.sqrt(2*sigma))**k)) )

            # for value in row:
                # if value >= 0:
                    # val_pdf = ( (k)/(math.sqrt(2*sigma)) * e ** (-1*(value/(math.sqrt(2*sigma))**k)) )
                #     pdf.append(val_pdf)
                # elif value < 0:
                #     val_pdf = 0
                #     pdf.append(val_pdf)
                # else:
                    # pdf.append(np.nan)
            # pdf = np.array(pdf)
        pdf_array.append(pdf)
    
    pdf_array = np.vstack(pdf_array)
    
    sh_new_data_array = np.array_split(sh_data_array,np.cumsum(seperator))[0:8]
    sh_new_pdf_array = np.array_split(pdf_array,np.cumsum(seperator))[0:8]
    
    cols = ['Point {}'.format(col) for col in range(1, 8)]
    rows = ['{} Est \nPDF'.format(row) for row in name_uni]
    
    return cols,rows, new_pdf_array, new_data_array, sh_new_data_array, sh_new_pdf_array       

def calculate_hourly_means(df):
    """
    Calculate hourly means for a time series DataFrame with timestamps at the middle of each hour.

    Args:
        df (pd.DataFrame): Input DataFrame with DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with hourly means and timestamps at the middle of each hour.
    """
    hourly_means = df.resample('H').mean()
    hourly_means.index = hourly_means.index + pd.Timedelta(minutes=30)

    return hourly_means
    
#%% Testing
'''
conda install -c conda-forge pygobject   to install gi
conda install -c conda-forge gtk3   to install gtk
'''
