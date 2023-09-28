#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will generate statitistics and also process outputs, statistics go live in the stats folder. 


Created on Thu Sep 28 09:15:20 2023
@author: af
"""
import pandas as pd


class r_stats:
    def __init__(self):
     a = 1
    @staticmethod
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

    def raw_to_useful(self, prim_tim_df,ukc3_tim_df,var,index_only):
        '''
        Primary purpose is to allign time dimensions for accurate comparison. Data should be sampled. 
        Depending on the variable you may want to interpolate values between but only for plotting not for 
        calculating statistics. 
        
        
        Need to make sure for this function to work you load in the time coords for 
        primea and for ukc3 as well as the variable you wish to use:
            Choices:
                Surface Height = sh
                Salinity       = sal
                
        Here it is really important to claify what time and shape dimension the data is in,
        if the data is sampled in 10 minute intervals this works, however is hourly intervas are
        used everything breaks apart. Ensure the code is robust enough to handle hourly 
        and 10 minute outputs or anything you may decide to throw at it.
        
        # try not to let hourly stuff get to here on the hour as it messes the whole thing up. 
        '''
        
        
        
        
        
        
        
        
        
        
        
        
        