# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:50:58 2023

@author: aafur
"""
import numpy as np
from math import cos, asin, sqrt

class Dist:
    def __init__(self):
        pass

    def find_nearest(array, value):
        '''
        Find the nearest possible number to a value within an array. E.g. You 
        may be looking for 1.32 but the arrays nearest is 1.311. That number will 
        become your desired variable. 
        

        Parameters
        ----------
        array : A numpy style array of values. Ideally 1x. 
        value : A single value, float or interger not string to be searched. 

        Returns
        -------
        The array index is returned for that value in the array. 

        '''
        
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    def dist_between_points(lat1, lon1, lat2, lon2):
        '''
        Generate the kilometric distance between two points using the haversine equation. 

        Parameters
        ----------
        lat1 : Latitude of point 1
        lon1 : Longitude of point 1
        lat2 : Latitude of point 2
        lon2 : Latitude of point 2

        Returns
        -------
        Returns a distance in between two points on a shere.

        '''
        p = 0.017453292519943295
        hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
        return 12742 * asin(sqrt(hav))
    
