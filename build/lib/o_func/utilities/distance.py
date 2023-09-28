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

    def find_nearest(array, value, index):
        '''
        Find the nearest possible number to a value(s) within an array. E.g. You 
        may be looking for 1.32 but the arrays nearest is 1.311. That number will 
        become your desired variable. 
        
        Values should be presented as a list or array 
        
        i = return index
        v = return value 
        Parameters
        ----------
        array : A numpy style array of values. Ideally 1x. 
        value : A single value, float or interger not string to be searched. 

        Returns
        -------
        The array index is returned for that value in the array. 

        '''
        def isarray(array):
            if not isinstance (array, np.ndarray):
                array = np.asarray(array)
                return array
            else:
                return array
        def isvalue(value):
            if len(value) == 1:
                if not isinstance (value, list):
                    return [value]
            else:
                return value
            
        new_value = isvalue(value)
        new_array = isarray(array)
        idx = []
        print(new_value)
        for i in new_value:
            
            print('i is ', i)
            
            idx.append((np.abs(new_array - i)).argmin())
        
        
        if index == 'v':
            return array[idx]
        elif index == 'i':
            return idx
    
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
    
    
def uk_bounds():
    '''
    Returns general boundaries of the PRIMEA grid intersections
    '''
    return [(-3.65,-2.75),(53.20,54.52)]

def uk_bounds_wide():
    '''
    Returns general boundaries of the PRIMEA grid intersections
    '''
    return [(-4,-2.5),(52,55)]

