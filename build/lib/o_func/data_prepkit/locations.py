#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:20:45 2024

@author: af
"""

def primea_bounds_for_ukc4_slicing():
    '''
    

    Returns
    -------
    bounds : These are the northerly and southerly bounds exact lon and lat as
    the grid for the NEMO file in UKC4. They can therefore be used to index 
    and slice datafiles for use within UKC4 files without relying on the actual
    index. 

    '''
    bounds = {'North':{'lon':-3.6295772, 'lat':54.488667},
              'South':{'lon':-3.5988138, 'lat':53.314304},
              # 'East' :{'lon':, 'lat':},
              # 'West' :{'lon':, 'lat':},
              }
    
    return bounds 

'''
lon[688,758]           # Northely point
Out[77]: -3.6295772

lon[601,758]           # Sourtherly Point
Out[78]: -3.5988138


lat[688,758]
Out[79]: 54.488667     #Northerly Point

lat[601,758]
Out[80]: 53.314304     #Southerly Point
'''