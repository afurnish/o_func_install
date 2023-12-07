#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:03:02 2023

@author: af
"""

from o_func.utilities.start import opsys; start_path = opsys()

from os.path import join
import pandas as pd


def tidconst():
    def process_string(input_str):
        # Remove trailing spaces
        stripped_str = input_str.strip()
    
        # Pad with spaces at the front
        padded_str = f"{stripped_str: >4}"
    
        return padded_str
    
    path = join(start_path, 'modelling_DATA/kent_estuary_project/tidal_boundary/tidal_conts_&_freqs.csv')
    # 
       
    tc = pd.read_csv(path)
    
    tc['Names'] = tc['Names'].apply(lambda x: process_string(x))
    tc['Names_lower'] = tc['Names_lower'].apply(lambda x: process_string(x))
    return tc
