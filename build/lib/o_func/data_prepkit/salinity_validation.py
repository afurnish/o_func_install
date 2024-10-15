#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:59:31 2024

@author: af
"""
from pathlib import Path
data = Path('/Volumes/PN/Original_Data/salinity')

for file in data.glob('*.csv'):
    print(file)