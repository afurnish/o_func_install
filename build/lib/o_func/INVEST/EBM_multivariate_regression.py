#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:25:26 2024

@author: af
"""
def run():
    # Conditional calculation of C_k
    if Q_r > Q_m:
        C_k = (Ro_s / 1000)**20 * (vel_tide / ur)**-0.5 * np.exp(-2000 * Eta)
    else:
        C_k = 200 * (Ro_s / 1000)**20 * (vel_tide / ur)**0.1 * np.exp(-2000 * Eta)
    
    print(C_k)
    return C_k

g = 9.81  # gravity (m/s²)

#%% The original 
import numpy as np

# Constants and inputs
Q_r = 115.57  # river inflow (m³/s)
Q_m = 240  # average river discharge (m³/s)
S_l = 35.95  # lower layer salinity (psu)
vel_tide = 0.027  # tidal velocity (m/s)
W_m = 200  # estuary width (m)
h = 5  # estuary height (m)

# Calculate ur
ur = Q_r / ((h / 2) * W_m) # river velocity is much higher here. 

# Calculate Ro_s
Ro_s = 1000 * (1 + (7.7 * (1E-4) * S_l))

# Calculate Fr_box
Fr_box = vel_tide / np.sqrt(h * g)

# Calculate Eta
Eta = (Fr_box**2) * (W_m / h)

C_k = run()

#%% Now to run it for one of my examples. However when C_k is set to 100, 

# Constants and inputs
for W_m, h in zip([ 8000,  4860, 10000,   540,  1770,   530, 10500,  4900],[4.75, 5.  , 7.  , 2.  , 6.25, 0.6 , 5.  , 3.  ]):
    # print(W_m)
    # print(h)
    W_m = 200
    # h = 5
    Q_r = 50  # river inflow (m³/s)
    Q_m = 51  # average river discharge (m³/s)
    S_l = 35.95  # lower layer salinity (psu)
    vel_tide = 0.8  # tidal velocity (m/s)
    
    # Calculate ur
    ur = Q_r / ((h / 2) * W_m) # river velocity is much slower here. But not the main issue. 
    
    # Calculate Ro_s
    Ro_s = 1000 * (1 + (7.7 * (1E-4) * S_l))
    
    # Calculate Fr_box
    Fr_box = vel_tide / np.sqrt(h * g)
    
    # Calculate Eta
    Eta = (Fr_box**2) * (W_m / h)
    
    
    if Q_r > Q_m:
        C_k = (Ro_s / 1000)**20 * (vel_tide / ur)**-0.5 * np.exp(-2000 * Eta)
    else:
        C_k = 200 * (Ro_s / 1000)**20 * (vel_tide / ur)**0.1 * np.exp(-2000 * Eta)
    
    print(C_k)