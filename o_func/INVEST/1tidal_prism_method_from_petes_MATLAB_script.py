# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:19:50 2025

@author: aafur
"""

from pathlib import Path

# Define the path to the data files
path = Path(r'D:\Original_Data\peter_salinity_matlab_INVEST\INVEST_flushing_residence_times\50m')
path_for_train = Path(r'/Volumes/PN/1temp_ELEMENTS_drive_files/Original_Data/peter_salinity_matlab_INVEST/INVEST_flushing_residence_times/50m')
path = path_for_train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

[plt.close() for i in range(10)]
# Load data from CSV files
S0 = pd.read_csv(path / Path(r'50m_RIBBLE_transect_SAL.csv'))
H0 = pd.read_csv(path / Path(r'50m_RIBBLE_transect_DEPTH.csv'))
U0 = pd.read_csv(path / Path(r'50m_RIBBLE_transect_velX.csv'))
V0 = pd.read_csv(path / Path(r'50m_RIBBLE_transect_velY.csv'))
T0 = pd.read_csv(path / Path(r'50m_RIBBLE_transect_TIDE.csv'))
D0 = pd.read_csv(path / Path(r'50m_RIBBLE_transect_DIS.csv'))

# Extract data arrays, excluding the first column (e.g., timestamps or labels)
S = S0.iloc[:, 1:].values
H = H0.iloc[:, 1:].values
U = U0.iloc[:, 1:].values
V = V0.iloc[:, 1:].values
T = T0.iloc[:, 1:].values
D = D0.iloc[:, 1:].values
# Calculate cross-section-averaged salinity
DY = 50  # Transect spacing in meters
DT = 3600  # Time interval in seconds
SF = []  # Salt flux list
F = []  # Flux list
Sbar = []  # Cross-section-averaged salinity

for it in range(len(H)):
    # Calculate salt flux and flux for each transect point
    SF_tmp = [S[it, ix] * H[it, ix] * U[it, ix] * DY * DT for ix in range(len(H[it]))]
    F_tmp = [H[it, ix] * U[it, ix] * DY * DT for ix in range(len(H[it]))]
    SF.append(sum(SF_tmp))  # Sum salt flux across transect
    F.append(sum(F_tmp))  # Sum flux across transect
    Sbar.append(SF[-1] / F[-1])  # Calculate cross-section-averaged salinity

# Alternative method for cross-section-averaged salinity (simple mean)
Sbar2 = np.mean(S, axis=1)
# Moving average of Sbar2 for smoothing
SBar2_movmean = pd.Series(Sbar2).rolling(window=25).mean()

# Plot salinity results
plt.figure()
plt.plot(Sbar, label='Sbar')
plt.plot(Sbar2, label='Sbar2')
plt.plot(SBar2_movmean, label='movmean(Sbar2)', linewidth=2)
plt.axis([0, 3000, 30, 34])
plt.legend()
plt.title('Ribble (M2-tide, Q=50 m3/s)')
plt.show()

# Calculate tidal prism and flushing time
range_upper = np.max(T)  # Maximum tidal range
range_lower = np.min(T)  # Minimum tidal range
area = 45276019.18  # Cross-sectional area in square meters
diff = abs(range_upper) + abs(range_lower)  # Total tidal range

# Calculate tidal prisms
                      #MSL volume 
# spring_tidal_prism = 61130717 + (area * range_upper)  # Spring tidal prism
spring_tidal_prism = 61130717 + (area * range_upper)  # Spring tidal prism

neap_tidal_prism = 61130717 - (area * range_lower)  # Neap tidal prism

#%%
# Volume and tidal prism calculations
Q_tidal = (np.sum(D, axis=1)*3600 )- (50 * 3600)  # Q_t will have shape (2880,) 
Q_surface_height = np.mean(T, axis=1) # timeseries of the surface height elevation
from scipy.signal import find_peaks

# Find high tides (peaks) and low tides (troughs)
high_tide_indices, _ = find_peaks(Q_tidal)  # Positive peaks
low_tide_indices, _ = find_peaks(-Q_tidal)  # Negative peaks

tidal_prisms = []  # List to store tidal prisms for each cycle

for i in range(len(low_tide_indices) - 1):
    start = low_tide_indices[i]
    end = low_tide_indices[i + 1]
    # Integrate using the trapezoidal rule
    prism = np.trapezoid(np.abs(Q_tidal[start:end+1]), dx=1)  # dx=1 hour
    tidal_prisms.append(prism)

tidal_prisms = np.array(tidal_prisms)
prism = tidal_prisms.mean()


V = 61130717  # Total volume (m^3) @ MSL 
Vs = V * 0.7  # Estimated spring tidal prism volume
V_ls = V - Vs  # Low spring tide volume
V_hs = V + Vs  # High spring tide volume
TPs = V_hs - V_ls  # Spring tidal prism volume
Vn = V * 0.15  # Estimated neap tidal prism volume
V_ln = V - Vn  # Low neap tide volume
V_hn = V + Vn  # High neap tide volume
TPn = V_hn - V_ln  # Neap tidal prism volume
Tm2 = 12.42 * 3600  # M2 tidal period in seconds
b = 0.885  # Flushing efficiency factor

# River flow rates (m^3/s)
Q = np.array([2, 5, 10, 20, 30, 40, 50])
Tf_TPs = []  # Flushing time for spring tides
Tf_TPn = []  # Flushing time for neap tides
Tf_TPm = []
#%
for q in Q:
    # Adjust tidal prism with river flow
    TPs = (V_hs - V_ls) + (q * Tm2)
    TPn = (V_hn - V_ln) + (q * Tm2)
    TPm = (prism) + + (q * Tm2) 
    Tf_TPs.append((V / (TPs * (1 - b)) * Tm2) / ( 3600))  # Spring flushing time in days
    Tf_TPn.append((V / (TPn * (1 - b)) * Tm2) / ( 3600))  # Neap flushing time in days
    Tf_TPm.append((V / (TPm * (1 - b)) * Tm2) / ( 3600)) 
# Plot flushing times for Ribble
plt.figure()
# plt.plot(Q, Tf_TPs, '^-', linewidth=2, label='High Tide')
# plt.plot(Q, Tf_TPn, '^-', linewidth=2, label='Low Tide')
plt.plot(Q, Tf_TPm, '^-', linewidth=2, label='Tidal Prism Over M2 Tide')

ft_delft = [72,72,72,71,71]
ft_river = [10,20,30,40,50]

plt.plot(ft_river, ft_delft, '^-', linewidth=2, label='Delft Res Times Over M2 Tide')

plt.legend()
plt.grid()
plt.axis([0, 55, 0,100])
plt.ylabel('Flushing Time (Hours)')
plt.xlabel('River Discharges (m$^3$/s)')
plt.title('Ribble')
plt.show()
print(Tf_TPm)
#%%
# Wyre estuary calculations
V = 61130717 * 0.2  # Reduced volume for Wyre estuary
Vs = V * 0.7  # Estimated spring tidal prism volume
V_ls = V - Vs  # Low spring tide volume
V_hs = V + Vs  # High spring tide volume
TPs = V_hs - V_ls  # Spring tidal prism volume
Vn = V * 0.15  # Estimated neap tidal prism volume
V_ln = V - Vn  # Low neap tide volume
V_hn = V + Vn  # High neap tide volume
TPn = V_hn - V_ln  # Neap tidal prism volume
Tm2 = 12.42 * 3600  # M2 tidal period in seconds
b = 0.8  # Flushing efficiency factor

Tf_TPs = []  # Flushing time for spring tides
Tf_TPn = []  # Flushing time for neap tides

for q in Q:
    # Adjust tidal prism with river flow
    TPs = (V_hs - V_ls) + (q * Tm2)
    TPn = (V_hn - V_ln) + (q * Tm2)
    Tf_TPs.append((V / (TPs * (1 - b)) * Tm2) / (24 * 3600))  # Spring flushing time in days
    Tf_TPn.append((V / (TPn * (1 - b)) * Tm2) / (24 * 3600))  # Neap flushing time in days

# Plot flushing times for Wyre
plt.figure()
plt.plot(Q, Tf_TPs, '^-', linewidth=2, label='Spring (TP)')
plt.plot(Q, Tf_TPn, '^-', linewidth=2, label='Neap (TP)')
plt.legend()
plt.grid()
plt.axis([0, 100, 0, 25])
plt.title('Wyre')
plt.show()

# Mersey estuary calculations
V = 391630951  # Total volume for Mersey estuary (m^3)
Vs = V * 0.7  # Estimated spring tidal prism volume
V_ls = V - Vs  # Low spring tide volume
V_hs = V + Vs  # High spring tide volume
TPs = V_hs - V_ls  # Spring tidal prism volume
Vn = V * 0.4  # Estimated neap tidal prism volume
V_ln = V - Vn  # Low neap tide volume
V_hn = V + Vn  # High neap tide volume
TPn = V_hn - V_ln  # Neap tidal prism volume
Tm2 = 12.42 * 3600  # M2 tidal period in seconds
b = 0.97  # Flushing efficiency factor

Tf_TPs = []  # Flushing time for spring tides
Tf_TPn = []  # Flushing time for neap tides
Tf_SBs = []  # Flushing time based on salinity balance (spring)
Tf_SBn = []  # Flushing time based on salinity balance (neap)

for q in Q:
    # Adjust tidal prism with river flow
    TPs = (V_hs - V_ls) + (q * Tm2)
    TPn = (V_hn - V_ln) + (q * Tm2)
    Tf_TPs.append((V / (TPs * (1 - b)) * Tm2) / (24 * 3600))  # Spring flushing time in days
    Tf_TPn.append((V / (TPn * (1 - b)) * Tm2) / (24 * 3600))  # Neap flushing time in days
    Sin = 33 - q / 50  # Inflow salinity (practical salinity units)
    Sout = 28 - q / 50  # Outflow salinity (practical salinity units)
    Tf_SBs.append((V * 1.2 * (Sin - Sout)) / (q * Sin) / (3600 * 24))  # Spring flushing time in days (salinity balance)
    Tf_SBn.append((V * 0.8 * (Sin - Sout)) / (q * Sin) / (3600 * 24))  # Neap flushing time in days (salinity balance)

# Plot flushing times for Mersey
plt.figure()
plt.plot(Q, Tf_TPs, '^-', linewidth=2, label='Spring (TP)')
plt.plot(Q, Tf_TPn, '^-', linewidth=2, label='Neap (TP)')
plt.plot(Q, Tf_SBs, '^-', linewidth=2, label='Spring (SB)')
plt.plot(Q, Tf_SBn, '^-', linewidth=2, label='Neap (SB)')
plt.legend()
plt.grid()
plt.axis([0, 100, 0, 25])
plt.title('Mersey')
plt.show()
