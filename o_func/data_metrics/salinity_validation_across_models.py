#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:04:47 2025

@author: af
"""
from pathlib import Path 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from o_func import opsys 
import os



start_path = Path(opsys('PNC'))

clim10 = start_path / 'modelling_DATA/kent_estuary_project/13.3D_finals/models/3d_10_layer_climatology_layer0/outputs/data_proc/salinity_validation_data.pkl'
realriv10 = start_path / 'modelling_DATA/kent_estuary_project/13.3D_finals/models/3d_10_layer_realriv_layer0/outputs/data_proc/salinity_validation_data.pkl'

fig_path = start_path / 'modelling_DATA/kent_estuary_project/river_boundary_conditions/figures'

with open(clim10, 'rb') as file:
    clim10_data = pickle.load(file)
    
with open(realriv10, 'rb') as file:
    realriv10_data = pickle.load(file)
    
#%% perform plotting operation

def _pair_stats(obs, mod):
    """RMSE, MAE, Bias, r, and OLS fit y = a*x + b for a model/obs pair."""
    obs = np.asarray(obs, float); mod = np.asarray(mod, float)
    m = np.isfinite(obs) & np.isfinite(mod)
    x, y = obs[m], mod[m]
    if x.size < 2:
        return dict(rmse=np.nan, mae=np.nan, bias=np.nan, r=np.nan, a=np.nan, b=np.nan)
    rmse = np.sqrt(np.mean((y - x) ** 2))
    mae  = np.mean(np.abs(y - x))
    bias = np.mean(y - x)
    r    = np.corrcoef(x, y)[0, 1]
    a, b = np.polyfit(x, y, 1)  # OLS: y = a*x + b
    return dict(rmse=rmse, mae=mae, bias=bias, r=r, a=a, b=b)

def salinity_overlay_single(clim10_data, realriv10_data, *,
                            xlim=(0, 35), ylim=(0, 35),
                            add_fits=True,
                            add_identity=True,
                            title='Surface salinity: models vs observations',
                            save=None, dpi=300):

    # ----- pull arrays -----
    # IRENE 10-layer, climatology forced
    obs_ic, mod_ic = clim10_data['obs_prim'],    clim10_data['mod_prim']
    # IRENE 10-layer, real-river forced
    obs_ir, mod_ir = realriv10_data['obs_prim'], realriv10_data['mod_prim']
    # UKC4 3-layer, climatology forced
    obs_uk, mod_uk = clim10_data['obs_ukc4'],    clim10_data['mod_ukc4']

    # ----- stats per pair -----
    S_ic = _pair_stats(obs_ic, mod_ic)
    S_ir = _pair_stats(obs_ir, mod_ir)
    S_uk = _pair_stats(obs_uk, mod_uk)

    # ----- figure -----
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    fig.subplots_adjust(bottom=0.33)  # leave room for the table under the plot

    # overlay scatters (UKC4, IRENE_C10, IRENE_R10)
    series = [
        (obs_uk, mod_uk, r'UKC4',          'C2', 's', S_uk),
        (obs_ic, mod_ic, r'IRENE$_{C10}$', 'C0', 'o', S_ic),
        (obs_ir, mod_ir, r'IRENE$_{R10}$', 'C3', '^', S_ir),
    ]
    for x, y, lab, col, mkr, S in series:
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=26, alpha=0.75, color=col, marker=mkr,
                   edgecolor='none', label=lab)
        if add_fits and np.isfinite(S['a']):
            xx = np.linspace(*xlim, 200)
            ax.plot(xx, S['a'] * xx + S['b'], color=col, lw=1.6, alpha=0.9)

    if add_identity:
        xx = np.linspace(*xlim, 200)
        ax.plot(xx, xx, 'k--', lw=1.0, label='y = x')

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Observed salinity [psu]')
    ax.set_ylabel('Modelled salinity [psu]')
    # ax.set_title(title, pad=10)  # uncomment if you want a title
    ax.legend(loc='lower right', fontsize=9, frameon=True)

    # ----- aligned stats "table" under the axes -----
    def _valid_mask(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        return np.isfinite(x) & np.isfinite(y)

    N_uk = _valid_mask(obs_uk, mod_uk).sum()
    N_ic = _valid_mask(obs_ic, mod_ic).sum()
    N_ir = _valid_mask(obs_ir, mod_ir).sum()

    header1 = "Skill Metrics vs Observations"
    header2 = f"{'Model':<12} {'RMSE':>6} {'MAE':>6} {'Bias':>7} {'r':>5} {'N':>5}"
    line_uk = f"{'UKC4':<12} {S_uk['rmse']:6.2f} {S_uk['mae']:6.2f} {S_uk['bias']:+7.2f} {S_uk['r']:5.2f} {N_uk:5d}"
    line_ic = f"{'IRENE_C10':<12} {S_ic['rmse']:6.2f} {S_ic['mae']:6.2f} {S_ic['bias']:+7.2f} {S_ic['r']:5.2f} {N_ic:5d}"
    line_ir = f"{'IRENE_R10':<12} {S_ir['rmse']:6.2f} {S_ir['mae']:6.2f} {S_ir['bias']:+7.2f} {S_ir['r']:5.2f} {N_ir:5d}"
    fig.text(
        0.5, 0.05,
        header1 + "\n" + header2 + "\n" + line_uk + "\n" + line_ic + "\n" + line_ir ,
        ha='center', va='bottom', fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.85', alpha=0.95)
    )

    if save:
        try:
            os.makedirs(getattr(save, 'parent', '.'), exist_ok=True)
        except Exception:
            pass
        plt.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig, ax

# %% Make the figure
fig, ax = salinity_overlay_single(
    clim10_data, realriv10_data,
    xlim=(0, 35), ylim=(0, 35),
    add_fits=True,
    title='Surface salinity: models vs observations',
    save=fig_path / 'salinity_overlay_single.png'
)
plt.show()



#%% Run figure plots 
fig, ax = salinity_overlay_single(
    clim10_data, realriv10_data,
    xlim=(0,35), ylim=(0,35),
    add_fits=True,  # set False if you only want y=x
    title='Surface salinity: models vs observations',
    save=fig_path / 'initial_salinity_validation.png'
)
plt.show()
