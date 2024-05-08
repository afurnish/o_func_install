#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:10:24 2024

@author: af
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_residence_time_diagram():
    fig, ax = plt.subplots(figsize=(8, 4))
    # Create a simple box to represent the reservoir
    reservoir = patches.Rectangle((0.3, 0.4), 0.4, 0.2, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(reservoir)
    
    # Add arrows for inflow and outflow
    ax.annotate('', xy=(0.25, 0.5), xytext=(0.3, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.7, 0.5), xytext=(0.75, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Text and annotations
    ax.text(0.15, 0.5, 'Inflow (Q)', va='center', ha='center', backgroundcolor='w')
    ax.text(0.85, 0.5, 'Outflow (Q)', va='center', ha='center', backgroundcolor='w')
    ax.text(0.5, 0.7, 'Reservoir (Volume V)', va='center', ha='center')
    ax.text(0.5, 0.1, 'Residence Time $\\tau = \\frac{V}{Q}$', va='center', ha='center', fontsize=12)
    
    # Settings
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

fig = draw_residence_time_diagram()
plt.show()