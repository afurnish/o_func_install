# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 18:17:52 2023

@author: aafur
"""
import matplotlib.pyplot as plt
from o_func.utilities import gr

class Plot:
    
    def __init__(self, figsize_preset = 'default', dpi=300):
       
       self.dpi = dpi
       self.figsize_preset = {
            'default': (21.7, 14),
            'whole_page': (10, 7),
            'a4_landscape' : (7.5, 11),
            'third_ofpage' : (7.5, gr(7.5)),
            'long_third' : (4, 7.5)

            # Add more named presets as needed...
        }
       self.figsize = self.figsize_preset.get(figsize_preset, figsize_preset)
       
    def create_figure(self):
        return plt.figure(figsize=self.figsize, dpi=self.dpi)

    def set_subplot_layout(self, ax_list, hspace=0.5):
        plt.subplots_adjust(hspace=hspace)


    
       
       
if __name__ == '__main__':
    plot_manager = Plot(figsize_preset = 'long_third')
    fig = plot_manager.create_figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(3, 1, 2)
    # ax3 = fig.add_subplot(3, 1, 3)
    plot_manager.set_subplot_layout(ax_list=[ax1], hspace=0.3)
