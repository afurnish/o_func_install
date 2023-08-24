# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:46:17 2023

@author: aafur
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from o_func.utilities.start import opsys; start_path = opsys()
from o_func.utilities.general import gr
from o_func import Plot

class VideoPlots:
    def __init__(self, main_dataset):
        
        self.yesno_video = 'n'
        self.main_dataset = main_dataset
        
    def video_speed(self):
        ''' Time should be a list of two numbers, t = 0, t = 1
         
        Parameters
        ----------
        time : List of 2 numbers t = 0 and t = 1

        Returns
        -------
        frame speed

        '''
        self.time = self.main_dataset.coords['time']
        diff_time = (self.time[1] - self.time[0]).values.astype('timedelta64[m]')
        print('Time Stamp is ', diff_time, ' minutes')
        fs = 24
        return fs
    
    def vid_plot(self, bounds):
        plt.rcParams['figure.autolayout'] = True

        # Path to mask layer
        UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        UKWEST = gpd.read_file(UKWEST_loc)
        
        self.yyy = self.main_dataset.mesh2d_face_x_bnd.mesh2d_face_y
        self.xxx = self.main_dataset.mesh2d_face_x_bnd.mesh2d_face_x

        self.wd  = self.main_dataset.mesh2d_waterdepth    #waterdepth              (m)
        self.sh  = self.main_dataset.mesh2d_s1            #water surface height    (m)
        self.sal = self.main_dataset.mesh2d_sa1           #salinity               (psu)
        
        i = 0
        s = 7
        plot_manager = Plot(figsize_preset = (gr(s)*0.6,s))
        fig = plot_manager.create_figure()
        ax = fig.add_subplot(1, 1, 1)
        plot_manager.set_subplot_layout(ax_list=[ax], hspace=0.2)
        
        plt.tricontourf(
        self.xxx,
        self.yyy,
        self.wd[i],
        levels= np.linspace( bounds[0][0],bounds[0][1],bounds[0][2]),
        cmap=cm.jet,
        extend='both'
        )
        
        cbar= plt.colorbar()
        UKWEST.plot(ax=ax, color="white")
        cbar.set_ticks( np.linspace( bounds[1][0],bounds[1][1],bounds[1][2] ))
        cbar.ax.tick_params(labelsize=1.25*s)
        cbar.set_label("water depth (m)", labelpad=+1, fontsize=1.3*s)
        ax.set_xlim( bounds[2][0],bounds[2][1])
        ax.set_ylim( bounds[3][0],bounds[3][1])
        plt.xlabel('Longitude', fontsize=1.6*s)
        plt.ylabel('Latitude', fontsize=1.6*s)
        # Set x and y tick label font size
        plt.xticks(fontsize=1.25*s)  # Change 8 to the desired font size
        plt.yticks(fontsize=1.25*s)  # Change 8 to the desired font size

        
        
#%% 
if __name__ == '__main__':
    #%% Importing Dependecies
    import os
    import xarray as xr
    import glob
    import time

    from o_func import DataChoice, DirGen

    #%% Making Directory paths
    main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
    make_paths = DirGen(main_path)
    sub_path = make_paths.dir_outputs('kent_2.0.0_no_wind')
    ### Finishing directory paths

    dc = DataChoice(os.path.join(main_path,'models'))
    fn = dc.dir_select()
    #%% Lazy loading dataset
    ### Large dataset path for testing 
    lp = glob.glob(r'F:\modelling_DATA\kent_estuary_project\5.Final\1.friction\SCW_runs\kent_2.0.0_wind\*.nc')[0]

    sp = time.time()
    #lp = glob.glob(os.path.join(fn[0],'*.nc'))[0]
    #Make chunk sizes 10 if it all goes to pot 
    main_dataset = xr.open_dataset(lp, 
                       chunks={'time': 'auto', 
                       'mesh2d_face_x_bnd': 'auto',
                       'mesh2d_face_y_bnd': 'auto'
                       }, 
                       engine = 'scipy'
                       )

    # plot_manager = Plot(figsize_preset = (40, 6))
    # fig = plot_manager.create_figure()
    # ax = fig.add_subplot(1, 1, 1)
    #plot_manager.set_subplot_layout(ax_list=[ax], hspace=0.3)
    wd_bounds = [(0,65,80),(0,60,7),(-3.65,-2.75),(53.20,54.52)]

    pv = VideoPlots(main_dataset)
    make_videos = pv.vid_plot(wd_bounds)
