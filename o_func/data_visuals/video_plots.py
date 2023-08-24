# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:46:17 2023

@author: aafur
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import multiprocessing as mp
import os
#from concurrent.futures import ThreadPoolExecutor



from o_func.utilities.start import opsys; start_path = opsys()
from o_func.utilities.general import gr

from o_func.data_prepkit import OpenNc

from o_func import Plot

class VideoPlots:
    def __init__(self, dataset, xxx, yyy, bounds, path, s = 7):
        
        # Video Speed attributes
        self.yesno_video = 'n'
        self.dataset = dataset
        
        #Data attributes
        self.xxx = xxx
        self.yyy = yyy
        self.wd = dataset
        self.bounds = bounds
        self.path = path
        
        #Time attributes
        self.time = self.wd.time
        #a = [str(i) for i in (new_data.surface_height.time.dt.strftime("%Y-%m-%d %H:%M").values)]
        self.str_time = [str(i) for i in (self.time.dt.strftime("%Y-%m-%d %H:%M").values)]
        
        #constant for figsizes
        self.s = s
        
        #name of run
        self.name_of_run = (self.wd).attrs['standard_name']
        
        #Setting up plot manager for video run 
        self.plot_manager = Plot(figsize_preset = (self.s*0.9,gr(self.s)))
        self.fig = self.plot_manager.create_figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.plot_manager.set_subplot_layout(ax_list=[self.ax], hspace=0.2)
        
    def video_speed(self):
        ''' Time should be a list of two numbers, t = 0, t = 1
         
        Parameters
        ----------
        time : List of 2 numbers t = 0 and t = 1

        Returns
        -------
        frame speed

        '''
        self.time = self.dataset.coords['time']
        diff_time = (self.time[1] - self.time[0]).values.astype('timedelta64[m]')
        print('Time Stamp is ', diff_time, ' minutes')
        fs = 24
        return fs
    
    # changed main dataset above here to just dataset. 

    def vid_plotter(self, i):
        
        
        #plt.clf()
        UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        UKWEST = gpd.read_file(UKWEST_loc)
        
        UKWEST.plot(ax = self.ax, color="red")
        
        UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        UKWEST = gpd.read_file(UKWEST_loc)
        
        # if i%100:
        #     print('h'+ str(i))
        im = self.ax.tricontourf(
        self.xxx,
        self.yyy,
        self.wd[i,:],
        levels= np.linspace( self.bounds[0][0],self.bounds[0][1],self.bounds[0][2]),
        cmap=cm.jet,
        extend='both'
        )
        UKWEST.plot(ax = plt.gca(), color="white")
        
        cbar= self.fig.colorbar(im, ax = self.ax)
        cbar.set_ticks( np.linspace(self.bounds[1][0],
                                    self.bounds[1][1],
                                    self.bounds[1][2]
                                    ))
        cbar.ax.tick_params(labelsize=1.25*self.s)
        cbar.set_label("water depth (m)", labelpad=+1, fontsize=1.3*self.s)
        
        self.ax.set_xlim(self.bounds[2][0],
                    self.bounds[2][1])
        self.ax.set_ylim(self.bounds[3][0],
                    self.bounds[3][1])
        plt.xlabel('Longitude', fontsize=1.6*self.s)
        plt.ylabel('Latitude', fontsize=1.6*self.s)
        # Set x and y tick label font size
        plt.xticks(fontsize=1.25*self.s)  # Change 8 to the desired font size
        plt.yticks(fontsize=1.25*self.s)  # Change 8 to the desired font size
        plt.title(self.str_time[i])
        
        
        #self.fig.savefig(os.path.join(self.path, 'kent_' + self.name_of_run + '_' + str(i) + '.png'))
        self.ax.clear()
    
    def vid_para_plot(self):
        #plt.rcParams['figure.autolayout'] = True
        start = time.time()
        pool = mp.Pool(12)
        jobs = []
        for item in range(100):
            # Use the `map` function to apply the `square` function to each number
          job = pool.apply_async(self.vid_plotter, (item, ))
          jobs.append(job)
          
        
        
        for job in jobs:
            job.get()
          
        pool.close()
        pool.join()
          #print(results)
        print("multiprocessing Plotting took {:.4f} seconds".format(time.time() - start))
   
        
        
    def vid_norm_plot(self):
        start = time.time()
        #plt.rcParams['figure.autolayout'] = True
        for i in range(100):
            self.vid_plotter(i)
        print("Normal Plotting took {:.4f} seconds".format(time.time() - start))

            

        
#%% 
if __name__ == '__main__':
    #%% Importing Dependecies
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

    
    wd_bounds = [(0,65,80),(0,60,7),(-3.65,-2.75),(53.20,54.52)]

    # Class to split the dataset into a managable chunk. 
    start_slice = OpenNc()
    new_data = start_slice.slice_nc(main_dataset)

#%% Vid prep
    #make path for dataset to live
    png_sh_path = make_paths.vid_var_path(var_choice='Surface_Height')
    pv = VideoPlots(dataset = new_data.surface_height,
                    xxx     = new_data.mesh2d_face_x,
                    yyy     = new_data.mesh2d_face_y,
                    bounds  = wd_bounds,
                    path    = png_sh_path
                    )
    
    # start = time.time()
    # make_videos = pv.vid_para_plot()
    # print("ThreadPoolExecutor took {:.4f} seconds".format(time.time() - start))
    
    
    # start = time.time()
    # make_videos = pv.vid_norm_plot()
    # print("Normal Plotting took {:.4f} seconds".format(time.time() - start))

    
    
    #make_videos = pv.vid_norm_plot()
    make_videos2 = pv.vid_para_plot()

