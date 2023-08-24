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
import xarray as xr
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
        self.units = (self.wd).attrs['units']
    
        
        #Setting up plot manager for video run 
        self.plot_manager = Plot(figsize_preset = (self.s*0.9,gr(self.s)))
        self.fig = self.plot_manager.create_figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.plot_manager.set_subplot_layout(ax_list=[self.ax], hspace=0.2)
        
        UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        self.UKWEST = gpd.read_file(UKWEST_loc)
        self.cbar = None  # Initialize the colorbar
        self.labels_set = False
        
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
    def set_common_labels(self):
         if not self.labels_set:
             plt.xlabel('Longitude', fontsize=1.6*self.s)
             plt.ylabel('Latitude', fontsize=1.6*self.s)
             plt.xticks(fontsize=1.25*self.s)
             plt.yticks(fontsize=1.25*self.s)
             self.labels_set = True

    def vid_plotter(self, i):
        
        
        #plt.clf()
        UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        UKWEST = gpd.read_file(UKWEST_loc)
        
        UKWEST.plot(ax = self.ax, color="red")
        
        UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        UKWEST = gpd.read_file(UKWEST_loc)
        
        #trying dry land introduction
        #cmap = ListedColormap(['blue', 'cyan', 'lightblue', 'white', 'grey'])
        #contour_levels = [dry_value - 0.1, dry_value, dry_value + 0.1, np.max(z)]

        
        
        
        im = self.ax.tricontourf(
        self.xxx,
        self.yyy,
        self.wd[i,:],
        levels= np.linspace( self.bounds[0][0],self.bounds[0][1],self.bounds[0][2]),
        cmap=cm.cool,
        extend='both'
        )
        UKWEST.plot(ax = plt.gca(), color="white")
        
        # cbar= self.fig.colorbar(im, ax = self.ax)
        # cbar.set_ticks( np.linspace(self.bounds[1][0],
        #                             self.bounds[1][1],
        #                             self.bounds[1][2]
        #                             ))
        # cbar.ax.tick_params(labelsize=1.25*self.s)
        # cbar.set_label("water depth (m)", labelpad=+1, fontsize=1.3*self.s)
        
        if self.cbar is None:
            self.cbar= self.fig.colorbar(im, ax = self.ax)
            self.cbar.set_ticks( np.linspace(self.bounds[1][0],
                                        self.bounds[1][1],
                                        self.bounds[1][2]
                                        ))
            self.cbar.ax.tick_params(labelsize=1.25*self.s)
            name = ' '.join([i.capitalize() for i in self.name_of_run.split('_')])
            
            self.cbar.set_label(name + f'({self.units})', labelpad=+1, fontsize=1.3*self.s)
            
        self.set_common_labels()
        self.ax.set_xlim(self.bounds[2][0],
                    self.bounds[2][1])
        self.ax.set_ylim(self.bounds[3][0],
                    self.bounds[3][1])
        # plt.xlabel('Longitude', fontsize=1.6*self.s)
        # plt.ylabel('Latitude', fontsize=1.6*self.s)
        # Set x and y tick label font size
        plt.xticks(fontsize=1.25*self.s)  # Change 8 to the desired font size
        plt.yticks(fontsize=1.25*self.s)  # Change 8 to the desired font size
        plt.title(self.str_time[i])
        
        
        self.fig.savefig(os.path.join(self.path, 'kent_' + self.name_of_run + '_' + str(i) + '.png'))
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
    bathy_path = glob.glob(r'F:\modelling_DATA\kent_estuary_project\6.Final2\models\01_kent_2.0.0_no_wind\2.0.0_wind_testing_4_months.dsproj_data\FlowFM\*.nc')
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

    
    wd_bounds = [(-4,5,70),(-4,5,16),(-3.65,-2.75),(53.20,54.52)]

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

    bd = xr.open_dataset(bathy_path[0], engine='scipy')
    from scipy.interpolate import griddata

    # Bathymetry data at node coordinates
    node_x = bd.mesh2d_node_z.mesh2d_node_x.values
    node_y = bd.mesh2d_node_z.mesh2d_node_y.values
    bathymetry_node = bd.mesh2d_node_z.values
    
    # Face coordinates
    face_x = bd.mesh2d_face_x.values
    face_y = bd.mesh2d_face_y.values
    
    # Reshape face coordinates for griddata input
    face_coords = np.column_stack((face_x, face_y))
    
    # Interpolate bathymetry data to face coordinates
    bathymetry_face = griddata((node_x, node_y), bathymetry_node, face_coords, method='linear')
    