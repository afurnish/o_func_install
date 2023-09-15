# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:46:17 2023

@author: aafur
"""

from matplotlib import cm
import matplotlib as matlib
matlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import multiprocessing as mp
import os
import xarray as xr
from matplotlib.colors import ListedColormap
import glob
import time

from joblib import Parallel, delayed

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
        
        #setting up dry land
        self.bathy_path = glob.glob(r'F:\modelling_DATA\kent_estuary_project\6.Final2\models\01_kent_2.0.0_no_wind\2.0.0_wind_testing_4_months.dsproj_data\FlowFM\*.nc')

        UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        self.UKWEST = gpd.read_file(UKWEST_loc)
        
        #testing dry land 
        bd = xr.open_dataset(self.bathy_path[0], engine='scipy')
        from scipy.interpolate import griddata

        # Bathymetry data at node coordinates
        node_x = bd.mesh2d_node_z.mesh2d_node_x.values
        node_y = bd.mesh2d_node_z.mesh2d_node_y.values
        bathymetry_node = bd.mesh2d_node_z.values
        
        # Face coordinates
        '''
        NOTE TO SELF, these coords are not the same as the nodes coords. 
        '''
        self.face_x = bd.mesh2d_face_x.values
        self.face_y = bd.mesh2d_face_y.values
        
        # Reshape face coordinates for griddata input
        ##face_coords = np.column_stack((self.face_x, self.face_y))
        face_coords = np.column_stack((self.xxx, self.yyy))
        
        # Interpolate bathymetry data to face coordinates
        self.bathymetry_face = griddata((node_x, node_y), bathymetry_node, face_coords, method='linear')
        
        
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
        # 
        
        #self.UKWEST.plot(ax = self.ax, color="red")
        
        # UKWEST_loc = start_path + r'modelling_DATA/kent_estuary_project/land_boundary/QGIS_Shapefiles/UK_WEST_POLYGON_NEGATIVE.shp'
        # UKWEST = gpd.read_file(UKWEST_loc)
        
        #trying dry land introduction
        colors = ['grey', 'blue', 'pink']
        cmap = ListedColormap(colors)
        
        
        # im = self.ax.tricontourf(
        # self.xxx,
        # self.yyy,
        # self.wd[i,:],
        # levels= np.linspace( self.bounds[0][0],self.bounds[0][1],self.bounds[0][2]),
        # cmap=cm.cool,
        # extend='both'
        # )
        
        im = self.ax.tricontourf(
        self.xxx,
        self.yyy,
        self.wd[i,:],
        levels= np.linspace( self.bounds[0][0],self.bounds[0][1],self.bounds[0][2]),
        cmap=cm.cool,
        extend='both'
        )
        
        
        #if self.vel != 'n':
           
        
        if self.land != 'n':
            
            # # Set the colors for dry land (where z matches bathymetry)
            tolerance=self.land
            abs_differences = np.abs(self.wd[i, :] - self.bathymetry_face)
            within_tolerance = abs_differences <= tolerance
            dry_land_indices = np.where(within_tolerance)
            self.ax.scatter(self.xxx[dry_land_indices], self.yyy[dry_land_indices], color='grey', s=1, alpha=1)


        self.UKWEST.plot(ax = plt.gca(), color="white")
        
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
        plt.close(self.fig)
    
    def vid_para_plot(self, num_of_figs = 10, land = 'n'):
        self.land = land
        print(self.land)
        print(num_of_figs)
        plt.rcParams['figure.autolayout'] = True
        start = time.time()
        pool = mp.Pool(os.cpu_count())
        jobs = []
        for item in range(num_of_figs):
            # Use the `map` function to apply the `square` function to each number
          job = pool.apply_async(self.vid_plotter, (item, ))
          jobs.append(job)

        for job in jobs:
            job.get()
          
        pool.close()
        pool.join()
        
        print("multiprocessing Plotting took {:.4f} seconds".format(time.time() - start))


    def joblib_para(self, num_of_figs = 10, land = 'n', vel = 'n'):
        self.land = land
        self.vel = vel
        results = Parallel(n_jobs=-1)(delayed(self.vid_plotter)(num_iters) for num_iters in range(num_of_figs))

        #self.vel_u = self.vel.
        #self.vel_v = self.vel 
        
        
          #print(results)
   
   
    def vid_norm_plot(self, num_of_figs = 10, land = 'n'):
        self.land = land

        start = time.time()
        #plt.rcParams['figure.autolayout'] = True
        for i in range(num_of_figs):
            self.vid_plotter(i)
        print("Normal Plotting took {:.4f} seconds".format(time.time() - start))

    def bathy_plot(self):
        spacing = (-40,20,200)
        
        # im = self.ax.tricontourf(
        # self.xxx,
        # self.yyy,
        # self.bathymetry_face,
        # levels= np.linspace(spacing[0], spacing[1], spacing[2]),
        # cmap=cm.cool,
        # extend='both'
        # )
        # Set the desired color range limits
        color_min = -40
        color_max = 20

        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(vmin=color_min, vmax=color_max)

        im = self.ax.scatter(self.xxx,self.yyy, c = self.bathymetry_face, s = 0.1, cmap = cmap, norm = norm)
        
        # self.UKWEST.plot(ax = plt.gca(), color="white")
        
        # if self.cbar is None:
        #     self.cbar= self.fig.colorbar(im, ax = self.ax)
        #     self.cbar.set_ticks( np.linspace(spacing[0],
        #                                 spacing[1],
        #                                 spacing[2]
        #                                 ))
        #     self.cbar.ax.tick_params(labelsize=1.25*self.s)
        #     name = ' '.join([i.capitalize() for i in self.name_of_run.split('_')])
            
        #     self.cbar.set_label(name + f'({self.units})', labelpad=+1, fontsize=1.3*self.s)
        
        # self.ax.set_xlim(self.bounds[2][0],
        #             self.bounds[2][1])
        # self.ax.set_ylim(self.bounds[3][0],
        #             self.bounds[3][1])
        
        self.fig.savefig(os.path.join(self.path, '00_kent_bathy.png'))
        print(os.path.join(self.path, '00_kent_bathy.png'))


        
#%% 
if __name__ == '__main__':
    #%% Importing Dependecies
    
    

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

    
    starrrr = time.time()
    #make_videos = pv.vid_norm_plot(num_of_figs=4, land = 'n')
    make_videos2 = pv.joblib_para(num_of_figs=200, land = 0.08)
    print('Time between: ', time.time() - starrrr)

    # pv = VideoPlots(dataset = new_data.surface_height,
    #                 xxx     = new_data.mesh2d_face_x,
    #                 yyy     = new_data.mesh2d_face_y,
    #                 bounds  = wd_bounds,
    #                 path    = png_sh_path
    #                 )
    pv.bathy_plot()
    #profiler.print_stats(sort='cumulative')  # Print the profiling results