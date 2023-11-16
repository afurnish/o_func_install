#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" # Generates the interpolated grid files found in the old UKC3 data. 

Does it need to be run for all cycles as its only purpose is to reshape the grid. Yes it does 
need to be rerun as it doesnt just generate a grid it reshapes the UKC3 datasets. You run it once per UKC3 grid cell. 

I need to do the reverse and interpolate PRIMEA onto UKC3 to achieve the best results. The best way to do this would be to use ESMF grid.
Can use this script in conjunction with ESMF to generate the correct ideas. 

Created on Thu Sep 28 09:35:38 2023
@author: af
"""
import os
import glob
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree


from o_func.utilities.start import opsys; start_path = opsys()
from o_func import DataChoice
import o_func.utilities as util
from o_func import uk_bounds_wide; uk_b = uk_bounds_wide()
# set default PRIMEA path but you can change it if grid changes. 
def_PRImea_PATH = os.path.join(start_path, 'modelling_DATA',
                                    'kent_estuary_project',
                                    '6.Final2','models',
                                    'kent_1.3.7_testing_4_days_UM_run',
                                    'kent_31_merged_map.nc')


class InterpolateGrid:
    '''
    Purpose is to interpolate one grid dataset onto another for comparison and statistics. 
    UKC3 will be mapped onto PRIMEA To create a higher resolution, equal quality dataset. 
    '''
    def __init__(self, ukc3_modcho,default_path = 'shelftmb', first_prim_path = def_PRImea_PATH):
        # Pick paths to extract UKC3 data. 
        self.default_path = default_path
        self.first_prim_path = first_prim_path
        ukc3_model_choice_options = {'og': {'name': 'ocean_only', 
                                            'data_path': os.path.join(start_path, 'Original_Data','UKC3','og', default_path )},
                                    'oa': {'name': 'ocean_atmosphere_coupled', 
                                           'data_path': os.path.join(start_path, 'Original_Data','UKC3','oa' , default_path )},
                                    'ow': {'name': 'ocean_wave_coupled', 
                                           'data_path': os.path.join(start_path, 'Original_Data','UKC3','ow' , default_path )},
                                    'owa': {'name': 'ocean_wave_atmosphere_coupled', 
                                            'data_path': os.path.join(start_path, 'Original_Data','UKC3','owa' , default_path )}
                                    }
        '''
        Here it will be possible to add in custom model settings when comparing the diff models. 
        '''
        
        if ukc3_modcho not in ukc3_model_choice_options:
            error_message = f'{ukc3_modcho} is not a valid key in the dictionary.'
            options_message = "Please choose from the following options:\n"
            
            for key, value in ukc3_model_choice_options.items():
                options_message += f'key: {key} Which corresponds to {value}\n'

            raise KeyError(error_message + options_message)
         
        self.ukc3_data_path = ukc3_model_choice_options[ukc3_modcho]['data_path']
        # Next stuff

    def load_ukc3(self):
        data_options = []
        #data_store_paths = []
        t_paths = sorted(glob.glob(os.path.join(self.ukc3_data_path, '*T.nc')))
        u_paths = sorted(glob.glob(os.path.join(self.ukc3_data_path, '*U.nc')))
        v_paths = sorted(glob.glob(os.path.join(self.ukc3_data_path, '*V.nc'))) 
        let = 'T', 'U', 'V' #This will determine what data gets extracted from file. 
        
        latitude_range = slice(uk_b[1][0], uk_b[1][1])
        longitude_range = slice(uk_b[0][0], uk_b[0][1])
        ukc3_dataset = []
        ukc3_all_datasets = []
        
        first_ukc3 = xr.open_dataset(t_paths[0], engine = 'netcdf4')
        self.lon_ukc3 = first_ukc3.nav_lon
        self.lat_ukc3 = first_ukc3.nav_lat
        lon_mean = np.nanmean(self.lon_ukc3, axis=0)
        lat_mean = np.nanmean(self.lat_ukc3, axis=1)
        slices = []
        slices.append(util.Dist.find_nearest(lat_mean, uk_b[1], index = 'i'))
        slices.append(util.Dist.find_nearest(lon_mean, uk_b[0], index = 'i'))
        #                   slices[0][0] slices[0][1]     slices[1][0] slices[1][1]
        # to slice slice(none), slice(531, 755), slice(743, 807) 
        print('slices', slices)
        for i, files in enumerate([t_paths, u_paths, v_paths]):
            if len(files) > 0:
                a = 0
                print('je')
                if i == 0:
                    data_options.append(let[i])
                    #data_store_paths.append(files) # ensures only availl  data is loaded. 
                    for i in files:
                        # print(i)
                        ds = xr.open_dataset(i, engine = 'netcdf4' , chunks= {'time_counter':240})

                        # print(ds.dims)
                        # print(ds.x)
                        # print(ds.y)
                        # print(ds.nav_lat)
                        # ds = ds.set_coords(['nav_lon', 'nav_lat'])
                        ###
                        ### If you get an oppertunity to fix this and slice it before you sample. 
                        ###
                        ds_sliced = ds.sel(y = slice(slices[0][0], slices[0][1]), x = slice(slices[1][0] ,slices[1][1]))
                        ukc3_dataset.append(ds_sliced)
                    ukc3_all_datasets.append(ukc3_dataset)
                    
                    print('successful')
        print('starting netcdf making')            
        # for i,file in enumerate(ukc3_all_datasets[0]):
        #     print(file)
        #     if a == 0:
        #         j = str(i)
        #         file.to_netcdf(f'test/{j}.nc')
        self.merged_dataset = xr.concat(ukc3_all_datasets[0], dim="time_counter")
        #print(merged_dataset)
        #merged_dataset = merged_dataset.chunk({'time_counter': 100})
        #print(merged_dataset)
        #merged_dataset.to_netcdf(path='test.nc', mode='w')
        # Optionally set chunk sizes
        #import dask.array as da

        #dask_ds = merged_dataset.chunk({'time_counter': 100})
        
        # Write the Dask-backed dataset to a NetCDF file
        #dask_ds.to_netcdf('output.nc', compute=False)
        
        # Compute and write the data
        #dask_ds.compute()
        
        
        combined = util.data_flattener(np.array(self.lon_ukc3), np.array(self.lat_ukc3))
        test = np.ravel(np.dstack([np.array(self.lon_ukc3).ravel()])[0]), np.ravel(np.dstack([np.array(self.lat_ukc3).ravel()])[0])
        self.tree = cKDTree(list(zip(np.ravel(np.dstack([np.array(self.lon_ukc3).ravel()])[0]), np.ravel(np.dstack([np.array(self.lat_ukc3).ravel()])[0]))))

        print(test[0].shape)
        print(test)
        #print(tree)
        
        return ukc3_dataset, self.lon_ukc3, self.lat_ukc3
    def load_primea(self):
        first_prim = xr.open_dataset(self.first_prim_path, engine = 'netcdf4')
        
        self.lon_prim = first_prim.mesh2d_face_x
        self.lat_prim = first_prim.mesh2d_face_y
        
    def make_paths(self):
        grid_path = os.path.join(start_path, 'modelling_DATA',
                                 'kent_estuary_project','grid',
                                 'grid_interpolation')
        #generate datastore path for each shaped grid. 
        self.grid_shape = str(self.lon_prim.shape[0])
        self.store_data = util.md([grid_path, 'for_grid_length_' + self.grid_shape])
        

    def interpolate_onto_grid(self, tot_timestep):
        from scipy.interpolate import griddata

        data_options = 'T'
        all_interp_list = []
        for i in range(tot_timestep):
            print(i)
            if self.default_path == 'shelftmb':
                a = 0
                if data_options == 'T':
                    sossheig = np.array(self.merged_dataset.sossheig[i,:,:]).ravel()
                    print(sossheig)
                    #all_interp_list.append(sossheig)
                    
                    print('sossheig shape', sossheig.shape)
                    print(self.lon_prim)
                    print(self.lat_prim)
                   
                    # for k in range(len(self.lon_prim)):
                    #     # distances, indices = self.tree.query([self.lon_prim[k], self.lat_prim[k]], k=1)
                    #     # #print(distances, indices)
                    #     # if k % 2000:
                    #     #     print(k)
                    #     # all_interp_list.append(indices)
                    
                    interpolated_data = griddata(
                            (np.array(self.lon_ukc3).ravel(), np.array(self.lon_ukc3).ravel()),  # Structured grid coordinates
                            sossheig,  # Values to interpolate
                            (self.lon_prim, self.lat_prim),  # Unstructured grid coordinates
                            method='linear',  # You can choose a different interpolation method
                            fill_value=np.nan  # Set non-overlapping points to NaN
)
                        #all_interp_list.append(np.array(sossheig)[indices][0][0])
                #     vosaline_top
                #     vosaline_mid
                #     vosaline_bot
                    
                #     votemper_top
                #     votemper_mid
                #     votemper_bot
                    
                #     runoffs
                # if data_options == 'U':
                #     vozocrtx_top
                #     vozocrtx_mid
                #     vozocrtx_bot
                #     vobtcrtx
                # if data_options == 'V':
                #     vomecrty_top
                #     vomecrty_mid
                #     vomecrty_bot
                #     vobtcrty
                
        return all_interp_list
    def write_to_netcdf(self):
        self.nc_file = os.path.join(self.store_data, 'ukc3_' + self.grid_shape + '.nc')
        
    def call_script(self):
        self.load_primea()
        ukc3_dataset = self.load_ukc3()
        self.make_paths()
        
        
        all_interp_list = self.interpolate_onto_grid(2)
        
        return ukc3_dataset, all_interp_list
    
if __name__ == '__main__':
    ig = InterpolateGrid(ukc3_modcho = 'og')
    ukc3_dataset, all_interp_list = ig.call_script()