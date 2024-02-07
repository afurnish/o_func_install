#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Python file that will create a folder and then populate it with a subset 
of a larger dataset to save on processing time. 

This calls a bash file, located within this folder. 

This sucessfully works to created the subsets needed

Created on Wed Dec  6 15:01:20 2023
@author: af
"""
from o_func.utilities import uk_bounds_wide, uk_bounds
from o_func import opsys; start_path = opsys()

import os
import glob
import xarray as xr
import numpy as np

import pkg_resources
import subprocess




def orig_data_paths():
    folder_paths = []
    base_path = os.path.join(start_path, "Original_Data","FES2014", "world")
    folder_list = ["eastward_velocity", "load_tide", "northward_velocity", "ocean_tide", "ocean_tide_extrapolated"]
    for folder_name in folder_list:
        folder_paths.append(os.path.join(base_path, folder_name))

    return folder_paths

def convert_to_0_to_360(longitudes):
    return (longitudes + 360) % 360

class FES_Slice:
    def __init__(self, example_data, bounds = uk_bounds()):
        self.lon, self.lat = uk_bounds_wide()[0], uk_bounds_wide()[1]
        self.example_data = example_data
    
    
    def sli(self):
        data = xr.open_dataset(self.example_data)
        # print(data)
    
        # Finding lower and upper indexes to slice from the main dataset
        self.li_lon = np.searchsorted(data.lon, convert_to_0_to_360(self.lon[0]), side='left')
        self.ui_lon = np.searchsorted(data.lon, convert_to_0_to_360(self.lon[1]), side='right')
        self.li_lat = np.searchsorted(data.lat, (self.lat[0]), side='left')
        self.ui_lat = np.searchsorted(data.lat, (self.lat[1]), side='right')
    
        

    def make_sli(self, folder_paths):
        # Get the path to the bash script within the package
        script_path = pkg_resources.resource_filename('o_func', 'data/bash/FES2014_slicer.sh')
        
        for folder in folder_paths:
            for file in glob.glob(os.path.join(folder, '*.nc')):
                # file is original name
                new_file = file.replace('/world/', '/'+self.sub_folder+'/')
            
                # input_file = '/Volumes/PN/Original_Data/FES2014/world/eastward_velocity/2n2.nc'
                # output_file = '/Volumes/PN/Original_Data/FES2014/PRIMEA_subset/eastward_velocity/2n2.nc'
                input_file = file
                output_file = new_file
        
                # subprocess.run(['bash', script_path, "-i", input_file, "-o", output_file, "-x", str(self.li_lon), "-y", str(self.ui_lon), "-X", str(self.li_lat), "-Y", str(self.ui_lat)])
                subprocess.run(['bash', script_path, "-i", input_file, "-o", output_file, "-x", str(self.li_lon), "-y", str(self.ui_lon), "-X", str(self.li_lat), "-Y", str(self.ui_lat)], capture_output=True)

                # Print the output of the Bash script
        
    def create_folders(self,sub_folder):
        self.sub_folder = sub_folder
        folder_paths = []
        # Combine paths
        base_path = os.path.join(start_path, "Original_Data","FES2014")

        main_folder_path = os.path.join(base_path, self.sub_folder)

        # Create the main folder
        if os.path.basename(main_folder_path).lower() in ["world", "raw"]:
            print("Error: Cannot overwrite main dataset. Choose a different sub_folder name.")
            return
        
        os.makedirs(main_folder_path, exist_ok=True)

      

        folder_list = ["eastward_velocity", "load_tide", "northward_velocity", "ocean_tide", "ocean_tide_extrapolated"]

        
        # Create subdirectories inside the subfolder
        for folder_name in folder_list:
            folder_paths.append(os.path.join(main_folder_path, folder_name))
            folder_path = os.path.join(main_folder_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
        return folder_paths
        
if __name__ == '__main__':
    
    # Example usage
    sub_folder = "UK_bounds"
    original_data_paths = orig_data_paths()
    
    example_data = glob.glob(os.path.join(original_data_paths[0], '*'))[0]
    
    fs = FES_Slice(example_data)
    folder_pathlist = fs.create_folders(sub_folder)
    fs.sli()
    fs.make_sli(original_data_paths)
