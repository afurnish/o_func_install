# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:41:25 2023

@author: aafur
"""
import xarray as xr
class OpenNc:
    def __init__(self):
        pass
    
    def slice_nc(self,main_dataset):
        '''
        Main Dataset should already be opened with an nc file. 
        '''
        self.main_dataset = main_dataset
        self.yyy = self.main_dataset.mesh2d_face_x_bnd.mesh2d_face_y
        self.xxx = self.main_dataset.mesh2d_face_x_bnd.mesh2d_face_x

        self.wd  = self.main_dataset.mesh2d_waterdepth    #waterdepth              (m)
        self.sh  = self.main_dataset.mesh2d_s1            #water surface height    (m)
        self.sal = self.main_dataset.mesh2d_sa1           #salinity               (psu)
        
        self.slice_dataset = xr.Dataset({
            "longitude": self.yyy,
            "latitude": self.xxx,
            "surface_height": self.sh,
            "water_depth": self.wd ,
            "salinity": self.sal
        })
        
        
        return self.slice_dataset
    
    
if __name__ == '__main__':
    start_slice = OpenNc()
    new_data = start_slice.slice_nc(main_dataset)
