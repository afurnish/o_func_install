#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to dictate choices made in selecting different data sets

Created on Wed Nov 30 14:15:02 2022
@author: af
"""
import glob
import os


from o_func.utilities.start import opsys; start_path = opsys()
from o_func.utilities import winc # winc used in Choices

#%%
class DataChoice:
    def __init__(self, location_of_choices , ms = 1):
        self.ms = ms
        self.location_of_choices = location_of_choices
    
    def multi_select(self, choice):
        if self.ms > 1:
            
            if choice == 'r':
                print("Choices reset.")
            else:
                print('Multiple selection, please make ', self.ms,' choices from the above.')

            
        
    def select(self, path, selection, input_message = '' ): 
        #generic_path_to_file = path + selection
        paths = []
        for i in range(self.ms):
        
            file_length = [str(i) for i in (list(range(1, len(selection) + 1)))]
            user_input = ''
            
            for index, item in enumerate(selection):
                input_message += f'{index+1}) {item}\n'
            
            input_message += '\nYour choice: '
            while user_input.lower() not in file_length:
                user_input = input(input_message)
            
            output = selection[int(user_input)-1]
            full_path = os.path.join(path,output)
    
            print('You picked: ' + output)
            print('Full Path: ' + full_path +'\n')
            
            paths.append(full_path)
        return paths
        
        
    def dir_select(self):
        path = self.location_of_choices
        def list_directories(directory_path):
            directories = [item for item in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, item))]
            return directories
        dirs = list_directories(path)
        
        input_message = "Select a directory by number:\n"      
        full_path = self.select(path,dirs,input_message)
        return full_path
        
    def file_select(self, data_type = 'default'):
        self.data_type = {
            'default': '',
            'csv': r'/*.csv',
            'pli' : r'/*.pli'
            # Add more named presets as needed...
        }
        
        path = self.location_of_choices
        generic_path_to_file = path + self.data_type[data_type] # formally csv etc
        
        files = []
        options = []
        # print(generic_path_to_file)
        for j, filename in enumerate(glob.glob(generic_path_to_file)):
            f = winc(filename)

            files.append(f)
            options.append(f.split('/')[-1])
        input_message = "Select a file by number:\n"
        full_path = self.select(path, options, input_message)
        return full_path
    
    def var_select(self):
        variables = [os.path.join('owa','shelftmb'), os.path.join('oa','shelftmb'), os.path.join('ow','shelftmb'), os.path.join('og','shelftmb')]
        path = self.location_of_choices
        
        full_path = self.select(path, variables)
        return full_path
        
    def var_name_select(self, variables):
        path = self.location_of_choices
        # This was originally set up to choose what dataset you wanted to print images for. 
        
        full_path = self.select(path, variables)
        return full_path
     
if __name__ == '__main__':
    
#     
    
# #%% Diff ways to use the choices program
#     directory_path = r'modelling_DATA/kent_estuary_project/' + \
#                         r'land_boundary/analysis/QGIS_shapefiles'
                        
#     directory_path2 = 'modelling_DATA/kent_estuary_project/tidal_boundary/delft_3dfm_inputs'
    
#     choice = DataChoice(start_path, directory_path)
    
#     file_path = choice.file_select(start_path, directory_path, data_type = 'csv')
#     folder_path = choice.dir_select(start_path, directory_path2)
    
    
# #%%
    
#     ### Finishing directory paths

    #%% 
    from o_func import DirGen
    main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
    make_paths = DirGen(main_path)
    sub_path = make_paths.dir_outputs('kent_2.0.0_no_wind')
    dc = DataChoice(os.path.join(main_path,'models'))
    fn = dc.dir_select()

#%%
    # variables = ['owa/shelftmb', 'oa/shelftmb', 'ow/shelftmb', 'og/shelftmb']
    # directory_path3 = r'Original Data/UKC3'
    # dc = DataChoice(os.path.join(start_path, directory_path3))
    # var_path = dc.var_select(variables)
