# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:53:58 2023

@author: aafur
"""
import os
import glob
from o_func import opsys3; start_path = opsys3()
import pkg_resources
import shutil
import o_func.utilities as util

class DirGen:
    def __init__(self, main_dir):
        path_dict = {
            "path_to_rgfgrid_files" : "files_rgfgrid",
            "path_to_qgis_files" : "files_qgis"
            }
        #print('len_path_dict', len(path_dict))
        
        if os.path.isdir(main_dir) == False:
            os.mkdir(main_dir)
        
        for key,value in path_dict.items(): 
            mp = os.path.join(main_dir,value)
            if os.path.isdir(mp) == False:
                os.mkdir(mp)
            
            
        self.main_dir = main_dir
        self.path_dict = path_dict
        
        
    def dir_outputs(self, model_name):
        # Identify if other models share the same name, then create their processing folder
        files = []
        for i, file in enumerate(glob.glob(os.path.join(self.main_dir, "*"))):
            files.append(file)
        # Checking if file already exits
        same_file = glob.glob(os.path.join(self.main_dir, "*"+model_name+"*"))
        # Sorting out numbering
        num_files = [
        folder for folder in os.listdir(self.main_dir)
            if os.path.isdir(os.path.join(self.main_dir, folder)) and "testing" not in folder
            ] # number of files but always ignore testing
        if model_name == 'testing_folder':
            new_num = '00'
        else:
            if len(same_file) == 0:
                new_num = str(len(num_files)-1).zfill(2)
                
            else:
                new_num = os.path.split(same_file[0])[-1][0:2]
            
        self.model_path = util.md([self.main_dir, new_num + '_' + model_name])
        
        # Making outputs folder for results and data 
        self.outputs = util.md([self.model_path, 'outputs'])
        self.giffs = util.md([self.outputs, 'giffs'])
        self.figures = util.md([self.outputs, 'figures'])
        self.data_proc = util.md([self.outputs, 'data_proc'])
        self.data_stats = util.md([self.outputs, 'data_stats'])
        
        # Making SCW run folder
        self.SCWrun = util.md([self.model_path, 'runSCW_' + model_name])
        self.logs = util.md([self.SCWrun, 'logs'])
        
        #path_to_mdu = os.path.exists(os.path.join(self.model_path,"*.dsproj_data","*.mdu"))
        #if path_to_mdu == False:
        self.DFM_Output = util.md([self.SCWrun, 'DFM_OUTPUT_kent_31' + model_name])
        #else:
            #file = os.path.split(glob.glob(path_to_mdu))[-1][:-3]
            #self.DFM_Output = util.md([self.SCWrun, file])
            
            # Copies across scw files needed to make model work. 
        data_directory = pkg_resources.resource_filename('o_func', 'data/SCW_files')
        data_files = pkg_resources.resource_listdir('o_func', 'data/SCW_files')
        for data_file in data_files:
            source_path = os.path.join(data_directory, data_file)
            destination_path = os.path.join(self.SCWrun, data_file)
            shutil.copy(source_path, destination_path)
            
        
        # Making MD README file. 
        textfile = os.path.join(self.model_path,'README.md')
        if os.path.exists(textfile) == False:
            with open(textfile, "w") as file:
                file.write("DATA INFORMATION for " + model_name + "\n\n")
                file.write("Friction Coefficient = "+ "\n")
                file.write("Friction Type = "+ "\n")
                
                file.write('\n############################ NOTES ############################\n\n')
                file.write('###############################################################')
            
        
# This function can be called directly from o_func
def write_directories(directory, model_name):
    main_path = os.path.join(start_path, directory)
    print(main_path)
    make_paths = DirGen(main_path)
    sub_path = make_paths.dir_outputs(model_name)
    
    
if __name__ == '__main__':
        
    main_path = os.path.join(start_path, 'testing_directory')
    make_paths = DirGen(main_path)
    sub_path = make_paths.dir_outputs('testing_folder')
    
    