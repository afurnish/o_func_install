# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:53:58 2023

@author: aafur
"""
import os
import glob
from o_func import opsys; start_path = opsys()
import pkg_resources
import shutil
import o_func.utilities as util

class DirGen:
    def __init__(self, main_dir):
        path_dict = {
            "path_to_rgfgrid_files" : "files_rgfgrid",
            "path_to_qgis_files" : "files_qgis",
            "path_to_boundary_conditions": "files_bc"
            }
        self.path_dict = path_dict
        
        #Makes main folder
        self.main_dir = util.md([main_dir])
            
        for key,value in path_dict.items(): 
            null = util.md([self.main_dir, value])
            del null
            
        self.main_models_path = util.md([self.main_dir, 'models'])
            

    def dir_outputs(self, model_name):
        # Identify if other models share the same name, then create their processing folder
        files = []
        for i, file in enumerate(glob.glob(os.path.join(self.main_models_path, "*"))):
            files.append(file)
        # Checking if file already exits
        same_file = glob.glob(os.path.join(self.main_models_path, "*"+model_name+"*"))
        # Sorting out numbering
        # num_files = [
        # folder for folder in os.listdir(self.main_dir)
        #     if os.path.isdir(os.path.join(self.main_dir, folder)) and "testing" not in folder
        #     ] # number of files but always ignore testing
        # if model_name == 'testing_folder':
        #     new_num = '00'
        # else:
        #     if len(same_file) == 0:
        #         new_num = str(len(num_files)-1).zfill(2)
                
        #     else:
        #         new_num = os.path.split(same_file[0])[-1][0:2]

        #self.model_path = util.md([self.main_dir, new_num + '_' + model_name])
        #self.model_path = util.md([self.main_models_path, new_num + '_' + model_name])
        self.model_path = util.md([self.main_models_path, model_name])
        # Making outputs folder for results and data 
        self.outputs = util.md([self.model_path, 'outputs'])
        self.inputs = util.md([self.model_path, 'inputs'])
        self.giffs = util.md([self.outputs, 'giffs'])
        self.figures = util.md([self.outputs, 'figures'])
        self.data_proc = util.md([self.outputs, 'data_proc'])
        self.data_stats = util.md([self.outputs, 'data_stats'])
        self.data_comp = util.md([self.outputs, 'compare'])
        
        self.Og = util.md([os.path.join(self.main_dir, 'files_bc'), 'UKO4g'])
        self.Coa = util.md([os.path.join(self.main_dir, 'files_bc'), 'UKC4oa'])
        self.Coaw = util.md([os.path.join(self.main_dir, 'files_bc'), 'UKC4oaw'])
        self.Cow = util.md([os.path.join(self.main_dir, 'files_bc'), 'UKC4ow'])
        
        #self.csv_dump1 = util.md([self.Og, 'dump_csv'])  # This is handled in the write boundary condition script. 
        #self.csv_dump2 = util.md([self.Coa, 'dump_csv'])
        #self.csv_dump3 = util.md([self.Coaw, 'dump_csv'])
        #self.csv_dump4 = util.md([self.Cow, 'dump_csv'])
        
        
        #Make video and images paths
        self.png_sh = util.md([self.figures, 'png_sh'])
        self.png_wd = util.md([self.figures, 'png_wd'])
        self.png_sal = util.md([self.figures, 'png_sal'])
        
        self.png_sal = util.md([self.figures, 'SanityCheck'])
        # This is specifically for figures that proves the validity of a choice 
        # For example you may want to plot tide gauge vs hourly average tide gauge. 
        
        
        
        # Making SCW run folder
        self.SCWrun = util.md([self.model_path, 'runSCW_' + model_name])
        self.logs = util.md([self.SCWrun, 'logs'])
        
        #path_to_mdu = os.path.exists(os.path.join(self.model_path,"*.dsproj_data","*.mdu"))
        #if path_to_mdu == False:
        self.DFM_Output = util.md([self.SCWrun, 'DFM_OUTPUT_kent_31']) # + model_name
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
        # print(self.model_path)
        
        return self.model_path , self.figures, self.data_stats
    
    def vid_var_path(self, var_choice = 'Surface_Height'):
        ''' Surface Height is the default. 
        '''
        var_mapping = {
            "Surface_Height" : self.png_sh,
            "Water_Depth" : self.png_wd,
            "Salinity" : self.png_sal
            }
        
        if var_choice in var_mapping:
            return var_mapping[var_choice]
        else:
            raise ValueError(f"Variable choice '{var_choice}' not found in the mapping.")

    def bc_outputs(self):
        '''
        Determines path directory for different condtion outputs from diff coupled
        states in UKC4 model runs. 

        Returns
        -------
        list
            DESCRIPTION.

        '''
        return [['UKC4g', 'UKC4oa','UKC4oaw','UKC4ow'],[self.Og, self.Coa, self.Coaw, self.Cow]]
         
         
        
    
# This function can be called directly from o_func
def write_directories(directory, model_name):
    ''' directory in this example should be the full path main folder in kent_model folder. 
        model name should be a reference frame. 
    '''
    
    main_path = os.path.join(start_path, directory)
    
    print(main_path)
    make_paths = DirGen(main_path)
    sub_path, fig_path, data_stats_path = make_paths.dir_outputs(model_name)
    print(sub_path)
    
    
if __name__ == '__main__':
        
    # main_path = os.path.join(start_path, 'testing_directory')
    # make_paths = DirGen(main_path)
    # sub_path = make_paths.dir_outputs('testing_folder')
    
    #from o_func.data_prepkit import DirGen
    # main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'6.Final2')
    main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'7.met_office')
    make_paths = DirGen(main_path)
    sub_path, fig_path, data_stats_path = make_paths.dir_outputs('bathymetry_testing_duddon_removed')
    
    # for model_input in ['oa', 'og', 'owa', 'ow']:
    #     for wind in ['yawind', 'nawind']:
    #         for flip in ['Flip', 'Orig']:
    #             sub_path, fig_path, data_stats_path = make_paths.dir_outputs(model_input + '_' + wind +'_'+ flip +'Forcing')
    
    # 
    #/media/af/PD/modelling_DATA/kent_estuary_project/6.Final2/models/kent_1.3.7_testing_4_days_UM_run