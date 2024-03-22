#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Automation of model creation
Created on Mon Mar  4 14:56:26 2024

@author: af
"""
import os
import subprocess
import shutil
import glob

from o_func import opsys; start_path = opsys()
from o_func import DirGen
main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'8.model_calibration')
make_paths = DirGen(main_path)
path_to_push = os.path.join(main_path, 'models/')
#%% Wind selector

#%% Input file selector

#%% Push to SCW
# def run_rsync(local_path, remote_path):
#     command = [
#         'rsync', '-avz', '-e', 'ssh',
#         '--include=*/', '--include=runSCW*/**', '--exclude=*',
#         local_path, remote_path
#     ]
#     try:
#         subprocess.run(command, check=True)
#         print("rsync completed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"rsync failed: {e}")
def run_rsync(local_path, remote_path):
    command = [
        'rsync', '-avz', '-e', 'ssh',
        '--include=*/',  # Include all directories to allow recursion.
        '--include=run*/**',  # Include "run" directories and their entire subtree.
        '--exclude=*',  # Exclude all other files and directories at the root level.
        local_path, remote_path
    ]
    try:
        subprocess.run(command, check=True)
        print("rsync completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"rsync failed: {e}")
        
def copy_contents(src_dir, dst_dir):
    """
    Copy the contents of src_dir to dst_dir, including files and subdirectories.
    The destination directory must exist.
    """
    # Iterate over all the directories and files in the source directory
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_item):
            # If the item is a directory, create it in the destination and copy its contents
            os.makedirs(dst_item, exist_ok=True)
            copy_contents(src_item, dst_item)  # Recursive call to copy subdirectory contents
        else:
            # If the item is a file, copy it directly
            shutil.copy2(src_item, dst_item)  # Use copy2 to preserve metadata

def update_val(config_path, key, new_value):
    """
    Update the value of a given key in the configuration file with a new value,
    preserving spaces around the "=" sign.
    
    Parameters:
    - config_path: The path to the configuration file.
    - key: The key whose value needs to be updated.
    - new_value: The new value to assign to the key, which will be treated as a string.
    """
    # Ensure the new value is treated as a string
    new_value_str = str(new_value)
    
    # Read the contents of the file
    with open(config_path, 'r') as file:
        lines = file.readlines()
    
    # Modify the line that contains the key
    for i, line in enumerate(lines):
        if line.strip().startswith(key):
            # Find the index of the "=" sign and preserve spaces around it
            equal_sign_index = line.find('=')
            if equal_sign_index != -1:
                # Preserve everything before the "=" sign (including spaces) and update the value
                lines[i] = line[:equal_sign_index+1] + ' ' + new_value_str + '\n'
                break  # Stop searching once the key is found and updated
    
    # Write the modified contents back to the file
    with open(config_path, 'w') as file:
        file.writelines(lines)
        
def update_submission_file(script_path, job_name=None, time_taken=None, partition=None):
    """
    Update job name, time taken, and partition in a SLURM submission script.

    Parameters:
    - script_path: Path to the SLURM submission script.
    - job_name: New job name to set.
    - time_taken: New time taken to set (format: days-hours:minutes).
    - partition: New partition to set.
    """
    # Read the contents of the file
    with open(script_path, 'r') as file:
        lines = file.readlines()
    
    # Modify the lines containing the parameters
    for i, line in enumerate(lines):
        if line.strip().startswith('#SBATCH --job-name=') and job_name:
            lines[i] = f'#SBATCH --job-name={job_name}\n'
        elif line.strip().startswith('#SBATCH --time=') and time_taken:
            lines[i] = f'#SBATCH --time={time_taken}\n'
        elif line.strip().startswith('#SBATCH --partition=') and partition:
            lines[i] = f'#SBATCH --partition={partition}\n'
    
    # Write the modified contents back to the file
    with open(script_path, 'w') as file:
        file.writelines(lines)



#%% Make and do the models
# Iterates through all the possible options I want to run. 

if __name__ == '__main__':
    # for model_input in ['oa', 'og', 'owa', 'ow']:
    #     for wind in ['yawind', 'nawind']:
    #         for flip in ['Flip', 'Orig']:
    #             sub_path, fig_path, data_stats_path = make_paths.dir_outputs(model_input + '_' + wind +'_'+ flip +'Forcing')


    # local_path = path_to_push
    # remote_path = 'b.osu903@hawklogin.cf.ac.uk:/home/b.osu903/testpush/'
    
    # run_rsync(local_path, remote_path)
    
    #%% Friction model generator to test friction characteristics
    
    # Doner model to be used - Input the river forcing files. 
    doner_model = os.path.join(start_path, 'modelling_DATA','kent_estuary_project','7.met_office','models','bathymetry_testing','runSCW_bathymetry_testing')           
    for model_input in ['oa']:
        for wind in ['nawind']:
            for flip in ['Orig']:
                for friction in ['0.000', '0.010','0.015', '0.020', '0.025', '0.030', '0.035', '0.040', '0.045', '0.050']:
                    sub_path, fig_path, data_stats_path = make_paths.dir_outputs(model_input + '_' + wind +'_'+ flip + '_m'+ friction +'_Forcing')
                    run_path = glob.glob(os.path.join(sub_path,'run*'))[0]
                    # copy contents from doner_model to sub_path here. 
                    copy_contents(doner_model, run_path)
                    
                    config_path = glob.glob(os.path.join(run_path,'*.mdu'))[0]
                    update_val(config_path, 'UnifFrictCoef1D', friction) 
                    update_val(config_path, 'UnifFrictCoef', friction)
                    update_val(config_path, 'UnifFrictCoefLin', friction)
                    
                    q_path = glob.glob(os.path.join(run_path,'*.q'))[0]
                    update_submission_file(q_path,
                                           job_name='m'+friction, 
                                           time_taken='00-03:00', 
                                           partition='htc') # dev or htc

                    
    remote_path = 'b.osu903@hawklogin.cf.ac.uk:/home/b.osu903/testpush/'
    run_rsync(path_to_push, remote_path)
    # local_path = path_to_push
    # remote_path = 'b.osu903@hawklogin.cf.ac.uk:/home/b.osu903/testpush/'
    
    # run_rsync(local_path, remote_path)
    