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
main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'9.friction_calibration')
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
# def run_rsync(local_path, remote_path):
#     command = [
#         'rsync', '-avz', '-e', 'ssh',
#         '--include=*/',  # Include all directories to allow recursion.
#         '--include=run*/**',  # Include "run" directories and their entire subtree.
#         '--exclude=*',  # Exclude all other files and directories at the root level.
#         local_path, remote_path
#     ]
#     try:
#         subprocess.run(command, check=True)
#         print("rsync completed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"rsync failed: {e}")
# def run_rsync(local_path, remote_path):
#     # Extract the remote host and the path
#     remote_host, remote_dir_path = remote_path.split(':', 1)
    
#     # Command to create the remote directory structure
#     mkdir_command = [
#         'ssh', remote_host,
#         f'mkdir -p {remote_dir_path}'
#     ]
    
#     # The rsync command, as before
#     rsync_command = [
#         'rsync', '-avz', '-e', 'ssh',
#         '--include=*/',  # Include all directories to allow recursion.
#         '--include=run*/**',  # Include "run" directories and their entire subtree.
#         '--exclude=*',  # Exclude all other files and directories at the root level.
#         local_path, remote_path
#     ]
    
#     try:
#         # First, create the directory structure on the remote side
#         print("Creating remote directory structure...")
#         subprocess.run(mkdir_command, check=True)
        
#         # Then, proceed with the rsync operation
#         print("Starting rsync...")
#         subprocess.run(rsync_command, check=True)
        
#         print("rsync completed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Operation failed: {e}")


def run_rsync(local_path, remote_path):
    # Extract the remote host and the base path from the remote_path
    remote_host, base_remote_dir_path = remote_path.split(':', 1)
    # Generate the duplicate directory path by replacing the initial part of the path with /scratch
    duplicate_remote_dir_path = '/scratch' + base_remote_dir_path.split('/home', 1)[1]
    
    # SSH command to create the original and duplicate directory structures
    mkdir_command = [
        'ssh', remote_host,
        f"mkdir -p {base_remote_dir_path} {duplicate_remote_dir_path}"
    ]
    
    # The rsync command
    rsync_command = [
        'rsync', '-avz', '-e', 'ssh',
        '--include=*/',  # Include all directories to allow recursion.
        '--include=run*/**',  # Include "run" directories and their entire subtree.
        '--exclude=*',  # Exclude all other files and directories at the root level.
        local_path, f"{remote_host}:{base_remote_dir_path}"
    ]
    
    try:
        # Create the directory structures on the remote side
        print("Creating remote directory structures...")
        subprocess.run(mkdir_command, check=True)
        
        # Proceed with the rsync operation
        print("Starting rsync...")
        subprocess.run(rsync_command, check=True)
        
        print("rsync completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Operation failed: {e}")

# Example usage
# run_rsync("local/path/", "user@remotehost:/remote/path/")
        
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

def copy_bc_files(src_dir, dst_dir, file_names):
    """
    Copy specific files from the source directory to the destination directory.

    Parameters:
    - src_dir: Source directory path.
    - dst_dir: Destination directory path.
    - file_names: A list of file names to copy.
    """
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    for file_name in file_names:
        src_file_path = os.path.join(src_dir, file_name)
        if os.path.isfile(src_file_path):  # Check if the specific file exists
            dst_file_path = os.path.join(dst_dir, file_name)
            shutil.copy2(src_file_path, dst_file_path)  # Use copy2 to preserve metadata
        else:
            print(f"File not found: {src_file_path}")

def update_forcingfile(file_path, new_forcing_file):
    # Derive the new quantity based on the forcing file name
    new_quantity = new_forcing_file.split('.')[0].lower() + 'bnd'
    
    # Template for the new section to add
    new_section_template = """
[boundary]
quantity={quantity}
locationfile={locationfile}
forcingfile={forcingfile}
return_time=0.0000000e+000
"""

    # Check if the new forcing file already exists in the content
    content = ''
    with open(file_path, 'r') as file:
        content = file.read()
        if new_forcing_file in content:
            print("The forcing file already exists in the file.")
            return False
    
    # Attempt to extract the locationfile from the last boundary section
    try:
        last_location_file = content.strip().split('locationfile=')[-1].split('\n')[0].strip()
    except IndexError:
        print("Failed to extract the last locationfile.")
        return False
    
    # Format the new section using the derived quantity, the last locationfile, and the new forcing file
    new_section = new_section_template.format(quantity=new_quantity, locationfile=last_location_file, forcingfile=new_forcing_file)
    
    # Append the new section to the file
    with open(file_path, 'a') as file:
        file.write(new_section)
    # print("New section added to the file.")
    return True
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
    donor_model = os.path.join(start_path, 'modelling_DATA','kent_estuary_project','7.met_office','models','bathymetry_testing','runSCW_bathymetry_testing')      
    temperature_donor = os.path.join(start_path,'modelling_DATA','kent_estuary_project','7.met_office','files_bc','UKC4oa','1')     
    for model_input in ['oa']:
        for wind in ['nawind']:
            for flip in ['Orig']:
                for friction in ['0.000', '0.005','0.010','0.015', '0.020', '0.025', '0.030', '0.035']:
                    sub_path, fig_path, data_stats_path = make_paths.dir_outputs(model_input + '_' + wind +'_'+ flip + '_m'+ friction +'_Forcing')
                    run_path = glob.glob(os.path.join(sub_path,'run*'))[0]
                    # copy contents from doner_model to sub_path here. 
                    copy_contents(donor_model, run_path)
                    
                    config_path = glob.glob(os.path.join(run_path,'*.mdu'))[0]
                    update_val(config_path, 'UnifFrictCoef1D', friction)  # This one is for rivers and channels, any 1D components
                    update_val(config_path, 'UnifFrictCoef', friction)    # I believe this is the main one to be concerned about. 
                    update_val(config_path, 'UnifFrictCoefLin', '0.000') # This is one that needs to be ignored. Therefore it is now 0. 
                    
                    q_path = glob.glob(os.path.join(run_path,'*.q'))[0]
                    update_submission_file(q_path,
                                           job_name='nol'+friction, # NO_L NO LINEAR FRICTION can only be 8 characters
                                           time_taken='00-04:00', 
                                           partition='htc') # dev or htc

                    # Add in temperature boundary condition
                    add_files = ['Temperature.bc']
                    copy_bc_files(temperature_donor, run_path, add_files) # Add in the temperature boundary forcing. 
                    [update_forcingfile(glob.glob(os.path.join(run_path,'*.ext'))[0], i) for i in add_files]
                    
    remote_path = 'b.osu903@hawklogin.cf.ac.uk:/home/b.osu903/kent/friction_testing'
    push_to_scw = input('Do you want to push to SCW? (y/n): ')
    if push_to_scw == 'y':
        run_rsync(path_to_push, remote_path)
    
    
    
    
    
    
#%% Extra content     
    # local_path = path_to_push
    # remote_path = 'b.osu903@hawklogin.cf.ac.uk:/home/b.osu903/testpush/'
    
    # run_rsync(local_path, remote_path)
    