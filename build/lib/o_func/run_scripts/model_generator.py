#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Automation of model creation
Created on Mon Mar  4 14:56:26 2024

@author: af
"""
import os
import subprocess

from o_func import opsys; start_path = opsys()
from o_func import DirGen
main_path = os.path.join(start_path, r'modelling_DATA','kent_estuary_project',r'8.model_calibration')
make_paths = DirGen(main_path)
path_to_push = os.path.join(main_path, 'models')
#%% Wind selector

#%% Input file selector

#%% Push to SCW
def run_rsync(local_path, remote_path):
    command = [
        'rsync', '-avz', '-e', 'ssh',
        '--include=*/', '--include=runSCW*/**', '--exclude=*',
        local_path, remote_path
    ]
    try:
        subprocess.run(command, check=True)
        print("rsync completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"rsync failed: {e}")

#%% Make and do the models
# Iterates through all the possible options I want to run. 

if __name__ == '__main__':
    for model_input in ['oa', 'og', 'owa', 'ow']:
        for wind in ['yawind', 'nawind']:
            for flip in ['Flip', 'Orig']:
                sub_path, fig_path, data_stats_path = make_paths.dir_outputs(model_input + '_' + wind +'_'+ flip +'Forcing')


    local_path = path_to_push
    remote_path = 'b.osu903@hawklogin.cf.ac.uk:/home/b.osu903/testpush/'
    
    run_rsync(local_path, remote_path)