#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Create a function to call start up folder choice

Created on Mon Oct 24 14:13:34 2022
@author: af
"""
import os
#from sys import platform
#import time

# def opsys():
    
#     print('\n')
#     print('Select Drive from list below...')
#     drive = ''
#     if platform == "win32":
#         drives = [ chr(x) + ":" for x in range(65,91) \
#                   if os.path.exists(chr(x) + ":") ]
#         print(drives)
#         options = [x[:-1] for x in drives]
#         while drive not in options:
#             drive = str(input("Windows Drive Letter only (capitalised): "))
#         start_path = drive + r':/'
#     elif platform == "linux" or platform == "linux2":
#         #edit this when you get a chance to work on linux systems
#         # This will not work again on a linux system. 
#         lin_path = os.path.expanduser('~')
#         lin_path2 = lin_path[6::]
#         drives = os.listdir('/media/aafur')
#         time.sleep(0.5)
#         print(drives)
#         print('\n')
#         time.sleep(0.5)
#         while drive not in drives:
#             drive = str(input("Linux drive name only (capitalised): "))
#         start_path = r'/media/' + lin_path2 + '/' + drive +'/'
#     elif platform == "darwin":
#         lin_path = r'/Volumes'
#         drives = os.listdir(lin_path)
#         print(drives)
#         print('\n')
#         while drive not in drives:
#             drive = str(input("Mac drive name only (capitalised): "))
#         start_path = lin_path + r'/' + drive + r'/'
        
#     print('start_path defined as: ' + start_path)
#     print('\n') 
#     return start_path

# def opsys2():

#     index = 0
#     empty = []
#     drive_label = "PD"
#     if os.name == "nt":  # Windows
#         drives = [d for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:")]
        
#         drives_to_clear = []
#         for d in drives:
#             drive_info = os.popen(f"vol {d}:").read().strip()
#             index = drive_info.find('PD',0)
#             if index == -1:
#                 drives_to_clear.append(d)
#         result = [item for item in drives if item not in drives_to_clear]
#         if result == empty:
#             raise Exception(f"Volume {drive_label} not found")

#         print('\nVolume PD has been detected\n')
#         start_path = result[0] + r':/'
               
                
#     #sort out windows or mac stuff later   
#     else:  # Linux or Mac
#         mount_points = [line.split()[1] for line in os.popen("mount").read().splitlines() if line.startswith("/dev/") and f"/{drive_label}" in line]
#         mount_points = os.popen("mount").read().splitlines()
#         drive = []
#         for mount in mount_points:
#             if mount.find('PD',0) != -1:
#                 drive.append(mount.split(' ')[2])
                
#         if drive == []:        
#             raise Exception(f"Drive {drive_label} not found")
#         start_path = drive[0] + r'/'
#     return start_path

def find_drive_label(drive_label):
    if os.name == "nt":  # Windows
        drives = [d for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:")]
        
        drives_to_clear = []
        for d in drives:
            drive_info = os.popen(f"vol {d}:").read().strip()
            if 'PN' not in drive_info:
                drives_to_clear.append(d)
                
        result = [item for item in drives if item not in drives_to_clear]
        if not result:
            raise Exception(f"Volume {drive_label} not found")
        
        return result[0] + ':\\'
        
    else:  # Linux or Mac
        mount_points = [line.split()[2] for line in os.popen("mount").read().splitlines() if line.startswith("/dev/") and f"/{drive_label}" in line]
        
        if not mount_points:
            raise Exception(f"Drive {drive_label} not found")
            
        return mount_points[0] + r'/'

def opsys(drive_label="PN"):
    start_path = find_drive_label(drive_label)
    #print(f'\nDrive {drive_label} has been detected\n')
    return start_path




