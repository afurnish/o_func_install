# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:36:58 2023

@author: aafur
"""
# from iris.analysis.cartography import unrotate_pole
# class Transform:
#     def __init__(self, lon, lat):
#         self.lon = lon 
#         self.lat = lat
#         #uk defined centre of rotation
#         # lon, lat = 40, -42
#         lons, lats = unrotate_pole(rotated_lons, rotated_lats,  lon, lat)
        
#     #def unrotate_pols():
import os
# def flip_and_relabel(file_path):
#     # Read the content of the file
    
#     splits = os.path.split(file_path)
#     file_location = splits[0]
#     file_name = splits[1]
#     # Read the content of the file
#     with open(file_path, 'r') as file:
#         content = file.readlines()

#     # Find the start and end indices of the data within [forcing]
#     start_indices = [i for i, line in enumerate(content) if line.startswith('[forcing]')]
#     end_indices = start_indices[1:] + [len(content)]

#     # Extract and reorder the datasets
#     reordered_data = []
#     for start, end in zip(reversed(start_indices), reversed(end_indices)):
#         dataset_lines = content[start:end]
#         reordered_data.extend(dataset_lines)

#     # Write the modified content to a new file
#     new_file_path = file_path.replace('.bc', '_reordered.bc')
#     with open(new_file_path, 'w') as new_file:
#         new_file.writelines(reordered_data)

#     print(f"File successfully reordered. Result saved to {new_file_path}")

def reorder_and_update_index(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Find the start and end indices of the data within [forcing]
    start_indices = [i for i, line in enumerate(content) if line.startswith('[forcing]')]
    end_indices = start_indices[1:] + [len(content)]

    # Extract and reorder the datasets
    reordered_data = []
    for i, (start, end) in enumerate(zip(reversed(start_indices), reversed(end_indices)), start=1):
        dataset_lines = content[start:end]

        # Update the index underneath [forcing]
        dataset_lines[1] = f"Name                            = 001_delft_ocean_boundary_UKC3_b601t688_length-87_points_{i:04d}\n"

        reordered_data.extend(dataset_lines)

    # Write the modified content to a new file
    new_file_path = file_path.replace('.bc', '_reordered_updated_index.bc')
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(reordered_data)

    print(f"File successfully reordered and updated. Result saved to {new_file_path}")

if __name__ == '__main__':
    # Specify the path to your file
    file_path = r'N:/modelling_DATA/kent_estuary_project/7.met_office/models/upside_down_ocean_bound/upside_down_no_hole_mesh.dsproj_data/FlowFM/WaterLevel.bc'
    
    # Call the function to flip and relabel the points
    reorder_and_update_index(file_path)
        
        