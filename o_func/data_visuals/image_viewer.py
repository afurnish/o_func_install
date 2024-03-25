#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Python Image viewer for comparative analysis
Created on Mon Mar  4 12:10:51 2024

@author: af
"""

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

from o_func import opsys; start_path = opsys()

def list_subdirectories(parent_path):
    """List all subdirectories in a given parent directory."""
    return [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]

def select_directories(subdirectories, max_selections=4):
    """Let the user select which subdirectories to use, up to a maximum."""
    print("Select up to 4 directories by index:")
    for i, directory in enumerate(subdirectories):
        print(f"{i}: {directory}")
    selected_indices = input("Enter the indices of the directories to view, separated by commas (e.g., 0,2,3): ")
    selected_indices = [int(x.strip()) for x in selected_indices.split(',')][:max_selections]
    return [subdirectories[i] for i in selected_indices]

def collect_unique_image_names(model_dirs, main_path):
    """Collect all unique .png image names across the selected model directories, including those in SanityCheck subfolders, ignoring files starting with '.'."""
    unique_images = set()
    for dir_name in model_dirs:
        figures_path = os.path.join(main_path, dir_name, 'outputs', 'figures')
        sanity_check_path = os.path.join(figures_path, 'SanityCheck')
        
        def add_png_files_from_path(path):
            for _, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.png') and not file.startswith('.'):
                        unique_images.add(file)
        
        add_png_files_from_path(figures_path)
        if os.path.exists(sanity_check_path):
            add_png_files_from_path(sanity_check_path)

    return sorted(list(unique_images))

def find_image_paths(model_dirs, main_path, image_name):
    """Find paths of the specified image in the selected model directories, checking both the figures directory and the SanityCheck subdirectory."""
    image_paths = []
    for dir_name in model_dirs:
        figures_path = os.path.join(main_path, dir_name, 'outputs', 'figures')
        sanity_check_path = os.path.join(figures_path, 'SanityCheck')
        
        image_path_figures = os.path.join(figures_path, image_name)
        image_path_sanity_check = os.path.join(sanity_check_path, image_name)
        
        if os.path.isfile(image_path_figures):
            image_paths.append(image_path_figures)
        elif os.path.isfile(image_path_sanity_check):
            image_paths.append(image_path_sanity_check)
        else:
            image_paths.append(None)
    
    return image_paths

def display_images(image_paths, direct_select):
    """Display images in a dynamic grid based on the number of images."""
    num_images = len(image_paths)
    
    # Determine the layout based on the number of images
    if num_images == 1:
        cols = 1
        rows = 1
    elif num_images == 2:
        cols = 2
        rows = 1
    elif num_images == 3:
        cols = 2  # Use 2 columns: first row will have 2 images, second row 1 image.
        rows = 2
    else:  # For 4 images, use a 2x2 grid
        cols = 2
        rows = 2
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_images == 1:
        axs = [axs]  # Make it iterable
    else:
        axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < num_images and image_paths[i]:  # Check if the image path exists and is within the selected range
            image = imread(image_paths[i])
            ax.imshow(image)
            tit_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_paths[i]))))
            ax.set_title(direct_select[i])
            ax.axis('off')
        elif i >= num_images:
            fig.delaxes(ax)  # Remove extra axes if they're not needed
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_path = os.path.join(start_path, 'modelling_DATA', 'kent_estuary_project', '8.model_calibration', 'models')
    
    subdirectories = list_subdirectories(main_path)
    selected_dirs = select_directories(subdirectories)
    
    if selected_dirs:
        unique_images = collect_unique_image_names(selected_dirs, main_path)
        print("Select an image to display:")
        for i, image_name in enumerate(unique_images):
            print(f"{i}: {image_name}")
        image_index = int(input("Enter the index of the image: "))
        image_name = unique_images[image_index]
     
        # Find and display the images
        image_paths = find_image_paths(selected_dirs, main_path, image_name)
        display_images(image_paths, selected_dirs)
    else:
        print("No directories selected or found.")
