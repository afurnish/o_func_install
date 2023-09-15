# -*- coding: utf-8 -*-
""" Finder 

An application for finding a string across all python code in the directory GitHub
Will only search in .py files.
 
Usage- 
    finder(line, directory)

line: 
    A string of the code or variable you want to find

directory:
    Starting directory to search. 
"""

# Created on Thu Apr 13 14:53:52 2023
# @author: aafur
import os
import fnmatch

def finder(line_to_search, starting_dir):
    # Define whether the search should be case-sensitive or case-insensitive
    case_sensitive = True
    
    # Function to check if a line of code exists in a file
    def check_line_in_file(file_path, line, starting_dir):
        with open(file_path, "r") as f:
            if case_sensitive:
                return line in f.read()
            else:
                return line.lower() in f.read().lower()
    
    # Search for Python files that contain the line of code
    found_files = []
    for dirpath, dirnames, filenames in os.walk(starting_dir):
        filenames = [filename for filename in filenames if not filename.startswith('.')]  # Exclude hidden files
        for filename in fnmatch.filter(filenames, "*.py"):
            file_path = os.path.join(dirpath, filename)
            if check_line_in_file(file_path, line_to_search, starting_dir):
                found_files.append(file_path)
    
    # Print the found file paths
    if found_files:
        print("Found Python files that contain the line of code:")
        for file_path in found_files:
            print(file_path.replace('\\','/'))
    else:
        print("No Python files were found that contain the line of code.")

if __name__ == '__main__':
    from o_func import opsys; start_path = opsys() # Aaron Code
    # Define the line of code to search for
    line_to_search = 'ERA_5_pressure.amp'#"sossheig"
    # Define the starting directory to search in
    #starting_dir = start_path + r"GitHub/python-oceanography/Delft 3D FM Suite 2019"
    starting_dir = start_path + r"GitHub"
    finder(line_to_search,starting_dir)
