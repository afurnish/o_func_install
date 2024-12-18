#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:11:40 2024

@author: af
"""

import os
import inspect
import pkgutil

def generate_package_dictionary(package_path, output_file):
    """
    Generates a dictionary of all scripts and functions in the package.
    
    Args:
        package_path (str): Path to the root directory of the package.
        output_file (str): Path to save the generated dictionary file.
    """
    package_name = os.path.basename(package_path)
    if not os.path.exists(package_path):
        raise FileNotFoundError(f"Package path '{package_path}' does not exist.")

    # Import the package dynamically
    package = __import__(package_name)
    package_path = package.__path__[0]

    package_dict = {}

    for module_info in pkgutil.walk_packages([package_path], f"{package_name}."):
        if module_info.ispkg:
            continue  # Skip sub-packages
        module_name = module_info.name
        try:
            module = __import__(module_name, fromlist=[''])
        except ImportError as e:
            print(f"Skipping {module_name} due to import error: {e}")
            continue

        functions = {}
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module_name:
                doc = inspect.getdoc(obj) or "No description available"
                functions[name] = doc

        if functions:
            package_dict[module_name] = functions

    # Write to text file
    with open(output_file, "w") as f:
        for module, funcs in package_dict.items():
            f.write(f"{module}:\n")
            for func, desc in funcs.items():
                f.write(f"  {func}:\n    {desc}\n")
            f.write("\n")

    print(f"Package dictionary saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Define your package path and output file
    package_home = "/media/af/PN/GitHub/o_func_install"  # Adjust to your package path
    output_file = os.path.join(package_home, "mypackage_dictionary.txt")

    generate_package_dictionary(package_home, output_file)

