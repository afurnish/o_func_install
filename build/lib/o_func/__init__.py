#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" O_FUNC
        - A python package of ocean functions. 

Import dependecies for entire package and have them here
so that classes and functions can be split up into multiple 
files for easier readability. 
"""
print('\n#### Loaded up o_func ####\n')      
from .data_visuals.plotting_figures_for_latex import Plot

from .utilities.finder import finder
from .utilities.start import opsys3
from .utilities.choices import DataChoice

from .data_prepkit.gen_directory import write_directories

# This is now you import the classes and functions in the main 
#from .near_neigh import *  ## This will import all functions from near neigh into func_o



#from .plotting_figures_for_latex import Plot
# Created on Wed Nov 23 16:51:36 2022

# @author: af

