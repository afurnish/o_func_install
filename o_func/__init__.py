#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" O_FUNC
        - A python package of ocean functions. 

Import dependecies for entire package and have them here
so that classes and functions can be split up into multiple 
files for easier readability. 
"""
from .data_visuals.plotting_figures_for_latex import Plot

from .utilities.finder import finder
from .utilities.start import opsys
from .utilities.choices import DataChoice

from .data_prepkit.gen_directory import write_directories
from .data_prepkit.gen_directory import DirGen


