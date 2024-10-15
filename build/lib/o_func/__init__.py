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
from .utilities.start import opsys, opsyst
from .utilities.choices import DataChoice
from .utilities.distance import uk_bounds, uk_bounds_wide


from .data_prepkit.gen_directory import write_directories
from .data_prepkit.gen_directory import DirGen

from .data_visuals.video_plots import VideoPlots


__authors__ = ['Aaron Furnish <afurnish@me.com>']

__version__ = "v0.0.1"
