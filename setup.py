#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:36:53 2022

@author: af
"""

from setuptools import setup, find_namespace_packages
setup(
    name='o_func',
    packages=find_namespace_packages(include=['o_func','o_func.*']),
    package_data={'o_func': ['data/SCW_files/*','data/bash/*', 'data/wind/*', 'data/SCW_files_2023/*'],},
    version='0.1.0',
    description='Ocean Processing for completion of PhD in Ocean Sciences at Bangor University',
    author='Aaron Andrew Furnish'
)