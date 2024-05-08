#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:29:59 2024

@author: af
"""

import pyproj



# Extract coordinates and convert them using pyproj
def convert_coordinates(data):
    transformer = pyproj.Transformer.from_crs("epsg:32630", "epsg:4326", always_xy=True)
    converted_data = []
    for line in data:
        x, y, z = map(float, line.split())
        lon, lat = transformer.transform(x, y)
        converted_data.append((lon, lat, z))
    return converted_data

if __name__ =='__main__':
    # Sample data from the user's provided text
    sample_data = [
        "332290.000000 5567970.000000 -7.68293",
        "332326.406250 5569168.500000 -20.3086",
        "332355.562500 5570087.000000 -24.9252",
        "332384.656250 5571004.500000 -26.729",
        "332413.718750 5571921.000000 -29.5954"
    ]
    converted_sample = convert_coordinates(sample_data)
    print(converted_sample)