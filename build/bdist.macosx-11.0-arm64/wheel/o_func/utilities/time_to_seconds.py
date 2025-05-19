#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:50:25 2024

@author: af
"""

from datetime import datetime, timedelta

# Define the start date and time
start_date_str = "20131031 00:00"
start_date = datetime.strptime(start_date_str, "%Y%m%d %H:%M")

# Add 3600 seconds (1 hour)
new_date = start_date + timedelta(seconds=10368000)

# Display the new date and time
print("New date and time:", new_date.strftime("%Y%m%d %H:%M:%S"))

def seconds_since(reference_date_str, current_date_str):
    # Convert the reference date and current date strings into datetime objects
    reference_date = datetime.strptime(reference_date_str, "%Y%m%d %H:%M:%S")
    current_date = datetime.strptime(current_date_str, "%Y%m%d %H:%M:%S")
    
    # Calculate the difference between the current date and the reference date
    time_difference = current_date - reference_date
    
    # Return the total number of seconds
    return time_difference.total_seconds()

# Example usage
reference_date_str = "20131031 00:00:00"
current_date_str = "20140228 00:00:00"

elapsed_seconds = seconds_since(reference_date_str, current_date_str)
print(f"Seconds since {reference_date_str}: {elapsed_seconds}")