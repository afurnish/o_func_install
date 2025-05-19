#!/bin/bash

# Parse command-line arguments
while getopts ":i:o:x:y:X:Y:" opt; do
  case $opt in
    i) input_file="$OPTARG";;
    o) output_file="$OPTARG";;
    x) lon_min="$OPTARG";;
    y) lon_max="$OPTARG";;
    X) lat_min="$OPTARG";;
    Y) lat_max="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done
echo "Input File: ${input_file}"
echo "Output File: ${output_file}"

# Use ncks to cut the dataset based on the specified indices
ncks -d lon,${lon_min},${lon_max} -d lat,${lat_min},${lat_max} "${input_file}" "${output_file}"

