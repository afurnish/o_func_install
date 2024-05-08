#!/bin/bash
# Usage: ./cut_salinity.sh input_path output_path

input_path=$1
output_path=$2

awk '/delft_ocean_boundary/ {p=1; count++} p; /^$/ && p {p=0; if(count==2) exit}' "$input_path" > "$output_path"
