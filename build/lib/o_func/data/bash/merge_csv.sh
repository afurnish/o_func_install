#!/bin/bash

# Check if there are at least two arguments (input files and output file)
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <output_file> <input_file1> <input_file2> ..."
  exit 1
fi

# Extract the output file name (first argument)
output_file="$1"
shift  # Remove the first argument

# Concatenate CSV files with an empty line in between
for file in "$@"; do
  cat "$file" >> "$output_file"
  echo "" >> "$output_file"
done