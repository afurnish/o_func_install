#!/bin/bash
# Usage: ./cut_salinity.sh input_path output_path
# The idea is to find patterns of delft_ocean_boundary
# The combination of these conditions means that awk will start printing 
# lines when it encounters the first delft_ocean_boundary, stop printing 
# after an empty line following the second occurrence of delft_ocean_boundary,
# and then exit.
#

input_path=$1
output_path=$2

# Ensure the output file is empty
> "$output_path"

# Ensure we received input and output paths
if [ -z "$input_path" ] || [ -z "$output_path" ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

# Debug information: Print input and output paths
echo "Input file: $input_path"
echo "Output file: $output_path"

# Find the line number where "Name = Alt_0001" occurs
line_number=$(grep -n "Name                            = Alt_0001" "$input_path" | cut -d: -f1)

# Debug information: Print the found line number
echo "Found line number: $line_number"

# If the line is found
if [ -n "$line_number" ]; then
    # Extract lines from the beginning up to the line before the found line
    head -n $((line_number - 1)) "$input_path" > "$output_path"
    
    # Debug information: Confirm lines were written to the output file
    echo "Lines written to output file up to line $((line_number - 1))"

    # Remove the last line from the output file
    sed -i '' -e '$d' "$output_path"
    sed -i '' -e '$d' "$output_path"
    sed -i '' -e '$d' "$output_path"
    
    # Debug information: Confirm the last line was removed
    echo "Last line removed from output file"
else
    echo "The specified line 'Name                            = Alt_0001' was not found."
    exit 1
fi

echo "Script completed successfully."