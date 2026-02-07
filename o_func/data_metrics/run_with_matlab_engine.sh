#!/bin/bash
# Fail on error
set -e

# Set MATLAB runtime path
export LD_LIBRARY_PATH=/home/af/MATLAB/R2025a/bin/glnxa64:$LD_LIBRARY_PATH

# Activate micromamba base first
source ~/micromamba/etc/profile.d/micromamba.sh

# Activate the geovista environment
micromamba activate geovista

# Run the Python script you pass as argument
python3 "$@"

