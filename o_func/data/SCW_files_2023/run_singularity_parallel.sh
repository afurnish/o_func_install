#!/bin/bash

# Usage:
# D-Flow FM computations using a Singularity container,
# either sequential, or parallel computations using one node.
# For parallel using multiple nodes: use submit_singularity.sh.
#
# To start:
# 1. Be sure that a Singularity container is available, 
#    together with an execute_singularity.sh script in the same folder.
# 2. Copy the run_singularity script into your working folder, i.e. the folder containing the dimr config file.
# 3. Modify the run_singularity script, see remarks below.
# 4. Execute the script from the command line.
#    You can feed the script to a queueing system.
#    Examples:
#      On local machine:
#      > ./run_singularity.sh
#      Using SGE queueing system:
#      > qsub ./run_singularity.sh
#
# "execute_singularity.sh -p 2": Parent level to mount:
# If your working folder does not contain all of the input files, then you must set the -p flag.
# Let's define the "top level" as the folder containing all of the input files.
# The value of -p must be the number of folder levels between the dimr config file and the top level.
# A higher value will not cause any harm, provided that folders exist at the higher levels.
# 

# Access the passed arguments
NTASKS=$1
KENT_31=$2
OUTPUT=$3

#
#
# --- You will need to change the lines below -----------------------------
 
# Set number of partitions (this script only works for one node)
nPart=${NTASKS}											# SET FOR PARALLEL PROCESSES

# Set the path to the folder containing the singularity image and the execute_singularity.sh script. For example: 
singularitydir=/apps/environment/delft3dfm/2023.02/singularity				# DO NOT CHANGE


# DIMR input file; must already exist!
dimrFile=dimr.xml									# SET NAME OF INPUT CONFIG XML


#
#
# --- You shouldn't need to change the lines below ------------------------

# stop after an error occurred:
set -e


# Replace number of processes in DIMR file
PROCESSSTR="$(seq -s " " 0 $((nPart-1)))"
sed -i "s/\(<process.*>\)[^<>]*\(<\/process.*\)/\1$PROCESSSTR\2/" $dimrFile

# Read MDU file from DIMR file
mduFile="$(sed -n 's/\r//; s/<inputFile>\(.*\).mdu<\/inputFile>/\1/p' $dimrFile)".mdu

#
#
# --- Execution part: modify if needed ------------------------------------

if [ "$nPart" == "1" ]; then
    # Sequential computation
    # "-p": See above. Arguments after "run_dimr.sh" are explained in run_dimr.sh
    $singularitydir/execute_singularity.sh -p 2 run_dimr.sh -m $dimrFile
else
    # Parallel computation on one node
    #

    # First: partitioning 
    # (You can re-use a partition if the input files and the number of partitions haven't changed)
    # Partitioning is executed by dflowfm, in the folder containing the mdu file
    #cd path/to/directory/containing/the/mdu/file
    echo partitioning...
    # "-p": See above. Arguments after "run_dflowfm.sh" are explained in run_dflowfm.sh
    bash $singularitydir/execute_singularity.sh -p 2 run_dflowfm.sh --partition:ndomains=$nPart:icgsolver=6 $mduFile
    
    # Jump back to the dimr config file folder to execute dimr
    #cd path/to/directory/containing/the/dimr_config/file
    # Second: computation
    echo computation...
    # mpiexec is executed inside run_dimr.sh    
    # "-p": See above. Arguments after "run_dimr.sh" are explained in run_dimr.sh
    bash $singularitydir/execute_singularity.sh -p 2 run_dimr.sh -m $dimrFile -c $nPart

    # Finally: combine output files    
    # Optionally merge the map output files together into one file
    #cd path/to/directory/containing/the/dflowfm/output/files
    # "-p": See above. Arguments after "run_dfmoutput.sh" are explained in run_dfmoutput.sh

    MODEL_NAME=${KENT_31}
    OUTPUT_DIR=${OUTPUT}
    
    input_files=""
    for ((i=0; i<nPart; i++)); do
        input_files+="${OUTPUT_DIR}/kent_31_$(printf "%04d" $i)_map.nc "
    done

    # Print the constructed input files list for verification
    echo "Input files: ${input_files}"
    bash $singularitydir/execute_singularity.sh -p 2 run_dfmoutput.sh -- -d mapmerge --infile ${input_files} --outfile ${OUTPUT_DIR}/${MODEL_NAME}_merged_map.nc
    #cd jump/back
fi

