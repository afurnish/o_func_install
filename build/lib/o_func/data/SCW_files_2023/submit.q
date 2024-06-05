#!/usr/bin/env bash

#SBATCH --output logs/run.out.%J        # Output file
#SBATCH --error logs/run.err.%J         # Error file
#SBATCH --job-name=riv_exmp
#SBATCH --nodes 1						# Total number of nodes
#SBATCH --ntasks 40               		# Total number of cores  CHANGE ME 
#SBATCH --account=scw1987				# Account CHANGE ME 
#SBATCH --time=00-20:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=osu903@bangor.ac.uk # CHANGE ME 
# --partition highmem

# Set filenaming conventions
KENT_31="kent_31"
OUTPUT_DIR="output"

echo " "
work_dir=$(pwd | cut -d/ -f 4-)
echo "home directory: " $work_dir
echo " "

# Load modules
module load delft3dfm/2023.02


# Save SLURM_NTASKS to a variable
NTASKS=${SLURM_NTASKS}

# Create scratch directory
home_dir=$PWD
scratch_dir="/scratch/${USER}/${work_dir}-${SLURM_JOB_ID}"
echo ">>>>>>>>>>>> creating scratch directory: ${scratch_dir}"
mkdir -p "${scratch_dir}"

# Copy files to scratch
echo ">>>>>>>>>>>> copying to scratch"
rsync -aP --exclude-from "exclude-list.txt" "${home_dir}/" "${scratch_dir}"
rsync -aP run_singularity_parallel.sh "${scratch_dir}"
cd "${scratch_dir}"

# Run
echo ">>>>>>>>>>>> run model"
start="$(date +%s)"

bash ./run_singularity_parallel.sh ${NTASKS} ${KENT_31} ${OUTPUT_DIR}

stop="$(date +%s)"
finish=$(( $stop-$start ))
echo delft3dfm ${SLURM_JOBID}: Job-Time ${finish} seconds. End Time is `date`.

#post_processing_of_data
cd ${OUTPUT_DIR}
rm *00* # removes all of the partitioned generated files. 
module load anaconda &>/dev/null
eval "$(/apps/languages/anaconda/2021.11/bin/conda shell.bash hook)"
conda activate phd
regrid.py ${KENT_31}_merged_map.nc
echo "Run Finished!"

# Run water quality merging only if water quality merging exisit. 

# Loop through each item in the current directory
for dir in */; do
  # Check if the directory name contains 'DFM_DELWAQ'
  if [[ "$dir" == *DFM_DELWAQ* ]]; then
    echo "Found directory with 'DFM_DELWAQ' in its name. Running water quality stitching..."
    
    # Perform water quality stitching together with waqmerge
    cd "${scratch_dir}"
    mkdir DELWAQ # Location of final waqmerge actual merged files.
    mkdir DFM_DELWAQ_kent_31 # Location of other waqmerge files to be discarded
    module load delft3dfm/2023.02
    # Load up the waqmerge from 2023 and run the .mdu file within it
    bash copy_waq.sh # moves from 00** files into DFM_DELWAQ_kent_31
    # Perform actual waqmerge after water quality files have been moved to single folder
    /apps/environment/delft3dfm/2023.02/singularity/execute_singularity.sh waqmerge kent_31.mdu
    bash move_waq.sh -s DFM_DELWAQ_kent_31 # moves the merged files into destination of DELWAQ
  fi
done

