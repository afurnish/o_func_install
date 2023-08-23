#!/usr/bin/env bash

#SBATCH --output logs/run.out.%J        # Output file
#SBATCH --error logs/run.err.%J         # Error file
#SBATCH --job-name=kent_31   		    # Name
#SBATCH --nodes 1						# Total number of nodes
#SBATCH --ntasks 40               		# Total number of cores  CHANGE ME 
#SBATCH --account=scw1987				# Account CHANGE ME 
#SBATCH --time=00-08:00			#wall time alloed dasy-hrs:min
#SBATCH --partition=htc         #dev or htc depending on which environmnet you want to run in
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=osu903@bangor.ac.uk # CHANGE ME 
# --partition highmem

echo " "
work_dir=$(pwd | cut -d/ -f 4-)
echo "home directory: " $work_dir
echo " "

# Load modules
module purge
module load delft3dfm/63285

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Create scratch directory
home_dir=$PWD
scratch_dir="/scratch/${USER}/${work_dir}-${SLURM_JOB_ID}"
echo ">>>>>>>>>>>> creating scratch directory: ${scratch_dir}"
mkdir "${scratch_dir}"

# Partition the file
echo ">>>>>>>>>>>> partioning files"
dflowfm --partition:ndomains=${SLURM_NTASKS}:icgsolver=7 kent_31.mdu # CHANGE .mcdu file name 

# Copy files to scratch
echo ">>>>>>>>>>>> copying to scratch"
rsync -aP --exclude-from "exclude-list.txt" "${home_dir}/" "${scratch_dir}"

# Run
echo ">>>>>>>>>>>> run model"
cd "${scratch_dir}"
run_dimr.sh -c "${SLURM_NTASKS}"

# Merge results files
cd DFM_OUTPUT_kent_31
module load delft3dfm/63285
dfmoutput mapmerge --infile kent_31_0000_map.nc kent_31_0001_map.nc kent_31_0002_map.nc kent_31_0003_map.nc kent_31_0004_map.nc kent_31_0005_map.nc kent_31_0006_map.nc kent_31_0007_map.nc kent_31_0008_map.nc kent_31_0009_map.nc kent_31_0010_map.nc kent_31_0011_map.nc kent_31_0012_map.nc kent_31_0013_map.nc kent_31_0014_map.nc kent_31_0015_map.nc kent_31_0016_map.nc kent_31_0017_map.nc kent_31_0018_map.nc kent_31_0019_map.nc kent_31_0020_map.nc kent_31_0021_map.nc kent_31_0022_map.nc kent_31_0023_map.nc kent_31_0024_map.nc kent_31_0025_map.nc kent_31_0026_map.nc kent_31_0027_map.nc kent_31_0028_map.nc kent_31_0029_map.nc kent_31_0030_map.nc kent_31_0031_map.nc kent_31_0032_map.nc kent_31_0033_map.nc kent_31_0034_map.nc kent_31_0035_map.nc kent_31_0036_map.nc kent_31_0037_map.nc kent_31_0038_map.nc kent_31_0039_map.nc # CHANGE result file names

#Finish
echo "Run Finished!"

# # Copy from scratch to home
# echo ">>>>>>>>>>>> copying from scratch to home"
# rsync -aP "${scratch_dir}" "${home_dir}/output" 

# # Remove files from scratch
# echo ">>>>>>>>>>>> removing files from scratch"
#  rm -rf "${scratch_dir}"
