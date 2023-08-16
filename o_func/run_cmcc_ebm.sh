#!/bin/bash
#BSUB -q s_short 
#BSUB -J cmccEBM
#BSUB -n 1
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -P 0310

touch J.out
touch J.err

##----working directories---##
R_OUT="./"
R_ebm_output="./output"
cd ${R_OUT}
rm *err *out

##---preliminary compile fortran code and load netcdf modules (README_How_to_use_CMCC_EBM)---##

##---- define Input parameters of  the test case---##
side= #logical param on river mouth location   #e.g. west
#side maybe {south, north,west,east}

year=
days=  
hours=
jday=

##---- define Input files (README_inputs_output_files_for_TestCase)---##
InFile_param=
InFile_ocean=
InFile_Qr=
InFile_vel_tide=
msk=
miss=

##execute the code with 11 external arguments: 4 files and 7 parameters
  #./cmcc_ebm_daily_Po.exe $InFile_Qr $InFile_param $InFile_vel_tide $InFile_ocean ${side} ${year} ${days} ${hours} ${jday}
  ./cmcc_ebm_daily_Po.exe $InFile_Qr $InFile_param $InFile_vel_tide $InFile_ocean ${side} ${year} ${days} ${hours} ${jday} ${miss} ${msk}

##-----Move output files---##
outFile=
mv ${outFile} ${R_ebm_output}/
