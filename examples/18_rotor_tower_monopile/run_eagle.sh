#!/bin/bash
#SBATCH --account=windse
#SBATCH --time=04:00:00
#SBATCH --nodes=3
#SBATCH --job-name=20mw_tut
#SBATCH --mail-user username@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=job_log.%j.out  # %j will be replaced with the job ID

# Load and activate your conda environment
source activate wisdem-env
# Run the design optimization
mpirun -np 92 python design_run.py
# Create convergence plots
python create_conv_plots.py
# Compare initial designs at 15MW and linearly scaled 15MW with final design found by WISDEM
compare_designs IEA15MW_FB.yaml IEA15MW_FB_scaled.yaml 20MW_opt/IEA20MW_FB.yaml --labels 15MW 15MWs 20MWopt
