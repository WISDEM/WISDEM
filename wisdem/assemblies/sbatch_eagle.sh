#!/bin/bash
#SBATCH --account=bar
#SBATCH --time=24:00:00
#SBATCH --job-name=Design1
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=36
#SBATCH --mail-user pbortolo@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=/projects/windse/importance_sampling/logs/job_DeISgn_Case1.%j.out


nDV = 11  # Number of design variables (x2 for central difference)
nOF = 100 # Number of openfast runs per finite-difference evaluation
nC  = $(( nDV + nDV * nOF ))   # Number of cores needed. Make sure to request an appropriate number of nodes = N / 36

source deactivate

module purge
module load conda
module load mkl/2019.1.144 cmake/3.12.3
module load gcc/8.2.0

conda activate wisdem-env

mpirun -np $nC python main.py
