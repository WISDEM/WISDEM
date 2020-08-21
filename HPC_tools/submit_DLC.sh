#!/bin/bash
#SBATCH --account=ssc
#SBATCH --time=0-04:00:00
#SBATCH --job-name=NREL5MW_rosco
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --mail-user nabbas@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=/home/nabbas/logs/job_5MWDLC.%j.out  # %j will be replaced with the job ID
#####SBATCH --mem=24GB
#SBATCH --mem=12GB
#####SBATCH --qos=high

#N=26
N=36

. /home/nabbas/Documents/batch_scripts/init_batch-env.sh

python3 /home/nabbas/Documents/pCrunch/runBatch/run_DLC_NREL5MW.py
