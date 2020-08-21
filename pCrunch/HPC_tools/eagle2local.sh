# ------ Sync DLC Data from eagle runs locally -------

# --- 5MW LAND LEGACY ---
outdir='/projects/ssc/nabbas/DLC_Analysis/5MW_Land/'
indir='../BatchOutputs/5MW_Land/5MW_Land_legacy/'
mkdir -p $indir;
rsync nabbas@eagle.hpc.nrel.gov:$outdir*.outb $indir
rsync nabbas@eagle.hpc.nrel.gov:$outdir/case_matrix.yaml $indir

# --- 5MW LAND ROSCO ---
outdir2='/projects/ssc/nabbas/DLC_Analysis/5MW_OC3Spar'
indir2='../BatchOutputs/5MW_Land/5MW_Land_ROSCO/'
mkdir -p $indir2;
rsync nabbas@eagle.hpc.nrel.gov:$outdir2*.outb $indir2
rsync nabbas@eagle.hpc.nrel.gov:$outdir2/case_matrix.yaml $indir2
