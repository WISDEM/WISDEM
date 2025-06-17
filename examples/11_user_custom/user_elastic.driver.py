import os
import sys
import time

from openmdao.utils.mpi import MPI
from wisdem.glue_code.runWISDEM import run_wisdem

## File management
run_dir = os.path.dirname(os.path.realpath(__file__))
fname_wt_input = run_dir + os.sep + "IEA-15-240-RWT_VolturnUS-S_user_elastic.yaml"
fname_modeling_options = run_dir + os.sep + "modeling_options_user_elastic.yaml"
fname_analysis_options = run_dir + os.sep + "analysis_options_null.yaml"

tt = time.time()
wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)


if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print("Run time: %f" % (time.time() - tt))
    sys.stdout.flush()
