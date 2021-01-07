import os
import sys
import time

from wisdem.commonse.mpi_tools import MPI
from wisdem.glue_code.runWISDEM import run_wisdem

## File management
run_dir = os.path.dirname(os.path.realpath(__file__))
fname_wt_input_oc3 = run_dir + os.sep + "nrel5mw-spar_oc3.yaml"
fname_modeling_options = run_dir + os.sep + "modeling_options.yaml"
fname_analysis_options = run_dir + os.sep + "analysis_options.yaml"


tt = time.time()
wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input_oc3, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print("Run time: %f" % (time.time() - tt))
    sys.stdout.flush()
