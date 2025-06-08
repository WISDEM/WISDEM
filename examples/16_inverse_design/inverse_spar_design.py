import os
import sys
import time

from openmdao.utils.mpi import MPI
from wisdem.glue_code.runWISDEM import run_wisdem

## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
wisdem_examples = os.path.dirname(os.path.dirname(run_dir))
fname_wt_input_oc3 = os.path.join(wisdem_examples, "09_floating", "nrel5mw-spar_oc3.yaml")
fname_modeling_options = os.path.join(wisdem_examples, "09_floating", "modeling_options_noRNA_spar.yaml")
fname_analysis_options = run_dir + os.sep + "analysis_options_spar.yaml"


wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input_oc3, fname_modeling_options, fname_analysis_options)
