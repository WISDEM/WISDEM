import os
import sys
import time

from openmdao.utils.mpi import MPI
from wisdem.glue_code.runWISDEM import run_wisdem
import numpy as np
from numpy.testing import assert_almost_equal


## File management
run_dir = os.path.dirname(os.path.realpath(__file__))
fname_wt_input = run_dir + os.sep + "IEA-15-240-RWT_VolturnUS-S_User_Props.yaml"
fname_modeling_options = run_dir + os.sep + "modeling_options_user_props.yaml"
fname_analysis_options = run_dir + os.sep + "analysis_options.yaml"


tt = time.time()
wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# Most of these are set in yaml, can use geometry overrides to set better or without yamls
values_to_check = {
    'drivese.hub_system_mass':  [69360],
    'drivese.hub_system_I':  [973520., 619970., 619970.,      0.,      0.,      0.],
    'drivese.generator_rotor_I': [1836784.0, 1836784.0, 1836784.0], # probably needs fixing on my implementation, only need x-axis
    'drivese.above_yaw_mass':  [675175.0],
    'drivese.yaw_mass':  [0],
    'drivese.above_yaw_cm': [-4.528, -0.14 ,  4.098],
    'drivese.above_yaw_I_TT': [ 9912933., 10862815., 10360761.,        0.,        0.,        0.],
    'drivese.rna_I_TT': [ 9912933., 10862815., 10360761.,        0.,        0.,        0.],
    'nacelle.uptilt': [0.10471976],
    'nacelle.overhang': [12.032],
    'nacelle.distance_tt_hub': [5.6141],
}

for output, value in values_to_check.items():
    assert_almost_equal(wt_opt[output],value)


if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print("Run time: %f" % (time.time() - tt))
    sys.stdout.flush()
