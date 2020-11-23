import os
from wisdem import run_wisdem


## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
#fname_wt_input         = mydir + os.sep + 'iea_rotor_drive.yaml'
fname_wt_input         = mydir + os.sep + '../02_reference_turbines' + os.sep + 'IEA-15-240-RWT.yaml'
fname_modeling_options = mydir + os.sep + 'modeling_options.yaml'
fname_analysis_options = mydir + os.sep + 'analysis_options.yaml'

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
