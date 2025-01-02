import os

from wisdem.glue_code.runWISDEM import run_wisdem

## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
fname_wt_input = os.path.join(run_dir, "IEA15MW_scaled.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options.yaml")

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
