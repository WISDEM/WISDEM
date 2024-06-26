import os
from wisdem.glue_code.runWISDEM import run_wisdem

## File management
run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep
wisdem_examples = os.path.dirname(os.path.dirname(run_dir))
fname_wt_input = os.path.join(wisdem_examples, "03_blade", "BAR_USC.yaml")
fname_modeling_options = os.path.join(wisdem_examples, "03_blade", "modeling_options.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_custom.yaml")

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

