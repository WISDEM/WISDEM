import os

import openmdao.api as om

import wisdem.inputs as sch
from wisdem.glue_code.gc_RunTools import PlotRecorder

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"
analysis_options = sch.load_analysis_yaml(fname_analysis_options)
wt_opt = om.Problem(reports=False, model=PlotRecorder(opt_options=analysis_options))
wt_opt.setup(derivatives=False)
wt_opt.run_model()
