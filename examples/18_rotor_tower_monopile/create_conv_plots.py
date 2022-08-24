import openmdao.api as om
from wisdem.glue_code.gc_RunTools import PlotRecorder
import wisdem.inputs as sch
import os


mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"
analysis_options = sch.load_analysis_yaml(fname_analysis_options)
wt_opt = om.Problem(model=PlotRecorder(opt_options=analysis_options))
wt_opt.setup(derivatives=False)
wt_opt.run_model()