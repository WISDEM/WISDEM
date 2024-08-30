import os
import openmdao.api as om
from wisdem.glue_code.gc_RunTools import PlotRecorder
import wisdem.inputs as sch

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

analysis_options_files = ["analysis_options_no_opt.yaml",
                          "analysis_options_aero.yaml",
                          "analysis_options_struct.yaml",
                          "analysis_options_aerostruct.yaml",
                          "analysis_options_user.yaml",
                          ]

for a in analysis_options_files:
    fname_analysis_options = os.path.join(mydir, a)
    analysis_options = sch.load_analysis_yaml(fname_analysis_options)
    wt_opt = om.Problem(model=PlotRecorder(opt_options=analysis_options))
    wt_opt.setup(derivatives=False)
    wt_opt.run_model()