import os

from wisdem import run_wisdem
from openmdao.utils.mpi import MPI
#from wisdem.postprocessing.compare_designs import run

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.join(mydir, "BAR_USC.yaml")
fname_modeling_options = os.path.join(mydir, "modeling_options.yaml")
analysis_options_files = ["analysis_options_no_opt.yaml",
                          "analysis_options_aero.yaml",
                          "analysis_options_struct.yaml",
                          "analysis_options_aerostruct.yaml",
                          "analysis_options_user.yaml",
                          ]

for a in analysis_options_files:
    fname_analysis_options = os.path.join(mydir, a)
    wt_opt, modeling_options, analysis_options = run_wisdem(
        fname_wt_input, fname_modeling_options, fname_analysis_options
    )

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0

if rank == 0:
    print(
        "RUN COMPLETED. RESULTS ARE AVAILABLE HERE: "
        + os.path.join(mydir, analysis_options["general"]["folder_output"])
    )
