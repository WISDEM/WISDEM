import os

from wisdem import run_wisdem
from wisdem.commonse.mpi_tools import MPI
from wisdem.postprocessing.compare_designs import run

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.join(os.path.dirname(mydir), "02_reference_turbines", "IEA-15-240-RWT.yaml")
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0

if rank == 0:
    print(
        "RUN COMPLETED. RESULTS ARE AVAILABLE HERE: "
        + os.path.join(mydir, analysis_options["general"]["folder_output"])
    )

run([wt_opt], ["optimized"], modeling_options, analysis_options)
