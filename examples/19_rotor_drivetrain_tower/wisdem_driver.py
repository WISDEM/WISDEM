import os

from wisdem import run_wisdem
from openmdao.utils.mpi import MPI
#from wisdem.postprocessing.compare_designs import run

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
refdir = os.path.join(os.path.dirname(mydir), "02_reference_turbines")
fname_wt_input = os.path.join(refdir, "IEA-3p4-130-RWT.yaml")
fname_modeling_options = os.path.join(refdir, "modeling_options_iea3p4.yaml")
fname_analysis_options = os.path.join(mydir, "analysis_options.yaml")

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
