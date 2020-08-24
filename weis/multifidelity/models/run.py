from wisdem.glue_code.runWISDEM import run_wisdem
from wisdem.commonse.mpi_tools import MPI
import os


## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
# fname_wt_input         = run_dir + 'IEAonshoreWT_3.yaml'
fname_wt_input = run_dir + "IEA-15-240-RWT_WISDEMieaontology4all.yaml"
fname_modeling_options_ccblade = run_dir  + "modeling_options_ccblade.yaml"
fname_modeling_options_openfast = run_dir + "modeling_options_openfast.yaml"
fname_analysis_options = run_dir + "analysis_options.yaml"
folder_output = run_dir + "it_0/"
fname_wt_output = folder_output + "temp.yaml"

# Run CCBlade
wt_opt_ccblade, analysis_options_ccblade, opt_options_ccblade = run_wisdem(
    fname_wt_input,
    fname_modeling_options_ccblade,
    fname_analysis_options)
    
# Run OpenFAST
wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_wisdem(
    fname_wt_input,
    fname_modeling_options_openfast,
    fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0

if rank == 0:
    print("Aerodynamic Cp CCblade    = " + str(wt_opt_ccblade["ccblade.CP"]))
    print("Aerodynamic Cp OpenFAST   = " + str(wt_opt_openfast["aeroelastic.Cp_out"]))
