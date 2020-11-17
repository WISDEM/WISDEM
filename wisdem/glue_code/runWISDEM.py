import numpy as np
import os, sys, time
import openmdao.api as om
from wisdem.glue_code.gc_LoadInputs   import WindTurbineOntologyPython
from wisdem.glue_code.gc_WT_InitModel import yaml2openmdao
from wisdem.glue_code.gc_PoseOptimization  import PoseOptimization
from wisdem.glue_code.glue_code       import WindPark
from wisdem.commonse.mpi_tools        import MPI
from wisdem.commonse                  import fileIO
from wisdem.yaml                      import load_yaml
from wisdem.yaml.gui                  import run as guirun

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

if MPI:
    #from openmdao.api import PetscImpl as impl
    #from mpi4py import MPI
    #from petsc4py import PETSc
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop

def run_wisdem(fname_wt_input, fname_modeling_options, fname_opt_options, overridden_values=None):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    # Initialize openmdao problem. If running with multiple processors in MPI, use parallel finite differencing equal to the number of cores used.
    # Otherwise, initialize the WindPark system normally. Get the rank number for parallelization. We only print output files using the root processor.
    myopt = PoseOptimization(modeling_options, opt_options)
    
    if MPI:

        n_DV = myopt.get_number_design_variables()
        
        # Extract the number of cores available
        max_cores = MPI.COMM_WORLD.Get_size()

        if max_cores / 2. != np.round(max_cores / 2.):
            raise ValueError('ERROR: the parallelization logic only works for an even number of cores available')

        # Define the color map for the parallelization, determining the maximum number of parallel finite difference (FD) evaluations based on the number of design variables (DV).
        n_FD = min([max_cores, n_DV])

        # Define the color map for the cores
        n_FD = max([n_FD, 1])
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(n_FD, 1)
        rank    = MPI.COMM_WORLD.Get_rank()
        color_i = color_map[rank]
        comm_i  = MPI.COMM_WORLD.Split(color_i, 1)
    else:
        color_i = 0
        rank = 0

    folder_output = opt_options['general']['folder_output']
    if rank == 0 and not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    if color_i == 0: # the top layer of cores enters
        if MPI:
            # Parallel settings for OpenMDAO
            wt_opt = om.Problem(model=om.Group(num_par_fd=n_FD), comm=comm_i)
            wt_opt.model.add_subsystem('comp', WindPark(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        else:
            # Sequential finite differencing
            wt_opt = om.Problem(model=WindPark(modeling_options = modeling_options, opt_options = opt_options))

        # If at least one of the design variables is active, setup an optimization
        if opt_options['opt_flag']:
            wt_opt = myopt.set_driver(wt_opt)
            wt_opt = myopt.set_objective(wt_opt)
            wt_opt = myopt.set_design_variables(wt_opt, wt_init)
            wt_opt = myopt.set_constraints(wt_opt)
            wt_opt = myopt.set_recorders(wt_opt)

        # Setup openmdao problem
        wt_opt.setup()

        # Load initial wind turbine data from wt_initial to the openmdao problem
        wt_opt = yaml2openmdao(wt_opt, modeling_options, wt_init)
        wt_opt = myopt.set_initial(wt_opt, wt_init)

        # If the user provides values in this dict, they overwrite
        # whatever values have been set by the yaml files.
        # This is useful for performing black-box wrapped optimization without
        # needing to modify the yaml files.
        if overridden_values is not None:
            for key in overridden_values:
                wt_opt[key][:] = overridden_values[key]

        # Place the last design variables from a previous run into the problem.
        # This needs to occur after the above setup() and yaml2openmdao() calls
        # so these values are correctly placed in the problem.
        wt_opt = myopt.set_restart(wt_opt)

        if 'check_totals' in opt_options['driver']:
            if opt_options['driver']['check_totals']:
                wt_opt.run_model()
                totals = wt_opt.compute_totals()

        if 'check_partials' in opt_options['driver']:
            if opt_options['driver']['check_partials']:
                wt_opt.run_model()
                checks = wt_opt.check_partials(compact_print=True)

        sys.stdout.flush()
        # Run openmdao problem
        if opt_options['opt_flag']:
            wt_opt.run_driver()
        else:
            wt_opt.run_model()

        if (not MPI) or (MPI and rank == 0):
            # Save data coming from openmdao to an output yaml file
            froot_out = os.path.join(folder_output, opt_options['general']['fname_output'])
            wt_initial.write_ontology(wt_opt, froot_out)

            # Save data to numpy and matlab arrays
            fileIO.save_data(froot_out, wt_opt)

    if rank == 0:
        return wt_opt, modeling_options, opt_options
    else:
        return [], [], []




def read_master_file( fyaml ):
    if os.path.exists(fyaml):
        print('...Reading master input file,',fyaml)
    else:
        raise FileNotFoundError('The master input file, '+fyaml+', cannot be found.')

    input_yaml = load_yaml(fyaml)

    check_list = ['geometry_file','modeling_file','analysis_file']
    for f in check_list:
        if not os.path.exists(input_yaml[f]):
            raise FileNotFoundError('The '+f+' entry, '+input_yaml[f]+', cannot be found.')

    return input_yaml



def wisdem_cmd():
    usg_msg = 'WISDEM command line launcher\n    Arguments: \n    wisdem : Starts GUI\n    wisdem input.yaml : Runs master yaml file that specifies geometry, modeling, and analysis files\n    wisdem geom.yaml modeling.yaml analysis.yaml : Runs specific geometry, modeling, and analysis files\n'

    # Look for help message
    help_flag = False
    for k in range(len(sys.argv)):
        if sys.argv[k] in ['-h','--help']:
            help_flag = True

    if help_flag:
        print(usg_msg)

    elif len(sys.argv) == 1:
        # Launch GUI
        guirun()

    elif len(sys.argv) == 2:
        # Grab master input file
        fyaml = sys.argv[1]
        if os.path.exists(fyaml):
            print('...Reading master input file,',fyaml)
        else:
            raise FileNotFoundError('The master input file, '+fyaml+', cannot be found.')
        yaml_dict = load_yaml( fyaml )

        check_list = ['geometry_file','modeling_file','analysis_file']
        for f in check_list:
            if not os.path.exists(yaml_dict[f]):
                raise FileNotFoundError('The '+f+' entry, '+yaml_dict[f]+', cannot be found.')

        # Run WISDEM (also saves output)
        wt_opt, modeling_options, opt_options = run_wisdem(yaml_dict['geometry_file'],
                                                           yaml_dict['modeling_file'],
                                                           yaml_dict['analysis_file'])

    elif len(sys.argv) == 4:
        check_list = ['geometry','modeling','analysis']
        for k, f in enumerate(sys.argv[1:]):
            if not os.path.exists(f):
                raise FileNotFoundError('The '+check_list[k]+' file, '+f+', cannot be found.')

        # Run WISDEM (also saves output)
        wt_opt, modeling_options, opt_options = run_wisdem(sys.argv[1], sys.argv[2], sys.argv[3])

    else:
        # As if asked for help
        print('Unrecognized set of inputs.  Usage:')
        print(usg_msg)

    sys.exit( 0 )




if __name__ == "__main__":

    ## File management
    run_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep + 'examples' + os.sep + 'reference_turbines_lcoe' + os.sep
    fname_wt_input         = run_dir + "IEA-15-240-RWT.yaml" #"reference_turbines/bar/BAR2010n.yaml"
    fname_modeling_options = run_dir + "modeling_options.yaml"
    fname_analysis_options = run_dir + "analysis_options.yaml"


    tt = time.time()
    wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    if rank == 0:
        print('Run time: %f'%(time.time()-tt))
        sys.stdout.flush()
