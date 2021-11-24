import os
import sys
import logging

import numpy as np
import openmdao.api as om

from wisdem.commonse import fileIO
from wisdem.commonse.mpi_tools import MPI
from wisdem.glue_code.glue_code import WindPark
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from wisdem.glue_code.gc_WT_InitModel import yaml2openmdao
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

if MPI:
    # from openmdao.api import PetscImpl as impl
    # from mpi4py import MPI
    # from petsc4py import PETSc
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop


def run_wisdem(fname_wt_input, fname_modeling_options, fname_opt_options, overridden_values=None, run_only=False):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    # Initialize openmdao problem. If running with multiple processors in MPI, use parallel finite differencing equal to the number of cores used.
    # Otherwise, initialize the WindPark system normally. Get the rank number for parallelization. We only print output files using the root processor.
    myopt = PoseOptimization(wt_init, modeling_options, opt_options)

    if MPI:

        n_DV = myopt.get_number_design_variables()

        # Extract the number of cores available
        max_cores = MPI.COMM_WORLD.Get_size()

        if max_cores > n_DV:
            raise ValueError("ERROR: please reduce the number of cores, currently set to " +
             str(max_cores) + ", to the number of finite differences " + str(n_DV) +
             ", which is equal to the number of design variables DV for forward differencing" + 
             " and DV times 2 for central differencing," + 
              " or the parallelization logic will not work")

        # Define the color map for the parallelization, determining the maximum number of parallel finite difference (FD) evaluations based on the number of design variables (DV).
        n_FD = min([max_cores, n_DV])

        # Define the color map for the cores
        n_FD = max([n_FD, 1])
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(n_FD, 1)
        rank = MPI.COMM_WORLD.Get_rank()
        color_i = color_map[rank]
        comm_i = MPI.COMM_WORLD.Split(color_i, 1)
    else:
        color_i = 0
        rank = 0

    folder_output = opt_options["general"]["folder_output"]

    if rank == 0:
        os.makedirs(folder_output, exist_ok=True)

        # create logger
        logger = logging.getLogger("wisdem/weis")
        logger.setLevel(logging.INFO)

        # create handlers
        ht = logging.StreamHandler()
        ht.setLevel(logging.WARNING)

        flog = os.path.join(folder_output, opt_options["general"]["fname_output"] + ".log")
        hf = logging.FileHandler(flog, mode="w")
        hf.setLevel(logging.INFO)

        # create formatters
        formatter_t = logging.Formatter("%(module)s:%(funcName)s:%(lineno)d %(levelname)s:%(message)s")
        formatter_f = logging.Formatter(
            "P%(process)d %(asctime)s %(module)s:%(funcName)s:%(lineno)d %(levelname)s:%(message)s"
        )

        # add formatter to handlers
        ht.setFormatter(formatter_t)
        hf.setFormatter(formatter_f)

        # add handlers to logger
        logger.addHandler(ht)
        logger.addHandler(hf)
        logger.info("Started")

    if color_i == 0:  # the top layer of cores enters
        if MPI:
            # Parallel settings for OpenMDAO
            wt_opt = om.Problem(model=om.Group(num_par_fd=n_FD), comm=comm_i)
            wt_opt.model.add_subsystem(
                "comp", WindPark(modeling_options=modeling_options, opt_options=opt_options), promotes=["*"]
            )
        else:
            # Sequential finite differencing
            wt_opt = om.Problem(model=WindPark(modeling_options=modeling_options, opt_options=opt_options))

        # If at least one of the design variables is active, setup an optimization
        if opt_options["opt_flag"] and not run_only:
            wt_opt = myopt.set_driver(wt_opt)
            wt_opt = myopt.set_objective(wt_opt)
            wt_opt = myopt.set_design_variables(wt_opt, wt_init)
            wt_opt = myopt.set_constraints(wt_opt)
            wt_opt = myopt.set_recorders(wt_opt)

        # Setup openmdao problem
        wt_opt.setup()

        # Load initial wind turbine data from wt_initial to the openmdao problem
        wt_opt = yaml2openmdao(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)

        # If the user provides values in this dict, they overwrite
        # whatever values have been set by the yaml files.
        # This is useful for performing black-box wrapped optimization without
        # needing to modify the yaml files.
        if overridden_values is not None:
            for key in overridden_values:
                wt_opt[key] = overridden_values[key]

        # Place the last design variables from a previous run into the problem.
        # This needs to occur after the above setup() and yaml2openmdao() calls
        # so these values are correctly placed in the problem.
        wt_opt = myopt.set_restart(wt_opt)

        if "check_totals" in opt_options["driver"] and not run_only:
            if opt_options["driver"]["check_totals"]:
                wt_opt.run_model()
                totals = wt_opt.compute_totals()

        if "check_partials" in opt_options["driver"] and not run_only:
            if opt_options["driver"]["check_partials"]:
                wt_opt.run_model()
                checks = wt_opt.check_partials(compact_print=True)

        sys.stdout.flush()

        if opt_options["driver"]["step_size_study"]["flag"] and not run_only:
            wt_opt.run_model()
            study_options = opt_options["driver"]["step_size_study"]
            step_sizes = study_options["step_sizes"]
            all_derivs = {}
            for idx, step_size in enumerate(step_sizes):
                wt_opt.model.approx_totals(method="fd", step=step_size, form=study_options["form"])

                if study_options["of"]:
                    of = study_options["of"]
                else:
                    of = None

                if study_options["wrt"]:
                    wrt = study_options["wrt"]
                else:
                    wrt = None

                derivs = wt_opt.compute_totals(of=of, wrt=wrt, driver_scaling=study_options["driver_scaling"])
                all_derivs[idx] = derivs
                all_derivs[idx]["step_size"] = step_size
            np.save("total_derivs.npy", all_derivs)

        # Run openmdao problem
        elif opt_options["opt_flag"] and not run_only:
            wt_opt.run_driver()
        else:
            wt_opt.run_model()

        if (not MPI) or (MPI and rank == 0):
            # Save data coming from openmdao to an output yaml file
            froot_out = os.path.join(folder_output, opt_options["general"]["fname_output"])
            wt_initial.write_ontology(wt_opt, froot_out)
            wt_initial.write_options(froot_out)

            # Save data to numpy and matlab arrays
            fileIO.save_data(froot_out, wt_opt)

    if rank == 0:
        return wt_opt, modeling_options, opt_options
    else:
        return [], [], []


def load_wisdem(frootin):
    froot = os.path.splitext(frootin)[0]
    fgeom = froot + ".yaml"
    fmodel = froot + "-modeling.yaml"
    fopt = froot + "-analysis.yaml"
    fpkl = froot + ".pkl"

    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPython(fgeom, fmodel, fopt)
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    wt_opt = om.Problem(model=WindPark(modeling_options=modeling_options, opt_options=opt_options))
    wt_opt.setup()

    wt_opt = fileIO.load_data(fpkl, wt_opt)

    return wt_opt, modeling_options, opt_options
