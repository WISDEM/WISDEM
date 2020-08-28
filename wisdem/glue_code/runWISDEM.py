import numpy as np
import os, sys, time
import openmdao.api as om
from wisdem.glue_code.gc_LoadInputs   import WindTurbineOntologyPython
from wisdem.glue_code.gc_WT_InitModel import yaml2openmdao
from wisdem.glue_code.glue_code       import WindPark
from wisdem.commonse.mpi_tools        import MPI
from wisdem.commonse                  import fileIO
from wisdem.schema                    import load_yaml

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
    blade_opt_options = opt_options['optimization_variables']['blade']
    tower_opt_options = opt_options['optimization_variables']['tower']
    control_opt_options = opt_options['optimization_variables']['control']
    if MPI:
        # Determine the number of design variables
        n_DV = 0
        if blade_opt_options['aero_shape']['twist']['flag']:
            n_DV += blade_opt_options['aero_shape']['twist']['n_opt'] - 2
        if blade_opt_options['aero_shape']['chord']['flag']:    
            n_DV += blade_opt_options['aero_shape']['chord']['n_opt'] - 3            
        if blade_opt_options['aero_shape']['af_positions']['flag']:
            n_DV += modeling_options['blade']['n_af_span'] - blade_opt_options['aero_shape']['af_positions']['af_start'] - 1
        if blade_opt_options['structure']['spar_cap_ss']['flag']:
            n_DV += blade_opt_options['structure']['spar_cap_ss']['n_opt'] - 2
        if blade_opt_options['structure']['spar_cap_ps']['flag'] and not blade_opt_options['structure']['spar_cap_ps']['equal_to_suction']:
            n_DV += blade_opt_options['structure']['spar_cap_ps']['n_opt'] - 2
        if opt_options['optimization_variables']['control']['tsr']['flag']:
            n_DV += 1
        if opt_options['optimization_variables']['control']['servo']['pitch_control']['flag']:
            n_DV += 2
        if opt_options['optimization_variables']['control']['servo']['torque_control']['flag']:
            n_DV += 2
        if tower_opt_options['outer_diameter']['flag']:
            n_DV += modeling_options['tower']['n_height']
        if tower_opt_options['layer_thickness']['flag']:
            n_DV += (modeling_options['tower']['n_height'] - 1) * modeling_options['tower']['n_layers']
        
        if opt_options['driver']['form'] == 'central':
            n_DV *= 2
        
        # Extract the number of cores available
        max_cores = MPI.COMM_WORLD.Get_size()

        if max_cores / 2. != np.round(max_cores / 2.):
            exit('ERROR: the parallelization logic only works for an even number of cores available')

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
            # If a step size for the driver-level finite differencing is provided, use that step size. Otherwise use a default value.
            if 'step_size' in opt_options['driver']:
                step_size = opt_options['driver']['step_size']
            else:
                step_size = 1.e-6
            
            # Solver has specific meaning in OpenMDAO
            wt_opt.model.approx_totals(method='fd', step=step_size, form=opt_options['driver']['form'])
            
            # Set optimization solver and options. First, Scipy's SLSQP
            if opt_options['driver']['solver'] == 'SLSQP':
                wt_opt.driver  = om.ScipyOptimizeDriver()
                wt_opt.driver.options['optimizer'] = opt_options['driver']['solver']
                wt_opt.driver.options['tol']       = opt_options['driver']['tol']
                wt_opt.driver.options['maxiter']   = opt_options['driver']['max_iter']
                
            # The next two optimization methods require pyOptSparse.
            elif opt_options['driver']['solver'] == 'CONMIN':
                try:
                    from openmdao.api import pyOptSparseDriver
                except:
                    exit('You requested the optimization solver CONMIN, but you have not installed the pyOptSparseDriver. Please do so and rerun.')
                wt_opt.driver = pyOptSparseDriver()
                wt_opt.driver.options['optimizer'] = opt_options['driver']['solver']
                wt_opt.driver.opt_settings['ITMAX']= opt_options['driver']['max_iter']
                
            elif opt_options['driver']['solver'] == 'SNOPT':
                try:
                    from openmdao.api import pyOptSparseDriver
                except:
                    exit('You requested the optimization solver SNOPT, but you have not installed the pyOptSparseDriver. Please do so and rerun.')
                wt_opt.driver = pyOptSparseDriver()
                try:    
                    wt_opt.driver.options['optimizer']                       = opt_options['driver']['solver']
                except:
                    exit('You requested the optimization solver SNOPT, but you have not installed it within the pyOptSparseDriver. Please do so and rerun.')
                wt_opt.driver.opt_settings['Major optimality tolerance']  = float(opt_options['driver']['tol'])
                wt_opt.driver.opt_settings['Major iterations limit']      = int(opt_options['driver']['max_major_iter'])
                wt_opt.driver.opt_settings['Iterations limit']            = int(opt_options['driver']['max_minor_iter'])
                wt_opt.driver.opt_settings['Major feasibility tolerance'] = float(opt_options['driver']['tol'])
                wt_opt.driver.opt_settings['Summary file']                = os.path.join(folder_output, 'SNOPT_Summary_file.txt')
                wt_opt.driver.opt_settings['Print file']                  = os.path.join(folder_output, 'SNOPT_Print_file.txt')
                if 'hist_file_name' in opt_options['driver']:
                    wt_opt.driver.hist_file = opt_options['driver']['hist_file_name']
                if 'verify_level' in opt_options['driver']:
                    wt_opt.driver.opt_settings['Verify level'] = opt_options['driver']['verify_level']
                # wt_opt.driver.declare_coloring()  
                if 'hotstart_file' in opt_options['driver']:
                    wt_opt.driver.hotstart_file = opt_options['driver']['hotstart_file']

            else:
                exit('The optimizer ' + opt_options['driver']['solver'] + 'is not yet supported!')

            # Set merit figure. Each objective has its own scaling.
            if opt_options['merit_figure'] == 'AEP':
                wt_opt.model.add_objective('sse.AEP', ref = -1.e6)
            elif opt_options['merit_figure'] == 'blade_mass':
                wt_opt.model.add_objective('elastic.precomp.blade_mass', ref = 1.e4)
            elif opt_options['merit_figure'] == 'LCOE':
                wt_opt.model.add_objective('financese.lcoe', ref = 0.1)
            elif opt_options['merit_figure'] == 'blade_tip_deflection':
                wt_opt.model.add_objective('tcons.tip_deflection_ratio')
            elif opt_options['merit_figure'] == 'tower_mass':
                wt_opt.model.add_objective('towerse.tower_mass')
            elif opt_options['merit_figure'] == 'tower_cost':
                wt_opt.model.add_objective('tcc.tower_cost')
            elif opt_options['merit_figure'] == 'Cp':
                if modeling_options['Analysis_Flags']['ServoSE']:
                    wt_opt.model.add_objective('sse.powercurve.Cp_regII', ref = -1.)
                else:
                    wt_opt.model.add_objective('ccblade.CP', ref = -1.)
            else:
                exit('The merit figure ' + opt_options['merit_figure'] + ' is not supported.')

            # Set optimization design variables.
            
            if blade_opt_options['aero_shape']['twist']['flag']:
                indices        = range(2, blade_opt_options['aero_shape']['twist']['n_opt'])
                wt_opt.model.add_design_var('blade.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)
                
            chord_options = blade_opt_options['aero_shape']['chord']
            if chord_options['flag']:
                indices  = range(3, chord_options['n_opt'] - 1)
                wt_opt.model.add_design_var('blade.opt_var.chord_opt_gain', indices = indices, lower=chord_options['min_gain'], upper=chord_options['max_gain'])
                
            if blade_opt_options['aero_shape']['af_positions']['flag']:
                n_af = modeling_options['blade']['n_af_span']
                indices  = range(blade_opt_options['aero_shape']['af_positions']['af_start'],n_af - 1)
                af_pos_init = wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['grid']
                lb_af    = np.zeros(n_af)
                ub_af    = np.zeros(n_af)
                for i in range(1,indices[0]):
                    lb_af[i]    = ub_af[i] = af_pos_init[i]
                for i in indices:
                    lb_af[i]    = 0.5*(af_pos_init[i-1] + af_pos_init[i]) + step_size
                    ub_af[i]    = 0.5*(af_pos_init[i+1] + af_pos_init[i]) - step_size
                lb_af[-1] = ub_af[-1] = 1.
                wt_opt.model.add_design_var('blade.opt_var.af_position', indices = indices, lower=lb_af[indices], upper=ub_af[indices])
                
            spar_cap_ss_options = blade_opt_options['structure']['spar_cap_ss']
            if spar_cap_ss_options['flag']:
                indices  = range(1,spar_cap_ss_options['n_opt'] - 1)
                wt_opt.model.add_design_var('blade.opt_var.spar_cap_ss_opt_gain', indices = indices, lower=spar_cap_ss_options['min_gain'], upper=spar_cap_ss_options['max_gain'])
                
            # Only add the pressure side design variables if we do set
            # `equal_to_suction` as False in the optimization yaml.
            spar_cap_ps_options = blade_opt_options['structure']['spar_cap_ps']
            if spar_cap_ps_options['flag'] and not spar_cap_ps_options['equal_to_suction']:
                indices  = range(1, spar_cap_ps_options['n_opt'] - 1)
                wt_opt.model.add_design_var('blade.opt_var.spar_cap_ps_opt_gain', indices = indices, lower=spar_cap_ps_options['min_gain'], upper=spar_cap_ps_options['max_gain'])
                                
            if tower_opt_options['outer_diameter']['flag']:
                wt_opt.model.add_design_var('tower.diameter', lower=tower_opt_options['outer_diameter']['lower_bound'], upper=tower_opt_options['outer_diameter']['upper_bound'], ref=5.)
                
            if tower_opt_options['layer_thickness']['flag']:
                wt_opt.model.add_design_var('tower.layer_thickness', lower=tower_opt_options['layer_thickness']['lower_bound'], upper=tower_opt_options['layer_thickness']['upper_bound'], ref=1e-2)
            
            # -- Control -- 
            if control_opt_options['tsr']['flag']:
                wt_opt.model.add_design_var('opt_var.tsr_opt_gain', lower=control_opt_options['tsr']['min_gain'], 
                                                                    upper=control_opt_options['tsr']['max_gain'])
            if control_opt_options['servo']['pitch_control']['flag']:
                wt_opt.model.add_design_var('control.PC_omega', lower=control_opt_options['servo']['pitch_control']['omega_min'], 
                                                                upper=control_opt_options['servo']['pitch_control']['omega_max'])
                wt_opt.model.add_design_var('control.PC_zeta', lower=control_opt_options['servo']['pitch_control']['zeta_min'], 
                                                               upper=control_opt_options['servo']['pitch_control']['zeta_max'])
            if control_opt_options['servo']['torque_control']['flag']:
                wt_opt.model.add_design_var('control.VS_omega', lower=control_opt_options['servo']['torque_control']['omega_min'], 
                                                                upper=control_opt_options['servo']['torque_control']['omega_max'])
                wt_opt.model.add_design_var('control.VS_zeta', lower=control_opt_options['servo']['torque_control']['zeta_min'], 
                                                               upper=control_opt_options['servo']['torque_control']['zeta_max'])

            # Set non-linear constraints
            blade_constraints = opt_options['constraints']['blade']
            if blade_constraints['strains_spar_cap_ss']['flag']:
                if blade_opt_options['structure']['spar_cap_ss']['flag']:
                    wt_opt.model.add_constraint('rlds.constr.constr_max_strainU_spar', upper= 1.0)
                else:
                    print('WARNING: the strains of the suction-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced.')
                
            if blade_constraints['strains_spar_cap_ps']['flag']:
                if blade_opt_options['structure']['spar_cap_ps']['flag'] or blade_opt_options['structure']['spar_cap_ps']['equal_to_suction']:
                    wt_opt.model.add_constraint('rlds.constr.constr_max_strainL_spar', upper= 1.0)
                else:
                    print('WARNING: the strains of the pressure-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced.')
                
            if blade_constraints['stall']['flag']:
                if blade_opt_options['aero_shape']['twist']['flag']:
                    wt_opt.model.add_constraint('stall_check.no_stall_constraint', upper= 1.0) 
                else:
                    print('WARNING: the margin to stall is set to be constrained, but twist is not an active design variable. The constraint is not enforced.')

            if blade_constraints['tip_deflection']['flag']:
                if blade_opt_options['structure']['spar_cap_ss']['flag'] or blade_opt_options['structure']['spar_cap_ps']['flag']:
                    wt_opt.model.add_constraint('tcons.tip_deflection_ratio', upper= blade_constraints['tip_deflection']['ratio'])
                else:
                    print('WARNING: the tip deflection is set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')
                
            if blade_constraints['chord']['flag']:
                if blade_opt_options['aero_shape']['chord']['flag']:
                    wt_opt.model.add_constraint('blade.pa.max_chord_constr', upper= 1.0)
                else:
                    print('WARNING: the max chord is set to be constrained, but chord is not an active design variable. The constraint is not enforced.')
                
            if blade_constraints['frequency']['flap_above_3P']:
                if blade_opt_options['structure']['spar_cap_ss']['flag'] or blade_opt_options['structure']['spar_cap_ps']['flag']:
                    wt_opt.model.add_constraint('rlds.constr.constr_flap_f_margin', upper= 0.0)
                else:
                    print('WARNING: the blade flap frequencies are set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')
                
            if blade_constraints['frequency']['edge_above_3P']:
                wt_opt.model.add_constraint('rlds.constr.constr_edge_f_margin', upper= 0.0)
                
            if blade_constraints['rail_transport']['flag']:
                if blade_constraints['rail_transport']['8_axle']:
                    wt_opt.model.add_constraint('elastic.rail.constr_LV_8axle_horiz',   lower = 0.8, upper= 1.0)
                    wt_opt.model.add_constraint('elastic.rail.constr_strainPS',         upper= 1.0)
                    wt_opt.model.add_constraint('elastic.rail.constr_strainSS',         upper= 1.0)
                elif blade_constraints['rail_transport']['4_axle']:
                    wt_opt.model.add_constraint('elastic.rail.constr_LV_4axle_horiz', upper= 1.0)
                else:
                    exit('You have activated the rail transport constraint module. Please define whether you want to model 4- or 8-axle flatcars.')
                    
            if opt_options['constraints']['blade']['moment_coefficient']['flag']:
                wt_opt.model.add_constraint('ccblade.CM', lower= opt_options['constraints']['blade']['moment_coefficient']['min'], upper= opt_options['constraints']['blade']['moment_coefficient']['max'])
            if opt_options['constraints']['blade']['match_cl_cd']['flag_cl'] or opt_options['constraints']['blade']['match_cl_cd']['flag_cd']:
                data_target = np.loadtxt(opt_options['constraints']['blade']['match_cl_cd']['filename'])
                eta_opt     = np.linspace(0., 1., opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
                target_cl   = np.interp(eta_opt, data_target[:,0], data_target[:,3])
                target_cd   = np.interp(eta_opt, data_target[:,0], data_target[:,4])
                eps_cl = 1.e-2
                if opt_options['constraints']['blade']['match_cl_cd']['flag_cl']:
                    wt_opt.model.add_constraint('ccblade.cl_n_opt', lower = target_cl-eps_cl, upper = target_cl+eps_cl)
                if opt_options['constraints']['blade']['match_cl_cd']['flag_cd']:
                    wt_opt.model.add_constraint('ccblade.cd_n_opt', lower = target_cd-eps_cl, upper = target_cd+eps_cl)
            if opt_options['constraints']['blade']['match_L_D']['flag_L'] or opt_options['constraints']['blade']['match_L_D']['flag_D']:
                data_target = np.loadtxt(opt_options['constraints']['blade']['match_L_D']['filename'])
                eta_opt     = np.linspace(0., 1., opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
                target_L   = np.interp(eta_opt, data_target[:,0], data_target[:,7])
                target_D   = np.interp(eta_opt, data_target[:,0], data_target[:,8])
            eps_L  = 1.e+2
            if opt_options['constraints']['blade']['match_L_D']['flag_L']:
                wt_opt.model.add_constraint('ccblade.L_n_opt', lower = target_L-eps_L, upper = target_L+eps_L)
            if opt_options['constraints']['blade']['match_L_D']['flag_D']:
                wt_opt.model.add_constraint('ccblade.D_n_opt', lower = target_D-eps_L, upper = target_D+eps_L)

            tower_constraints = opt_options['constraints']['tower']
            if tower_constraints['height_constraint']['flag']:
                wt_opt.model.add_constraint('towerse.height_constraint',
                    lower=tower_constraints['height_constraint']['lower_bound'],
                    upper=tower_constraints['height_constraint']['upper_bound'])
                    
            if tower_constraints['stress']['flag']:
                wt_opt.model.add_constraint('towerse.post.stress', upper=1.0)
                
            if tower_constraints['global_buckling']['flag']:
                wt_opt.model.add_constraint('towerse.post.global_buckling', upper=1.0)
                
            if tower_constraints['shell_buckling']['flag']:
                wt_opt.model.add_constraint('towerse.post.shell_buckling', upper=1.0)
                
            if tower_constraints['constr_d_to_t']['flag']:
                wt_opt.model.add_constraint('towerse.constr_d_to_t', upper=0.0)
                
            if tower_constraints['constr_taper']['flag']:
                wt_opt.model.add_constraint('towerse.constr_taper', lower=0.0)
                
            if tower_constraints['slope']['flag']:
                wt_opt.model.add_constraint('towerse.slope', upper=1.0)
                
            if tower_constraints['frequency_1']['flag']:
                wt_opt.model.add_constraint('towerse.tower.f1',
                    lower=tower_constraints['frequency_1']['lower_bound'],
                    upper=tower_constraints['frequency_1']['upper_bound'])
            
            control_constraints = opt_options['constraints']['control']
            
            # Set recorder on the OpenMDAO driver level using the `optimization_log`
            # filename supplied in the optimization yaml
            if opt_options['recorder']['flag']:
                recorder = om.SqliteRecorder(os.path.join(folder_output, opt_options['recorder']['file_name']))
                wt_opt.driver.add_recorder(recorder)
                wt_opt.add_recorder(recorder)
                
                wt_opt.driver.recording_options['excludes'] = ['*_df']
                wt_opt.driver.recording_options['record_constraints'] = True 
                wt_opt.driver.recording_options['record_desvars'] = True 
                wt_opt.driver.recording_options['record_objectives'] = True
        
        # Setup openmdao problem
        wt_opt.setup()
        
        # Load initial wind turbine data from wt_initial to the openmdao problem
        wt_opt = yaml2openmdao(wt_opt, modeling_options, wt_init)
        wt_opt['blade.opt_var.s_opt_twist']   = np.linspace(0., 1., blade_opt_options['aero_shape']['twist']['n_opt'])
        if blade_opt_options['aero_shape']['twist']['flag']:
            init_twist_opt = np.interp(wt_opt['blade.opt_var.s_opt_twist'], wt_init['components']['blade']['outer_shape_bem']['twist']['grid'], wt_init['components']['blade']['outer_shape_bem']['twist']['values'])
            lb_twist = np.array(blade_opt_options['aero_shape']['twist']['lower_bound'])
            ub_twist = np.array(blade_opt_options['aero_shape']['twist']['upper_bound'])
            wt_opt['blade.opt_var.twist_opt_gain']    = (init_twist_opt - lb_twist) / (ub_twist - lb_twist)
            if max(wt_opt['blade.opt_var.twist_opt_gain']) > 1. or min(wt_opt['blade.opt_var.twist_opt_gain']) < 0.:
                print('Warning: the initial twist violates the upper or lower bounds of the twist design variables.')
                
        blade_constraints = opt_options['constraints']['blade']
        wt_opt['blade.opt_var.s_opt_chord']  = np.linspace(0., 1., blade_opt_options['aero_shape']['chord']['n_opt'])
        wt_opt['blade.ps.s_opt_spar_cap_ss'] = np.linspace(0., 1., blade_opt_options['structure']['spar_cap_ss']['n_opt'])
        wt_opt['blade.ps.s_opt_spar_cap_ps'] = np.linspace(0., 1., blade_opt_options['structure']['spar_cap_ps']['n_opt'])
        wt_opt['rlds.constr.max_strainU_spar'] = blade_constraints['strains_spar_cap_ss']['max']
        wt_opt['rlds.constr.max_strainL_spar'] = blade_constraints['strains_spar_cap_ps']['max']
        wt_opt['stall_check.stall_margin'] = blade_constraints['stall']['margin'] * 180. / np.pi
        
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
        if 'warmstart_file' in opt_options['driver']:
            
            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict
            db = SqliteDict(opt_options['driver']['warmstart_file'])

            # Grab the last iteration's design variables
            last_key = db['last']
            desvars = db[last_key]['xuser']
            
            # Obtain the already-setup OM problem's design variables
            if wt_opt.model._static_mode:
                design_vars = wt_opt.model._static_design_vars
            else:
                design_vars = wt_opt.model._design_vars
            
            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = wt_opt.model._var_allprocs_prom2abs_list['output']
            abs2prom = {}
            for key in design_vars:
                abs2prom[prom2abs[key][0]] = key

            # Loop through each design variable
            for key in desvars:
                prom_key = abs2prom[key]
                
                # Scale each DV based on the OM scaling from the problem.
                # This assumes we're running the same problem with the same scaling
                scaler = design_vars[prom_key]['scaler']
                adder = design_vars[prom_key]['adder']
                
                if scaler is None:
                    scaler = 1.0
                if adder is None:
                    adder = 0.0
                
                scaled_dv = desvars[key] / scaler - adder
                
                # Special handling for blade twist as we only have the
                # last few control points as design variables
                if 'twist_opt_gain' in key:
                    wt_opt[key][2:] = scaled_dv
                else:
                    wt_opt[key][:] = scaled_dv

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
    usg_msg = 'WISDEM command line launcher\n  Arguments: wisdem input.yaml'

    # Look for help message
    help_flag = len(sys.argv) == 1
    for k in range(len(sys.argv)):
        if sys.argv[k] in ['-h','--help']:
            help_flag = True

    if help_flag:
        print(usg_msg)
        sys.exit( 0 )

    # Warn for unparsed arguments
    if len(sys.argv) > 2:
        ignored = ''
        for k in range(2,len(sys.argv)): ignored += ' '+sys.argv[k]
        print('WARNING: The following arguments will be ignored,',ignored)
        print(usg_msg)

    # Grab master input file
    yaml_dict = read_master_file( sys.argv[1] )
    
    # Run WISDEM (also saves output)
    wt_opt, modeling_options, opt_options = run_wisdem(yaml_dict['geometry_file'],
                                                       yaml_dict['modeling_file'],
                                                       yaml_dict['analysis_file'])

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