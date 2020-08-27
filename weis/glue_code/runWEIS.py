import numpy as np
import os, sys, time
import openmdao.api as om
from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel import yaml2openmdao
from weis.glue_code.glue_code         import WindPark
from wisdem.commonse.mpi_tools        import MPI
from wisdem.commonse                  import fileIO
from wisdem.schema                    import load_yaml
from wisdem.glue_code.runWISDEM       import read_master_file, wisdem_cmd
from weis.glue_code.gc_ROSCOInputs    import assign_ROSCO_values

if MPI:
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop

def run_weis(fname_wt_input, fname_modeling_options, fname_opt_options):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_opt_options)
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
        if opt_options['optimization_variables']['control']['servo']['flap_control']['flag']:
            n_DV += 2
        if 'dac' in blade_opt_options:
            if blade_opt_options['dac']['te_flap_end']['flag']:
                n_DV += modeling_options['blade']['n_te_flaps']
            if blade_opt_options['dac']['te_flap_ext']['flag']:
                n_DV += modeling_options['blade']['n_te_flaps']
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

        # Define the color map for the parallelization, determining the maximum number of parallel finite difference (FD) evaluations based on the number of design variables (DV). OpenFAST on/off changes things.
        if modeling_options['Analysis_Flags']['OpenFAST']:
            # If openfast is called, the maximum number of FD is the number of DV, if we have the number of cores available that doubles the number of DVs, otherwise it is half of the number of DV (rounded to the lower integer). We need this because a top layer of cores calls a bottom set of cores where OpenFAST runs.
            if max_cores > 2. * n_DV:
                n_FD = n_DV
            else:
                n_FD = int(np.floor(max_cores / 2))
            # The number of OpenFAST runs is the minimum between the actual number of requested OpenFAST simulations, and the number of cores available (minus the number of DV, which sit and wait for OF to complete)
            
            # need to calculate the number of OpenFAST runs from the user input
            n_OF_runs = 0
            if modeling_options['openfast']['dlc_settings']['run_power_curve']:
                if modeling_options['openfast']['dlc_settings']['Power_Curve']['turbulent_power_curve']:
                    n_OF_runs += len(modeling_options['openfast']['dlc_settings']['Power_Curve']['U'])*len(modeling_options['openfast']['dlc_settings']['Power_Curve']['Seeds'])
                else:
                    n_OF_runs += len(modeling_options['openfast']['dlc_settings']['Power_Curve']['U'])
            if modeling_options['openfast']['dlc_settings']['run_IEC'] or modeling_options['openfast']['dlc_settings']['run_blade_fatigue']:
                for dlc in modeling_options['openfast']['dlc_settings']['IEC']:
                    dlc_vars = list(dlc.keys())
                    # Number of wind speeds
                    if 'U' not in dlc_vars:
                        if dlc['DLC'] == 1.4: # assuming 1.4 is run at [V_rated-2, V_rated, V_rated] and +/- direction change
                            n_U = 6
                        elif dlc['DLC'] == 5.1: # assuming 1.4 is run at [V_rated-2, V_rated, V_rated]
                            n_U = 3
                        elif dlc['DLC'] in [6.1, 6.3]: # assuming V_50 for [-8, 8] deg yaw error
                            n_U = 2
                        else:
                            print('Warning: for OpenFAST DLC %1.1f specified in the Analysis Options, wind speeds "U" must be provided'%dlc['DLC'])
                    else:
                        n_U = len(dlc['U'])
                    # Number of seeds
                    if 'Seeds' not in dlc_vars:
                        if dlc['DLC'] == 1.4: # not turbulent
                            n_Seeds = 1
                        else:
                            print('Warning: for OpenFAST DLC %1.1f specified in the Analysis Options, turbulent seeds "Seeds" must be provided'%dlc['DLC'])
                    else:
                        n_Seeds = len(dlc['Seeds'])

                    n_OF_runs += n_U*n_Seeds

            n_DV = max([n_DV, 1])
            max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
            n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])

            modeling_options['openfast']['dlc_settings']['n_OF_runs'] = n_OF_runs
        else:
            # If OpenFAST is not called, the number of parallel calls to compute the FDs is just equal to the minimum of cores available and DV
            n_FD = min([max_cores, n_DV])
            n_OF_runs_parallel = 1
        
        # Define the color map for the cores (how these are distributed between finite differencing and openfast runs)
        n_FD = max([n_FD, 1])
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(n_FD, n_OF_runs_parallel)
        rank    = MPI.COMM_WORLD.Get_rank()
        color_i = color_map[rank]
        comm_i  = MPI.COMM_WORLD.Split(color_i, 1)
    else:
        color_i = 0
        rank = 0

    folder_output = opt_options['general']['folder_output']
    if rank == 0 and not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    if color_i == 0: # the top layer of cores enters, the others sit and wait to run openfast simulations
        if MPI:
            if modeling_options['Analysis_Flags']['OpenFAST']:
                # Parallel settings for OpenFAST
                modeling_options['openfast']['analysis_settings']['mpi_run']           = True
                modeling_options['openfast']['analysis_settings']['mpi_comm_map_down'] = comm_map_down
                modeling_options['openfast']['analysis_settings']['cores']             = n_OF_runs_parallel            
            # Parallel settings for OpenMDAO
            wt_opt = om.Problem(model=om.Group(num_par_fd=n_FD), comm=comm_i)
            wt_opt.model.add_subsystem('comp', WindPark(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        else:
            # Sequential finite differencing and openfast simulations
            modeling_options['openfast']['analysis_settings']['cores'] = 1
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
                    exit('You requested the optimization solver SNOPT which requires pyOptSparse to be installed, but it cannot be found. Please install pyOptSparse and rerun.')
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
            elif opt_options['merit_figure'] == 'My_std':   # for DAC optimization on root-flap-bending moments
                wt_opt.model.add_objective('aeroelastic.My_std', ref = 1.e6)
            elif opt_options['merit_figure'] == 'DEL_RootMyb':   # for DAC optimization on root-flap-bending moments
                wt_opt.model.add_objective('aeroelastic.DEL_RootMyb', ref = 1.e5)
            elif opt_options['merit_figure'] == 'flp1_std':   # for DAC optimization on flap angles - TORQUE 2020 paper (need to define time constant in ROSCO)
                wt_opt.model.add_objective('aeroelastic.flp1_std')  #1.e-8)
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
                                
            if 'dac' in blade_opt_options:
                if blade_opt_options['dac']['te_flap_end']['flag']:
                    wt_opt.model.add_design_var('blade.opt_var.te_flap_end', lower=blade_opt_options['dac']['te_flap_end']['min_end'], upper=blade_opt_options['dac']['te_flap_end']['max_end'])
                if blade_opt_options['dac']['te_flap_ext']['flag']:
                    wt_opt.model.add_design_var('blade.opt_var.te_flap_ext', lower=blade_opt_options['dac']['te_flap_ext']['min_ext'], upper=blade_opt_options['dac']['te_flap_ext']['max_ext'])
                    
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
            if 'flap_control' in control_opt_options['servo']:
                if control_opt_options['servo']['flap_control']['flag']:
                    wt_opt.model.add_design_var('control.Flp_omega', lower=control_opt_options['servo']['flap_control']['omega_min'], 
                                                                     upper=control_opt_options['servo']['flap_control']['omega_max'])
                    wt_opt.model.add_design_var('control.Flp_zeta', lower=control_opt_options['servo']['flap_control']['zeta_min'], 
                                                                    upper=control_opt_options['servo']['flap_control']['zeta_max'])

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
                    wt_opt.model.add_constraint('tcons.tip_deflection_ratio', upper= 1.0)
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
                
            # if blade_constraints['frequency']['flap_below_3P']:
            #     wt_opt.model.add_constraint('rlds.constr.constr_flap_f_below_3P', upper= 1.0)
                
            # if blade_constraints['frequency']['edge_below_3P']:
            #     wt_opt.model.add_constraint('rlds.constr.constr_edge_f_below_3P', upper= 1.0)
                
            # if blade_constraints['frequency']['flap_above_3P'] and blade_constraints['frequency']['flap_below_3P']:
            #     exit('The blade flap frequency is constrained to be both above and below 3P. Please check the constraint flags.')
                
            # if blade_constraints['frequency']['edge_above_3P'] and blade_constraints['frequency']['edge_below_3P']:
            #     exit('The blade edge frequency is constrained to be both above and below 3P. Please check the constraint flags.')
                
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
            if control_constraints['flap_control']['flag']:
                if modeling_options['Analysis_Flags']['OpenFAST'] != True:
                    exit('Please turn on the call to OpenFAST if you are trying to optimize trailing edge flaps.')
                wt_opt.model.add_constraint('sse_tune.tune_rosco.Flp_Kp',
                    lower = control_constraints['flap_control']['min'],
                    upper = control_constraints['flap_control']['max'])
                wt_opt.model.add_constraint('sse_tune.tune_rosco.Flp_Ki', 
                    lower = control_constraints['flap_control']['min'],
                    upper = control_constraints['flap_control']['max'])    
            
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
        if opt_options['opt_flag']:
            wt_opt.setup()
        else:
            # If we're not performing optimization, we don't need to allocate
            # memory for the derivative arrays.
            wt_opt.setup(derivatives=False)
        
        # Load initial wind turbine data from wt_initial to the openmdao problem
        wt_opt = yaml2openmdao(wt_opt, modeling_options, wt_init)
        wt_opt = assign_ROSCO_values(wt_opt, modeling_options, wt_init['control'])
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
            wt_initial.update_ontology_control(wt_opt)
            wt_initial.write_ontology(wt_opt, froot_out)
            
            # Save data to numpy and matlab arrays
            fileIO.save_data(froot_out, wt_opt)

    if MPI and modeling_options['Analysis_Flags']['OpenFAST']:
        # subprocessor ranks spin, waiting for FAST simulations to run
        sys.stdout.flush()
        if rank in comm_map_up.keys():
            subprocessor_loop(comm_map_up)
        sys.stdout.flush()

        # close signal to subprocessors
        if rank == 0:
            subprocessor_stop(comm_map_down)
        sys.stdout.flush()

        
    if rank == 0:
        return wt_opt, modeling_options, opt_options
    else:
        return [], [], []

if __name__ == "__main__":

    ## File management
    run_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep + 'examples' + os.sep + 'rotor_opt' + os.sep
    fname_wt_input         = run_dir + "IEA-15-240-RWT.yaml"
    fname_modeling_options = run_dir + "modeling_options.yaml"
    fname_analysis_options = run_dir + "analysis_options.yaml"


    tt = time.time()
    wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)
    
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    if rank == 0:
        print('Run time: %f'%(time.time()-tt))
        sys.stdout.flush()

        # print(wt_opt['aeroelastic.C_miners_SC_PS'])
