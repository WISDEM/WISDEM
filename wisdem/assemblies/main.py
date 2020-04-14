import numpy as np
import os
import matplotlib.pyplot as plt
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem, SqliteRecorder, ScipyOptimizeDriver, CaseReader
from wisdem.assemblies.load_IEA_yaml import WindTurbineOntologyPython, WindTurbineOntologyOpenMDAO, yaml2openmdao
from wisdem.assemblies.run_tools import Opt_Data, Convergence_Trends_Opt, Outputs_2_Screen
from wisdem.assemblies.wt_land_based import WindPark
from wisdem.commonse.mpi_tools import MPI


def run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output):
    # Main function to run a wind turbine wisdem assembly
    
    # Get the rank number for parallelization. We only print output files
    # using the root processor.
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    
    # Create an Opt_Data instance, which is a container for the optimization
    # parameters as described in by the yaml file. Also set the folder output.
    optimization_data       = Opt_Data()
    optimization_data.fname_opt_options = fname_opt_options
    optimization_data.folder_output     = folder_output    
    
    # Load yaml for turbine description into a pure python data structure.
    wt_initial                   = WindTurbineOntologyPython()
    analysis_options, wt_init    = wt_initial.initialize(fname_wt_input, fname_analysis_options)
    
    opt_options = optimization_data.initialize()
    
    # Assume we're not doing optimization unless an opt_flag is true in the
    # optimization yaml
    opt_options['opt_flag']    = False
    
    # Loop through all blade optimization variables, setting opt_flag to true
    # if any of these variables are set to optimize. If it's not an optimization
    # DV, set the number of optimization points to be the same as the number
    # of discretization points. 
    blade_opt_options = opt_options['optimization_variables']['blade']
    
    if blade_opt_options['aero_shape']['twist']['flag']:
        opt_options['opt_flag'] = True
    else:
        blade_opt_options['aero_shape']['twist']['n_opt'] = analysis_options['rotorse']['n_span']
        
    if blade_opt_options['aero_shape']['chord']['flag']:
        opt_options['opt_flag'] = True
    else:
        blade_opt_options['aero_shape']['chord']['n_opt'] = analysis_options['rotorse']['n_span']
        
    if blade_opt_options['aero_shape']['af_positions']['flag']:
        opt_options['opt_flag'] = True
        
    if blade_opt_options['structure']['spar_cap_ss']['flag']:
        opt_options['opt_flag'] = True
    else:
        blade_opt_options['structure']['spar_cap_ss']['n_opt'] = analysis_options['rotorse']['n_span']
        
    if blade_opt_options['structure']['spar_cap_ps']['flag']:
        opt_options['opt_flag'] = True
    else:
        blade_opt_options['structure']['spar_cap_ps']['n_opt'] = analysis_options['rotorse']['n_span']
        
    if opt_options['optimization_variables']['control']['tsr']['flag']:
        opt_options['opt_flag'] = True
        
    # 'dac' is an optional setting, so we check if it's in the yaml-produced dict
    if 'dac' in blade_opt_options:
        if blade_opt_options['dac']['te_flap_end']['flag'] or blade_opt_options['dac']['te_flap_ext']['flag']:
            opt_options['opt_flag'] = True
            
    # Loop through all tower optimization variables, setting opt_flag to true
    # if any of these variables are set to optimize. If it's not an optimization
    # DV, set the number of optimization points to be the same as the number
    # of discretization points. 
    tower_opt_options = opt_options['optimization_variables']['tower']
    
    if tower_opt_options['outer_diameter']['flag']:
        opt_options['opt_flag'] = True
        
    if tower_opt_options['layer_thickness']['flag']:
        opt_options['opt_flag'] = True
    
        
    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    # Initialize openmdao problem. If running with multiple processors in MPI,
    # use parallel finite differencing equal to the number of cores used.
    # Otherwise, initialize the WindPark system normally.
    # with other OM problems?
    if MPI:
        num_par_fd = MPI.COMM_WORLD.Get_size()
        wt_opt = Problem(model=Group(num_par_fd=num_par_fd))
        wt_opt.model.add_subsystem('comp', WindPark(analysis_options = analysis_options, opt_options = opt_options), promotes=['*'])
    else:
        wt_opt = Problem(model=WindPark(analysis_options = analysis_options, opt_options = opt_options))
    
    # If a step size for the driver-level finite differencing is provided,
    # use that step size. Otherwise use a default value.
    if 'step_size' in opt_options['driver']:
        step_size = opt_options['driver']['step_size']
    else:
        step_size = 1.e-6
    wt_opt.model.approx_totals(method='fd', step=step_size, form='central')    
    
    # After looping through the optimization options yaml above, if opt_flag
    # became true then we set up an optimization problem
    # Solver has specific meaning in OpenMDAO
    if opt_options['opt_flag']:
        
        # Set optimization solver and options. First, Scipy's SLSQP
        if opt_options['driver']['solver'] == 'SLSQP':
            wt_opt.driver  = ScipyOptimizeDriver()
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
            wt_opt.driver.opt_settings['Major iterations limit']      = int(opt_options['driver']['max_iter'])
            wt_opt.driver.opt_settings['Iterations limit']            = int(opt_options['driver']['max_function_calls'])
            wt_opt.driver.opt_settings['Major feasibility tolerance'] = float(opt_options['driver']['tol'])
            # wt_opt.driver.opt_settings['Summary file'] = 'SNOPT_Summary_file.txt'
            # wt_opt.driver.opt_settings['Print file'] = 'SNOPT_Print_file.txt'
            # wt_opt.driver.opt_settings['Major step limit'] = opt_options['driver']['step_size']
            wt_opt.driver.hist_file = 'tower_opt.db'

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
            wt_opt.model.add_objective('sse.powercurve.Cp_regII', ref = -1.)
        elif opt_options['merit_figure'] == 'My_std':   # for DAC optimization on root-flap-bending moments
            wt_opt.model.add_objective('aeroelastic.My_std')  #1.e-8)
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
            indices  = range(2, chord_options['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.chord_opt_gain', indices = indices, lower=chord_options['min_gain'], upper=chord_options['max_gain'])
            
        if blade_opt_options['aero_shape']['af_positions']['flag']:
            n_af = analysis_options['blade']['n_af_span']
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
            
        if opt_options['optimization_variables']['control']['tsr']['flag']:
            wt_opt.model.add_design_var('opt_var.tsr_opt_gain', lower=opt_options['optimization_variables']['control']['tsr']['min_gain'], upper=opt_options['optimization_variables']['control']['tsr']['max_gain'])
            
        if 'dac' in blade_opt_options:
            if blade_opt_options['dac']['te_flap_end']['flag']:
                wt_opt.model.add_design_var('blade.opt_var.te_flap_end', lower=blade_opt_options['dac']['te_flap_end']['min_end'], upper=blade_opt_options['dac']['te_flap_end']['max_end'])
            if blade_opt_options['dac']['te_flap_ext']['flag']:
                wt_opt.model.add_design_var('blade.opt_var.te_flap_ext', lower=blade_opt_options['dac']['te_flap_ext']['min_ext'], upper=blade_opt_options['dac']['te_flap_ext']['max_ext'])
                
        if tower_opt_options['outer_diameter']['flag']:
            wt_opt.model.add_design_var('tower.diameter', lower=tower_opt_options['outer_diameter']['lower_bound'], upper=tower_opt_options['outer_diameter']['upper_bound'], ref=5.)
            
        if tower_opt_options['layer_thickness']['flag']:
            wt_opt.model.add_design_var('tower.layer_thickness', lower=tower_opt_options['layer_thickness']['lower_bound'], upper=tower_opt_options['layer_thickness']['upper_bound'], ref=1e-2)
        

        # Set non-linear constraints
        blade_constraints = opt_options['constraints']['blade']
        if blade_constraints['strains_spar_cap_ss']['flag']:
            wt_opt.model.add_constraint('rlds.constr.constr_max_strainU_spar', upper= 1.0)
            
        if blade_constraints['strains_spar_cap_ps']['flag']:
            wt_opt.model.add_constraint('rlds.constr.constr_max_strainL_spar', upper= 1.0)
            
        if blade_constraints['stall']['flag']:
            wt_opt.model.add_constraint('sse.stall_check.no_stall_constraint', upper= 1.0) 
            
        if blade_constraints['tip_deflection']['flag']:
            wt_opt.model.add_constraint('tcons.tip_deflection_ratio', upper= 1.0)
            
        if blade_constraints['chord']['flag']:
            wt_opt.model.add_constraint('blade.pa.max_chord_constr', upper= 1.0)
            
        if blade_constraints['frequency']['flap_above_3P']:
            wt_opt.model.add_constraint('rlds.constr.constr_flap_f_above_3P', upper= 1.0)
            
        if blade_constraints['frequency']['edge_above_3P']:
            wt_opt.model.add_constraint('rlds.constr.constr_edge_f_above_3P', upper= 1.0)
            
        if blade_constraints['frequency']['flap_below_3P']:
            wt_opt.model.add_constraint('rlds.constr.constr_flap_f_below_3P', upper= 1.0)
            
        if blade_constraints['frequency']['edge_below_3P']:
            wt_opt.model.add_constraint('rlds.constr.constr_edge_f_below_3P', upper= 1.0)
            
        if blade_constraints['frequency']['flap_above_3P'] and blade_constraints['frequency']['flap_below_3P']:
            exit('The blade flap frequency is constrained to be both above and below 3P. Please check the constraint flags.')
            
        if blade_constraints['frequency']['edge_above_3P'] and blade_constraints['frequency']['edge_below_3P']:
            exit('The blade edge frequency is constrained to be both above and below 3P. Please check the constraint flags.')
            
        if blade_constraints['rail_transport']['flag']:
            if blade_constraints['rail_transport']['8_axle']:
                wt_opt.model.add_constraint('elastic.rail.LV_constraint_8axle', upper= 1.0)
            elif blade_constraints['rail_transport']['4_axle']:
                wt_opt.model.add_constraint('elastic.rail.LV_constraint_4axle', upper= 1.0)
            else:
                exit('You have activated the rail transport constraint module. Please define whether you want to model 4- or 8-axle flatcars.')
                
                
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
            
        if tower_constraints['weldability']['flag']:
            wt_opt.model.add_constraint('towerse.weldability', upper=0.0)
            
        if tower_constraints['manufacturability']['flag']:
            wt_opt.model.add_constraint('towerse.manufacturability', lower=0.0)
            
        if tower_constraints['slope']['flag']:
            wt_opt.model.add_constraint('towerse.slope', upper=1.0)
            
        if tower_constraints['frequency_1']['flag']:
            wt_opt.model.add_constraint('towerse.tower.f1',
                lower=tower_constraints['frequency_1']['lower_bound'],
                upper=tower_constraints['frequency_1']['upper_bound'])
            
        
        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        wt_opt.driver.add_recorder(SqliteRecorder(opt_options['optimization_log']))
        wt_opt.driver.recording_options['includes'] = ['sse.AEP, elastic.precomp.blade_mass, financese.lcoe', 'rlds.constr.constr_max_strainU_spar', 'rlds.constr.constr_max_strainL_spar', 'tcons.tip_deflection_ratio', 'sse.stall_check.no_stall_constraint', 'pc.tsr_opt', ]
        wt_opt.driver.recording_options['record_objectives']  = True
        wt_opt.driver.recording_options['record_constraints'] = True
        wt_opt.driver.recording_options['record_desvars']     = True
    
    # Setup openmdao problem
    wt_opt.setup()
    
    # Load initial wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, analysis_options, wt_init)
    wt_opt['blade.pa.s_opt_twist']   = np.linspace(0., 1., blade_opt_options['aero_shape']['twist']['n_opt'])
    if blade_opt_options['aero_shape']['twist']['flag']:
        init_twist_opt = np.interp(wt_opt['blade.pa.s_opt_twist'], wt_init['components']['blade']['outer_shape_bem']['twist']['grid'], wt_init['components']['blade']['outer_shape_bem']['twist']['values'])
        lb_twist = np.array(blade_opt_options['aero_shape']['twist']['lower_bound'])
        ub_twist = np.array(blade_opt_options['aero_shape']['twist']['upper_bound'])
        wt_opt['blade.opt_var.twist_opt_gain']    = (init_twist_opt - lb_twist) / (ub_twist - lb_twist)
        if max(wt_opt['blade.opt_var.twist_opt_gain']) > 1. or min(wt_opt['blade.opt_var.twist_opt_gain']) < 0.:
            print('Warning: the initial twist violates the upper or lower bounds of the twist design variables.')
            
    blade_constraints = opt_options['constraints']['blade']
    wt_opt['blade.pa.s_opt_chord']       = np.linspace(0., 1., blade_opt_options['aero_shape']['chord']['n_opt'])
    wt_opt['blade.ps.s_opt_spar_cap_ss'] = np.linspace(0., 1., blade_opt_options['structure']['spar_cap_ss']['n_opt'])
    wt_opt['blade.ps.s_opt_spar_cap_ps'] = np.linspace(0., 1., blade_opt_options['structure']['spar_cap_ps']['n_opt'])
    wt_opt['rlds.constr.max_strainU_spar'] = blade_constraints['strains_spar_cap_ss']['max']
    wt_opt['rlds.constr.max_strainL_spar'] = blade_constraints['strains_spar_cap_ps']['max']
    wt_opt['sse.stall_check.stall_margin'] = blade_constraints['stall']['margin'] * 180. / np.pi

    if 'check_totals' in opt_options['driver']:
        if opt_options['driver']['check_totals']:
            wt_opt.run_model()
            totals = wt_opt.compute_totals()
    
    if 'check_partials' in opt_options['driver']:
        if opt_options['driver']['check_partials']:
            wt_opt.run_model()
            checks = wt_opt.check_partials(compact_print=True)
            
    # Run openmdao problem
    wt_opt.run_driver()

    if rank == 0:
        # Save data coming from openmdao to an output yaml file
        wt_initial.write_ontology(wt_opt, fname_wt_output)

    return wt_opt, analysis_options, opt_options


if __name__ == "__main__":
    ## File management
    
    run_dir = os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) + os.sep + 'assemblies' + os.sep + 'reference_turbines' + os.sep
    fname_wt_input         = run_dir + "nrel5mw/nrel5mw_mod_update.yaml" #"reference_turbines/bar/BAR2010n.yaml"
    fname_analysis_options = run_dir + "analysis_options.yaml"
    fname_opt_options      = run_dir + "optimization_options.yaml"
    fname_wt_output        = run_dir + "nrel5mw/nrel5mw_mod_update_output.yaml"
    folder_output          = run_dir + 'nrel5mw/'

    wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

        # Get the rank number for parallelization
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    if rank == 0:
        # Printing and plotting results
        print('AEP in GWh = ' + str(wt_opt['sse.AEP']*1.e-6))
        print('Nat frequencies blades in Hz = ' + str(wt_opt['sse.curvefem_rated.freq']))
        print('Tip tower clearance in m     = ' + str(wt_opt['tcons.blade_tip_tower_clearance']))
        print('Tip deflection constraint    = ' + str(wt_opt['tcons.tip_deflection_ratio']))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainU_spar'], label='spar ss')
        plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainL_spar'], label='spar ps')
        plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainU_te'], label='te ss')
        plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainL_te'], label='te ps')
        plt.ylim([-5e-3, 5e-3])
        plt.xlabel('r [m]')
        plt.ylabel('strain [-]')
        plt.legend()
        plt.show()
