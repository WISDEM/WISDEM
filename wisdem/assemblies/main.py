import numpy as np
import os
import matplotlib.pyplot as plt
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem, SqliteRecorder, ScipyOptimizeDriver, CaseReader
from wisdem.assemblies.load_IEA_yaml import WindTurbineOntologyPython, WindTurbineOntologyOpenMDAO, yaml2openmdao
from wisdem.assemblies.run_tools import Opt_Data, Convergence_Trends_Opt, Outputs_2_Screen
from wisdem.assemblies.wt_land_based import WindPark
from wisdem.commonse.mpi_tools import MPI


def run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output):
    # Main to run a wind turbine wisdem assembly
    
    # Get the rank number for parallelization
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    
    # Optimization options
    optimization_data       = Opt_Data()
    optimization_data.fname_opt_options = fname_opt_options
    optimization_data.folder_output     = folder_output    
    
    # Load yaml data into a pure python data structure
    wt_initial                   = WindTurbineOntologyPython()
    analysis_options, wt_init    = wt_initial.initialize(fname_wt_input, fname_analysis_options)
    
    opt_options = optimization_data.initialize()
    opt_options['opt_flag']    = False
    if opt_options['optimization_variables']['blade']['aero_shape']['twist']['flag'] == True:
        opt_options['opt_flag'] = True
    else:
        opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'] = analysis_options['rotorse']['n_span']
    if opt_options['optimization_variables']['blade']['aero_shape']['chord']['flag'] == True:
        opt_options['opt_flag'] = True
    else:
        opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt'] = analysis_options['rotorse']['n_span']
    if opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['flag'] == True:
        opt_options['opt_flag'] = True
    else:
        opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt'] = analysis_options['rotorse']['n_span']
    if opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['flag'] == True:
        opt_options['opt_flag'] = True
    else:
        opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt'] = analysis_options['rotorse']['n_span']
    if opt_options['optimization_variables']['control']['tsr']['flag'] == True:
        opt_options['opt_flag'] = True
    if 'dac' in opt_options['optimization_variables']['blade'].keys():
        if opt_options['optimization_variables']['blade']['dac']['te_flap_end']['flag'] == True or opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['flag'] == True:
            opt_options['opt_flag'] = True

    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    # Initialize openmdao problem
    if MPI:
        num_par_fd = MPI.COMM_WORLD.Get_size()
        wt_opt = Problem(model=Group(num_par_fd=num_par_fd))
        wt_opt.model.add_subsystem('comp', WindPark(analysis_options = analysis_options, opt_options = opt_options), promotes=['*'])
    else:
        wt_opt = Problem(model=WindPark(analysis_options = analysis_options, opt_options = opt_options))
    
    if 'step_size' in opt_options['driver']:
        step_size = opt_options['driver']['step_size']
    else:
        step_size = 1.e-6
    wt_opt.model.approx_totals(method='fd', step=step_size, form='central')    
    
    if opt_options['opt_flag'] == True:
        # Set optimization solver and options
        if opt_options['driver']['solver'] == 'SLSQP':
            wt_opt.driver  = ScipyOptimizeDriver()
            wt_opt.driver.options['optimizer'] = opt_options['driver']['solver']
            wt_opt.driver.options['tol']       = opt_options['driver']['tol']
            wt_opt.driver.options['maxiter']   = opt_options['driver']['max_iter']
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
            wt_opt.driver.opt_settings['Major optimality tolerance'] = float(opt_options['driver']['tol'])
            wt_opt.driver.opt_settings['Major iterations limit']           = int(opt_options['driver']['max_iter'])
            wt_opt.driver.opt_settings['Iterations limit']           = int(opt_options['driver']['max_function_calls'])
            wt_opt.driver.opt_settings['Major feasibility tolerance']= float(opt_options['driver']['tol'])
            # wt_opt.driver.opt_settings['Summary file'] = 'SNOPT_Summary_file.txt'
            # wt_opt.driver.opt_settings['Print file'] = 'SNOPT_Print_file.txt'
            # wt_opt.driver.opt_settings['Major step limit'] = opt_options['driver']['step_size']

        else:
            exit('The optimizer ' + opt_options['driver']['solver'] + 'is not yet supported!')

        # Set merit figure
        if opt_options['merit_figure'] == 'AEP':
            wt_opt.model.add_objective('sse.AEP', scaler = -1.e-6)
        elif opt_options['merit_figure'] == 'blade_mass':
            wt_opt.model.add_objective('elastic.precomp.blade_mass', scaler = 1.e-4)
        elif opt_options['merit_figure'] == 'LCOE':
            wt_opt.model.add_objective('financese.lcoe', scaler = 1.e+1)
        elif opt_options['merit_figure'] == 'blade_tip_deflection':
            wt_opt.model.add_objective('tcons.tip_deflection_ratio')
        elif opt_options['merit_figure'] == 'Cp':
            wt_opt.model.add_objective('sse.powercurve.Cp_cutin', scaler = -1.)
        else:
            exit('The merit figure ' + opt_options['merit_figure'] + ' is not supported.')

        # Set optimization variables
        if opt_options['optimization_variables']['blade']['aero_shape']['twist']['flag'] == True:
            indices        = range(2,opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
            wt_opt.model.add_design_var('blade.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)
        if opt_options['optimization_variables']['blade']['aero_shape']['chord']['flag'] == True:
            indices  = range(2,opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.chord_opt_gain', indices = indices, lower=opt_options['optimization_variables']['blade']['aero_shape']['chord']['min_gain'], upper=opt_options['optimization_variables']['blade']['aero_shape']['chord']['max_gain'])
        if opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['flag'] == True:
            indices  = range(1,opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.spar_cap_ss_opt_gain', indices = indices, lower=opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['min_gain'], upper=opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['max_gain'])
        if opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['flag'] == True and opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['equal_to_suction'] == False:
            indices  = range(1,opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.spar_cap_ps_opt_gain', indices = indices, lower=opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['min_gain'], upper=opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['max_gain'])
        if opt_options['optimization_variables']['control']['tsr']['flag'] == True:
            wt_opt.model.add_design_var('opt_var.tsr_opt_gain', lower=opt_options['optimization_variables']['control']['tsr']['min_gain'], upper=opt_options['optimization_variables']['control']['tsr']['max_gain'])
        if 'dac' in opt_options['optimization_variables']['blade'].keys():
            if opt_options['optimization_variables']['blade']['dac']['te_flap_end']['flag'] == True:
                wt_opt.model.add_design_var('blade.opt_var.te_flap_end', lower=opt_options['optimization_variables']['blade']['dac']['te_flap_end']['min_end'], upper=opt_options['optimization_variables']['blade']['dac']['te_flap_end']['max_end'])
            if opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['flag'] == True:
                wt_opt.model.add_design_var('blade.opt_var.te_flap_ext', lower=opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['min_ext'], upper=opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['max_ext'])
        

        # Set non-linear constraints
        if opt_options['constraints']['blade']['strains_spar_cap_ss']['flag']:
            wt_opt.model.add_constraint('rlds.constr.constr_max_strainU_spar', upper= 1.)
        if opt_options['constraints']['blade']['strains_spar_cap_ps']['flag']:
            wt_opt.model.add_constraint('rlds.constr.constr_max_strainL_spar', upper= 1.)
        if opt_options['constraints']['blade']['stall']['flag']:
            wt_opt.model.add_constraint('sse.stall_check.no_stall_constraint', upper= 1.) 
        if opt_options['constraints']['blade']['tip_deflection']['flag']:
            wt_opt.model.add_constraint('tcons.tip_deflection_ratio',    upper= 1.0) 
        if opt_options['constraints']['blade']['chord']['flag']:
            wt_opt.model.add_constraint('blade.pa.max_chord_constr',     upper= 1.0) 
        if opt_options['constraints']['blade']['rail_transport']['flag']:
            if opt_options['constraints']['blade']['rail_transport']['8_axle']:
                wt_opt.model.add_constraint('elastic.rail.LV_constraint_8axle',    upper= 1.0)
            elif opt_options['constraints']['blade']['rail_transport']['4_axle']:
                wt_opt.model.add_constraint('elastic.rail.LV_constraint_4axle',    upper= 1.0)
            else:
                exit('You have activated the rail transport constraint module. Please define whether you want to model 4- or 8-axle flatcars.')
        
        # Set recorder
        wt_opt.driver.add_recorder(SqliteRecorder(opt_options['optimization_log']))
        wt_opt.driver.recording_options['includes'] = ['sse.AEP, elastic.precomp.blade_mass, financese.lcoe', 'rlds.constr.constr_max_strainU_spar', 'rlds.constr.constr_max_strainL_spar', 'tcons.tip_deflection_ratio', 'sse.stall_check.no_stall_constraint', 'pc.tsr_opt']
        wt_opt.driver.recording_options['record_objectives']  = True
        wt_opt.driver.recording_options['record_constraints'] = True
        wt_opt.driver.recording_options['record_desvars']     = True
    
    # Setup openmdao problem
    wt_opt.setup()
    
    # Load initial wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, analysis_options, wt_init)
    wt_opt['blade.pa.s_opt_twist']   = np.linspace(0., 1., opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
    if opt_options['optimization_variables']['blade']['aero_shape']['twist']['flag'] == True:
        init_twist_opt = np.interp(wt_opt['blade.pa.s_opt_twist'], wt_init['components']['blade']['outer_shape_bem']['twist']['grid'], wt_init['components']['blade']['outer_shape_bem']['twist']['values'])
        lb_twist = np.array(opt_options['optimization_variables']['blade']['aero_shape']['twist']['lower_bound'])
        ub_twist = np.array(opt_options['optimization_variables']['blade']['aero_shape']['twist']['upper_bound'])
        wt_opt['blade.opt_var.twist_opt_gain']    = (init_twist_opt - lb_twist) / (ub_twist - lb_twist)
        if max(wt_opt['blade.opt_var.twist_opt_gain']) > 1. or min(wt_opt['blade.opt_var.twist_opt_gain']) < 0.:
            print('Warning: the initial twist violates the upper or lower bounds of the twist design variables.')
    wt_opt['blade.pa.s_opt_chord']       = np.linspace(0., 1., opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt'])
    wt_opt['blade.ps.s_opt_spar_cap_ss'] = np.linspace(0., 1., opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt'])
    wt_opt['blade.ps.s_opt_spar_cap_ps'] = np.linspace(0., 1., opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt'])
    wt_opt['rlds.constr.max_strainU_spar'] = opt_options['constraints']['blade']['strains_spar_cap_ss']['max']
    wt_opt['rlds.constr.max_strainL_spar'] = opt_options['constraints']['blade']['strains_spar_cap_ps']['max']
    wt_opt['sse.stall_check.stall_margin'] = opt_options['constraints']['blade']['stall']['margin'] * 180. / np.pi

    # Build and run openmdao problem
    wt_opt.run_driver()

    if rank == 0:
        # Save data coming from openmdao to an output yaml file
        wt_initial.write_ontology(wt_opt, fname_wt_output)

    return wt_opt, analysis_options, opt_options


if __name__ == "__main__":
    ## File management
    fname_wt_input         = "reference_turbines/nrel5mw/nrel5mw_mod_update.yaml" #"reference_turbines/bar/BAR2010n.yaml"
    fname_analysis_options = "reference_turbines/analysis_options.yaml"
    fname_opt_options      = "reference_turbines/optimization_options.yaml"
    fname_wt_output        = "reference_turbines/bar/BAR2010n_noRE_output.yaml"
    folder_output          = 'temp/'

    wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

        # Get the rank number for parallelization
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    if rank == 0:
        # Printing and plotting results
        print('AEP in GWh = ' + str(wt_opt['sse.AEP']*1.e-6))
        print('Nat frequencies blades in Hz = ' + str(wt_opt['elastic.curvefem.freq']))
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
