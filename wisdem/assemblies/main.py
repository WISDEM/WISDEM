import numpy as np
import os
import matplotlib.pyplot as plt
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem, SqliteRecorder, ScipyOptimizeDriver, CaseReader
from wisdem.assemblies.load_IEA_yaml import WindTurbineOntologyPython, WindTurbineOntologyOpenMDAO, yaml2openmdao
from wisdem.assemblies.run_tools import Opt_Data, Convergence_Trends_Opt, Outputs_2_Screen
from wisdem.assemblies.wt_land_based import WindPark

def run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output):
    # Main to run a wind turbine wisdem assembly
    
    # Optimization options
    optimization_data       = Opt_Data()
    optimization_data.fname_opt_options = fname_opt_options
    optimization_data.folder_output     = folder_output    
    
    # Load yaml data into a pure python data structure
    wt_initial                   = WindTurbineOntologyPython()
    analysis_options, wt_init    = wt_initial.initialize(fname_wt_input, fname_analysis_options)
    
    opt_options = optimization_data.initialize()
    opt_flag    = False
    if opt_options['blade_aero']['opt_twist'] == True:
        opt_flag = True
    else:
        opt_options['blade_aero']['n_opt_twist'] = analysis_options['rotorse']['n_span']
    if opt_options['blade_aero']['opt_chord'] == True:
        opt_flag = True
    else:
        opt_options['blade_aero']['n_opt_chord'] = analysis_options['rotorse']['n_span']
    if opt_options['blade_struct']['opt_spar_cap_ss'] == True:
        opt_flag = True
    else:
        opt_options['blade_aero']['n_opt_spar_cap_ss'] = analysis_options['rotorse']['n_span']
    if opt_options['blade_struct']['opt_spar_cap_ps'] == True:
        opt_flag = True
    else:
        opt_options['blade_aero']['n_opt_spar_cap_ps'] = analysis_options['rotorse']['n_span']

    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    # Initialize openmdao problem
    wt_opt          = Problem()
    wt_opt.model    = WindPark(analysis_options = analysis_options, opt_options = opt_options)
    wt_opt.model.approx_totals(method='fd')
    
    if opt_flag == True:
        # Set optimization solver and options
        wt_opt.driver  = ScipyOptimizeDriver()
        wt_opt.driver.options['optimizer'] = opt_options['driver']['solver']
        wt_opt.driver.options['tol']       = opt_options['driver']['tol']
        wt_opt.driver.options['maxiter']   = opt_options['driver']['max_iter']

        # Set merit figure
        if opt_options['merit_figure'] == 'AEP':
            wt_opt.model.add_objective('ra.AEP', scaler = -1.e-6)
        elif opt_options['merit_figure'] == 'blade_mass':
            wt_opt.model.add_objective('rlds.blade_mass', scaler = 1.e-4)
        elif opt_options['merit_figure'] == 'LCOE':
            wt_opt.model.add_objective('financese.lcoe', scaler = 1.e+2)
        else:
            exit('The merit figure ' + opt_options['merit_figure'] + ' is not supported.')
        
        # Set optimization variables
        if opt_options['blade_aero']['opt_twist'] == True:
            indices        = range(2,opt_options['blade_aero']['n_opt_twist'])
            wt_opt.model.add_design_var('param.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)
        if opt_options['blade_aero']['opt_chord'] == True:
            indices  = range(2,opt_options['blade_aero']['n_opt_chord'] - 1)
            wt_opt.model.add_design_var('param.opt_var.chord_opt_gain', indices = indices, lower=opt_options['blade_aero']['min_gain_chord'], upper=opt_options['blade_aero']['max_gain_chord'])
        if opt_options['blade_struct']['opt_spar_cap_ss'] == True:
            indices  = range(2,opt_options['blade_struct']['n_opt_spar_ss'] - 1)
            wt_opt.model.add_design_var('param.opt_var.spar_ss_opt_gain', indices = indices, lower=opt_options['blade_struct']['min_gain_spar_cap_ss'], upper=opt_options['blade_struct']['max_gain_spar_cap_ss'])
        if opt_options['blade_struct']['opt_spar_cap_ps'] == True:
            indices  = range(2,opt_options['blade_struct']['n_opt_spar_ps'] - 1)
            wt_opt.model.add_design_var('param.opt_var.spar_ps_opt_gain', indices = indices, lower=opt_options['blade_struct']['min_gain_spar_cap_ps'], upper=opt_options['blade_struct']['max_gain_spar_cap_ps'])

        # Set non-linear constraints
        wt_opt.model.add_constraint('rlds.pbeam.strainU_spar', upper= 1.) 
        wt_opt.model.add_constraint('rlds.pbeam.strainL_spar', upper= 1.) 
        wt_opt.model.add_constraint('tcons.tip_deflection_ratio',    upper= 1.0) 
        
        # Set recorder
        wt_opt.driver.add_recorder(SqliteRecorder(opt_options['optimization_log']))
        wt_opt.driver.recording_options['includes'] = ['ra.AEP, rlds.blade_mass, financese.lcoe']
        wt_opt.driver.recording_options['record_objectives']  = True
        wt_opt.driver.recording_options['record_constraints'] = True
        wt_opt.driver.recording_options['record_desvars']     = True
    
    # Setup openmdao problem
    wt_opt.setup()
    
    # Load initial wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, analysis_options, wt_init)
    wt_opt['param.pa.s_opt_twist']   = np.linspace(0., 1., opt_options['blade_aero']['n_opt_twist'])
    wt_opt['param.pa.s_opt_chord']   = np.linspace(0., 1., opt_options['blade_aero']['n_opt_chord'])
    wt_opt['param.ps.s_opt_spar_cap_ss'] = np.linspace(0., 1., opt_options['blade_struct']['n_opt_spar_cap_ss'])
    wt_opt['param.ps.s_opt_spar_cap_ps'] = np.linspace(0., 1., opt_options['blade_struct']['n_opt_spar_cap_ps'])
    wt_opt['rlds.constr.min_strainU_spar'] = -0.003
    wt_opt['rlds.constr.max_strainU_spar'] =  0.003
    wt_opt['rlds.constr.min_strainL_spar'] = -0.003
    wt_opt['rlds.constr.max_strainL_spar'] =  0.003

    # Build and run openmdao problem
    wt_opt.run_driver()

    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_wt_output)

    return wt_opt, analysis_options, opt_options


if __name__ == "__main__":
    ## File management
    fname_wt_input         = "wisdem/wisdem/assemblies/reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    fname_analysis_options = "wisdem/wisdem/assemblies/reference_turbines/analysis_options.yaml"
    fname_opt_options      = "wisdem/wisdem/assemblies/reference_turbines/optimization_options.yaml"
    fname_wt_output        = "wisdem/wisdem/assemblies/reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    folder_output          = 'temp/'

    wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

    # Printing and plotting results
    print('AEP in GWh = ' + str(wt_opt['ra.AEP']*1.e-6))
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