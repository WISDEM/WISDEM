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
    if opt_options['optimization_variables']['blade']['aero_shape']['twist']['flag'] == True:
        opt_flag = True
    else:
        opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'] = analysis_options['rotorse']['n_span']
    if opt_options['optimization_variables']['blade']['aero_shape']['chord']['flag'] == True:
        opt_flag = True
    else:
        opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt'] = analysis_options['rotorse']['n_span']
    if opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['flag'] == True:
        opt_flag = True
    else:
        opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt'] = analysis_options['rotorse']['n_span']
    if opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['flag'] == True:
        opt_flag = True
    else:
        opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt'] = analysis_options['rotorse']['n_span']
    if 'dac' in opt_options['optimization_variables']['blade'].keys():
        if opt_options['optimization_variables']['blade']['dac']['te_flap_end']['flag'] == True or opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['flag'] == True:
            opt_flag = True

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
            wt_opt.model.add_objective('sse.AEP', scaler = -1.e-6)
        elif opt_options['merit_figure'] == 'blade_mass':
            wt_opt.model.add_objective('elastic.precomp.blade_mass', scaler = 1.e-4)
        elif opt_options['merit_figure'] == 'LCOE':
            wt_opt.model.add_objective('financese.lcoe', scaler = 1.e+2)
        else:
            exit('The merit figure ' + opt_options['merit_figure'] + ' is not supported.')
        
        # Set optimization variables
        if opt_options['optimization_variables']['blade']['aero_shape']['twist']['flag'] == True:
            indices        = range(2,opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
            wt_opt.model.add_design_var('param.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)
        if opt_options['optimization_variables']['blade']['aero_shape']['chord']['flag'] == True:
            indices  = range(2,opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt'] - 1)
            wt_opt.model.add_design_var('param.opt_var.chord_opt_gain', indices = indices, lower=opt_options['optimization_variables']['blade']['aero_shape']['chord']['min_gain'], upper=opt_options['optimization_variables']['blade']['aero_shape']['chord']['max_gain'])
        if opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['flag'] == True:
            indices  = range(2,opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt'] - 1)
            wt_opt.model.add_design_var('param.opt_var.spar_cap_ss_opt_gain', indices = indices, lower=opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['min_gain'], upper=opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['max_gain'])
        if opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['flag'] == True:
            indices  = range(2,opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt'] - 1)
            wt_opt.model.add_design_var('param.opt_var.spar_cap_ps_opt_gain', indices = indices, lower=opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['min_gain'], upper=opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['max_gain'])
        if 'dac' in opt_options['optimization_variables']['blade'].keys():
            if opt_options['optimization_variables']['blade']['dac']['te_flap_end']['flag'] == True:
                wt_opt.model.add_design_var('param.opt_var.te_flap_end', lower=opt_options['optimization_variables']['blade']['dac']['te_flap_end']['min_end'], upper=opt_options['optimization_variables']['blade']['dac']['te_flap_end']['max_end'])
            if opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['flag'] == True:
                wt_opt.model.add_design_var('param.opt_var.te_flap_ext', lower=opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['min_ext'], upper=opt_options['optimization_variables']['blade']['dac']['te_flap_ext']['max_ext'])
        

        # Set non-linear constraints
        wt_opt.model.add_constraint('rlds.constr.constr_max_strainU_spar', upper= 1.) 
        wt_opt.model.add_constraint('rlds.constr.constr_min_strainU_spar', upper= 1.) 
        wt_opt.model.add_constraint('rlds.constr.constr_max_strainL_spar', upper= 1.) 
        wt_opt.model.add_constraint('rlds.constr.constr_min_strainL_spar', upper= 1.) 
        wt_opt.model.add_constraint('sse.stall_check.no_stall_constraint', upper= 1.) 
        # wt_opt.model.add_constraint('tcons.tip_deflection_ratio',    upper= 1.0) 
        
        # Set recorder
        wt_opt.driver.add_recorder(SqliteRecorder(opt_options['optimization_log']))
        wt_opt.driver.recording_options['includes'] = ['sse.AEP, elastic.precomp.blade_mass, financese.lcoe']
        wt_opt.driver.recording_options['record_objectives']  = True
        wt_opt.driver.recording_options['record_constraints'] = True
        wt_opt.driver.recording_options['record_desvars']     = True
    
    # Setup openmdao problem
    wt_opt.setup()
    
    # Load initial wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, analysis_options, wt_init)
    wt_opt['param.pa.s_opt_twist']   = np.linspace(0., 1., opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
    wt_opt['param.pa.s_opt_chord']   = np.linspace(0., 1., opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt'])
    wt_opt['param.ps.s_opt_spar_cap_ss'] = np.linspace(0., 1., opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt'])
    wt_opt['param.ps.s_opt_spar_cap_ps'] = np.linspace(0., 1., opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt'])
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
    fname_wt_input         = "reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    fname_analysis_options = "reference_turbines/analysis_options.yaml"
    fname_opt_options      = "reference_turbines/optimization_options.yaml"
    fname_wt_output        = "reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    folder_output          = 'temp/'

    wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

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
