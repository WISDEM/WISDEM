import ruamel_yaml as ry
import numpy as np
import os
import matplotlib.pyplot as plt
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem, SqliteRecorder, ScipyOptimizeDriver, CaseReader

class Opt_Data(object):
    # Pure python class to set the optimization parameters:

    def __init__(self):
        
        self.opt_options = {}

        # Save data
        self.fname_opt_options    = ''
        self.folder_output        = ''

    def initialize(self):

        self.opt_options = self.load_yaml(self.fname_opt_options)

        self.opt_options['folder_output']    = self.folder_output
        self.opt_options['optimization_log'] = self.folder_output + self.opt_options['recorder']['file_name']

        # # Optimization variables
        # self.opt_options['blade_aero'] = {}
        # self.opt_options['blade_aero']['opt_twist']         = opt_data['optimization_variables']['blade']['aero_shape']['twist']['flag']
        # self.opt_options['blade_aero']['n_opt_twist']       = opt_data['optimization_variables']['blade']['aero_shape']['twist']['n_opt']
        # self.opt_options['blade_aero']['lower_bound_twist'] = opt_data['optimization_variables']['blade']['aero_shape']['twist']['lower_bound']
        # self.opt_options['blade_aero']['upper_bound_twist'] = opt_data['optimization_variables']['blade']['aero_shape']['twist']['upper_bound']

        # self.opt_options['blade_aero']['opt_chord']         = opt_data['optimization_variables']['blade']['aero_shape']['chord']['flag']
        # self.opt_options['blade_aero']['n_opt_chord']       = opt_data['optimization_variables']['blade']['aero_shape']['chord']['n_opt']
        # self.opt_options['blade_aero']['min_gain_chord']    = opt_data['optimization_variables']['blade']['aero_shape']['chord']['min_gain']
        # self.opt_options['blade_aero']['max_gain_chord']    = opt_data['optimization_variables']['blade']['aero_shape']['chord']['max_gain']

        # self.opt_options['blade_struct'] = {}
        # self.opt_options['blade_struct']['opt_spar_cap_ss']         = opt_data['optimization_variables']['blade']['structure']['spar_cap_ss']['flag']
        # self.opt_options['blade_struct']['n_opt_spar_cap_ss']       = opt_data['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt']
        # self.opt_options['blade_struct']['spar_cap_ss_var']         = opt_data['optimization_variables']['blade']['structure']['spar_cap_ss']['name']
        # self.opt_options['blade_struct']['min_gain_spar_cap_ss']    = opt_data['optimization_variables']['blade']['structure']['spar_cap_ss']['min_gain']
        # self.opt_options['blade_struct']['max_gain_spar_cap_ss']    = opt_data['optimization_variables']['blade']['structure']['spar_cap_ss']['max_gain']
        
        # self.opt_options['blade_struct']['opt_spar_cap_ps']         = opt_data['optimization_variables']['blade']['structure']['spar_cap_ps']['flag']
        # self.opt_options['blade_struct']['n_opt_spar_cap_ps']       = opt_data['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt']
        # self.opt_options['blade_struct']['spar_cap_ps_var']         = opt_data['optimization_variables']['blade']['structure']['spar_cap_ps']['name']
        # self.opt_options['blade_struct']['min_gain_spar_cap_ps']    = opt_data['optimization_variables']['blade']['structure']['spar_cap_ps']['min_gain']
        # self.opt_options['blade_struct']['max_gain_spar_cap_ps']    = opt_data['optimization_variables']['blade']['structure']['spar_cap_ps']['max_gain']

        # self.opt_options['blade_struct']['te_ps_opt']         = opt_data['optimization_variables']['blade']['structure']['te_ps']['flag']
        # self.opt_options['blade_struct']['n_opt_te_ps']       = opt_data['optimization_variables']['blade']['structure']['te_ps']['n_opt']
        # self.opt_options['blade_struct']['te_ps_var']         = opt_data['optimization_variables']['blade']['structure']['te_ps']['name']
        # self.opt_options['blade_struct']['min_gain_te_ps']    = opt_data['optimization_variables']['blade']['structure']['te_ps']['min_gain']
        # self.opt_options['blade_struct']['max_gain_te_ps']    = opt_data['optimization_variables']['blade']['structure']['te_ps']['max_gain']

        # self.opt_options['blade_struct']['te_ss_opt']         = opt_data['optimization_variables']['blade']['structure']['te_ss']['flag']
        # self.opt_options['blade_struct']['n_opt_te_ss']       = opt_data['optimization_variables']['blade']['structure']['te_ss']['n_opt']
        # self.opt_options['blade_struct']['te_ss_var']         = opt_data['optimization_variables']['blade']['structure']['te_ss']['name']
        # self.opt_options['blade_struct']['min_gain_te_ss']    = opt_data['optimization_variables']['blade']['structure']['te_ss']['min_gain']
        # self.opt_options['blade_struct']['max_gain_te_ss']    = opt_data['optimization_variables']['blade']['structure']['te_ss']['max_gain']

        # if 'dac' in opt_data['optimization_variables']['blade'].keys():
        #     self.opt_options['dac']['te_ss_opt']         = opt_data['optimization_variables']['blade']['structure']['te_ss']['flag']
        #     self.opt_options['dac']['n_opt_te_ss']       = opt_data['optimization_variables']['blade']['structure']['te_ss']['n_opt']
        #     self.opt_options['dac']['te_ss_var']         = opt_data['optimization_variables']['blade']['structure']['te_ss']['name']
        #     self.opt_options['dac']['min_gain_te_ss']    = opt_data['optimization_variables']['blade']['structure']['te_ss']['min_gain']
        #     self.opt_options['dac']['max_gain_te_ss']    = opt_data['optimization_variables']['blade']['structure']['te_ss']['max_gain']

        # # Merit figure
        # self.opt_options['merit_figure']    = opt_data['merit_figure']

        # # Optimization driver options
        # self.opt_options['driver'] = {}
        # self.opt_options['driver']['solver']    = opt_data['driver']['solver']
        # self.opt_options['driver']['max_iter']  = opt_data['driver']['max_iter']
        # self.opt_options['driver']['tol']       = opt_data['driver']['tol']

        return self.opt_options

    def load_yaml(self, fname_input):
        """ Load optimization options """
        with open(fname_input, 'r') as myfile:
            inputs = myfile.read()
        yaml = ry.YAML()
        
        return yaml.load(inputs)

class Convergence_Trends_Opt(ExplicitComponent):
    def initialize(self):
        
        self.options.declare('opt_options')
        
    def compute(self, inputs, outputs):
        
        folder_output       = self.options['opt_options']['folder_output']
        optimization_log    = self.options['opt_options']['optimization_log']

        if os.path.exists(optimization_log):
        
            cr = CaseReader(optimization_log)
            cases = cr.list_cases()
            rec_data = {}
            iterations = []
            for i, casei in enumerate(cases):
                iterations.append(i)
                it_data = cr.get_case(casei)
                
                # parameters = it_data.get_responses()
                for parameters in [it_data.get_responses(), it_data.get_design_vars()]:
                    for j, param in enumerate(parameters.keys()):
                        if i == 0:
                            rec_data[param] = []
                        rec_data[param].append(parameters[param])

            for param in rec_data.keys():
                fig, ax = plt.subplots(1,1,figsize=(5.3, 4))
                ax.plot(iterations, rec_data[param])
                ax.set(xlabel='Number of Iterations' , ylabel=param)
                fig_name = 'Convergence_trend_' + param + '.png'
                fig.savefig(folder_output + fig_name)
                plt.close(fig)

class Outputs_2_Screen(ExplicitComponent):
    # Class to print outputs on screen
    def setup(self):
        
        self.add_input('aep', val=0.0, units = 'GW * h')
        self.add_input('blade_mass', val=0.0, units = 'kg')
        self.add_input('lcoe', val=0.0, units = 'USD/MW/h')
    def compute(self, inputs, outputs):
        print('########################################')
        print('Objectives')
        print('Turbine AEP: {:8.10f} GWh'.format(inputs['aep'][0]))
        print('Blade Mass:  {:8.10f} kg'.format(inputs['blade_mass'][0]))
        print('LCOE:        {:8.10f} USD/MWh'.format(inputs['lcoe'][0]))
        print('########################################')