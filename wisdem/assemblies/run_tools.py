try:
    import ruamel_yaml as ry
except:
    try:
        import ruamel.yaml as ry
    except:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')
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
        self.add_input('Mxyz', val=0.0, units = 'N*m')
        self.add_input('My_std', val=0.0, units = 'N*m')
        self.add_input('flp1_std', val=0.0, units = 'deg')

    def compute(self, inputs, outputs):
        print('########################################')
        print('Objectives')
        print('Turbine AEP: {:8.10f} GWh'.format(inputs['aep'][0]))
        print('Blade Mass:  {:8.10f} kg'.format(inputs['blade_mass'][0]))
        print('LCOE:        {:8.10f} USD/MWh'.format(inputs['lcoe'][0]))
        print('Std(Myroot): {:8.10f} Nm'.format(inputs['My_std'][0]))
        print('Std(FLAP1):  {:8.10f} deg'.format(inputs['flp1_std'][0]))
        print('########################################')
