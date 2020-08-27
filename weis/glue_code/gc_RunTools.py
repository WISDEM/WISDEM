import os
import matplotlib.pyplot as plt
import openmdao.api as om
from wisdem.commonse.mpi_tools import MPI

class Outputs_2_Screen(om.ExplicitComponent):
    # Class to print outputs on screen
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        self.add_input('aep', val=0.0, units = 'GW * h')
        self.add_input('blade_mass', val=0.0, units = 'kg')
        self.add_input('lcoe', val=0.0, units = 'USD/MW/h')
        self.add_input('My_std', val=0.0, units = 'N*m')
        self.add_input('flp1_std', val=0.0, units = 'deg')
        self.add_input('PC_omega', val=0.0, units = 'rad/s')
        self.add_input('PC_zeta', val=0.0)
        self.add_input('VS_omega', val=0.0, units='rad/s')
        self.add_input('VS_zeta', val=0.0)
        self.add_input('Flp_omega', val=0.0, units='rad/s')
        self.add_input('Flp_zeta', val=0.0)
        self.add_input('tip_deflection', val=0.0, units='m')

    def compute(self, inputs, outputs):
        print('########################################')
        print('Objectives')
        print('Turbine AEP: {:8.10f} GWh'.format(inputs['aep'][0]))
        print('Blade Mass:  {:8.10f} kg'.format(inputs['blade_mass'][0]))
        print('LCOE:        {:8.10f} USD/MWh'.format(inputs['lcoe'][0]))
        print('Tip Defl.:   {:8.10f} m'.format(inputs['tip_deflection'][0]))
        if self.options['modeling_options']['Analysis_Flags']['OpenFAST'] == True: 
            if self.options['opt_options']['optimization_variables']['control']['servo']['pitch_control']['flag'] == True:
                print('Pitch PI gain inputs: pc_omega = {:2.3f}, pc_zeta = {:2.3f}'.format(inputs['PC_omega'][0], inputs['PC_zeta'][0]))
            if self.options['opt_options']['optimization_variables']['control']['servo']['torque_control']['flag'] == True:
                print('Torque PI gain inputs: vs_omega = {:2.3f}, vs_zeta = {:2.3f}'.format(inputs['VS_omega'][0], inputs['VS_zeta'][0]))
            if self.options['opt_options']['optimization_variables']['control']['servo']['flap_control']['flag'] == True:
                print('Flap PI gain inputs: flp_omega = {:2.3f}, flp_zeta = {:2.3f}'.format(inputs['Flp_omega'][0], inputs['Flp_zeta'][0]))
            if self.options['modeling_options']['airfoils']['n_tab'] > 1:
                print('Std(Myroot): {:8.10f} Nm'.format(inputs['My_std'][0]))
                print('Std(FLAP1):  {:8.10f} deg'.format(inputs['flp1_std'][0]))
        
        print('########################################')