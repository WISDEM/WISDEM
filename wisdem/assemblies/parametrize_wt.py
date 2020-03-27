import numpy as np
import os
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from wisdem.rotorse.parametrize_rotor import ParametrizeBladeAero, ParametrizeBladeStruct

class WT_Parametrize(Group):
    # Openmdao group to parametrize the wind turbine based on the optimization variables
    
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')
        
    def setup(self):
        analysis_options = self.options['analysis_options']
        opt_options     = self.options['opt_options']
        
        # Optimization parameters initialized as indipendent variable component
        opt_var = IndepVarComp()
        opt_var.add_output('twist_opt_gain',   val = 0.5 * np.ones(opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt']))
        opt_var.add_output('chord_opt_gain',   val = np.ones(opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt']))
        opt_var.add_output('spar_cap_ss_opt_gain', val = np.ones(opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt']))
        opt_var.add_output('spar_cap_ps_opt_gain', val = np.ones(opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt']))
        
        self.add_subsystem('opt_var',opt_var)

        # Analysis components
        self.add_subsystem('pa',    ParametrizeBladeAero(blade_init_options = analysis_options['blade'], opt_options = opt_options)) # Parameterize aero (chord and twist)
        self.add_subsystem('ps',    ParametrizeBladeStruct(blade_init_options = analysis_options['blade'], opt_options = opt_options)) # Parameterize struct (spar caps ss and ps)

        # Connections to blade aero parametrization
        self.connect('opt_var.twist_opt_gain',    'pa.twist_opt_gain')
        self.connect('opt_var.chord_opt_gain',    'pa.chord_opt_gain')
        
        # Connections to blade struct parametrization
        self.connect('opt_var.spar_cap_ss_opt_gain','ps.spar_cap_ss_opt_gain')
        self.connect('opt_var.spar_cap_ps_opt_gain','ps.spar_cap_ps_opt_gain')
