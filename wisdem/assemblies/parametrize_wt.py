import numpy as np
import os
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from wisdem.rotorse.wt_rotor import ParametrizeBladeAero, ParametrizeBladeStruct

class WT_Parametrize(Group):
    # Openmdao group to parametrize the wind turbine based on the optimization variables
    
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        opt_options     = self.options['opt_options']
        
        # Optimization parameters initialized as indipendent variable component
        opt_var = IndepVarComp()
        opt_var.add_output('twist_opt_gain',   val = 0.5 * np.ones(opt_options['blade_aero']['n_opt_twist']))
        opt_var.add_output('chord_opt_gain',   val = np.ones(opt_options['blade_aero']['n_opt_chord']))
        opt_var.add_output('spar_ss_opt_gain', val = np.ones(opt_options['blade_struct']['n_opt_spar_ss']))
        opt_var.add_output('spar_ps_opt_gain', val = np.ones(opt_options['blade_struct']['n_opt_spar_ps']))
        self.add_subsystem('opt_var',opt_var)

        # Analysis components
        self.add_subsystem('pa',    ParametrizeBladeAero(blade_init_options = wt_init_options['blade'], opt_options = opt_options)) # Parameterize aero (chord and twist)
        self.add_subsystem('ps',    ParametrizeBladeStruct(blade_init_options = wt_init_options['blade'], opt_options = opt_options)) # Parameterize struct (spar caps ss and ps)

        # Connections to blade aero parametrization
        self.connect('opt_var.twist_opt_gain',    'pa.twist_opt_gain')
        self.connect('opt_var.chord_opt_gain',    'pa.chord_opt_gain')
        
        # Connections to blade struct parametrization
        self.connect('opt_var.spar_ss_opt_gain','ps.spar_ss_opt_gain')
        self.connect('opt_var.spar_ps_opt_gain','ps.spar_ps_opt_gain')
