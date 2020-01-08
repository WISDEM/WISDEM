import numpy as np
import os
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from wisdem.rotorse.rotor_aeropower import RotorAeroPower
from wisdem.rotorse.rotor_structure_simple import RotorStructure

class ParametrizeBladeAero(ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the aerodynamic only analysis of the wind turbine rotor
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        blade_init_options = self.options['blade_init_options']
        opt_options        = self.options['opt_options']
        n_span             = blade_init_options['n_span']
        self.n_opt_twist   = n_opt_twist        = opt_options['blade_aero']['n_opt_twist']
        self.n_opt_chord   = n_opt_chord        = opt_options['blade_aero']['n_opt_chord']

        self.add_input('s',               val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        # Blade twist
        self.add_input('twist_original',  val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is the one defined in the yaml.')
        self.add_input('s_opt_twist',     val=np.zeros(n_opt_twist),            desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist angle')
        self.add_input('twist_opt_gain',  val=0.5 * np.ones(n_opt_twist),       desc='1D array of the non-dimensional gains to optimize the blade spanwise distribution of the twist angle')
        # Blade chord
        self.add_input('chord_original',  val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span. The chord is the one defined in the yaml.')
        self.add_input('s_opt_chord',     val=np.zeros(n_opt_chord),            desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade chord')
        self.add_input('chord_opt_gain',  val=np.ones(n_opt_chord),             desc='1D array of the non-dimensional gains to optimize the blade spanwise distribution of the chord')

        self.add_output('twist_param',    val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is the result of the parameterization.')
        self.add_output('chord_param',    val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span. The chord is the result of the parameterization.')

    def compute(self, inputs, outputs):

        twist_opt_gain_nd       = inputs['twist_opt_gain']
        twist_upper             = np.ones(self.n_opt_twist) * 20.  / 180. * np.pi
        twist_lower             = np.ones(self.n_opt_twist) * -20. / 180. * np.pi
        twist_opt_gain_rad      = twist_opt_gain_nd * (twist_upper - twist_lower) + twist_lower
        twist_opt_gain_rad_interp   = np.interp(inputs['s'], inputs['s_opt_twist'], twist_opt_gain_rad)
        outputs['twist_param']  = inputs['twist_original'] + twist_opt_gain_rad_interp

        chord_opt_gain_nd       = inputs['chord_opt_gain']
        chord_opt_gain_m_interp = np.interp(inputs['s'], inputs['s_opt_chord'], chord_opt_gain_nd)
        outputs['chord_param']  = inputs['chord_original'] * chord_opt_gain_m_interp

class WT_Rotor(Group):
    # Openmdao group to run the aerostructural analysis of the wind turbine rotor
    
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        opt_options     = self.options['opt_options']
        
        # Optimization parameters initialized as indipendent variable component
        opt_var = IndepVarComp()
        opt_var.add_output('twist_opt_gain', val = 0.5 * np.ones(opt_options['blade_aero']['n_opt_twist']))
        opt_var.add_output('chord_opt_gain', val = np.ones(opt_options['blade_aero']['n_opt_chord']))
        self.add_subsystem('opt_var',opt_var)

        # Analysis components
        self.add_subsystem('param',     ParametrizeBladeAero(blade_init_options = wt_init_options['blade'], opt_options = opt_options))
        self.add_subsystem('ra',        RotorAeroPower(wt_init_options      = wt_init_options))
        self.add_subsystem('rs',        RotorStructure(wt_init_options      = wt_init_options, opt_options = opt_options))
        # self.add_subsystem('rc',        RotorCost(wt_init_options      = wt_init_options))

        # Connections to blade parametrization
        self.connect('opt_var.twist_opt_gain',       'param.twist_opt_gain')
        self.connect('opt_var.chord_opt_gain',       'param.chord_opt_gain')
        self.connect('param.twist_param',           ['ra.theta','rs.theta'])
        self.connect('param.twist_param',            'rs.tip_pos.theta_tip',   src_indices=[-1])
        self.connect('param.chord_param',           ['ra.chord','rs.chord'])

        # Connection from ra to rs for the rated conditions
        # self.connect('ra.powercurve.rated_V',        'rs.aero_rated.V_load')
        self.connect('ra.powercurve.rated_V',        'rs.gust.V_hub')
        self.connect('rs.gust.V_gust',              ['rs.aero_gust.V_load'])
        self.connect('ra.powercurve.rated_Omega',   ['rs.Omega_load', 'rs.aeroloads_Omega', 'rs.curvefem.Omega'])
        self.connect('ra.powercurve.rated_pitch',   ['rs.pitch_load', 'rs.aeroloads_pitch'])
