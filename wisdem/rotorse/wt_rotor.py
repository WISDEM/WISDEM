import numpy as np
import os
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from wisdem.rotorse.rotor_aeropower import RotorAeroPower
from wisdem.rotorse.rotor_structure_simple import RotorStructure

class ParametrizeBladeAero(ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the outer shape of the wind turbine rotor blades
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        blade_init_options = self.options['blade_init_options']
        opt_options        = self.options['opt_options']
        n_span             = blade_init_options['n_span']
        self.n_opt_twist   = n_opt_twist = opt_options['blade_aero']['n_opt_twist']
        self.n_opt_chord   = n_opt_chord = opt_options['blade_aero']['n_opt_chord']
        
        # Inputs
        self.add_input('s',               val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        # Blade twist
        self.add_input('twist_original',  val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is the one defined in the yaml.')
        self.add_input('s_opt_twist',     val=np.zeros(n_opt_twist),            desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist angle')
        self.add_input('twist_opt_gain',  val=0.5 * np.ones(n_opt_twist),       desc='1D array of the non-dimensional gains to optimize the blade spanwise distribution of the twist angle')
        # Blade chord
        self.add_input('chord_original',  val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span. The chord is the one defined in the yaml.')
        self.add_input('s_opt_chord',     val=np.zeros(n_opt_chord),            desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade chord')
        self.add_input('chord_opt_gain',  val=np.ones(n_opt_chord),             desc='1D array of the non-dimensional gains to optimize the blade spanwise distribution of the chord')

        # Outputs
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


class ParametrizeBladeStruct(ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the structural design of the wind turbine rotor blades
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        blade_init_options = self.options['blade_init_options']
        self.opt_options   = opt_options   = self.options['opt_options']
        self.n_span        = n_span        = blade_init_options['n_span']
        self.n_layers      = n_layers      = blade_init_options['n_layers']
        self.n_opt_spar_ss = n_opt_spar_ss = opt_options['blade_struct']['n_opt_spar_ss']
        self.n_opt_spar_ps = n_opt_spar_ps = opt_options['blade_struct']['n_opt_spar_ps']

        # Inputs
        self.add_input('s',                     val=np.zeros(n_span),       desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        # Blade spar suction side
        self.add_discrete_input('layer_name',   val=n_layers * [''],        desc='1D array of the names of the layers modeled in the blade structure.')
        self.add_input('layer_thickness_original',val=np.zeros((n_layers, n_span)), units='m',desc='2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('s_opt_spar_ss',         val=np.zeros(n_opt_spar_ss),desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side')
        self.add_input('spar_ss_opt_gain',      val=np.ones(n_opt_spar_ss), desc='1D array of the non-dimensional gains to optimize the blade spanwise distribution of the spar caps suction side')
        # Blade spar suction side
        self.add_output('s_opt_spar_ps',         val=np.zeros(n_opt_spar_ps),desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap pressure side')
        self.add_input('spar_ps_opt_gain',      val=np.ones(n_opt_spar_ps), desc='1D array of the non-dimensional gains to optimize the blade spanwise distribution of the spar caps pressure side')
        
        # Outputs
        self.add_output('layer_thickness_param', val=np.zeros((n_layers, n_span)), units='m',desc='2D array of the thickness of the layers of the blade structure after the parametrization. The first dimension represents each layer, the second dimension represents each entry along blade span.')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        spar_ss_name = self.opt_options['blade_struct']['spar_ss_var']
        spar_ps_name = self.opt_options['blade_struct']['spar_ps_var']

        for i in range(self.n_layers):
            if discrete_inputs['layer_name'][i] == spar_ss_name:
                spar_ss_opt_gain_nd = inputs['spar_ss_opt_gain']
                opt_gain_m_interp   = np.interp(inputs['s'], outputs['s_opt_spar_ss'], spar_ss_opt_gain_nd)
            elif discrete_inputs['layer_name'][i] == spar_ps_name:
                spar_ps_opt_gain_nd = inputs['spar_ps_opt_gain']
                opt_gain_m_interp   = np.interp(inputs['s'], outputs['s_opt_spar_ps'], spar_ps_opt_gain_nd)
            else:
                opt_gain_m_interp = np.ones(self.n_span)

            outputs['layer_thickness_param'][i,:] = inputs['layer_thickness_original'][i,:] * opt_gain_m_interp

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
        opt_var.add_output('twist_opt_gain',   val = 0.5 * np.ones(opt_options['blade_aero']['n_opt_twist']))
        opt_var.add_output('chord_opt_gain',   val = np.ones(opt_options['blade_aero']['n_opt_chord']))
        opt_var.add_output('spar_ss_opt_gain', val = np.ones(opt_options['blade_struct']['n_opt_spar_ss']))
        opt_var.add_output('spar_ps_opt_gain', val = np.ones(opt_options['blade_struct']['n_opt_spar_ps']))
        self.add_subsystem('opt_var',opt_var)

        # Analysis components
        self.add_subsystem('pa',    ParametrizeBladeAero(blade_init_options   = wt_init_options['blade'], opt_options = opt_options))
        self.add_subsystem('ps',    ParametrizeBladeStruct(blade_init_options = wt_init_options['blade'], opt_options = opt_options))
        self.add_subsystem('ra',    RotorAeroPower(wt_init_options      = wt_init_options))
        self.add_subsystem('rs',    RotorStructure(wt_init_options      = wt_init_options, opt_options = opt_options))
        # self.add_subsystem('rc',        RotorCost(wt_init_options      = wt_init_options))

        # Connections to blade aero parametrization
        self.connect('opt_var.twist_opt_gain',    'pa.twist_opt_gain')
        self.connect('opt_var.chord_opt_gain',    'pa.chord_opt_gain')
        
        # Connections from blade aero parametrization to rotorse
        self.connect('pa.twist_param',           ['ra.theta','rs.theta'])
        self.connect('pa.twist_param',            'rs.tip_pos.theta_tip',   src_indices=[-1])
        self.connect('pa.chord_param',           ['ra.chord','rs.chord'])

        # Connections to blade struct parametrization
        self.connect('opt_var.spar_ss_opt_gain','ps.spar_ss_opt_gain')
        self.connect('opt_var.spar_ps_opt_gain','ps.spar_ps_opt_gain')

        # Connections from blade struct parametrization to rotorse
        self.connect('ps.layer_thickness_param', 'rs.precomp.layer_thickness')
        self.connect('ps.s_opt_spar_ss',   'rs.constr.s_opt_spar_ss')
        self.connect('ps.s_opt_spar_ps',   'rs.constr.s_opt_spar_ps')

        # Connection from ra to rs for the rated conditions
        # self.connect('ra.powercurve.rated_V',        'rs.aero_rated.V_load')
        self.connect('ra.powercurve.rated_V',        'rs.gust.V_hub')
        self.connect('rs.gust.V_gust',              ['rs.aero_gust.V_load'])
        self.connect('ra.powercurve.rated_Omega',   ['rs.Omega_load', 'rs.aeroloads_Omega', 'rs.curvefem.Omega'])
        self.connect('ra.powercurve.rated_pitch',   ['rs.pitch_load', 'rs.aeroloads_pitch'])
