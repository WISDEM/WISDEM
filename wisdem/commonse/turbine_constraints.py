from openmdao.api import Group, ExplicitComponent
from .csystem import DirectionVector
from .utilities import interp_with_deriv
from wisdem.commonse import NFREQ
import numpy as np


# class TowerModes(ExplicitComponent):
#     def setup(self):

#         self.add_input('tower_freq', val=np.zeros(2), units='Hz', desc='First natural frequencies of tower (and substructure)')
#         self.add_input('rotor_omega', val=0.0, units='rpm', desc='rated rotor rotation speed')
#         self.add_input('gamma_freq', val=0.0, desc='partial safety factor for fatigue')
#         self.add_discrete_input('blade_number', 3, desc='number of rotor blades')

#         self.add_output('frequencyNP_margin_low', val=np.zeros(2), desc='Upper bound constraint of tower/structure frequency to blade passing frequency with margin')
#         self.add_output('frequencyNP_margin_high', val=np.zeros(2), desc='Lower bound constraint of tower/structure frequency to blade passing frequency with margin')
#         self.add_output('frequency1P_margin_low', val=np.zeros(2), desc='Upper bound constraint of tower/structure frequency to rotor frequency with margin')
#         self.add_output('frequency1P_margin_high', val=np.zeros(2), desc='Lower bound constraint of tower/structure frequency to rotor frequency with margin')

#         # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
#     def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
#         freq_struct = inputs['tower_freq']
#         gamma       = inputs['gamma_freq']
#         oneP        = (inputs['rotor_omega']/60.0)
#         oneP_high   = oneP * gamma
#         oneP_low    = oneP / gamma
#         threeP      = oneP * discrete_inputs['blade_number']
#         threeP_high = threeP * gamma
#         threeP_low  = threeP / gamma
        
#         # Compute margins between (N/3)P and structural frequencies
#         indicator_high = threeP_high * np.ones(freq_struct.shape)
#         indicator_high[freq_struct < threeP_low] = 1e-16
#         outputs['frequencyNP_margin_high'] = freq_struct / indicator_high

#         indicator_low = threeP_low * np.ones(freq_struct.shape)
#         indicator_low[freq_struct > threeP_high] = 1e30
#         outputs['frequencyNP_margin_low']  = freq_struct / indicator_low

#         # Compute margins between 1P and structural frequencies
#         indicator_high = oneP_high * np.ones(freq_struct.shape)
#         indicator_high[freq_struct < oneP_low] = 1e-16
#         outputs['frequency1P_margin_high'] = freq_struct / indicator_high

#         indicator_low = oneP_low * np.ones(freq_struct.shape)
#         indicator_low[freq_struct > oneP_high] = 1e30
#         outputs['frequency1P_margin_low']  = freq_struct / indicator_low

class TowerModes(ExplicitComponent):
    def setup(self):

        self.add_input('tower_freq', val=np.zeros(2), units='Hz', desc='First natural frequencies of tower (and substructure)')
        self.add_input('rotor_omega', val=0.0, units='rpm', desc='rated rotor rotation speed')
        self.add_input('gamma_freq', val=0.0, desc='partial safety factor for fatigue')
        self.add_discrete_input('blade_number', 3, desc='number of rotor blades')

        self.add_output('frequencyNP_margin', val=np.zeros(2), desc='constraint on tower/structure frequency to blade passing frequency with margin')
        self.add_output('frequency1P_margin', val=np.zeros(2), desc='constraint on tower/structure frequency to rotor frequency with margin')

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        freq_struct = inputs['tower_freq']
        gamma       = inputs['gamma_freq']
        oneP        = (inputs['rotor_omega']/60.0)
        oneP_high   = oneP * gamma
        oneP_low    = oneP / gamma
        threeP      = oneP * discrete_inputs['blade_number']
        threeP_high = threeP * gamma
        threeP_low  = threeP / gamma
        
        outputs['frequencyNP_margin'] = np.array([min([f-threeP_low, threeP_high-f]) for f in freq_struct]).flatten()
        outputs['frequency1P_margin'] = np.array([min([f-oneP_low, oneP_high-f]) for f in freq_struct]).flatten()


class MaxTipDeflection(ExplicitComponent):
    def initialize(self):
        self.options.declare('nFullTow')
        
    def setup(self):
        nFullTow = self.options['nFullTow']

        self.add_discrete_input('downwind',       val=False)
        self.add_input('tip_deflection', val=0.0,       units='m',  desc='Blade tip deflection in yaw x-direction')
        self.add_input('Rtip',           val=0.0,       units='m',  desc='Blade tip location in z_b')
        self.add_input('precurveTip',    val=0.0,       units='m',  desc='Blade tip location in x_b')
        self.add_input('presweepTip',    val=0.0,       units='m',  desc='Blade tip location in y_b')
        self.add_input('precone',        val=0.0,       units='deg',desc='Rotor precone angle')
        self.add_input('tilt',           val=0.0,       units='deg',desc='Nacelle uptilt angle')
        self.add_input('overhang',       val=0.0,       units='m',  desc='Horizontal distance between hub and tower-top axis')
        self.add_input('z_full',         val=np.zeros(nFullTow),units='m',  desc='z-coordinates of tower at fine-section nodes')
        self.add_input('d_full',         val=np.zeros(nFullTow),units='m',  desc='Diameter of tower at fine-section nodes')
        self.add_input('gamma_m',        val=0.0, desc='safety factor on materials')

        self.add_output('tip_deflection_ratio',      val=0.0,           desc='Ratio of blade tip deflectiion towardsa the tower and clearance between undeflected blade tip and tower')
        self.add_output('blade_tip_tower_clearance', val=0.0, units='m',desc='Clearance between undeflected blade tip and tower in x-direction of yaw c.s.')

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack variables
        z_tower     = inputs['z_full']
        d_tower     = inputs['d_full']
        overhang    = inputs['overhang']
        tt2hub      = overhang * np.sin(inputs['tilt'] / 180. * np.pi)
        precone     = inputs['precone']
        tilt        = inputs['tilt']
        delta       = inputs['tip_deflection']
        upwind      = not discrete_inputs['downwind']
        
        # Coordinates of blade tip in yaw c.s.
        blade_yaw = DirectionVector(inputs['precurveTip'], inputs['presweepTip'], inputs['Rtip']).\
                    bladeToAzimuth(precone).azimuthToHub(180.0).hubToYaw(tilt)

        # Find the radius of tower where blade passes
        z_interp = z_tower[-1] + tt2hub + blade_yaw.z
        
        d_interp, ddinterp_dzinterp, ddinterp_dtowerz, ddinterp_dtowerd = interp_with_deriv(z_interp, z_tower, d_tower)
        r_interp = 0.5 * d_interp
        drinterp_dzinterp = 0.5 * ddinterp_dzinterp
        drinterp_dtowerz  = 0.5 * ddinterp_dtowerz
        drinterp_dtowerd  = 0.5 * ddinterp_dtowerd

        # Max deflection before strike
        if upwind:
            parked_margin = overhang - blade_yaw.x - r_interp
        else:
            parked_margin = -overhang + blade_yaw.x - r_interp
        outputs['blade_tip_tower_clearance']   = parked_margin
        outputs['tip_deflection_ratio']        = delta * inputs['gamma_m'] / parked_margin

    
class TurbineConstraints(Group):

    def initialize(self):
        self.options.declare('nFull')
        
    def setup(self):
        nFull = self.options['nFull']
        
        self.add_subsystem('modes', TowerModes(), promotes=['*'])
        self.add_subsystem('tipd', MaxTipDeflection(nFullTow=nFull), promotes=['*'])
