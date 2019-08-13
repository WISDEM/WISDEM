from __future__ import print_function
import numpy as np
import warnings
import copy, time

from openmdao.api import ExplicitComponent, Group, IndepVarComp

from wisdem.commonse.akima import Akima, akima_interp_with_derivs
from wisdem.ccblade.ccblade_component import CCBladeGeometry
from wisdem.ccblade import CCAirfoil
from wisdem.airfoilprep import Airfoil
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp

from scipy.interpolate import PchipInterpolator

NINPUT = 5



class BladeGeometry(ExplicitComponent):
    def initialize(self):
        self.options.declare('RefBlade')
    
    def setup(self):
        self.refBlade = RefBlade = self.options['RefBlade']
        npts   = len(self.refBlade['pf']['s'])
        NINPUT = len(self.refBlade['ctrl_pts']['r_in'])
        NAF    = len(self.refBlade['outer_shape_bem']['airfoil_position']['grid'])

        # variables
        self.add_discrete_input('blade_in_overwrite', val={}, desc='optional input blade that can be used to overwrite RefBlade from initialization, first intended for the inner loop of a nested optimization')

        self.add_input('bladeLength', val=0.0, units='m', desc='blade length (if not precurved or swept) otherwise length of blade before curvature')
        self.add_input('r_max_chord', val=0.0, desc='location of max chord on unit radius')
        self.add_input('chord_in', val=np.zeros(NINPUT), units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
        self.add_input('theta_in', val=np.zeros(NINPUT), units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
        self.add_input('precurve_in', val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_input('presweep_in', val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_input('sparT_in', val=np.zeros(NINPUT), units='m', desc='thickness values of spar cap that linearly vary from non-cylinder position to tip')
        self.add_input('teT_in', val=np.zeros(NINPUT), units='m', desc='thickness values of trailing edge panels that linearly vary from non-cylinder position to tip')
        # self.add_input('thickness_in', val=np.zeros(NINPUT), desc='relative thickness of airfoil distribution control points')
        self.add_input('airfoil_position', val=np.zeros(NAF), desc='spanwise position of airfoils')

        # parameters
        self.add_input('hubFraction', val=0.0, desc='hub location as fraction of radius')

        # Blade geometry outputs
        self.add_output('Rhub', val=0.0, units='m', desc='dimensional radius of hub')
        self.add_output('Rtip', val=0.0, units='m', desc='dimensional radius of tip')
        self.add_output('r_pts', val=np.zeros(npts), units='m', desc='dimensional aerodynamic grid')
        self.add_output('r_in', val=np.zeros(NINPUT), units='m', desc='Spline control points for inputs')
        self.add_output('max_chord', val=0.0, units='m', desc='maximum chord length')
        self.add_output('chord', val=np.zeros(npts), units='m', desc='chord at airfoil locations')
        self.add_output('theta', val=np.zeros(npts), units='deg', desc='twist at airfoil locations')
        self.add_output('precurve', val=np.zeros(npts), units='m', desc='precurve at airfoil locations')
        self.add_output('presweep', val=np.zeros(npts), units='m', desc='presweep at structural locations')
        # self.add_output('sparT', val=np.zeros(npts), units='m', desc='dimensional spar cap thickness distribution')
        # self.add_output('teT', val=np.zeros(npts), units='m', desc='dimensional trailing-edge panel thickness distribution')
        self.add_output('rthick', val=np.zeros(npts), desc='relative thickness of airfoil distribution')

        self.add_output('hub_diameter', val=0.0, units='m')
        self.add_output('diameter', val=0.0, units='m')
        
        self.add_discrete_output('airfoils', val=[], desc='Spanwise coordinates for aerodynamic analysis')
        self.add_output('le_location', val=np.zeros(npts), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_output('chord_ref', val=np.zeros(npts), desc='Chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c)')

        # Blade layup outputs
        self.add_discrete_output('materials', val=np.zeros(npts), desc='material properties of composite materials')
        
        self.add_discrete_output('upperCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for upper surface')
        self.add_discrete_output('lowerCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for lower surface')
        self.add_discrete_output('websCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for shear webs')
        self.add_discrete_output('profile', val=np.zeros(npts), desc='list of CompositeSection profiles')
        
        self.add_discrete_output('sector_idx_strain_spar_ss', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for spar (PreComp definition of sector)')
        self.add_discrete_output('sector_idx_strain_spar_ps', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for spar (PreComp definition of sector)')
        self.add_discrete_output('sector_idx_strain_te_ss', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for trailing edge (PreComp definition of sector)')
        self.add_discrete_output('sector_idx_strain_te_ps', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for trailing edge (PreComp definition of sector)')

        self.add_discrete_output('blade_out', val={}, desc='updated blade dictionary for ontology')

        self.add_output('sparT_in_out', val=np.zeros(NINPUT), units='m', desc='thickness values of spar cap that linearly vary from non-cylinder position to tip, pass through for nested optimization')

        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        if discrete_inputs['blade_in_overwrite'] != {}:
            blade = copy.deepcopy(discrete_inputs['blade_in_overwrite'])
        else:
            blade = copy.deepcopy(self.refBlade)

        NINPUT = len(blade['ctrl_pts']['r_in'])

        Rhub = inputs['hubFraction'] * inputs['bladeLength']
        Rtip = Rhub + inputs['bladeLength']

        outputs['Rhub']     = Rhub
        outputs['Rtip']     = Rtip
        # r_in                 = np.r_[0.0, blade['ctrl_pts']['r_cylinder'].tolist(), np.linspace(inputs['r_max_chord'], 1.0, NINPUT-2)]
        # outputs['r_in']     = Rhub + (Rtip-Rhub)*np.r_[0.0, blade['ctrl_pts']['r_cylinder'].tolist(), np.linspace(inputs['r_max_chord'], 1.0, NINPUT-2)]
        r_in                 = np.r_[0.,
                                     np.linspace(blade['ctrl_pts']['r_cylinder'], inputs['r_max_chord'], num=3).flatten()[:-1],
                                     np.linspace(inputs['r_max_chord'], 1., NINPUT-3).flatten()]
        outputs['r_in']     = Rhub + (Rtip-Rhub)*np.array(r_in)

        blade['ctrl_pts']['bladeLength']  = inputs['bladeLength']
        blade['ctrl_pts']['r_in']         = r_in
        blade['ctrl_pts']['chord_in']     = inputs['chord_in']
        blade['ctrl_pts']['theta_in']     = inputs['theta_in']
        blade['ctrl_pts']['precurve_in']  = inputs['precurve_in']
        blade['ctrl_pts']['presweep_in']  = inputs['presweep_in']
        blade['ctrl_pts']['sparT_in']     = inputs['sparT_in']
        blade['ctrl_pts']['teT_in']       = inputs['teT_in']
        blade['ctrl_pts']['r_max_chord']  = inputs['r_max_chord']
        # blade['ctrl_pts']['thickness_in'] = inputs['thickness_in']

        #check that airfoil positions are increasing
        correct_af_position = False
        airfoil_position = copy.deepcopy(inputs['airfoil_position']).tolist()
        for i in reversed(range(1,len(airfoil_position))):
            if airfoil_position[i] <= airfoil_position[i-1]:
                airfoil_position[i-1] = airfoil_position[i] - 0.001
                correct_af_position = True

        if correct_af_position:
            blade['outer_shape_bem']['airfoil_position']['grid'] = airfoil_position
            warning_corrected_airfoil_position = "Airfoil spanwise positions must be increasing.  Changed from: %s to: %s" % (inputs['airfoil_position'].tolist(), airfoil_position)
            warnings.warn(warning_corrected_airfoil_position)
        else:
            blade['outer_shape_bem']['airfoil_position']['grid'] = inputs['airfoil_position'].tolist()

        # Update
        refBlade = ReferenceBlade()
        refBlade.verbose        = False
        refBlade.NINPUT         = len(outputs['r_in'])
        refBlade.NPTS           = len(blade['pf']['s'])
        refBlade.analysis_level = blade['analysis_level']
        if blade['analysis_level'] < 3:
            refBlade.spar_var   = blade['precomp']['spar_var']
            refBlade.te_var     = blade['precomp']['te_var']
        
        # blade_out = blade
        blade_out = refBlade.update(blade)
        
        # Although the inputs get mirrored to outputs, this is still necessary so that the user can designate the inputs as design variables
        outputs['hub_diameter']           = 2.0*Rhub
        outputs['r_pts']                  = Rhub + (Rtip-Rhub)*np.array(blade_out['pf']['s'])
        outputs['diameter']               = 2.0*outputs['r_pts'][-1]

        outputs['chord']                  = blade_out['pf']['chord']
        outputs['max_chord']              = max(blade_out['pf']['chord'])
        outputs['theta']                  = blade_out['pf']['theta']
        outputs['precurve']               = blade_out['pf']['precurve']
        outputs['presweep']               = blade_out['pf']['presweep']
        outputs['rthick']                 = blade_out['pf']['rthick']

        discrete_outputs['airfoils']               = blade_out['airfoils']
        outputs['le_location']            = blade_out['pf']['p_le']
        discrete_outputs['upperCS']                = blade_out['precomp']['upperCS']
        discrete_outputs['lowerCS']                = blade_out['precomp']['lowerCS']
        discrete_outputs['websCS']                 = blade_out['precomp']['websCS']
        discrete_outputs['profile']                = blade_out['precomp']['profile']
        outputs['chord_ref']              = blade_out['pf']['chord']
        discrete_outputs['materials']              = blade_out['precomp']['materials']
        
        # Assumptions:
        # - pressure and suction side regions are the same (i.e. spar cap is the Nth region on both side)
        # - if the composite layer is divided into multiple regions (i.e. if the spar cap is split into 3 regions due to the web locations),
        #   the middle region is selected with int(n_reg/2), note for an even number of regions, this rounds up
        discrete_outputs['sector_idx_strain_spar_ss'] = blade_out['precomp']['sector_idx_strain_spar_ss']
        discrete_outputs['sector_idx_strain_spar_ps'] = blade_out['precomp']['sector_idx_strain_spar_ps']
        discrete_outputs['sector_idx_strain_te_ss']   = blade_out['precomp']['sector_idx_strain_te_ss']
        discrete_outputs['sector_idx_strain_te_ps']   = blade_out['precomp']['sector_idx_strain_te_ps']

        discrete_outputs['blade_out'] = blade_out
        outputs['sparT_in_out'] = inputs['sparT_in']
        
class Location(ExplicitComponent):
    def setup(self):
        self.add_input('hub_height', val=0.0, units='m', desc='Tower top hub height')
        self.add_output('wind_zvec', val=np.zeros(1), units='m', desc='Tower top hub height as vector')
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['wind_zvec'] = np.array([ np.float(inputs['hub_height']) ])

    def compute_partials(self, inputs, J):
        J['wind_zvec','hub_height'] = np.ones(1)


        
class TurbineClass(ExplicitComponent):
    def setup(self):
        # parameters
        self.add_discrete_input('turbine_class', val='I', desc='IEC turbine class')
        self.add_input('V_mean_overwrite', val=0., desc='overwrite value for mean velocity for using user defined CDFs')

        # outputs should be constant
        self.add_output('V_mean', shape=1, units='m/s', desc='IEC mean wind speed for Rayleigh distribution')
        self.add_output('V_extreme1', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_output('V_extreme50', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_output('V_extreme_full', shape=2, units='m/s', desc='IEC extreme wind speed at hub height')
        
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        turbine_class = discrete_inputs['turbine_class'].upper()

        if turbine_class == 'I':
            Vref = 50.0
        elif turbine_class == 'II':
            Vref = 42.5
        elif turbine_class == 'III':
            Vref = 37.5
        elif turbine_class == 'IV':
            Vref = 30.0
        else:
            raise ValueError('turbine_class input must be I/II/III/IV')

        if inputs['V_mean_overwrite'] == 0.:
            outputs['V_mean'] = 0.2*Vref
        else:
            outputs['V_mean'] = inputs['V_mean_overwrite']
        outputs['V_extreme1'] = 0.8*Vref
        outputs['V_extreme50'] = 1.4*Vref
        outputs['V_extreme_full'][0] = 1.4*Vref # for extreme cases TODO: check if other way to do
        outputs['V_extreme_full'][1] = 1.4*Vref



class RotorGeometry(Group):
    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('topLevelFlag',default=False)
    
    def setup(self):
        RefBlade = self.options['RefBlade']
        topLevelFlag   = self.options['topLevelFlag']
        NINPUT = len(RefBlade['ctrl_pts']['r_in'])
        NAF    = len(RefBlade['outer_shape_bem']['airfoil_position']['grid'])

        # Independent variables that are unique to TowerSE
        if topLevelFlag:
            geomIndeps = IndepVarComp()
            geomIndeps.add_output('bladeLength', 0.0, units='m')
            geomIndeps.add_output('hubFraction', 0.0)
            geomIndeps.add_output('r_max_chord', 0.0)
            geomIndeps.add_output('chord_in', np.zeros(NINPUT),units='m')
            geomIndeps.add_output('theta_in', np.zeros(NINPUT), units='deg')
            geomIndeps.add_output('precurve_in', np.zeros(NINPUT), units='m')
            geomIndeps.add_output('presweep_in', np.zeros(NINPUT), units='m')
            geomIndeps.add_output('precurveTip', 0.0, units='m')
            geomIndeps.add_output('presweepTip', 0.0, units='m')
            geomIndeps.add_output('precone', 0.0, units='deg')
            geomIndeps.add_output('tilt', 0.0, units='deg')
            geomIndeps.add_output('yaw', 0.0, units='deg')
            geomIndeps.add_discrete_output('nBlades', 3)
            geomIndeps.add_discrete_output('downwind', False)
            geomIndeps.add_discrete_output('turbine_class', val='I', desc='IEC turbine class')
            geomIndeps.add_discrete_output('blade_in_overwrite', val={}, desc='IEC turbine class')
            geomIndeps.add_output('V_mean_overwrite', val=0.0, desc='optional overwrite value for mean velocity for using user defined CDFs')
            geomIndeps.add_output('airfoil_posision', val=np.zeros(NAF))
            geomIndeps.add_output('sparT_in', val=np.zeros(NINPUT), units='m', desc='spar cap thickness parameters')
            geomIndeps.add_output('teT_in', val=np.zeros(NINPUT), units='m', desc='trailing-edge thickness parameters')
            self.add_subsystem('geomIndeps', geomIndeps, promotes=['*'])
            
        # --- Rotor Definition ---
        self.add_subsystem('loc', Location(), promotes=['*'])
        self.add_subsystem('turbineclass', TurbineClass(), promotes=['turbine_class','V_mean_overwrite'])
        #self.add_subsystem('spline0', BladeGeometry(RefBlade))
        self.add_subsystem('spline', BladeGeometry(RefBlade=RefBlade), promotes=['*'])
        self.add_subsystem('geom', CCBladeGeometry(), promotes=['precone','precurveTip'])

        # connections to spline0
        #self.connect('r_max_chord', 'spline0.r_max_chord')
        #self.connect('chord_in', 'spline0.chord_in')
        #self.connect('theta_in', 'spline0.theta_in')
        #self.connect('precurve_in', 'spline0.precurve_in')
        #self.connect('presweep_in', 'spline0.presweep_in')
        #self.connect('bladeLength', 'spline0.bladeLength')
        #self.connect('hubFraction', 'spline0.hubFraction')
        #self.connect('sparT_in', 'spline0.sparT_in')
        #self.connect('teT_in', 'spline0.teT_in')

        # connections to spline
        #self.connect('r_max_chord', 'spline.r_max_chord')
        #self.connect('chord_in', 'spline.chord_in')
        #self.connect('theta_in', 'spline.theta_in')
        #self.connect('precurve_in', 'spline.precurve_in')
        #self.connect('presweep_in', 'spline.presweep_in')
        #self.connect('bladeLength', 'spline.bladeLength')
        #self.connect('hubFraction', 'spline.hubFraction')
        #self.connect('sparT_in', 'spline.sparT_in')
        #self.connect('teT_in', 'spline.teT_in')

        # connections to geom
        self.connect('Rtip', 'geom.Rtip')
        #self.connect('precone', 'geom.precone')
        #self.connect('precurveTip', 'geom.precurveTip')


if __name__ == "__main__":

    # Turbine Ontology input
    fname_input = "turbine_inputs/nrel5mw_mod.yaml"

    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose = True
    refBlade.NINPUT  = NINPUT
    refBlade.NPITS   = 50

    refBlade.spar_var = ['Spar_Cap_SS', 'Spar_Cap_PS']
    refBlade.te_var   = 'TE_reinforcement'

    blade = refBlade.initialize(fname_input)
    rotor = RotorGeometry(blade)
