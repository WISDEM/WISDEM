from __future__ import print_function
import numpy as np
import warnings
import copy, time

from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem

from wisdem.commonse.akima import Akima, akima_interp_with_derivs
from wisdem.ccblade.ccblade_component import CCBladeGeometry
from wisdem.ccblade import CCAirfoil
from wisdem.airfoilprep import Airfoil
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.rotorse.precomp import PreComp, Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from wisdem.rotorse.rotor_cost import blade_cost_model

from scipy.interpolate import PchipInterpolator

NINPUT = 8



class BladeGeometry(ExplicitComponent):
    def initialize(self):
        self.options.declare('RefBlade')

        # Blade Cost Model Options
        self.options.declare('verbosity',           default=False)
        self.options.declare('tex_table',           default=False)
        self.options.declare('generate_plots',      default=False)
        self.options.declare('show_plots',          default=False)
        self.options.declare('show_warnings',       default=False)
        self.options.declare('discrete',            default=False)
        self.options.declare('user_update_routine', default=None)
    
    def setup(self):
        self.refBlade = RefBlade = self.options['RefBlade']
        npts    = len(self.refBlade['pf']['s'])
        NINPUT  = len(self.refBlade['ctrl_pts']['r_in'])
        NAF     = len(self.refBlade['outer_shape_bem']['airfoil_position']['grid'])
        NAFgrid = len(self.refBlade['airfoils_aoa'])
        NRe     = len(self.refBlade['airfoils_Re'])

        self.add_input('bladeLength',   val=0.0, units='m', desc='blade length (if not precurved or swept) otherwise length of blade before curvature')
        self.add_input('r_max_chord',   val=0.0, desc='location of max chord on unit radius')
        self.add_input('chord_in',      val=np.zeros(NINPUT), units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
        self.add_input('theta_in',      val=np.zeros(NINPUT), units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
        self.add_input('precurve_in',   val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_input('presweep_in',   val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_input('sparT_in',      val=np.zeros(NINPUT), units='m', desc='thickness values of spar cap that linearly vary from non-cylinder position to tip')
        self.add_input('teT_in',        val=np.zeros(NINPUT), units='m', desc='thickness values of trailing edge panels that linearly vary from non-cylinder position to tip')
        # self.add_input('leT_in',        val=np.zeros(NINPUT), units='m', desc='thickness values of leading edge panels that linearly vary from non-cylinder position to tip')
        self.add_input('airfoil_position', val=np.zeros(NAF), desc='spanwise position of airfoils')

        # parameters
        self.add_input('hubFraction', val=0.0, desc='hub location as fraction of radius')

        # Blade geometry outputs
        self.add_output('Rhub', val=0.0, units='m', desc='dimensional radius of hub')
        self.add_output('Rtip', val=0.0, units='m', desc='dimensional radius of tip')
        self.add_output('r',            val=np.zeros(npts), units='m', desc='dimensional aerodynamic grid')
        self.add_output('r_in',         val=np.zeros(NINPUT), units='m', desc='Spline control points for inputs')
        self.add_output('max_chord',    val=0.0, units='m', desc='maximum chord length')
        self.add_output('chord',        val=np.zeros(npts), units='m', desc='chord at airfoil locations')
        self.add_output('theta',        val=np.zeros(npts), units='deg', desc='twist at airfoil locations')
        self.add_output('precurve',     val=np.zeros(npts), units='m', desc='precurve at airfoil locations')
        self.add_output('presweep',     val=np.zeros(npts), units='m', desc='presweep at structural locations')
        self.add_output('rthick',       val=np.zeros(npts), desc='relative thickness of airfoil distribution')
        self.add_output('le_location',  val=np.zeros(npts))
        self.add_output('hub_diameter', val=0.0, units='m')
        self.add_output('diameter',     val=0.0, units='m')

        # Airfoil properties
        # self.add_discrete_output('airfoils', val=[], desc='Spanwise coordinates for aerodynamic analysis')
        self.add_output('airfoils_cl',  val=np.zeros((NAFgrid, npts, NRe)), desc='lift coefficients, spanwise')
        self.add_output('airfoils_cd',  val=np.zeros((NAFgrid, npts, NRe)), desc='drag coefficients, spanwise')
        self.add_output('airfoils_cm',  val=np.zeros((NAFgrid, npts, NRe)), desc='moment coefficients, spanwise')
        self.add_output('airfoils_aoa', val=np.zeros((NAFgrid)), units='deg', desc='angle of attack grid for polars')
        self.add_output('airfoils_Re',  val=np.zeros((NRe)), desc='Reynolds numbers of polars')
        
        # Airfoil coordinates
        self.add_output('airfoils_coord_x',  val=np.zeros((200, npts)), desc='x airfoil coordinate, spanwise')
        self.add_output('airfoils_coord_y',  val=np.zeros((200, npts)), desc='y airfoil coordinate, spanwise')
        
        # Beam properties
        self.add_output('z',            val=np.zeros(npts), units='m',      desc='locations of properties along beam')
        self.add_output('EA',           val=np.zeros(npts), units='N',      desc='axial stiffness')
        self.add_output('EIxx',         val=np.zeros(npts), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_output('EIyy',         val=np.zeros(npts), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_output('EIxy',         val=np.zeros(npts), units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_output('GJ',           val=np.zeros(npts), units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_output('rhoA',         val=np.zeros(npts), units='kg/m',   desc='mass per unit length')
        self.add_output('rhoJ',         val=np.zeros(npts), units='kg*m',   desc='polar mass moment of inertia per unit length')
        self.add_output('Tw_iner',      val=np.zeros(npts), units='m',      desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_output('x_ec',         val=np.zeros(npts), units='m',      desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_output('y_ec',         val=np.zeros(npts), units='m',      desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_output('flap_iner',    val=np.zeros(npts), units='kg/m',   desc='Section flap inertia about the Y_G axis per unit length.')
        self.add_output('edge_iner',    val=np.zeros(npts), units='kg/m',   desc='Section lag inertia about the X_G axis per unit length')
        self.add_output('eps_crit_spar',    val=np.zeros(npts), desc='critical strain in spar from panel buckling calculation')
        self.add_output('eps_crit_te',      val=np.zeros(npts), desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_output('xu_strain_spar',   val=np.zeros(npts), desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('xl_strain_spar',   val=np.zeros(npts), desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('yu_strain_spar',   val=np.zeros(npts), desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('yl_strain_spar',   val=np.zeros(npts), desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('xu_strain_te',     val=np.zeros(npts), desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('xl_strain_te',     val=np.zeros(npts), desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_output('yu_strain_te',     val=np.zeros(npts), desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('yl_strain_te',     val=np.zeros(npts), desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')

        # Blade Cost
        self.add_output('total_blade_cost', val=0.0, units='USD', desc='total blade cost')
        self.add_output('total_blade_mass', val=0.0, units='USD', desc='total blade cost')

        #
        self.add_discrete_output('blade_out', val={}, desc='updated blade dictionary for ontology')
        self.add_output('sparT_in_out', val=np.zeros(NINPUT), units='m', desc='thickness values of spar cap that linearly vary from non-cylinder position to tip, pass through for nested optimization')
        
        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # initialization point
        # if discrete_inputs['blade_in_overwrite'] != {}:
        #     blade = copy.deepcopy(discrete_inputs['blade_in_overwrite'])
        # else:
        blade = copy.deepcopy(self.refBlade)
        NINPUT = len(blade['ctrl_pts']['r_in'])

        # Set inputs to update blade geometry
        Rhub = inputs['hubFraction'] * inputs['bladeLength']
        Rtip = Rhub + inputs['bladeLength']

        outputs['Rhub']     = Rhub
        outputs['Rtip']     = Rtip
        
        r_in = blade['ctrl_pts']['r_in']
        
        outputs['r_in'] = Rhub + (Rtip-Rhub)*np.array(r_in)

        blade['ctrl_pts']['bladeLength']  = inputs['bladeLength']
        blade['ctrl_pts']['r_in']         = r_in
        blade['ctrl_pts']['chord_in']     = inputs['chord_in']
        blade['ctrl_pts']['theta_in']     = inputs['theta_in']
        blade['ctrl_pts']['precurve_in']  = inputs['precurve_in']
        blade['ctrl_pts']['presweep_in']  = inputs['presweep_in']
        blade['ctrl_pts']['sparT_in']     = inputs['sparT_in']
        blade['ctrl_pts']['teT_in']       = inputs['teT_in']
        # blade['ctrl_pts']['leT_in']       = inputs['leT_in']
        blade['ctrl_pts']['r_max_chord']  = inputs['r_max_chord']

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
        refBlade.verbose             = False
        refBlade.NINPUT              = len(outputs['r_in'])
        refBlade.NPTS                = len(blade['pf']['s'])
        refBlade.analysis_level      = blade['analysis_level']
        refBlade.user_update_routine = self.options['user_update_routine']
        if blade['analysis_level'] < 3:
            refBlade.spar_var        = blade['precomp']['spar_var']
            refBlade.te_var          = blade['precomp']['te_var']
            # if 'le_var' in blade['precomp']:
            #     refBlade.le_var     = blade['precomp']['le_var']
        
        blade_out = refBlade.update(blade)
        
        # Get geometric outputs
        outputs['hub_diameter'] = 2.0*Rhub
        outputs['r']            = Rhub + (Rtip-Rhub)*np.array(blade_out['pf']['s'])
        outputs['diameter']     = 2.0*outputs['r'][-1]

        outputs['chord']        = blade_out['pf']['chord']
        outputs['max_chord']    = max(blade_out['pf']['chord'])
        outputs['theta']        = blade_out['pf']['theta']
        outputs['precurve']     = blade_out['pf']['precurve']
        outputs['presweep']     = blade_out['pf']['presweep']
        outputs['rthick']       = blade_out['pf']['rthick']
        outputs['le_location']  = blade_out['pf']['p_le']

        # airfoils  = blade_out['airfoils']
        outputs['airfoils_cl']  = blade_out['airfoils_cl']
        outputs['airfoils_cd']  = blade_out['airfoils_cd']
        outputs['airfoils_cm']  = blade_out['airfoils_cm']
        outputs['airfoils_aoa'] = blade_out['airfoils_aoa']
        outputs['airfoils_Re']  = blade_out['airfoils_Re']
        
        
        
        upperCS   = blade_out['precomp']['upperCS']
        lowerCS   = blade_out['precomp']['lowerCS']
        websCS    = blade_out['precomp']['websCS']
        profile   = blade_out['precomp']['profile']
        materials = blade_out['precomp']['materials']
        
        for i in range(len(profile)):
            outputs['airfoils_coord_x'][:,i] = blade_out['profile'][:,0,i]
            outputs['airfoils_coord_y'][:,i] = blade_out['profile'][:,1,i]
        
        # Assumptions:
        # - if the composite layer is divided into multiple regions (i.e. if the spar cap is split into 3 regions due to the web locations),
        #   the middle region is selected with int(n_reg/2), note for an even number of regions, this rounds up
        sector_idx_strain_spar_ss = blade_out['precomp']['sector_idx_strain_spar_ss']
        sector_idx_strain_spar_ps = blade_out['precomp']['sector_idx_strain_spar_ps']
        sector_idx_strain_te_ss   = blade_out['precomp']['sector_idx_strain_te_ss']
        sector_idx_strain_te_ps   = blade_out['precomp']['sector_idx_strain_te_ps']


        # Get Beam Properties        
        beam = PreComp(outputs['r'], outputs['chord'], outputs['theta'], outputs['le_location'], 
                       outputs['precurve'], outputs['presweep'], profile, materials, upperCS, lowerCS, websCS, 
                       sector_idx_strain_spar_ps, sector_idx_strain_spar_ss, sector_idx_strain_te_ps, sector_idx_strain_te_ss)
        EIxx, EIyy, GJ, EA, EIxy, x_ec, y_ec, rhoA, rhoJ, Tw_iner, flap_iner, edge_iner = beam.sectionProperties()

        outputs['eps_crit_spar'] = beam.panelBucklingStrain(sector_idx_strain_spar_ss)
        outputs['eps_crit_te'] = beam.panelBucklingStrain(sector_idx_strain_te_ss)

        xu_strain_spar, xl_strain_spar, yu_strain_spar, yl_strain_spar = beam.criticalStrainLocations(sector_idx_strain_spar_ss, sector_idx_strain_spar_ps)
        xu_strain_te, xl_strain_te, yu_strain_te, yl_strain_te = beam.criticalStrainLocations(sector_idx_strain_te_ss, sector_idx_strain_te_ps)
        
        outputs['z']         = outputs['r']
        outputs['EIxx']      = EIxx
        outputs['EIyy']      = EIyy
        outputs['GJ']        = GJ
        outputs['EA']        = EA
        outputs['EIxy']      = EIxy
        outputs['x_ec']      = x_ec
        outputs['y_ec']      = y_ec
        outputs['rhoA']      = rhoA
        outputs['rhoJ']      = rhoJ
        outputs['Tw_iner']   = Tw_iner
        outputs['flap_iner'] = flap_iner
        outputs['edge_iner'] = edge_iner

        outputs['xu_strain_spar'] = xu_strain_spar
        outputs['xl_strain_spar'] = xl_strain_spar
        outputs['yu_strain_spar'] = yu_strain_spar
        outputs['yl_strain_spar'] = yl_strain_spar
        outputs['xu_strain_te']   = xu_strain_te
        outputs['xl_strain_te']   = xl_strain_te
        outputs['yu_strain_te']   = yu_strain_te
        outputs['yl_strain_te']   = yl_strain_te
        

        # Blade cost model
        bcm             = blade_cost_model(options=self.options) # <------------- options, import blade cost model
        bcm.name        = blade_out['config']['name']
        bcm.materials   = materials
        bcm.upperCS     = upperCS
        bcm.lowerCS     = lowerCS
        bcm.websCS      = websCS
        bcm.profile     = profile
        bcm.chord       = outputs['chord']
        bcm.r           = (outputs['r'] - outputs['Rhub'])/(outputs['Rtip'] - outputs['Rhub']) * float(inputs['bladeLength'])
        bcm.bladeLength = float(inputs['bladeLength'])
        bcm.le_location              = outputs['le_location']
        blade_cost, blade_mass       = bcm.execute_blade_cost_model()

        outputs['total_blade_cost'] = blade_cost
        outputs['total_blade_mass'] = blade_mass

        #
        discrete_outputs['blade_out'] = blade_out
        outputs['sparT_in_out'] = inputs['sparT_in']
        
class Location(ExplicitComponent):
    def setup(self):
        self.add_input('hub_height', val=0.0, units='m', desc='Tower top hub height')
        self.add_output('wind_zvec', val=np.zeros(1), units='m', desc='Tower top hub height as vector')
        #self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['wind_zvec'] = np.array([ np.float(inputs['hub_height']) ])
        '''

    def compute_partials(self, inputs, J):
        J['wind_zvec','hub_height'] = np.ones(1)
        '''


        
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
        
        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

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
        
        # Blade Cost Model Options
        self.options.declare('verbosity',        default=False)
        self.options.declare('tex_table',        default=False)
        self.options.declare('generate_plots',   default=False)
        self.options.declare('show_plots',       default=False)
        self.options.declare('show_warnings',    default=False)
        self.options.declare('discrete',         default=False)
        self.options.declare('user_update_routine', default=None)
        
    def setup(self):
        RefBlade = self.options['RefBlade']
        topLevelFlag   = self.options['topLevelFlag']
        NINPUT = len(RefBlade['ctrl_pts']['r_in'])
        NAF    = len(RefBlade['outer_shape_bem']['airfoil_position']['grid'])
        
        verbosity           = self.options['verbosity']
        tex_table           = self.options['tex_table']     
        generate_plots      = self.options['generate_plots']
        show_plots          = self.options['show_plots']
        show_warnings       = self.options['show_warnings']
        discrete            = self.options['discrete']
        user_update_routine = self.options['user_update_routine']
        
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
            # geomIndeps.add_output('precurveTip', 0.0, units='m')
            # geomIndeps.add_output('presweepTip', 0.0, units='m')
            geomIndeps.add_output('precone', 0.0, units='deg')
            geomIndeps.add_output('tilt', 0.0, units='deg')
            geomIndeps.add_output('yaw', 0.0, units='deg')
            geomIndeps.add_discrete_output('nBlades', 3)
            geomIndeps.add_discrete_output('downwind', False)
            geomIndeps.add_discrete_output('turbine_class', val='I', desc='IEC turbine class')
            geomIndeps.add_discrete_output('blade_in_overwrite', val={}, desc='IEC turbine class')
            geomIndeps.add_output('V_mean_overwrite', val=0.0, desc='optional overwrite value for mean velocity for using user defined CDFs')
            geomIndeps.add_output('airfoil_position', val=np.zeros(NAF))
            geomIndeps.add_output('sparT_in', val=np.zeros(NINPUT), units='m', desc='spar cap thickness parameters')
            geomIndeps.add_output('teT_in', val=np.zeros(NINPUT), units='m', desc='trailing-edge thickness parameters')
            # geomIndeps.add_output('leT_in', val=np.zeros(NINPUT), units='m', desc='leading-edge thickness parameters')
            self.add_subsystem('geomIndeps', geomIndeps, promotes=['*'])
            
        # --- Rotor Definition ---
        self.add_subsystem('loc', Location(), promotes=['*'])
        self.add_subsystem('turbineclass', TurbineClass(), promotes=['*'])
        #self.add_subsystem('spline0', BladeGeometry(RefBlade))
        self.add_subsystem('spline', BladeGeometry(RefBlade=RefBlade, 
                                               verbosity=verbosity,
                                               tex_table=tex_table,
                                               generate_plots=generate_plots,
                                               show_plots=show_plots,
                                               show_warnings =show_warnings ,
                                               discrete=discrete,
                                               user_update_routine=user_update_routine), promotes=['*'])
        self.add_subsystem('geom', CCBladeGeometry(NINPUT = NINPUT), promotes=['precone','precurve_in', 'presweep_in',
                                                                               'precurveTip','presweepTip','R','Rtip'])

        
def Init_RotorGeometry_wRefBlade(rotor, blade):
    rotor['precone']          = blade['config']['cone_angle']
    rotor['bladeLength']      = blade['ctrl_pts']['bladeLength'] #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    rotor['r_max_chord']      = blade['ctrl_pts']['r_max_chord']  # 0.23577 #(Float): location of max chord on unit radius
    rotor['chord_in']         = np.array(blade['ctrl_pts']['chord_in']) # np.array([3.2612, 4.3254, 4.5709, 3.7355, 2.69923333, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in']         = np.array(blade['ctrl_pts']['theta_in']) # np.array([0.0, 13.2783, 12.30514836,  6.95106536,  2.72696309, -0.0878099]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in']      = np.array(blade['ctrl_pts']['precurve_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in']      = np.array(blade['ctrl_pts']['presweep_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['sparT_in']         = np.array(blade['ctrl_pts']['sparT_in']) # np.array([0.0, 0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in']           = np.array(blade['ctrl_pts']['teT_in']) # np.array([0.0, 0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # if 'le_var' in blade['precomp']['le_var']:
    #     rotor['leT_in']       = np.array(blade['ctrl_pts']['leT_in']) ## (Array, m): leading-edge thickness parameters
    rotor['airfoil_position'] = np.array(blade['outer_shape_bem']['airfoil_position']['grid'])
    rotor['hubFraction']      = blade['config']['hubD']/2./blade['pf']['r'][-1] #0.025  # (Float): hub location as fraction of radius
    rotor['hub_height']       = blade['config']['hub_height']  # (Float, m): hub height
    rotor['turbine_class']    = blade['config']['turbine_class'].upper() #TURBINE_CLASS['I']  # (Enum): IEC turbine class

    return rotor


if __name__ == "__main__":

    # Turbine Ontology input
    # fname_input = "turbine_inputs/nrel5mw_mod_update.yaml"
    fname_input = "/mnt/c/Users/egaertne/WISDEM2/wisdem/Design_Opt/nrel15mw/inputs/NREL15MW_prelim_v3.0.yaml"

    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose  = True
    refBlade.NINPUT   = NINPUT
    refBlade.NPTS     = 50
    refBlade.validate = False
    refBlade.fname_schema = "turbine_inputs/IEAontology_schema.yaml"

    refBlade.spar_var = ['Spar_Cap_SS', 'Spar_Cap_PS']
    refBlade.te_var   = 'TE_reinforcement'
    # refBlade.le_var       = 'le_reinf'
    blade = refBlade.initialize(fname_input)

    # setup
    rotor = Problem()
    rotor.model = RotorGeometry(RefBlade=blade, topLevelFlag=True)
    rotor.setup()
    rotor = Init_RotorGeometry_wRefBlade(rotor, blade)
    rotor.run_driver()

    print(rotor['total_blade_cost'])
