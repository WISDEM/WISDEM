from wisdem.ccblade import CCAirfoil, CCBlade as CCBlade
from openmdao.api import ExplicitComponent
import numpy as np
import wisdem.ccblade._bem as _bem


cosd = lambda x: np.cos(np.deg2rad(x))
sind = lambda x: np.sin(np.deg2rad(x))

class CCBladeGeometry(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('NINPUT')
        
    def setup(self):
        NINPUT = self.options['NINPUT']
        
        self.add_input('Rtip', val=0.0, units='m', desc='tip radius')
        self.add_input('precurve_in', val=np.zeros(NINPUT), desc='prebend distribution', units='m')
        self.add_input('presweep_in', val=np.zeros(NINPUT), desc='presweep distribution', units='m')
        self.add_input('precone', val=0.0, desc='precone angle', units='deg')
        
        self.add_output('R', val=0.0, units='m', desc='rotor radius')
        self.add_output('diameter', val=0.0, units='m')
        self.add_output('precurveTip', val=0.0, units='m', desc='tip prebend')
        self.add_output('presweepTip', val=0.0, units='m', desc='tip sweep')
        
        self.declare_partials('R', ['Rtip', 'precone'])
        self.declare_partials('diameter', ['Rtip', 'precone'])
        
        self.declare_partials(['R', 'diameter'], 'precurve_in', rows=[0], cols=[NINPUT-1])
        
        self.declare_partials('precurveTip', 'precurve_in', val=1., rows=[0], cols=[NINPUT-1])
        self.declare_partials('presweepTip', 'presweep_in', val=1., rows=[0], cols=[NINPUT-1])
        
    def compute(self, inputs, outputs):
        Rtip           = inputs['Rtip']
        precone        = inputs['precone']
        
        outputs['precurveTip']  = inputs['precurve_in'][-1]
        outputs['presweepTip']  = inputs['presweep_in'][-1]
        
        outputs['R'] = Rtip * cosd(precone) + outputs['precurveTip'] * sind(precone)
        outputs['diameter'] = outputs['R'] * 2

    def compute_partials(self, inputs, J):
        Rtip           = inputs['Rtip']
        precone        = inputs['precone']
        precurveTip  = inputs['precurve_in'][-1]
        
        J['R', 'precurve_in'] = sind(precone)
        J['R', 'Rtip'] = cosd(precone)
        J['R', 'precone'] = (-Rtip*sind(precone) + precurveTip*cosd(precone))*np.pi/180.0
        
        J['diameter', 'precurve_in'] = 2.0 * J['R', 'precurve_in']
        J['diameter', 'Rtip'] = 2.0 * J['R', 'Rtip']
        J['diameter', 'precone'] = 2.0 * J['R', 'precone']
        

class CCBladePower(ExplicitComponent):
    def initialize(self):
        self.options.declare('naero')
        self.options.declare('npower')

        self.options.declare('n_aoa_grid')
        self.options.declare('n_Re_grid')

        
    def setup(self):
        self.naero = naero = self.options['naero']
        npower = self.options['npower']
        n_aoa_grid = self.options['n_aoa_grid']
        n_Re_grid  = self.options['n_Re_grid']

        """blade element momentum code"""

        # inputs
        self.add_input('Uhub', val=np.zeros(npower), units='m/s', desc='hub height wind speed')
        self.add_input('Omega', val=np.zeros(npower), units='rpm', desc='rotor rotation speed')
        self.add_input('pitch', val=np.zeros(npower), units='deg', desc='blade pitch setting')

        # outputs
        self.add_output('T', val=np.zeros(npower), units='N', desc='rotor aerodynamic thrust')
        self.add_output('Q', val=np.zeros(npower), units='N*m', desc='rotor aerodynamic torque')
        self.add_output('P', val=np.zeros(npower), units='W', desc='rotor aerodynamic power')

        
        # (potential) variables
        self.add_input('r', val=np.zeros(naero), units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord', val=np.zeros(naero), units='m', desc='chord length at each section')
        self.add_input('theta', val=np.zeros(naero),  units='deg', desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub', val=0.0, units='m', desc='hub radius')
        self.add_input('Rtip', val=0.0, units='m', desc='tip radius')
        self.add_input('hub_height', val=0.0, units='m', desc='hub height')
        self.add_input('precone', val=0.0, desc='precone angle', units='deg')
        self.add_input('tilt', val=0.0, desc='shaft tilt', units='deg')
        self.add_input('yaw', val=0.0, desc='yaw error', units='deg')

        # TODO: I've not hooked up the gradients for these ones yet.
        self.add_input('precurve', val=np.zeros(naero), units='m', desc='precurve at each section')
        self.add_input('precurveTip', val=0.0, units='m', desc='precurve at tip')

        # parameters
        self.add_input('airfoils_cl', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa', val=np.zeros((n_aoa_grid)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re', val=np.zeros((n_Re_grid)), desc='Reynolds numbers of polars')
        # self.add_discrete_input('airfoils', val=[0]*naero, desc='CCAirfoil instances')
        self.add_discrete_input('nBlades', val=0, desc='number of blades')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='density of air')
        self.add_input('mu', val=0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_input('shearExp', val=0.0, desc='shear exponent')
        self.add_discrete_input('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss', val=True, desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss', val=True, desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd', val=True, desc='use drag coefficient in computing induction factors')

        self.declare_partials(['P', 'T', 'Q'],['precone', 'tilt', 'hub_height', 'Rhub', 'Rtip', 'yaw',
                                               'Uhub', 'Omega', 'pitch', 'r', 'chord', 'theta',
                                               'precurve', 'precurveTip'])

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs['r']
        chord = inputs['chord']
        theta = inputs['theta']
        Rhub = inputs['Rhub']
        Rtip = inputs['Rtip']
        hub_height = inputs['hub_height']
        precone = inputs['precone']
        tilt = inputs['tilt']
        yaw = inputs['yaw']
        precurve = inputs['precurve']
        precurveTip = inputs['precurveTip']
        # airfoils = discrete_inputs['airfoils']
        B = discrete_inputs['nBlades']
        rho = inputs['rho']
        mu = inputs['mu']
        shearExp = inputs['shearExp']
        nSector = discrete_inputs['nSector']
        tiploss = discrete_inputs['tiploss']
        hubloss = discrete_inputs['hubloss']
        wakerotation = discrete_inputs['wakerotation']
        usecd = discrete_inputs['usecd']
        Uhub = inputs['Uhub']
        Omega = inputs['Omega']
        pitch = inputs['pitch']

        af = [None]*self.naero
        for i in range(self.naero):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][:,i,:], inputs['airfoils_cd'][:,i,:], inputs['airfoils_cm'][:,i,:])
        
        ccblade = CCBlade(r, chord, theta, af, Rhub, Rtip, B,
            rho, mu, precone, tilt, yaw, shearExp, hub_height,
            nSector, precurve, precurveTip, tiploss=tiploss, hubloss=hubloss,
            wakerotation=wakerotation, usecd=usecd, derivatives=True)

        # power, thrust, torque
        myoutputs, self.derivs = ccblade.evaluate(Uhub, Omega, pitch, coefficients=False)
        outputs['T'] = myoutputs['T']
        outputs['Q'] = myoutputs['Q']
        outputs['P'] = myoutputs['P']
        

    def compute_partials(self, inputs, J, discrete_inputs):

        dP = self.derivs['dP']
        dT = self.derivs['dT']
        dQ = self.derivs['dQ']

        J['P', 'precone']     = dP['dprecone']
        J['P', 'tilt']        = dP['dtilt']
        J['P', 'hub_height']  = dP['dhubHt']
        J['P', 'Rhub']        = dP['dRhub']
        J['P', 'Rtip']        = dP['dRtip']
        J['P', 'yaw']         = dP['dyaw']
        J['P', 'Uhub']        = dP['dUinf']
        J['P', 'Omega']       = dP['dOmega']
        J['P', 'pitch']       = dP['dpitch']
        J['P', 'r']           = dP['dr']
        J['P', 'chord']       = dP['dchord']
        J['P', 'theta']       = dP['dtheta']
        J['P', 'precurve']    = dP['dprecurve']
        J['P', 'precurveTip'] = dP['dprecurveTip']

        J['T', 'precone']     = dT['dprecone']
        J['T', 'tilt']        = dT['dtilt']
        J['T', 'hub_height']  = dT['dhubHt']
        J['T', 'Rhub']        = dT['dRhub']
        J['T', 'Rtip']        = dT['dRtip']
        J['T', 'yaw']         = dT['dyaw']
        J['T', 'Uhub']        = dT['dUinf']
        J['T', 'Omega']       = dT['dOmega']
        J['T', 'pitch']       = dT['dpitch']
        J['T', 'r']           = dT['dr']
        J['T', 'chord']       = dT['dchord']
        J['T', 'theta']       = dT['dtheta']
        J['T', 'precurve']    = dT['dprecurve']
        J['T', 'precurveTip'] = dT['dprecurveTip']

        J['Q', 'precone']     = dQ['dprecone']
        J['Q', 'tilt']        = dQ['dtilt']
        J['Q', 'hub_height']  = dQ['dhubHt']
        J['Q', 'Rhub']        = dQ['dRhub']
        J['Q', 'Rtip']        = dQ['dRtip']
        J['Q', 'yaw']         = dQ['dyaw']
        J['Q', 'Uhub']        = dQ['dUinf']
        J['Q', 'Omega']       = dQ['dOmega']
        J['Q', 'pitch']       = dQ['dpitch']
        J['Q', 'r']           = dQ['dr']
        J['Q', 'chord']       = dQ['dchord']
        J['Q', 'theta']       = dQ['dtheta']
        J['Q', 'precurve']    = dQ['dprecurve']
        J['Q', 'precurveTip'] = dQ['dprecurveTip']

        


    
class CCBladeLoads(ExplicitComponent):
    # OpenMDAO component that given a rotor speed, pitch angle, and wind speed, computes the aerodynamic forces along the blade span
    def initialize(self):
        self.options.declare('analysis_options')
        
    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
        self.n_span        = n_span    = blade_init_options['n_span']
        af_init_options = self.options['analysis_options']['airfoils']
        self.n_aoa         = n_aoa     = af_init_options['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = af_init_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = af_init_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1

        # inputs
        self.add_input('V_load',        val=0.0, units='m/s', desc='hub height wind speed')
        self.add_input('Omega_load',    val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_input('pitch_load',    val=0.0, units='deg', desc='blade pitch setting')
        self.add_input('azimuth_load',  val=0.0, units='deg', desc='blade azimuthal location')

        # (potential) variables
        self.add_input('r',             val=np.zeros(n_span),   units='m',      desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',         val=np.zeros(n_span),   units='m',      desc='chord length at each section')
        self.add_input('theta',         val=np.zeros(n_span),   units='deg',    desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',          val=0.0,                units='m',      desc='hub radius')
        self.add_input('Rtip',          val=0.0,                units='m',      desc='tip radius')
        self.add_input('hub_height',    val=0.0,                units='m',      desc='hub height')
        self.add_input('precone',       val=0.0,                units='deg',    desc='precone angle')
        self.add_input('tilt',          val=0.0,                units='deg',    desc='shaft tilt')
        self.add_input('yaw',           val=0.0,                units='deg',    desc='yaw error')
        self.add_input('precurve',      val=np.zeros(n_span),   units='m', desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                units='m', desc='precurve at tip')

        # parameters
        # self.add_discrete_input('airfoils', val=[0]*naero, desc='CCAirfoil instances')
        self.add_input('airfoils_cl',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',  val=np.zeros((n_aoa)), units='deg',         desc='angle of attack grid for polars')
        self.add_input('airfoils_Re',   val=np.zeros((n_Re)),                       desc='Reynolds numbers of polars')

        self.add_discrete_input('nBlades',  val=0,                      desc='number of blades')
        self.add_input('rho',               val=0.0, units='kg/m**3',   desc='density of air')
        self.add_input('mu',                val=0.0, units='kg/(m*s)',  desc='dynamic viscosity of air')
        self.add_input('shearExp',          val=0.0,                    desc='shear exponent')
        self.add_discrete_input('nSector',  val=4,                      desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',  val=True,                   desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',  val=True,                   desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True,               desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',    val=True,                   desc='use drag coefficient in computing induction factors')

        # outputs
        self.add_output('loads_r',      val=np.zeros(n_span), units='m',    desc='radial positions along blade going toward tip')
        self.add_output('loads_Px',     val=np.zeros(n_span), units='N/m',  desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py',     val=np.zeros(n_span), units='N/m',  desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz',     val=np.zeros(n_span), units='N/m',  desc='distributed loads in blade-aligned z-direction')

        arange = np.arange(n_span)
        self.declare_partials('loads_Px', ['Omega_load', 'Rhub', 'Rtip', 'V_load', 'azimuth_load', 'chord', 'hub_height', 'pitch_load', 'precone', 'precurve', 'r', 'theta', 'tilt', 'yaw'])
        self.declare_partials('loads_Py', ['Omega_load', 'Rhub', 'Rtip', 'V_load', 'azimuth_load', 'chord', 'hub_height', 'pitch_load', 'precone', 'precurve', 'r', 'theta', 'tilt', 'yaw'])
        self.declare_partials('loads_Pz', '*', dependent=False)
        self.declare_partials('loads_r', 'r', val=1., rows=arange, cols=arange)
        self.declare_partials('*', 'airfoils*', dependent=False)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs['r']
        chord = inputs['chord']
        theta = inputs['theta']
        Rhub = inputs['Rhub']
        Rtip = inputs['Rtip']
        hub_height = inputs['hub_height']
        precone = inputs['precone']
        tilt = inputs['tilt']
        yaw = inputs['yaw']
        precurve = inputs['precurve']
        precurveTip = inputs['precurveTip']
        B = discrete_inputs['nBlades']
        rho = inputs['rho']
        mu = inputs['mu']
        shearExp = inputs['shearExp']
        nSector = discrete_inputs['nSector']
        tiploss = discrete_inputs['tiploss']
        hubloss = discrete_inputs['hubloss']
        wakerotation = discrete_inputs['wakerotation']
        usecd = discrete_inputs['usecd']
        V_load = inputs['V_load']
        Omega_load = inputs['Omega_load']
        pitch_load = inputs['pitch_load']
        azimuth_load = inputs['azimuth_load']
        
        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None]*self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])

        ccblade = CCBlade(r, chord, theta, af, Rhub, Rtip, B,
            rho, mu, precone, tilt, yaw, shearExp, hub_height,
            nSector, precurve, precurveTip, tiploss=tiploss, hubloss=hubloss,
            wakerotation=wakerotation, usecd=usecd, derivatives=True)

        # distributed loads
        loads, self.derivs = ccblade.distributedAeroLoads(V_load, Omega_load, pitch_load, azimuth_load)
        Np = loads['Np']
        Tp = loads['Tp']
        
        # unclear why we need this output at all
        outputs['loads_r'] = r

        # conform to blade-aligned coordinate system
        outputs['loads_Px'] = Np
        outputs['loads_Py'] = -Tp
        outputs['loads_Pz'][:] = 0.

    def compute_partials(self, inputs, J, discrete_inputs):
        dNp = self.derivs['dNp']
        dTp = self.derivs['dTp']
    
        J['loads_Px',     'r']             = dNp['dr']
        J['loads_Px',     'chord']         = dNp['dchord']
        J['loads_Px',     'theta']         = dNp['dtheta']
        J['loads_Px',     'Rhub']          = np.squeeze(dNp['dRhub'])
        J['loads_Px',     'Rtip']          = np.squeeze(dNp['dRtip'])
        J['loads_Px',     'hub_height']    = np.squeeze(dNp['dhubHt'])
        J['loads_Px',     'precone']       = np.squeeze(dNp['dprecone'])
        J['loads_Px',     'tilt']          = np.squeeze(dNp['dtilt'])
        J['loads_Px',     'yaw']           = np.squeeze(dNp['dyaw'])
        J['loads_Px',     'V_load']        = np.squeeze(dNp['dUinf'])
        J['loads_Px',     'Omega_load']    = np.squeeze(dNp['dOmega'])
        J['loads_Px',     'pitch_load']    = np.squeeze(dNp['dpitch'])
        J['loads_Px',     'azimuth_load']  = np.squeeze(dNp['dazimuth'])
        J['loads_Px',     'precurve']      = dNp['dprecurve']
    
        J['loads_Py',     'r']             = -dTp['dr']
        J['loads_Py',     'chord']         = -dTp['dchord']
        J['loads_Py',     'theta']         = -dTp['dtheta']
        J['loads_Py',     'Rhub']          = -np.squeeze(dTp['dRhub'])
        J['loads_Py',     'Rtip']          = -np.squeeze(dTp['dRtip'])
        J['loads_Py',     'hub_height']    = -np.squeeze(dTp['dhubHt'])
        J['loads_Py',     'precone']       = -np.squeeze(dTp['dprecone'])
        J['loads_Py',     'tilt']          = -np.squeeze(dTp['dtilt'])
        J['loads_Py',     'yaw']           = -np.squeeze(dTp['dyaw'])
        J['loads_Py',     'V_load']        = -np.squeeze(dTp['dUinf'])
        J['loads_Py',     'Omega_load']    = -np.squeeze(dTp['dOmega'])
        J['loads_Py',     'pitch_load']    = -np.squeeze(dTp['dpitch'])
        J['loads_Py',     'azimuth_load']  = -np.squeeze(dTp['dazimuth'])
        J['loads_Py',     'precurve']      = -dTp['dprecurve']
    

class AeroHubLoads(ExplicitComponent):
    # OpenMDAO component that computes the aerodynamic loading at hub center
    
    def initialize(self):
        self.options.declare('analysis_options')

    def setup(self):
        
        blade_init_options = self.options['analysis_options']['blade']
        self.n_span        = n_span    = blade_init_options['n_span']
        af_init_options = self.options['analysis_options']['airfoils']
        self.n_aoa         = n_aoa     = af_init_options['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = af_init_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = af_init_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1

        # inputs
        self.add_input('V_load',        val=0.0, units='m/s', desc='hub height wind speed')
        self.add_input('Omega_load',    val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_input('pitch_load',    val=0.0, units='deg', desc='blade pitch setting')

        # (potential) variables
        self.add_input('r',             val=np.zeros(n_span),   units='m',      desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',         val=np.zeros(n_span),   units='m',      desc='chord length at each section')
        self.add_input('theta',         val=np.zeros(n_span),   units='deg',    desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',          val=0.0,                units='m',      desc='hub radius')
        self.add_input('Rtip',          val=0.0,                units='m',      desc='tip radius')
        self.add_input('hub_height',    val=0.0,                units='m',      desc='hub height')
        self.add_input('precone',       val=0.0,                units='deg',    desc='precone angle')
        self.add_input('tilt',          val=0.0,                units='deg',    desc='shaft tilt')
        self.add_input('yaw',           val=0.0,                units='deg',    desc='yaw error')
        self.add_input('precurve',      val=np.zeros(n_span),   units='m', desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                units='m', desc='precurve at tip')

        # parameters
        # self.add_discrete_input('airfoils', val=[0]*naero, desc='CCAirfoil instances')
        self.add_input('airfoils_cl',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',  val=np.zeros((n_aoa)), units='deg',         desc='angle of attack grid for polars')
        self.add_input('airfoils_Re',   val=np.zeros((n_Re)),                       desc='Reynolds numbers of polars')

        self.add_discrete_input('nBlades',  val=0,                      desc='number of blades')
        self.add_input('rho',               val=0.0, units='kg/m**3',   desc='density of air')
        self.add_input('mu',                val=0.0, units='kg/(m*s)',  desc='dynamic viscosity of air')
        self.add_input('shearExp',          val=0.0,                    desc='shear exponent')
        self.add_discrete_input('tiploss',  val=True,                   desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',  val=True,                   desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True,               desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',    val=True,                   desc='use drag coefficient in computing induction factors')

        # outputs
        self.add_output('Fxyz_blade_aero',  val=np.zeros((3,6)),  units='N',   desc='Forces at blade root from aerodynamic loading in the blade c.s.')
        self.add_output('Mxyz_blade_aero',  val=np.zeros((3,6)),  units='N*m', desc='Moments at blade root from aerodynamic loading in the blade c.s.')
        self.add_output('Fxyz_hub_aero',    val=np.zeros(3),      units='N',   desc='Forces at hub center from aerodynamic loading in the hub c.s.')
        self.add_output('Mxyz_hub_aero',    val=np.zeros(3),      units='N*m', desc='Moments at hub center from aerodynamic loading in the hub c.s.')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs['r']
        chord = inputs['chord']
        theta = inputs['theta']
        Rhub = inputs['Rhub']
        Rtip = inputs['Rtip']
        hub_height = inputs['hub_height']
        precone = inputs['precone']
        tilt = inputs['tilt']
        yaw = inputs['yaw']
        precurve = inputs['precurve']
        precurveTip = inputs['precurveTip']
        B = discrete_inputs['nBlades']
        rho = inputs['rho']
        mu = inputs['mu']
        shearExp = inputs['shearExp']
        nSector = 1
        tiploss = discrete_inputs['tiploss']
        hubloss = discrete_inputs['hubloss']
        wakerotation = discrete_inputs['wakerotation']
        usecd = discrete_inputs['usecd']
        V_load = inputs['V_load']
        Omega_load = inputs['Omega_load']
        pitch_load = inputs['pitch_load']
        from wisdem.commonse.csystem import DirectionVector

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        presweep = np.zeros_like(r)

        n = len(r)
        dx_dx = np.eye(3*n)

        x_az, x_azd, y_az, y_azd, z_az, z_azd, cone, coned, s, sd = _bem.definecurvature_dv2(r, dx_dx[:, :n],
                                                                                             precurve, dx_dx[:, n:2*n],
                                                                                             presweep, dx_dx[:, 2*n:],
                                                                                             0.0, np.zeros(3*n))

        totalCone = precone + np.degrees(cone)
        s = r[0] + s

        af = [None]*self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])

        ccblade = CCBlade(r, chord, theta, af, Rhub, Rtip, B,
            rho, mu, precone, tilt, yaw, shearExp, hub_height,
            nSector, precurve, precurveTip, tiploss=tiploss, hubloss=hubloss,
            wakerotation=wakerotation, usecd=usecd, derivatives=True)

        azimuth_blades  = np.linspace(0, 360, B+1)

        # distributed loads
        for i_blade in range(B):
            loads, self.derivs = ccblade.distributedAeroLoads(V_load, Omega_load, pitch_load, azimuth_blades[i_blade])
            Np = loads['Np']
            Tp = loads['Tp']
            
            # conform to blade-aligned coordinate system
            Px = Np
            Py = -Tp
            Pz = 0.*Np

            # Integrate to get shear forces
            Fx = np.trapz(Px, r)
            Fy = np.trapz(Py, r)
            Fz = np.trapz(Pz, r)
            Fxy= np.sqrt(Fx**2 + Fy**2)
            Fyz= np.sqrt(Fy**2 + Fz**2)
            Fxz= np.sqrt(Fx**2 + Fz**2)

            # loads in azimuthal c.s.
            P = DirectionVector(Px, Py, Pz).bladeToAzimuth(totalCone)

            # distributed bending load in azimuth coordinate ysstem
            az = DirectionVector(x_az, y_az, z_az)
            Mp = az.cross(P)

            # Integrate to obtain moments
            Mx = np.trapz(Mp.x, r)
            My = np.trapz(Mp.y, r)
            Mz = np.trapz(Mp.z, r)
            Mxy= np.sqrt(Mx**2 + My**2)
            Myz= np.sqrt(My**2 + Mz**2)
            Mxz= np.sqrt(Mx**2 + Mz**2)

            outputs['Fxyz_blade_aero'][i_blade, :] = np.array([Fx, Fy, Fz, Fxy, Fyz, Fxz])
            outputs['Mxyz_blade_aero'][i_blade, :] = np.array([Mx, My, Mz, Mxy, Myz, Mxz])
        
        
        # Initialize summation
        F_hub_tot = np.zeros((3,))
        M_hub_tot = np.zeros((3,))
        # Convert from blade to hub c.s.
        for i_blade in range(B):
            myF = DirectionVector.fromArray(outputs['Fxyz_blade_aero'][i_blade,:]).azimuthToHub(azimuth_blades[i_blade])
            myM = DirectionVector.fromArray(outputs['Mxyz_blade_aero'][i_blade,:]).azimuthToHub(azimuth_blades[i_blade])
            
            F_hub_tot += myF.toArray()
            M_hub_tot += myM.toArray()

        # Now sum over all blades
        outputs['Fxyz_hub_aero'] = F_hub_tot
        outputs['Mxyz_hub_aero'] = M_hub_tot


