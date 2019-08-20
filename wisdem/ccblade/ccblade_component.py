from wisdem.ccblade import CCAirfoil, CCBlade as CCBlade
from openmdao.api import ExplicitComponent
import numpy as np

cosd = lambda x: np.cos(np.deg2rad(x))
sind = lambda x: np.sin(np.deg2rad(x))

class CCBladeGeometry(ExplicitComponent):
    def setup(self):
        self.add_input('Rtip', val=0.0, units='m', desc='tip radius')
        self.add_input('precurveTip', val=0.0, units='m', desc='tip radius')
        self.add_input('precone', val=0.0, desc='precone angle', units='deg')
        self.add_output('R', val=0.0, units='m', desc='rotor radius')
        self.add_output('diameter', val=0.0, units='m')

        self.declare_partials('R', '*')
        self.declare_partials('diameter', '*')
        self.declare_partials('diameter', 'R')
        
    def compute(self, inputs, outputs):

        self.Rtip = inputs['Rtip']
        self.precurveTip = inputs['precurveTip']
        self.precone = inputs['precone']

        self.R = self.Rtip*cosd(self.precone) + self.precurveTip*sind(self.precone)

        outputs['R'] = self.R
        outputs['diameter'] = self.R*2

    def compute_partials(self, inputs, J):

        J_sub = np.array([[cosd(self.precone), sind(self.precone),
            (-self.Rtip*sind(self.precone) + self.precurveTip*sind(self.precone))*np.pi/180.0]])

        J['R', 'Rtip'] = J_sub[0][0]
        J['R', 'precurveTip'] = J_sub[0][1]
        J['R', 'precone'] = J_sub[0][2]
        J['diameter', 'Rtip'] = 2.0*J_sub[0][0]
        J['diameter', 'precurveTip'] = 2.0*J_sub[0][1]
        J['diameter', 'precone'] = 2.0*J_sub[0][2]
        J['diameter', 'R'] = 2.0

        

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

        self.r = inputs['r']
        self.chord = inputs['chord']
        self.theta = inputs['theta']
        self.Rhub = inputs['Rhub']
        self.Rtip = inputs['Rtip']
        self.hub_height = inputs['hub_height']
        self.precone = inputs['precone']
        self.tilt = inputs['tilt']
        self.yaw = inputs['yaw']
        self.precurve = inputs['precurve']
        self.precurveTip = inputs['precurveTip']
        # self.airfoils = discrete_inputs['airfoils']
        self.B = discrete_inputs['nBlades']
        self.rho = inputs['rho']
        self.mu = inputs['mu']
        self.shearExp = inputs['shearExp']
        self.nSector = discrete_inputs['nSector']
        self.tiploss = discrete_inputs['tiploss']
        self.hubloss = discrete_inputs['hubloss']
        self.wakerotation = discrete_inputs['wakerotation']
        self.usecd = discrete_inputs['usecd']
        self.Uhub = inputs['Uhub']
        self.Omega = inputs['Omega']
        self.pitch = inputs['pitch']

        af = [None]*self.naero
        for i in range(self.naero):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][:,i,:], inputs['airfoils_cd'][:,i,:], inputs['airfoils_cm'][:,i,:])
        
        self.ccblade = CCBlade(self.r, self.chord, self.theta, af, self.Rhub, self.Rtip, self.B,
            self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hub_height,
            self.nSector, self.precurve, self.precurveTip, tiploss=self.tiploss, hubloss=self.hubloss,
            wakerotation=self.wakerotation, usecd=self.usecd, derivatives=True)

        # power, thrust, torque
        self.P, self.T, self.Q, self.M, self.dP, self.dT, self.dQ \
            = self.ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficients=False)
        outputs['T'] = self.T
        outputs['Q'] = self.Q
        outputs['P'] = self.P

    def compute_partials(self, inputs, J, discrete_inputs):

        dP = self.dP
        dT = self.dT
        dQ = self.dQ
        
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
    def initialize(self):
        self.options.declare('naero')
        self.options.declare('npower')

        self.options.declare('n_aoa_grid')
        self.options.declare('n_Re_grid')
        
    def setup(self):
        self.naero = naero = self.options['naero']
        npower     = self.options['npower']
        n_aoa_grid = self.options['n_aoa_grid']
        n_Re_grid  = self.options['n_Re_grid']
        """blade element momentum code"""

        # inputs
        self.add_input('V_load', val=0.0, units='m/s', desc='hub height wind speed')
        self.add_input('Omega_load', val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_input('pitch_load', val=0.0, units='deg', desc='blade pitch setting')
        self.add_input('azimuth_load', val=0.0, units='deg', desc='blade azimuthal location')

        # outputs
        self.add_output('loads_r', val=np.zeros(naero), units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads_Px', val=np.zeros(naero), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py', val=np.zeros(naero), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz', val=np.zeros(naero), units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_output('loads_V', val=0.0, units='m/s', desc='hub height wind speed')
        self.add_output('loads_Omega', val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_output('loads_pitch', val=0.0, units='deg', desc='pitch angle')
        self.add_output('loads_azimuth', val=0.0, units='deg', desc='azimuthal angle')
        
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
        # self.add_discrete_input('airfoils', val=[0]*naero, desc='CCAirfoil instances')
        self.add_input('airfoils_cl', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa', val=np.zeros((n_aoa_grid)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re', val=np.zeros((n_Re_grid)), desc='Reynolds numbers of polars')

        self.add_discrete_input('nBlades', val=0, desc='number of blades')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='density of air')
        self.add_input('mu', val=0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_input('shearExp', val=0.0, desc='shear exponent')
        self.add_discrete_input('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss', val=True, desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss', val=True, desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd', val=True, desc='use drag coefficient in computing induction factors')

        self.declare_partials('loads_r', ['r', 'Rhub', 'Rtip'])
        self.declare_partials(['loads_Px', 'loads_Py'],
                              ['r', 'chord', 'theta', 'Rhub', 'Rtip', 'hub_height', 'precone', 'tilt',
                               'yaw', 'V_load', 'Omega_load', 'pitch_load', 'azimuth_load', 'precurve'])
        self.declare_partials('loads_V', 'V_load')
        self.declare_partials('loads_Omega', 'Omega_load')
        self.declare_partials('loads_pitch', 'pitch_load')
        self.declare_partials('loads_azimuth', 'azimuth_load')

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        self.r = inputs['r']
        self.chord = inputs['chord']
        self.theta = inputs['theta']
        self.Rhub = inputs['Rhub']
        self.Rtip = inputs['Rtip']
        self.hub_height = inputs['hub_height']
        self.precone = inputs['precone']
        self.tilt = inputs['tilt']
        self.yaw = inputs['yaw']
        self.precurve = inputs['precurve']
        self.precurveTip = inputs['precurveTip']
        # self.airfoils = discrete_inputs['airfoils']
        self.B = discrete_inputs['nBlades']
        self.rho = inputs['rho']
        self.mu = inputs['mu']
        self.shearExp = inputs['shearExp']
        self.nSector = discrete_inputs['nSector']
        self.tiploss = discrete_inputs['tiploss']
        self.hubloss = discrete_inputs['hubloss']
        self.wakerotation = discrete_inputs['wakerotation']
        self.usecd = discrete_inputs['usecd']
        self.V_load = inputs['V_load']
        self.Omega_load = inputs['Omega_load']
        self.pitch_load = inputs['pitch_load']
        self.azimuth_load = inputs['azimuth_load']


        if len(self.precurve) == 0:
            self.precurve = np.zeros_like(self.r)

        # airfoil files
        # n = len(self.airfoils)
        af = [None]*self.naero
        for i in range(self.naero):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][:,i,:], inputs['airfoils_cd'][:,i,:], inputs['airfoils_cm'][:,i,:])
        # af = self.airfoils

        self.ccblade = CCBlade(self.r, self.chord, self.theta, af, self.Rhub, self.Rtip, self.B,
            self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hub_height,
            self.nSector, self.precurve, self.precurveTip, tiploss=self.tiploss, hubloss=self.hubloss,
            wakerotation=self.wakerotation, usecd=self.usecd, derivatives=True)

        # distributed loads
        Np, Tp, self.dNp, self.dTp \
            = self.ccblade.distributedAeroLoads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)
        
        
        # concatenate loads at root/tip
        outputs['loads_r'] = self.r

        # conform to blade-aligned coordinate system
        outputs['loads_Px'] = Np
        outputs['loads_Py'] = -Tp
        outputs['loads_Pz'] = 0*Np

        # return other outputs needed
        outputs['loads_V'] = self.V_load
        outputs['loads_Omega'] = self.Omega_load
        outputs['loads_pitch'] = self.pitch_load
        outputs['loads_azimuth'] = self.azimuth_load

    def compute_partials(self, inputs, J, discrete_inputs=None):

        dNp = self.dNp
        dTp = self.dTp
        n = len(self.r)

        dr_dr = np.eye(n)
        dr_dRhub = np.zeros(n)
        dr_dRtip = np.zeros(n)
        dr_dRhub[0] = 1.0
        dr_dRtip[-1] = 1.0

        dV = np.zeros(4*n+10)
        dV[3*n+6] = 1.0
        dOmega = np.zeros(4*n+10)
        dOmega[3*n+7] = 1.0
        dpitch = np.zeros(4*n+10)
        dpitch[3*n+8] = 1.0
        dazimuth = np.zeros(4*n+10)
        dazimuth[3*n+9] = 1.0

        zero = np.zeros(self.naero)
        J['loads_r',      'r']             = dr_dr
        J['loads_r',      'Rhub']          = dr_dRhub
        J['loads_r',      'Rtip']          = dr_dRtip
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
        J['loads_V',      'V_load']        = 1.0
        J['loads_Omega',  'Omega_load']    = 1.0
        J['loads_pitch',  'pitch_load']    = 1.0
        J['loads_azimuth', 'azimuth_load'] = 1.0

        
    
'''
def common_io(group, varspeed, varpitch):

    regulated = varspeed or varpitch

    # add inputs
    group.add_input('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at')
    group.add_input('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve')
    group.add_input('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)')
    if varspeed:
        group.add_input('control:Vin', units='m/s', desc='cut-in wind speed')
        group.add_input('control:Vout', units='m/s', desc='cut-out wind speed')
        group.add_input('control:ratedPower', units='W', desc='rated power')
        group.add_input('control:minOmega', units='rpm', desc='minimum allowed rotor rotation speed')
        group.add_input('control:maxOmega', units='rpm', desc='maximum allowed rotor rotation speed')
        group.add_input('control:tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        group.add_input('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
    else:
        group.add_input('control:Vin', units='m/s', desc='cut-in wind speed')
        group.add_input('control:Vout', units='m/s', desc='cut-out wind speed')
        group.add_input('control:ratedPower', units='W', desc='rated power')
        group.add_input('control:Omega', units='rpm', desc='fixed rotor rotation speed')
        group.add_input('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        group.add_input('control:npts', val=20, desc='number of points to evalute aero code to generate power curve')


    # # add slots (must replace)
    # group.add('geom', Slot(GeomtrySetupBase))
    # group.add('analysis', Slot(AeroBase))
    # group.add('dt', Slot(DrivetrainLossesBase))
    # group.add('cdf', Slot(CDFBase))


    # add outputs
    group.add_output('AEP', units='kW*h', desc='annual energy production')
    group.add_output('V', units='m/s', desc='wind speeds (power curve)')
    group.add_output('P', units='W', desc='power (power curve)')
    group.add_output('diameter', units='m', desc='rotor diameter')
    if regulated:
        group.add_output('ratedConditions:V', units='m/s', desc='rated wind speed')
        group.add_output('ratedConditions:Omega', units='rpm', desc='rotor rotation speed at rated')
        group.add_output('ratedConditions:pitch', units='deg', desc='pitch setting at rated')
        group.add_output('ratedConditions:T', units='N', desc='rotor aerodynamic thrust at rated')
        group.add_output('ratedConditions:Q', units='N*m', desc='rotor aerodynamic torque at rated')


def common_configure(group, varspeed, varpitch):

    regulated = varspeed or varpitch

    # add components
    group.add('geom', GeomtrySetupBase())

    if varspeed:
        group.add('setup', SetupRunVarSpeed(20))
    else:
        group.add('setup', SetupRunFixedSpeed())

    group.add('analysis', AeroBase())
    group.add('dt', DrivetrainLossesBase())

    if varspeed or varpitch:
        group.add('powercurve', RegulatedPowerCurve(20))
        group.add('brent', Brent())
        group.brent.workflow.add(['powercurve'])
    else:
        group.add('powercurve', UnregulatedPowerCurve())

    group.add('cdf', CDFBase())
    group.add('aep', AEP(200))

    if regulated:
        group.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'brent', 'cdf', 'aep'])
    else:
        group.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'powercurve', 'cdf', 'aep'])


    # connections to setup
    group.connect('control', 'setup.control')
    group.connect('npts_coarse_power_curve', 'setup.npts')
    if varspeed:
        group.connect('geom.R', 'setup.R')


    # connections to analysis
    group.connect('setup.Uhub', 'analysis.Uhub')
    group.connect('setup.Omega', 'analysis.Omega')
    group.connect('setup.pitch', 'analysis.pitch')
    group.analysis.run_case = 'power'


    # connections to drivetrain
    group.connect('analysis.P', 'dt.aeroPower')
    group.connect('analysis.Q', 'dt.aeroTorque')
    group.connect('analysis.T', 'dt.aeroThrust')
    group.connect('control:ratedPower', 'dt.ratedPower')


    # connections to powercurve
    group.connect('control', 'powercurve.control')
    group.connect('setup.Uhub', 'powercurve.Vcoarse')
    group.connect('dt.power', 'powercurve.Pcoarse')
    group.connect('analysis.T', 'powercurve.Tcoarse')
    group.connect('npts_spline_power_curve', 'powercurve.npts')

    if regulated:
        group.connect('geom.R', 'powercurve.R')

        # setup Brent method to find rated speed
        group.connect('control:Vin', 'brent.lower_bound')
        group.connect('control:Vout', 'brent.upper_bound')
        group.brent.add_inputeter('powercurve.Vrated', low=-1e-15, high=1e15)
        group.brent.add_constraint('powercurve.residual = 0')
        group.brent.invalid_bracket_return = 1.0


    # connections to cdf
    group.connect('powercurve.V', 'cdf.x')


    # connections to aep
    group.connect('cdf.F', 'aep.CDF_V')
    group.connect('powercurve.P', 'aep.P')
    group.connect('AEP_loss_factor', 'aep.lossFactor')


    # connections to outputs
    group.connect('powercurve.V', 'V')
    group.connect('powercurve.P', 'P')
    group.connect('aep.AEP', 'AEP')
    group.connect('2*geom.R', 'diameter')
    if regulated:
        group.connect('powercurve.ratedConditions', 'ratedConditions')





def common_io_with_ccblade(group, varspeed, varpitch, cdf_type):

    regulated = varspeed or varpitch

    # add inputs
    group.add_input('r_af', units='m', desc='locations where airfoils are defined on unit radius')
    group.add_input('r_max_chord')
    group.add_input('chord_sub', units='m', desc='chord at control points')
    group.add_input('theta_sub', units='deg', desc='twist at control points')
    group.add_input('Rhub', units='m', desc='hub radius')
    group.add_input('Rtip', units='m', desc='tip radius')
    group.add_input('hub_height', units='m')
    group.add_input('precone', desc='precone angle', units='deg')
    group.add_input('tilt', val=0.0, desc='shaft tilt', units='deg')
    group.add_input('yaw', val=0.0, desc='yaw error', units='deg')
    group.add_input('airfoil_files', desc='names of airfoil file')
    group.add_input('idx_cylinder', desc='location where cylinder section ends on unit radius')
    group.add_input('nBlades', val=3, desc='number of blades')
    group.add_input('rho', val=1.225, units='kg/m**3', desc='density of air')
    group.add_input('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air')
    group.add_input('shearExp', val=0.2, desc='shear exponent')
    group.add_input('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
    group.add_input('tiploss', val=True, desc='include Prandtl tip loss model')
    group.add_input('hubloss', val=True, desc='include Prandtl hub loss model')
    group.add_input('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
    group.add_input('usecd', val=True, desc='use drag coefficient in computing induction factors')
    group.add_input('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at')
    group.add_input('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve')
    group.add_input('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)')

    if varspeed:
        group.add_input('control:Vin', units='m/s', desc='cut-in wind speed')
        group.add_input('control:Vout', units='m/s', desc='cut-out wind speed')
        group.add_input('control:ratedPower', units='W', desc='rated power')
        group.add_input('control:minOmega', units='rpm', desc='minimum allowed rotor rotation speed')
        group.add_input('control:maxOmega', units='rpm', desc='maximum allowed rotor rotation speed')
        group.add_input('control:tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        group.add_input('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
    else:
        group.add_input('control:Vin', units='m/s', desc='cut-in wind speed')
        group.add_input('control:Vout', units='m/s', desc='cut-out wind speed')
        group.add_input('control:ratedPower', units='W', desc='rated power')
        group.add_input('control:Omega', units='rpm', desc='fixed rotor rotation speed')
        group.add_input('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        group.add_input('control:npts', val=20, desc='number of points to evalute aero code to generate power curve')

    group.add_input('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'])
    group.add_input('cdf_mean_wind_speed', units='m/s', desc='mean wind speed of site cumulative distribution function')

    if cdf_type == 'weibull':
        group.add_input('weibull_shape_factor', desc='(shape factor of weibull distribution)')

    # outputs
    group.add_output('AEP', units='kW*h', desc='annual energy production')
    group.add_output('V', units='m/s', desc='wind speeds (power curve)')
    group.add_output('P', units='W', desc='power (power curve)')
    group.add_output('diameter', units='m')
    if regulated:
        group.add_output('ratedConditions:V', units='m/s', desc='rated wind speed')
        group.add_output('ratedConditions:Omega', units='rpm', desc='rotor rotation speed at rated')
        group.add_output('ratedConditions:pitch', units='deg', desc='pitch setting at rated')
        group.add_output('ratedConditions:T', units='N', desc='rotor aerodynamic thrust at rated')
        group.add_output('ratedConditions:Q', units='N*m', desc='rotor aerodynamic torque at rated')



def common_configure_with_ccblade(group, varspeed, varpitch, cdf_type):
    common_configure(group, varspeed, varpitch)

    # put in parameterization for CCBlade
    group.add('spline', GeometrySpline())
    group.replace('geom', CCBladeGeometry())
    group.replace('analysis', CCBlade())
    group.replace('dt', CSMDrivetrain())
    if cdf_type == 'rayleigh':
        group.replace('cdf', RayleighCDF())
    elif cdf_type == 'weibull':
        group.replace('cdf', WeibullWithMeanCDF())


    # add spline to workflow
    group.driver.workflow.add('spline')

    # connections to spline
    group.connect('r_af', 'spline.r_af')
    group.connect('r_max_chord', 'spline.r_max_chord')
    group.connect('chord_sub', 'spline.chord_sub')
    group.connect('theta_sub', 'spline.theta_sub')
    group.connect('idx_cylinder', 'spline.idx_cylinder')
    group.connect('Rhub', 'spline.Rhub')
    group.connect('Rtip', 'spline.Rtip')

    # connections to geom
    group.connect('Rtip', 'geom.Rtip')
    group.connect('precone', 'geom.precone')

    # connections to analysis
    group.connect('spline.r', 'analysis.r')
    group.connect('spline.chord', 'analysis.chord')
    group.connect('spline.theta', 'analysis.theta')
    group.connect('spline.precurve', 'analysis.precurve')
    group.connect('Rhub', 'analysis.Rhub')
    group.connect('Rtip', 'analysis.Rtip')
    group.connect('hub_height', 'analysis.hub_height')
    group.connect('precone', 'analysis.precone')
    group.connect('tilt', 'analysis.tilt')
    group.connect('yaw', 'analysis.yaw')
    group.connect('airfoil_files', 'analysis.airfoil_files')
    group.connect('nBlades', 'analysis.nBlades')
    group.connect('rho', 'analysis.rho')
    group.connect('mu', 'analysis.mu')
    group.connect('shearExp', 'analysis.shearExp')
    group.connect('nSector', 'analysis.nSector')
    group.connect('tiploss', 'analysis.tiploss')
    group.connect('hubloss', 'analysis.hubloss')
    group.connect('wakerotation', 'analysis.wakerotation')
    group.connect('usecd', 'analysis.usecd')

    # connections to dt
    group.connect('drivetrainType', 'dt.drivetrainType')
    group.dt.missing_deriv_policy = 'assume_zero'  # TODO: openmdao bug remove later

    # connnections to cdf
    group.connect('cdf_mean_wind_speed', 'cdf.xbar')
    if cdf_type == 'weibull':
        group.connect('weibull_shape_factor', 'cdf.k')




class RotorAeroVSVP(Group):

    def configure(self):
        varspeed = True
        varpitch = True
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroVSFP(Group):

    def configure(self):
        varspeed = True
        varpitch = False
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroFSVP(Group):

    def configure(self):
        varspeed = False
        varpitch = True
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroFSFP(Group):

    def configure(self):
        varspeed = False
        varpitch = False
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)

        
class RotorAeroVSVPWithCCBlade(Group):
    def setup(self, cdf_type='weibull'):
        self.cdf_type = cdf_type

    def configure(self):
        varspeed = True
        varpitch = True
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)


class RotorAeroVSFPWithCCBlade(Group):

    def setup(self, cdf_type='weibull'):
        self.cdf_type = cdf_type

    def configure(self):
        varspeed = True
        varpitch = False
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)



class RotorAeroFSVPWithCCBlade(Group):

    def setup(self, cdf_type='weibull'):
        self.cdf_type = cdf_type

    def configure(self):
        varspeed = False
        varpitch = True
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)



class RotorAeroFSFPWithCCBlade(Group):

    def setup(self, cdf_type='weibull'):
        self.cdf_type = cdf_type

    def configure(self):
        varspeed = False
        varpitch = False
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)


'''
