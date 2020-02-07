import copy
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from openmdao.api import ExplicitComponent, Group
from wisdem.ccblade.ccblade_component import CCBladeLoads, AeroHubLoads
import wisdem.ccblade._bem as _bem
from wisdem.commonse.utilities import rotate, arc_length
from wisdem.commonse.akima import Akima
from wisdem.commonse import gravity
from wisdem.commonse.csystem import DirectionVector
from wisdem.rotorse import RPM2RS, RS2RPM
import wisdem.pBeam._pBEAM as _pBEAM

class GustETM(ExplicitComponent):
    # OpenMDAO component that generates an "equivalent gust" wind speed by summing an user-defined wind speed at hub height with 3 times sigma. sigma is the turbulent wind speed standard deviation for the extreme turbulence model, see IEC-61400-1 Eq. 19 paragraph 6.3.2.3
    
    def setup(self):
        # Inputs
        self.add_input('V_mean', val=0.0, units='m/s', desc='IEC average wind speed for turbine class')
        self.add_input('V_hub', val=0.0, units='m/s', desc='hub height wind speed')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('std', val=3, desc='number of standard deviations for strength of gust')

        # Output
        self.add_output('V_gust', val=0.0, units='m/s', desc='gust wind speed')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        V_mean = inputs['V_mean']
        V_hub = inputs['V_hub']
        turbulence_class = discrete_inputs['turbulence_class']
        std = discrete_inputs['std']

        if turbulence_class == 'A':
            Iref = 0.16
        elif turbulence_class == 'B':
            Iref = 0.14
        elif turbulence_class == 'C':
            Iref = 0.12

        c = 2.0
        sigma = c * Iref * (0.072*(V_mean/c + 3)*(V_hub/c - 4) + 10)
        V_gust = V_hub + std*sigma
        outputs['V_gust'] = V_gust

class BladeCurvature(ExplicitComponent):
    # OpenMDAO component that computes the 3D curvature of the blade
    def initialize(self):
        self.options.declare('analysis_options')

    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
        n_span    = blade_init_options['n_span']

        # Inputs
        self.add_input('r',         val=np.zeros(n_span), units='m',      desc='location in blade z-coordinate')
        self.add_input('precurve',  val=np.zeros(n_span), units='m',      desc='location in blade x-coordinate')
        self.add_input('presweep',  val=np.zeros(n_span), units='m',      desc='location in blade y-coordinate')
        self.add_input('precone',   val=0.0,              units='deg',    desc='precone angle')

        # Outputs
        self.add_output('3d_curv',  val=np.zeros(n_span),units='deg',    desc='total cone angle from precone and curvature')
        self.add_output('x_az',     val=np.zeros(n_span), units='m',      desc='location of blade in azimuth x-coordinate system')
        self.add_output('y_az',     val=np.zeros(n_span), units='m',      desc='location of blade in azimuth y-coordinate system')
        self.add_output('z_az',     val=np.zeros(n_span), units='m',      desc='location of blade in azimuth z-coordinate system')
        self.add_output('s',        val=np.zeros(n_span), units='m',      desc='cumulative path length along blade')

    def compute(self, inputs, outputs):

        r = inputs['r']
        precurve = inputs['precurve']
        presweep = inputs['presweep']
        precone = inputs['precone']

        n = len(r)
        dx_dx = np.eye(3*n)

        x_az, x_azd, y_az, y_azd, z_az, z_azd, cone, coned, s, sd = _bem.definecurvature_dv2(r, dx_dx[:, :n],
                                                                                             precurve, dx_dx[:, n:2*n],
                                                                                             presweep, dx_dx[:, 2*n:],
                                                                                             0.0, np.zeros(3*n))

        totalCone = precone + np.degrees(cone)
        s = r[0] + s

        outputs['3d_curv'] = totalCone
        outputs['x_az'] = x_az
        outputs['y_az'] = y_az
        outputs['z_az'] = z_az
        outputs['s'] = s

class TotalLoads(ExplicitComponent):
    # OpenMDAO component that takes as input the rotor configuration (tilt, cone), the blade twist and mass distributions, and the blade aerodynamic loading, and computes the total loading including gravity and centrifugal forces
    def initialize(self):
        self.options.declare('analysis_options')

    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
        n_span    = blade_init_options['n_span']

        # Inputs
        self.add_input('r',                 val=np.zeros(n_span),   units='m',      desc='radial positions along blade going toward tip')
        self.add_input('aeroloads_Px',      val=np.zeros(n_span),   units='N/m',    desc='distributed loads in blade-aligned x-direction')
        self.add_input('aeroloads_Py',      val=np.zeros(n_span),   units='N/m',    desc='distributed loads in blade-aligned y-direction')
        self.add_input('aeroloads_Pz',      val=np.zeros(n_span),   units='N/m',    desc='distributed loads in blade-aligned z-direction')
        self.add_input('aeroloads_Omega',   val=0.0,                units='rpm',    desc='rotor rotation speed')
        self.add_input('aeroloads_pitch',   val=0.0,                units='deg',    desc='pitch angle')
        self.add_input('aeroloads_azimuth', val=0.0,                units='deg',    desc='azimuthal angle')
        self.add_input('theta',             val=np.zeros(n_span),   units='deg',    desc='structural twist')
        self.add_input('tilt',              val=0.0,                units='deg',    desc='tilt angle')
        self.add_input('3d_curv',           val=np.zeros(n_span),   units='deg',    desc='total cone angle from precone and curvature')
        self.add_input('z_az',              val=np.zeros(n_span),   units='m',      desc='location of blade in azimuth z-coordinate system')
        self.add_input('rhoA',              val=np.zeros(n_span),   units='kg/m',   desc='mass per unit length')
        self.add_input('dynamicFactor',     val=1.0,                                desc='a dynamic amplification factor to adjust the static deflection calculation') #)

        # Outputs
        self.add_output('Px_af', val=np.zeros(n_span), desc='total distributed loads in airfoil x-direction')
        self.add_output('Py_af', val=np.zeros(n_span), desc='total distributed loads in airfoil y-direction')
        self.add_output('Pz_af', val=np.zeros(n_span), desc='total distributed loads in airfoil z-direction')


    def compute(self, inputs, outputs):

        dynamicFactor = inputs['dynamicFactor']
        r = inputs['r']
        theta = inputs['theta']
        tilt = inputs['tilt']
        totalCone = inputs['3d_curv']
        z_az = inputs['z_az']
        rhoA = inputs['rhoA']


        # totalCone = precone
        # z_az = r*cosd(precone)
        totalCone = totalCone
        z_az = z_az

        # keep all in blade c.s. then rotate all at end

        # rename
        # aero = aeroloads

        # --- aero loads ---

        # interpolate aerodynamic loads onto structural grid
        P_a = DirectionVector(0, 0, 0)
        myakima = Akima(inputs['r'], inputs['aeroloads_Px'])
        P_a.x, dPax_dr, dPax_daeror, dPax_daeroPx = myakima(r)

        myakima = Akima(inputs['r'], inputs['aeroloads_Py'])
        P_a.y, dPay_dr, dPay_daeror, dPay_daeroPy = myakima(r)

        myakima = Akima(inputs['r'], inputs['aeroloads_Pz'])
        P_a.z, dPaz_dr, dPaz_daeror, dPaz_daeroPz = myakima(r)


        # --- weight loads ---

        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -rhoA*gravity)

        P_w = weight.yawToHub(tilt).hubToAzimuth(inputs['aeroloads_azimuth'])\
            .azimuthToBlade(totalCone)


        # --- centrifugal loads ---

        # azimuthal c.s.
        Omega = inputs['aeroloads_Omega']*RPM2RS
        load = DirectionVector(0.0, 0.0, rhoA*Omega**2*z_az)

        P_c = load.azimuthToBlade(totalCone)


        # --- total loads ---
        P = P_a + P_w + P_c

        # rotate to airfoil c.s.
        theta = np.array(theta) + inputs['aeroloads_pitch']
        P = P.bladeToAirfoil(theta)

        Px_af = dynamicFactor * P.x
        Py_af = dynamicFactor * P.y
        Pz_af = dynamicFactor * P.z

        outputs['Px_af'] = Px_af
        outputs['Py_af'] = Py_af
        outputs['Pz_af'] = Pz_af

class RunpBEAM(ExplicitComponent):
    def initialize(self):
        self.options.declare('analysis_options')

    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
        self.n_span = n_span = blade_init_options['n_span']
        self.n_freq = n_freq = blade_init_options['n_freq']

        # all inputs/outputs in airfoil coordinate system
        self.add_input('Px_af', val=np.zeros(n_span), desc='distributed load (force per unit length) in airfoil x-direction')
        self.add_input('Py_af', val=np.zeros(n_span), desc='distributed load (force per unit length) in airfoil y-direction')
        self.add_input('Pz_af', val=np.zeros(n_span), desc='distributed load (force per unit length) in airfoil z-direction')

        self.add_input('xu_strain_spar',    val=np.zeros(n_span), desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_input('xl_strain_spar',    val=np.zeros(n_span), desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_input('yu_strain_spar',    val=np.zeros(n_span), desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_input('yl_strain_spar',    val=np.zeros(n_span), desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_input('xu_strain_te',      val=np.zeros(n_span), desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_input('xl_strain_te',      val=np.zeros(n_span), desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_input('yu_strain_te',      val=np.zeros(n_span), desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_input('yl_strain_te',      val=np.zeros(n_span), desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')

        self.add_input('r',     val=np.zeros(n_span), units='m',        desc='locations of properties along beam')
        self.add_input('EA',    val=np.zeros(n_span), units='N',        desc='axial stiffness')
        self.add_input('EIxx',  val=np.zeros(n_span), units='N*m**2',   desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('EIyy',  val=np.zeros(n_span), units='N*m**2',   desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('EIxy',  val=np.zeros(n_span), units='N*m**2',   desc='coupled flap-edge stiffness')
        self.add_input('GJ',    val=np.zeros(n_span), units='N*m**2',   desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_input('rhoA',  val=np.zeros(n_span), units='kg/m',     desc='mass per unit length')
        self.add_input('rhoJ',  val=np.zeros(n_span), units='kg*m',     desc='polar mass moment of inertia per unit length')
        self.add_input('x_ec',  val=np.zeros(n_span), units='m',        desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_input('y_ec',  val=np.zeros(n_span), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')

        # outputs
        # self.add_output('blade_mass',       val=0.0,              units='kg',       desc='mass of one blades')
        # self.add_output('blade_moment_of_inertia', val=0.0,       units='kg*m**2',  desc='out of plane moment of inertia of a blade')
        self.add_output('freq_pbeam',       val=np.zeros(n_freq), units='Hz',       desc='first nF natural frequencies of blade')
        self.add_output('freq_distance',    val=0.0,              desc='ration of 2nd and 1st natural frequencies, should be ratio of edgewise to flapwise')
        self.add_output('dx',               val=np.zeros(n_span), desc='deflection of blade section in airfoil x-direction')
        self.add_output('dy',               val=np.zeros(n_span), desc='deflection of blade section in airfoil y-direction')
        self.add_output('dz',               val=np.zeros(n_span), desc='deflection of blade section in airfoil z-direction')
        self.add_output('strainU_spar',     val=np.zeros(n_span), desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_output('strainL_spar',     val=np.zeros(n_span), desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')
        self.add_output('strainU_te',       val=np.zeros(n_span), desc='strain in trailing-edge panels on upper surface at location xu,yu_te with loads P_te')
        self.add_output('strainL_te',       val=np.zeros(n_span), desc='strain in trailing-edge panels on lower surface at location xl,yl_te with loads P_te')
        
    def principalCS(self, EIyy_in, EIxx_in, y_ec_in, x_ec_in, EA, EIxy):

        # rename (with swap of x, y for profile c.s.)
        EIxx , EIyy = EIyy_in , EIxx_in
        x_ec , y_ec = y_ec_in , x_ec_in
        self.EA     = EA
        EIxy        = EIxy

        # translate to elastic center
        EIxx -= y_ec**2*EA
        EIyy -= x_ec**2*EA
        EIxy -= x_ec*y_ec*EA

        # get rotation angle
        alpha = 0.5*np.arctan2(2*EIxy, EIyy-EIxx)

        self.EI11 = EIxx - EIxy*np.tan(alpha)
        self.EI22 = EIyy + EIxy*np.tan(alpha)

        # get moments and positions in principal axes
        self.ca = np.cos(alpha)
        self.sa = np.sin(alpha)

    def strain(self, blade, xu, yu, xl, yl):

        Vx, Vy, Fz, Mx, My, Tz = blade.shearAndBending()

        # use profile c.s. to use Hansen's notation
        Vx, Vy = Vy, Vx
        Mx, My = My, Mx
        xu, yu = yu, xu
        xl, yl = yl, xl

        # convert to principal xes
        M1 = Mx*self.ca + My*self.sa
        M2 = -Mx*self.sa + My*self.ca

        x = xu*self.ca + yu*self.sa
        y = -xu*self.sa + yu*self.ca

        # compute strain
        strainU = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)  # negative sign because 3 is opposite of z

        x = xl*self.ca + yl*self.sa
        y = -xl*self.sa + yl*self.ca

        strainL = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)

        return strainU, strainL

    def compute(self, inputs, outputs):

        Px = inputs['Px_af']
        Py = inputs['Py_af']
        Pz = inputs['Pz_af']
        xu_strain_spar = inputs['xu_strain_spar']
        xl_strain_spar = inputs['xl_strain_spar']
        yu_strain_spar = inputs['yu_strain_spar']
        yl_strain_spar = inputs['yl_strain_spar']
        xu_strain_te = inputs['xu_strain_te']
        xu_strain_te = inputs['xu_strain_te']
        xl_strain_te = inputs['xl_strain_te']
        yu_strain_te = inputs['yu_strain_te']
        yl_strain_te = inputs['yl_strain_te']

        # outputs
        nsec = self.n_span
        
        # create finite element objects
        p_section = _pBEAM.SectionData(nsec, inputs['r'], inputs['EA'], inputs['EIxx'],
            inputs['EIyy'], inputs['GJ'], inputs['rhoA'], inputs['rhoJ'])
        p_tip = _pBEAM.TipData()  # no tip mass
        p_base = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base


        # ----- tip deflection -----

        # evaluate displacements
        p_loads = _pBEAM.Loads(nsec, Px, Py, Pz)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

        # now computed in rotor elasticity!
        # --- moments of inertia
        # blade_moment_of_inertia = blade.outOfPlaneMomentOfInertia()
        # --- mass --- 
        # blade_mass = blade.mass()
        # ----- natural frequencies ----
        freq = blade.naturalFrequencies(self.n_freq)

        # ----- strain -----
        self.principalCS(inputs['EIyy'], inputs['EIxx'], inputs['y_ec'], inputs['x_ec'], inputs['EA'], inputs['EIxy'])
        strainU_spar, strainL_spar = self.strain(blade, xu_strain_spar, yu_strain_spar, xl_strain_spar, yl_strain_spar)
        strainU_te, strainL_te = self.strain(blade, xu_strain_te, yu_strain_te, xl_strain_te, yl_strain_te)

        outputs['freq_pbeam'] = freq
        outputs['freq_distance'] = np.float(freq[1]/freq[0])
        outputs['dx'] = dx
        outputs['dy'] = dy
        outputs['dz'] = dz
        outputs['strainU_spar'] = strainU_spar
        outputs['strainL_spar'] = strainL_spar
        outputs['strainU_te'] = strainU_te
        outputs['strainL_te'] = strainL_te

class TipDeflection(ExplicitComponent):
    # OpenMDAO component that computes the blade deflection at tip in yaw x-direction
    def setup(self):
        # Inputs
        self.add_input('dx_tip',        val=0.0,                    desc='deflection at tip in airfoil x-direction')
        self.add_input('dy_tip',        val=0.0,                    desc='deflection at tip in airfoil y-direction')
        self.add_input('dz_tip',        val=0.0,                    desc='deflection at tip in airfoil z-direction')
        self.add_input('theta_tip',     val=0.0,    units='deg',    desc='twist at tip section')
        self.add_input('pitch_load',    val=0.0,    units='deg',    desc='blade pitch angle')
        self.add_input('tilt',          val=0.0,    units='deg',    desc='tilt angle')
        self.add_input('3d_curv_tip',   val=0.0,    units='deg',    desc='total coning angle including precone and curvature')
        self.add_input('dynamicFactor', val=1.0,                    desc='a dynamic amplification factor to adjust the static deflection calculation') #)
        # Outputs
        self.add_output('tip_deflection', val=0.0,  units='m',      desc='deflection at tip in yaw x-direction')

    def compute(self, inputs, outputs):

        dx            = inputs['dx_tip']
        dy            = inputs['dy_tip']
        dz            = inputs['dz_tip']
        theta         = inputs['theta_tip']
        pitch         = inputs['pitch_load']
        azimuth       = 180.0 # The blade is assumed in front of the tower, although the loading may correspond to another azimuthal position
        tilt          = inputs['tilt']
        totalConeTip  = inputs['3d_curv_tip']
        dynamicFactor = inputs['dynamicFactor']

        theta = theta + pitch

        dr = DirectionVector(dx, dy, dz)
        delta = dr.airfoilToBlade(theta).bladeToAzimuth(totalConeTip).azimuthToHub(azimuth).hubToYaw(tilt)

        tip_deflection = dynamicFactor * delta.x

        outputs['tip_deflection'] = tip_deflection

class DesignConstraints(ExplicitComponent):
    # OpenMDAO component that formulates constraints on user-defined maximum strains, frequencies   
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')

    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
        self.n_span = n_span = blade_init_options['n_span']
        self.n_freq = n_freq = blade_init_options['n_freq']
        self.opt_options   = opt_options   = self.options['opt_options']
        self.n_opt_spar_cap_ss = n_opt_spar_cap_ss = opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt']
        self.n_opt_spar_cap_ps = n_opt_spar_cap_ps = opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt']
        # Inputs strains
        self.add_input('strainU_spar',     val=np.zeros(n_span), desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_input('strainL_spar',     val=np.zeros(n_span), desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')

        self.add_input('min_strainU_spar', val=0.0, desc='minimum strain in spar cap suction side')
        self.add_input('max_strainU_spar', val=0.0, desc='minimum strain in spar cap pressure side')
        self.add_input('min_strainL_spar', val=0.0, desc='maximum strain in spar cap suction side')
        self.add_input('max_strainL_spar', val=0.0, desc='maximum strain in spar cap pressure side')
        
        self.add_input('s',                     val=np.zeros(n_span),       desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('s_opt_spar_cap_ss',         val=np.zeros(n_opt_spar_cap_ss),desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side')
        self.add_input('s_opt_spar_cap_ps',         val=np.zeros(n_opt_spar_cap_ss),desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side')

        # Input frequencies
        self.add_input('rated_Omega', val=0.0,                units='rpm', desc='rotor rotation speed at rated')
        self.add_input('delta_f',     val=1.1,                             desc='minimum margin between 3P and edge frequency')
        self.add_input('freq',        val=np.zeros(n_freq),   units='Hz',  desc='first nF natural frequencies')

        # Outputs
        self.add_output('constr_min_strainU_spar',     val=np.zeros(n_opt_spar_cap_ss), desc='constraint for minimum strain in spar cap suction side')
        self.add_output('constr_max_strainU_spar',     val=np.zeros(n_opt_spar_cap_ss), desc='constraint for maximum strain in spar cap suction side')
        self.add_output('constr_min_strainL_spar',     val=np.zeros(n_opt_spar_cap_ps), desc='constraint for minimum strain in spar cap pressure side')
        self.add_output('constr_max_strainL_spar',     val=np.zeros(n_opt_spar_cap_ps), desc='constraint for maximum strain in spar cap pressure side')
        self.add_output('constr_flap_f_above_3P',      val=0.0,                     desc='constraint on flap blade frequency to stay above 3P + delta')
        self.add_output('constr_edge_f_above_3P',      val=0.0,                     desc='constraint on edge blade frequency to stay above 3P + delta')

    def compute(self, inputs, outputs):
        
        # Constraints on blade strains
        s               = inputs['s']
        s_opt_spar_cap_ss   = inputs['s_opt_spar_cap_ss']
        s_opt_spar_cap_ps   = inputs['s_opt_spar_cap_ps']
        
        strainU_spar    = inputs['strainU_spar']
        strainL_spar    = inputs['strainL_spar']
        min_strainU_spar= inputs['min_strainU_spar']
        max_strainU_spar= inputs['max_strainU_spar']
        min_strainL_spar= inputs['min_strainL_spar']
        max_strainL_spar= inputs['max_strainL_spar']
        
        outputs['constr_min_strainU_spar'] = abs(np.interp(s_opt_spar_cap_ss, s, strainU_spar)) / abs(min_strainU_spar)
        outputs['constr_max_strainU_spar'] = abs(np.interp(s_opt_spar_cap_ss, s, strainU_spar)) / max_strainU_spar
        outputs['constr_min_strainL_spar'] = abs(np.interp(s_opt_spar_cap_ps, s, strainL_spar)) / abs(min_strainL_spar)
        outputs['constr_max_strainL_spar'] = abs(np.interp(s_opt_spar_cap_ps, s, strainL_spar)) / max_strainL_spar

        # Constraints on blade frequencies
        threeP = 3. * inputs['rated_Omega'] / 60.
        flap_f = inputs['freq'][0] # assuming the flap frequency is the first lowest
        edge_f = inputs['freq'][1] # assuming the edge frequency is the second lowest
        delta  = inputs['delta_f']
        outputs['constr_flap_f_above_3P'] = (threeP * delta) / flap_f
        outputs['constr_edge_f_above_3P'] = (threeP * delta) / edge_f
        
class RotorLoadsDeflStrains(Group):
    # OpenMDAO group to compute the blade elastic properties, deflections, and loading
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')
    def setup(self):
        analysis_options = self.options['analysis_options']
        opt_options     = self.options['opt_options']

        # Load blade with rated conditions and compute aerodynamic forces
        promoteListAeroLoads =  ['r', 'theta', 'chord', 'Rtip', 'Rhub', 'hub_height', 'precone', 'tilt', 'airfoils_aoa', 'airfoils_Re', 'airfoils_cl', 'airfoils_cd', 'airfoils_cm', 'nBlades', 'rho', 'mu', 'Omega_load','pitch_load']
        # self.add_subsystem('aero_rated',        CCBladeLoads(analysis_options = analysis_options), promotes=promoteListAeroLoads)
        self.add_subsystem('gust',              GustETM())
        self.add_subsystem('aero_gust',         CCBladeLoads(analysis_options = analysis_options), promotes=promoteListAeroLoads)
        # self.add_subsystem('aero_storm_1yr',    CCBladeLoads(analysis_options = analysis_options), promotes=promoteListAeroLoads)
        # self.add_subsystem('aero_storm_50yr',   CCBladeLoads(analysis_options = analysis_options), promotes=promoteListAeroLoads)
        # Add centrifugal and gravity loading to aero loading
        promotes=['tilt','theta','rhoA','z','totalCone','z_az']
        self.add_subsystem('curvature',         BladeCurvature(analysis_options = analysis_options),  promotes=['r','precone','precurve','presweep','3d_curv','z_az'])
        promoteListTotalLoads = ['r', 'theta', 'tilt', 'rhoA', '3d_curv', 'z_az', 'aeroloads_Omega', 'aeroloads_pitch']
        # self.add_subsystem('tot_loads_rated',       TotalLoads(analysis_options = analysis_options),      promotes=promoteListTotalLoads)
        self.add_subsystem('tot_loads_gust',        TotalLoads(analysis_options = analysis_options),      promotes=promoteListTotalLoads)
        # self.add_subsystem('tot_loads_storm_1yr',   TotalLoads(analysis_options = analysis_options),      promotes=promoteListTotalLoads)
        # self.add_subsystem('tot_loads_storm_50yr',  TotalLoads(analysis_options = analysis_options),      promotes=promoteListTotalLoads)
        promoteListpBeam = ['r','EA','EIxx','EIyy','EIxy','GJ','rhoA','rhoJ','x_ec','y_ec','xu_strain_spar','xl_strain_spar','yu_strain_spar','yl_strain_spar','xu_strain_te','xl_strain_te','yu_strain_te','yl_strain_te']
        self.add_subsystem('pbeam',     RunpBEAM(analysis_options = analysis_options),      promotes=promoteListpBeam)
        self.add_subsystem('tip_pos',   TipDeflection(),                                  promotes=['tilt','pitch_load'])
        self.add_subsystem('aero_hub_loads', AeroHubLoads(analysis_options = analysis_options), promotes = promoteListAeroLoads)
        self.add_subsystem('constr',    DesignConstraints(analysis_options = analysis_options, opt_options = opt_options))

        # Aero loads to total loads
        # self.connect('aero_rated.loads_Px',     'tot_loads_rated.aeroloads_Px')
        # self.connect('aero_rated.loads_Py',     'tot_loads_rated.aeroloads_Py')
        # self.connect('aero_rated.loads_Pz',     'tot_loads_rated.aeroloads_Pz')
        self.connect('aero_gust.loads_Px',      'tot_loads_gust.aeroloads_Px')
        self.connect('aero_gust.loads_Py',      'tot_loads_gust.aeroloads_Py')
        self.connect('aero_gust.loads_Pz',      'tot_loads_gust.aeroloads_Pz')
        # self.connect('aero_storm_1yr.loads_Px', 'tot_loads_storm_1yr.aeroloads_Px')
        # self.connect('aero_storm_1yr.loads_Py', 'tot_loads_storm_1yr.aeroloads_Py')
        # self.connect('aero_storm_1yr.loads_Pz', 'tot_loads_storm_1yr.aeroloads_Pz')
        # self.connect('aero_storm_50yr.loads_Px', 'tot_loads_storm_50yr.aeroloads_Px')
        # self.connect('aero_storm_50yr.loads_Py', 'tot_loads_storm_50yr.aeroloads_Py')
        # self.connect('aero_storm_50yr.loads_Pz', 'tot_loads_storm_50yr.aeroloads_Pz')

        # Total loads to strains
        self.connect('tot_loads_gust.Px_af', 'pbeam.Px_af')
        self.connect('tot_loads_gust.Py_af', 'pbeam.Py_af')
        self.connect('tot_loads_gust.Pz_af', 'pbeam.Pz_af')

        # Blade distributed deflections to tip deflection
        self.connect('pbeam.dx', 'tip_pos.dx_tip', src_indices=[-1])
        self.connect('pbeam.dy', 'tip_pos.dy_tip', src_indices=[-1])
        self.connect('pbeam.dz', 'tip_pos.dz_tip', src_indices=[-1])
        self.connect('3d_curv',  'tip_pos.3d_curv_tip', src_indices=[-1])

        # Strains from pbeam to constraint
        self.connect('pbeam.strainU_spar', 'constr.strainU_spar')
        self.connect('pbeam.strainL_spar', 'constr.strainL_spar')

         