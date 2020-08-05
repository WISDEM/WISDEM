import openmdao.api as om
import numpy as np

# Application factor to include ring/housing/carrier weight
Kr = 0.4

#-----------------------------------

def V_planetary(U, B, K):
    sunU = 0.5*U - 1.0
    V = 1./U + 1./U/B + 1./B/(0.5*U-1.0) + sunU + sunU**2 + K*(U-1.)**2/B + K*(U-1.)**2/B/sunU
    return V
#-----------------------------------

def V_parallel(U):
    V = (1. + 1./U + U + U**2)
    return V
#-----------------------------------
def volumeEEP(x, n_planets, torque, Kr1=Kr, K_r2=Kr):
    # Safety factor?
    Kgamma = 1.1 * np.ones(n_planets.shape)
    Kgamma[n_planets>=5] = 1.35

    ## TODO: Stage torque or Q-factor? (not the same thing)

    # Individual stage torques
    Q_stage = torque / np.cumprod(x)

    # Volume
    V = (Q_stage[0] * Kgamma[0] * V_planetary(x[0], n_planets[0], K_r1) +
         Q_stage[1] * Kgamma[1] * V_planetary(x[1], n_planets[1], K_r2) +
         Q_stage[2]             * V_parallel( x[2])/np.prod(x) )
    return 2*V
#-----------------------------------

def volumeEPP(x, n_planets, torque, Kr1=Kr):
    # Safety factor?
    Kgamma = 1.1 if n_planets[0] < 5 else 1.35

    ## TODO: Stage torque or Q-factor? (not the same thing)
    
    # Individual stage torques
    Q_stage = torque / np.cumprod(x)
    
    V = (Q_stage[0] * Kgamma * V_planetary(x[0], n_planets[0], K_r1) +
         Q_stage[1]          * V_parallel( x[1])/np.prod(x[:2]) +
         Q_stage[2]          * V_parallel( x[2])/np.prod(x) )
    return 2*V
#-----------------------------------
        

class Gearbox(om.ExplicitComponent):
    """
    Gearbox class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
    Approach computes the gearbox weight based on a surface durability criteria.
    
    Parameters
    ----------
    direct_drive : boolean
        True if system is direct drive        
    gear_configuration : string
        string that represents the configuration of the gearbox (stage number and types)
    shaft_factor : string
        normal or short shaft length
    n_planets : numpy array[3]
        number of planets in each stage
    gear_ratio : float
        overall gearbox speedup ratio
    rotor_rpm : float, [rpm]
        rotor rpm at rated power
    D_rotor : float, [m]
        rotor diameter
    Q_rotor : float, [N*m]
        rotor torque at rated power
    s_gearbox : float, [m]
        gearbox position along x-axis
    
    Returns
    -------
    stage_masses : numpy array[3], [kg]
        individual gearbox stage gearbox_masses
    gearbox_mass : float, [kg]
        overall component mass
    gearbox_cm : numpy array[3], [m]
        Gearbox center of mass [x,y,z] measure along shaft from bedplate
    gearbox_I : numpy array[3], [kg*m**2]
        Gearbox mass moments of inertia [Ixx, Iyy, Izz] around its center of mass
    L_gearbox : float, [m]
        length of gearbox
    H_gearbox : float, [m]
        height of gearbox
    D_gearbox : float, [m]
        diameter of gearbox
    
    """

    def setup(self):

        self.add_discrete_input('direct_drive', False)
        self.add_discrete_input('gear_configuration', val='eep')
        self.add_discrete_input('shaft_factor', val='normal')
        self.add_discrete_input('planet_numbers', val=np.array([0, 0, 0]))
        self.add_input('gear_ratio', val=1.0)
        self.add_input('D_rotor', val=0.0, units='m')
        self.add_input('Q_rotor', val=0.0, units='N*m')

        self.add_output('stage_ratios', val=np.zeros(3))
        self.add_output('gearbox_mass', 0.0, units='kg')
        self.add_output('gearbox_I', np.zeros(3), units='kg*m**2')
        self.add_output('L_gearbox', 0.0, units='m')
        self.add_output('D_gearbox', 0.0, units='m')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        if discrete_inputs['direct_drive']: return

        # Unpack inputs
        config         = inputs['gear_configuration']
        shaft_factor   = inputs['shaft_factor']
        gear_ratio     = inputs['gear_ratio']
        n_planets      = inputs['planet_numbers']
        rotor_diameter = inputs['rotor_diameter']
        rotor_torque   = inputs['rotor_torque']

        # Known configuration checks
        if not config.lower() in ['eep','eep_2','eep_3','epp']:
            raise ValueError('Invalid value for gearbox_configuration.  Must be one of: eep, eep_2, eep_3, epp')
        n_stage = 3

        # Optimize stage ratios
        def constr(x, ratio):
            return np.prod(x) - ratio

        x0     = gear_ratio ** (1.0 / n_stage) * np.ones(n_stage)
        bounds = [(0.0, 1e2)]*n_stage
        const  = {}
        const['type'] = 'eq'
        const['fun']  = constr
        const['args'] = [gear_ratio]

        if config.lower() == 'eep':
            result = minimize(lambda x: VolumeEEP(x, n_planets, torque), x0, method='slsqp', bounds=bounds,
                              tol=1e-3, constraints=const, options={'maxiter': 100} )
            ratio_stage = result.x

        elif config == 'eep_3':
            # fixes last stage ratio at 3
            const['args']  = [gear_ratio/3.0]
            result = minimize(lambda x: VolumeEEP(np.r_[x, 3.0], n_planets, torque), x0[:2], method='slsqp', bounds=bounds[:2],
                              tol=1e-3, constraints=const, options={'maxiter': 100} )
            ratio_stage = np.r_[result.x, 3.0]

        elif config == 'eep_2':
            # fixes final stage ratio at 2
            const['args']  = [gear_ratio/2.0]
            result = minimize(lambda x: VolumeEEP(np.r_[x, 2.0], n_planets, torque), x0[:2], method='slsqp', bounds=bounds[:2],
                              tol=1e-3, constraints=const, options={'maxiter': 100} )
            ratio_stage = np.r_[result.x, 2.0]

        elif config == 'epp':
            result = minimize(lambda x: VolumeEPP(x, n_planets, torque), x0, method='slsqp', bounds=bounds,
                              tol=1e-3, constraints=const, options={'maxiter': 100} )
            ratio_stage = result.x

        # Get final volume
        if config.lower().find('eep') >= 0:
            vol = volumeEEP(ratio_stage, n_planets, torque)
        else:
            vol = volumeEPP(ratio_stage, n_planets, torque)

        ## TODO: None of this seems right
        # Application factor for weight estimate
        Ka = 0.6

        # Make conservative K factor assumption
        Kfact = 1100.0

        # Unit conversion from Nm to inlb and vice-versa
        Kunit = 8.029 # 8.85075 to convert from N-m to in-lb?

        # Shaft length factor
        Kshaft = 1.0 if shaft_factor == 'normal' else 1.25

        # All factors into the mass
        m_gearbox = Kshaft * Kunit * Ka / Kfact * vol.sum()

        # calculate mass properties
        L_gearbox = (0.012 * rotor_diameter)
        R_gearbox = 0.5 * 0.75 * 0.015 * rotor_diameter

        I = np.zeros(3)
        I[0] = 0.5*R_gearbox**2
        I[1:] = (1./12.)*(3*R_gearbox**2 + L_gearbox**2)

        # Store outputs
        outputs['stage_ratios'] = ratio_stage
        outputs['gearbox_mass'] = m_gearbox
        outputs['gearbox_I'] = I
        outputs['D_gearbox'] = 2*R_gearbox
        outputs['L_gearbox'] = L_gearbox

