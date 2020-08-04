import openmdao.api as om
import numpy as np


class Gearbox(om.ExplicitComponent):
    """
    Gearbox class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
    
    Parameters
    ----------
    direct_drive : boolean
        True if system is direct drive        
    gear_configuration : string
        string that represents the configuration of the gearbox (stage number and types)
    shaft_factor : string
        normal or short shaft length
    planet_numbers : numpy array[3]
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
        self.add_discrete_input('planet_numbers', val=np.array([0, 0, 0, ]))
        self.add_input('gear_ratio', val=1.0)
        self.add_input('rotor_rpm', val=0.0, units='rpm')
        self.add_input('D_rotor', val=0.0, units='m')
        self.add_input('Q_rotor', val=0.0, units='N*m')
        self.add_input('s_gearbox', val=0.00, units='m')

        self.add_output('stage_masses', val=np.zeros(3), units='kg')
        self.add_output('gearbox_mass', 0.0, units='kg')
        self.add_output('gearbox_cm', np.zeros(3), units='m')
        self.add_output('gearbox_I', np.zeros(3), units='kg*m**2')
        self.add_output('L_gearbox', 0.0, units='m')
        self.add_output('H_gearbox', 0.0, units='m')
        self.add_output('D_gearbox', 0.0, units='m')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        if discrete_inputs['direct_drive']: return

        '''
        gear_configuration = gear_configuration #Str(iotype='in', desc='string that represents the configuration of the gearbox (stage number and types)')
        shaft_factor = shaft_factor #Str(iotype='in', desc = 'normal or short shaft length')
        debug = debug
        #variables
        gear_ratio = gear_ratio #Float(iotype='in', desc='overall gearbox speedup ratio')
        planet_numbers = planet_numbers #Array(np.array([0.0,0.0,0.0,]), iotype='in', desc='number of planets in each stage')
        rotor_rpm = rotor_rpm #Float(iotype='in', desc='rotor rpm at rated power')
        rotor_diameter = rotor_diameter #Float(iotype='in', desc='rotor diameter')
        rotor_torque = rotor_torque #Float(iotype='in', units='N*m', desc='rotor torque at rated power')
        gearbox_input_cm = gearbox_input_cm #Float(0,iotype = 'in', units='m', desc ='gearbox position along x-axis')
    
        # outputs
        stage_masses = np.zeros(4) #Array(np.array([0.0, 0.0, 0.0, 0.0]), iotype='out', units='kg', desc='individual gearbox stage masses')
        gearbox_mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        gearbox_cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        gearbox_I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    
        gearbox_length = 0.0 #Float(iotype='out', units='m', desc='gearbox length')
        gearbox_height = 0.0 #Float(iotype='out', units='m', desc='gearbox height')
        gearbox_diameter = 0.0 #Float(iotype='out', units='m', desc='gearbox diameter')

        # initialize stage ratios
        stageRatio = np.zeros([3, 1])

        # filled in when ebxWeightEst is called
        stageTorque = np.zeros([len(stageRatio), 1])
        # filled in when ebxWeightEst is called
        stageMass = np.zeros([len(stageRatio), 1])
        stageType = stageTypeCalc(gear_configuration)
        stageRatio = stageRatioCalc(gear_ratio, planet_numbers, gear_configuration)

        m = gearboxWeightEst(gear_configuration, gear_ratio, planet_numbers, shaft_factor, rotor_torque)
        gearbox_mass = float(m)
        stage_masses = stageMass
        # calculate mass properties

        gearbox_length = (0.012 * rotor_diameter)
        gearbox_height = (0.015 * rotor_diameter)
        gearbox_diameter = (0.75 * gearbox_height)

        cm0 = gearbox_input_cm
        cm1 = 0.0
        # TODO validate or adjust factor. origin is modified to be above
        # bedplate top
        cm2 = 0.4 * gearbox_height
        gearbox_cm = np.array([cm0, cm1, cm2])

        I0 = gearbox_mass * (gearbox_diameter ** 2) / 8 \
          + (gearbox_mass / 2) * (gearbox_height ** 2) / 8
        I1 = gearbox_mass * (   0.5 * (gearbox_diameter ** 2) 
                                 + (2/3) * (gearbox_length ** 2) 
                                 + 0.25 * (gearbox_height ** 2)) / 8
        I2 = I1
        gearbox_I = np.array([I0, I1, I2])
        
        if debug:
            sys.stderr.write('GBOX: Mass {:.1f} kg  Len/Ht/Diam (m) {:.2f} {:.2f} {:.2f}\n'.format(gearbox_mass, 
                             gearbox_length, gearbox_height, gearbox_diameter))

        return(stage_masses.flatten(), gearbox_mass, gearbox_cm, gearbox_I.flatten(), gearbox_length, gearbox_height, gearbox_diameter)

    def stageTypeCalc(self, config):
        temp = []
        for character in config:
                if character == 'e':
                    temp.append(2)
                if character == 'p':
                    temp.append(1)
        return temp

    def stageMassCalc(self, indStageRatio, indNp, indStageType):
        # Computes the mass of an individual gearbox stage.

        # Application factor to include ring/housing/carrier weight
        Kr = 0.4
        Kgamma = 1.1

        # Select gamma by number of planets?
        if indNp == 3:
            Kgamma = 1.1
        elif indNp == 4:
            Kgamma = 1.1
        elif indNp == 5:
            Kgamma = 1.35

        if indStageType == 1: # parallel
            indStageMass = 1.0 + indStageRatio + indStageRatio**2 + (1.0 / indStageRatio)

        elif indStageType == 2: # epicyclic
            sunRatio = 0.5 * indStageRatio - 1.0
            indStageMass = Kgamma * ( (1 / indNp) 
                                    + (1 / (indNp * sunRatio))
                                    + sunRatio 
                                    + sunRatio**2 
                                    + Kr * ((indStageRatio - 1)**2) / indNp 
                                    + Kr * ((indStageRatio - 1)**2) / (indNp * sunRatio))
            
        return indStageMass

    def gearboxWeightEst(self, config, overallRatio, planet_numbers, shaft_factor, torque):
        # Computes the gearbox weight based on a surface durability criteria.

        ## Define Application Factors ##
        # Application factor for weight estimate
        Ka = 0.6
        Kshaft = 0.0
        Kfact = 0.0

        # K factor for pitting analysis
        #if rotor_torque < 200.0:
        if torque < 200.0:
            Kfact = 850.0
        #elif rotor_torque < 700.0:
        elif torque < 700.0:
            Kfact = 950.0
        else:
            Kfact = 1100.0

        # Unit conversion from Nm to inlb and vice-versa
        Kunit = 8.029 # 8.85075 to convert from N-m to in-lb?

        # Shaft length factor
        try:
            shaft_factor in ['normal','short']
        except ValueError:
            print("Invalid shaft_factor.  Must be either 'normal' or 'short'")
        else:
            if shaft_factor == 'normal':
                Kshaft = 1.0
            elif shaft_factor == 'short':
                Kshaft = 1.25

        # Individual stage torques
        #torqueTemp = rotor_torque
        torqueTemp = torque
        for s in range(len(stageRatio)):
            stageTorque[s] = torqueTemp / stageRatio[s]
            torqueTemp = stageTorque[s]
            stageMass[s] = Kunit * Ka / Kfact * stageTorque[s] \
                * stageMassCalc(stageRatio[s], planet_numbers[s], stageType[s])
            if debug:
                sys.stderr.write('GBOX::gbWE(): stage {} mass {:8.1f} kg  torque {:9.1f} N-m\n'.format(s, 
                                 stageMass[s][0], stageTorque[s][0]))

        gearboxWeight = (sum(stageMass)) * Kshaft

        return gearboxWeight

    def stageRatioCalc(self, overallRatio, planet_numbers, config):
        # Calculates individual stage ratios using either: Sunderland empirical relationships or a constrained optimization routine.

        K_r = 0

        x = np.zeros([3, 1])

        try:
            config in ['eep','eep_2','eep_3','epp']
        except ValueError:
            print("Invalid value for gearbox_configuration.  Must be one of: 'eep','eep_2','eep_3','epp'")
        else:
            if config == 'eep':
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r1 = 0
                K_r2 = 0  # 2nd stage structure weight coefficient
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) +
                    (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) + \
                    (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 +
                     K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
    
            elif config == 'eep_3':
                # fixes last stage ratio at 3
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r1 = 0
                K_r2 = 0.8  # 2nd stage structure weight coefficient
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 + K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                def constr3(x, overallRatio):
                    return x[2] - 3.0
    
                def constr4(x, overallRatio):
                    return 3.0 - x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2, constr3, constr4], consargs=[overallRatio], rhoend=1e-7)
    
            elif config == 'eep_2':
                # fixes final stage ratio at 2
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r1 = 0
                K_r2 = 1.6  # 2nd stage structure weight coefficient
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 + K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
            elif config == 'epp':
                # fixes last stage ratio at 3
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r = 0
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 +
                        K_r * ((x[0] - 1.0)**2) / B_1 + K_r * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                    + (1.0 / (x[0] * x[1])) * (1.0 + (1.0 / x[1]) + x[1] + x[1]**2) \
                    + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
    
            else:  # Should not execute since try/except checks for acceptable gearbox configuration types
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                K_r = 0.0
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1) + (x[0] / 2.0 - 1.0)**2 + K_r * ((x[0] - 1.0)**2) / B_1 + K_r * ((x[0] - 1)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1])) * (1.0 + (1.0 / x[1]) + x[1] + x[1]**2) \
                         + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                   return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
    
            return x
        
        '''
