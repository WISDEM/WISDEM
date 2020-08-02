
import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util

#-------------------------------------------------------------------------

class MainBearing(om.ExplicitComponent):
    ''' 
    MainBearings class is used to represent the main bearing components of a wind turbine drivetrain.
    '''
        
    def setup(self):
        self.add_discrete_input('bearing_type', 'CARB', desc='bearing mass type')
        self.add_input('D_bearing', 0.0, units='m', desc='bearing diameter/facewidth')
        self.add_input('D_shaft', 0.0, units='m', desc='Diameter of LSS shaft at bearing location')

        self.add_output('mb_max_defl_ang', 0.0, units='rad', desc='Maximum allowable deflection angle')
        self.add_output('mb_mass', 0.0, units='kg', desc='overall component mass')
        #self.add_output('mb_cm',   np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('mb_I',    np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        if type(discrete_inputs['bearing_type']) != type(''): raise ValueError('Bearing type input must be a string')
        btype = discrete_inputs['bearing_type'].upper()
        D_shaft = inputs['D_shaft']
        D_bearing = inputs['D_bearing']
        
        # assume low load rating for bearing
        if btype == 'CARB':  # p = Fr, so X=1, Y=0
            face_width = 0.2663 * D_shaft + .0435
            mass = 1561.4 * D_shaft**2.6007
            max_ang = np.deg2rad(0.5)

        elif btype == 'CRB':
            face_width = 0.1136 * D_shaft
            mass = 304.19 * D_shaft**1.8885
            max_ang = np.deg2rad(4.0 / 60.0)

        elif btype == 'SRB':
            face_width = 0.2762 * D_shaft
            mass = 876.7 * D_shaft**1.7195
            max_ang = 0.078

        #elif btype == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        #    face_width = 0.0839
        #    mass = 229.47 * D_shaft**1.8036
        #    max_ang = 0.002

        #elif btype == 'TRB1':
        #    face_width = 0.0740
        #    mass = 92.863 * D_shaft**.8399
        #    max_ang = 3.0 / 60.0 / 180.0 * np.pi

        elif btype == 'TRB':
            face_width = 0.1499 * D_shaft
            mass = 543.01 * D_shaft**1.9043
            max_ang = np.deg2rad(3.0 / 60.0)

        else:
            raise ValueError('Bearing type must be CARB / CRB / SRB / TRB')
            
        # add housing weight, but pg 23 of report says factor is 2.92 whereas this is 2.963
        mass += mass*(8000.0/2700.0)  

        # Consider the bearings a torus for MoI (https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
        I0 = 0.25*mass  * (4*(0.5*D_shaft)**2 + 3*(0.5*D_bearing)**2)
        I1 = 0.125*mass * (4*(0.5*D_shaft)**2 + 5*(0.5*D_bearing)**2)
        I = np.r_[I0, I1, I1]
        outputs['mb_mass'] = mass
        outputs['mb_I'] = I
        outputs['mb_max_defl_ang'] = max_ang
        

#-------------------------------------------------------------------

class HighSpeedSide(om.ExplicitComponent):
    '''
    HighSpeedShaft class
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def setup(self):

        self.add_discrete_input('direct_drive', False)

        self.add_input('rotor_diameter', 0.0, units='m', desc='rotor diameter')
        self.add_input('rotor_torque', 0.0, units='N*m', desc='rotor torque at rated power')
        self.add_input('gear_ratio', 1.0, desc='overall gearbox ratio')
        self.add_input('D_shaft_end', 0.0, units='m', desc='low speed shaft outer diameter')
        self.add_input('s_rotor', 0.0, units='m', desc='Generator rotor attachment to shaft s-coordinate')
        self.add_input('s_gearbox', 0.0, units='m', desc='Gearbox s-coordinate measured from bedplate')
        self.add_input('hss_input_length', 0.0, units='m', desc='high speed shaft length determined by user. Default 0.5m')
        self.add_input('rho', 0.0, units='kg/m**3', desc='material density')

        # returns
        self.add_output('hss_mass', 0.0, units='kg', desc='overall component mass')
        self.add_output('hss_cm', 0.0, units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('hss_I', np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('hss_length', 0.0, units='m', desc='length of high speed shaft')
        self.add_output('hss_diameter', 0.0, units='m', desc='diameter of high speed shaft')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        direct      = discrete_inputs['direct_drive']
        D_rotor     = float(inputs['rotor_diameter'])
        Q_rotor     = float(inputs['rotor_torque'])
        gear_ratio  = float(inputs['gear_ratio'])
        D_shaft     = float(inputs['D_shaft_end'])
        s_rotor     = float(inputs['s_rotor'])
        s_gearbox   = float(inputs['s_gearbox'])
        L_hss_input = float(inputs['hss_input_length'])
        rho         = float(inputs['rho'])

        # Regression based sizing
        # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines
        # DD Brake scaling derived by J.Keller under FOA 1981 support project
        if direct:
            m_hss_shaft = 0.0
            m_brake     = 1220. * 1e-6 * Q_rotor
            D_hss_shaft = L_hss_shaft = 0.0
        else:
            m_hss_shaft = 0.025 * Q_rotor / gear_ratio
            m_brake     = 0.5 * m_hss_shaft 
            D_hss_shaft = 1.5 * D_shaft # based on WindPACT relationships for full HSS / mechanical brake assembly
            L_hss_shaft = L_hss_input if L_hss_input > 0.0 else m_hss_shaft / (np.pi * (0.5*D_hss_shaft)**2 * rho)
        mass = m_brake + m_hss_shaft

    
        # Assume brake disc diameter and simple MoI
        D_disc = 0.01*D_rotor
        I      = np.zeros(3)
        I[0]   = 0.5*m_brake*(0.5*D_disc)**2
        I[1:]  = 0.25*m_brake*(0.5*D_disc)**2

        if direct:
            cm = s_rotor
        else:
            cm = 0.5*(s_rotor + s_gearbox)
  
            I[0]  += m_hss_shaft *     (0.5*D_hss_shaft)**2                   / 2.
            I[1:] += m_hss_shaft * (3.*(0.5*D_hss_shaft)**2 + L_hss_shaft**2) / 12.

        outputs['hss_mass'] = mass
        outputs['hss_cm'] = cm
        outputs['hss_I'] = I
        outputs['hss_length'] = L_hss_shaft
        outputs['hss_diameter'] = D_hss_shaft
        
#----------------------------------------------------------------------------------------------
        

class GeneratorSimple(om.ExplicitComponent):
    '''Generator class
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

        
    def setup(self):
        # variables
        self.add_discrete_input('direct_drive', False)
        self.add_input('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_input('machine_rating', val=0.0, units='kW', desc='machine rating of generator')
        self.add_input('rotor_torque', 0.0, units='N*m', desc='rotor torque at rated power')

        #returns
        self.add_output('R_generator', val=0.0, units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('generator_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('generator_I', val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        direct  = discrete_inputs['direct_drive']
        rating  = float(inputs['machine_rating'])
        D_rotor = float(inputs['rotor_diameter'])
        Q_rotor = float(inputs['rotor_torque'])
  
        if direct:
            massCoeff = 1e-3 * 37.68
            mass = massCoeff * Q_rotor
        else:
            massCoeff = np.mean([6.4737, 10.51, 5.34])
            massExp   = 0.9223
            mass = (massCoeff * rating ** massExp)
        outputs['generator_mass'] = mass
        
        # calculate mass properties
        length = 1.8 * 0.015 * D_rotor
        R_generator = 0.5 * 0.015 * D_rotor
        outputs['R_generator'] = R_generator
        
        I = np.zeros(3)
        I[0] = 0.5*R_generator**2
        I[1:] = (1./12.)*(3*R_generator**2 + length**2)
        outputs['generator_I'] = mass*I

        
#-------------------------------------------------------------------------------

class Electronics(om.ExplicitComponent):
    ''' Estimate mass of electronics based on rating, rotor diameter, and tower top diameter.  Empirical only, no load analysis.'''
        
    def setup(self):

        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating of the turbine')
        self.add_input('rotor_diameter', 0.0, units='m', desc='rotor diameter of turbine')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')

        self.add_output('electronics_mass', 0.0, units='kg', desc='overall component mass')
        self.add_output('electronics_cm', np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('electronics_I', np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        
    def compute(self, inputs, outputs):

        # Unpack inputs
        rating  = float(inputs['machine_rating'])
        D_rotor = float(inputs['rotor_diameter'])
        D_top   = float(inputs['D_top'])
    
        # Correlation based trends, assume box
        mass   = 2.4445*rating + 1599.0
        sides  = 0.015*D_rotor

        # CM location, just assume off to the side of the bedplate
        cm = np.zeros(3)
        cm[1] = 0.5*D_top + 0.5*sides
        cm[2] = 0.5*sides

        # MoI
        I = (1./6.)*mass*sides**2*np.ones(3)
        
        # Outputs
        outputs['electronics_mass'] = mass
        outputs['electronics_cm'] = cm
        outputs['electronics_I'] = I
        


#---------------------------------------------------------------------------------------------------------------

class YawSystem(om.ExplicitComponent):
    ''' Estimate mass of yaw system based on rotor diameter and tower top diameter.  Empirical only, no load analysis.'''

    def setup(self):
        # variables
        self.add_input('rotor_diameter', 0.0, units='m', desc='rotor diameter')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')
        self.add_input('rho', 0.0, units='kg/m**3', desc='material density')
        
        # outputs
        self.add_output('yaw_mass', 0.0, units='kg', desc='overall component mass')
        self.add_output('yaw_cm', np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('yaw_I', np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

    def compute(self, inputs, outputs):

        # Unpack inputs
        D_rotor = float(inputs['rotor_diameter'])
        D_top   = float(inputs['D_top'])
        rho     = float(inputs['rho'])

        # Estimate the number of yaw motors (borrowed from old DriveSE utilities)
        n_motors = 2*np.ceil(D_rotor / 30.0) - 2
  
        # Assume same yaw motors as Vestas V80 for now: Bonfiglioli 709T2M
        m_motor = 190.0
  
        # Assume friction plate surface width is 1/10 the diameter and thickness scales with rotor diameter
        m_frictionPlate = rho * np.pi*D_top * (0.1*D_top) * (1e-3*D_rotor)

        # Total mass estimate
        outputs['yaw_mass'] = m_frictionPlate + n_motors*m_motor
  
        # Assume cm is at tower top (cm=0,0,0) and mass is non-rotating (I=0,..), so leave at default value of 0s
        outputs['yaw_cm'] = np.zeros(3)
        outputs['yaw_I']  = np.zeros(3)

#---------------------------------------------------------------------------------------------------------------

class MiscNacelleComponents(om.ExplicitComponent):
    ''' Estimate mass properties of miscellaneous other ancillary components in the nacelle.'''

    def setup(self):
        self.add_discrete_input('upwind', True, desc='Flag whether the design is upwind or downwind') 

        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating of the turbine')
        self.add_input('L_bedplate', 0.0, units='m', desc='Length of bedplate') 
        self.add_input('H_bedplate', 0.0, units='m', desc='height of bedplate')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')
        self.add_input('bedplate_mass', 0.0, units='kg', desc='Bedplate mass')
        self.add_input('bedplate_cm', np.zeros(3), units='m', desc='Bedplate center of mass')
        self.add_input('bedplate_I', np.zeros(6), units='kg*m**2', desc='Bedplate mass moment of inertia about base')
        self.add_input('R_generator', 0.0, units='m', desc='Generatour outer diameter')
        self.add_input('overhang',0.0, units='m', desc='Overhang of rotor from tower along x-axis in yaw-aligned c.s.') 
        self.add_input('s_rotor', 0.0, units='m', desc='Generator rotor attachment to shaft s-coordinate')
        self.add_input('s_stator', 0.0, units='m', desc='Generator stator attachment to nose s-coordinate')
        self.add_input('rho_fiberglass', 0.0, units='kg/m**3', desc='material density of fiberglass')
        
        self.add_output('hvac_mass', 0.0, units='kg', desc='component mass')
        self.add_output('hvac_cm', 0.0, units='m', desc='component center of mass')
        self.add_output('hvac_I', np.zeros(3), units='m', desc='component mass moments of inertia')
        
        self.add_output('mainframe_mass', 0.0, units='kg', desc='component mass')
        self.add_output('mainframe_cm', np.zeros(3), units='m', desc='component center of mass')
        self.add_output('mainframe_I', np.zeros(6), units='m', desc='component mass moments of inertia')
        
        self.add_output('cover_mass', 0.0, units='kg', desc='component mass')
        self.add_output('cover_cm', np.zeros(3), units='m', desc='component center of mass')
        self.add_output('cover_I', np.zeros(3), units='m', desc='component mass moments of inertia')


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        upwind      = discrete_inputs['upwind']
        rating      = float(inputs['machine_rating'])
        L_bedplate  = float(inputs['L_bedplate'])
        H_bedplate  = float(inputs['H_bedplate'])
        D_bedplate  = float(inputs['D_top'])
        R_generator = float(inputs['R_generator'])
        m_bedplate  = float(inputs['bedplate_mass'])
        cm_bedplate = inputs['bedplate_cm']
        I_bedplate  = inputs['bedplate_I']
        overhang    = float(inputs['overhang'])
        s_rotor     = float(inputs['s_rotor'])
        s_stator    = float(inputs['s_stator'])
        rho_fiberglass = float(inputs['rho_fiberglass'])

        # For the nacelle cover, imagine a box from the bedplate to the hub in length and around the generator in width, height, with 10% margin in each dim
        L_cover  = 1.1 * (overhang + 0.5*D_bedplate)
        W_cover  = 1.1 * 2*R_generator
        H_cover  = 1.1 * (R_generator + np.maximum(R_generator,H_bedplate))
        A_cover  = 2*(L_cover*W_cover + L_cover*H_cover + H_cover*W_cover)
        t_cover  = 0.04 # 5cm thick walls?
        m_cover  = A_cover * t_cover * rho_fiberglass
        cm_cover = np.array([0.5*L_cover-0.5*D_bedplate, 0.0, 0.5*H_cover])
        I_cover  = m_cover*np.array([H_cover**2 + W_cover**2 - (H_cover-t_cover)**2 - (W_cover-t_cover)**2,
                                     H_cover**2 + L_cover**2 - (H_cover-t_cover)**2 - (L_cover-t_cover)**2,
                                     W_cover**2 + L_cover**2 - (W_cover-t_cover)**2 - (L_cover-t_cover)**2]) / 12.
        if upwind: cm_cover[0] *= -1.0
        outputs['cover_mass'] = m_cover
        outputs['cover_cm']   = cm_cover
        outputs['cover_I' ]   = I_cover
        
        # Regression based estimate on HVAC mass
        m_hvac       = 0.08 * rating
        cm_hvac      = 0.5*(s_rotor + s_stator)
        I_hvac       = m_hvac * (0.75*R_generator)**2
        outputs['hvac_mass'] = m_hvac
        outputs['hvac_cm']   = cm_hvac
        outputs['hvac_I' ]   = I_hvac*np.array([1.0, 0.5, 0.5])

        # Platforms as a fraction of bedplate mass and bundling it to call it 'mainframe'
        platforms_coeff = 0.125
        m_mainframe  = platforms_coeff * m_bedplate
        I_mainframe  = platforms_coeff * I_bedplate
        outputs['mainframe_mass'] = m_mainframe
        outputs['mainframe_cm']   = np.zeros(3)
        outputs['mainframe_I' ]   = I_mainframe

#--------------------------------------------
class NacelleSystemAdder(om.ExplicitComponent): #added to drive to include electronics
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def setup(self):
        self.add_discrete_input('upwind', True, desc='Flag whether the design is upwind or downwind') 
        self.add_discrete_input('uptower', True, desc='Power electronics are placed in the nacelle at the tower top')

        self.add_input('tilt', 0.0, units='deg', desc='Shaft tilt') 

        self.add_input('mb1_mass', 0.0, units='kg', desc='component mass')
        self.add_input('mb1_cm', 0.0, units='m', desc='component CM')
        self.add_input('mb1_I', np.zeros(3), units='kg*m**2', desc='component I')

        self.add_input('mb2_mass', 0.0, units='kg', desc='component mass')
        self.add_input('mb2_cm', 0.0, units='m', desc='component CM')
        self.add_input('mb2_I', np.zeros(3), units='kg*m**2', desc='component I')

        self.add_input('gearbox_mass', 0.0, units='kg', desc='component mass')
        self.add_input('gearbox_cm', np.zeros(3), units='m', desc='component CM')
        self.add_input('gearbox_I', np.zeros(3), units='kg*m**2', desc='component I')

        self.add_input('hss_mass', 0.0, units='kg', desc='component mass')
        self.add_input('hss_cm', 0.0, units='m', desc='component CM')
        self.add_input('hss_I', np.zeros(3), units='kg*m**2', desc='component I')

        self.add_input('generator_mass', 0.0, units='kg', desc='component mass')
        self.add_input('generator_cm', 0.0, units='m', desc='component CM')
        self.add_input('generator_I', np.zeros(3), units='kg*m**2', desc='component I')

        self.add_input('nose_mass', val=0.0, units='kg', desc='Nose mass')
        self.add_input('nose_cm', val=0.0, units='m', desc='Nose center of mass along nose axis from bedplate')
        self.add_input('nose_I', val=np.zeros(3), units='kg*m**2', desc='Nose moment of inertia around cm in axial (hub-aligned) c.s.')

        self.add_input('lss_mass', val=0.0, units='kg', desc='LSS mass')
        self.add_input('lss_cm', val=0.0, units='m', desc='LSS center of mass along shaft axis from bedplate')
        self.add_input('lss_I', val=np.zeros(3), units='kg*m**2', desc='LSS moment of inertia around cm in axial (hub-aligned) c.s.')
        
        self.add_input('electronics_mass', 0.0, units='kg', desc='component mass')
        self.add_input('electronics_cm', np.zeros(3), units='m', desc='component CM')
        self.add_input('electronics_I', np.zeros(3), units='kg*m**2', desc='component I')

        self.add_input('yaw_mass', 0.0, units='kg', desc='overall component mass')
        self.add_input('yaw_cm', np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_input('yaw_I', np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        self.add_input('bedplate_mass', 0.0, units='kg', desc='component mass')
        self.add_input('bedplate_cm', np.zeros(3), units='m', desc='component CM')
        self.add_input('bedplate_I', np.zeros(6), units='kg*m**2', desc='component I')

        self.add_input('hvac_mass', 0.0, units='kg', desc='component mass')
        self.add_input('hvac_cm', 0.0, units='m', desc='component center of mass')
        self.add_input('hvac_I', np.zeros(3), units='m', desc='component mass moments of inertia')
        
        self.add_input('mainframe_mass', 0.0, units='kg', desc='component mass')
        self.add_input('mainframe_cm', np.zeros(3), units='m', desc='component center of mass')
        self.add_input('mainframe_I', np.zeros(6), units='m', desc='component mass moments of inertia')
        
        self.add_input('cover_mass', 0.0, units='kg', desc='component mass')
        self.add_input('cover_cm', np.zeros(3), units='m', desc='component center of mass')
        self.add_input('cover_I', np.zeros(3), units='m', desc='component mass moments of inertia')
        
        # returns
        self.add_output('other_mass', 0.0, units='kg', desc='overall component mass')
        self.add_output('nacelle_mass', 0.0, units='kg', desc='overall component mass')
        self.add_output('nacelle_cm', np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('nacelle_I', np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        Cup  = -1.0 if discrete_inputs['upwind'] else 1.0
        tilt = float(np.deg2rad(inputs['tilt']))
        
        components = ['mb1','mb2','lss','hss','gearbox','generator','hvac',
                      'nose','bedplate','mainframe','yaw','cover']
        if discrete_inputs['uptower']: components.append('electronics')

        # Mass and CofM summaries first because will need them for I later
        m_nac = 0.0
        cm_nac = np.zeros(3)
        for k in components:
            m_i  = inputs[k+'_mass']
            cm_i = inputs[k+'_cm']

            # If cm is (x,y,z) then it is already in tower-top c.s.  If it is a scalar, it is in distance from bedplate and we have to convert
            if len(cm_i) == 1:
                cm_i = cm_i * np.array([Cup*np.cos(tilt), 0.0, np.sin(tilt)])
            
            m_nac  += m_i
            cm_nac += m_i*cm_i

        # Complete CofM calculation
        cm_nac /= m_nac

        # Now find total I about nacelle CofM
        I_nac  = util.assembleI( np.zeros(6) )
        for k in components:
            m_i  = inputs[k+'_mass']
            cm_i = inputs[k+'_cm']
            I_i  = inputs[k+'_I']

            # Rotate MofI if in hub c.s.
            if len(cm_i) == 1:
                cm_i = cm_i * np.array([Cup*np.cos(tilt), 0.0, np.sin(tilt)])
                I_i  = util.rotateI(I_i, -Cup*tilt, axis='y')
            else:
                I_i  = np.r_[I_i, np.zeros(3)]
                
            r       = cm_i - cm_nac
            I_nac  += util.assembleI(I_i) + m_i*(np.dot(r, r)*np.eye(3) - np.outer(r, r))

        outputs['nacelle_mass'] = m_nac
        outputs['nacelle_cm']   = cm_nac
        outputs['nacelle_I']    = util.unassembleI(I_nac)
        outputs['other_mass']   = (inputs['hss_mass'] + inputs['hvac_mass'] + inputs['mainframe_mass'] +
                                   inputs['yaw_mass'] + inputs['cover_mass'] + inputs['electronics_mass'])

#--------------------------------------------


class RNA_Adder(om.ExplicitComponent):
    
    def setup(self):

        self.add_discrete_input('upwind', True, desc='Flag whether the design is upwind or downwind') 
        self.add_input('tilt', 0.0, units='deg', desc='Shaft tilt') 
        self.add_input('L_drive',0.0, units='m', desc='Length of drivetrain from bedplate to hub flang') 

        self.add_input('blades_mass', 0.0, units='kg', desc='Mass of all bladea')
        self.add_input('hub_system_mass', 0.0, units='kg', desc='Mass of hub system (hub + spinner + pitch)')
        self.add_input('nacelle_mass', 0.0, units='kg', desc='Mass of nacelle system')

        self.add_input('hub_system_cm', 0.0, units='m', desc='Hub center of mass from hub flange in hub c.s.')
        self.add_input('nacelle_cm', np.zeros(3), units='m', desc='Nacelle center of mass relative to tower top in yaw-aligned c.s.')

        self.add_input('blades_I', np.zeros(6), units='kg*m**2', desc='Mass moments of inertia of all blades about hub center')
        self.add_input('hub_system_I', np.zeros(3), units='kg*m**2', desc='Mass moments of inertia of hub system about its CofM')
        self.add_input('nacelle_I', np.zeros(6), units='kg*m**2', desc='Mass moments of inertia of nacelle about its CofM')

        # outputs
        self.add_output('rotor_mass', 0.0, units='kg', desc='Mass of blades and hub system')
        self.add_output('rna_mass', 0.0, units='kg', desc='Total RNA mass')
        self.add_output('rna_cm', np.zeros(3), units='m', desc='RNA center of mass relative to tower top in yaw-aligned c.s.')
        self.add_output('rna_I_TT', np.zeros(6), units='kg*m**2', desc='Mass moments of inertia of RNA about tower top in yaw-aligned coordinate system')


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        Cup  = -1.0 if discrete_inputs['upwind'] else 1.0
        tilt = float(np.deg2rad(inputs['tilt']))
        
        rotor_mass = inputs['blades_mass'] + inputs['hub_system_mass']
        nac_mass   = inputs['nacelle_mass']
        
        # rna mass
        outputs['rotor_mass'] = rotor_mass
        outputs['rna_mass']   = rotor_mass + nac_mass

        # rna cm
        hub_cm  = inputs['hub_system_cm']
        L_drive = inputs['L_drive']
        hub_cm  = (L_drive+hub_cm) * np.array([Cup*np.cos(tilt), 0.0, np.sin(tilt)])
        outputs['rna_cm'] = (rotor_mass*hub_cm + nac_mass*inputs['nacelle_cm']) / outputs['rna_mass']

        # rna I
        hub_I    = util.assembleI( util.rotateI(inputs['hub_system_I'], -Cup*tilt, axis='y') )
        blades_I = util.assembleI(inputs['blades_I'])
        nac_I    = util.assembleI(inputs['nacelle_I'])
        rotor_I  = blades_I + hub_I

        R = hub_cm
        rotor_I_TT = rotor_I + rotor_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        R = inputs['nacelle_cm']
        nac_I_TT = nac_I + nac_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        outputs['rna_I_TT'] = util.unassembleI(rotor_I_TT + nac_I_TT)
#--------------------------------------------
