
import numpy as np
import openmdao.api as om

#-------------------------------------------------------------------------

class MainBearing(om.ExplicitComponent):
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

        
    def setup(self):
        self.add_discrete_input('bearing_type', val='CARB', desc='bearing mass type')
        self.add_input('D_bearing', val=0.0, units='m', desc='bearing diameter/facewidth')
        self.add_input('D_shaft', val=0.0, units='m', desc='Diameter of LSS shaft at bearing location')

        self.add_output('mb_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('mb_cm',   val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('mb_I',    val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # assume low load rating for bearing
        if btype == 'CARB':  # p = Fr, so X=1, Y=0
            face_width = 0.2663 * D_shaft + .0435
            mass = 1561.4 * D_shaft**2.6007
            Bearing_Limit = 0.5 / 180 * pi

        elif btype == 'CRB':
            face_width = 0.1136 * D_shaf
            mass = 304.19 * D_shaft**1.8885
            Bearing_Limit = 4.0 / 60.0 / 180.0 * pi

        elif btype == 'SRB':
            face_width = 0.2762 * D_shaft
            mass = 876.7 * D_shaft**1.7195
            Bearing_Limit = 0.078

        #elif btype == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        #    face_width = 0.0839
        #    mass = 229.47 * D_shaft**1.8036
        #    Bearing_Limit = 0.002

        #elif btype == 'TRB1':
        #    face_width = 0.0740
        #    mass = 92.863 * D_shaft**.8399
        #    Bearing_Limit = 3.0 / 60.0 / 180.0 * pi

        elif btype == 'TRB':
            face_width = 0.1499 * D_shaf
            tmass = 543.01 * D_shaft**1.9043
            Bearing_Limit = 3.0 / 60.0 / 180.0 * pi
        
        # add housing weight, but pg 23 of report says factor is 2.92 whereas this is 2.963
        mass += mass*(8000.0/2700.0)  

        b1I0 = mass * (0.5*D_bearing)** 2
        I = np.r_[b1I0, 0.5*b1I0*np.ones(2)]
        

#-------------------------------------------------------------------

class HighSpeedSide(om.ExplicitComponent):
    '''
    HighSpeedShaft class
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def setup(self):

        # variables
        self.add_input('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_input('rotor_torque', val=0.0, units='N*m', desc='rotor torque at rated power')
        self.add_input('gear_ratio', val=0.0, desc='overall gearbox ratio')
        self.add_input('D_shaft_end', val=0.0, units='m', desc='low speed shaft outer diameter')
        self.add_input('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_input('gearbox_height', val=0.0, units='m', desc='gearbox height')
        self.add_input('gearbox_cm', val=np.zeros(3), units='m', desc='gearbox cm [x,y,z]')
        self.add_input('hss_input_length', val=0.0, units='m', desc='high speed shaft length determined by user. Default 0.5m')

        # returns
        self.add_output('hss_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('hss_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('hss_I', val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('hss_length', val=0.0, units='m', desc='length of high speed shaft')
        self.add_output('hss_diameter', val=0.0, units='m', desc='diameter of high speed shaft')

    def compute(self, inputs, outputs):

        # Unpack inputs
        D_rotor     = float(inputs['rotor_diameter'])
        Q_rotor     = float(inputs['rotor_torque'])
        gear_ratio  = float(inputs['gear_ratio'])
        D_shaft     = float(inputs['D_shaft_end'])
        L_gearbox   = float(inputs['gearbox_length'])
        H_gearbox   = float(inputs['gearbox_height'])
        cm_gearbox  = float(inputs['gearbox_cm'])
        L_hss_input = float(inputs['hss_input_length'])

        # Regression based sizing
        # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines
        # DD Brake scaling derived by J.Keller under FOA 1981 support project
        if direct:
            m_hss_shaft = 0.0
            m_brake     = 1220. * 1e-6 * Q_rotor
        else:
            m_hss_shaft = 0.025 * Q_rotor / gear_ratio
            m_brake     = 0.5 * m_hss_shaft 
        mass = m_brake + m_hss_shaft
  
        D_hss_shaft = 1.5 * D_shaft # based on WindPACT relationships for full HSS / mechanical brake assembly
        L_hss_shaft = L_hss_input if L_hss_input > 0.0 else m_hss_shaft / (np.pi * (0.5*D_hss_shaft)**2 * rho)
  
        # Assume brake disc diameter and simple MoI
        D_disc = 0.01*D_rotor
        I      = np.zeros(3)
        I[0]   = 0.5*m_brake*(0.5*D_disc)**2
        I[1:]  = 0.25*m_brake*(0.5*D_disc)**2

        cm = np.zeros(3)
        if direct:
            cm[0] = x_rotor
            cm[2] = z_rotor
        else:
            cm = cm_gearbox.copy()
            cm[0] += 0.5*L_gearbox + 0.5*hss_length
            cm[2] += 0.2*H_gearbox
  
            I[0]  += m_hss_shaft *     (0.5*D_hss_shaft)**2                   / 2.
            I[1:] += m_hss_shaft * (3.*(0.5*D_hss_shaft)**2 + L_hss_shaft**2) / 12.

        outputs['hss_mass'] = mass
        outputs['hss_cm'] = cm
        outputs['hss_I'] = I
        outputs['hss_length'] = L_hss_shaft
        outputs['hss_diameter'] = D_hss_shaft
        
        
        
#-------------------------------------------------------------------------------

class Electronics(om.ExplicitComponent):
    ''' Estimate mass of electronics based on rating, rotor diameter, and tower top diameter.  Empirical only, no load analysis.'''
        
    def setup(self):

        self.add_input('machine_rating', val=0.0, units='kW', desc='machine rating of the turbine')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')
        self.add_input('rotor_diameter', val=0.0, units='m', desc='rotor diameter of turbine')
        #self.add_discrete_input('uptower', True)

        self.add_output('electronics_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('electronics_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('electronics_I', val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        
    def compute(self, inputs, outputs):

        # Unpack inputs
        rating  = inputs['machine_rating']
        D_top   = float(inputs['D_top'])
        D_rotor = float(inputs['rotor_diameter'])
    
        # Correlation based trends
        mass   = 2.4445*rating + 1599.0
        width  = 0.5*D_top
        height = 0.016*D_rotor #similar to gearbox
        length = 0.012*D_rotor #similar to gearbox

        # CM location- can be a user input
        cm = np.zeros(3)
        if direct_drive:
            cm[0] = inputs['x_bedplate'][0]
            cm[2] = inputs['x_bedplate_outer'][0] + 0.5*height
        else:
            cm[0] = inputs['x_electronics']
            cm[2] = 0.6*inputs['cm_generator'][2]

        # MoI
        def get_I(d1,d2,mass):
            return (d1**2 + d2**2)/12.
            
        I = np.zeros(3)
        I[0] = mass*get_I(height, width)
        I[1] = mass*get_I(length, height)
        I[2] = mass*get_I(length, width)

        # Outputs
        outputs['electronics_mass'] = mass
        outputs['electronics_cm'] = cm
        outputs['electronics_I'] = I
        


#---------------------------------------------------------------------------------------------------------------

class YawSystem(om.ExplicitComponent):
    ''' Estimate mass of yaw system based on rotor diameter and tower top diameter.  Empirical only, no load analysis.'''

    def setup(self):
        # variables
        self.add_input('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        
        # outputs
        self.add_output('yaw_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('yaw_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('yaw_I', val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

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
        outputs['yaw_mass'] = m_frictionPlate frictionPlateMass + n_motors*m_motor
  
        # Assume cm is at tower top (cm=0,0,0) and mass is non-rotating (I=0,..), so just set them directly
        outputs['yaw_cm'] = np.zeros(3)
        outputs['yaw_I']  = np.zeros(6)
        

#---------------------------------------------------------------------------------------------------------------

class MiscNacelleComponents(om.ExplicitComponent):
    ''' Estimate mass properties of miscellaneous other ancillary components in the nacelle.'''

    def setup(self):
        # variables
        self.add_input('L_bedplate', 0.0, units='m', desc='Length of bedplate') 
        self.add_input('H_bedplate', 0.0, units='m', desc='height of bedplate')
        self.add_input('D_bedplate_base', val=np.zeros(n_points), units='m', desc='Bedplate diameters')
        self.add_input('rho_fiberglass', val=0.0, units='kg/m**3', desc='material density of fiberglass')
        
        # outputs
        self.add_output('hvac_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('hvac_cm', val=np.zeros(3), units='m', desc='component center of mass')
        self.add_output('hvac_I', val=np.zeros(3), units='m', desc='component mass moments of inertia')
        
        self.add_output('mainframe_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('mainframe_cm', val=np.zeros(3), units='m', desc='component center of mass')
        self.add_output('mainframe_I', val=np.zeros(3), units='m', desc='component mass moments of inertia')
        
        self.add_output('cover_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('cover_cm', val=np.zeros(3), units='m', desc='component center of mass')
        self.add_output('cover_I', val=np.zeros(3), units='m', desc='component mass moments of inertia')


    def compute(self, inputs, outputs):

        # Unpack inputs
        D_generator = float(inputs['generator_outer_diameter'])
        L_bedplate  = float(inputs['L_bedplate'])
        H_bedplate  = float(inputs['H_bedplate'])
        D_bedplate  = float(inputs['D_bedplate_base'])
        D_generator = float(inputs['D_generator'])
        overhang    = float(inputs['overhang'])
        rho_fiberglass = float(inputs['rho_fiberglass'])

        # For the nacelle cover, imagine a box from the bedplate to the hub in length and around the generator in width, height, with 10% margin in each dim
        L_cover  = 1.1 * (overhang + 0.5*D_bedplate[-1])
        W_cover  = 1.1 * D_generator
        H_cover  = 1.1 * (0.5*D_generator + H_bedplate)
        A_cover  = 2*(L_cover*W_cover + L_cover*H_cover + H_cover*W_cover)
        t_cover  = 0.05 # cm thick walls?
        m_cover  = A_cover * t_cover * rho_fiberglass
        cm_cover = np.array([0.5*L_cover-0.5*D_bedplate[-1], 0.0, 0.5*H_cover])
        I_cover  = m_cover*np.array([H_cover**2 + W_cover**2 - (H_cover-t_cover)**2 - (W_cover-t_cover)**2,
                                     H_cover**2 + L_cover**2 - (H_cover-t_cover)**2 - (L_cover-t_cover)**2,
                                     W_cover**2 + L_cover**2 - (W_cover-t_cover)**2 - (L_cover-t_cover)**2]) / 12.
        if upwind: cm_cover[0] *= -1.0
        outputs['cover_mass'] = m_cover
        outputs['cover_cm']   = cm_cover
        outputs['cover_I' ]   = I_cover
        
        # Regression based estimate on HVAC mass
        m_hvac       = 0.08 * rating
        cm_hvac      = cm_generator.copy()
        I_hvac       = (m_hvac / m_generator) * I_generator
        outputs['hvac_mass'] = m_hvac
        outputs['hvac_cm']   = cm_hvac
        outputs['hvac_I' ]   = I_hvac

        # Platforms as a fraction of bedplate mass and bundling it to call it 'mainframe'
        m_mainframe  = platforms_coeff * m_bedplate
        cm_mainframe = cm_bedplate.copy()
        I_mainframe  = platforms_coeff * I_bedplate
        outputs['mainframe_mass'] = m_mainframe
        outputs['mainframe_cm']   = cm_mainframe
        outputs['mainframe_I' ]   = I_mainframe

#--------------------------------------------
class NacelleSystemAdder(om.ExplicitComponent): #added to drive to include electronics
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def setup(self):
        self.add_discrete_input('crane', val=True, desc='onboard crane present')

        self.add_input('machine_rating', val=0.0, units='kW', desc='machine rating')

        self.add_input('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('lss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('lss_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        self.add_input('mb1_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('mb2_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('mb1_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('mb2_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('mb1_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('mb2_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        self.add_input('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('gearbox_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('gearbox_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        self.add_input('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('hss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('hss_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        self.add_input('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('generator_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('generator_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        self.add_input('electronics_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('electronics_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('electronics_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        self.add_input('yaw_mass', val=0.0, units='kg', desc='mass of yaw system')
        self.add_input('mainframe_mass', val=0.0, units='kg', desc='component mass')

        self.add_input('bedplate_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('bedplate_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('bedplate_length', val=0.0, units='m', desc='component length')
        self.add_input('bedplate_width', val=0.0, units='m', desc='component width')
        self.add_input('bedplate_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        # returns

        self.add_output('nacelle_length', val=0.0, units='m', desc='component length')
        self.add_output('nacelle_width', val=0.0, units='m', desc='component width')
        self.add_output('nacelle_height', val=0.0, units='m', desc='component height')
        self.add_output('nacelle_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('nacelle_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('nacelle_I', val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        
    def compute(self, inputs, outputs):
        # Unpack inputs
        yaw_mass = yaw_mass
        lss_mass = lss_mass
        mb1_mass = mb1_mass
        mb2_mass = mb2_mass
        gearbox_mass = gearbox_mass
        hss_mass = hss_mass
        generator_mass = generator_mass
        bedplate_mass = bedplate_mass
        mainframe_mass = mainframe_mass
        lss_cm = lss_cm
        mb1_cm = mb1_cm
        mb2_cm = mb2_cm
        cm_gearbox = gearbox_cm
        hss_cm = hss_cm
        generator_cm = generator_cm
        bedplate_cm = bedplate_cm
        lss_I = lss_I
        mb1_I = mb1_I
        mb2_I = mb2_I
        gearbox_I = gearbox_I
        hss_I = hss_I
        generator_I = generator_I
        bedplate_I = bedplate_I
        electronics_mass = electronics_mass
        electronics_cm = electronics_cm
        electronics_I = electronics_I
        rating = inputs['machine_rating']
        lss_mass = lss_mass
        mb1_mass = mb1_mass
        mb2_mass = mb2_mass
        gearbox_mass = gearbox_mass
        hss_mass = hss_mass
        generator_mass = generator_mass
        bedplate_mass = bedplate_mass
        bedplate_length = bedplate_length
        bedplate_width = bedplate_width
        electronics_mass = electronics_mass
        
        # yaw system weight calculations based on total system mass above yaw system
        m_above_yaw =  (lss_mass + 
                                mb1_mass + mb2_mass + 
                                gearbox_mass + 
                                hss_mass + 
                                generator_mass + 
                                mainframe_mass + 
                                electronics_mass +
                                electrical_mass + 
                                converter_mass + 
                                hvac_mass +
                                cover_mass)

        length      = bedplate_length                              # nacelle length [m] based on bedplate length
        width       = bedplate_width                        # nacelle width [m] based on bedplate width
        height      = (2.0 / 3.0) * length                         # nacelle height [m] calculated based on cladding area
        
        # returns
        nacelle_mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        nacelle_cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), units='m', iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        nacelle_I = np.zeros(3) # Array(np.array([0.0, 0.0, 0.0]), units='kg*m**2', iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        # aggregation of nacelle mass
        nacelle_mass = (m_above_yaw + yaw_mass)
  
        # calculation of mass center and moments of inertia
        nacelle_cm = ( (lss_mass*lss_cm
                           + electronics_mass*electronics_cm 
                           + mb1_mass*mb1_cm 
                           + mb2_mass*mb2_cm 
                           + gearbox_mass*cm_gearbox 
                           + hss_mass*hss_cm 
                           + generator_mass*generator_cm 
                           + mainframe_mass*bedplate_cm 
                           + yaw_mass*np.zeros(3))
                      / (lss_mass + mb1_mass + mb2_mass + gearbox_mass +
                         hss_mass + generator_mass + mainframe_mass + yaw_mass) )
  
        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems

        def appendI(xmass, xcm, xI):
            r    = xcm - nacelle_cm
            Icg  = assembleI( np.r_[xI, np.zeros(3)] ) # not used
            Iadd = xmass*(np.dot(r, r)*np.eye(3) - np.outer(r, r))
            return Iadd

        I   = np.zeros((3,3))
        I += appendI(lss_mass, lss_cm, lss_I)
        I += appendI(hss_mass, hss_cm, hss_I)
        I += appendI(mb1_mass, mb1_cm, mb1_I)
        I += appendI(mb2_mass, mb2_cm, mb2_I)
        I += appendI(gearbox_mass, cm_gearbox, gearbox_I)
        I += appendI(electronics_mass, electronics_cm, electronics_I)
        I += appendI(generator_mass, generator_cm, generator_I)
        # Mainframe mass includes bedplate mass and other components that assume the bedplate cm
        I += appendI(mainframe_mass, bedplate_cm, (mainframe_mass/bedplate_mass)*bedplate_I)
        nacelle_I = unassembleI(I)


