
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
            tmass = 304.19 * D_shaft**1.8885
            Bearing_Limit = 4.0 / 60.0 / 180.0 * pi

        elif btype == 'SRB':
            face_width = 0.2762 * D_shaft
            mass = 876.7 * D_shaft**1.7195
            Bearing_Limit = 0.078

        elif btype == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
            face_width = 0.0839
            mass = 229.47 * D_shaft**1.8036
            Bearing_Limit = 0.002

        elif btype == 'TRB1':
            face_width = 0.0740
            mass = 92.863 * D_shaft**.8399
            Bearing_Limit = 3.0 / 60.0 / 180.0 * pi

        elif btype == 'TRB2':
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
        

#-------------------------------------------------------------------------------

class AboveYawMassAdder_OM(om.ExplicitComponent):
    ''' AboveYawMassAdder_OM class
          The AboveYawMassAdder_OM class is used to represent the masses of all parts of a wind turbine drivetrain that
          are above the yaw system.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    def initialize(self):
        self.options.declare('debug', default=False)

    def setup(self):
        # variables
        self.add_input('machine_rating', val=0.0, units='kW', desc='machine rating')
        self.add_input('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('mb1_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('mb2_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('bedplate_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('bedplate_length', val=0.0, units='m', desc='component length')
        self.add_input('bedplate_width', val=0.0, units='m', desc='component width')
        self.add_input('electronics_mass', val=0.0, units='kg', desc='component mass')

        self.add_discrete_input('crane', val=True, desc='onboard crane present')
        
        # returns
        self.add_output('electrical_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('converter_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('hvac_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('controls_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('platforms_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('crane_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('mainframe_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('cover_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('above_yaw_mass', val=0.0, units='kg', desc='total mass above yaw system')
        self.add_output('nacelle_length', val=0.0, units='m', desc='component length')
        self.add_output('nacelle_width', val=0.0, units='m', desc='component width')
        self.add_output('nacelle_height', val=0.0, units='m', desc='component height')
        

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        aboveyawmass = dc.AboveYawMassAdder(discrete_inputs['crane'])

        (outputs['electrical_mass'], outputs['converter_mass'], outputs['hvac_mass'], outputs['controls_mass'], 
         outputs['platforms_mass'], outputs['crane_mass'], outputs['mainframe_mass'], outputs['cover_mass'], 
         outputs['above_yaw_mass'], outputs['nacelle_length'], outputs['nacelle_width'], outputs['nacelle_height']) \
            = aboveyawmass.compute(inputs['machine_rating'], inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], 
                    inputs['gearbox_mass'], inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], 
                    inputs['bedplate_length'], inputs['bedplate_width'], inputs['electronics_mass'])
        
        if self.options['debug']:
            print('AYMA IN: {:.1f} kW BPl {:.1f} m BPw {:.1f} m'.format(
                  inputs['machine_rating'],inputs['bedplate_length'], inputs['bedplate_width']))
            print('AYMA IN  masses (kg): LSS {:.1f} MB1 {:.1f} MB2 {:.1f} GBOX {:.1f} HSS {:.1f} GEN {:.1f} BP {:.1f} TFRM {:.1f}'.format(
                  inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'],
                  inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], inputs['electronics_mass']))
            print('AYMA OUT masses (kg) : E {:.1f} VSE {:.1f} HVAC {:.1f} CNTL {:.1f} PTFM {:.1f} CRN {:.1f} MNFRM {:.1f} CVR {:.1f} AYM {:.1f}'.format( 
                  outputs['electrical_mass'], outputs['converter_mass'], outputs['hvac_mass'], outputs['controls_mass'],
                  outputs['platforms_mass'], outputs['crane_mass'], outputs['mainframe_mass'], outputs['cover_mass'],
                  outputs['above_yaw_mass']))
            print('AYMA OUT nacelle (m): L {:.2f} W {:.2f} H {:.2f}'.format( 
                 outputs['nacelle_length'], outputs['nacelle_width'], outputs['nacelle_height']))


        # variables
        rating = inputs['machine_rating']
        self.lss_mass = lss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb1_mass = mb1_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb2_mass = mb2_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.gearbox_mass = gearbox_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.hss_mass = hss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.generator_mass = generator_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.bedplate_mass = bedplate_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.bedplate_length = bedplate_length #Float(iotype = 'in', units='m', desc='component length')
        self.bedplate_width = bedplate_width #Float(iotype = 'in', units='m', desc='component width')
        self.electronics_mass = electronics_mass #Float(iotype = 'in', units='kg', desc='component mass')
    
        # electronic systems, hydraulics and controls
        self.electrical_mass = 0.0
        
        self.converter_mass = 0 #2.4445*rating + 1599.0 accounted for in electronics calcs
        
        self.hvac_mass = 0.08 * rating
        
        self.controls_mass     = 0.0
        
        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        self.platforms_mass = 0.125 * self.bedplate_mass
        
        if (self.crane):
            self.crane_mass =  3000.0
        else:
            self.crane_mass = 0.0
            
        self.mainframe_mass  = self.bedplate_mass + self.crane_mass + self.platforms_mass
        
        nacelleCovArea      = 2 * (self.bedplate_length ** 2)              # this calculation is based on Sunderland
        self.cover_mass = (84.1 * nacelleCovArea) / 2          # this calculation is based on Sunderland - divided by 2 in order to approach CSM
        
        # yaw system weight calculations based on total system mass above yaw system
        m_above_yaw =  (self.lss_mass + 
                                self.mb1_mass + self.mb2_mass + 
                                self.gearbox_mass + 
                                self.hss_mass + 
                                self.generator_mass + 
                                self.mainframe_mass + 
                                self.electronics_mass +
                                self.electrical_mass + 
                                self.converter_mass + 
                                self.hvac_mass +
                                self.cover_mass)

        self.length      = self.bedplate_length                              # nacelle length [m] based on bedplate length
        self.width       = self.bedplate_width                        # nacelle width [m] based on bedplate width
        self.height      = (2.0 / 3.0) * self.length                         # nacelle height [m] calculated based on cladding area

        return(self.electrical_mass, self.converter_mass, self.hvac_mass, self.controls_mass, self.platforms_mass, self.crane_mass, \
               self.mainframe_mass, self.cover_mass, m_above_yaw, self.length, self.width, self.height)
        

#--------------------------------------------
class NacelleSystemAdder_OM(om.ExplicitComponent): #added to drive to include electronics
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def setup(self):

        # variables
        self.add_input('above_yaw_mass', val=0.0, units='kg', desc='mass above yaw system')
        self.add_input('yaw_mass', val=0.0, units='kg', desc='mass of yaw system')
        self.add_input('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('mb1_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('mb2_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('bedplate_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('mainframe_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('lss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('mb1_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('mb2_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('gearbox_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('hss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('generator_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('bedplate_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('lss_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('mb1_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('mb2_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('gearbox_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('hss_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('generator_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('bedplate_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_input('electronics_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('electronics_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('electronics_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        # returns
        self.add_output('nacelle_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('nacelle_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('nacelle_I', val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        
    def compute(self, inputs, outputs):
        nacelleadder = dc.NacelleSystemAdder()

        (outputs['nacelle_mass'], outputs['nacelle_cm'], outputs['nacelle_I']) \
                    = nacelleadder.compute(inputs['above_yaw_mass'], inputs['yaw_mass'], inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'], \
                      inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], inputs['mainframe_mass'], \
                      inputs['lss_cm'], inputs['mb1_cm'], inputs['mb2_cm'], inputs['gearbox_cm'], inputs['hss_cm'], inputs['generator_cm'], inputs['bedplate_cm'], \
                      inputs['lss_I'], inputs['mb1_I'], inputs['mb2_I'], inputs['gearbox_I'], inputs['hss_I'], inputs['generator_I'], inputs['bedplate_I'], \
                      inputs['electronics_mass'], inputs['electronics_cm'], inputs['electronics_I'])

        # variables
        m_above_yaw = above_yaw_mass #Float(iotype='in', units='kg', desc='mass above yaw system')
        self.yaw_mass = yaw_mass #Float(iotype='in', units='kg', desc='mass of yaw system')
        self.lss_mass = lss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb1_mass = mb1_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb2_mass = mb2_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.gearbox_mass = gearbox_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.hss_mass = hss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.generator_mass = generator_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.bedplate_mass = bedplate_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mainframe_mass = mainframe_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.lss_cm = lss_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.mb1_cm = mb1_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.mb2_cm = mb2_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        cm_gearbox = gearbox_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.hss_cm = hss_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.generator_cm = generator_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.bedplate_cm = bedplate_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.lss_I = lss_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.mb1_I = mb1_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.mb2_I = mb2_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.gearbox_I = gearbox_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.hss_I = hss_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.generator_I = generator_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.bedplate_I = bedplate_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.electronics_mass = electronics_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.electronics_cm = electronics_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.electronics_I = electronics_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    
        # returns
        self.nacelle_mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.nacelle_cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), units='m', iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.nacelle_I = np.zeros(3) # Array(np.array([0.0, 0.0, 0.0]), units='kg*m**2', iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        # aggregation of nacelle mass
        self.nacelle_mass = (m_above_yaw + self.yaw_mass)
  
        # calculation of mass center and moments of inertia
        self.nacelle_cm = ( (self.lss_mass*self.lss_cm
                           + self.electronics_mass*self.electronics_cm 
                           + self.mb1_mass*self.mb1_cm 
                           + self.mb2_mass*self.mb2_cm 
                           + self.gearbox_mass*cm_gearbox 
                           + self.hss_mass*self.hss_cm 
                           + self.generator_mass*self.generator_cm 
                           + self.mainframe_mass*self.bedplate_cm 
                           + self.yaw_mass*np.zeros(3))
                      / (self.lss_mass + self.mb1_mass + self.mb2_mass + self.gearbox_mass +
                         self.hss_mass + self.generator_mass + self.mainframe_mass + self.yaw_mass) )
  
        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems

        def appendI(xmass, xcm, xI):
            r    = xcm - self.nacelle_cm
            Icg  = assembleI( np.r_[xI, np.zeros(3)] ) # not used
            Iadd = xmass*(np.dot(r, r)*np.eye(3) - np.outer(r, r))
            return Iadd

        I   = np.zeros((3,3))
        I += appendI(self.lss_mass, self.lss_cm, self.lss_I)
        I += appendI(self.hss_mass, self.hss_cm, self.hss_I)
        I += appendI(self.mb1_mass, self.mb1_cm, self.mb1_I)
        I += appendI(self.mb2_mass, self.mb2_cm, self.mb2_I)
        I += appendI(self.gearbox_mass, cm_gearbox, self.gearbox_I)
        I += appendI(self.electronics_mass, self.electronics_cm, self.electronics_I)
        I += appendI(self.generator_mass, self.generator_cm, self.generator_I)
        # Mainframe mass includes bedplate mass and other components that assume the bedplate cm
        I += appendI(self.mainframe_mass, self.bedplate_cm, (self.mainframe_mass/self.bedplate_mass)*self.bedplate_I)
        self.nacelle_I = unassembleI(I)

        return(self.nacelle_mass, self.nacelle_cm, self.nacelle_I)

