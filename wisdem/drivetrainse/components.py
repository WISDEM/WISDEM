
import numpy as np
import openmdao.api as om


# -------------------------------------------------
# Bearing support functions
# returns facewidth, mass for bearings without fatigue analysis
def resize_for_bearings(D_shaft, btype, deriv):
    # assume low load rating for bearing
    if btype == 'CARB':  # p = Fr, so X=1, Y=0
        out = [D_shaft, .2663 * D_shaft + .0435, 1561.4 * D_shaft**2.6007]
        if deriv == True:
            out.extend([1., .2663, 1561.4 * 2.6007 * D_shaft**1.6007])
    elif btype == 'SRB':
        out = [D_shaft, .2762 * D_shaft, 876.7 * D_shaft**1.7195]
        if deriv == True:
            out.extend([1., .2762, 876.7 * 1.7195 * D_shaft**0.7195])
    elif btype == 'TRB1':
        out = [D_shaft, .0740, 92.863 * D_shaft**.8399]
        if deriv == True:
            out.extend([1., 0., 92.863 * 0.8399 * D_shaft**(0.8399 - 1.)])
    elif btype == 'CRB':
        out = [D_shaft, .1136 * D_shaft, 304.19 * D_shaft**1.8885]
        if deriv == True:
            out.extend([1., .1136, 304.19 * 1.8885 * D_shaft**0.8885])
    elif btype == 'TRB2':
        out = [D_shaft, .1499 * D_shaft, 543.01 * D_shaft**1.9043]
        if deriv == True:
            out.extend([1., .1499, 543.01 * 1.9043 * D_shaft**.9043])
    elif btype == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        out = [D_shaft, .0839, 229.47 * D_shaft**1.8036]
        if deriv == True:
            out.extend([1.0, 0.0, 229.47 * 1.8036 * D_shaft**0.8036])

    # shaft diameter, facewidth, mass. if deriv==True, provides derivatives.
    return out

#-------------------------------------------------------------------------

class MainBearing_OM(om.ExplicitComponent):
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def initialize(self):
        self.options.declare('bearing_position', default='main')
        
    def setup(self):
        # variables
        self.add_input('bearing_mass', val=0.0, units='kg', desc='bearing mass from LSS model')
        self.add_input('lss_diameter', val=0.0, units='m', desc='lss outer diameter at main bearing')
        self.add_input('lss_design_torque', val=0.0, units='N*m', desc='lss design torque')
        self.add_input('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_input('lss_mb_cm', val=np.array([0., 0., 0.]), units='m', desc='x,y,z location from shaft model')

        # returns
        self.add_output('mb_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('mb_cm',   val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('mb_I',    val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        
    def compute(self, inputs, outputs):

        mb = dc.MainBearing(self.options['bearing_position'])
        
        (outputs['mb_mass'], outputs['mb_cm'], outputs['mb_I']) \
            = mb.compute(inputs['bearing_mass'], inputs['lss_diameter'], inputs['lss_design_torque'], inputs['rotor_diameter'], inputs['lss_mb_cm'])

        
        self.bearing_mass = bearing_mass #Float(iotype ='in', units = 'kg', desc = 'bearing mass from LSS model')
        self.lss_diameter = lss_diameter #Float(iotype='in', units='m', desc='lss outer diameter at main bearing')
        self.lss_design_torque = lss_design_torque #Float(iotype='in', units='N*m', desc='lss design torque')
        D_rotor = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.location = location #Array(np.array([0.,0.,0.]),iotype = 'in', units = 'm', desc = 'x,y,z location from shaft model')

        # returns
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I  = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        self.mass = self.bearing_mass
        self.mass += self.mass * (8000.0 / 2700.0)  # add housing weight
            # see Section 2.2.4.2 in report which gives a factor of 2.92 - this is 2.963

        # calculate mass properties
        inDiam = self.lss_diameter
        depth = (inDiam * 1.5) # not used

        try:
            self.bearing_position in ['main','second']
        except ValueError:
            print("Invalid variable assignment: bearing position must be 'main' or 'second'.")
        else:
            if self.bearing_position == 'main':
                if self.location[0] != 0.0:
                    cm = self.location
                else:
                    cmMB = np.zeros(3)
                    cmMB = ([- (0.035 * D_rotor),  0.0, 0.025 * D_rotor])
                    cm = cmMB
                
                b1I0 = (self.mass * inDiam ** 2) / 4.0
                self.cm = cm
                self.I = np.array([b1I0, b1I0 / 2.0, b1I0 / 2.0])
            else:
                if self.mass > 0 and self.location[0] != 0.0:
                    cm = self.location
                else:
                    cm = np.zeros(3)
                    self.mass = 0.
        
                b2I0 = (self.mass * inDiam ** 2) / 4.0
                self.cm = cm
                self.I = np.array([b2I0, b2I0 / 2.0, b2I0 / 2.0])

        return (self.mass, self.cm, self.I.flatten())
        

#-------------------------------------------------------------------

class HighSpeedSide_OM(om.ExplicitComponent):
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
        self.add_input('lss_diameter', val=0.0, units='m', desc='low speed shaft outer diameter')
        self.add_input('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_input('gearbox_height', val=0.0, units='m', desc='gearbox height')
        self.add_input('gearbox_cm', val=np.zeros(3), units='m', desc='gearbox cm [x,y,z]')
        self.add_input('hss_input_length', val=0.0, units='m', desc='high speed shaft length determined by user. Default 0.5m')

        # returns
        self.add_output('hss_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('hss_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('hss_I', val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('hss_length', val=0.0, units='m', desc='length of high speed shaft')

        self.hss = dc.HighSpeedSide()

    def compute(self, inputs, outputs):

        (outputs['hss_mass'], outputs['hss_cm'], outputs['hss_I'], outputs['hss_length']) \
            = self.hss.compute(inputs['rotor_diameter'], inputs['rotor_torque'], inputs['gear_ratio'], inputs['lss_diameter'], inputs['gearbox_length'], inputs['gearbox_height'], inputs['gearbox_cm'], inputs['hss_input_length'])

        # variables
        D_rotor = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.rotor_torque = rotor_torque #Float(iotype='in', units='N*m', desc='rotor torque at rated power')
        self.gear_ratio = gear_ratio #Float(iotype='in', desc='overall gearbox ratio')
        self.lss_diameter = lss_diameter #Float(iotype='in', units='m', desc='low speed shaft outer diameter')
        self.gearbox_length = gearbox_length #Float(iotype = 'in', units = 'm', desc='gearbox length')
        self.gearbox_height = gearbox_height #Float(iotype='in', units = 'm', desc = 'gearbox height')
        self.gearbox_cm = gearbox_cm #Array(iotype = 'in', units = 'm', desc = 'gearbox cm [x,y,z]')
        self.length_in = length_in #Float(iotype = 'in', units = 'm', desc = 'high speed shaft length determined by user. Default 0.5m')
    
        # returns
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.length = 0.0 #Float(iotype='out', desc='length of high speed shaft')

        # compute masses, dimensions and cost
        design_torque = self.rotor_torque / self.gear_ratio               # design torque [Nm] based on rotor torque and Gearbox ratio
        massFact = 0.025                                 # mass matching factor default value
        highSpeedShaftMass = (massFact * design_torque)
  
        mechBrakeMass = (0.5 * highSpeedShaftMass)      # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines
  
        self.mass = (mechBrakeMass + highSpeedShaftMass)
  
        diameter = (1.5 * self.lss_diameter)                     # based on WindPACT relationships for full HSS / mechanical brake assembly
        if self.length_in == 0:
            self.hss_length = 0.5+D_rotor/127.
        else:
            self.hss_length = self.length_in
        hss_length = self.hss_length
  
        matlDensity = 7850. # material density kg/m^3
  
        # calculate mass properties
        cm = np.zeros(3)
        cm[0]   = self.gearbox_cm[0]+self.gearbox_length/2+hss_length/2
        cm[1]   = self.gearbox_cm[1]
        cm[2]   = self.gearbox_cm[2]+self.gearbox_height*0.2
        self.cm = cm
  
        I = np.zeros(3)
        I[0]    = 0.25 * hss_length * 3.14159 * matlDensity * (diameter ** 2) * (self.gear_ratio**2) * (diameter ** 2) / 8.
        I[1]    = self.mass * ((3/4.) * (diameter ** 2) + (hss_length ** 2)) / 12.
        I[2]    = I[1]
        self.I = I

        return(self.mass, self.cm, self.I, self.hss_length)

        
        
        
#-------------------------------------------------------------------------------

class Transformer(om.ExplicitComponent):
    ''' Estimate mass of transformer based on rating, rotor diameter, and tower top diameter.  Empirical only, no load analysis.'''
        
    def setup(self):

        self.add_input('machine_rating', val=0.0, units='kW', desc='machine rating of the turbine')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')
        self.add_input('rotor_diameter', val=0.0, units='m', desc='rotor diameter of turbine')
        #self.add_discrete_input('uptower', True)

        self.add_output('transformer_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('transformer_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('transformer_I', val=np.zeros(3), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        
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
            cm[0] = inputs['x_transformer']
            cm[2] = 0.6*inputs['cm_generator'][2]

        # MoI
        def get_I(d1,d2,mass):
            return (d1**2 + d2**2)/12.
            
        I = np.zeros(3)
        I[0] = mass*get_I(height, width)
        I[1] = mass*get_I(length, height)
        I[2] = mass*get_I(length, width)

        # Outputs
        outputs['transformer_mass'] = mass
        outputs['transformer_cm'] = cm
        outputs['transformer_I'] = I
        


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
        self.add_input('transformer_mass', val=0.0, units='kg', desc='component mass')

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
                    inputs['bedplate_length'], inputs['bedplate_width'], inputs['transformer_mass'])
        
        if self.options['debug']:
            print('AYMA IN: {:.1f} kW BPl {:.1f} m BPw {:.1f} m'.format(
                  inputs['machine_rating'],inputs['bedplate_length'], inputs['bedplate_width']))
            print('AYMA IN  masses (kg): LSS {:.1f} MB1 {:.1f} MB2 {:.1f} GBOX {:.1f} HSS {:.1f} GEN {:.1f} BP {:.1f} TFRM {:.1f}'.format(
                  inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'],
                  inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], inputs['transformer_mass']))
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
        self.transformer_mass = transformer_mass #Float(iotype = 'in', units='kg', desc='component mass')
    
        # electronic systems, hydraulics and controls
        self.electrical_mass = 0.0
        
        self.converter_mass = 0 #2.4445*rating + 1599.0 accounted for in transformer calcs
        
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
                                self.transformer_mass +
                                self.electrical_mass + 
                                self.converter_mass + 
                                self.hvac_mass +
                                self.cover_mass)

        self.length      = self.bedplate_length                              # nacelle length [m] based on bedplate length
        self.width       = self.bedplate_width                        # nacelle width [m] based on bedplate width
        self.height      = (2.0 / 3.0) * self.length                         # nacelle height [m] calculated based on cladding area

        return(self.electrical_mass, self.converter_mass, self.hvac_mass, self.controls_mass, self.platforms_mass, self.crane_mass, \
               self.mainframe_mass, self.cover_mass, m_above_yaw, self.length, self.width, self.height)
        

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
        

#--------------------------------------------
class NacelleSystemAdder_OM(om.ExplicitComponent): #added to drive to include transformer
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
        self.add_input('transformer_mass', val=0.0, units='kg', desc='component mass')
        self.add_input('transformer_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_input('transformer_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

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
                      inputs['transformer_mass'], inputs['transformer_cm'], inputs['transformer_I'])

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
        self.gearbox_cm = gearbox_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
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
        self.transformer_mass = transformer_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.transformer_cm = transformer_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.transformer_I = transformer_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    
        # returns
        self.nacelle_mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.nacelle_cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), units='m', iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.nacelle_I = np.zeros(3) # Array(np.array([0.0, 0.0, 0.0]), units='kg*m**2', iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        # aggregation of nacelle mass
        self.nacelle_mass = (m_above_yaw + self.yaw_mass)
  
        # calculation of mass center and moments of inertia
        self.nacelle_cm = ( (self.lss_mass*self.lss_cm
                           + self.transformer_mass*self.transformer_cm 
                           + self.mb1_mass*self.mb1_cm 
                           + self.mb2_mass*self.mb2_cm 
                           + self.gearbox_mass*self.gearbox_cm 
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
        I += appendI(self.gearbox_mass, self.gearbox_cm, self.gearbox_I)
        I += appendI(self.transformer_mass, self.transformer_cm, self.transformer_I)
        I += appendI(self.generator_mass, self.generator_cm, self.generator_I)
        # Mainframe mass includes bedplate mass and other components that assume the bedplate cm
        I += appendI(self.mainframe_mass, self.bedplate_cm, (self.mainframe_mass/self.bedplate_mass)*self.bedplate_I)
        self.nacelle_I = unassembleI(I)

        return(self.nacelle_mass, self.nacelle_cm, self.nacelle_I)

