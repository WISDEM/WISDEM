"""
nacelle.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from math import *
from common import SubComponent
import numpy as np
from zope.interface import implements

# -------------------------------------------------

class LowSpeedShaft():
    implements(SubComponent)
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''

    def __init__(self, RotorDiam, RotorMass, RotorTorque):
        ''' 
        Initializes low speed shaft component 
        
        Parameters
        ----------
        RotorDiam : float
          The wind turbine rotor diameter [m]
        RotorMass : float
          The wind turbine rotor mass [kg]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        designTQ : float
          Design torque for the low speed shaft based on input torque from the rotor at rated speed accounting for drivetrain losses - multiplied by a safety factor
        designBL : float
          Design bending load based on low speed shaft based on rotor mass
        '''

        self.designTQ = 0.00
        self.designBL = 0.00

        self.update_mass(RotorDiam, RotorMass, RotorTorque)
        
    def update_mass(self,RotorDiam, RotorMass, RotorTorque):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine low speed shaft component.
        
        The compute method determines and sets the attributes for the low speed shaft component based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.        
        
        Parameters
        ----------
        RotorDiam : float
          The wind turbine rotor diameter [m]
        RotorMass : float
          The wind turbine rotor mass [kg]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
          
        '''
        # compute masses, dimensions and cost
        ioratio   = 0.100                                    # constant value for inner/outer diameter ratio (should depend on LSS type)                                                  
        hollow    = 1/(1-(ioratio)**4)                    # hollowness factor based on diameter ratio

        TQsafety       = 3.0                                    # safety factor for design torque applied to rotor torque
        self.designTQ =(TQsafety * RotorTorque)            # LSS design torque [Nm]

        lenFact        = 0.03                                   # constant value for length as a function of rotor diameter (should depend on LSS type) 
        self.length = (lenFact * RotorDiam )              # LSS shaft length [m]                                                                  
        maFact         = 5                                 # moment arm factor from shaft lenght (should depend on shaft type)
        self.mmtArm    = self.length / maFact            # LSS moment arm [m] - from hub to first main bearing
        BLsafety       = 1.25                                   # saftey factor on bending load
        g              = 9.81                              # gravitational constant [m / s^2]
        self.designBL  = BLsafety * g * RotorMass          # LSS design bending load [N]                                                  
        self.bendMom   = self.designBL * self.mmtArm       # LSS design bending moment [Nm]

        yieldst        = 371000000.0                             # BS1503-622 yield stress [Pa] (should be adjusted depending on material type)
        endurstsp      = 309000000.0                             # BS1503-625 specimen endurance limit [Pa] (should be adjusted depending on material type)
        endurFact      = 0.23                                    # factor for specimen to component endurance limit 
                                                                # (0.75 surface condition * 0.65 size * 0.52 reliability * 1 temperature * 0.91 stress concentration)
        endurst        = endurstsp * endurFact                   # endurance limit [Pa] for LSS
        SOsafety       = 3.25                               # Soderberg Line approach factor for safety 
        self.diameter = (((32/pi)*hollow*SOsafety*((self.designTQ / yieldst)**2+(self.bendMom/endurst)**2)**(0.5))**(1./3.))                               
                                                            # outer diameter [m] computed by Westinghouse Code Formula based on Soderberg Line approach to fatigue design
        inDiam    = self.diameter * ioratio            # inner diameter [m]

        
        massFact       = 1.25                                    # mass weight factor (should depend on LSS/drivetrain type, currently from windpact modifications to account for flange weight)                                                                       
        steeldens      = 7860                                    # steel density [kg / m^3]

        self.mass = (massFact*(pi/4)*(self.diameter**2-inDiam**2)*self.length*steeldens)      # mass of LSS [kg]  

        # calculate mass properties        
        cm = np.array([0.0,0.0,0.0])
        cm[0] = - (0.035 - 0.01) * RotorDiam            # cm based on WindPACT work - halfway between locations of two main bearings
        cm[1] = 0.0
        cm[2] = 0.025 * RotorDiam                      
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        I[0]  = self.mass * (inDiam ** 2 + self.diameter ** 2) / 8
        I[1]  = self.mass * (inDiam ** 2 + self.diameter ** 2 + (4 / 3) * (self.length ** 2)) / 16
        I[2]  = I[1]
        self.I = I

#-------------------------------------------------------------------------------

class Bearing():
	  implements(SubComponent)
	  '''
	     Bearing class for a generic bearing
	  '''

class MainBearings():
    implements(SubComponent) 
    ''' MainBearings class          
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.           
    '''
    
    def __init__(self,lss, RotorSpeed, RotorDiam):
        ''' Initializes main bearings component 

        Parameters
        ----------
        lss : LowSpeedShaft object
          The low speed shaft object of a wind turbine drivetrain
        RotorSpeed : float
          Speed of the rotor at rated power [rpm]
        RotorDiam : float
          The wind turbine rotor diameter [m]
        inDiam : float
          inner diameter of bearings - equivalent to low speed shaft outer diameter
        '''

        self.mainBearing = Bearing()
        self.secondBearing = Bearing()
        self.inDiam     = 0.0

        self.update_mass(lss, RotorSpeed, RotorDiam)       
        
    def update_mass(self,lss, RotorSpeed, RotorDiam):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine main bearing components.
        
        The compute method determines and sets the attributes for the main bearing components based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
        
        Parameters
        ----------
        lss : LowSpeedShaft object
          The low speed shaft object of a wind turbine drivetrain
        RotorSpeed : float
          Speed of the rotor at rated power [rpm]
        RotorDiam : float
          The wind turbine rotor diameter [m]          
        '''
      
        # compute masses, dimensions and cost
        g = 9.81                                           # gravitational constant [m / s^2]
        design1SL = (4.0 / 3.0) * lss.designTQ + lss.mass * (g / 2.0)
                                                           # front bearing static design load [N] based on default equation (should depend on LSS type)
        design2SL = (1.0 / 3.0) * lss.designTQ - lss.mass * (g / 2.0)
                                                           # rear bearing static design load [N] based on default equation (should depend on LSS type)
        design1DL = 2.29 * design1SL * (RotorSpeed ** 0.3)
                                                           # front bearing dynamic design load [N]
        design2DL = 2.29 * design2SL * (RotorSpeed ** 0.3)
                                                           # rear bearing dynamic design load [N]

        ratingDL  = 17.96 * ((lss.diameter * 1000.0) ** 1.9752)  # basic dynamic load rating for a bearing given inside diameter based on catalogue regression

        massFact  = 0.25                                 # bearing weight factor (should depend on drivetrain type) - using to adjust data closer to cost and scaling model estimates

        if (design1DL < ratingDL):
            b1mass = massFact * (26.13 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.77)
                                                           # bearing mass [kg] for single row bearing (based on catalogue data regression)
            h1mass = massFact * (67.44 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.64)
                                                           # bearing housing mass [kg] for single row bearing (based on catalogue data regression) 
        else:
            b1mass = massFact * 1.7 * (26.13 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.77)
                                                          # bearing mass [kg] for double row bearing (based on catalogue data regression) 
            h1mass = massFact * 1.5 * (67.44 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.64)
                                                          # bearing housing mass [kg] for double row bearing (based on catalogue data regression) 

        if (design2DL < ratingDL):
            b2mass = massFact * (26.13 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.77)
                                                           # bearing mass [kg] for single row bearing (based on catalogue data regression)
            h2mass = massFact * (67.44 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.64)
                                                           # bearing housing mass [kg] for single row bearing (based on catalogue data regression) 
        else:
            b2mass = massFact * 1.7 * (26.13 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.77)
                                                          # bearing mass [kg] for double row bearing (based on catalogue data regression) 
            h2mass = massFact * 1.5 * (67.44 * (10 ** (-6))) * ((lss.diameter * 1000.0) ** 2.64)
                                                          # bearing housing mass [kg] for double row bearing (based on catalogue data regression) 

        self.mainBearing.mass = b1mass + h1mass
        self.secondBearing.mass = b2mass + h2mass

        self.mass = (self.mainBearing.mass + self.secondBearing.mass)

        # calculate mass properties
        inDiam  = lss.diameter
        self.depth = (inDiam * 1.5)

        cmMB = np.array([0.0,0.0,0.0])
        cmMB = ([- (0.035 * RotorDiam), 0.0, 0.025 * RotorDiam])
        self.mainBearing.cm = cmMB

        cmSB = np.array([0.0,0.0,0.0])
        cmSB = ([- (0.01 * RotorDiam), 0.0, 0.025 * RotorDiam])
        self.secondBearing.cm = cmSB

        cm = np.array([0.0,0.0,0.0])
        for i in (range(0,3)):
            # calculate center of mass
            cm[i] = (self.mainBearing.mass * self.mainBearing.cm[i] + self.secondBearing.mass * self.secondBearing.cm[i]) \
                      / (self.mainBearing.mass + self.secondBearing.mass)
        self.cm = cm
       
        self.b1I0 = (b1mass * inDiam ** 2 ) / 4 + (h1mass * self.depth ** 2) / 4
        self.mainBearing.I = ([self.b1I0, self.b1I0 / 2, self.b1I0 / 2]) 

        self.b2I0  = (b2mass * inDiam ** 2 ) / 4 + (h2mass * self.depth ** 2) / 4
        self.secondBearing.I = ([self.b2I0, self.b2I0 / 2, self.b2I0 / 2])

        I = np.array([0.0, 0.0, 0.0])
        for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
            # calculate moments around CM
            # sum moments around each components CM
            I[i]  =  self.mainBearing.I[i] + self.secondBearing.I[i]
            # translate to nacelle CM using parallel axis theorem
            for j in (range(0,3)): 
                if i != j:
                    I[i] +=  (self.mainBearing.mass * (self.mainBearing.cm[i] - self.cm[i]) ** 2) + \
                                  (self.secondBearing.mass * (self.secondBearing.cm[i] - self.cm[i]) ** 2)
        self.I = I

#-------------------------------------------------------------------------------

class Gearbox():
    implements(SubComponent)  
    ''' Gearbox class          
          The Gearbox class is used to represent the gearbox component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.           
    '''
    
    def __init__(self,iDsgn,RotorTorque,GearRatio,GearConfig,Bevel,RotorDiam):
        '''
        Initializes gearbox component 
          
        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains though the configuration can be single, double or triple staged with any combination of epicyclic and parallel stages.
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        GearConfig : str
          String that represents the configuration of the gearbox (stage number and types).
          Possible configurations include 'e', 'p', 'pp', 'ep', 'ee', 'epp', 'eep', 'eee'.
        Bevel : int
          Flag for the presence of a bevel stage - 1 if present, 0 if not; typically it is not present.
        RotorDiam : float
          The wind turbine rotor diameter [m]
        stagemass : array
          Array of stage masses for the gearbox [kg]
        '''

        self.stagemass = [None, 0.0, 0.0, 0.0, 0.0]

        self.update_mass(iDsgn,RotorTorque,GearRatio,GearConfig,Bevel,RotorDiam)         
        
    def update_mass(self,iDsgn,RotorTorque,GearRatio,GearConfig,Bevel,RotorDiam):

        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine gearbox component.
        
        The compute method determines and sets the attributes for the gearbox component based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
        Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.
        
        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains though the configuration can be single, double or triple staged with any combination of epicyclic and parallel stages.
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        GearConfig : str
          String that represents the configuration of the gearbox (stage number and types).
          Possible configurations include 'e', 'p', 'pp', 'ep', 'ee', 'epp', 'eep', 'eee'.
        Bevel : int
          Flag for the presence of a bevel stage - 1 if present, 0 if not; typically it is not present.
        RotorDiam : float
          The wind turbine rotor diameter [m]
          
        '''    

        # compute masses, dimensions and cost
        overallweightFact = 1.00                          # default weight factor 1.0 (should depend on drivetrain design)
        self.stagemass = [None, 0.0, 0.0, 0.0, 0.0]       # TODO: problem initializing stagemass and accessing in compute
 
        # find weight of each stage depending on configuration
        # Gear ratio reduced for each stage based on principle that mixed epicyclic/parallel trains have a final stage ratio set at 1:2.5
        if GearConfig == 'p':
            self.stagemass[1] = self.__getParallelStageWeight(RotorTorque,GearRatio)
        if GearConfig == 'e':
            self.stagemass[1] = self.__getEpicyclicStageWeight(RotorTorque,GearRatio)
        if GearConfig == 'pp':
            self.stagemass[1] = self.__getParallelStageWeight(RotorTorque,GearRatio**0.5)
            self.stagemass[2] = self.__getParallelStageWeight(RotorTorque,GearRatio**0.5)
        if GearConfig == 'ep':
            self.stagemass[1] = self.__getEpicyclicStageWeight(RotorTorque,GearRatio/2.5)
            self.stagemass[2] = self.__getParallelStageWeight(RotorTorque,2.5)
        if GearConfig == 'ee':
            self.stagemass[1] = self.__getEpicyclicStageWeight(RotorTorque,GearRatio**0.5)
            self.stagemass[2] = self.__getEpicyclicStageWeight(RotorTorque,GearRatio**0.5)
        if GearConfig == 'eep':
            U1 = (GearRatio/3.0)**0.5
            U2 = (GearRatio/3.0)**0.5
            U3 = 3.0
            self.stagemass[1] = self.__getEpicyclicStageWeight(RotorTorque,1,U1,U2,U3)  #different than sunderland
            self.stagemass[2] = self.__getEpicyclicStageWeight(RotorTorque,2,U1,U2,U3)
            self.stagemass[3] = self.__getParallelStageWeight(RotorTorque,3,U1,U2,U3)
        if GearConfig == 'epp':
            U1 = GearRatio**0.33*1.4
            U2 = (GearRatio**0.33/1.18)
            U3 = (GearRatio**0.33/1.18)
            self.stagemass[1] = self.__getEpicyclicStageWeight(RotorTorque,1,U1,U2,U3)    # not in sunderland
            self.stagemass[2] = self.__getParallelStageWeight(RotorTorque,2,U1,U2,U3)
            self.stagemass[3] = self.__getParallelStageWeight(RotorTorque,3,U1,U2,U3)
        if GearConfig == 'eee':
            self.stagemass[1] = self.__getEpicyclicStageWeight(RotorTorque,GearRatio**(0.33))
            self.stagemass[2] = self.__getEpicyclicStageWeight(RotorTorque,GearRatio**(0.33))
            self.stagemass[3] = self.__getEpicyclicStageWeight(RotorTorque,GearRatio**(0.33))
        if GearConfig == 'ppp':
            self.stagemass[1] = self.__getParallelStageWeight(RotorTorque,GearRatio**(0.33))
            self.stagemass[2] = self.__getParallelStageWeight(RotorTorque,GearRatio**(0.33))
            self.stagemass[3] = self.__getParallelStageWeight(RotorTorque,GearRatio**(0.33))


        if (Bevel):
            self.stagemass[4] = 0.0454 * (RotorTorque ** 0.85)

        mass = 0.0
        for i in range(1,4):
            mass += self.stagemass[i]
        mass     *= overallweightFact  
        self.mass = (mass)

        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0]   = cm[1] = 0.0
        cm[2]   = 0.025 * RotorDiam
        self.cm = cm

        self.length = (0.012 * RotorDiam)
        self.height = (0.015 * RotorDiam)
        self.diameter = (0.75 * self.height)

        I = np.array([0.0, 0.0, 0.0])
        I[0] = self.mass * (self.diameter ** 2 ) / 8 + (self.mass / 2) * (self.height ** 2) / 8
        I[1] = self.mass * (0.5 * (self.diameter ** 2) + (2 / 3) * (self.length ** 2) + 0.25 * (self.height ** 2)) / 8
        I[2] = I[1]       
        self.I = I

    def __getParallelStageWeight(self,RotorTorque,stage,StageRatio1,StageRatio2,StageRatio3):

        ''' 
          This method calculates the stage weight for a parallel stage in a gearbox based on the input torque, stage number, and stage ratio for each individual stage.
        '''
        
        serviceFact     = 1.00                                # default service factor for a gear stage is 1.75 based on full span VP (should depend on control type)
        applicationFact = 0.4                             # application factor ???        
        stageweightFact = 8.029 #/2                           # stage weight factor applied to each Gearbox stage

        if (RotorTorque * serviceFact) < 200000.0:       # design factor for design and manufacture of Gearbox
            designFact = 925.0
        elif (RotorTorque * serviceFact) < 700000.0:
            designFact = 1000.0
        else:
            designFact = 1100.0                            # TODO: should be an exception for all 2 stage Gearboxes to have designFact = 1000
            
        if stage == 1:
            Qr         = RotorTorque
            StageRatio = StageRatio1
        elif stage == 2:
            Qr         = RotorTorque/StageRatio1
            StageRatio = StageRatio2
        elif stage == 3:
            Qr         = RotorTorque/(StageRatio1*StageRatio2)
            StageRatio = StageRatio3

        gearFact = applicationFact / designFact          # Gearbox factor for design, manufacture and application of Gearbox
        
        gearweightFact = 1 + (1 / StageRatio) + StageRatio + (StageRatio ** 2)
                                                         # Gearbox weight factor for relationship of stage ratio required and relative stage volume

        stageWeight = stageweightFact * Qr * serviceFact * gearFact * gearweightFact
                                                         # forumula for parallel gearstage weight based on sunderland model

        return stageWeight

    def __getEpicyclicStageWeight(self,RotorTorque,stage,StageRatio1,StageRatio2,StageRatio3):
        ''' 
          This method calculates the stage weight for a epicyclic stage in a gearbox based on the input torque, stage number, and stage ratio for each individual stage
        '''

        serviceFact     = 1.00                                # default service factor for a gear stage is 1.75 based on full span VP (should depend on control type)
        applicationFact = 0.4                             # application factor ???        
        stageweightFact = 8.029/12                          # stage weight factor applied to each Gearbox stage
        OptWheels       = 3.0                                    # default optional wheels (should depend on stage design)

        if (RotorTorque * serviceFact) < 200000.0:       # design factor for design and manufacture of Gearbox
            designFact = 850.0
        elif (RotorTorque * serviceFact) < 700000.0:
            designFact = 950.0
        else:
            designFact = 1100.0
           
        if stage == 1:
            Qr         = RotorTorque
            StageRatio = StageRatio1
        elif stage == 2:
            Qr         = RotorTorque/StageRatio1
            StageRatio = StageRatio2
        elif stage == 3:
            Qr         = RotorTorque/(StageRatio1*StageRatio2)
            StageRatio = StageRatio3

        gearFact = applicationFact / designFact          # Gearbox factor for design, manufacture and application of Gearbox
       
        sunwheelratio  = (StageRatio / 2.0) - 1             # sun wheel ratio for epicyclic Gearbox stage based on stage ratio
        gearweightFact = (1 / OptWheels) + (1 / (OptWheels * sunwheelratio)) + sunwheelratio + \
                         ((1 + sunwheelratio) / OptWheels) * ((StageRatio - 1.) ** 2)
                                                         # Gearbox weight factor for relationship of stage ratio required and relative stage volume

        stageWeight    = stageweightFact * Qr * serviceFact * gearFact * gearweightFact
                                                         # forumula for epicyclic gearstage weight based on sunderland model

        return stageWeight

    def getStageMass(self):
        '''
        This method returns an array of the stage masses for individual stages in a gearbox
       
        Returns
        -------
        self.stagemass : array
           Array of individual stage masses for a gearbox
        '''

        return self.stagemass        

#-------------------------------------------------------------------------------

class HighSpeedShaft():
	  implements(SubComponent)
	  ''' 
	     Basic class for a high speed shaft.
	  '''

class MechanicalBrake():
	  implements(SubComponent)
	  '''
	     Basic class for mechanical brake.
	  '''
              
class HighSpeedSide():
    implements(SubComponent)
    ''' 
    HighSpeedShaft class          
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''

    def __init__(self,MachineRating,RotorTorque,GearRatio,RotorDiam,lssOutDiam):
        ''' Initializes high speed shaft and mechanical brake component 
          
        Parameters
        ----------
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        lssOutDiam : float
          outer diameter of low speed shaft [m]        
        '''

        self.HSS = HighSpeedShaft()
        self.MechBrake = MechanicalBrake()
     
        self.update_mass(MachineRating,RotorTorque,GearRatio,RotorDiam,lssOutDiam)        
        
    def update_mass(self,MachineRating,RotorTorque,GearRatio,RotorDiam,lssOutDiam): 
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine high speed shaft and brake.
        
        The compute method determines and sets the attributes for the HSS and mehanical brakes based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
        
          
        Parameters
        ----------
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        lssOutDiam : float
          outer diameter of low speed shaft [m]
        '''
        
        # compute masses, dimensions and cost        
        designTQ = RotorTorque / GearRatio               # design torque [Nm] based on rotor torque and Gearbox ratio
        massFact = 0.025                                 # mass matching factor default value
        self.HSS.mass = (massFact * designTQ)
        
        self.MechBrake.mass = (0.5 * self.HSS.mass)      # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines

        self.mass = (self.MechBrake.mass + self.HSS.mass)                      

        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0]   = 0.5 * (0.0125 * RotorDiam)
        cm[1]   = 0.0
        cm[2]   = 0.025 * RotorDiam
        self.cm = cm

        self.diameter = (1.5 * lssOutDiam)                     # based on WindPACT relationships for full HSS / mechanical brake assembly
        self.length = (0.025)

        I = np.array([0.0, 0.0, 0.0])
        I[0]    = 0.25 * self.length * 3.14159 * (self.diameter ** 2) * GearRatio * (self.diameter ** 2) / 8
        I[1]    = self.mass * ((3/4) * (self.diameter ** 2) + (self.length ** 2)) / 12
        I[2]    = I[1]      
        self.I = I

#-------------------------------------------------------------------------------

class Generator():
    implements(SubComponent)
    '''Generator class          
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.           
    '''
    
    def __init__(self,iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio):
        ''' Initializes generator component 
        
        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains.
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        '''    

        self.update_mass(iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio)         
        
    def update_mass(self,iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine generator component.
        
        The compute method determines and sets the attributes for the generator component based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
        Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.
      
        
        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains.
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        '''

        massCoeff = [None, 6.4737, 10.51 ,  5.34  , 37.68  ]           
        massExp   = [None, 0.9223, 0.9223,  0.9223, 1      ]

        CalcRPM    = 80 / (RotorDiam*0.5*pi/30)
        CalcTorque = (MachineRating*1.1) / (CalcRPM * pi/30)
        
        if (iDsgn < 4):
            self.mass = (massCoeff[iDsgn] * MachineRating ** massExp[iDsgn])   
        else:  # direct drive
            self.mass = (massCoeff[iDsgn] * CalcTorque ** massExp[iDsgn])  

        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0]  = 0.0125 * RotorDiam
        cm[1]  = 0.0
        cm[2]  = 0.025 * RotorDiam
        self.cm = cm

        self.length = (1.6 * 0.015 * RotorDiam)
        self.depth = (0.015 * RotorDiam)
        self.width = (0.5 * self.depth)

        I = np.array([0.0, 0.0, 0.0])
        I[0]   = ((4.86 * (10 ** (-5))) * (RotorDiam ** 5.333)) + (((2/3) * self.mass) * (self.depth ** 2 + self.width ** 2) / 8)
        I[1]   = (I[0] / 2) / (GearRatio ** 2) + ((1/3) * self.mass * (self.length ** 2) / 12) + (((2 / 3) * self.mass) * \
                   (self.depth ** 2 + self.width ** 2 + (4/3) * (self.length ** 2)) / 16 )
        I[2]   = I[1]                           
        self.I = I

#-------------------------------------------------------------------------------

class Bedplate():
    implements(SubComponent)
    ''' Bedplate class         
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.           
    '''
    
    def __init__(self,iDsgn,RotorTorque,RotorMass,RotorThrust,RotorDiam,TowerTopDiam):
        ''' Initializes bedplate component 
        
        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains.
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        RotorMass : float
          Mass of the rotor [kg]
        RotorThrust : float
          Maximum thrust from the rotor applied to the drivetrain under extreme conditions [N]
        RotorDiam : float
          The wind turbine rotor diameter [m]
        TowerTopDiam : float
          Diameter of the turbine tower top [m]        
        '''

        self.update_mass(iDsgn,RotorTorque,RotorMass,RotorThrust,RotorDiam,TowerTopDiam)       
        
    def update_mass(self,iDsgn,RotorTorque,RotorMass,RotorThrust,RotorDiam,TowerTopDiam):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine bedplate component.
        
        The compute method determines and sets the attributes for the bedplate component based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
      
        
        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains.
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        RotorMass : float
          Mass of the rotor [kg]
        RotorThrust : float
          Maximum thrust from the rotor applied to the drivetrain under extreme conditions [N]
        RotorDiam : float
          The wind turbine rotor diameter [m]
        TowerTopDiam : float
          Diameter of the turbine tower top [m]
          
        '''

        # compute masses, dimensions and cost
        # bedplate sizing based on superposition of loads for rotor torque, thurst, weight         #TODO: only handles bedplate for a traditional drivetrain configuration
        bedplateWeightFact = 2.86                                   # toruqe weight factor for bedplate (should depend on drivetrain, bedplate type)

        torqueweightCoeff = 0.00368                   # regression coefficient multiplier for bedplate weight based on rotor torque
        MassFromTorque    = bedplateWeightFact * (torqueweightCoeff * RotorTorque)
                                                                  
        thrustweightCoeff = 0.00158                                 # regression coefficient multiplier for bedplate weight based on rotor thrust
        MassFromThrust    = bedplateWeightFact * (thrustweightCoeff * (RotorThrust * TowerTopDiam))

        rotorweightCoeff    = 0.015                                    # regression coefficient multiplier for bedplate weight based on rotor weight
        MassFromRotorWeight = bedplateWeightFact * (rotorweightCoeff * (RotorMass * TowerTopDiam))
        
        # additional weight ascribed to bedplate area
        BPlengthFact    = 1.5874                                       # bedplate length factor (should depend on drivetrain, bedplate type)
        nacellevolFact  = 0.052                                      # nacelle volume factor (should depend on drivetrain, bedplate type)
        self.length = (BPlengthFact * nacellevolFact * RotorDiam)     # bedplate length [m] calculated as a function of rotor diameter
        self.width = (self.length / 2.0)                              # bedplate width [m] assumed to be half of bedplate length
        self.area       = self.length * self.width                        # bedplate area [m^2]
        self.height = ((2 / 3) * self.length)                         # bedplate height [m] calculated based on cladding area
        areaweightCoeff = 100                                       # regression coefficient multiplier for bedplate weight based on bedplate area
        MassFromArea    = bedplateWeightFact * (areaweightCoeff * self.area)
    
        # total mass is calculated based on adding masses attributed to rotor torque, thrust, weight and bedplate area
        TotalMass = MassFromTorque + MassFromThrust + MassFromRotorWeight + MassFromArea

        # for single-stage and multi-generator - bedplate mass based on regresstion to rotor diameter
        # for geared and direct drive - bedplate mass based on total mass as calculated above
        massCoeff    = [None,22448,1.29490,1.72080,22448 ]
        massExp      = [None,    0,1.9525, 1.9525 ,    0 ]
        massCoeff[1] = TotalMass  
        ddweightfact = 0.55                                         # direct drive bedplate weight assumed to be 55% of modular geared type
        massCoeff[4] = TotalMass * ddweightfact

        self.mass = (massCoeff[iDsgn] * RotorDiam ** massExp[iDsgn] )
        
        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0] = cm[1] = 0.0
        cm[2] = 0.0122 * RotorDiam                             # half distance from shaft to yaw axis
        self.cm = cm

        self.depth = (self.length / 2.0)

        I = np.array([0.0, 0.0, 0.0])
        I[0]  = self.mass * (self.width ** 2 + self.depth ** 2) / 8
        I[1]  = self.mass * (self.depth ** 2 + self.width ** 2 + (4/3) * self.length ** 2) / 16
        I[2]  = I[1]                          
        self.I = I

#-------------------------------------------------------------------------------
   
class YawSystem():
    implements(SubComponent)
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
 
    def __init__(self,RotorDiam,RotorThrust,TowerTopDiam,AboveYawMass):
        ''' Initializes yaw system 
        
        Parameters
        ----------
        RotorDiam : float
          The wind turbine rotor diameter [m]
        RotorThrust : float
          Maximum thrust from the rotor applied to the drivetrain under extreme conditions [N]
        TowerTopDiam : float
          Diameter of the turbine tower top [m]
        AboveYawMass : float
          Mass of the system above the yaw bearing [kg]
        '''

        self.update_mass(RotorDiam,RotorThrust,TowerTopDiam,AboveYawMass)         
        
    def update_mass(self,RotorDiam,RotorThrust,TowerTopDiam,AboveYawMass):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine yaw system.
        
        The compute method determines and sets the attributes for the yaw system component based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies [2].
     
        
        Parameters
        ----------
        RotorDiam : float
          The wind turbine rotor diameter [m]
        RotorThrust : float
          Maximum thrust from the rotor applied to the drivetrain under extreme conditions [N]
        TowerTopDiam : float
          Diameter of the turbine tower top [m]
        AboveYawMass : float
          Mass of the system above the yaw bearing [kg]  
        '''
        
        # yaw weight depends on moment due to weight of components above yaw bearing and moment due to max thrust load
        #AboveYawMass = 350000 # verboseging number based on 5 MW RNA mass
        yawfactor = 0.41 * (2.4 * (10 ** (-3)))                   # should depend on rotor configuration: blade number and hub type
        weightMom = AboveYawMass * RotorDiam                    # moment due to weight above yaw system
        thrustMom = RotorThrust * TowerTopDiam                  # moment due to rotor thrust
        self.mass = (yawfactor * (0.4 * weightMom + 0.975 * thrustMom))
        
        # calculate mass properties
        # yaw system assumed to be collocated to tower top center              
        cm = np.array([0.0,0.0,0.0])
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        self.I = I

#-------------------------------------------------------------------------------

class NacelleSystem(): # changed name to nacelle - need to rename, move code pieces, develop configurations ***
    implements(SubComponent)
    ''' NacelleSystem class       
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self,RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane):
        ''' 
        Initializes nacelle system 
        
        Parameters
        ----------
        lss  : LowSpeedShaft()
        mbg  : MainBearings()
        gear : Gearbox()
        hss  : HighSpeedShaft()
        gen  : Generator()
        bpl  : Bedplate()
        yaw  : YawSystem()            
        crane   : bool
          flag for presence of service crane up-tower
        econnectionsCost : float     
          cost for electrical cabling
        econnectionsMass : float    
          mass uptower for electrical cabling [kg]
        vspdEtronicsCost : float    
          cost for variable speed electronics
        vspdEtronicsMass : float    
          mass for variable speed electronics [kg]
        hydrCoolingCost : float     
          cost for hydraulics and HVAC system
        hydrCoolingMass : float     
          mass for hydraulics and HVAC system [kg]
        ControlsCost : float  
          cost for controls up      
        ControlsMass : float  
          mass uptower for controls [kg]      
        nacellePlatformsCost : float
          cost for nacelle platforms
        nacellePlatformsMass : float
          mass for nacelle platforms [kg]
        craneCost : float     
          cost for service crane uptower       
        craneMass : float           
          mass for service crane uptower [kg]
        mainframeCost : float       
          cost for mainframe including bedplate, service crane and platforms
        mainframeMass : float       
          mass for mainframe including bedplate, service crane and platforms [kg]
        nacelleCovCost : float      
          cost for nacelle cover
        nacelleCovMass : float  
          mass for nacelle cover [kg] 
        '''

        # create instances of input variables and initialize values
        self.lss = LowSpeedShaft(RotorDiam,RotorMass,RotorTorque)
        self.mbg = MainBearings(self.lss, RotorSpeed, RotorDiam)
        self.gear = Gearbox(iDsgn,RotorTorque,GearRatio,GearConfig,Bevel,RotorDiam)
        self.stagemass = self.gear.getStageMass() #return gearbox stage masses
        self.hss = HighSpeedSide(MachineRating,RotorTorque,GearRatio,RotorDiam,self.lss.diameter)
        self.gen = Generator(iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio)
        self.bpl = Bedplate(iDsgn,RotorTorque,RotorMass,RotorThrust,RotorDiam,TowerTopDiam) 
        self.aboveYawMass = 0.0
        self.yaw = YawSystem(RotorDiam,RotorThrust,TowerTopDiam,self.aboveYawMass)

        # initialize default status for onboard crane and onshore/offshroe
        self.crane   = True
        self.onshore = True

        # initialize other drivetrain components
        self.econnectionsCost     = 0.0
        self.econnectionsMass     = 0.0
        self.vspdEtronicsCost     = 0.0
        self.vspdEtronicsMass     = 0.0
        self.hydrCoolingCost      = 0.0
        self.hydrCoolingMass      = 0.0
        self.ControlsCost         = 0.0
        self.ControlsMass         = 0.0
        self.nacellePlatformsCost = 0.0
        self.nacellePlatformsMass = 0.0
        self.craneCost            = 0.0
        self.craneMass            = 0.0
        self.mainframeCost        = 0.0
        self.mainframeMass        = 0.0
        self.nacelleCovCost       = 0.0
        self.nacelleCovMass       = 0.0

        self.update_mass(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    def update_mass(self,RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine nacelle and all of its sub components.
        
        The compute method determines and sets the attributes for the nacelle and all of its sub components based on the inputs described in the parameter section below. 
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
        Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.
      
        
        Parameters
        ----------
        RotorMass : float
            mass of rotor [kg]
        RotorDiam : float
            diameter of rotor [m]
        RotorTorque : float
            input torque from rotor at rated power accounting for drivetrain losses [Nm]
        RotorThrust : float
            maximum input thrust from rotor for extreme conditions [N]
        RotorSpeed : float
            rotor speed at rated power [rpm]
        iDsgn : int
            Drivetrain design type (1 to 4)
        MachineRating : float
            Machine rating for turbine [kW]
        GearRatio : float
            gear ratio for full gearbox (high speed shaft to low speed shaft speed)
        GearConfig : str
            number of gears and types in a string; allowable configurations include 'p', 'e', 'pp', 'ep', 'ee', 'epp', 'eep', 'eee'
        Bevel : int
            flag for presence of a bevel stage (0 if not present, 1 if present)
        TowerTopDiam : float
            diameter of tower top [m]
        crane : bool
            boolean for crane present on-board
        '''        
        self.crane = crane
        self.GearConfig = GearConfig
        
        #computation of mass for main drivetrian subsystems     
        self.lss.update_mass(RotorDiam,RotorMass,RotorTorque)
        self.mbg.update_mass(self.lss, RotorSpeed, RotorDiam)
        self.gear.update_mass(iDsgn,RotorTorque,GearRatio,GearConfig,Bevel,RotorDiam)
        self.stagemass = self.gear.getStageMass() #return gearbox stage masses
        self.hss.update_mass(MachineRating,RotorTorque,GearRatio,RotorDiam,self.lss.diameter)
        self.gen.update_mass(iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio)
        self.bpl.update_mass(iDsgn,RotorTorque,RotorMass,RotorThrust,RotorDiam,TowerTopDiam)

        # electronic systems, hydraulics and controls 
        self.econnectionsMass = 0.0

        self.vspdEtronicsMass = 0.0
               
        self.hydrCoolingMass = 0.08 * MachineRating
 
        self.ControlsMass     = 0.0

        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        self.nacellePlatformsMass = 0.125 * self.bpl.mass

        if (self.crane):
            self.craneMass =  3000.0
        else:
            self.craneMass = 0.0
 
        self.mainframeMass  = self.bpl.mass + self.craneMass + self.nacellePlatformsMass     
        
        nacelleCovArea      = 2 * (self.bpl.length ** 2)              # this calculation is based on Sunderland
        self.nacelleCovMass = (84.1 * nacelleCovArea) / 2          # this calculation is based on Sunderland - divided by 2 in order to approach CSM

        self.length      = self.bpl.length                              # nacelle length [m] based on bedplate length
        self.width       = self.bpl.width                        # nacelle width [m] based on bedplate width
        self.height      = (2.0 / 3.0) * self.length                         # nacelle height [m] calculated based on cladding area
        
        # yaw system weight calculations based on total system mass above yaw system
        self.aboveYawMass =  self.lss.mass + \
                    self.mbg.mass + \
                    self.gear.mass + \
                    self.hss.mass + \
                    self.gen.mass + \
                    self.mainframeMass + \
                    self.econnectionsMass + \
                    self.vspdEtronicsMass + \
                    self.hydrCoolingMass + \
                    self.nacelleCovMass
        self.yaw.update_mass(RotorDiam,RotorThrust,TowerTopDiam,self.aboveYawMass)   # yaw mass calculation based on Sunderalnd model

        # aggregation of nacelle mass
        self.mass = (self.aboveYawMass + self.yaw.mass)

        # calculation of mass center and moments of inertia
        cm = np.array([0.0,0.0,0.0])
        for i in (range(0,3)):
            # calculate center of mass
            cm[i] = (self.lss.mass * self.lss.cm[i] + 
                    self.mbg.mainBearing.mass * self.mbg.mainBearing.cm[i] + self.mbg.secondBearing.mass * self.mbg.secondBearing.cm[i] + \
                    self.gear.mass * self.gear.cm[i] + self.hss.mass * self.hss.cm[i] + \
                    self.gen.mass * self.gen.cm[i] + self.bpl.mass * self.bpl.cm[i] ) / \
                    (self.lss.mass + self.mbg.mainBearing.mass + self.mbg.secondBearing.mass + \
                    self.gear.mass + self.hss.mass + self.gen.mass + self.bpl.mass)
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
            # calculate moments around CM
            # sum moments around each components CM
            I[i]  =  self.lss.I[i] + self.mbg.mainBearing.I[i] + self.mbg.secondBearing.I[i] + self.gear.I[i] + \
                          self.hss.I[i] + self.gen.I[i] + self.bpl.I[i]
            # translate to nacelle CM using parallel axis theorem
            for j in (range(0,3)): 
                if i != j:
                    I[i] +=  self.lss.mass * (self.lss.cm[i] - cm[i]) ** 2 + \
                                  self.mbg.mainBearing.mass * (self.mbg.mainBearing.cm[i] - cm[i]) ** 2 + \
                                  self.mbg.secondBearing.mass * (self.mbg.secondBearing.cm[i] - cm[i]) ** 2 + \
                                  self.gear.mass * (self.gear.cm[i] - cm[i]) ** 2 + \
                                  self.hss.mass * (self.hss.cm[i] - cm[i]) ** 2 + \
                                  self.gen.mass * (self.gen.cm[i] - cm[i]) ** 2 + \
                                  self.bpl.mass * (self.bpl.cm[i] - cm[i]) ** 2
        self.I = I

    def getNacelleComponentMasses(self):
        """ Returns detailed nacelle assembly masses
        
        detailedMasses : array_like of float
           detailed masses for nacelle components
        """
        
        detailedMasses = [self.lss.mass, self.mbg.mass, self.gear.mass, self.hss.mass, self.gen.mass, self.vspdEtronicsMass, \
                self.econnectionsMass, self.hydrCoolingMass, \
                self.ControlsMass, self.yaw.mass, self.mainframeMass, self.nacelleCovMass]

        return detailedMasses

#------------------------------------------------------------------

def example():

    # test of module for turbine data set

    # NREL 5 MW Rotor Variables
    RotorDiam = 126.0 # m
    RotorSpeed = 12.13 # m/s
    RotorTorque = 4365248.74 # Nm
    RotorThrust = 500930.84 # N
    RotorMass = 142585.75 # kg

    # NREL 5 MW Drivetrain variables
    iDsgn = 1 # geared 3-stage Gearbox with induction generator machine
    MachineRating = 5000.0 # kW
    GearRatio = 97.0 # 97:1 as listed in the 5 MW reference document
    GearConfig = 'eep' # epicyclic-epicyclic-parallel
    Bevel = 0 # no bevel stage
    crane = True # onboard crane present

    # NREL 5 MW Tower Variables
    TowerTopDiam = 3.78 # m

    print '----- NREL 5 MW Turbine -----'
    nace = NacelleSystem(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, \
                   MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    print 'Nacelle system model results'
    print 'Low speed shaft %8.1f kg  %6.2f m length %6.2f m OD %6.2f m Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.lss.mass , nace.lss.length,nace.lss.diameter, nace.lss.I[0], nace.lss.I[1], nace.lss.I[2], nace.lss.cm[0], nace.lss.cm[1], nace.lss.cm[2])
    # 31257.3 kg
    print 'Main bearings   %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' % (nace.mbg.mass , nace.mbg.I[0], nace.mbg.I[1], nace.mbg.I[2], nace.mbg.cm[0], nace.mbg.cm[1], nace.mbg.cm[2])
    # 9731.4 kg
    print 'Gearbox         %8.1f kg %6.2f m length %6.2f m diameter %6.2f m height %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gear.mass, nace.gear.length, nace.gear.diameter, nace.gear.height, nace.gear.I[0], nace.gear.I[1], nace.gear.I[2], nace.gear.cm[0], nace.gear.cm[1], nace.gear.cm[2] )
    # 30237.6 kg
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.stagemass[1], nace.stagemass[2], nace.stagemass[3])
    print 'High speed shaft & brakes  %8.1f kg %6.2f m length %6.2f m OD %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.hss.mass, nace.hss.length, nace.hss.diameter, nace.hss.I[0], nace.hss.I[1], nace.hss.I[2], nace.hss.cm[0], nace.hss.cm[1], nace.hss.cm[2])
    # 1492.4 kg
    print 'Generator       %8.1f kg %6.2f m length %6.2f m width %6.2f m depth %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gen.mass, nace.gen.length, nace.gen.width, nace.gen.depth, nace.gen.I[0], nace.gen.I[1], nace.gen.I[2], nace.gen.cm[0], nace.gen.cm[1], nace.gen.cm[2])
    # 16699.9 kg
    print 'Variable speed electronics %8.1f kg' % (nace.vspdEtronicsMass)
    # 0.0 kg
    print 'Overall mainframe %8.1f kg %6.2f m length %6.2f m width' % (nace.mainframeMass, nace.bpl.length, nace.bpl.width)
    # 96932.9 kg
    print '     Bedplate     %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
         % (nace.bpl.mass, nace.bpl.I[0], nace.bpl.I[1], nace.bpl.I[2], nace.bpl.cm[0], nace.bpl.cm[1], nace.bpl.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.econnectionsMass)
    # 0.0 kg
    print 'HVAC system     %8.1f kg' % (nace.hydrCoolingMass )
    # 400.0 kg
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.nacelleCovMass , nace.height, nace.width, nace.length)
    # 9097.4 kg
    print 'Yaw system      %8.1f kg' % (nace.yaw.mass )
    # 11878.2 kg
    print 'Overall nacelle:  %8.1f kg cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.mass, nace.cm[0], nace.cm[1], nace.cm[2], nace.I[0], nace.I[1], nace.I[2]  )
    # 207727.1


def example2():

    # WindPACT 1.5 MW Drivetrain variables
    iDsgn = 1 # geared 3-stage Gearbox with induction generator machine
    MachineRating = 1500 # machine rating [kW]
    GearRatio = 87.965
    GearConfig = 'epp'
    Bevel = 0
    crane = True

    # WindPACT 1.5 MW Rotor Variables
    airdensity = 1.225 # air density [kg / m^3]
    MaxTipSpeed = 80 # max tip speed [m/s]
    RotorDiam = 70 # rotor diameter [m]
    RotorSpeed = 21.830
    DrivetrainEfficiency = 0.95
    RotorTorque = (MachineRating * 1000 / DrivetrainEfficiency) / (RotorSpeed * (pi / 30)) 
        # rotor torque [Nm] calculated from max / rated rotor speed and machine rating
    RotorThrust = 324000 
    RotorMass = 28560 # rotor mass [kg]

    # WindPACT 1.5 MW Tower Variables
    TowerTopDiam = 2.7 # tower top diameter [m]

    print '----- WindPACT 1.5 MW Turbine -----'
    nace = NacelleSystem(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, \
                   MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    print 'Nacelle system model results'
    print 'Low speed shaft %8.1f kg  %6.2f m length %6.2f m OD %6.2f m Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.lss.mass , nace.lss.length,nace.lss.diameter, nace.lss.I[0], nace.lss.I[1], nace.lss.I[2], nace.lss.cm[0], nace.lss.cm[1], nace.lss.cm[2])
    # 31257.3 kg
    print 'Main bearings   %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' % (nace.mbg.mass , nace.mbg.I[0], nace.mbg.I[1], nace.mbg.I[2], nace.mbg.cm[0], nace.mbg.cm[1], nace.mbg.cm[2])
    # 9731.4 kg
    print 'Gearbox         %8.1f kg %6.2f m length %6.2f m diameter %6.2f m height %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gear.mass, nace.gear.length, nace.gear.diameter, nace.gear.height, nace.gear.I[0], nace.gear.I[1], nace.gear.I[2], nace.gear.cm[0], nace.gear.cm[1], nace.gear.cm[2] )
    # 30237.6 kg
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.stagemass[1], nace.stagemass[2], nace.stagemass[3])
    print 'High speed shaft & brakes  %8.1f kg %6.2f m length %6.2f m OD %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.hss.mass, nace.hss.length, nace.hss.diameter, nace.hss.I[0], nace.hss.I[1], nace.hss.I[2], nace.hss.cm[0], nace.hss.cm[1], nace.hss.cm[2])
    # 1492.4 kg
    print 'Generator       %8.1f kg %6.2f m length %6.2f m width %6.2f m depth %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gen.mass, nace.gen.length, nace.gen.width, nace.gen.depth, nace.gen.I[0], nace.gen.I[1], nace.gen.I[2], nace.gen.cm[0], nace.gen.cm[1], nace.gen.cm[2])
    # 16699.9 kg
    print 'Variable speed electronics %8.1f kg' % (nace.vspdEtronicsMass)
    # 0.0 kg
    print 'Overall mainframe %8.1f kg %6.2f m length %6.2f m width' % (nace.mainframeMass, nace.bpl.length, nace.bpl.width)
    # 96932.9 kg
    print '     Bedplate     %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
         % (nace.bpl.mass, nace.bpl.I[0], nace.bpl.I[1], nace.bpl.I[2], nace.bpl.cm[0], nace.bpl.cm[1], nace.bpl.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.econnectionsMass)
    # 0.0 kg
    print 'HVAC system     %8.1f kg' % (nace.hydrCoolingMass )
    # 400.0 kg
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.nacelleCovMass , nace.height, nace.width, nace.length)
    # 9097.4 kg
    print 'Yaw system      %8.1f kg' % (nace.yaw.mass)
    # 11878.2 kg
    print 'Overall nacelle:  %8.1f kg cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.mass, nace.cm[0], nace.cm[1], nace.cm[2], nace.I[0], nace.I[1], nace.I[2]  )
    # 207727.1
 
    # GRC Drivetrain variables
    nace = Nacelle()
    iDesign = 1 # geared 3-stage Gearbox with induction generator machine
    MachineRating = 750 # machine rating [kW]
    GearRatio = 81.491 
    GearConfig = 'epp'
    Bevel = 0

    # GRC Rotor Variables
    airdensity = 1.225 # air density [kg / m^3]
    MaxTipSpeed = 8 # max tip speed [m/s]
    RotorDiam = 48.2 # rotor diameter [m]
    #RotorSpeed = MaxTipSpeed / ((RotorDiam / 2) * (pi / 30)) # max / rated rotor speed [rpm] calculated from max tip speed and rotor diamter
    RotorSpeed = 22
    DrivetrainEfficiency = 0.944
    RotorTorque = (MachineRating * 1000 / DrivetrainEfficiency) / (RotorSpeed * (pi / 30)) # rotor torque [Nm] calculated from max / rated rotor speed and machine rating
    RotorThrust = 159000 # based on windpact 750 kW design (GRC information not available)
    RotorMass = 13200 # rotor mass [kg]

    # Tower Variables
    TowerTopDiam = 2 # tower top diameter [m] - not given

    print '----- GRC 750 kW Turbine -----'
    nace.update_mass(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, \
                   MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    print 'Nacelle system model results'
    print 'Low speed shaft %8.1f kg  %6.2f m length %6.2f m OD %6.2f m Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.lss.mass , nace.lss.length,nace.lss.diameter, nace.lss.I[0], nace.lss.I[1], nace.lss.I[2], nace.lss.cm[0], nace.lss.cm[1], nace.lss.cm[2])
    # 31257.3 kg
    print 'Main bearings   %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' % (nace.mbg.mass , nace.mbg.I[0], nace.mbg.I[1], nace.mbg.I[2], nace.mbg.cm[0], nace.mbg.cm[1], nace.mbg.cm[2])
    # 9731.4 kg
    print 'Gearbox         %8.1f kg %6.2f m length %6.2f m diameter %6.2f m height %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gear.mass, nace.gear.length, nace.gear.diameter, nace.gear.height, nace.gear.I[0], nace.gear.I[1], nace.gear.I[2], nace.gear.cm[0], nace.gear.cm[1], nace.gear.cm[2] )
    # 30237.6 kg
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.stagemass[1], nace.stagemass[2], nace.stagemass[3])
    print 'High speed shaft & brakes  %8.1f kg %6.2f m length %6.2f m OD %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.hss.mass, nace.hss.length, nace.hss.diameter, nace.hss.I[0], nace.hss.I[1], nace.hss.I[2], nace.hss.cm[0], nace.hss.cm[1], nace.hss.cm[2])
    # 1492.4 kg
    print 'Generator       %8.1f kg %6.2f m length %6.2f m width %6.2f m depth %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gen.mass, nace.gen.length, nace.gen.width, nace.gen.depth, nace.gen.I[0], nace.gen.I[1], nace.gen.I[2], nace.gen.cm[0], nace.gen.cm[1], nace.gen.cm[2])
    # 16699.9 kg
    print 'Variable speed electronics %8.1f kg' % (nace.vspdEtronicsMass)
    # 0.0 kg
    print 'Overall mainframe %8.1f kg %6.2f m length %6.2f m width' % (nace.mainframeMass, nace.bpl.length, nace.bpl.width)
    # 96932.9 kg
    print '     Bedplate     %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
         % (nace.bpl.mass, nace.bpl.I[0], nace.bpl.I[1], nace.bpl.I[2], nace.bpl.cm[0], nace.bpl.cm[1], nace.bpl.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.econnectionsMass)
    # 0.0 kg
    print 'HVAC system     %8.1f kg' % (nace.hydrCoolingMass )
    # 400.0 kg
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.nacelleCovMass , nace.height, nace.width, nace.length)
    # 9097.4 kg
    print 'Yaw system      %8.1f kg' % (nace.yaw.mass)
    # 11878.2 kg
    print 'Overall nacelle:  %8.1f kg cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.mass, nace.cm[0], nace.cm[1], nace.cm[2], nace.I[0], nace.I[1], nace.I[2]  )
    # 207727.1


if __name__ == '__main__':
    ''' Main runs through tests of several drivetrain configurations with known component masses and dimensions '''
    
    # todo: adjust to use rotor model interface

    example()
    
    #example2()