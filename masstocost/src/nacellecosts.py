"""
nacellecosts.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from config import *
from common import ComponentCost
from zope.interface import implements

# -------------------------------------------------

class LowSpeedShaftCost():
    implements(ComponentCost)

    def __init__(self, lssMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine low speed shaft component.       
          
          Parameters
          ----------
          lssMass : float
            lss mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(lssMass, curr_yr, curr_mon)
    
    def update_cost(self, lssMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine low speed shaft component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          lssMass : float
            lss mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon
        
        # calculate component cost        
        LowSpeedShaftCost2002 = 3.3602 * lssMass + 13587      # equation adjusted to be based on mass rather than rotor diameter using data from CSM
        lssCostEsc            = ppi.compute('IPPI_LSS')
        self.cost = (LowSpeedShaftCost2002 * lssCostEsc )

#-------------------------------------------------------------------------------

class MainBearingsCost(): 
    implements(ComponentCost)

    def __init__(self, bearingsMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine maing bearings.       
          
          Parameters
          ----------
          bearingsMass : float
            bearing mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(bearingsMass, curr_yr, curr_mon)
    
    def update_cost(self, bearingsMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine main bearings.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          bearingsMass : float
            bearing mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        # calculate component cost
        bearingCostEsc       = ppi.compute('IPPI_BRN')

        brngSysCostFactor = 17.6 # $/kg                  # cost / unit mass from CSM
        Bearings2002 = (bearingsMass) * brngSysCostFactor
        self.cost    = (( Bearings2002 ) * bearingCostEsc ) / 4   # div 4 to account for bearing cost mass differences CSM to Sunderland  
             

#-------------------------------------------------------------------------------

class GearboxCost():  
    implements(ComponentCost)

    def __init__(self, gearboxMass, iDesign, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine gearbox component.       
          
          Parameters
          ----------
          gearboxMass : float
            gearbox mass [kg]
          iDesign : int
            machine configuration 1 conventional, 2 medium speed, 3 multi-gen, 4 direct-drive
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(gearboxMass, iDesign, curr_yr, curr_mon)
    
    def update_cost(self, gearboxMass, iDesign, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine gearbox component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          gearboxMass : float
            gearbox mass [kg]]
          iDesign : int
            machine configuration 1 conventional, 2 medium speed, 3 multi-gen, 4 direct-drive
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        # calculate component cost                                              
        GearboxCostEsc     = ppi.compute('IPPI_GRB')

        costCoeff = [None, 16.45  , 74.101     ,   15.25697015,  0 ]
        costExp   = [None,  1.2491,  1.002     ,    1.2491    ,  0 ]

        if iDesign == 1:                                 
          Gearbox2002 = 16.9 * gearboxMass - 25066          # for traditional 3-stage gearbox, use mass based cost equation from NREL CSM
        else:
          Gearbox2002 = costCoeff[iDsgn] * (MachineRating ** costCoeff[iDesign])        # for other drivetrain configurations, use NREL CSM equation based on machine rating

        self.cost   = Gearbox2002 * GearboxCostEsc      

#-------------------------------------------------------------------------------
              
class HighSpeedShaftCost():
    implements(ComponentCost)

    def __init__(self, mechBrakeMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine mechanical brake and HSS component.       
          
          Parameters
          ----------
          mechBrakeMass : float
            mechBrake mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(mechBrakeMass, curr_yr, curr_mon)
    
    def update_cost(self, mechBrakeMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine mechanical brake and HSS component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          mechBrakeMass : float
            mechBrake mass [kg]]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon 
        # calculate component cost
        mechBrakeCostEsc     = ppi.compute('IPPI_BRK')
        mechBrakeCost2002    = 10 * mechBrakeMass                  # mechanical brake system cost based on $10 / kg multiplier from CSM model (inverse relationship)
        self.cost            = mechBrakeCostEsc * mechBrakeCost2002                                

#-------------------------------------------------------------------------------

class GeneratorCost():
    implements(ComponentCost)

    def __init__(self, generatorMass, iDesign, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine generator component.       
          
          Parameters
          ----------
          generatorMass : float
            generator mass [kg]
          iDesign : int
            machine configuration 1 conventional, 2 medium speed, 3 multi-gen, 4 direct-drive
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(generatorMass, iDesign, curr_yr, curr_mon)
    
    def update_cost(self, generatorMass, iDesign, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine generator component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          generatorMass : float
            generator mass [kg]]
          iDesign : int
            machine configuration 1 conventional, 2 medium speed, 3 multi-gen, 4 direct-drive
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon
                                                        
        # calculate component cost                                      #TODO: only handles traditional drivetrain configuration at present
        generatorCostEsc     = ppi.compute('IPPI_GEN')
        costCoeff = [None, 65    , 54.73 ,  48.03 , 219.33 ] # $/kW - from 'Generators' worksheet

        GeneratorCost2002 = 19.697 * generatorMass + 9277.3
        self.cost         = GeneratorCost2002 * generatorCostEsc 
                       

#-------------------------------------------------------------------------------

class BedplateCost():
    implements(ComponentCost)

    def __init__(self, bedplateMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine bedplate component.       
          
          Parameters
          ----------
          bedplateMass : float
            bedplate mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(bedplateMass, curr_yr, curr_mon)
    
    def update_cost(self, bedplateMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine bedplate component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          bedplateMass : float
            bedplate mass [kg]]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        #calculate component cost                                    # TODO: cost needs to be adjusted based on look-up table or a materials, mass and manufacturing equation            
        BedplateCostEsc     = ppi.compute('IPPI_MFM')

        costCoeff = [None, 9.48850 , 303.96000, 17.92300 , 627.280000 ]
        costExp   = [None, 1.9525, 1.0669, 1.6716, 0.85]

        self.cost2002 = 0.9461 * bedplateMass + 17799                   # equation adjusted based on mass / cost relationships for components documented in NREL CSM
        self.cost     = self.cost2002 * BedplateCostEsc
      

#-------------------------------------------------------------------------------
   
class YawSystemCost():
    implements(ComponentCost)

    def __init__(self, yawSystemMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine yaw system.       
          
          Parameters
          ----------
          yawSystemMass : float
            yawSystem mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(yawSystemMass, curr_yr, curr_mon)
    
    def update_cost(self, yawSystemMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine yaw system.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          yawSystemMass : float
            yawSystem mass [kg]]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        # calculate component cost
        yawDrvBearingCostEsc = ppi.compute('IPPI_YAW')

        YawDrvBearing2002 = 8.3221 * yawSystemMass + 2708.5          # cost / mass relationship derived from NREL CSM data
        self.cost         = YawDrvBearing2002 * yawDrvBearingCostEsc 
               

#-------------------------------------------------------------------------------

class NacelleSystemCost(): # changed name to nacelle - need to rename, move code pieces, develop configurations ***
    implements(ComponentCost)

    def __init__(self, lssMass, bearingsMass, gearboxMass, mechBrakeMass, generatorMass, bedplateMass, yawSystemMass, machineRating, iDesign, offshore, curr_yr, curr_mon, crane=False):
        '''
          Initial computation of the costs for the wind turbine gearbox component.       
          
          Parameters
          ----------
          lssMass : float
            Low speed shaft mass [kg]
          bearingsMass : float
            bearing mass [kg]
          gearboxMass : float
            Gearbox mass [kg]
          mechBrakeMass : float
            High speed shaft mass [kg]
          bedplateMass : float
            Bedplate mass [kg]
          yawSystemMass : float
            Yaw system mass [kg]
          MachineRating : float
            Machine rating for turbine [kW]
          iDesign : int
            machine configuration 1 conventional, 2 medium speed, 3 multi-gen, 4 direct-drive
          offshore : bool
            boolean true if it is offshore
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          crane : bool
              boolean for crane present on-board          
        '''

        self.lss = LowSpeedShaftCost(lssMass, curr_yr, curr_mon)
        self.bearings = MainBearingsCost(bearingsMass, curr_yr, curr_mon)
        self.gearbox = GearboxCost(gearboxMass, iDesign, curr_yr, curr_mon)
        self.hss = HighSpeedShaftCost(mechBrakeMass, curr_yr, curr_mon)
        self.generator = GeneratorCost(generatorMass, iDesign, curr_yr, curr_mon)
        self.bedplate = BedplateCost(bedplateMass, curr_yr, curr_mon)
        self.yawsystem = YawSystemCost(yawSystemMass, curr_yr, curr_mon)
        
        self.update_cost(lssMass, bearingsMass, gearboxMass, mechBrakeMass, generatorMass, bedplateMass, yawSystemMass, machineRating, iDesign, offshore, curr_yr, curr_mon, crane)
    
    def update_cost(self, lssMass, bearingsMass, gearboxMass, mechBrakeMass, generatorMass, bedplateMass, yawSystemMass, machineRating, iDesign, offshore, curr_yr, curr_mon, crane=False):

        '''
          Computes the costs for the wind turbine gearbox component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          lssMass : float
            Low speed shaft mass [kg]
          bearingsMass : float
            bearing mass [kg]
          gearboxMass : float
            Gearbox mass [kg]
          mechBrakeMass : float
            High speed shaft mass [kg]
          bedplateMass : float
            Bedplate mass [kg]
          yawSystemMass : float
            Yaw system mass [kg]
          MachineRating : float
            Machine rating for turbine [kW]
          iDesign : int
            machine configuration 1 conventional, 2 medium speed, 3 multi-gen, 4 direct-drive
          offshore : bool
            boolean true if it is offshore
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          crane : bool
              boolean for crane present on-board          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        self.lss.update_cost(lssMass, curr_yr, curr_mon)
        self.bearings.update_cost(bearingsMass, curr_yr, curr_mon)
        self.gearbox.update_cost(gearboxMass, iDesign, curr_yr, curr_mon)
        self.hss.update_cost(mechBrakeMass, curr_yr, curr_mon)
        self.generator.update_cost(generatorMass, iDesign, curr_yr, curr_mon)
        self.bedplate.update_cost(bedplateMass, curr_yr, curr_mon)
        self.yawsystem.update_cost(yawSystemMass, curr_yr, curr_mon)

        # calculations of mass and cost for other systems not included above as main drivetrain load-bearing components
        # Cost Escalators - should be obtained from PPI tables
        BedplateCostEsc      = ppi.compute('IPPI_MFM')
        VspdEtronicsCostEsc  = ppi.compute('IPPI_VSE')
        nacelleCovCostEsc    = ppi.compute('IPPI_NAC')
        hydrCoolingCostEsc   = ppi.compute('IPPI_HYD')
        econnectionsCostEsc  = ppi.compute('IPPI_ELC')
        controlsCostEsc      = ppi.compute('IPPI_CTL')

        # electronic systems, hydraulics and controls
        econnectionsCost2002  = 40.0 * machineRating  # 2002
        self.econnectionsCost = econnectionsCost2002 * econnectionsCostEsc
               
        VspdEtronics2002      = 79.32 * machineRating
        self.vspdEtronicsCost = VspdEtronics2002 * VspdEtronicsCostEsc         

        hydrCoolingCost2002  = 12.0 * machineRating # 2002
        self.hydrCoolingCost = hydrCoolingCost2002 * hydrCoolingCostEsc   

        if (not offshore):
            ControlsCost2002  = 35000.0 # initial approximation 2002
            self.controlsCost = ControlsCost2002 * controlsCostEsc 
        else:
            ControlsCost2002  = 55900.0 # initial approximation 2002
            self.controlsCost = ControlsCost2002 * controlsCostEsc      

        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        nacellePlatformsMass = 0.125 * bedplateMass
        NacellePlatforms2002 = 8.7 * nacellePlatformsMass

        if (crane):
            craneCost2002  = 12000.0
        else:
            craneCost2002  = 0.0

        # aggregation of mainframe components: bedplate, crane and platforms into single mass and cost
        BaseHardwareCost2002  = self.bedplate.cost2002 * 0.7
        MainFrameCost2002   = (NacellePlatforms2002 + craneCost2002  + \
                          BaseHardwareCost2002 )
        self.mainframeCost  = MainFrameCost2002 * BedplateCostEsc + self.bedplate.cost       
        
        nacelleCovCost2002  = 11.537 * machineRating + (3849.7)
        self.nacelleCovCost = nacelleCovCost2002 * nacelleCovCostEsc 
        
        
        # aggregation of nacelle costs
        self.cost = self.lss.cost + \
                    self.bearings.cost + \
                    self.gearbox.cost + \
                    self.hss.cost + \
                    self.generator.cost + \
                    self.yawsystem.cost + \
                    self.econnectionsCost + \
                    self.vspdEtronicsCost + \
                    self.hydrCoolingCost + \
                    self.mainframeCost + \
                    self.controlsCost + \
                    self.nacelleCovCost

#------------------------------------------------------------------

def example():

    # test of module for turbine data set
    
    ref_yr   = 2002
    ref_mon  =    9
    curr_yr  = 2009
    curr_mon =  12
    
    ppi.ref_yr   = ref_yr
    ppi.ref_mon  = ref_mon

    lssMass = 31257.3
    bearingsMass = 9731.41
    gearboxMass = 30237.60
    hssMass = 1492.45
    generatorMass = 16699.85
    bedplateMass = 93090.6
    yawSystemMass = 11878.24
    machineRating = 5000.0
    iDesign = 1
    crane = True
    offshore = False

    lss = LowSpeedShaftCost(lssMass, curr_yr, curr_mon)
    bearings = MainBearingsCost(bearingsMass, curr_yr, curr_mon)
    gearbox = GearboxCost(gearboxMass, iDesign, curr_yr, curr_mon)
    hss = HighSpeedShaftCost(hssMass, curr_yr, curr_mon)
    generator = GeneratorCost(generatorMass, iDesign, curr_yr, curr_mon)
    bedplate = BedplateCost(bedplateMass, curr_yr, curr_mon)
    yawsystem = YawSystemCost(yawSystemMass, curr_yr, curr_mon)

    print "LSS cost is ${0:.2f} USD".format(lss.cost) # $183363.52
    print "Main bearings cost is ${0:.2f} USD".format(bearings.cost) # $56660.71
    print "Gearbox cost is ${0:.2f} USD".format(gearbox.cost) # $648030.18
    print "HSS cost is ${0:.2f} USD".format(hss.cost) # $15218.20
    print "Generator cost is ${0:.2f} USD".format(generator.cost) # $435157.75
    print "Bedplate cost is ${0:.2f} USD".format(bedplate.cost)
    print "Yaw system cost is ${0:.2f} USD".format(yawsystem.cost) # $137609.38

    nacelle = NacelleSystemCost(lssMass, bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, machineRating, iDesign, offshore, curr_yr, curr_mon, crane)
    print "Overall nacelle cost is ${0:.2f} USD".format(nacelle.cost) # $2884227.08

if __name__ == '__main__':

    example()