"""
turbinecost.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from config import *
from common import ComponentCost
from zope.interface import implements
from rotorcosts import RotorCost
from nacellecosts import NacelleSystemCost
from towercost import TowerCost

#-------------------------------------------------------------------------------        

class TurbineCost():
    implements(ComponentCost)

    def __init__(self, bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, towerMass, lssMass, \
                         bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, \
                         machineRating, iDesign, offshore, curr_yr, curr_mon, crane, advanced):
        '''
          Initial computation of the costs for the wind turbine.

          Parameters
          ----------
          bladeMass : float
            blade mass [kg]
          bladeNum : int
            Number of blades on rotor
          hubMass : float
            hub mass [kg]
          pitchSystemMass : float
            pitch system mass [kg]
          spinnerMass : float
            spinner mass [kg]
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
          towerMass : float
            mass [kg] of the wind turbine tower
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month 
          crane : bool
              boolean for crane present on-board
          advanced : bool
            boolean for advanced (using carbon) or basline (all fiberglass) blade        
        '''

        self.rotor = RotorCost(bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon, advanced)
        self.nacelle = NacelleSystemCost(lssMass, bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, machineRating, iDesign, offshore, curr_yr, curr_mon, crane)
        self.tower = TowerCost(towerMass, curr_yr, curr_mon)
        
        self.update_cost(bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, towerMass, lssMass, \
                         bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, \
                         machineRating, iDesign, offshore, curr_yr, curr_mon, crane, advanced)
    
    def update_cost(self, bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, towerMass, lssMass, \
                         bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, \
                         machineRating, iDesign, offshore, curr_yr, curr_mon, crane, advanced):

        '''
          Computes the costs for the wind turbine.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          bladeMass : float
            blade mass [kg]
          bladeNum : int
            Number of blades on rotor
          hubMass : float
            hub mass [kg]
          pitchSystemMass : float
            pitch system mass [kg]
          spinnerMass : float
            spinner mass [kg]
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
          towerMass : float
            mass [kg] of the wind turbine tower
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month 
          crane : bool
              boolean for crane present on-board
          advanced : bool
            boolean for advanced (using carbon) or basline (all fiberglass) blade          
        '''

        self.rotor.update_cost(bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon, advanced)
        self.nacelle.update_cost(lssMass, bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, machineRating, iDesign, offshore, curr_yr, curr_mon, crane)
        self.tower.update_cost(towerMass, curr_yr, curr_mon)
        
        self.cost = self.rotor.cost + self.nacelle.cost + self.tower.cost
        
#-------------------------------------------------------------------------------       

def example():

    # simple test of module
    
    ref_yr  = 2002
    ref_mon =    9
    curr_yr = 2009
    curr_mon =  12
    
    ppi.ref_yr   = ref_yr
    ppi.ref_mon  = ref_mon

    bladeMass = 17650.67  # inline with the windpact estimates
    advanced = True
    hubMass = 31644.5
    pitchSystemMass = 17004.0
    spinnerMass = 1810.5
    bladeNum = 3

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
    offshore = True

    towerMass = 434559.0

    turbine = TurbineCost(bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, towerMass, lssMass, \
                         bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, \
                         machineRating, iDesign, offshore, curr_yr, curr_mon, crane, advanced)
    
    print "Turbine cost is ${0:.2f} USD".format(turbine.cost) # $5350414.10
    print
    print "Overall rotor cost with 3 advanced blades is ${0:.2f} USD".format(turbine.rotor.cost)
    print "Advanced blade cost is ${0:.2f} USD".format(turbine.rotor.blade.cost)
    print "Cost of 3 blades is ${0:.2f} USD".format(turbine.rotor.blade.cost * 3)
    print "Hub cost is ${0:.2f} USD".format(turbine.rotor.hub.cost)   # $175513.50
    print "Pitch cost is ${0:.2f} USD".format(turbine.rotor.pitch.cost)  # $535075.0
    print "Spinner cost is ${0:.2f} USD".format(turbine.rotor.spinner.cost)  # $10509.00
    print
    print "Overall nacelle cost is ${0:.2f} USD".format(turbine.nacelle.cost) # $2884227.08
    print "LSS cost is ${0:.2f} USD".format(turbine.nacelle.lss.cost) # $183363.52
    print "Main bearings cost is ${0:.2f} USD".format(turbine.nacelle.bearings.cost) # $56660.71
    print "Gearbox cost is ${0:.2f} USD".format(turbine.nacelle.gearbox.cost) # $648030.18
    print "HSS cost is ${0:.2f} USD".format(turbine.nacelle.hss.cost) # $15218.20
    print "Generator cost is ${0:.2f} USD".format(turbine.nacelle.generator.cost) # $435157.75
    print "Bedplate cost is ${0:.2f} USD".format(turbine.nacelle.bedplate.cost)
    print "Yaw system cost is ${0:.2f} USD".format(turbine.nacelle.yawsystem.cost) # $137609.38
    print
    print "Tower cost is ${0:.2f} USD".format(turbine.tower.cost) # $987180.30    

if __name__ == "__main__":  

    example()