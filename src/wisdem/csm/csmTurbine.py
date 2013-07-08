"""
csmTurbine.py

Created by George Scott on 2012-08-01.
Modified by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from csmPPI import *
from config import *
from csmBlades import csmBlades
from csmHub import csmHub
from csmNacelle import csmNacelle
from csmTower import csmTower

import sys
        
#-------------------------------------------------------------------------------

class csmTurbine(object):
    ''' Turbine class 
        inputs are adjusted to be only those needed for mass and cost calculations; these include:
            hubHeight=90.0, machineRating=5000.0, maxTipSpd=80.0, rotorDiam=126.0, dtDesign=1, nblades = 3, maxEfficiency=0.910201, ratedWindSpd=12.0, idepth=1)       
    '''
    def __init__(self):

        """
           Initialize properties for operations and maintenance costs.
        """
        
        # output variables
        self.mass = 0.0
        self.cost = 0.0

        # input variable assignment
        self.hubHeight = 90
        self.machineRating = 5000
        self.dtDesign = 1 # 1: 3-stage planetary, 2: single low-speed, 3: multi-stage, 4: direct drive
        self.iDepth = 1  # 1: Land, 2: < 30m, 3: < 60m, 4: >= 60m
        self.rotorDiam = 126
        self.nblades = 3
        self.maxEfficiency = 0.90201
        self.ratedWindSpd = 11.506
        self.altitude = 0.0
        self.thrustCoeff = 0.50                                     
        self.maxTipSpd = 80.0
        self.crane = 1
        self.advancedBlade = 0
        self.airDensity = 1.225 # std air density
        self.offshore  = 0  # 0 = onshore

        # initialize ppi index calculator
        ref_yr  = 2002                    # always use 2002 as reference year for now
        ref_mon =    9
        self.curr_yr = 2009
        self.curr_mon = 12
        self.ppi = PPI(ref_yr, ref_mon, self.curr_yr, self.curr_mon)
        
        # Initialize the components of the turbine
        self.blades  = csmBlades()
        self.hub     = csmHub()    
        self.nac     = csmNacelle()
        self.tower   = csmTower()

        # calculate derivative input parameters for nacelle calculations       # todo - these should come from AEP/rotor module
        self.ratedHubPower  = self.machineRating / self.maxEfficiency 
        self.rotorSpeed     = (self.maxTipSpd/(0.5*self.rotorDiam)) * (60.0 / (2*pi))
        self.maximumThrust  = self.airDensity * self.thrustCoeff * pi * self.rotorDiam**2 * (self.ratedWindSpd**2) / 8
        self.rotorTorque = self.ratedHubPower/(self.rotorSpeed*(pi/30))*1000   # NREL internal version

        pass
    
    def compute(self, hubHeight=90.0, machineRating=5000.0, maxTipSpd=80.0, rotorDiam=126.0, dtDesign=1, nblades = 3, \
                       maxEfficiency=0.90201, ratedWindSpd = 11.5064, altitude=0.0, thrustCoeff=0.50, seaDepth=0.0, crane=True, advancedBlade = False, \
                       advancedBedplate = 1, year = 2009, month = 12, Verbose=0):
        """
        Computes the turbine component masses and costs using the NREL Cost and Scaling Model method.
        
        Parameters
        ----------
        hubHeight : float
          hub height [m] of the wind turbines at the site
        machineRating : float
          machine rating [kW] for the wind turbine at the site
        maxTipSpd : float
          maximum allowable tip speed [m/s] for the wind turbine
        rotorDiam : float
          rotor diameter [m] of the wind turbines at the site
        dtDesign : int
          drivetrain design [1 = 3-stage geared, 2 = single-stage, 3 = multi-gen, 4 = direct drive]
        nblades : int
          number of rotor blades
        maxEfficiency : float
          maximum efficiency of the drivetrain
        ratedWindSpd : float
          wind speed at which turbine produced rated power [m/s]
        altitude : float
          altitude [m] of wind farm above sea level for an onshore plant
        seaDepth : float
          sea depth [m] which is 0.0 or negative for an onshore project
        crane : bool
          boolean for the presence of an on-board service crane
        advancedBlade : bool
          boolean for the presence of an advanced wind turbine blade configuration
        advancedBedplate : int
          bedplate design for standard, modular or integrated
        year : int
          project start year [year]
        month : int
          project start month [month]
        
        """

        # input variable assignment
        self.hubHeight = hubHeight
        self.machineRating = machineRating
        self.dtDesign = dtDesign # 1: 3-stage planetary, 2: single low-speed, 3: multi-stage, 4: direct drive

        if seaDepth == 0.0:            # type of plant # 1: Land, 2: < 30m, 3: < 60m, 4: >= 60m
            self.iDepth = 1
        elif seaDepth < 30:
            self.iDepth = 2
        elif seaDepth < 60:
            self.iDepth = 3
        else:
            self.iDepth = 4
        self.seaDepth = seaDepth

        self.rotorDiam = rotorDiam
        self.nblades = nblades
        self.maxEfficiency = maxEfficiency
        self.ratedWindSpd = ratedWindSpd
        self.altitude = altitude
        self.thrustCoeff = thrustCoeff                                     
        self.maxTipSpd = maxTipSpd
        self.crane = crane
        self.advancedBlade = advancedBlade
        self.advancedBedplate = advancedBedplate
        self.year = year
        self.month = month

        if (self.iDepth == 1):   # TODO - crane assignment should be an input
            self.offshore  = 0  # 0 = onshore
        else:
            self.offshore  = 1  # 1 = offshore

        # initialize ppi index calculator
        self.ppi.curr_yr = self.year
        self.ppi.curr_mon = self.month

        # Compute air density (todo: this is redundant from csm AEP, calculation of environmental variables of interest should probably be its own model)        
        ssl_pa     = 101300  # std sea-level pressure in Pa
        gas_const  = 287.15  # gas constant for air in J/kg/K
        gravity    = 9.80665 # standard gravity in m/sec/sec
        lapse_rate = 0.0065  # temp lapse rate in K/m
        ssl_temp   = 288.15  # std sea-level temp in K
        
        self.airDensity = (ssl_pa * (1-((lapse_rate*(self.altitude + self.hubHeight))/ssl_temp))**(gravity/(lapse_rate*gas_const))) / \
            (gas_const*(ssl_temp-lapse_rate*(self.altitude + self.hubHeight)))

        # calaculate derivative input parameters for nacelle calculations       # todo - these should come from AEP/rotor module
        self.ratedHubPower  = self.machineRating / self.maxEfficiency 
        self.rotorSpeed     = (self.maxTipSpd/(0.5*self.rotorDiam)) * (60.0 / (2*pi))
        self.maximumThrust  = self.airDensity * self.thrustCoeff * pi * self.rotorDiam**2 * (self.ratedWindSpd**2) / 8
        self.rotorTorque = self.ratedHubPower/(self.rotorSpeed*(pi/30))*1000   # NREL internal version
        
        # sub-component computations for mass, cost and dimensions          
        self.blades.compute(self.rotorDiam,self.advancedBlade,self.curr_yr,self.curr_mon)
        
        self.hub.compute(self.blades.getMass(), self.rotorDiam,self.nblades,self.curr_yr,self.curr_mon)

        self.rotorMass = self.blades.getMass() * self.nblades + self.hub.getMass()
        self.rotorCost = self.blades.getCost() * self.nblades + self.hub.getCost()
        
        self.nac.compute(self.rotorDiam, self.machineRating, self.rotorMass, self.rotorSpeed, \
                      self.maximumThrust, self.rotorTorque, self.dtDesign, self.offshore, \
                      self.crane, self.advancedBedplate, self.curr_yr, self.curr_mon)

        self.tower.compute(self.rotorDiam, self.hubHeight, self.curr_yr, self.curr_mon)        
        
        self.cost = \
            self.rotorCost + \
            self.nac.cost + \
            self.tower.cost

        self.marCost = 0.0
        if (self.offshore == 1): # offshore - add marinization - NOTE: includes Foundation cost (not included in CSM.xls)
            marCoeff = 0.10 # 10%
            self.marCost = marCoeff * self.cost
            
        self.cost += self.marCost
            
        self.mass = \
            self.rotorMass + \
            self.nac.mass + \
            self.tower.mass

        pass

    def getMass(self):
        """ 
        Provides the overall turbine mass.

        Returns
        -------
        mass : float
            Total mass for wind turbine [kg]
        """

        return self.mass
        
    def getCost(self):
        """ 
        Provides the turbine capital costs for a single turbine.

        Returns
        -------
        cost : float
            Turbine capital costs [USD]
        """

        return self.cost

    def cm_print(self):  # print cost and mass of components
        print 'Turbine Components for %9.2f kW rated turbine' % (self.machineRating)        
        print '  Rotor      %9.2f $K %9.2f kg' % (self.rotorCost, self.rotorMass)
        print '  Nacelle    %9.2f $K %9.2f kg' % (self.nac.cost, self.nac.mass)
        print '  Tower      %9.2f $K %9.2f kg' % (self.tower.cost, self.tower.mass)
        if (self.offshore == 1):
            print '  Marinization %7.2f $K' % (self.marCost)
        print 'TURBINE TOTAL %8.2f $K %9.2f kg' % (self.cost, self.mass)
        print        

# ------------------------------------------------------------

def example():

    # simple test of module

    ref_yr  = 2002
    ref_mon =    9
    curr_yr = 2009
    curr_mon =  12
    
    ppi.ref_yr   = ref_yr
    ppi.ref_mon  = ref_mon
    ppi.curr_yr  = curr_yr
    ppi.curr_mon = curr_mon
    
    turb = csmTurbine()

    # 5 MW onshore turbine    
    turb.compute()
    
    print "Onshore Configuration 5 MW turbine"
    turb.cm_print()
    
    print "Offshore Configuration 5 MW turbine"
    # 5 MW offshore turbine with 20.0 m depth
    turb.compute(90.0, 5000.0, 80.0, 126.0, 1, 3, 0.90201, 11.5064, 0.0, 0.50, 20.0, True, False, 1, 2009, 12)
    
    turb.cm_print()


if __name__ == "__main__":  #TODO - update based on changes to csm Turbine

    example()