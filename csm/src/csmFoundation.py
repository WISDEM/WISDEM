""" 
csmFoundation.py
    
Created by George Scott 2012
Modified  by Katherine Dykes 2012
Copyright (c)  NREL. All rights reserved.

"""

from csmPPI import *
from config import *

import sys

#-------------------------------------------------------------------------------        

class csmFoundation(object):

    """ NREL Cost and Scaling Balance of Station foundation cost model. """

    def __init__(self):

        """
           Initialize properties for balance of plant foundation cost.
        """  

        self.cost = 0.0

        pass
        
    def compute(self, MachineRating,HubHt,RotorDiam,SeaDepth, curr_yr=2009, curr_mon=12, verbose=0):
        """
        Computes the balance of station foundation costs for a wind plant using the NREL Cost and Scaling Model method.
        
        Parameters
        ----------

        MachineRating : float
          machine rating [kW] for the wind turbine at the site
        HubHt : float
          hub height [m] of the wind turbines at the site
        RotorDiam : float
          rotor diameter [m] of the wind turbines at the site
        SeaDepth : float
          sea depth [m] which is 0.0 or negative for an onshore project
        curr_yr : int
          project start year [year]
        curr_mon : int
          project start month [month]
        
        """  
        ppi.ref_yr = 2002
        ppi.ref_mon = 9
        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon
        
        if SeaDepth == 0.0: # 1: land-based, 2: shallow, 3: 30-60m, 4: >60m
          fdnType = 1
        elif SeaDepth < 30:
          fdnType = 2
        elif SeaDepth < 60:
          fdnType = 3
        else:
          fdnType = 4
        
        self.ftype = fdnType 
        
        if (self.ftype == 1): # land
            fcCoeff = 303.23
            fcExp   = 0.4037
            SweptArea = (RotorDiam*0.5)**2.0 * pi
            self.cost = fcCoeff * (HubHt*SweptArea)**fcExp
            fndnCostEscalator = ppi.compute('IPPI_FND')
        elif (self.ftype == 2):
            sscf = 300.0 # $/kW
            self.cost = sscf*MachineRating
            ppi.ref_yr = 2003                                     # reference year for offshore foundations is 2003 (as with offshore BOS variables)
            fndnCostEscalator = ppi.compute('IPPI_MPF')
            ppi.ref_yr = 2002
        elif (self.ftype == 3):
            sscf = 450.0 # $/kW
            self.cost = sscf*MachineRating
            ppi.ref_yr = 2003
            fndnCostEscalator = ppi.compute('IPPI_OAI')
            ppi.ref_yr = 2002
        elif (self.ftype == 4):
            self.cost = 0.0
            fndnCostEscalator = 1.0

        self.cost *= fndnCostEscalator
        
        if (verbose > 0):
        	print "foundation cost: {0}".format(self.cost)
        
    def getCost(self):
        """ 
        Provides the overall balance of station costs for the plant.

        Returns
        -------
        cost : float
            Balance of plant foundation costs [USD]
        """

        return self.cost

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
    
    foundation = csmFoundation()
    
    rotorDiameter = 126.0
    hubHeight = 90.0
    ratedPower = 5000.0
    seaDepth = 0.0
    
    print "Onshore foundation cost"
    foundation.compute(ratedPower, hubHeight, rotorDiameter, seaDepth, curr_yr, curr_mon, 1)
    
    seaDepth = 20.0

    print "Offshore foundation cost"
    foundation.compute(ratedPower, hubHeight, rotorDiameter, seaDepth, curr_yr, curr_mon, 1)	        

if __name__ == "__main__":

    example()
