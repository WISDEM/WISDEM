"""
foundation.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

import sys

#-------------------------------------------------------------------------------        

class foundation(object):

    """
    Foundation class based on the NREL Offshore Balance of Station Model for foundation cost.
    """

    def __init__(self):
        """
        Initialize the parameters for the wind turbine monopile foundation
        
        Parameters
        ----------
        cost : float
          monopile foundation cost [USD]
        """

        self.cost = 0.0

        pass
        
    def compute(self, ratedPower, seaDepth, monopileMass=0.0, transitionMass=0.0, verbose=0):
        """
        Compute cost for a wind turbine foundation based on input masses
        
        Parameters
        ----------
        ratedPower : float
          project turbine rated power [MW]
        seaDepth : float
          project sea depth [m]
        monopileMass : float
          mass [kg] of monopile
        transitionMass : float
          mass [kg] of monopile transition piece
        """
        
        if ratedPower == 4.0:
            secondaryMass = ((35 + (ratedPower/1000)*5) + 0.8*(18 + seaDepth))*1000
        else:
            secondaryMass = ((35) + 0.8*(18 + seaDepth))*1000         
        secondaryCosts = secondaryMass * 7.250
        monopileCosts = monopileMass * 2.250
        transitionCosts = transitionMass * 2.750
        self.cost = monopileCosts + transitionCosts + secondaryCosts
        
        if (verbose > 0):
          print "foundation cost: {0}".format(self.cost)
        
    def getCost(self):
        """ 
        Provides the cost for the wind turbine monopile foundation.

        Returns
        -------
        cost : float
            Wind turbine monopile foundation cost [USD]
        """

        return self.cost

# -------------------------------------------------------------------------

def example():

    # simple test of module
    
    fdn = foundation()

    ratedPower = 5000.0
    seaDepth = 20.0
    monopileMass = 763000.00
    transitionMass = 415000.00
    
    fdn.compute(ratedPower, seaDepth, monopileMass, transitionMass, 1)     

if __name__ == "__main__":

    example()