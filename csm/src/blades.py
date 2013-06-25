"""
bladeCost.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

import sys

from csmPPI import *
from config import *

#-------------------------------------------------------------------------------

class BladeCost(object):
    ''' bladeCost class
    
          This class provides a representation of a wind turbine blade.
            
    '''
    
    def __init__(self):
        """
        Initialize properties for blade cost model
           
        Parameters
        ----------
        cost : float
          individual wind turbine blade cost
        """        
        self.cost = 0.0

        pass
        
    def compute(self, mass,diam = 126.0, advanced=False,curr_yr=2009,curr_mon=9,verbose=0):
        """
        Computes wind turbine blade cost based on mass
           
        Parameters
        ----------
        mass : float
          individual wind turbine blade mass [kg]
        diam : float
          rotor diameter [m] of wind turbine - currently unused
        advanced : bool
          boolean for advanced (using carbon) or basline (all fiberglass) blade
        curr_yr : int
          project start year
        curr_mon : int
          project start month
        """ 
        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon

        # initialize inputs to execute
        self.advanced = advanced # 0 = baseline, 1 = advanced
        self.mass = mass
        self.diam = diam
         
        ppi_labor  = ppi.compute('IPPI_BLL')

        if (self.advanced == True):
            ppi.ref_yr = 2003
            ppi_mat   = ppi.compute('IPPI_BLA')
            slope   = 13.0 #14.0 from model
            intercept     = 5813.9
        else:
            ppi_mat   = ppi.compute('IPPI_BLD')
            slope   = 8.0
            intercept     = 21465.0
            
        laborCoeff    = 2.7445         # todo: ignoring labor impacts for now
        laborExp      = 2.5025
        
        self.bladeCostCurrent = (slope*self.mass + intercept)*ppi_mat
        
        self.cost = self.bladeCostCurrent
        
        if (verbose > 0):
            print '  blades cost      %6.1f K$  ' % (self.getCost())
    
    def getCost(self):
        """ 
        Provides the turbine blade capital costs for a single blade.

        Returns
        -------
        cost : float
            Single blade cost [USD]
        """

        return self.cost

        
#-------------------------------------------------------------------------------        

def example():
	
    # simple test of module
    blade = bladeCost()

    ref_yr  = 2002
    ref_mon =    9
    curr_yr = 2009
    curr_mon =  12
    
    ppi.ref_yr   = ref_yr
    ppi.ref_mon  = ref_mon
    ppi.curr_yr  = curr_yr
    ppi.curr_mon = curr_mon
    
    # Test 1
    bladeMass = 25614.38
    rotorDiam = 126.0
    advanced = False

    print "Conventional blade:"
    blade.compute(bladeMass,rotorDiam,advanced,curr_yr, curr_mon, 1)
    
    # Test 2
    bladeMass = 17650.67  # inline with the windpact estimates
    advanced = True
    
    print "Advanced blade: "
    blade.compute(bladeMass,rotorDiam,advanced,curr_yr, curr_mon, 1)

if __name__ == "__main__":  #TODO - update based on changes to csm Turbine

    example()
    
