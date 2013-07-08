"""
csmBlades.py

Created by George Scott on 2012-08-01.
Modified by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

import sys

from csmPPI import *
from config import *

#-------------------------------------------------------------------------------

class csmBlades(object):
    ''' csmBlades class
    
          This class provides a representation of a wind turbine blade.
            
    '''
    
    def __init__(self):
        """
        Initialize the parameters for the wind turbine blade
        
        Parameters
        ----------
        mass : float
          Individual wind turbine blade mass [kg]
        cost : float
          individual wind turbine blade cost [USD]
        """

        self.mass = 0.0
        self.cost = 0.0

        pass
        
    def compute(self, diam=126.0,advanced=False,curr_yr=2009,curr_mon=9,verbose=0):
        """
        Compute mass and cost for a single wind turbine blade by calling computeMass and computeCost
        
        Parameters
        ----------
        diam : float
          rotor diameter [m] of the turbine
        advanced : bool
          advanced blade configuration boolean
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """
        self.computeMass(diam,advanced)
        
        self.computeCost(diam,advanced,curr_yr,curr_mon)

        if (verbose > 0):
            print '  blades        %6.1f K$  %8.1f kg' % (self.getCost()   , self.getMass())
    
    def computeMass(self,diam,advanced=False):
        """
        Compute mass for a single wind turbine blade using NREL cost and scaling model
        
        Parameters
        ----------
        diam : float
          rotor diameter [m] of the turbine
        advanced : bool
          advanced blade configuration boolean
        """

        # initialize inputs to execute
        self.advanced = advanced # 0 = baseline, 1 = advanced
        self.diam = diam
         
        if (self.advanced == True):
            massCoeff = 0.4948
            massExp   = 2.5300
        else:
            massCoeff = 0.1452 
            massExp   = 2.9158
        
        self.mass = (massCoeff*(self.diam/2.0000)**massExp)
      
    def computeCost(self,diam,advanced=False,curr_yr=2009,curr_mon=9):
        """
        Compute cost for a single wind turbine blade using NREL cost and scaling model
        
        Parameters
        ----------
        diam : float
          rotor diameter [m] of the turbine
        advanced : bool
          advanced blade configuration boolean
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """
        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon

        # initialize inputs to execute
        self.advanced = advanced # 0 = baseline, 1 = advanced
        self.diam = diam
         
        ppi_labor  = ppi.compute('IPPI_BLL')

        if (self.advanced == True):
            ppi.ref_yr = 2003
            ppi_mat   = ppi.compute('IPPI_BLA')
            slopeR3   = 0.4019
            intR3     = -21051.0000
        else:
            ppi_mat   = ppi.compute('IPPI_BLD')
            slopeR3   = 0.4019
            intR3     = -955.2400
            
        laborCoeff    = 2.7445
        laborExp      = 2.5025
        
        self.bladeCostCurrent = ( (slopeR3*(self.diam/2.0000)**3.0000 + (intR3))*ppi_mat + \
                                  (laborCoeff*(self.diam/2.0000)**laborExp)*ppi_labor    ) / (1.0000-0.2800)
        self.cost = self.bladeCostCurrent

    def getMass(self):
        """ 
        Provides the mass for the wind turbine blade.

        Returns
        -------
        mass : float
            Wind turbine blade mass [kg]
        """

        return self.mass
        
    def getCost(self):
        """ 
        Provides the cost for the wind turbine blade.

        Returns
        -------
        cost : float
            Wind turbine blade cost [USD]
        """

        return self.cost
        
#-------------------------------------------------------------------------------        

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
    
    blades = csmBlades()
    print "Conventional blade design:"
    blades.compute(126,False,curr_yr, curr_mon, 1)
    print "Advanced blade design:"
    blades.compute(126,True,curr_yr, curr_mon, 1)

if __name__ == "__main__":  #TODO - update based on changes to csm Turbine

    example()
