"""
towercost.py

Created by George Scott on 2012-08-01.
Modified by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from config import *
from common import ComponentCost
from zope.interface import implements

#-------------------------------------------------------------------------------        

class TowerCost():  # TODO: advanced tower not included here
    implements(ComponentCost)

    def __init__(self, towerMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine tower component.       
          
          Parameters
          ----------
          towerMass : float
            mass [kg] of the wind turbine tower
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month        
        '''
        
        self.update_cost(towerMass, curr_yr, curr_mon)
    
    def update_cost(self, towerMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine tower component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          towerMass : float
            mass [kg] of the wind turbine tower
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month         
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        twrCostEscalator  = ppi.compute('IPPI_TWR')
        
        twrCostCoeff      = 1.5 # $/kg    
        
        self.towerCost2002 = towerMass * twrCostCoeff               
        self.cost = self.towerCost2002 * twrCostEscalator
        
#-------------------------------------------------------------------------------       

def example():

    # simple test of module
    
    ref_yr  = 2002
    ref_mon =    9
    curr_yr = 2009
    curr_mon =  12
    
    ppi.ref_yr   = ref_yr
    ppi.ref_mon  = ref_mon

    towerMass = 434559.0
    tower = TowerCost(towerMass, curr_yr, curr_mon)
    
    print "Tower cost is ${0:.2f} USD".format(tower.cost) # $987180.30

if __name__ == "__main__":  

    example()