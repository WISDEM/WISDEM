"""
rotorcosts.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from config import *
from common import ComponentCost
from zope.interface import implements

#-------------------------------------------------------------------------------

class BladeCost():
    implements(ComponentCost)

    def __init__(self, bladeMass, curr_yr, curr_mon, advanced = True):
        '''
          Initial computation of the costs for the wind turbine blade component.       
          
          Parameters
          ----------
          bladeMass : float
            blade mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          advanced : bool
            boolean for advanced (using carbon) or basline (all fiberglass) blade          
        '''
        
        self.update_cost(bladeMass, curr_yr, curr_mon, advanced = True)
    
    def update_cost(self, bladeMass, curr_yr, curr_mon, advanced = True):

        '''
          Computes the costs for the wind turbine blade component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          bladeMass : float
            blade mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          advanced : bool
            boolean for advanced (using carbon) or basline (all fiberglass) blade          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon
         
        ppi_labor  = ppi.compute('IPPI_BLL')

        if (advanced == True):
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
        
        self.cost = ((slope*bladeMass + intercept)*ppi_mat)

# -----------------------------------------------------------------------------------------------

class HubCost():
    implements(ComponentCost)

    def __init__(self, hubMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine hub component.       
          
          Parameters
          ----------
          hubMass : float
            hub mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(hubMass, curr_yr, curr_mon)
    
    def update_cost(self, hubMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine hub component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          hubMass : float
            hub mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        #calculate system costs
        ppi_labor  = ppi.compute('IPPI_BLL')
            
        laborCoeff    = 2.7445
        laborExp      = 2.5025    

        hubCost2002      = (hubMass * 4.25) # $/kg
        hubCostEscalator = ppi.compute('IPPI_HUB')
        self.cost = (hubCost2002 * hubCostEscalator )
        
#-------------------------------------------------------------------------------

class PitchCost():
    implements(ComponentCost)

    def __init__(self, pitchSystemMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine pitch system.       
          
          Parameters
          ----------
          pitchSystemMass : float
            pitch system mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(pitchSystemMass, curr_yr, curr_mon)
    
    def update_cost(self, pitchSystemMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine pitch system component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          pitchSystemMass : float
            pitch system mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        #calculate system costs
        ppi_labor  = ppi.compute('IPPI_BLL')
            
        laborCoeff    = 2.7445
        laborExp      = 2.5025    

        pitchSysCost2002     = 2.28 * (0.0808 * (pitchSystemMass ** 1.4985))            # new cost based on mass - x1.328 for housing proportion
        bearingCostEscalator = ppi.compute('IPPI_PMB')
        self.cost = (bearingCostEscalator * pitchSysCost2002)
        
#-------------------------------------------------------------------------------

class SpinnerCost():
    implements(ComponentCost)

    def __init__(self, spinnerMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine spinner component.       
          
          Parameters
          ----------
          spinnerMass : float
            spinner mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.update_cost(spinnerMass, curr_yr, curr_mon)
    
    def update_cost(self, spinnerMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine spinner component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          spinnerMass : float
            spinner mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''

        # assign input variables
        ppi.curr_yr   = curr_yr
        ppi.curr_mon   = curr_mon

        #calculate system costs
        ppi_labor  = ppi.compute('IPPI_BLL')
            
        laborCoeff    = 2.7445
        laborExp      = 2.5025    

        spinnerCostEscalator = ppi.compute('IPPI_NAC')
        self.cost = (spinnerCostEscalator * (5.57*spinnerMass))

#-------------------------------------------------------------------------------

class HubSystemCost():
    implements(ComponentCost)

    def __init__(self, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon):
        '''
          Initial computation of the costs for the wind turbine hub component.       
          
          Parameters
          ----------
          hubMass : float
            hub mass [kg]
          pitchSystemMass : float
            pitch system mass [kg]
          spinnerMass : float
            spinner mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.hub = HubCost(hubMass, curr_yr, curr_mon)
        self.pitch = PitchCost(pitchSystemMass, curr_yr, curr_mon)
        self.spinner = SpinnerCost(spinnerMass, curr_yr, curr_mon)
        
        self.update_cost(hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon)
    
    def update_cost(self, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon):

        '''
          Computes the costs for the wind turbine hub component.
          Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.       
          
          Parameters
          ----------
          hubMass : float
            hub mass [kg]
          pitchSystemMass : float
            pitch system mass [kg]
          spinnerMass : float
            spinner mass [kg]
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          
        '''
        
        self.hub.update_cost(hubMass, curr_yr, curr_mon)
        self.pitch.update_cost(pitchSystemMass, curr_yr, curr_mon)
        self.spinner.update_cost(spinnerMass, curr_yr, curr_mon)
        
        self.cost = self.hub.cost + self.pitch.cost + self.spinner.cost

#-------------------------------------------------------------------------------

class RotorCost():
    implements(ComponentCost)

    def __init__(self, bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon, advanced = True):
        '''
          Initial computation of the costs for the wind turbine hub component.       
          
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
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          advanced : bool
            boolean for advanced (using carbon) or basline (all fiberglass) blade           
        '''
        
        self.hub = HubCost(hubMass, curr_yr, curr_mon)
        self.pitch = PitchCost(pitchSystemMass, curr_yr, curr_mon)
        self.spinner = SpinnerCost(spinnerMass, curr_yr, curr_mon)
        self.blade = BladeCost(bladeMass, curr_yr, curr_mon, advanced)
        
        self.update_cost(bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon, advanced)
    
    def update_cost(self, bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon, advanced = True):

        '''
          Computes the costs for the wind turbine hub component.
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
          curr_yr : int
            Project start year
          curr_mon : int
            Project start month
          advanced : bool
            boolean for advanced (using carbon) or basline (all fiberglass) blade           
        '''
        
        self.hub.update_cost(hubMass, curr_yr, curr_mon)
        self.pitch.update_cost(pitchSystemMass, curr_yr, curr_mon)
        self.spinner.update_cost(spinnerMass, curr_yr, curr_mon)
        self.blade.update_cost(bladeMass, curr_yr, curr_mon, advanced)
        
        self.cost = self.blade.cost * bladeNum + self.hub.cost + self.pitch.cost + self.spinner.cost        

# ------------------------------------------------------------------------------------------------      

def example():
  
    # simple test of module

    ref_yr  = 2002
    ref_mon =    9
    curr_yr = 2009
    curr_mon =  12
    
    ppi.ref_yr   = ref_yr
    ppi.ref_mon  = ref_mon

    # NREL 5 MW turbine
    print "NREL 5 MW turbine test"

    # Blade Test 1
    bladeMass = 25614.38
    advanced = False

    blade = BladeCost(bladeMass, curr_yr, curr_mon, advanced)
    print "Conventional blade: ${0:.2f} USD".format(blade.cost)
  
    # Blade Test 2
    bladeMass = 17650.67  # inline with the windpact estimates
    advanced = True

    blade.update_cost(bladeMass,curr_yr, curr_mon, advanced)
    print "Advanced blade: ${0:.2f} USD".format(blade.cost)

    hubMass = 31644.5
    pitchSystemMass = 17004.0
    spinnerMass = 1810.5
    curr_yr = 2009
    curr_mon = 12

    hub = HubCost(hubMass, curr_yr, curr_mon)
    pitch = PitchCost(pitchSystemMass, curr_yr, curr_mon)
    spinner = SpinnerCost(spinnerMass, curr_yr, curr_mon)

    print "Hub cost is ${0:.2f} USD".format(hub.cost)   # $175513.50
    print "Pitch cost is ${0:.2f} USD".format(pitch.cost)  # $535075.0
    print "Spinner cost is ${0:.2f} USD".format(spinner.cost)  # $10509.00

    hubsystem = HubSystemCost(hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon)
    
    print "Hub system cost is ${0:.2f} USD".format(hubsystem.cost)
    
    bladeNum = 3
    rotor = RotorCost(bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, curr_yr, curr_mon, advanced)
    
    print "Overall rotor cost with 3 advanced blades is ${0:.2f} USD".format(rotor.cost)

if __name__ == "__main__":

    example()