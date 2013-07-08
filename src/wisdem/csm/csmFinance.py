"""
csmFin.py

Created by George Scott on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from math import *
from csmPPI import *

#----------------------------------------------------------------------

class csmFinance(object):
    """
    This class is a simplified model used to determine the cost of energy and levelized cost of energy for a wind plant based on the NREL Cost and Scaling Model.   
    """

    def __init__(self):
        """
           Initialize properties for csmFinance
        """
        # initialize variables
        self.ratedpwr = 5000
        self.tcc = 0.0
        self.om = 0.0
        self.llc = 0.0
        self.lrc = 0.0
        self.bos = 0.0
        self.aep = 0.0
        self.fixedChargeRate = 0.12
        self.constructionFinancingRate = 0.03
        self.taxRate = 0.4
        self.discountRate = 0.08
        self.constructionTime = 1
        self.projLifetime = 20
        self.turbineNum = 50

        # initialize output variables
        self.COE = 0.0
        self.LCOE = 0.0
  
        
    def compute(self, ratedpwr, tcc, om, llc, lrc, bos, aep, fcr, constructionrate, taxrate, discountrate, constructiontime, projlifetime, turbineNum):
        """
        Computes a wind plant cost of energy and levelized cost of energy using the NREL Cost and Scaling Model method.
        
        Parameters
        ----------
        ratedpwr : float
           Wind turbine rated power [kW]
        tcc : float
           Turbine Capital Costs (for entire plant) [USD]
        om : float
           Annual Operations and Maintenance Costs (for entire plant) (USD)
        llc : float
           Annual Land Lease Costs for wind plant [USD]
        lrc : float
           Levelized Replacement Costs (for entire plant) [USD]
        bos : float
           Balance of Station Costs (for entire plant) [USD]
        aep : float
           Annual energy production (for entire plant) [kWh]
        fcr : float
           Fixed charge rate
        constructionrate : float
           Construction financing rate
        taxrate : float
           Project tax rate
        discountrate : float
           Project discount rate
        constructiontime : float
           Time for construction of plant [years]
        projlifetime : float
           Project lifetime [years]
        turbineNum : float
           Number of turbines in plant
        
        """

        # initialize variables
        self.ratedpwr = ratedpwr
        self.tcc = tcc
        self.om = om
        self.llc = llc
        self.lrc = lrc
        self.bos = bos
        self.aep = aep
        self.fixedChargeRate = fcr
        self.constructionFinancingRate = constructionrate
        self.taxRate = taxrate
        self.discountRate = discountrate
        self.constructionTime = constructiontime
        self.projLifetime = projlifetime
        self.turbineNum = turbineNum

        #print [self.tcc, self.bos, self.aep, self.om, self.llc, self.lrc, self.taxRate, self.fixedChargeRate]
        #compute COE and LCOE values
        self.COE = ((self.tcc + self.bos)* (1+self.constructionFinancingRate) * self.fixedChargeRate / self.aep) + (self.om * (1-self.taxRate) + self.llc + self.lrc) / self.aep                        

        iccKW = ((self.bos + self.tcc)*(1+self.constructionFinancingRate)) / (self.ratedpwr * self.turbineNum)
        amortFactor = (1 + 0.5*((1+self.discountRate)**self.constructionTime-1)) * \
                      ((self.discountRate)/(1-(1+self.discountRate)**(-1.0*self.projLifetime)))
        capFact = self.aep / (8760 * self.ratedpwr * self.turbineNum)
        self.LCOE = (iccKW*amortFactor) / (8760*capFact) + (self.om/self.aep)

    def getCOE(self):
        """
        Returns the wind plant cost of energy
        
        Returns
        -------
        COE : float
           Wind Plant Cost of Energy [$/kWh]
        """

        return self.COE

    def getLCOE(self):
        """
        Returns the wind plant levelized cost of energy
        
        Returns
        -------
        LCOE : float
           Wind Plant Levelized Cost of Energy [$/kWh]
        """
        return self.LCOE
        
# --------------------------------------------

def example():
  
    # simple test of module
    
    lcoe = csmFinance()

    ratedpwr = 5000.0
    tcc = 80291265.250
    om = 1888344.846
    llc = 291344.633
    lrc = 880488.338
    bos = 29893055.879
    aep = 245869523.851
    fcr = 0.12
    constructionrate = 0.03
    taxrate = 0.4
    discountrate = 0.07
    constructiontime = 1
    projlifetime = 20
    turbineNum = 50
    
    print "Onshore"
    lcoe.compute(ratedpwr, tcc, om, llc, lrc, bos, aep, \
         fcr, constructionrate, taxrate, discountrate, constructiontime, projlifetime, turbineNum)
    print "LCOE %6.6f" % (lcoe.getLCOE())
    print "COE %6.6f" % (lcoe.getCOE())
    print

    tcc = 89736230.413
    om = 5267299.667
    llc = 291344.633
    lrc = 1365725.806
    bos = 97150303.809 + 12236758.693

    print "Offshore"
    lcoe.compute(ratedpwr, tcc, om, llc, lrc, bos, aep, \
         fcr, constructionrate, taxrate, discountrate, constructiontime, projlifetime, turbineNum)
    print "LCOE %6.6f" % (lcoe.getLCOE())
    print "COE %6.6f" % (lcoe.getCOE())


if __name__ == "__main__":

    example()