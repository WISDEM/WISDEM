""" 
csmPowerCurve.py
    
Created by George Scott 2012
Modified  by Katherine Dykes 2012
Copyright (c)  NREL. All rights reserved.

"""

from csmPPI import *
from csmFoundation import csmFoundation

class csmBOS:

    """ NREL Cost and Scaling Balance of Station cost model. """
 
    def __init__(self):
        """
           Initialize properties for balance of plant costs including default inputs
        """      
        # initialize foundation
        self.fdn = csmFoundation()

        # initialize model input variables
        self.iDepth = 1
        self.machineRating = 5000.0
        self.hubHt = 90.0
        self.rtrDiam = 126
        self.tcc = 5508624.347
        self.ppi = PPI(2002, 9, 2009, 13)

        # initialize model output variables
        self.cost = 0
        
        self.foundations    = 0.0 # 80
        self.transportation = 0.0 # 60
        self.roadsCivil     = 0.0 #128
        self.portStaging    = 0.0 #128
        self.installation   = 0.0 # 82
        self.electrical     = 0.0 #230
        self.engPermits     = 0.0 # 36
        self.pai   = 0
        self.scour = 0
        
        self.lPrmtsCostCoeff1 = 9.94E-04 
        self.lPrmtsCostCoeff2 = 20.31 
        self.oPrmtsCostFactor = 37.0 # $/kW (2003)
        self.scourCostFactor =  55.0 # $/kW (2003)
        self.ptstgCostFactor =  20.0 # $/kW (2003)
        self.ossElCostFactor = 260.0 # $/kW (2003) shallow
        self.ostElCostFactor = 290.0 # $/kW (2003) transitional
        self.ostSTransFactor  =  25.0 # $/kW (2003)
        self.ostTTransFactor  =  77.0 # $/kW (2003)
        self.osInstallFactor  = 100.0 # $/kW (2003) shallow & trans
        self.paiCost         = 60000.0 # per turbine
        
        self.suretyBRate     = 0.03  # 3% of ICC
        self.suretyBond      = 0.0
        pass

        
    def compute(self,seaDepth,machineRating,hubHt,rtrDiam,tcc, year, month,verbose=0.0):
        """
        Computes the balance of station costs for a wind plant using the NREL Cost and Scaling Model method.
        
        Parameters
        ----------
        seaDepth : float
          sea depth [m] which is 0.0 or negative for an onshore project
        machineRating : float
          machine rating [kW] for the wind turbine at the site
        hubHt : float
          hub height [m] of the wind turbines at the site
        rtrDiam : float
          rotor diameter [m] of the wind turbines at the site
        tcc : float
          turbine capital costs [USD] for a single wind turbine in the project
        year : int
          project start year [year]
        month : int
          project start month [month]
        
        """      
        # reset variable values
        self.foundations    = 0.0 # 80
        self.transportation = 0.0 # 60
        self.roadsCivil     = 0.0 #128
        self.portStaging    = 0.0 #128
        self.installation   = 0.0 # 82
        self.electrical     = 0.0 #230
        self.engPermits     = 0.0 # 36
        self.pai   = 0
        self.scour = 0

        #set variables
        if seaDepth == 0:            # type of plant # 1: Land, 2: < 30m, 3: < 60m, 4: >= 60m
            self.iDepth = 1
        elif seaDepth < 30:
            self.iDepth = 2
        elif seaDepth < 60:
            self.iDepth = 3
        else:
            self.iDepth = 4

        self.machineRating = machineRating
        self.hubHt = hubHt
        self.rtrDiam = rtrDiam
        self.tcc = tcc 

        # initialize self.ppi index calculator
        if self.iDepth == 1:
            ref_yr  = 2002                   
            ref_mon =    9
        else:
            ref_yr = 2003
            ref_mon = 9
        self.ppi = PPI(ref_yr, ref_mon, year, month)
        
        # foundation cost calculations
        self.fdn.compute(machineRating, hubHt, rtrDiam, seaDepth, year, month)
        
        # cost calculations
        tpC1  =0.00001581
        tpC2  =-0.0375
        tpInt =54.7
        tFact = tpC1*self.machineRating*self.machineRating + tpC2*self.machineRating + tpInt   

        if (self.iDepth == 1):
            self.engPermits  = (self.lPrmtsCostCoeff1 * self.machineRating * self.machineRating) + \
                               (self.lPrmtsCostCoeff2 * self.machineRating)
            self.ppi.ref_mon = 3
            self.engPermits *= self.ppi.compute('IPPI_LPM') 
            self.ppi.ref_mon = 9
            
            elC1  = 3.49E-06
            elC2  = -0.0221
            elInt = 109.7
            eFact = elC1*self.machineRating*self.machineRating + elC2*self.machineRating + elInt
            self.electrical = self.machineRating * eFact * self.ppi.compute('IPPI_LEL')
            
            rcC1  = 2.17E-06
            rcC2  = -0.0145
            rcInt =69.54
            rFact = rcC1*self.machineRating*self.machineRating + rcC2*self.machineRating + rcInt
            self.roadsCivil = self.machineRating * rFact * self.ppi.compute('IPPI_RDC')
             
            iCoeff = 1.965
            iExp   = 1.1736
            self.installation = iCoeff * ((self.hubHt*self.rtrDiam)**iExp) * self.ppi.compute('IPPI_LAI')
          
            self.transportation = self.machineRating * tFact * self.ppi.compute('IPPI_TPT')
             
            pass
        elif (self.iDepth == 2):  # offshore shallow
            self.pai            = self.paiCost                          * self.ppi.compute('IPPI_PAE')
            self.portStaging    = self.ptstgCostFactor  * self.machineRating * self.ppi.compute('IPPI_STP')
            self.engPermits     = self.oPrmtsCostFactor * self.machineRating * self.ppi.compute('IPPI_OPM')
            self.scour          = self.scourCostFactor  * self.machineRating * self.ppi.compute('IPPI_STP')
            self.installation   = self.osInstallFactor  * self.machineRating * self.ppi.compute('IPPI_OAI')            
            self.electrical     = self.ossElCostFactor  * self.machineRating * self.ppi.compute('IPPI_OEL')
            self.transportation = self.machineRating * tFact * self.ppi.compute('IPPI_TPT')
            pass 
        elif (self.iDepth == 3):  # offshore transitional depth
            self.installation   = self.osInstallFactor  * self.machineRating * self.ppi.compute('IPPI_OAI')
            self.pai            = self.paiCost                          * self.ppi.compute('IPPI_PAE')
            self.electrical     = self.ostElCostFactor  * self.machineRating * self.ppi.compute('IPPI_OEL')
            self.portStaging    = self.ptstgCostFactor  * self.machineRating * self.ppi.compute('IPPI_STP')
            self.engPermits     = self.oPrmtsCostFactor * self.machineRating * self.ppi.compute('IPPI_OPM')
            self.scour          = self.scourCostFactor  * self.machineRating * self.ppi.compute('IPPI_STP')
            turbTrans           = self.ostTTransFactor  * self.machineRating * self.ppi.compute('IPPI_TPT') 
            supportTrans        = self.ostSTransFactor  * self.machineRating * self.ppi.compute('IPPI_OAI') 
            self.transportation = turbTrans + supportTrans
            
        elif (self.iDepth == 4):  # offshore deep
            print "\ncsmBOS: Add costCat 4 code\n\n"
            pass
        
        self.cost = self.fdn.getCost() + \
                    self.transportation + \
                    self.roadsCivil     + \
                    self.portStaging    + \
                    self.installation   + \
                    self.electrical     + \
                    self.engPermits     + \
                    self.pai            + \
                    self.scour       

        if (self.iDepth > 1):
            self.suretyBond = self.suretyBRate * (self.tcc / 1.10 + self.cost)
            self.cost += self.suretyBond

        if (verbose > 0):
            self.dump()

    def getCost(self):
        """ 
        Provides the overall balance of station costs for the plant.

        Returns
        -------
        cost : float
            Balance of plant costs [USD]
        """
        
        return self.cost

    def getDetailedCosts(self):
        """ 
        Provides the detailed balance of station costs for the plant.

        Returns
        -------
        detailedCosts : array_like of float
            Balance of plant costs [USD] broken down into components: foundation, transporation, roads & civil,
            ports and staging, installation and assembly, electrical, permits, miscenalleous, scour, surety bond
        """
        self.detailedCosts = [self.fdn.getCost(), self.transportation, self.roadsCivil, self.portStaging, self.installation, \
                self.electrical, self.engPermits, self.pai, self.scour, self.suretyBond] 
        
        return self.detailedCosts 
 
    def dump(self):
        print
        print "BOS: "
        print "  foundation     %8.3f $" % self.fdn.getCost()
        print "  transportation %8.3f $" % self.transportation 
        print "  roadsCivil     %8.3f $" % self.roadsCivil     
        print "  portStaging    %8.3f $" % self.portStaging    
        print "  installation   %8.3f $" % self.installation   
        print "  electrical     %8.3f $" % self.electrical     
        print "  engPermits     %8.3f $" % self.engPermits     
        print "  pai            %8.3f $" % self.pai            
        print "  scour          %8.3f $" % self.scour       
        print "TOTAL            %8.3f $" % self.cost       
        print "  surety bond    %8.3f $" % self.suretyBond       
        print
    
#------------------------------------------------------------------

def example():
	
    # simple test of module
    
    bos = csmBOS()
    
    seaDepth = 0.0
    machineRating = 5000.0
    hubHt = 90.0
    rtrDiam = 126.0
    tcc = 5508624.347
    year = 2009
    month = 12
    
    bos.compute(seaDepth,machineRating,hubHt,rtrDiam,tcc, year, month, 1)

    print 'BOS cost onshore   %9.3f '          % bos.getCost()
    
    seaDepth = 20.0
    tcc = 6087803.6
    
    bos.compute(seaDepth,machineRating,hubHt,rtrDiam,tcc, year, month, 1)
    
    print 'BOS cost offshore   %9.3f '          % bos.getCost()    
    
if __name__ == "__main__":

    example()

#------------------------------------------------------------------