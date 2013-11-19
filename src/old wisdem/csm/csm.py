# csm integrating model

from csmTurbine import csmTurbine
from csmAEP import csmAEP
from csmPowerCurve import csmPowerCurve
from csmDriveEfficiency import DrivetrainEfficiencyModel, csmDriveEfficiency
from csmBOS import csmBOS
from csmOM import csmOM
from csmFinance import csmFinance

from math import *
import numpy as np

class csm(object):

    """
    Integrating class for all the NREL Cost and Scaling Model
    """

    def __init__(self, drivetrainDesign=1):
        
        self.turb = csmTurbine()
        self.aep = csmAEP()
        self.powerCurve = csmPowerCurve()
        self.drivetrain = csmDriveEfficiency(drivetrainDesign)
        self.bos = csmBOS()
        self.om = csmOM()
        self.fin = csmFinance()

    def compute(self, hubHeight, ratedPower, maxTipSpd, rotorDiam, dtDesign, nblades, \
                       maxEfficiency, ratedWindSpd, altitude, thrustCoeff, seaDepth, crane, advancedBlade, \
                       advancedBedplate, year, month, maxCp, maxTipSpdRatio, cutInWS, cutOutWS, \
                       airDensity,shearExp,ws50m,weibullK, \
                       soilingLosses, arrayLosses, availability, \
                       fcr, constructionrate, taxrate, discountrate, \
                       constructiontime, projlifetime, turbineNum):
      
        self.turb.compute(hubHeight, ratedPower, maxTipSpd, rotorDiam, dtDesign, nblades, \
                         maxEfficiency, ratedWindSpd, altitude, thrustCoeff, seaDepth, crane, advancedBlade, \
                         advancedBedplate, year, month)
        
        self.powerCurve.compute(self.drivetrain, hubHeight, ratedPower,maxTipSpd,rotorDiam,  \
                   maxCp, maxTipSpdRatio, cutInWS, cutOutWS, \
                   altitude, airDensity)
        
        self.powercurve = np.array(self.powerCurve.powerCurve)
  
        self.aep.compute(self.powercurve, ratedPower, hubHeight, shearExp,ws50m,weibullK, \
                  soilingLosses, arrayLosses, availability)
        
        self.bos.compute(seaDepth,ratedPower,hubHeight,rotorDiam,self.turb.cost, year, month)
        
        self.om.compute(self.aep.aep, seaDepth, ratedPower, year, month)
        
        self.fin.compute(ratedPower, self.turb.cost, self.om.cost, self.om.llc, self.om.lrc, \
                 self.bos.cost, self.aep.aep, fcr, constructionrate, taxrate, discountrate, \
                 constructiontime, projlifetime, turbineNum)


def example():

    #Default Cost and Scaling Model inputs for 5 MW turbine (onshore)    
    hubHeight=90.0
    ratedPower=5000.0
    maxTipSpd=80.0
    rotorDiam=126.0
    dtDesign=1
    nblades = 3
    maxEfficiency=0.90201
    ratedWindSpd = 11.5064
    altitude=0.0
    thrustCoeff=0.50
    seaDepth=0.0
    crane=True
    advancedBlade = False
    advancedBedplate = 1
    year = 2009
    month = 12
    maxCp=0.482
    maxTipSpdRatio = 7.55
    cutInWS = 4.0
    cutOutWS = 25.0
    altitude=0.0
    airDensity = 0.0
    shearExp=0.143
    ws50m=8.25
    weibullK=2.4
    soilingLosses = 0.0
    arrayLosses = 0.10
    availability = 0.95
    fcr = 0.12
    constructionrate = 0.03
    taxrate = 0.4
    discountrate = 0.08
    constructiontime = 1
    projlifetime = 20
    turbineNum = 50
    
    csmtest = csm(dtDesign)
    csmtest.compute(hubHeight, ratedPower, maxTipSpd, rotorDiam, dtDesign, nblades, \
                       maxEfficiency, ratedWindSpd, altitude, thrustCoeff, seaDepth, crane, advancedBlade, \
                       advancedBedplate, year, month, maxCp, maxTipSpdRatio, cutInWS, cutOutWS, \
                       airDensity,shearExp,ws50m,weibullK, \
                       soilingLosses, arrayLosses, availability, \
                       fcr, constructionrate, taxrate, discountrate, \
                       constructiontime, projlifetime, turbineNum)
                       
    print "LCOE: {0}".format(csmtest.fin.LCOE)
    print "COE: {0}".format(csmtest.fin.COE)

if __name__ == "__main__":

    example()