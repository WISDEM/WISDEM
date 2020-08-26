# 1 ---------

from NREL_CSM.config import *
from NREL_CSM.csm import csm

# 1 ---------
# 2 ---------

#Default Cost and Scaling Model inputs for 5 MW turbine (onshore)    
ppi.curr_yr  = 2009
ppi.curr_mon = 12

hubHeight=90.0
ratedPower=5000.0
maxTipSpd=80.0
rotorDiam=126.0
dtDesign=1
nblades = 3
altitude=0.0
thrustCoeff=0.50
seaDepth=20.0
crane=True
advancedBlade = True
advancedBedplate = 0
advancedTower = False
year = 2009
month = 12
maxCp=0.488
maxTipSpdRatio = 7.525
cutInWS = 3.0
cutOutWS = 25.0
airDensity = 0.0
shearExp=0.1
ws50m=8.02
weibullK=2.15
soilingLosses = 0.0
arrayLosses = 0.10
availability = 0.941
fcr = 0.12
taxrate = 0.4
discountrate = 0.07
constructiontime = 1
projlifetime = 20
turbineNum = 100

# 2 ----------
# 3 ----------

csmtest = csm(dtDesign)
csmtest.compute(hubHeight, ratedPower, maxTipSpd, rotorDiam, dtDesign, nblades, altitude, thrustCoeff, seaDepth, crane, advancedBlade,  advancedBedplate, advancedTower, year, month, maxCp, maxTipSpdRatio, cutInWS, cutOutWS, \
                  airDensity, shearExp, ws50m, weibullK, soilingLosses, arrayLosses, availability, fcr, taxrate, discountrate, constructiontime, projlifetime, turbineNum)

# 3 ----------
# 4 ----------

print "LCOE %9.8f" % (csmtest.fin.LCOE)
print "COE %9.8f"%(csmtest.fin.COE)
print "AEP %9.5f"%(csmtest.aep.aep / 1000.0)
print "BOS %9.5f"%(csmtest.bos.cost / 1000.0)
print "TCC %9.5f"%(csmtest.turb.cost / 1000.0)
print "OM %9.5f"%(csmtest.om.cost / 1000.0)
print "LRC %9.5f"%(csmtest.om.lrc / 1000.0)
print "LLC %9.5f"%(csmtest.om.llc / 1000.0)

# 4 ---------