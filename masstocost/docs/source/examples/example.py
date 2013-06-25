# 1 ---------

from masstocost.src.turbinecost import TurbineCost
from masstocost.src.config import *

# simple example of the turbine cost model

ref_yr  = 2002
ref_mon =    9
curr_yr = 2009
curr_mon =  12

ppi.ref_yr   = ref_yr
ppi.ref_mon  = ref_mon

# 1 ---------
# 2 ---------

# NREL 5 MW turbine component masses based on Sunderland model approach
bladeMass = 17650.67  
hubMass = 31644.5
pitchSystemMass = 17004.0
spinnerMass = 1810.5

lssMass = 31257.3
bearingsMass = 9731.41
gearboxMass = 30237.60
hssMass = 1492.45
generatorMass = 16699.85
bedplateMass = 93090.6
yawSystemMass = 11878.24

towerMass = 434559.0

# 2 ---------
# 3 ---------

# Additional non-mass cost model input variables
bladeNum = 3
advanced = True # use advanced blade mass-cost curve
machineRating = 5000.0
iDesign = 1 # conventional 3-stage geared drivetrain system
crane = True # onboard crane is present
offshore = True # turbine is for an offshore application

# 3 ---------
# 4 --------- 

turbine = TurbineCost(bladeMass, bladeNum, hubMass, pitchSystemMass, spinnerMass, towerMass, lssMass, \
                     bearingsMass, gearboxMass, hssMass, generatorMass, bedplateMass, yawSystemMass, \
                     machineRating, iDesign, offshore, curr_yr, curr_mon, crane, advanced)

# 4 ----------
# 5 ----------

print "Turbine cost is ${0:.2f} USD".format(turbine.cost) 
print
print "Overall rotor cost with 3 advanced blades is ${0:.2f} USD".format(turbine.rotor.cost)
print "Advanced blade cost is ${0:.2f} USD".format(turbine.rotor.blade.cost)
print "Cost of 3 blades is ${0:.2f} USD".format(turbine.rotor.blade.cost * 3)
print "Hub cost is ${0:.2f} USD".format(turbine.rotor.hub.cost)  
print "Pitch cost is ${0:.2f} USD".format(turbine.rotor.pitch.cost) 
print "Spinner cost is ${0:.2f} USD".format(turbine.rotor.spinner.cost) 
print
print "Overall nacelle cost is ${0:.2f} USD".format(turbine.nacelle.cost)
print "LSS cost is ${0:.2f} USD".format(turbine.nacelle.lss.cost)
print "Main bearings cost is ${0:.2f} USD".format(turbine.nacelle.bearings.cost) 
print "Gearbox cost is ${0:.2f} USD".format(turbine.nacelle.gearbox.cost) 
print "HSS cost is ${0:.2f} USD".format(turbine.nacelle.hss.cost) 
print "Generator cost is ${0:.2f} USD".format(turbine.nacelle.generator.cost) 
print "Bedplate cost is ${0:.2f} USD".format(turbine.nacelle.bedplate.cost)
print "Yaw system cost is ${0:.2f} USD".format(turbine.nacelle.yawsystem.cost) 
print
print "Tower cost is ${0:.2f} USD".format(turbine.tower.cost)

# 5 ---------- 