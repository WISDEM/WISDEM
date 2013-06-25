# 1 ---------

from sunderpact.src.hubsystem import HubSystem
from sunderpact.src.nacelle import NacelleSystem

# 1 ---------
# 2 ---------

# simple test of hubsystem sunderpact model

# NREL 5 MW turbine
BladeMass = 17740.0 # kg
RotorDiam = 126.0 # m
BladeNum  = 3
hubDiam   = 0.0 # m
RootMoment= 0.0 # Nm
AirDensity= 1.225 # kg/(m^3)
Solidity  = 0.0517 
RatedWindSpeed = 11.05 # m/s

# 2 ----------
hub = HubSystem(BladeMass, RotorDiam, BladeNum, hubDiam, RatedWindSpeed, RootMoment, AirDensity, Solidity)

# 3 ----------
# 4 ----------

print "NREL 5 MW turbine test"
print "Hub Components"
print '  hub         {0:8.1f} kg'.format(hub.hub.mass)  # 31644.47
print '  pitch mech  {0:8.1f} kg'.format(hub.pitchSystem.mass) # 17003.98
print '  nose cone   {0:8.1f} kg'.format(hub.spinner.mass) # 1810.50
print 'HUB TOTAL     {0:8.1f} kg'.format(hub.mass) # 50458.95
print 'cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(hub.cm[0], hub.cm[1], hub.cm[2])
print 'I {0:6.1f} {1:6.1f} {2:6.1f}'.format(hub.I[0], hub.I[1], hub.I[2])

# 4 ---------
# 5 ---------

# simple test of nacelle sunderpact model

# NREL 5 MW Rotor Variables
RotorDiam = 126.0 # m
RotorSpeed = 12.13 # m/s
RotorTorque = 4365248.74 # Nm
RotorThrust = 500930.84 # N
RotorMass = 142585.75 # kg

# NREL 5 MW Drivetrain variables
iDsgn = 1 # geared 3-stage Gearbox with induction generator machine
MachineRating = 5000.0 # kW
GearRatio = 97.0 # 97:1 as listed in the 5 MW reference document
GearConfig = 'eep' # epicyclic-epicyclic-parallel
Bevel = 0 # no bevel stage
crane = True # onboard crane present

# NREL 5 MW Tower Variables
TowerTopDiam = 3.78 # m

# 5 ---------
# 6 ---------

nace = NacelleSystem(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, \
               MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)
               
# 6 ---------
# 7 ---------

print '----- NREL 5 MW Turbine -----'
print 'Nacelle system model results'
print 'Low speed shaft %8.1f kg' % (nace.lss.mass)
print 'Main bearings   %8.1f kg' % (nace.mbg.mass)
print 'Gearbox         %8.1f kg' % (nace.gear.mass)
print 'High speed shaft & brakes  %8.1f kg' % (nace.hss.mass)
print 'Generator       %8.1f kg' % (nace.gen.mass)
print 'Variable speed electronics %8.1f kg' % (nace.vspdEtronicsMass)
print 'Overall mainframe %8.1f kg' % (nace.mainframeMass)
print '     Bedplate     %8.1f kg' % (nace.bpl.mass)
print 'electrical connections  %8.1f kg' % (nace.econnectionsMass)
print 'HVAC system     %8.1f kg' % (nace.hydrCoolingMass )
print 'Nacelle cover:   %8.1f kg' % (nace.nacelleCovMass)
print 'Yaw system      %8.1f kg' % (nace.yaw.mass)
print 'Overall nacelle:  %8.1f kg cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.mass, nace.cm[0], nace.cm[1], nace.cm[2], nace.I[0], nace.I[1], nace.I[2]  )

# 7 ---------