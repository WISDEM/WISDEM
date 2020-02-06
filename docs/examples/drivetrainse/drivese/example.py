# 1 ---------

from math import pi
import numpy as np
from drivese.hub import HubSE
from drivese.drive import Drive4pt

# 1 ---------
# 2 ---------

# simple test of hub model

# NREL 5 MW turbine

hubS = HubSE()
hubS.rotor_diameter = 126.0 # m
hubS.blade_number  = 3
hubS.blade_root_diameter   = 3.542

hubS.L_rb = 0.5
hubS.gamma = 5.0
hubS.MB1_location = np.array([-0.5, 0.0, 0.0])
hubS.machine_rating = 5000.0

# 2 ----------
# 3 ----------

hubS.run()

# 3 ----------
# 4 ----------

print "Estimate of Hub Component Sizes for the NREL 5 MW Reference Turbine"
print "Hub Components"
print '  Hub: {0:8.1f} kg'.format(hubS.hub.mass)  # 31644.47
print '  Pitch system: {0:8.1f} kg'.format(hubS.pitchSystem.mass) # 17003.98
print '  Nose cone: {0:8.1f} kg'.format(hubS.spinner.mass) # 1810.50
print 'Hub system total: {0:8.1f} kg'.format(hubS.hub_system_mass) # 50458.95
print '    cm {0:6.2f} {1:6.2f} {2:6.2f} [m, m, m]'.format(hubS.hub_system_cm[0], hubS.hub_system_cm[1], hubS.hub_system_cm[2])
print '    I {0:6.1f} {1:6.1f} {2:6.1f} [kg*m^2, kg*m^2, kg*m^2]'.format(hubS.hub_system_I[0], hubS.hub_system_I[1], hubS.hub_system_I[2])
print

# 4 ---------
# 5 ---------

# test of drivetrain model

# NREL 5 MW Rotor Variables
nace = Drive4pt()
nace.rotor_diameter = 126.0 # m
nace.rotor_speed = 12.1 # #rpm m/s
nace.machine_rating = 5000.0
nace.DrivetrainEfficiency = 0.95
nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm
nace.rotor_thrust = 599610.0 # N
nace.rotor_mass = 0.0 #accounted for in F_z # kg
nace.rotor_speed = 12.1 #rpm
nace.rotor_bending_moment = -16665000.0 # Nm same as rotor_bending_moment_y
nace.rotor_bending_moment_x = 330770.0# Nm
nace.rotor_bending_moment_y = -16665000.0 # Nm
nace.rotor_bending_moment_z = 2896300.0 # Nm
nace.rotor_force_x = 599610.0 # N
nace.rotor_force_y = 186780.0 # N
nace.rotor_force_z = -842710.0 # N

# NREL 5 MW Drivetrain variables
nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
nace.machine_rating = 5000.0 # kW
nace.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
#nace.bevel = 0 # no bevel stage
nace.crane = True # onboard crane present
nace.shaft_angle = 5.0 #deg
nace.shaft_ratio = 0.10
nace.Np = [3,3,1]
nace.ratio_type = 'optimal'
nace.shaft_type = 'normal'
nace.uptower_transformer=False
nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
nace.carrier_mass = 8000.0 # estimated
nace.mb1Type = 'CARB'
nace.mb2Type = 'SRB'
nace.flange_length = 0.5 #m
nace.overhang = 5.0
nace.gearbox_cm = 0.1
nace.hss_length = 1.5
nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs
nace.blade_number=3
nace.cut_in=3. #cut-in m/s
nace.cut_out=25. #cut-out m/s
nace.Vrated=11.4 #rated windspeed m/s
nace.weibull_k = 2.2 # windepeed distribution shape parameter
nace.weibull_A = 9. # windspeed distribution scale parameter
nace.T_life=20. #design life in years
nace.IEC_Class_Letter = 'A'
nace.L_rb = 1.912 # length from hub center to main bearing, leave zero if unknown

# NREL 5 MW Tower Variables
nace.tower_top_diameter = 3.78 # m

# 5 ---------
# 6 ---------

nace.run()

# 6 ---------
# 7 ---------

print "Estimate of Nacelle Component Sizes for the NREL 5 MW Reference Turbine"
print 'Low speed shaft: {0:8.1f} kg'.format(nace.lowSpeedShaft.mass)
print 'Main bearings: {0:8.1f} kg'.format(nace.mainBearing.mass + nace.secondBearing.mass)
print 'Gearbox: {0:8.1f} kg'.format(nace.gearbox.mass)
print 'High speed shaft & brakes: {0:8.1f} kg'.format(nace.highSpeedSide.mass)
print 'Generator: {0:8.1f} kg'.format(nace.generator.mass)
print 'Variable speed electronics: {0:8.1f} kg'.format(nace.above_yaw_massAdder.vs_electronics_mass)
print 'Overall mainframe:{0:8.1f} kg'.format(nace.above_yaw_massAdder.mainframe_mass)
print '     Bedplate: {0:8.1f} kg'.format(nace.bedplate.mass)
print 'Electrical connections: {0:8.1f} kg'.format(nace.above_yaw_massAdder.electrical_mass)
print 'HVAC system: {0:8.1f} kg'.format(nace.above_yaw_massAdder.hvac_mass )
print 'Nacelle cover: {0:8.1f} kg'.format(nace.above_yaw_massAdder.cover_mass)
print 'Yaw system: {0:8.1f} kg'.format(nace.yawSystem.mass)
print 'Overall nacelle: {0:8.1f} kg'.format(nace.nacelle_mass, nace.nacelle_cm[0], nace.nacelle_cm[1], nace.nacelle_cm[2], nace.nacelle_I[0], nace.nacelle_I[1], nace.nacelle_I[2]  )
print '    cm {0:6.2f} {1:6.2f} {2:6.2f} [m, m, m]'.format(nace.nacelle_cm[0], nace.nacelle_cm[1], nace.nacelle_cm[2])
print '    I {0:6.1f} {1:6.1f} {2:6.1f} [kg*m^2, kg*m^2, kg*m^2]'.format(nace.nacelle_I[0], nace.nacelle_I[1], nace.nacelle_I[2])
# 7 ---------