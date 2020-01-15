# 1 ---------

# A simple test of turbine_costsse model
from turbine_costsse.turbine_costsse import Turbine_CostsSE

turbine = Turbine_CostsSE()

# 1 ---------
# 2 ---------

# NREL 5 MW turbine component masses based on Sunderland model approach

# Rotor
turbine.blade_mass = 17650.67  # inline with the windpact estimates
turbine.hub_mass = 31644.5
turbine.pitch_system_mass = 17004.0
turbine.spinner_mass = 1810.5

# Drivetrain and Nacelle
turbine.low_speed_shaft_mass = 31257.3
#bearingsMass = 9731.41
turbine.main_bearing_mass = 9731.41 / 2
turbine.second_bearing_mass = 9731.41 / 2
turbine.gearbox_mass = 30237.60
turbine.high_speed_side_mass = 1492.45
turbine.generator_mass = 16699.85
turbine.bedplate_mass = 93090.6
turbine.yaw_system_mass = 11878.24

# Tower
turbine.tower_mass = 434559.0

# 2 ---------
# 3 ---------

# Additional non-mass cost model input variables
turbine.machine_rating = 5000.0
turbine.advanced = True
turbine.blade_number = 3
turbine.drivetrain_design = 'geared'
turbine.crane = True
turbine.offshore = True

# Target year for analysis results
turbine.year = 2010
turbine.month =  12

# 3 ---------
# 4 --------- 

turbine.run()

# 4 ----------
# 5 ----------

print "The results for the NREL 5 MW Reference Turbine in an offshore 20 m water depth location are:"
print
print "Overall rotor cost with 3 advanced blades is ${0:.2f} USD".format(turbine.rotorCC.cost)
print "Blade cost is ${0:.2f} USD".format(turbine.rotorCC.bladeCC.cost)
print "Hub cost is ${0:.2f} USD".format(turbine.rotorCC.hubCC.cost)
print "Pitch system cost is ${0:.2f} USD".format(turbine.rotorCC.pitchSysCC.cost)
print "Spinner cost is ${0:.2f} USD".format(turbine.rotorCC.spinnerCC.cost)
print
print "Overall nacelle cost is ${0:.2f} USD".format(turbine.nacelleCC.cost)
print "LSS cost is ${0:.2f} USD".format(turbine.nacelleCC.lssCC.cost)
print "Main bearings cost is ${0:.2f} USD".format(turbine.nacelleCC.bearingsCC.cost)
print "Gearbox cost is ${0:.2f} USD".format(turbine.nacelleCC.gearboxCC.cost)
print "Hight speed side cost is ${0:.2f} USD".format(turbine.nacelleCC.hssCC.cost)
print "Generator cost is ${0:.2f} USD".format(turbine.nacelleCC.generatorCC.cost)
print "Bedplate cost is ${0:.2f} USD".format(turbine.nacelleCC.bedplateCC.cost)
print "Yaw system cost is ${0:.2f} USD".format(turbine.nacelleCC.yawSysCC.cost)
print
print "Tower cost is ${0:.2f} USD".format(turbine.towerCC.cost)
print
print "The overall turbine cost is ${0:.2f} USD".format(turbine.turbine_cost)
print

# 5 ---------- 
# 6 ----------

# A simple test of nrel_csm_tcc model
from turbine_costsse.nrel_csm_tcc import tcc_csm_assembly
import numpy as np

trb = tcc_csm_assembly()

# 6 ----------
# 7 ----------

# NREL 5 MW main parameters
trb.rotor_diameter = 126.0
trb.advanced_blade = True
trb.blade_number = 3
trb.hub_height = 90.0    
trb.machine_rating = 5000.0
trb.offshore = True
trb.drivetrain_design = 'geared'
trb.crane = True
trb.advanced_bedplate = 0
trb.advanced_tower = False

# 7 ----------
# 8 ----------

# Rotor force calculations for nacelle inputs
maxTipSpd = 80.0
maxEfficiency = 0.90201
ratedWindSpd = 11.5064
thrustCoeff = 0.50
airDensity = 1.225

ratedHubPower  = trb.machine_rating / maxEfficiency 
rotorSpeed     = (maxTipSpd/(0.5*trb.rotor_diameter)) * (60.0 / (2*np.pi))
trb.rotor_thrust  = airDensity * thrustCoeff * np.pi * trb.rotor_diameter**2 * (ratedWindSpd**2) / 8
trb.rotor_torque = ratedHubPower/(rotorSpeed*(np.pi/30))*1000

# 8 -----------

# Target year for analysis results
trb.year = 2009
trb.month = 12

# 8 -----------
# 9 -----------

trb.run()

# 9 -----------
# 10 ----------

print "The results for the NREL 5 MW Reference Turbine in an offshore 20 m water depth location are:"
print "Overall turbine mass is {0:.2f} kg".format(trb.turbine_mass)
print "Overall turbine cost is ${0:.2f} USD".format(trb.turbine_cost)

# 10 ----------
# 11 ----------
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015

turbine = Turbine_CostsSE_2015()

# 11 ----------
# 12 ----------

turbine.blade_mass = 17650.67  # inline with the windpact estimates
turbine.hub_mass = 31644.5
turbine.pitch_system_mass = 17004.0
turbine.spinner_mass = 1810.5
turbine.low_speed_shaft_mass = 31257.3
#bearingsMass = 9731.41
turbine.main_bearing_mass = 9731.41 / 2
turbine.second_bearing_mass = 9731.41 / 2 #KLD - revisit this in new model
turbine.gearbox_mass = 30237.60
turbine.high_speed_side_mass = 1492.45
turbine.generator_mass = 16699.85
turbine.bedplate_mass = 93090.6
turbine.yaw_system_mass = 11878.24
turbine.tower_mass = 434559.0
turbine.variable_speed_elec_mass = 0. #obsolete - using transformer #Float(iotype='in', units='kg', desc='component mass [kg]')
turbine.hydraulic_cooling_mass = 400. #Float(iotype='in', units='kg', desc='component mass [kg]')
turbine.nacelle_cover_mass = 6837. #Float(iotype='in', units='kg', desc='component mass [kg]')
turbine.other_mass = 8220. #Float(iotype='in', units='kg', desc='component mass [kg]')
turbine.transformer_mass = 11485. #Float(iotype='in', units='kg', desc='component mass [kg]')    

# 12 ----------
# 13 ----------
# other inputs
turbine.machine_rating = 5000.0
turbine.blade_number = 3
turbine.crane = True
turbine.offshore = True
turbine.bearing_number = 2

# 13 ----------
# 14 ----------

turbine.run()

# 14 ----------
# 15 ----------

print "The results for the NREL 5 MW Reference Turbine in an offshore 20 m water depth location are:"
print
print "Overall rotor cost with 3 advanced blades is ${0:.2f} USD".format(turbine.rotorCC.cost)
print "Blade cost is ${0:.2f} USD".format(turbine.rotorCC.bladeCC.cost)
print "Hub cost is ${0:.2f} USD".format(turbine.rotorCC.hubCC.cost)
print "Pitch system cost is ${0:.2f} USD".format(turbine.rotorCC.pitchSysCC.cost)
print "Spinner cost is ${0:.2f} USD".format(turbine.rotorCC.spinnerCC.cost)
print
print "Overall nacelle cost is ${0:.2f} USD".format(turbine.nacelleCC.cost)
print "LSS cost is ${0:.2f} USD".format(turbine.nacelleCC.lssCC.cost)
print "Main bearings cost is ${0:.2f} USD".format(turbine.nacelleCC.bearingsCC.cost)
print "Gearbox cost is ${0:.2f} USD".format(turbine.nacelleCC.gearboxCC.cost)
print "High speed side cost is ${0:.2f} USD".format(turbine.nacelleCC.hssCC.cost)
print "Generator cost is ${0:.2f} USD".format(turbine.nacelleCC.generatorCC.cost)
print "Bedplate cost is ${0:.2f} USD".format(turbine.nacelleCC.bedplateCC.cost)
print "Yaw system cost is ${0:.2f} USD".format(turbine.nacelleCC.yawSysCC.cost)
print "Variable speed electronics cost is ${0:.2f} USD".format(turbine.nacelleCC.vsCC.cost)
print "HVAC cost is ${0:.2f} USD".format(turbine.nacelleCC.hydraulicCC.cost)    
print "Electrical connections cost is ${0:.2f} USD".format(turbine.nacelleCC.elecCC.cost)
print "Controls cost is ${0:.2f} USD".format(turbine.nacelleCC.controlsCC.cost)
print "Mainframe cost is ${0:.2f} USD".format(turbine.nacelleCC.mainframeCC.cost)
print "Transformer cost is ${0:.2f} USD".format(turbine.nacelleCC.transformerCC.cost)
print
print "Tower cost is ${0:.2f} USD".format(turbine.towerCC.cost)
print
print "The overall turbine cost is ${0:.2f} USD".format(turbine.turbine_cost)
print

# 15 ----------
# 16 ----------

# simple test of module
from turbine_costsse.nrel_csm_tcc_2015 import nrel_csm_tcc_2015

trb = nrel_csm_tcc_2015()

# 16 ----------
# 17 ----------

trb.rotor_diameter = 100.0
trb.turbine_class = 'II/III'
trb.blade_has_carbon = True
trb.blade_number = 3    
trb.machine_rating = 2000.0
trb.hub_height = 80.0
trb.bearing_number = 2
trb.crane = True
trb.offshore = False

# 17 ----------
# 18 ----------

# Rotor force calculations for nacelle inputs
maxTipSpd = 80.0
maxEfficiency = 0.9

ratedHubPower  = trb.machine_rating*1000. / maxEfficiency 
rotorSpeed     = (maxTipSpd/(0.5*trb.rotor_diameter)) * (60.0 / (2*np.pi))
trb.rotor_torque = ratedHubPower/(rotorSpeed*(np.pi/30))

# 18 ----------
# 19 ----------

trb.run()

# 19 ----------
# 20 ----------

print "The results for the NREL 5 MW Reference Turbine in an offshore 20 m water depth location are:"
print "Overall turbine mass is {0:.2f} kg".format(trb.turbine_mass)
print "Overall turbine cost is ${0:.2f} USD".format(trb.turbine_cost)

# 20 ----------