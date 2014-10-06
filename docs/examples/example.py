# 1 ---------

# A simple test of WISDEM using the NREL CSM
from wisdem.lcoe.lcoe_csm_assembly import lcoe_csm_assembly

lcoe = lcoe_csm_assembly()

# 1 ---------
# 2 ---------

# NREL 5 MW turbine specifications and plant attributes
lcoe.machine_rating = 5000.0 # Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
lcoe.rotor_diameter = 126.0 # Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
lcoe.max_tip_speed = 80.0 # Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
lcoe.hub_height = 90.0 # Float(units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level')
lcoe.sea_depth = 20.0 # Float(units = 'm', iotype='in', desc = 'sea depth for offshore wind project')

# Parameters
lcoe.drivetrain_design = 'geared' # Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
lcoe.altitude = 0.0 # Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
lcoe.turbine_number = 100 # Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
lcoe.year = 2009 # Int(2009, iotype='in', desc = 'year of project start')
lcoe.month = 12 # Int(12, iotype='in', desc = 'month of project start')

# Extra AEP inputs
lcoe.max_power_coefficient = 0.488 #Float(0.488, iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
lcoe.opt_tsr = 7.525 #Float(7.525, iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
lcoe.cut_in_wind_speed = 3.0 #Float(3.0, units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
lcoe.cut_out_wind_speed = 25.0 #Float(25.0, units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
lcoe.hub_height = 90.0 #Float(90.0, units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
lcoe.altitude = 0.0 #Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
#lcoe.air_density = Float(0.0, units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
lcoe.drivetrain_design = 'geared' #Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
lcoe.shear_exponent = 0.1 #Float(0.1, iotype='in', desc= 'shear exponent for wind plant') #TODO - could use wind model here
lcoe.wind_speed_50m = 8.02 #Float(8.35, units = 'm/s', iotype='in', desc='mean annual wind speed at 50 m height')
lcoe.weibull_k= 2.15 #Float(2.1, iotype='in', desc = 'weibull shape factor for annual wind speed distribution')
lcoe.soiling_losses = 0.0 #Float(0.0, iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines')
lcoe.array_losses = 0.10 #Float(0.06, iotype='in', desc = 'energy losses due to turbine interactions - across entire plant')
lcoe.availability = 0.941 #Float(0.94287630736, iotype='in', desc = 'average annual availbility of wind turbines at plant')
lcoe.turbine_number = 100 #Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
lcoe.thrust_coefficient = 0.50 #Float(0.50, iotype='in', desc='thrust coefficient at rated power')

# Extra TCC inputs
lcoe.blade_number = 3 #Int(3, iotype='in', desc = 'number of rotor blades')
lcoe.offshore = True #Bool(True, iotype='in', desc = 'boolean for offshore')
lcoe.advanced_blade = True #Bool(False, iotype='in', desc = 'boolean for use of advanced blade curve')
lcoe.crane = True #Bool(True, iotype='in', desc = 'boolean for presence of a service crane up tower')
lcoe.advanced_bedplate = 0 #Int(0, iotype='in', desc= 'indicator for drivetrain bedplate design 0 - conventional')   
lcoe.advanced_tower = False #Bool(False, iotype='in', desc = 'advanced tower configuration')

# Extra Finance inputs
lcoe.fixed_charge_rate = 0.12 #Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation')
lcoe.construction_finance_rate = 0.00 #Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs')
lcoe.tax_rate = 0.4 #Float(0.4, iotype = 'in', desc = 'tax rate applied to operations')
lcoe.discount_rate = 0.07 #Float(0.07, iotype = 'in', desc = 'applicable project discount rate')
lcoe.construction_time = 1.0 #Float(1.0, iotype = 'in', desc = 'number of years to complete project construction')
lcoe.project_lifetime = 20.0 #Float(20.0, iotype = 'in', desc = 'project lifetime for LCOE calculation')

# 2 ---------
# 3 ---------

lcoe.run()

# 3 ---------
# 4 --------- 

print "Cost of Energy results for a 500 MW offshore wind farm using the NREL 5 MW reference turbine"
print "LCOE: ${0:.4f} USD/kWh".format(lcoe.lcoe)
print "COE: ${0:.4f} USD/kWh".format(lcoe.coe)
print
print "AEP per turbine: {0:1f} kWh/turbine".format(lcoe.net_aep / lcoe.turbine_number)
print "Turbine Cost: ${0:2f} USD".format(lcoe.turbine_cost)
print "BOS costs per turbine: ${0:2f} USD/turbine".format(lcoe.bos_costs / lcoe.turbine_number)
print "OPEX per turbine: ${0:2f} USD/turbine".format(lcoe.avg_annual_opex / lcoe.turbine_number)
print

# 4 ----------
# 5 ----------

# A simple test of WISDEM using the NREL CSM along with the ECN Offshore OPEX Model
from wisdem.lcoe.lcoe_csm_ecn_assembly import lcoe_csm_ecn_assembly

lcoe = lcoe_csm_ecn_assembly('C:/Models/ECN Model/ECN O&M Model.xls') # Substitute your own path to the ECN Model

# 5 ---------- 
# 6 ----------

# NREL 5 MW turbine specifications and plant attributes
lcoe.machine_rating = 5000.0 # Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
lcoe.rotor_diameter = 126.0 # Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
lcoe.max_tip_speed = 80.0 # Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
lcoe.hub_height = 90.0 # Float(units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level')
lcoe.sea_depth = 20.0 # Float(units = 'm', iotype='in', desc = 'sea depth for offshore wind project')

# Parameters
lcoe.drivetrain_design = 'geared' # Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
lcoe.altitude = 0.0 # Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
lcoe.turbine_number = 100 # Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
lcoe.year = 2009 # Int(2009, iotype='in', desc = 'year of project start')
lcoe.month = 12 # Int(12, iotype='in', desc = 'month of project start')

# Extra AEP inputs
lcoe.max_power_coefficient = 0.488 #Float(0.488, iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
lcoe.opt_tsr = 7.525 #Float(7.525, iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
lcoe.cut_in_wind_speed = 3.0 #Float(3.0, units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
lcoe.cut_out_wind_speed = 25.0 #Float(25.0, units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
lcoe.hub_height = 90.0 #Float(90.0, units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
lcoe.altitude = 0.0 #Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
#lcoe.air_density = Float(0.0, units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
lcoe.drivetrain_design = 'geared' #Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
lcoe.shear_exponent = 0.1 #Float(0.1, iotype='in', desc= 'shear exponent for wind plant') #TODO - could use wind model here
lcoe.wind_speed_50m = 8.02 #Float(8.35, units = 'm/s', iotype='in', desc='mean annual wind speed at 50 m height')
lcoe.weibull_k= 2.15 #Float(2.1, iotype='in', desc = 'weibull shape factor for annual wind speed distribution')
lcoe.soiling_losses = 0.0 #Float(0.0, iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines')
lcoe.array_losses = 0.10 #Float(0.06, iotype='in', desc = 'energy losses due to turbine interactions - across entire plant')
lcoe.availability = 0.941 #Float(0.94287630736, iotype='in', desc = 'average annual availbility of wind turbines at plant')
lcoe.turbine_number = 100 #Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
lcoe.thrust_coefficient = 0.50 #Float(0.50, iotype='in', desc='thrust coefficient at rated power')

# Extra TCC inputs
lcoe.blade_number = 3 #Int(3, iotype='in', desc = 'number of rotor blades')
lcoe.offshore = True #Bool(True, iotype='in', desc = 'boolean for offshore')
lcoe.advanced_blade = True #Bool(False, iotype='in', desc = 'boolean for use of advanced blade curve')
lcoe.crane = True #Bool(True, iotype='in', desc = 'boolean for presence of a service crane up tower')
lcoe.advanced_bedplate = 0 #Int(0, iotype='in', desc= 'indicator for drivetrain bedplate design 0 - conventional')   
lcoe.advanced_tower = False #Bool(False, iotype='in', desc = 'advanced tower configuration')

# Extra Finance inputs
lcoe.fixed_charge_rate = 0.12 #Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation')
lcoe.construction_finance_rate = 0.00 #Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs')
lcoe.tax_rate = 0.4 #Float(0.4, iotype = 'in', desc = 'tax rate applied to operations')
lcoe.discount_rate = 0.07 #Float(0.07, iotype = 'in', desc = 'applicable project discount rate')
lcoe.construction_time = 1.0 #Float(1.0, iotype = 'in', desc = 'number of years to complete project construction')
lcoe.project_lifetime = 20.0 #Float(20.0, iotype = 'in', desc = 'project lifetime for LCOE calculation')

# 6 ----------
# 7 ----------

lcoe.run()

# 7 ----------
# 8 ----------

print "Cost of Energy results for a 500 MW offshore wind farm using the NREL 5 MW reference turbine"
print "LCOE: ${0:.4f} USD/kWh".format(lcoe.lcoe)
print "COE: ${0:.4f} USD/kWh".format(lcoe.coe)
print
print "AEP per turbine: {0:1f} kWh/turbine".format(lcoe.net_aep / lcoe.turbine_number)
print "Turbine Cost: ${0:2f} USD".format(lcoe.turbine_cost)
print "BOS costs per turbine: ${0:2f} USD/turbine".format(lcoe.bos_costs / lcoe.turbine_number)
print "OPEX per turbine: ${0:2f} USD/turbine".format(lcoe.avg_annual_opex / lcoe.turbine_number)

# 8 -----------
# 9 -----------
from wisdem.lcoe.lcoe_se_csm_assembly import lcoe_se_assembly

with_new_nacelle = True
with_landbos = False
flexible_blade = False
with_3pt_drive = False
sea_depth = 0.0
wind_class = 'I'

# === Create LCOE SE assembly ========
lcoe_se = lcoe_se_assembly(with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive)

# 9 -----------
