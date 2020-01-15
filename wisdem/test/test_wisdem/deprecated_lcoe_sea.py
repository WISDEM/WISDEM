from wisdem.lcoe.lcoe_se_assembly import lcoe_se_assembly
import numpy as np
import unittest
# NREL 5 MW in land-based wind plant with high winds (as class I)
class test_lcoe_se(unittest.TestCase):
  def test1(self):
    wind_class = 'I'
    sea_depth = 0.0
    with_new_nacelle = True
    with_landbos = False
    flexible_blade = False
    with_3pt_drive = False
    with_ecn_opex = False
    ecn_file = ''
    with_openwind=False
    ow_file=None
    ow_wkbook=None
    
    lcoe_se = lcoe_se_assembly(with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)
    
    # === Set assembly variables and objects ===
    lcoe_se.sea_depth = sea_depth # 0.0 for land-based turbine
    lcoe_se.turbine_number = 100
    lcoe_se.year = 2009
    lcoe_se.month = 12
    
    rotor = lcoe_se.rotor
    nacelle = lcoe_se.nacelle
    tower = lcoe_se.tower
    tcc_a = lcoe_se.tcc_a
    # bos_a = lcoe_se.bos_a
    # opex_a = lcoe_se.opex_a
    aep_a = lcoe_se.aep_a
    fin_a = lcoe_se.fin_a
    
    # Turbine ===========
    from wisdem.reference_turbines.nrel5mw.nrel5mw import configure_nrel5mw_turbine
    configure_nrel5mw_turbine(lcoe_se,wind_class,lcoe_se.sea_depth)
    
    # tcc ====
    lcoe_se.advanced_blade = True
    lcoe_se.offshore = False
    lcoe_se.assemblyCostMultiplier = 0.30
    lcoe_se.profitMultiplier = 0.20
    lcoe_se.overheadCostMultiplier = 0.0
    lcoe_se.transportMultiplier = 0.0
    
    # for new landBOS
    # === new landBOS ===
    if with_landbos:
        lcoe_se.voltage = 137
        lcoe_se.distInter = 5
        lcoe_se.terrain = 'FLAT_TO_ROLLING'
        lcoe_se.layout = 'SIMPLE'
        lcoe_se.soil = 'STANDARD'
    
    # aep ==== # based on COE review for land-based machines
    if not with_openwind:
        lcoe_se.array_losses = 0.059
        lcoe_se.A = 8.9 # weibull of 7.25 at 50 m with shear exp of 0.143
        lcoe_se.k = 2.0
    lcoe_se.other_losses = 0.101
    if not with_ecn_opex:
        lcoe_se.availability = 0.94
    
    # fin ===
    lcoe_se.fixed_charge_rate = 0.095
    lcoe_se.construction_finance_rate = 0.0
    lcoe_se.tax_rate = 0.4
    lcoe_se.discount_rate = 0.07
    lcoe_se.construction_time = 1.0
    lcoe_se.project_lifetime = 20.0
    
    # Set plant level inputs ===
    shearExp = 0.2 #TODO : should be an input to lcoe
    #rotor.cdf_reference_height_wind_speed = 90.0
    if not with_openwind:
        lcoe_se.array_losses = 0.1
    lcoe_se.other_losses = 0.0
    if not with_ecn_opex:
        lcoe_se.availability = 0.98
    rotor.turbulence_class = 'B'
    lcoe_se.multiplier = 2.23
    
    if wind_class == 'Offshore':
        # rotor.cdf_reference_mean_wind_speed = 8.4 # TODO - aep from its own module
        # rotor.cdf_reference_height_wind_speed = 50.0
        # rotor.weibull_shape = 2.1
        shearExp = 0.14 # TODO : should be an input to lcoe
        lcoe_se.array_losses = 0.15
        if not with_ecn_opex:
            lcoe_se.availability = 0.96
        lcoe_se.offshore = True
        lcoe_se.multiplier = 2.33
        lcoe_se.fixed_charge_rate = 0.118
    
    rotor.shearExp = shearExp
    tower.wind1.shearExp = shearExp
    tower.wind2.shearExp = shearExp
    
    # ====
    
    # === Run default assembly and print results
    lcoe_se.run()
    # ====
    self.assertEqual(np.round(lcoe_se.rotor.mass_all_blades, 2), 54674.80)
    self.assertEqual(np.round(lcoe_se.maxdeflection.ground_clearance, 2), 28.47)
    # === Print ===
    
    print "Key Turbine Outputs for NREL 5 MW Reference Turbine"
    print 'mass rotor blades:{0:.2f} (kg) '.format(lcoe_se.rotor.mass_all_blades)
    print 'mass hub system: {0:.2f} (kg) '.format(lcoe_se.hub.hub_system_mass)
    print 'mass nacelle: {0:.2f} (kg) '.format(lcoe_se.nacelle.nacelle_mass)
    print 'mass tower: {0:.2f} (kg) '.format(lcoe_se.tower.mass)
    print 'maximum tip deflection: {0:.2f} (m) '.format(lcoe_se.maxdeflection.max_tip_deflection)
    print 'ground clearance: {0:.2f} (m) '.format(lcoe_se.maxdeflection.ground_clearance)
    print
    print "Key Plant Outputs for wind plant with NREL 5 MW Turbine"
    #print "LCOE: ${0:.4f} USD/kWh".format(lcoe_se.lcoe) # not in base output set (add to assembly output if desired)
    print "COE: ${0:.4f} USD/kWh".format(lcoe_se.coe)
    print
    print "AEP per turbine: {0:.1f} kWh/turbine".format(lcoe_se.net_aep / lcoe_se.turbine_number)
    print "Turbine Cost: ${0:.2f} USD".format(lcoe_se.turbine_cost)
    print "BOS costs per turbine: ${0:.2f} USD/turbine".format(lcoe_se.bos_costs / lcoe_se.turbine_number)
    print "OPEX per turbine: ${0:.2f} USD/turbine".format(lcoe_se.avg_annual_opex / lcoe_se.turbine_number)    
    
if __name__ == "__main__":
    unittest.main()
