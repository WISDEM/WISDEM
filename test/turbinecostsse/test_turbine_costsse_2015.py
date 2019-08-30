from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
import unittest
import numpy as np
class TestNewAssembly(unittest.TestCase):
  def test1(self):

    turbine = Turbine_CostsSE_2015()

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

    # other inputs
    turbine.machine_rating = 5000.0
    turbine.blade_number = 3
    turbine.crane = True
    turbine.offshore = True
    turbine.bearing_number = 2

    turbine.run()
    self.assertEqual(np.round(turbine.rotorCC.cost, 2), 1292397.85)
    self.assertEqual(np.round(turbine.nacelleCC.generatorCC.cost, 2), 207078.14)
    self.assertEqual(np.round(turbine.nacelleCC.transformerCC.cost, 2), 215918.00)
    self.assertEqual(np.round(turbine.turbine_cost, 2), 4664967.03)
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

if __name__ == "__main__":
    unittest.main()
