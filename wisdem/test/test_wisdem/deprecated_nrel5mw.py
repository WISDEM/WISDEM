import unittest
import numpy as np
from wisdem.turbinese.turbine import TurbineSE
from wisdem.reference_turbines.nrel5mw.nrel5mw import configure_nrel5mw_turbine

class NREL_WISDEMTestCase(unittest.TestCase):

  def setUp(self):
        pass
        
  def tearDown(self):
        pass
        
    
  def test_NREL_WISDEM_Turbine(self):
    turbine = TurbineSE()
    turbine.sea_depth = 0.0 # 0.0 for land-based turbine
    wind_class = 'I'

    configure_nrel5mw_turbine(turbine,wind_class,turbine.sea_depth)

    # === run ===
    turbine.run()
    self.assertEqual(np.round(turbine.rotor.mass_all_blades, 1), 54674.8)
    self.assertEqual(np.round(turbine.maxdeflection.ground_clearance, 1), 28.5)
    print 'mass rotor blades (kg) =', turbine.rotor.mass_all_blades
    print 'mass hub system (kg) =', turbine.hubSystem.hub_system_mass
    print 'mass nacelle (kg) =', turbine.nacelle.nacelle_mass
    print 'mass tower (kg) =', turbine.tower.mass
    print 'maximum tip deflection (m) =', turbine.maxdeflection.max_tip_deflection
    print 'ground clearance (m) =', turbine.maxdeflection.ground_clearance
    # print

        
        
if __name__ == "__main__":
    unittest.main()
    
