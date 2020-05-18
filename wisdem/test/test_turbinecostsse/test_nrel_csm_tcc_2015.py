from openmdao.api import Problem
import wisdem.turbine_costsse.nrel_csm_tcc_2015 as nct2015
import nrel_csm_mass_2015
import unittest
import numpy as np

class TestAll(unittest.TestCase):

    def testMass(self):

        # simple test of module
        prob = om.Problem()
        prob.model = nct2015.nrel_csm_mass_2015()
        prob.setup()

        prob['rotor_diameter'] = 126.0
        prob['turbine_class'] = 1
        prob['blade_has_carbon'] = False
        prob['blade_number'] = 3    
        prob['machine_rating'] = 5000.0
        prob['hub_height'] = 90.0
        prob['main_bearing_number'] = 2
        prob['crane'] = True
        prob['max_tip_speed'] = 80.0
        prob['max_efficiency'] = 0.90

        prob.run_model()
        
        self.assertEqual(np.round(prob['blade_mass'], 2), 18590.66820649)
        self.assertEqual(np.round(prob['hub_mass'], 2), 44078.53687493)
        self.assertEqual(np.round(prob['pitch_system_mass'], 2), 10798.90594644)
        self.assertEqual(np.round(prob['spinner_mass'], 2), 973.)
        self.assertEqual(np.round(prob['lss_mass'], 2), 22820.27928238)
        self.assertEqual(np.round(prob['main_bearing_mass'], 2), 2245.41649102)
        self.assertEqual(np.round(prob['rated_rpm'], 2), 12.1260909)
        self.assertEqual(np.round(prob['rotor_torque'], 2), 4375000.)
        self.assertEqual(np.round(prob['gearbox_mass'], 2), 43468.32086769)
        self.assertEqual(np.round(prob['hss_mass'], 2), 994.7)
        self.assertEqual(np.round(prob['generator_mass'], 2), 14900.)
        self.assertEqual(np.round(prob['bedplate_mass'], 2), 41765.26095285)
        self.assertEqual(np.round(prob['yaw_mass'], 2), 12329.96247921)
        self.assertEqual(np.round(prob['hvac_mass'], 2), 400.)
        self.assertEqual(np.round(prob['cover_mass'], 2), 6836.69)
        self.assertEqual(np.round(prob['platforms_mass'], 2), 8220.65761911)
        self.assertEqual(np.round(prob['transformer_mass'], 2), 11485.)
        self.assertEqual(np.round(prob['tower_mass'], 2), 182336.48057717)
        self.assertEqual(np.round(prob['hub_system_mass'], 2), 55850.44282136)
        self.assertEqual(np.round(prob['rotor_mass'], 2), 111622.44744083)
        self.assertEqual(np.round(prob['nacelle_mass'], 2), 167711.70418327)
        self.assertEqual(np.round(prob['turbine_mass'], 2), 461670.63220128)


    def testMassAndCost(self):

        # simple test of module
        prob = om.Problem()
        prob.model = nct2015.nrel_csm_2015()
        prob.setup()

        prob['rotor_diameter'] = 126.0
        prob['turbine_class'] = 1
        prob['blade_has_carbon'] = False
        prob['blade_number'] = 3    
        prob['machine_rating'] = 5000.0
        prob['hub_height'] = 90.0
        prob['main_bearing_number'] = 2
        prob['crane'] = True
        prob['max_tip_speed'] = 80.0
        prob['max_efficiency'] = 0.90

        prob.run_model()

        self.assertEqual(np.round(prob['blade_mass'], 2), 18590.66820649)
        self.assertEqual(np.round(prob['hub_mass'], 2), 44078.53687493)
        self.assertEqual(np.round(prob['pitch_system_mass'], 2), 10798.90594644)
        self.assertEqual(np.round(prob['spinner_mass'], 2), 973.)
        self.assertEqual(np.round(prob['lss_mass'], 2), 22820.27928238)
        self.assertEqual(np.round(prob['main_bearing_mass'], 2), 2245.41649102)
        self.assertEqual(np.round(prob['rated_rpm'], 2), 12.1260909)
        self.assertEqual(np.round(prob['rotor_torque'], 2), 4375000.)
        self.assertEqual(np.round(prob['gearbox_mass'], 2), 43468.32086769)
        self.assertEqual(np.round(prob['hss_mass'], 2), 994.7)
        self.assertEqual(np.round(prob['generator_mass'], 2), 14900.)
        self.assertEqual(np.round(prob['bedplate_mass'], 2), 41765.26095285)
        self.assertEqual(np.round(prob['yaw_mass'], 2), 12329.96247921)
        self.assertEqual(np.round(prob['hvac_mass'], 2), 400.)
        self.assertEqual(np.round(prob['cover_mass'], 2), 6836.69)
        self.assertEqual(np.round(prob['platforms_mass'], 2), 8220.65761911)
        self.assertEqual(np.round(prob['transformer_mass'], 2), 11485.)
        self.assertEqual(np.round(prob['tower_mass'], 2), 182336.48057717)
        self.assertEqual(np.round(prob['hub_system_mass'], 2), 55850.44282136)
        self.assertEqual(np.round(prob['rotor_mass'], 2), 111622.44744083)
        self.assertEqual(np.round(prob['nacelle_mass'], 2), 167711.70418327)
        self.assertEqual(np.round(prob['turbine_mass'], 2), 461670.63220128)

        self.assertEqual(np.round(prob['blade_cost'], 2), 271423.75581475)
        self.assertEqual(np.round(prob['hub_cost'], 2), 171906.29381221)
        self.assertEqual(np.round(prob['pitch_system_cost'], 2), 238655.82141628)
        self.assertEqual(np.round(prob['spinner_cost'], 2), 10800.3)
        self.assertEqual(np.round(prob['hub_system_mass_tcc'], 2), 55850.44282136)
        self.assertEqual(np.round(prob['hub_system_cost'], 2), 421362.41522849)
        self.assertEqual(np.round(prob['rotor_cost'], 2), 1235633.68267274)
        self.assertEqual(np.round(prob['rotor_mass_tcc'], 2), 111622.44744083)
        self.assertEqual(np.round(prob['lss_cost'], 2), 271561.32346034)
        self.assertEqual(np.round(prob['main_bearing_cost'], 2), 10104.37420958)
        self.assertEqual(np.round(prob['gearbox_cost'], 2), 560741.33919321)
        self.assertEqual(np.round(prob['hss_cost'], 2), 6763.96)
        self.assertEqual(np.round(prob['generator_cost'], 2), 184760.)
        self.assertEqual(np.round(prob['bedplate_cost'], 2), 121119.25676326)
        self.assertEqual(np.round(prob['yaw_system_cost'], 2), 102338.68857747)
        self.assertEqual(np.round(prob['hvac_cost'], 2), 49600.)
        self.assertEqual(np.round(prob['controls_cost'], 2), 105750.)
        self.assertEqual(np.round(prob['converter_cost'], 2), 0.)
        self.assertEqual(np.round(prob['elec_cost'], 2), 209250.)
        self.assertEqual(np.round(prob['cover_cost'], 2), 38969.133)
        self.assertEqual(np.round(prob['platforms_cost'], 2), 101273.24528671)
        self.assertEqual(np.round(prob['transformer_cost'], 2), 215918.)
        self.assertEqual(np.round(prob['nacelle_cost'], 2), 1988253.69470015)
        self.assertEqual(np.round(prob['nacelle_mass_tcc'], 2), 159491.04656417)
        self.assertEqual(np.round(prob['tower_parts_cost'], 2), 528775.7936738)
        self.assertEqual(np.round(prob['tower_cost'], 2), 528775.7936738)
        self.assertEqual(np.round(prob['turbine_mass_tcc'], 2), 453449.97458217)
        self.assertEqual(np.round(prob['turbine_cost'], 2), 3752663.17104668)
        self.assertEqual(np.round(prob['turbine_cost_kW'], 2), 750.53263421)
        
        

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAll))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

