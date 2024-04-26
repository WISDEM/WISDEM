import unittest

import openmdao.api as om

import wisdem.nrelcsm.nrel_csm_mass_2015 as nct2015


class TestAll(unittest.TestCase):
    def testMass(self):
        # simple test of module
        prob = om.Problem(reports=False)
        prob.model = nct2015.nrel_csm_mass_2015()
        prob.setup()

        prob["rotor_diameter"] = 126.0
        prob["turbine_class"] = 1
        prob["blade_has_carbon"] = False
        prob["blade_number"] = 3
        prob["machine_rating"] = 5000.0
        prob["tower_length"] = 90.0
        prob["main_bearing_number"] = 2
        prob["crane"] = True
        prob["max_tip_speed"] = 80.0
        prob["max_efficiency"] = 0.90

        prob.run_model()

        self.assertAlmostEqual(float(prob["blade_mass"][0]), 18590.66820649, 2)
        self.assertAlmostEqual(float(prob["hub_mass"][0]), 44078.53687493, 2)
        self.assertAlmostEqual(float(prob["pitch_system_mass"][0]), 10798.90594644, 2)
        self.assertAlmostEqual(float(prob["spinner_mass"][0]), 973.0, 2)
        self.assertAlmostEqual(float(prob["lss_mass"][0]), 22820.27928238, 2)
        self.assertAlmostEqual(float(prob["main_bearing_mass"][0]), 2245.41649102, 2)
        self.assertAlmostEqual(float(prob["rated_rpm"][0]), 12.1260909, 2)
        self.assertAlmostEqual(float(prob["rotor_torque"][0]), 4375.0, 2)
        self.assertAlmostEqual(float(prob["gearbox_mass"][0]), 21875.0, 2)
        self.assertAlmostEqual(float(prob["hss_mass"][0]), 994.7, 2)
        self.assertAlmostEqual(float(prob["generator_mass"][0]), 14900.0, 2)
        self.assertAlmostEqual(float(prob["bedplate_mass"][0]), 41765.26095285, 2)
        self.assertAlmostEqual(float(prob["yaw_mass"][0]), 12329.96247921, 2)
        self.assertAlmostEqual(float(prob["hvac_mass"][0]), 400.0, 2)
        self.assertAlmostEqual(float(prob["cover_mass"][0]), 6836.69, 2)
        self.assertAlmostEqual(float(prob["platforms_mass"][0]), 8220.65761911, 2)
        self.assertAlmostEqual(float(prob["transformer_mass"][0]), 11485.0, 2)
        self.assertAlmostEqual(float(prob["tower_mass"][0]), 182336.48057717, 2)
        self.assertAlmostEqual(float(prob["hub_system_mass"][0]), 55850.44282136, 2)
        self.assertAlmostEqual(float(prob["rotor_mass"][0]), 111622.44744083, 2)
        self.assertAlmostEqual(float(prob["nacelle_mass"][0]),  151455.88331558352, 2)
        self.assertAlmostEqual(float(prob["turbine_mass"][0]), 445414.81133358914, 2)

    def testMassAndCost(self):
        # simple test of module
        prob = om.Problem(reports=False)
        prob.model = nct2015.nrel_csm_2015()
        prob.setup()

        prob["rotor_diameter"] = 126.0
        prob["turbine_class"] = 1
        prob["blade_has_carbon"] = False
        prob["blade_number"] = 3
        prob["machine_rating"] = 5000.0
        prob["tower_length"] = 90.0
        prob["main_bearing_number"] = 2
        prob["crane"] = True
        prob["max_tip_speed"] = 80.0
        prob["max_efficiency"] = 0.90

        prob.run_model()

        self.assertAlmostEqual(float(prob["blade_mass"][0]), 18590.66820649, 2)
        self.assertAlmostEqual(float(prob["hub_mass"][0]), 44078.53687493, 2)
        self.assertAlmostEqual(float(prob["pitch_system_mass"][0]), 10798.90594644, 2)
        self.assertAlmostEqual(float(prob["spinner_mass"][0]), 973.0, 2)
        self.assertAlmostEqual(float(prob["lss_mass"][0]), 22820.27928238, 2)
        self.assertAlmostEqual(float(prob["main_bearing_mass"][0]), 2245.41649102, 2)
        self.assertAlmostEqual(float(prob["rated_rpm"][0]), 12.1260909, 2)
        self.assertAlmostEqual(float(prob["rotor_torque"][0]), 4375.0, 2)
        self.assertAlmostEqual(float(prob["gearbox_mass"][0]), 21875.0, 2)
        self.assertAlmostEqual(float(prob["hss_mass"][0]), 994.7, 2)
        self.assertAlmostEqual(float(prob["generator_mass"][0]), 14900.0, 2)
        self.assertAlmostEqual(float(prob["bedplate_mass"][0]), 41765.26095285, 2)
        self.assertAlmostEqual(float(prob["yaw_mass"][0]), 12329.96247921, 2)
        self.assertAlmostEqual(float(prob["hvac_mass"][0]), 400.0, 2)
        self.assertAlmostEqual(float(prob["cover_mass"][0]), 6836.69, 2)
        self.assertAlmostEqual(float(prob["platforms_mass"][0]), 8220.65761911, 2)
        self.assertAlmostEqual(float(prob["transformer_mass"][0]), 11485.0, 2)
        self.assertAlmostEqual(float(prob["tower_mass"][0]), 182336.48057717, 2)
        self.assertAlmostEqual(float(prob["hub_system_mass"][0]), 55850.44282136, 2)
        self.assertAlmostEqual(float(prob["rotor_mass"][0]), 111622.44744083, 2)
        self.assertAlmostEqual(float(prob["nacelle_mass"][0]),  151455.88331558352, 2)
        self.assertAlmostEqual(float(prob["turbine_mass"][0]), 445414.81133358914, 2)

        self.assertAlmostEqual(float(prob["blade_cost"][0]), 271423.75581475, 2)
        self.assertAlmostEqual(float(prob["hub_cost"][0]), 171906.29381221, 2)
        self.assertAlmostEqual(float(prob["pitch_system_cost"][0]), 238655.82141628, 2)
        self.assertAlmostEqual(float(prob["spinner_cost"][0]), 10800.3, 2)
        self.assertAlmostEqual(float(prob["hub_system_mass_tcc"][0]), 55850.44282136, 2)
        self.assertAlmostEqual(float(prob["hub_system_cost"][0]), 421362.41522849, 2)
        self.assertAlmostEqual(float(prob["rotor_cost"][0]), 1235633.68267274, 2)
        self.assertAlmostEqual(float(prob["rotor_mass_tcc"][0]), 111622.44744083, 2)
        self.assertAlmostEqual(float(prob["lss_cost"][0]), 271561.32346034, 2)
        self.assertAlmostEqual(float(prob["main_bearing_cost"][0]), 10104.37420958, 2)
        self.assertAlmostEqual(float(prob["gearbox_cost"][0]), 218750.0 , 2)
        self.assertAlmostEqual(float(prob["hss_cost"][0]), 6763.96, 2)
        self.assertAlmostEqual(float(prob["generator_cost"][0]), 184760.0, 2)
        self.assertAlmostEqual(float(prob["bedplate_cost"][0]), 121119.25676326, 2)
        self.assertAlmostEqual(float(prob["yaw_system_cost"][0]), 102338.68857747, 2)
        self.assertAlmostEqual(float(prob["hvac_cost"][0]), 49600.0, 2)
        self.assertAlmostEqual(float(prob["controls_cost"][0]), 105750.0, 2)
        self.assertAlmostEqual(float(prob["converter_cost"][0]), 0.0, 2)
        self.assertAlmostEqual(float(prob["elec_cost"][0]), 209250.0, 2)
        self.assertAlmostEqual(float(prob["cover_cost"][0]), 38969.133, 2)
        self.assertAlmostEqual(float(prob["platforms_cost"][0]), 101273.24528671, 2)
        self.assertAlmostEqual(float(prob["transformer_cost"][0]), 215918.0, 2)
        self.assertAlmostEqual(float(prob["nacelle_cost"][0]), 1665612.9280069391, 2)
        self.assertAlmostEqual(float(prob["nacelle_mass_tcc"][0]), 151455.88331558352 , 2)
        self.assertAlmostEqual(float(prob["tower_parts_cost"][0]), 528775.7936738, 2)
        self.assertAlmostEqual(float(prob["tower_cost"][0]), 528775.7936738, 2)
        self.assertAlmostEqual(float(prob["turbine_mass_tcc"][0]), 445414.81133358914, 2)
        self.assertAlmostEqual(float(prob["turbine_cost"][0]), 3430022.404353479, 2)
        self.assertAlmostEqual(float(prob["turbine_cost_kW"][0]), 686.0044808706958, 2)


if __name__ == "__main__":
    unittest.main()
