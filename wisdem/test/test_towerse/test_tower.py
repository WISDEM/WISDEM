import copy
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.towerse.tower as tow
from wisdem.towerse import RIGID
from wisdem.commonse import gravity as g


class TestTowerSE(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        # Store analysis options
        self.modeling_options = {}
        self.modeling_options["materials"] = {}
        self.modeling_options["materials"]["n_mat"] = 1

        self.modeling_options["flags"] = {}
        self.modeling_options["flags"]["monopile"] = False

        self.modeling_options["WISDEM"] = {}
        self.modeling_options["WISDEM"]["TowerSE"] = {}
        self.modeling_options["WISDEM"]["TowerSE"]["buckling_method"] = "eurocode"
        self.modeling_options["WISDEM"]["TowerSE"]["buckling_length"] = 30.0
        self.modeling_options["WISDEM"]["TowerSE"]["n_height_tower"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["n_height_monopile"] = 0
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers_monopile"] = 0
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["n_refine"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["wind"] = "PowerWind"
        self.modeling_options["WISDEM"]["TowerSE"]["nLC"] = 1

        self.modeling_options["WISDEM"]["TowerSE"]["soil_springs"] = False
        self.modeling_options["WISDEM"]["TowerSE"]["gravity_foundation"] = False

        self.modeling_options["WISDEM"]["TowerSE"]["gamma_f"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_m"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_n"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_b"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_fatigue"] = 1.0

        # Simplified the options available to the user
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"] = {}
        # self.modeling_options['TowerSE']['frame3dd']['DC']      = 80.0
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["shear"] = True
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["geom"] = True
        # self.modeling_options['TowerSE']['frame3dd']['dx']      = -1
        # self.modeling_options['TowerSE']['frame3dd']['nM']      = 6
        # self.modeling_options['TowerSE']['frame3dd']['Mmethod'] = 1
        # self.modeling_options['TowerSE']['frame3dd']['lump']    = 0
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["tol"] = 1e-9
        # self.modeling_options['TowerSE']['frame3dd']['shift']   = 0.0
        # self.modeling_options['TowerSE']['frame3dd']['add_gravity'] = True

    def testProblemLand(self):

        prob = om.Problem()
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["hub_height"] = 80.0
        prob["transition_piece_mass"] = 0.0
        prob["transition_piece_cost"] = 0.0
        prob["gravity_foundation_mass"] = 0.0

        prob["tower_s"] = np.linspace(0, 1, 3)
        prob["tower_foundation_height"] = 0.0
        prob["tower_height"] = 80.0
        # prob['tower_section_height'] = 40.0*np.ones(2)
        prob["tower_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["tower_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["tower_outfitting_factor"] = 1.0
        prob["tower_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8

        prob["yaw"] = 0.0
        prob["rna_mass"] = 2e5
        prob["rna_I"] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob["rna_cg"] = np.array([-3.0, 0.0, 1.0])
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["wind.Uref"] = 15.0
        prob["pre.rna_F"] = 1e3 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob["pre.rna_M"] = 1e4 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4 * (5.0 ** 2 - 4.9 ** 2) * np.pi
        npt.assert_equal(prob["z_start"], 0.0)
        npt.assert_equal(prob["transition_piece_height"], 0.0)
        npt.assert_equal(prob["suctionpile_depth"], 0.0)
        npt.assert_equal(prob["z_param"], np.array([0.0, 40.0, 80.0]))

        self.assertEqual(prob["height_constraint"], 0.0)
        self.assertEqual(prob["tower_cost"], prob["cm.cost"])
        npt.assert_equal(prob["tower_I_base"], prob["cm.I_base"])
        npt.assert_almost_equal(prob["tower_center_of_mass"], 40.0)
        npt.assert_equal(prob["tower_section_center_of_mass"], prob["cm.section_center_of_mass"])
        self.assertEqual(prob["monopile_mass"], 0.0)
        self.assertEqual(prob["monopile_cost"], 0.0)
        npt.assert_almost_equal(prob["tower_mass"], mass_dens * 80.0)

        npt.assert_equal(prob["pre.kidx"], np.array([0], dtype=np.int_))
        npt.assert_equal(prob["pre.kx"], np.array([RIGID]))
        npt.assert_equal(prob["pre.ky"], np.array([RIGID]))
        npt.assert_equal(prob["pre.kz"], np.array([RIGID]))
        npt.assert_equal(prob["pre.ktx"], np.array([RIGID]))
        npt.assert_equal(prob["pre.kty"], np.array([RIGID]))
        npt.assert_equal(prob["pre.ktz"], np.array([RIGID]))

        npt.assert_equal(prob["pre.midx"], np.array([6, 0, 0]))
        npt.assert_equal(prob["pre.m"], np.array([2e5, 0, 0]))
        npt.assert_equal(prob["pre.mrhox"], np.array([-3.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mrhoy"], np.array([0.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mrhoz"], np.array([1.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mIxx"], np.array([1e5, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mIyy"], np.array([1e5, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mIzz"], np.array([2e5, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mIxy"], np.zeros(3))
        npt.assert_equal(prob["pre.mIxz"], np.zeros(3))
        npt.assert_equal(prob["pre.mIyz"], np.zeros(3))

        npt.assert_equal(prob["pre.plidx"], np.array([6]))
        npt.assert_equal(prob["pre.Fx"], np.array([2e3]))
        npt.assert_equal(prob["pre.Fy"], np.array([3e3]))
        npt.assert_equal(prob["pre.Fz"], np.array([4e3]))
        npt.assert_equal(prob["pre.Mxx"], np.array([2e4]))
        npt.assert_equal(prob["pre.Myy"], np.array([3e4]))
        npt.assert_equal(prob["pre.Mzz"], np.array([4e4]))

    def testProblemFixedPile(self):
        self.modeling_options["WISDEM"]["TowerSE"]["n_height_monopile"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers_monopile"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = 5
        self.modeling_options["WISDEM"]["TowerSE"]["soil_springs"] = True
        self.modeling_options["WISDEM"]["TowerSE"]["gravity_foundation"] = False
        self.modeling_options["flags"]["monopile"] = True

        prob = om.Problem()
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["hub_height"] = 80.0
        prob["water_depth"] = 30.0
        prob["transition_piece_mass"] = 1e2
        prob["transition_piece_cost"] = 1e3
        prob["gravity_foundation_mass"] = 0.0  # 1e4

        prob["tower_s"] = np.linspace(0, 1, 3)
        prob["tower_foundation_height"] = 0.0
        prob["tower_height"] = 60.0
        prob["tower_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["tower_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["tower_outfitting_factor"] = 1.0
        hval = np.array([15.0, 30.0])
        prob["monopile_s"] = np.cumsum(np.r_[0, hval]) / hval.sum()
        prob["monopile_foundation_height"] = -45.0
        prob["monopile_height"] = hval.sum()
        prob["monopile_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["monopile_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["monopile_outfitting_factor"] = 1.0
        prob["tower_layer_materials"] = prob["monopile_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8

        prob["outfitting_factor"] = 1.0
        prob["yaw"] = 0.0
        prob["G_soil"] = 1e7
        prob["nu_soil"] = 0.5
        prob["rna_mass"] = 2e5
        prob["rna_I"] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob["rna_cg"] = np.array([-3.0, 0.0, 1.0])
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.3351e-3
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["Hsig_wave"] = 0.0
        prob["Tsig_wave"] = 1e3
        prob["wind.Uref"] = 15.0
        prob["pre.rna_F"] = 1e3 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob["pre.rna_M"] = 1e4 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4 * (5.0 ** 2 - 4.9 ** 2) * np.pi
        npt.assert_equal(prob["z_start"], -45.0)
        npt.assert_equal(prob["transition_piece_height"], 0.0)
        npt.assert_equal(prob["suctionpile_depth"], 15.0)
        npt.assert_equal(prob["z_param"], np.array([-45.0, -30.0, 0.0, 30.0, 60.0]))

        self.assertEqual(prob["height_constraint"], 20.0)
        npt.assert_almost_equal(prob["tower_cost"], (60.0 / 105.0) * prob["cm.cost"])
        npt.assert_equal(prob["tower_I_base"][:2], prob["cm.I_base"][:2] + 1e2 * 45 ** 2)
        npt.assert_equal(prob["tower_I_base"][2:], prob["cm.I_base"][2:])
        npt.assert_almost_equal(
            prob["tower_center_of_mass"],
            (7.5 * mass_dens * 105.0 + 0.0 * 1e2) / (mass_dens * 105 + 1e2),
        )
        npt.assert_equal(prob["tower_section_center_of_mass"], prob["cm.section_center_of_mass"])
        npt.assert_almost_equal(prob["monopile_cost"], (45.0 / 105.0) * prob["cm.cost"] + 1e3)
        npt.assert_almost_equal(prob["monopile_mass"], mass_dens * 45.0 + 1e2)
        npt.assert_almost_equal(prob["tower_mass"], mass_dens * 60.0)

        npt.assert_equal(prob["pre.kidx"], np.arange(4, dtype=np.int_))
        npt.assert_array_less(prob["pre.kx"], RIGID)
        npt.assert_array_less(prob["pre.ky"], RIGID)
        npt.assert_array_less(prob["pre.kz"][0], RIGID)
        npt.assert_array_less(prob["pre.ktx"], RIGID)
        npt.assert_array_less(prob["pre.kty"], RIGID)
        npt.assert_array_less(prob["pre.ktz"], RIGID)
        npt.assert_array_less(0.0, prob["pre.kx"])
        npt.assert_array_less(0.0, prob["pre.ky"])
        npt.assert_array_less(0.0, prob["pre.kz"][0])
        npt.assert_array_less(0.0, prob["pre.ktx"])
        npt.assert_array_less(0.0, prob["pre.kty"])
        npt.assert_array_less(0.0, prob["pre.ktz"])
        npt.assert_equal(0.0, prob["pre.kz"][1:])

        npt.assert_equal(prob["pre.midx"], np.array([12, 6, 0]))
        npt.assert_equal(prob["pre.m"], np.array([2e5, 1e2, 0]))
        npt.assert_equal(prob["pre.mrhox"], np.array([-3.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mrhoy"], np.array([0.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mrhoz"], np.array([1.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mIxx"], np.array([1e5, 1e2 * 25 * 0.5, 0]))
        npt.assert_equal(prob["pre.mIyy"], np.array([1e5, 1e2 * 25 * 0.5, 0]))
        npt.assert_equal(prob["pre.mIzz"], np.array([2e5, 1e2 * 25, 0]))
        npt.assert_equal(prob["pre.mIxy"], np.zeros(3))
        npt.assert_equal(prob["pre.mIxz"], np.zeros(3))
        npt.assert_equal(prob["pre.mIyz"], np.zeros(3))

        npt.assert_equal(prob["pre.plidx"], np.array([12]))
        npt.assert_equal(prob["pre.Fx"], np.array([2e3]))
        npt.assert_equal(prob["pre.Fy"], np.array([3e3]))
        npt.assert_equal(prob["pre.Fz"], np.array([4e3]))
        npt.assert_equal(prob["pre.Mxx"], np.array([2e4]))
        npt.assert_equal(prob["pre.Myy"], np.array([3e4]))
        npt.assert_equal(prob["pre.Mzz"], np.array([4e4]))
        npt.assert_almost_equal(prob["tower.base_F"], [4.61183362e04, 1.59353875e03, -2.94077236e07], 0)
        npt.assert_almost_equal(prob["tower.base_M"], [-248566.38259147, -3286049.81237828, 40000.0], 0)

    def testProblemFixedPile_GBF(self):
        self.modeling_options["WISDEM"]["TowerSE"]["n_height_monopile"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers_monopile"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = 5
        self.modeling_options["WISDEM"]["TowerSE"]["soil_springs"] = False
        self.modeling_options["WISDEM"]["TowerSE"]["gravity_foundation"] = True
        self.modeling_options["flags"]["monopile"] = True

        prob = om.Problem()
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["hub_height"] = 80.0
        prob["water_depth"] = 30.0
        prob["transition_piece_mass"] = 1e2
        prob["transition_piece_cost"] = 1e3
        prob["gravity_foundation_mass"] = 1e4

        prob["tower_s"] = np.linspace(0, 1, 3)
        prob["tower_foundation_height"] = 0.0
        prob["tower_height"] = 60.0
        prob["tower_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["tower_layer_thickness"] = 0.1 * np.ones(3).reshape((1, 3))
        prob["tower_outfitting_factor"] = 1.0
        hval = np.array([15.0, 30.0])
        prob["monopile_s"] = np.cumsum(np.r_[0, hval]) / hval.sum()
        prob["monopile_foundation_height"] = -45.0
        prob["monopile_height"] = hval.sum()
        prob["monopile_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["monopile_layer_thickness"] = 0.1 * np.ones(3).reshape((1, 3))
        prob["monopile_outfitting_factor"] = 1.0
        prob["tower_layer_materials"] = prob["monopile_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8

        prob["outfitting_factor"] = 1.0
        prob["yaw"] = 0.0
        prob["rna_mass"] = 2e5
        prob["rna_I"] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob["rna_cg"] = np.array([-3.0, 0.0, 1.0])
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.3351e-3
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["Hsig_wave"] = 0.0
        prob["Tsig_wave"] = 1e3
        prob["wind.Uref"] = 15.0
        prob["pre.rna_F"] = 1e3 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob["pre.rna_M"] = 1e4 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4 * (5.0 ** 2 - 4.9 ** 2) * np.pi
        npt.assert_equal(prob["z_start"], -45.0)
        npt.assert_equal(prob["transition_piece_height"], 0.0)
        npt.assert_equal(prob["suctionpile_depth"], 15.0)
        npt.assert_equal(prob["z_param"], np.array([-45.0, -30.0, 0.0, 30.0, 60.0]))

        self.assertEqual(prob["height_constraint"], 20.0)
        npt.assert_almost_equal(prob["tower_cost"], (60.0 / 105.0) * prob["cm.cost"])
        npt.assert_equal(prob["tower_I_base"][:2], prob["cm.I_base"][:2] + 1e2 * 45 ** 2)
        npt.assert_equal(prob["tower_I_base"][2:], prob["cm.I_base"][2:])
        npt.assert_almost_equal(
            prob["tower_center_of_mass"],
            (7.5 * mass_dens * 105.0 + 0.0 * 1e2 + (-45) * 1e4) / (mass_dens * 105 + 1e2 + 1e4),
        )
        npt.assert_equal(prob["tower_section_center_of_mass"], prob["cm.section_center_of_mass"])
        npt.assert_almost_equal(prob["monopile_cost"], (45.0 / 105.0) * prob["cm.cost"] + 1e3)
        npt.assert_almost_equal(prob["monopile_mass"], mass_dens * 45.0 + 1e2 + 1e4)
        npt.assert_almost_equal(prob["tower_mass"], mass_dens * 60.0)

        npt.assert_equal(prob["pre.kidx"], 0)
        npt.assert_equal(prob["pre.kx"], RIGID)
        npt.assert_equal(prob["pre.ky"], RIGID)
        npt.assert_equal(prob["pre.kz"], RIGID)
        npt.assert_equal(prob["pre.ktx"], RIGID)
        npt.assert_equal(prob["pre.kty"], RIGID)
        npt.assert_equal(prob["pre.ktz"], RIGID)

        npt.assert_equal(prob["pre.midx"], np.array([12, 6, 0]))
        npt.assert_equal(prob["pre.m"], np.array([2e5, 1e2, 1e4]))
        npt.assert_equal(prob["pre.mrhox"], np.array([-3.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mrhoy"], np.array([0.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mrhoz"], np.array([1.0, 0.0, 0.0]))
        npt.assert_equal(prob["pre.mIxx"], np.array([1e5, 1e2 * 25 * 0.5, 1e4 * 25 * 0.25]))
        npt.assert_equal(prob["pre.mIyy"], np.array([1e5, 1e2 * 25 * 0.5, 1e4 * 25 * 0.25]))
        npt.assert_equal(prob["pre.mIzz"], np.array([2e5, 1e2 * 25, 1e4 * 25 * 0.5]))
        npt.assert_equal(prob["pre.mIxy"], np.zeros(3))
        npt.assert_equal(prob["pre.mIxz"], np.zeros(3))
        npt.assert_equal(prob["pre.mIyz"], np.zeros(3))

        npt.assert_equal(prob["pre.plidx"], np.array([12]))
        npt.assert_equal(prob["pre.Fx"], np.array([2e3]))
        npt.assert_equal(prob["pre.Fy"], np.array([3e3]))
        npt.assert_equal(prob["pre.Fz"], np.array([4e3]))
        npt.assert_equal(prob["pre.Mxx"], np.array([2e4]))
        npt.assert_equal(prob["pre.Myy"], np.array([3e4]))
        npt.assert_equal(prob["pre.Mzz"], np.array([4e4]))

        npt.assert_almost_equal(prob["tower.base_F"], [3.74393291e04, 1.84264671e03, -3.39826364e07], 0)
        npt.assert_almost_equal(prob["tower.base_M"], [-294477.83027742, -2732413.3684215, 40000.0], 0)

    def testAddedMassForces(self):
        self.modeling_options["WISDEM"]["TowerSE"]["n_height_monopile"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers_monopile"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = 5
        self.modeling_options["WISDEM"]["TowerSE"]["soil_springs"] = False
        self.modeling_options["WISDEM"]["TowerSE"]["gravity_foundation"] = False
        self.modeling_options["flags"]["monopile"] = True

        prob = om.Problem()
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["hub_height"] = 80.0
        prob["water_depth"] = 30.0
        prob["transition_piece_mass"] = 0.0
        prob["transition_piece_cost"] = 0.0
        prob["gravity_foundation_mass"] = 0.0

        prob["tower_s"] = np.linspace(0, 1, 3)
        prob["tower_foundation_height"] = 0.0
        prob["tower_height"] = 60.0
        prob["tower_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["tower_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["tower_outfitting_factor"] = 1.0
        hval = np.array([15.0, 30.0])
        prob["monopile_s"] = np.cumsum(np.r_[0, hval]) / hval.sum()
        prob["monopile_foundation_height"] = -45.0
        prob["monopile_height"] = hval.sum()
        prob["monopile_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["monopile_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["monopile_outfitting_factor"] = 1.0
        prob["tower_layer_materials"] = prob["monopile_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8

        prob["yaw"] = 0.0
        # prob["G_soil"] = 1e7
        # prob["nu_soil"] = 0.5
        prob["rna_mass"] = 0.0
        prob["rna_I"] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob["rna_cg"] = np.array([-3.0, 0.0, 1.0])
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.3351e-3
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["Hsig_wave"] = 0.0
        prob["Tsig_wave"] = 1e3
        prob["wind.Uref"] = 15.0
        prob["pre.rna_F"] = 1e3 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob["pre.rna_M"] = 1e4 * np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )
        prob.run_model()

        myFz = copy.copy(prob["tower.tower_Fz"])

        prob["rna_mass"] = 1e4
        prob.run_model()
        myFz[3:] -= 1e4 * g
        npt.assert_almost_equal(prob["tower.tower_Fz"], myFz)

        prob["transition_piece_mass"] = 1e2
        prob.run_model()
        myFz[3:6] -= 1e2 * g
        npt.assert_almost_equal(prob["tower.tower_Fz"], myFz)

        prob["gravity_foundation_mass"] = 1e3
        prob.run_model()
        # myFz[0] -= 1e3*g
        npt.assert_almost_equal(prob["tower.tower_Fz"], myFz)

    def test15MWmode_shapes(self):
        # --- geometry ----
        h_param = np.array(
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 14.1679]
        )
        d_param = np.array(
            [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                9.92647687,
                9.44319282,
                8.83283769,
                8.15148167,
                7.38976138,
                6.90908962,
                6.74803581,
                6.57231775,
                6.5,
            ]
        )
        t_param = np.array(
            [
                0.05534138,
                0.05534138,
                0.05344902,
                0.05150928,
                0.04952705,
                0.04751736,
                0.04551709,
                0.0435267,
                0.04224176,
                0.04105759,
                0.0394965,
                0.03645589,
                0.03377851,
                0.03219233,
                0.03070819,
                0.02910109,
                0.02721289,
                0.02400931,
                0.0208264,
                0.02399756,
            ]
        )

        self.modeling_options["WISDEM"]["TowerSE"]["n_height_tower"] = len(d_param)
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = len(d_param)

        prob = om.Problem()
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        # Set common and then customized parameters
        prob["hub_height"] = prob["wind_reference_height"] = 30 + 146.1679
        # prob["foundation_height"] = 0.0  # -30.0

        prob["tower_s"] = np.cumsum(np.r_[0.0, h_param]) / h_param.sum()
        prob["tower_foundation_height"] = 0.0  # 15.0
        prob["water_depth"] = 0.0  # 15.0
        prob["tower_height"] = h_param.sum()
        prob["tower_outer_diameter_in"] = d_param
        prob["tower_layer_thickness"] = t_param.reshape((1, len(t_param)))
        prob["tower_outfitting_factor"] = 1.0
        prob["tower_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 210e9 * np.ones((1, 3))
        prob["G_mat"] = 79.3e9 * np.ones((1, 3))
        prob["rho_mat"] = 7850.0
        prob["sigma_y_mat"] = 345e6

        prob["yaw"] = 0.0
        prob["transition_piece_mass"] = 0.0  # 100e3
        prob["transition_piece_cost"] = 0.0  # 100e3
        # prob['G_soil'] = 140e6
        # prob['nu_soil'] = 0.4
        prob["shearExp"] = 0.11
        prob["rho_air"] = 1.225
        prob["z0"] = 0.0
        prob["mu_air"] = 1.7934e-5
        prob["life"] = 20.0

        mIxx = 379640227.0
        mIyy = 224477294.0
        mIzz = 182971949.0
        mIxy = 0.0
        mIxz = -7259625.38
        mIyz = 0.0
        prob["rna_mass"] = 1007537.0
        prob["rna_I"] = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
        prob["rna_cg"] = np.array([-5.019, 0.0, 0.0])

        prob["wind.Uref"] = 0.0  # 20.00138038
        prob["pre.rna_F"] = np.zeros(3)  # np.array([3569257.70891496, -22787.83765441, -404483.54819059])
        prob["pre.rna_M"] = np.zeros(3)  # np.array([68746553.1515807, 16045443.38557568, 1811078.988995])

        # # --- run ---
        prob.run_model()
        """
        Natural Frequencies (Hz): [ 0.2161   0.21842  1.1091   1.167    1.2745   2.3611   2.5877   5.1233  5.2111   9.9725  10.007   10.151   16.388   16.4     18.092   21.813 23.955   23.958   30.184   33.706  ]

        Polynomial fit coefficients to modal displacements (x^2, x^3, x^4, x^5, x^6)
        1st Fore-aft    = [1.11422342, -2.73438505, 6.84397071, -5.97959674, 1.75578766]
        2nd Fore-aft    = [-48.86125831, 82.74454067, -156.79260263, 208.53125496, -84.62193469]
        1st Side-side   = [1.10492357, -2.71587869, 6.80247339, -5.93612744, 1.74460918]
        2nd Side-side   = [48.9719383, -89.25323746, 183.04839183, -226.34534799, 84.57825533]
        """

    def testExampleRegression(self):
        # --- geometry ----
        h_param = np.diff(np.array([0.0, 43.8, 87.6]))
        d_param = np.array([6.0, 4.935, 3.87])
        t_param = 1.3 * np.array([0.027, 0.023, 0.019])
        z_foundation = 0.0
        yaw = 0.0
        Koutfitting = 1.07

        # --- material props ---
        E = 210e9
        G = 80.8e9
        rho = 8500.0
        sigma_y = 450.0e6

        # --- extra mass ----
        m = np.array([285598.8])
        mIxx = 1.14930678e08
        mIyy = 2.20354030e07
        mIzz = 1.87597425e07
        mIxy = 0.0
        mIxz = 5.03710467e05
        mIyz = 0.0
        mI = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
        mrho = np.array([-1.13197635, 0.0, 0.50875268])
        # -----------

        # --- wind ---
        wind_zref = 90.0
        wind_z0 = 0.0
        shearExp = 0.2
        cd_usr = -1.0
        # ---------------

        # --- wave ---
        water_depth = 0.0
        soilG = 140e6
        soilnu = 0.4
        # ---------------

        # --- costs ---
        material_cost = 5.0
        labor_cost = 100.0 / 60.0
        painting_cost = 30.0
        # ---------------

        # two load cases.  TODO: use a case iterator

        # # --- loading case 1: max Thrust ---
        wind_Uref1 = 11.73732
        Fx1 = 1284744.19620519
        Fy1 = 0.0
        Fz1 = -2914124.84400512 + m * g
        Mxx1 = 3963732.76208099
        Myy1 = -2275104.79420872
        Mzz1 = -346781.68192839
        # # ---------------

        # # --- loading case 2: max wind speed ---
        wind_Uref2 = 70.0
        Fx2 = 930198.60063279
        Fy2 = 0.0
        Fz2 = -2883106.12368949 + m * g
        Mxx2 = -1683669.22411597
        Myy2 = -2522475.34625363
        Mzz2 = 147301.97023764
        # # ---------------

        # --- fatigue ---
        life = 20.0
        # ---------------

        self.modeling_options["WISDEM"]["TowerSE"]["n_height_tower"] = len(d_param)
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = len(d_param)
        self.modeling_options["WISDEM"]["TowerSE"]["nLC"] = 2
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_f"] = 1.35
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_m"] = 1.3
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_n"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_b"] = 1.1
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_fatigue"] = 1.35 * 1.3 * 1.0

        def fill_prob():
            prob = om.Problem()
            prob.model = tow.TowerSE(modeling_options=self.modeling_options)
            prob.setup()

            if self.modeling_options["WISDEM"]["TowerSE"]["wind"] == "PowerWind":
                prob["shearExp"] = shearExp

            # assign values to params

            # --- geometry ----
            prob["hub_height"] = h_param.sum()
            prob["water_depth"] = water_depth
            # prob['tower_section_height'] = h_param
            prob["tower_s"] = np.cumsum(np.r_[0.0, h_param]) / h_param.sum()
            prob["tower_foundation_height"] = z_foundation
            prob["tower_height"] = h_param.sum()
            prob["tower_outer_diameter_in"] = d_param
            # prob['tower_wall_thickness'] = t_param
            prob["tower_layer_thickness"] = t_param.reshape((1, len(t_param)))
            prob["tower_outfitting_factor"] = Koutfitting
            prob["tower_layer_materials"] = ["steel"]
            prob["material_names"] = ["steel"]
            prob["yaw"] = yaw
            # prob["G_soil"] = soilG
            # prob["nu_soil"] = soilnu
            # --- material props ---
            prob["E_mat"] = E * np.ones((1, 3))
            prob["G_mat"] = G * np.ones((1, 3))
            prob["rho_mat"] = rho
            prob["sigma_y_mat"] = sigma_y

            # --- extra mass ----
            prob["rna_mass"] = m
            prob["rna_I"] = mI
            prob["rna_cg"] = mrho
            # -----------

            # --- costs ---
            prob["unit_cost"] = material_cost
            prob["labor_cost_rate"] = labor_cost
            prob["painting_cost_rate"] = painting_cost
            # -----------

            # --- wind & wave ---
            prob["wind_reference_height"] = wind_zref
            prob["z0"] = wind_z0
            prob["cd_usr"] = cd_usr
            prob["rho_air"] = 1.225
            prob["mu_air"] = 1.7934e-5

            # --- fatigue ---
            prob["life"] = life
            # ---------------

            # # --- loading case 1: max Thrust ---
            prob["wind1.Uref"] = wind_Uref1

            prob["pre1.rna_F"] = np.r_[Fx1, Fy1, Fz1]
            prob["pre1.rna_M"] = np.r_[Mxx1, Myy1, Mzz1]
            # # ---------------

            # # --- loading case 2: max Wind Speed ---
            prob["wind2.Uref"] = wind_Uref2

            prob["pre2.rna_F"] = np.r_[Fx2, Fy2, Fz2]
            prob["pre2.rna_M"] = np.r_[Mxx2, Myy2, Mzz2]

            return prob

        # # --- run ---
        prob = fill_prob()
        prob.run_model()

        npt.assert_almost_equal(prob["z_full"], [0.0, 14.6, 29.2, 43.8, 58.4, 73.0, 87.6])
        npt.assert_almost_equal(prob["d_full"], [6.0, 5.645, 5.29, 4.935, 4.58, 4.225, 3.87])
        npt.assert_almost_equal(prob["t_full"], [0.0325, 0.0325, 0.0325, 0.0273, 0.0273, 0.0273])

        npt.assert_almost_equal(prob["tower_mass"], [370541.14008246])
        npt.assert_almost_equal(prob["tower_center_of_mass"], [38.78441074])
        npt.assert_almost_equal(prob["constr_d_to_t"], [168.23076923, 161.26373626])
        npt.assert_almost_equal(prob["constr_taper"], [0.8225, 0.78419453])
        npt.assert_almost_equal(prob["wind1.Uref"], [11.73732])
        npt.assert_almost_equal(prob["tower1.f1"], [0.33214436], 5)
        npt.assert_almost_equal(prob["tower1.top_deflection"], [0.6988131])
        npt.assert_almost_equal(
            prob["post1.constr_stress"], [0.3844339, 0.3436128, 0.2856628, 0.2421312, 0.1121663, 0.0623614]
        )
        npt.assert_almost_equal(
            prob["post1.constr_global_buckling"], [0.5170422, 0.4829785, 0.4351583, 0.4221748, 0.3168518, 0.2755187]
        )
        npt.assert_almost_equal(
            prob["post1.constr_shell_buckling"], [0.2371124, 0.1861889, 0.1282914, 0.1073705, 0.0295743, 0.0130323]
        )
        npt.assert_almost_equal(prob["wind2.Uref"], [70.0])
        npt.assert_almost_equal(prob["tower2.f1"], [0.33218936], 5)
        npt.assert_almost_equal(prob["tower2.top_deflection"], [0.6440434])
        npt.assert_almost_equal(
            prob["post2.constr_stress"], [0.3728837, 0.3137352, 0.2421504, 0.18487, 0.0662662, 0.0471034]
        )
        npt.assert_almost_equal(
            prob["post2.constr_global_buckling"], [0.5064959, 0.4570302, 0.3978452, 0.373074, 0.276448, 0.2668201]
        )
        npt.assert_almost_equal(
            prob["post2.constr_shell_buckling"], [0.2258567, 0.1599288, 0.0970113, 0.0700506, 0.0156922, 0.0100741]
        )
        npt.assert_almost_equal(prob["tower1.base_F"][0], 1300347.476206353, 2)  # 1.29980269e06, 2)
        npt.assert_array_less(np.abs(prob["tower1.base_F"][1]), 1e2, 2)
        npt.assert_almost_equal(prob["tower1.base_F"][2], -6.31005811e06, 2)
        npt.assert_almost_equal(prob["tower1.base_M"], [4.14775052e06, 1.10758024e08, -3.46827499e05], 0)
        npt.assert_almost_equal(prob["tower2.base_F"][0], 1617231.046083178, 2)
        npt.assert_array_less(np.abs(prob["tower2.base_F"][1]), 1e2, 2)
        npt.assert_almost_equal(prob["tower2.base_F"][2], -6.27903939e06, 2)
        npt.assert_almost_equal(prob["tower2.base_M"], [-1.76120197e06, 1.12569564e08, 1.47321336e05], 0)

        # Now regression on DNV-GL C202 methods
        self.modeling_options["WISDEM"]["TowerSE"]["buckling_method"] = "dnvgl"
        prob = fill_prob()
        prob.run_model()

        npt.assert_almost_equal(prob["z_full"], [0.0, 14.6, 29.2, 43.8, 58.4, 73.0, 87.6])
        npt.assert_almost_equal(prob["d_full"], [6.0, 5.645, 5.29, 4.935, 4.58, 4.225, 3.87])
        npt.assert_almost_equal(prob["t_full"], [0.0325, 0.0325, 0.0325, 0.0273, 0.0273, 0.0273])

        npt.assert_almost_equal(prob["tower_mass"], [370541.14008246])
        npt.assert_almost_equal(prob["tower_center_of_mass"], [38.78441074])
        npt.assert_almost_equal(prob["constr_d_to_t"], [168.23076923, 161.26373626])
        npt.assert_almost_equal(prob["constr_taper"], [0.8225, 0.78419453])
        npt.assert_almost_equal(prob["wind1.Uref"], [11.73732])
        npt.assert_almost_equal(prob["tower1.f1"], [0.33214436], 5)
        npt.assert_almost_equal(prob["tower1.top_deflection"], [0.6988131])
        npt.assert_almost_equal(
            prob["post1.constr_stress"], [0.3844339, 0.3436128, 0.2856628, 0.2421312, 0.1121663, 0.0623614]
        )
        npt.assert_almost_equal(
            prob["post1.constr_global_buckling"], [0.6274373, 0.5691916, 0.4884754, 0.454831, 0.2769742, 0.2022617]
        )
        npt.assert_almost_equal(
            prob["post1.constr_shell_buckling"], [0.0357574, 0.0318343, 0.0281723, 0.0347622, 0.0310088, 0.0276982]
        )
        npt.assert_almost_equal(prob["wind2.Uref"], [70.0])
        npt.assert_almost_equal(prob["tower2.f1"], [0.33218936], 5)
        npt.assert_almost_equal(prob["tower2.top_deflection"], [0.6440434])
        npt.assert_almost_equal(
            prob["post2.constr_stress"], [0.3728837, 0.3137352, 0.2421504, 0.18487, 0.0662662, 0.0471034]
        )
        npt.assert_almost_equal(
            prob["post2.constr_global_buckling"], [0.616188, 0.532396, 0.4323648, 0.378272, 0.2126512, 0.1896404]
        )
        npt.assert_almost_equal(
            prob["post2.constr_shell_buckling"], [0.0393843, 0.0397039, 0.0368479, 0.0463734, 0.0428983, 0.0394461]
        )
        npt.assert_almost_equal(prob["tower1.base_F"][0], 1300347.476206353, 2)  # 1.29980269e06, 2)
        npt.assert_array_less(np.abs(prob["tower1.base_F"][1]), 1e2, 2)
        npt.assert_almost_equal(prob["tower1.base_F"][2], -6.31005811e06, 2)
        npt.assert_almost_equal(prob["tower1.base_M"], [4.14775052e06, 1.10758024e08, -3.46827499e05], 0)
        npt.assert_almost_equal(prob["tower2.base_F"][0], 1617231.046083178, 2)
        npt.assert_array_less(np.abs(prob["tower2.base_F"][1]), 1e2, 2)
        npt.assert_almost_equal(prob["tower2.base_F"][2], -6.27903939e06, 2)
        npt.assert_almost_equal(prob["tower2.base_M"], [-1.76120197e06, 1.12569564e08, 1.47321336e05], 0)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTowerSE))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
