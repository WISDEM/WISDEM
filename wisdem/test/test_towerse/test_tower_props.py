import unittest

import numpy as np
import numpy.testing as npt
import wisdem.towerse.tower_props as tow
from wisdem.commonse.utilities import nodal2sectional

npts = 100
myones = np.ones((npts,))


class TestDiscretization(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params["section_height"] = np.arange(1, 5)
        self.params["diameter"] = 5.0 * np.ones(5)
        self.params["wall_thickness"] = 0.05 * np.ones(4)
        self.params["foundation_height"] = 0.0

    def testRefine2(self):
        mydis = tow.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        npt.assert_array_equal(self.unknowns["z_param"], np.array([0.0, 1.0, 3.0, 6.0, 10.0]))
        npt.assert_array_equal(self.unknowns["z_full"], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]))
        npt.assert_array_equal(self.unknowns["d_full"], 5.0)
        npt.assert_array_equal(self.unknowns["t_full"], 0.05)

    def testFoundation(self):
        self.params["foundation_height"] = -30.0
        mydis = tow.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        npt.assert_array_equal(self.unknowns["z_param"], np.array([0.0, 1.0, 3.0, 6.0, 10.0]) - 30.0)
        npt.assert_array_equal(self.unknowns["z_full"], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]) - 30.0)
        npt.assert_array_equal(self.unknowns["d_full"], 5.0)
        npt.assert_array_equal(self.unknowns["t_full"], 0.05)

    def testRefine3(self):
        mydis = tow.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        for k in self.unknowns["z_param"]:
            self.assertIn(k, self.unknowns["z_full"])

    def testRefineInterp(self):
        self.params["diameter"] = np.array([5.0, 5.0, 6.0, 7.0, 7.0])
        self.params["wall_thickness"] = 1e-2 * np.array([5.0, 5.0, 6.0, 7.0])
        mydis = tow.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        npt.assert_array_equal(self.unknowns["z_param"], np.array([0.0, 1.0, 3.0, 6.0, 10.0]))
        npt.assert_array_equal(self.unknowns["z_full"], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]))
        npt.assert_array_equal(self.unknowns["d_full"], np.array([5.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.0, 7.0]))
        npt.assert_array_equal(self.unknowns["t_full"], 1e-2 * np.array([5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0]))


class TestMass(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params["d_full"] = 2.0 * 10.0 * myones
        self.params["t_full"] = 0.5 * np.ones((npts - 1,))
        self.params["z_full"] = np.linspace(0, 50.0, npts)
        self.params["rho"] = 5.0
        self.params["outfitting_factor"] = 1.5
        self.params["material_cost_rate"] = 1.5
        self.params["labor_cost_rate"] = 1.0
        self.params["painting_cost_rate"] = 10.0

        self.cm = tow.CylinderMass(nPoints=npts)

    def testRegular(self):
        # Straight column
        self.cm.compute(self.params, self.unknowns)

        expect = np.pi * (10.0 ** 2 - 9.5 ** 2) * 5.0 * 1.5 * (50.0 / (npts - 1))
        m = expect * (npts - 1)
        Iax = 0.5 * m * (10.0 ** 2 + 9.5 ** 2)
        Ix = (1 / 12.0) * m * (3 * (10.0 ** 2 + 9.5 ** 2) + 50 * 50) + m * 25 * 25  # parallel axis on last term
        z_avg, _ = nodal2sectional(self.params["z_full"])
        self.assertAlmostEqual(self.unknowns["mass"].sum(), m)
        npt.assert_almost_equal(self.unknowns["mass"], expect)
        npt.assert_almost_equal(self.unknowns["section_center_of_mass"], z_avg)
        self.assertAlmostEqual(self.unknowns["center_of_mass"], 25.0)
        npt.assert_almost_equal(self.unknowns["I_base"], [Ix, Ix, Iax, 0.0, 0.0, 0.0], decimal=5)

        """
    def testFrustum(self):
        # Frustum shell
        self.params['t_full'] = np.array([0.5, 0.4, 0.3])
        self.params['d_full'] = 2*np.array([10.0, 8.0, 6.0])
        self.wave.compute(self.params, self.unknowns)

        expect = np.pi/3.0*5.0*1.5*np.array([20.0, 30.0])*np.array([9.75*1.4+7.8*1.3, 7.8*1.1+5.85*1.0])
        m = expect*(npts-1)
        self.assertAlmostEqual(self.unknowns['mass'].sum(), m)
        npt.assert_almost_equal(self.unknowns['mass'].sum(), expect)
        """


class TestProps(unittest.TestCase):
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

    def testDiscYAML_Land_1Material(self):

        # Test land based, 1 material
        self.inputs["water_depth"] = 0.0
        self.inputs["tower_s"] = np.linspace(0, 1, 5)
        self.inputs["tower_layer_thickness"] = 0.25 * np.ones((1, 5))
        self.inputs["tower_foundation_height"] = 0.0
        self.inputs["tower_height"] = 1e2
        self.inputs["tower_outer_diameter_in"] = 8 * np.ones(5)
        self.inputs["tower_outfitting_factor"] = 1.1
        self.discrete_inputs["tower_layer_materials"] = ["steel"]
        self.inputs["monopile_s"] = np.empty(0)
        self.inputs["monopile_layer_thickness"] = np.empty((0, 0))
        self.inputs["monopile_foundation_height"] = 0.0
        self.inputs["monopile_height"] = 0.0
        self.inputs["monopile_outer_diameter_in"] = np.empty(0)
        self.inputs["monopile_outfitting_factor"] = 0.0
        self.discrete_inputs["monopile_layer_materials"] = [""]
        self.inputs["E_mat"] = 1e9 * np.ones((1, 3))
        self.inputs["G_mat"] = 1e8 * np.ones((1, 3))
        self.inputs["sigma_y_mat"] = np.array([1e7])
        self.inputs["rho_mat"] = np.array([1e4])
        self.inputs["unit_cost_mat"] = np.array([1e1])
        self.discrete_inputs["material_names"] = ["steel"]
        myobj = tow.DiscretizationYAML(
            n_height_tower=5, n_height_monopile=0, n_layers_tower=1, n_layers_monopile=0, n_mat=1
        )
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["tower_section_height"], 25.0 * np.ones(4))
        npt.assert_equal(self.outputs["tower_outer_diameter"], self.inputs["tower_outer_diameter_in"])
        npt.assert_equal(self.outputs["tower_wall_thickness"], 0.25 * np.ones(4))
        npt.assert_equal(self.outputs["outfitting_factor"], 1.1 * np.ones(4))
        npt.assert_equal(self.outputs["E"], 1e9 * np.ones(4))
        npt.assert_equal(self.outputs["G"], 1e8 * np.ones(4))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.ones(4))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.ones(4))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.ones(4))
        npt.assert_equal(self.outputs["z_start"], 0.0)
        npt.assert_equal(self.outputs["transition_piece_height"], 0.0)
        npt.assert_equal(self.outputs["suctionpile_depth"], 0.0)

    def testDiscYAML_Land_2Materials(self):
        # Test land based, 2 materials
        self.inputs["water_depth"] = 0.0
        self.inputs["tower_s"] = np.linspace(0, 1, 5)
        self.inputs["tower_layer_thickness"] = np.array([[0.2, 0.2, 0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.1]])
        self.inputs["tower_foundation_height"] = 0.0
        self.inputs["tower_height"] = 1e2
        self.inputs["tower_outer_diameter_in"] = 8 * np.ones(5)
        self.inputs["tower_outfitting_factor"] = 1.1
        self.discrete_inputs["tower_layer_materials"] = ["steel", "other"]
        self.inputs["monopile_s"] = np.empty(0)
        self.inputs["monopile_layer_thickness"] = np.empty((0, 0))
        self.inputs["monopile_foundation_height"] = 0.0
        self.inputs["monopile_height"] = 0.0
        self.inputs["monopile_outer_diameter_in"] = np.empty(0)
        self.inputs["monopile_outfitting_factor"] = 0.0
        self.discrete_inputs["monopile_layer_materials"] = [""]
        self.inputs["E_mat"] = 1e9 * np.vstack((np.ones((1, 3)), 2 * np.ones((1, 3))))
        self.inputs["G_mat"] = 1e8 * np.vstack((np.ones((1, 3)), 2 * np.ones((1, 3))))
        self.inputs["sigma_y_mat"] = np.array([1e7, 2e7])
        self.inputs["rho_mat"] = np.array([1e4, 2e4])
        self.inputs["unit_cost_mat"] = np.array([1e1, 2e1])
        self.discrete_inputs["material_names"] = ["steel", "other"]
        myobj = tow.DiscretizationYAML(
            n_height_tower=5, n_height_monopile=0, n_layers_tower=1, n_layers_monopile=0, n_mat=2
        )
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # Define mixtures
        v = np.r_[np.mean([0.2, 0]), np.mean([0.1, 0.0])]
        vv = v / v.sum()
        x = np.r_[1, 2]
        xx1 = np.sum(x * vv)  # Mass weighted
        xx2 = 0.5 * np.sum(vv * x) + 0.5 / np.sum(vv / x)  # Volumetric method
        xx3 = np.sum(x * x * vv) / xx1  # Mass-cost weighted
        npt.assert_equal(self.outputs["tower_section_height"], 25.0 * np.ones(4))
        npt.assert_equal(self.outputs["tower_outer_diameter"], self.inputs["tower_outer_diameter_in"])
        npt.assert_almost_equal(self.outputs["tower_wall_thickness"], np.array([0.2, 0.2, v.sum(), 0.1]))
        npt.assert_equal(self.outputs["outfitting_factor"], 1.1 * np.ones(4))
        npt.assert_almost_equal(self.outputs["E"], 1e9 * np.array([1, 1, xx2, 2]))
        npt.assert_almost_equal(self.outputs["G"], 1e8 * np.array([1, 1, xx2, 2]))
        npt.assert_almost_equal(self.outputs["sigma_y"], 1e7 * np.array([1, 1, xx2, 2]))
        npt.assert_almost_equal(self.outputs["rho"], 1e4 * np.array([1, 1, xx1, 2]))
        npt.assert_almost_equal(self.outputs["unit_cost"], 1e1 * np.array([1, 1, xx3, 2]))
        npt.assert_equal(self.outputs["z_start"], 0.0)
        npt.assert_equal(self.outputs["transition_piece_height"], 0.0)
        npt.assert_equal(self.outputs["suctionpile_depth"], 0.0)

    def testDiscYAML_Monopile_1Material(self):
        self.inputs["water_depth"] = 30.0
        self.inputs["tower_s"] = np.linspace(0, 1, 5)
        self.inputs["tower_layer_thickness"] = 0.25 * np.ones((1, 5))
        self.inputs["tower_height"] = 1e2
        self.inputs["tower_foundation_height"] = 10.0
        self.inputs["tower_outer_diameter_in"] = 8 * np.ones(5)
        self.inputs["tower_outfitting_factor"] = 1.1
        self.discrete_inputs["tower_layer_materials"] = ["steel"]
        self.inputs["monopile_s"] = np.linspace(0, 1, 5)
        self.inputs["monopile_layer_thickness"] = 0.5 * np.ones((1, 5))
        self.inputs["monopile_foundation_height"] = -40.0
        self.inputs["monopile_height"] = 50.0
        self.inputs["monopile_outer_diameter_in"] = 10 * np.ones(5)
        self.inputs["monopile_outer_diameter_in"][-1] = 8
        self.inputs["monopile_outfitting_factor"] = 1.2
        self.discrete_inputs["monopile_layer_materials"] = ["steel"]
        self.inputs["E_mat"] = 1e9 * np.ones((1, 3))
        self.inputs["G_mat"] = 1e8 * np.ones((1, 3))
        self.inputs["sigma_y_mat"] = np.array([1e7])
        self.inputs["rho_mat"] = np.array([1e4])
        self.inputs["unit_cost_mat"] = np.array([1e1])
        self.discrete_inputs["material_names"] = ["steel"]
        myobj = tow.DiscretizationYAML(
            n_height_tower=5, n_height_monopile=5, n_layers_tower=1, n_layers_monopile=1, n_mat=1
        )
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["tower_section_height"], np.r_[10.0, 15.0, 12.5 * np.ones(2), 25 * np.ones(4)])
        npt.assert_equal(self.outputs["tower_outer_diameter"], np.r_[10 * np.ones(4), 8 * np.ones(5)])
        npt.assert_equal(self.outputs["tower_wall_thickness"], np.r_[0.5 * np.ones(4), 0.25 * np.ones(4)])
        npt.assert_equal(self.outputs["outfitting_factor"], np.r_[1.2 * np.ones(4), 1.1 * np.ones(4)])
        npt.assert_equal(self.outputs["E"], 1e9 * np.ones(8))
        npt.assert_equal(self.outputs["G"], 1e8 * np.ones(8))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.ones(8))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.ones(8))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.ones(8))
        npt.assert_equal(self.outputs["z_start"], -40.0)
        npt.assert_equal(self.outputs["transition_piece_height"], 10.0)
        npt.assert_equal(self.outputs["suctionpile_depth"], 10.0)

    def testDiscYAML_Monopile_PileShort(self):
        self.inputs["water_depth"] = 60.0
        self.inputs["tower_s"] = np.linspace(0, 1, 5)
        self.inputs["tower_layer_thickness"] = 0.25 * np.ones((1, 5))
        self.inputs["tower_height"] = 1e2
        self.inputs["tower_foundation_height"] = 10.0
        self.inputs["tower_outer_diameter_in"] = 8 * np.ones(5)
        self.inputs["tower_outfitting_factor"] = 1.1
        self.discrete_inputs["tower_layer_materials"] = ["steel"]
        self.inputs["monopile_s"] = np.linspace(0, 1, 5)
        self.inputs["monopile_layer_thickness"] = 0.5 * np.ones((1, 5))
        self.inputs["monopile_foundation_height"] = -40.0
        self.inputs["monopile_height"] = 50.0
        self.inputs["monopile_outer_diameter_in"] = 10 * np.ones(5)
        self.inputs["monopile_outer_diameter_in"][-1] = 8
        self.inputs["monopile_outfitting_factor"] = 1.2
        self.discrete_inputs["monopile_layer_materials"] = ["steel"]
        self.inputs["E_mat"] = 1e9 * np.ones((1, 3))
        self.inputs["G_mat"] = 1e8 * np.ones((1, 3))
        self.inputs["sigma_y_mat"] = np.array([1e7])
        self.inputs["rho_mat"] = np.array([1e4])
        self.inputs["unit_cost_mat"] = np.array([1e1])
        self.discrete_inputs["material_names"] = ["steel"]
        myobj = tow.DiscretizationYAML(
            n_height_tower=5, n_height_monopile=5, n_layers_tower=1, n_layers_monopile=1, n_mat=1
        )
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["tower_section_height"], np.r_[12.5 * np.ones(4), 25 * np.ones(4)])
        npt.assert_equal(self.outputs["tower_outer_diameter"], np.r_[10 * np.ones(4), 8 * np.ones(5)])
        npt.assert_equal(self.outputs["tower_wall_thickness"], np.r_[0.5 * np.ones(4), 0.25 * np.ones(4)])
        npt.assert_equal(self.outputs["outfitting_factor"], np.r_[1.2 * np.ones(4), 1.1 * np.ones(4)])
        npt.assert_equal(self.outputs["E"], 1e9 * np.ones(8))
        npt.assert_equal(self.outputs["G"], 1e8 * np.ones(8))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.ones(8))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.ones(8))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.ones(8))
        npt.assert_equal(self.outputs["z_start"], -40.0)
        npt.assert_equal(self.outputs["transition_piece_height"], 10.0)
        npt.assert_equal(self.outputs["suctionpile_depth"], -20.0)

    def testDiscYAML_Monopile_RedoPileNodes(self):
        self.inputs["water_depth"] = 30.0
        self.inputs["tower_s"] = np.linspace(0, 1, 5)
        self.inputs["tower_layer_thickness"] = 0.25 * np.ones((1, 5))
        self.inputs["tower_height"] = 1e2
        self.inputs["tower_foundation_height"] = 10.0
        self.inputs["tower_outer_diameter_in"] = 8 * np.ones(5)
        self.inputs["tower_outfitting_factor"] = 1.1
        self.discrete_inputs["tower_layer_materials"] = ["steel"]
        self.inputs["monopile_s"] = np.linspace(0, 1, 20)
        self.inputs["monopile_layer_thickness"] = 0.5 * np.ones((1, 20))
        self.inputs["monopile_foundation_height"] = -40.0
        self.inputs["monopile_height"] = 50.0
        self.inputs["monopile_outer_diameter_in"] = 10 * np.ones(20)
        self.inputs["monopile_outer_diameter_in"][-1] = 8
        self.inputs["monopile_outfitting_factor"] = 1.2
        self.discrete_inputs["monopile_layer_materials"] = ["steel"]
        self.inputs["E_mat"] = 1e9 * np.ones((1, 3))
        self.inputs["G_mat"] = 1e8 * np.ones((1, 3))
        self.inputs["sigma_y_mat"] = np.array([1e7])
        self.inputs["rho_mat"] = np.array([1e4])
        self.inputs["unit_cost_mat"] = np.array([1e1])
        self.discrete_inputs["material_names"] = ["steel"]
        myobj = tow.DiscretizationYAML(
            n_height_tower=5, n_height_monopile=20, n_layers_tower=1, n_layers_monopile=1, n_mat=1
        )
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["tower_section_height"][0], 10.0)
        # npt.assert_almost_equal(self.outputs["tower_section_height"][1:4], (2.63157895*4-10)/4)
        npt.assert_almost_equal(
            self.outputs["tower_section_height"][4:], np.r_[2.63157895 * np.ones(15), 25 * np.ones(4)]
        )
        npt.assert_equal(self.outputs["tower_outer_diameter"], np.r_[10 * np.ones(19), 8 * np.ones(5)])
        npt.assert_equal(self.outputs["tower_wall_thickness"], np.r_[0.5 * np.ones(19), 0.25 * np.ones(4)])
        npt.assert_equal(self.outputs["outfitting_factor"], np.r_[1.2 * np.ones(19), 1.1 * np.ones(4)])
        npt.assert_equal(self.outputs["E"], 1e9 * np.ones(23))
        npt.assert_equal(self.outputs["G"], 1e8 * np.ones(23))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.ones(23))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.ones(23))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.ones(23))
        npt.assert_equal(self.outputs["z_start"], -40.0)
        npt.assert_equal(self.outputs["transition_piece_height"], 10.0)
        npt.assert_equal(self.outputs["suctionpile_depth"], 10.0)

    def testDiscYAML_Monopile_DifferentMaterials(self):
        self.inputs["water_depth"] = 30.0
        self.inputs["tower_s"] = np.linspace(0, 1, 5)
        self.inputs["tower_layer_thickness"] = 0.25 * np.ones((1, 5))
        self.inputs["tower_foundation_height"] = 10.0
        self.inputs["tower_height"] = 1e2
        self.inputs["tower_outer_diameter_in"] = 8 * np.ones(5)
        self.inputs["tower_outfitting_factor"] = 1.1
        self.discrete_inputs["tower_layer_materials"] = ["steel"]
        self.inputs["monopile_s"] = np.linspace(0, 1, 5)
        self.inputs["monopile_layer_thickness"] = 0.5 * np.ones((1, 5))
        self.inputs["monopile_foundation_height"] = -40.0
        self.inputs["monopile_height"] = 50.0
        self.inputs["monopile_outer_diameter_in"] = 10 * np.ones(5)
        self.inputs["monopile_outer_diameter_in"][-1] = 8
        self.inputs["monopile_outfitting_factor"] = 1.2
        self.discrete_inputs["monopile_layer_materials"] = ["other"]
        self.inputs["E_mat"] = 1e9 * np.vstack((np.ones((1, 3)), 2 * np.ones((1, 3))))
        self.inputs["G_mat"] = 1e8 * np.vstack((np.ones((1, 3)), 2 * np.ones((1, 3))))
        self.inputs["sigma_y_mat"] = np.array([1e7, 2e7])
        self.inputs["rho_mat"] = np.array([1e4, 2e4])
        self.inputs["unit_cost_mat"] = np.array([1e1, 2e1])
        self.discrete_inputs["material_names"] = ["steel", "other"]
        myobj = tow.DiscretizationYAML(
            n_height_tower=5, n_height_monopile=5, n_layers_tower=1, n_layers_monopile=1, n_mat=2
        )
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["tower_section_height"], np.r_[10.0, 15.0, 12.5 * np.ones(2), 25 * np.ones(4)])
        npt.assert_equal(self.outputs["tower_outer_diameter"], np.r_[10 * np.ones(4), 8 * np.ones(5)])
        npt.assert_equal(self.outputs["tower_wall_thickness"], np.r_[0.5 * np.ones(4), 0.25 * np.ones(4)])
        npt.assert_equal(self.outputs["outfitting_factor"], np.r_[1.2 * np.ones(4), 1.1 * np.ones(4)])
        npt.assert_equal(self.outputs["E"], 1e9 * np.r_[2 * np.ones(4), np.ones(4)])
        npt.assert_equal(self.outputs["G"], 1e8 * np.r_[2 * np.ones(4), np.ones(4)])
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.r_[2 * np.ones(4), np.ones(4)])
        npt.assert_equal(self.outputs["rho"], 1e4 * np.r_[2 * np.ones(4), np.ones(4)])
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.r_[2 * np.ones(4), np.ones(4)])
        npt.assert_equal(self.outputs["z_start"], -40.0)
        npt.assert_equal(self.outputs["transition_piece_height"], 10.0)
        npt.assert_equal(self.outputs["suctionpile_depth"], 10.0)

    def testDiscYAML_Bad_Inputs(self):
        self.inputs["water_depth"] = 0.0
        self.inputs["tower_s"] = np.linspace(0, 1, 5)
        self.inputs["tower_layer_thickness"] = 0.25 * np.ones((1, 5))
        self.inputs["tower_foundation_height"] = 0.0
        self.inputs["tower_height"] = 1e2
        self.inputs["tower_outer_diameter_in"] = 8 * np.ones(5)
        self.inputs["tower_outfitting_factor"] = 1.1
        self.discrete_inputs["tower_layer_materials"] = ["steel"]
        self.inputs["monopile_s"] = np.empty(0)
        self.inputs["monopile_layer_thickness"] = np.empty((0, 0))
        self.inputs["monopile_foundation_height"] = 0.0
        self.inputs["monopile_height"] = 0.0
        self.inputs["monopile_outer_diameter_in"] = np.empty(0)
        self.inputs["monopile_outfitting_factor"] = 0.0
        self.discrete_inputs["monopile_layer_materials"] = [""]
        self.inputs["E_mat"] = 1e9 * np.ones((1, 3))
        self.inputs["G_mat"] = 1e8 * np.ones((1, 3))
        self.inputs["sigma_y_mat"] = np.array([1e7])
        self.inputs["rho_mat"] = np.array([1e4])
        self.inputs["unit_cost_mat"] = np.array([1e1])
        self.discrete_inputs["material_names"] = ["steel"]
        myobj = tow.DiscretizationYAML(
            n_height_tower=5, n_height_monopile=0, n_layers_tower=1, n_layers_monopile=0, n_mat=1
        )

        try:
            self.inputs["tower_layer_thickness"][0, -2:] = 0.0
            myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
            self.assertTrue(False)  # Shouldn't get here
        except ValueError:
            self.assertTrue(True)

        try:
            self.inputs["tower_layer_thickness"][0, -2:] = 0.25
            self.inputs["tower_outer_diameter_in"][-1] = -1.0
            myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
            self.assertTrue(False)  # Shouldn't get here
        except ValueError:
            self.assertTrue(True)

        try:
            self.inputs["tower_layer_thickness"][0, -2:] = 0.25
            self.inputs["tower_outer_diameter_in"][-1] = 8.0
            self.inputs["tower_s"][-1] = self.inputs["tower_s"][-2]
            myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
            self.assertTrue(False)  # Shouldn't get here
        except ValueError:
            self.assertTrue(True)

    def testTowerDisc(self):
        # Test Land
        self.inputs["hub_height"] = 100.0
        self.inputs["z_param"] = np.array([0.0, 40.0, 80.0])
        self.inputs["z_full"] = np.linspace(0.0, 80.0, 7)
        self.inputs["rho"] = 1e3 * np.ones(2)
        self.inputs["outfitting_factor"] = 1.1 * np.ones(2)
        self.inputs["unit_cost"] = 5.0 * np.ones(2)
        self.inputs["E"] = 6.0 * np.ones(2)
        self.inputs["G"] = 7.0 * np.ones(2)
        self.inputs["sigma_y"] = 8.0 * np.ones(2)
        self.inputs["Az"] = 9.0 * np.ones(6)
        self.inputs["Jz"] = 10.0 * np.ones(6)
        self.inputs["Ixx"] = 11.0 * np.ones(6)
        self.inputs["Iyy"] = 11.0 * np.ones(6)
        myobj = tow.TowerDiscretization(n_height=3)
        myobj.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs["height_constraint"], 20.0)
        npt.assert_equal(self.outputs["rho_full"], self.inputs["rho"][0] * np.ones(6))
        npt.assert_equal(self.outputs["E_full"], self.inputs["E"][0] * np.ones(6))
        npt.assert_equal(self.outputs["G_full"], self.inputs["G"][0] * np.ones(6))
        npt.assert_equal(self.outputs["sigma_y_full"], self.inputs["sigma_y"][0] * np.ones(6))
        npt.assert_equal(self.outputs["unit_cost_full"], self.inputs["unit_cost"][0] * np.ones(6))

        nout = 2
        npt.assert_almost_equal(self.outputs["sec_loc"], np.linspace(0, 1, nout))
        # npt.assert_equal(self.outputs["str_tw"], np.zeros(nout))
        # npt.assert_equal(self.outputs["tw_iner"], np.zeros(nout))
        npt.assert_equal(self.outputs["mass_den"], 1e3 * 9 * np.ones(nout))
        npt.assert_equal(self.outputs["foreaft_iner"], 1e3 * 11 * np.ones(nout))
        npt.assert_equal(self.outputs["sideside_iner"], 1e3 * 11 * np.ones(nout))
        npt.assert_equal(self.outputs["foreaft_stff"], 6 * 11 * np.ones(nout))
        npt.assert_equal(self.outputs["sideside_stff"], 6 * 11 * np.ones(nout))
        npt.assert_equal(self.outputs["tor_stff"], 7 * 10 * np.ones(nout))
        npt.assert_equal(self.outputs["axial_stff"], 6 * 9 * np.ones(nout))
        # npt.assert_equal(self.outputs["cg_offst"], np.zeros(nout))
        # npt.assert_equal(self.outputs["sc_offst"], np.zeros(nout))
        # npt.assert_equal(self.outputs["tc_offst"], np.zeros(nout))

    def testTowerMass(self):

        self.inputs["z_full"] = np.array([-50.0, -30, 0.0, 40.0, 80.0])
        self.inputs["d_full"] = 10 * np.ones(5)
        self.inputs["cylinder_mass"] = 1e3 * np.ones(4)
        self.inputs["cylinder_cost"] = 1e5
        self.inputs["cylinder_center_of_mass"] = 10.0
        self.inputs["cylinder_section_center_of_mass"] = self.inputs["z_full"][:-1] + 0.5 * np.diff(
            self.inputs["z_full"]
        )
        self.inputs["cylinder_I_base"] = 1e4 * np.r_[np.ones(3), np.zeros(3)]
        self.inputs["transition_piece_height"] = 20.0
        self.inputs["transition_piece_mass"] = 1e2
        self.inputs["transition_piece_cost"] = 1e3
        self.inputs["gravity_foundation_mass"] = 1e2

        myobj = tow.TowerMass(n_height=5)
        myobj.compute(self.inputs, self.outputs)

        self.assertEqual(self.outputs["structural_cost"], self.inputs["cylinder_cost"] + 1e3)
        npt.assert_equal(self.outputs["tower_I_base"], self.inputs["cylinder_I_base"])
        self.assertEqual(
            self.outputs["tower_center_of_mass"], (4 * 1e3 * 10.0 + 1e2 * 20.0 + 1e2 * -50.0) / (4 * 1e3 + 2e2)
        )
        npt.assert_equal(self.outputs["tower_section_center_of_mass"], self.inputs["cylinder_section_center_of_mass"])
        self.assertEqual(self.outputs["monopile_mass"], 1e3 * 2.5 + 2 * 1e2)
        self.assertEqual(self.outputs["monopile_cost"], self.inputs["cylinder_cost"] * 2.5 / 4.0 + 1e3)
        self.assertEqual(self.outputs["tower_mass"], 1e3 * (4 - 2.5))
        self.assertEqual(self.outputs["tower_cost"], self.inputs["cylinder_cost"] * 1.5 / 4.0)
        npt.assert_equal(self.outputs["transition_piece_I"], 1e2 * 25 * np.r_[0.5, 0.5, 1.0, np.zeros(3)])
        npt.assert_equal(self.outputs["gravity_foundation_I"], 0.5 * 1e2 * 25 * np.r_[0.5, 0.5, 1.0, np.zeros(3)])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDiscretization))
    suite.addTest(unittest.makeSuite(TestMass))
    suite.addTest(unittest.makeSuite(TestProps))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
