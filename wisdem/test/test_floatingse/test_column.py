import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.floatingse.column as column
from wisdem.commonse import gravity as g
from wisdem.commonse.utilities import nodal2sectional
from wisdem.commonse.vertical_cylinder import get_nfull

NHEIGHT = 6
NPTS = get_nfull(NHEIGHT)
myones = np.ones((NPTS,))
secones = np.ones((NPTS - 1,))


class TestInputs(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

    def testDiscYAML_1Material(self):

        # Test land based, 1 material
        self.inputs["s"] = np.linspace(0, 1, 5)
        self.inputs["layer_thickness"] = 0.25 * np.ones((1, 4))
        self.inputs["height"] = 1e2
        self.inputs["outer_diameter_in"] = 8 * np.ones(5)
        self.discrete_inputs["layer_materials"] = ["steel"]
        self.inputs["E_mat"] = 1e9 * np.ones((1, 3))
        self.inputs["G_mat"] = 1e8 * np.ones((1, 3))
        self.inputs["sigma_y_mat"] = np.array([1e7])
        self.inputs["rho_mat"] = np.array([1e4])
        self.inputs["unit_cost_mat"] = np.array([1e1])
        self.discrete_inputs["material_names"] = ["steel"]
        myobj = column.DiscretizationYAML(n_height=5, n_layers=1, n_mat=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["section_height"], 25.0 * np.ones(4))
        npt.assert_equal(self.outputs["outer_diameter"], self.inputs["outer_diameter_in"])
        npt.assert_equal(self.outputs["wall_thickness"], 0.25 * np.ones(4))
        npt.assert_equal(self.outputs["E"], 1e9 * np.ones(4))
        npt.assert_equal(self.outputs["G"], 1e8 * np.ones(4))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.ones(4))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.ones(4))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.ones(4))

    def testDiscYAML_2Materials(self):
        # Test land based, 2 materials
        self.inputs["s"] = np.linspace(0, 1, 5)
        self.inputs["layer_thickness"] = np.array([[0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.1, 0.1]])
        self.inputs["height"] = 1e2
        self.inputs["outer_diameter_in"] = 8 * np.ones(5)
        self.discrete_inputs["layer_materials"] = ["steel", "other"]
        self.inputs["E_mat"] = 1e9 * np.vstack((np.ones((1, 3)), 2 * np.ones((1, 3))))
        self.inputs["G_mat"] = 1e8 * np.vstack((np.ones((1, 3)), 2 * np.ones((1, 3))))
        self.inputs["sigma_y_mat"] = np.array([1e7, 2e7])
        self.inputs["rho_mat"] = np.array([1e4, 2e4])
        self.inputs["unit_cost_mat"] = np.array([1e1, 2e1])
        self.discrete_inputs["material_names"] = ["steel", "other"]
        myobj = column.DiscretizationYAML(n_height=5, n_layers=1, n_mat=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["section_height"], 25.0 * np.ones(4))
        npt.assert_equal(self.outputs["outer_diameter"], self.inputs["outer_diameter_in"])
        npt.assert_equal(self.outputs["wall_thickness"], np.array([0.25, 0.25, 0.1, 0.1]))
        npt.assert_equal(self.outputs["E"], 1e9 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["G"], 1e8 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.array([1, 1, 2, 2]))


class TestBulk(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        self.inputs["z_full"] = np.linspace(0, 1, NPTS)
        self.inputs["d_full"] = 10.0 * myones
        self.inputs["t_full"] = 0.05 * secones
        self.inputs["rho_full"] = 1e3 * secones
        self.inputs["bulkhead_locations"] = np.array([0.0, 1.0, 3.0, 5.0]) / 5.0
        nbulk = len(self.inputs["bulkhead_locations"])
        self.inputs["bulkhead_thickness"] = 0.05 * np.ones(nbulk)
        self.inputs["unit_cost_full"] = 1.0 * secones
        self.inputs["painting_cost_rate"] = 10.0
        self.inputs["labor_cost_rate"] = 2.0
        self.inputs["shell_mass"] = 500.0 * np.ones(NPTS - 1)

        self.bulk = column.BulkheadProperties(n_height=NHEIGHT, n_bulkhead=nbulk)

    def testAll(self):
        self.bulk.compute(self.inputs, self.outputs)

        R_i = 0.5 * 10 - 0.05
        m_bulk = np.pi * 1e3 * R_i ** 2 * 0.05
        expect = np.zeros(self.inputs["z_full"].size - 1)
        expect[[0, 3, 9, NPTS - 2]] = m_bulk
        ind = expect > 0.0
        npt.assert_almost_equal(self.outputs["bulkhead_mass"], expect)

        J0 = 0.50 * m_bulk * R_i ** 2
        I0 = 0.25 * m_bulk * R_i ** 2

        z_bulk = s_bulk = self.inputs["bulkhead_locations"]

        I = np.zeros(6)
        I[2] = 4.0 * J0
        I[0] = I0 + m_bulk * z_bulk[0] ** 2
        I[0] += I0 + m_bulk * z_bulk[1] ** 2
        I[0] += I0 + m_bulk * z_bulk[2] ** 2
        I[0] += I0 + m_bulk * z_bulk[3] ** 2
        I[1] = I[0]
        npt.assert_almost_equal(self.outputs["bulkhead_I_keel"], I)

        A = np.pi * R_i ** 2
        Kp_exp = 10.0 * 2 * A * ind.sum()
        self.inputs["painting_cost_rate"] = 10.0
        self.inputs["unit_cost_full"] = 1.0 * secones
        self.inputs["labor_cost_rate"] = 0.0
        self.bulk.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs["bulkhead_cost"], Kp_exp + m_bulk * ind.sum())

        self.inputs["painting_cost_rate"] = 0.0
        self.inputs["unit_cost_full"] = 0.0 * secones
        self.inputs["labor_cost_rate"] = 1.0
        self.bulk.compute(self.inputs, self.outputs)
        self.assertGreater(self.outputs["bulkhead_cost"], 2e3)


class TestBuoyancyTank(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        self.inputs["d_full"] = 10.0 * myones
        self.inputs["z_full"] = np.linspace(0, 1, NPTS) - 0.5
        self.inputs["rho_full"] = 1e3 * secones

        self.inputs["buoyancy_tank_diameter"] = 12.0
        self.inputs["buoyancy_tank_height"] = 0.25
        self.inputs["buoyancy_tank_location"] = 0.0
        self.inputs["unit_cost_full"] = 1.0 * secones
        self.inputs["labor_cost_rate"] = 2.0
        self.inputs["painting_cost_rate"] = 10.0
        self.inputs["shell_mass"] = 500.0 * np.ones(NPTS - 1)

        self.box = column.BuoyancyTankProperties(n_height=NHEIGHT)

    def testNormal(self):
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6 * 6 - 5 * 5)
        V_box = A_box * 0.25
        A_box = 2 * A_box + 0.25 * 2 * np.pi * 6
        m_expect = A_box * (6.0 / 50.0) * 1e3
        self.assertEqual(self.outputs["buoyancy_tank_mass"], m_expect)
        self.assertEqual(self.outputs["buoyancy_tank_cg"], -0.5 + 0.5 * 0.25)
        self.assertAlmostEqual(self.outputs["buoyancy_tank_displacement"], V_box)
        # self.assertEqual(self.outputs['buoyancy_tank_I_keel'], 0.0)

        self.inputs["unit_cost_full"] = 1.0 * secones
        self.inputs["labor_cost_rate"] = 0.0
        self.inputs["painting_cost_rate"] = 10.0
        self.box.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs["buoyancy_tank_cost"], m_expect + 10 * 2 * 1.5 * A_box)

        self.inputs["unit_cost_full"] = 0.0 * secones
        self.inputs["labor_cost_rate"] = 1.0
        self.inputs["painting_cost_rate"] = 0.0
        self.box.compute(self.inputs, self.outputs)
        self.assertGreater(self.outputs["buoyancy_tank_cost"], 1e3)

    def testTopAbove(self):
        self.inputs["buoyancy_tank_height"] = 0.75
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6 * 6 - 5 * 5)
        V_box = np.pi * (6 * 6 - 5 * 5) * 0.5
        m_expect = (2 * A_box + 0.75 * 2 * np.pi * 6) * (6.0 / 50.0) * 1e3
        self.assertAlmostEqual(self.outputs["buoyancy_tank_mass"], m_expect)
        self.assertAlmostEqual(self.outputs["buoyancy_tank_cg"], -0.5 + 0.5 * 0.75)
        self.assertAlmostEqual(self.outputs["buoyancy_tank_displacement"], V_box)

    def testBottomAbove(self):
        self.inputs["buoyancy_tank_location"] = 0.6
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6 * 6 - 5 * 5)
        V_box = np.pi * (6 * 6 - 5 * 5) * 0.0
        m_expect = (2 * A_box + 0.25 * 2 * np.pi * 6) * (6.0 / 50.0) * 1e3
        self.assertAlmostEqual(self.outputs["buoyancy_tank_mass"], m_expect)
        self.assertAlmostEqual(self.outputs["buoyancy_tank_cg"], 0.1 + 0.5 * 0.25)
        self.assertAlmostEqual(self.outputs["buoyancy_tank_displacement"], V_box)

    def testTooNarrow(self):
        self.inputs["buoyancy_tank_diameter"] = 8.0
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6 * 6 - 5 * 5)
        V_box = np.pi * (6 * 6 - 5 * 5) * 0.0
        m_expect = (2 * A_box + 0.25 * 2 * np.pi * 6) * (6.0 / 50.0) * 1e3
        self.assertAlmostEqual(self.outputs["buoyancy_tank_mass"], 0.0)
        self.assertEqual(self.outputs["buoyancy_tank_cg"], -0.5 + 0.5 * 0.25)
        self.assertAlmostEqual(self.outputs["buoyancy_tank_displacement"], 0.0)


class TestStiff(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        resid = None

        inputs["t_web"] = 0.5 * secones
        inputs["t_flange"] = 0.3 * secones
        inputs["h_web"] = 1.0 * secones
        inputs["w_flange"] = 2.0 * secones
        inputs["L_stiffener"] = 0.1 * secones
        inputs["L_stiffener"][int(NPTS / 2) :] = 0.05
        inputs["rho_full"] = 1e3 * secones
        inputs["unit_cost_full"] = 1.0 * secones
        inputs["labor_cost_rate"] = 2.0
        inputs["painting_cost_rate"] = 10.0
        inputs["shell_mass"] = 500.0 * np.ones(NPTS - 1)

        inputs["t_full"] = 0.5 * secones
        inputs["d_full"] = 2 * 10.0 * myones
        inputs["d_full"][1::2] = 2 * 8.0
        inputs["z_full"] = np.linspace(0, 1, NPTS) - 0.5
        inputs["z_param"] = np.linspace(0, 1, NHEIGHT) - 0.5

        stiff = column.StiffenerProperties(n_height=NHEIGHT)

        stiff.compute(inputs, outputs)

        Rwo = 9 - 0.5
        Rwi = Rwo - 1.0
        Rfi = Rwi - 0.3
        V1 = np.pi * (Rwo ** 2 - Rwi ** 2) * 0.5
        V2 = np.pi * (Rwi ** 2 - Rfi ** 2) * 2.0
        V = V1 + V2
        expect = V * 1e3
        actual = outputs["stiffener_mass"]

        # Test Mass
        self.assertAlmostEqual(actual.sum(), expect * (0.5 / 0.1 + 0.5 / 0.05))

        # Test cost
        A = 2 * (np.pi * (Rwo ** 2 - Rwi ** 2) + 2 * np.pi * 0.5 * (Rfi + Rwi) * (0.3 + 2)) - 2 * np.pi * Rwi * 0.5
        inputs["unit_cost_full"] = 1.0 * secones
        inputs["labor_cost_rate"] = 0.0
        inputs["painting_cost_rate"] = 10.0
        stiff.compute(inputs, outputs)
        self.assertAlmostEqual(outputs["stiffener_cost"], (expect + 10 * 2 * A) * (0.5 / 0.1 + 0.5 / 0.05))

        inputs["unit_cost_full"] = 0.0 * secones
        inputs["labor_cost_rate"] = 1.0
        inputs["painting_cost_rate"] = 0.0
        stiff.compute(inputs, outputs)
        self.assertGreater(outputs["stiffener_cost"], 1e3)

        # Test moment
        inputs["L_stiffener"] = 1.2 * secones
        stiff.compute(inputs, outputs)
        I_web = column.I_tube(Rwi, Rwo, 0.5, V1 * 1e3)
        I_fl = column.I_tube(Rfi, Rwi, 2.0, V2 * 1e3)
        I_sec = I_web + I_fl
        z_sec = 0.6 + 1e-6

        I = np.zeros(6)
        I[2] = I_sec[0, 2]
        I[0] += I_sec[0, 0] + expect * z_sec ** 2.0
        I[1] = I[0]

        npt.assert_almost_equal(outputs["stiffener_I_keel"], I)
        npt.assert_equal(outputs["flange_spacing_ratio"], 2 * 2.0 / 1.2)
        npt.assert_equal(outputs["stiffener_radius_ratio"], 1.8 / 9.0)


class TestBallast(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        resid = None
        inputs["t_full"] = 0.5 * secones
        inputs["d_full"] = 2 * 10.0 * myones
        inputs["z_full"] = np.linspace(0, 1, NPTS) - 0.5
        inputs["permanent_ballast_height"] = 1.0
        inputs["permanent_ballast_density"] = 2e3
        inputs["ballast_cost_rate"] = 10.0
        inputs["rho_water"] = 1e3

        ball = column.BallastProperties(n_height=NHEIGHT)

        ball.compute(inputs, outputs)

        area = np.pi * 9.5 ** 2
        m_perm = area * 1.0 * 2e3
        cg_perm = inputs["z_full"][0] + 0.5

        I_perm = np.zeros(6)
        I_perm[2] = 0.5 * m_perm * 9.5 ** 2
        I_perm[0] = m_perm * (3 * 9.5 ** 2 + 1.0 ** 2) / 12.0 + m_perm * 0.5 ** 2
        I_perm[1] = I_perm[0]

        # Unused!
        h_expect = 1e6 / area / 1000.0
        m_expect = m_perm + 1e6
        cg_water = inputs["z_full"][0] + 1.0 + 0.5 * h_expect
        cg_expect = (m_perm * cg_perm + 1e6 * cg_water) / m_expect

        self.assertAlmostEqual(outputs["ballast_mass"].sum(), m_perm)
        self.assertAlmostEqual(outputs["ballast_z_cg"], cg_perm)
        npt.assert_almost_equal(outputs["ballast_I_keel"], I_perm)


class TestGeometry(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        resid = None

        this_nheight = 3
        this_npts = column.get_nfull(this_nheight)
        this_ones = np.ones(this_npts - 1)

        inputs["z_full_in"] = np.linspace(0, 50.0, this_npts)
        inputs["z_param_in"] = np.array([0.0, 20.0, 50.0])
        inputs["section_height"] = np.array([20.0, 30.0])
        inputs["freeboard"] = 15.0
        inputs["water_depth"] = 100.0
        inputs["stiffener_web_thickness"] = np.array([0.5, 0.5])
        inputs["stiffener_flange_thickness"] = np.array([0.3, 0.3])
        inputs["stiffener_web_height"] = np.array([1.0, 1.0])
        inputs["stiffener_flange_width"] = np.array([2.0, 2.0])
        inputs["stiffener_spacing"] = np.array([0.1, 0.1])
        inputs["Hsig_wave"] = 5.0
        inputs["max_draft"] = 70.0
        inputs["unit_cost"] = 1.0 * np.ones(2)
        inputs["E"] = 2e9 * np.ones(2)
        inputs["G"] = 2e7 * np.ones(2)
        inputs["rho"] = 7850 * np.ones(2)
        inputs["sigma_y"] = 3e9 * np.ones(2)

        geom = column.ColumnGeometry(n_height=this_nheight)

        geom.compute(inputs, outputs)
        self.assertEqual(outputs["draft"], 35.0)
        self.assertEqual(outputs["draft"], np.sum(inputs["section_height"]) - inputs["freeboard"])
        self.assertEqual(outputs["draft"], -1 * outputs["z_full"][0])
        self.assertEqual(outputs["draft"], -1 * outputs["z_param"][0])
        self.assertEqual(outputs["draft_margin"], 0.5)
        npt.assert_equal(outputs["z_param"], np.array([-35.0, -15.0, 15.0]))
        npt.assert_equal(outputs["z_full"], inputs["z_full_in"] - 35)
        npt.assert_equal(outputs["t_web"], 0.5 * this_ones)
        npt.assert_equal(outputs["t_flange"], 0.3 * this_ones)
        npt.assert_equal(outputs["h_web"], 1.0 * this_ones)
        npt.assert_equal(outputs["w_flange"], 2.0 * this_ones)
        npt.assert_equal(outputs["L_stiffener"], 0.1 * this_ones)


class TestProperties(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        # For Geometry call
        this_nheight = 3
        this_npts = column.get_nfull(this_nheight)
        this_sec = np.ones(this_npts - 1)
        this_ones = np.ones(this_npts)
        self.inputs["z_full_in"] = np.linspace(0, 50.0, this_npts)
        self.inputs["z_section"], _ = nodal2sectional(self.inputs["z_full_in"])

        self.inputs["z_param_in"] = np.array([0.0, 20.0, 50.0])
        self.inputs["section_height"] = np.array([20.0, 30.0])
        self.inputs["freeboard"] = 15.0
        self.inputs["fairlead"] = 10.0
        self.inputs["water_depth"] = 100.0
        self.inputs["Hsig_wave"] = 5.0
        self.inputs["max_draft"] = 70.0

        self.inputs["t_full"] = 0.5 * this_sec
        self.inputs["d_full"] = 2 * 10.0 * this_ones

        self.inputs["stack_mass_in"] = 0.0

        self.inputs["shell_I_keel"] = 1e5 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs["stiffener_I_keel"] = 2e5 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs["bulkhead_I_keel"] = 3e5 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs["buoyancy_tank_I_keel"] = 5e6 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs["ballast_I_keel"] = 2e3 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        self.inputs["buoyancy_tank_diameter"] = 15.0

        self.inputs["rho_water"] = 1e3
        self.inputs["bulkhead_mass"] = 10.0 * this_sec
        self.inputs["bulkhead_z_cg"] = -10.0
        self.inputs["shell_mass"] = 500.0 * this_sec
        self.inputs["stiffener_mass"] = 100.0 * this_sec
        self.inputs["ballast_mass"] = 20.0 * this_sec
        self.inputs["ballast_z_cg"] = -35.0

        self.inputs["buoyancy_tank_mass"] = 20.0
        self.inputs["buoyancy_tank_cg"] = -15.0
        self.inputs["buoyancy_tank_location"] = 0.3
        self.inputs["buoyancy_tank_displacement"] = 300.0
        self.inputs["outfitting_factor"] = 1.05

        self.inputs["shell_cost"] = 1.0
        self.inputs["stiffener_cost"] = 2.0
        self.inputs["bulkhead_cost"] = 3.0
        self.inputs["buoyancy_tank_cost"] = 4.0
        self.inputs["ballast_cost"] = 5.0

        self.inputs["mooring_mass"] = 50.0
        self.inputs["mooring_vertical_load"] = 25.0
        self.inputs["mooring_restoring_force"] = 1e5
        self.inputs["mooring_cost"] = 1e4

        self.inputs["outfitting_cost_rate"] = 1.0

        self.inputs["unit_cost"] = 1.0 * np.ones(2)
        self.inputs["E"] = 2e9 * np.ones(2)
        self.inputs["G"] = 2e7 * np.ones(2)
        self.inputs["rho"] = 7850 * np.ones(2)
        self.inputs["sigma_y"] = 3e9 * np.ones(2)

        self.inputs["stiffener_web_thickness"] = np.array([0.5, 0.5])
        self.inputs["stiffener_flange_thickness"] = np.array([0.3, 0.3])
        self.inputs["stiffener_web_height"] = np.array([1.0, 1.0])
        self.inputs["stiffener_flange_width"] = np.array([2.0, 2.0])
        self.inputs["stiffener_spacing"] = np.array([0.1, 0.1])

        self.geom = column.ColumnGeometry(n_height=this_nheight)
        self.set_geometry()

        self.mycolumn = column.ColumnProperties(n_height=this_nheight)

    def set_geometry(self):
        tempUnknowns = {}
        self.geom.compute(self.inputs, tempUnknowns)
        for pairs in tempUnknowns.items():
            self.inputs[pairs[0]] = pairs[1]
        self.inputs["z_section"], _ = nodal2sectional(self.inputs["z_full"])

    def testColumnMassCG(self):
        self.mycolumn.compute_column_mass_cg(self.inputs, self.outputs)
        ibox = self.mycolumn.ibox

        bulk = self.inputs["bulkhead_mass"]
        bulkcg = self.inputs["bulkhead_z_cg"]
        stiff = self.inputs["stiffener_mass"]
        shell = self.inputs["shell_mass"]
        box = self.inputs["buoyancy_tank_mass"]
        boxcg = self.inputs["buoyancy_tank_cg"]
        m_ballast = self.inputs["ballast_mass"]
        cg_ballast = self.inputs["ballast_z_cg"]

        m_column = bulk.sum() + stiff.sum() + shell.sum() + box
        m_out = 0.05 * m_column
        m_expect = m_column + m_ballast.sum() + m_out

        mysec = stiff + shell + bulk
        mysec[ibox] += box
        mysec += m_ballast
        mysec += m_out / len(mysec)

        mycg = (box * boxcg + bulk.sum() * bulkcg + np.dot(stiff + shell, self.inputs["z_section"])) / m_column
        cg_system = ((m_column + m_out) * mycg + m_ballast.sum() * cg_ballast) / m_expect

        Iones = np.r_[np.ones(3), np.zeros(3)]
        I_expect = 1.05 * 5.6e6 * Iones + 2e3 * Iones
        I_expect[0] = I_expect[1] = I_expect[0] - m_expect * (cg_system - self.inputs["z_full"][0]) ** 2

        self.assertAlmostEqual(self.outputs["column_total_mass"].sum(), m_expect)
        self.assertAlmostEqual(self.outputs["z_center_of_mass"], cg_system)

        self.assertAlmostEqual(self.outputs["column_structural_mass"], m_column + m_out)
        self.assertAlmostEqual(self.outputs["column_outfitting_mass"], m_out)
        npt.assert_almost_equal(self.outputs["column_total_mass"], mysec)
        npt.assert_almost_equal(self.outputs["I_column"], I_expect)

    def testBalance(self):
        rho_w = self.inputs["rho_water"]

        self.mycolumn.compute_column_mass_cg(self.inputs, self.outputs)
        self.mycolumn.balance_column(self.inputs, self.outputs)

        V_column = np.pi * 100.0 * 35.0
        V_box = self.inputs["buoyancy_tank_displacement"]
        box_cg = self.inputs["buoyancy_tank_cg"]
        V_expect = V_column + V_box
        cb_expect = (-17.5 * V_column + V_box * box_cg) / V_expect
        Ixx = 0.25 * np.pi * 1e4
        Axx = np.pi * 1e2
        self.assertAlmostEqual(self.outputs["displaced_volume"].sum(), V_expect)
        self.assertAlmostEqual(self.outputs["hydrostatic_force"].sum(), V_expect * rho_w * g)
        self.assertAlmostEqual(self.outputs["z_center_of_buoyancy"], cb_expect)
        self.assertAlmostEqual(self.outputs["Iwater"], Ixx)
        self.assertAlmostEqual(self.outputs["Awater"], Axx)

        m_a = np.zeros(6)
        m_a[:2] = V_expect * rho_w
        m_a[2] = 0.5 * (8.0 / 3.0) * rho_w * 10.0 ** 3
        m_a[3:5] = np.pi * rho_w * 100.0 * ((0 - cb_expect) ** 3.0 - (-35 - cb_expect) ** 3.0) / 3.0
        npt.assert_almost_equal(self.outputs["column_added_mass"], m_a, decimal=-4)

        # Test if everything under water
        dz = -1.5 * self.inputs["z_full"][-1]
        self.inputs["z_section"] += dz
        self.inputs["z_full"] += dz
        self.mycolumn.balance_column(self.inputs, self.outputs)
        V_column = np.pi * 100.0 * 50.0
        V_expect = V_column + V_box
        cb_expect = (V_column * (-25.0 + self.inputs["z_full"][-1]) + V_box * box_cg) / V_expect
        self.assertAlmostEqual(self.outputs["displaced_volume"].sum(), V_expect)
        self.assertAlmostEqual(self.outputs["hydrostatic_force"].sum(), V_expect * rho_w * g)
        self.assertAlmostEqual(self.outputs["z_center_of_buoyancy"], cb_expect)

        # Test taper- check hydrostatic via Archimedes within 1%
        self.inputs["d_full"][5] -= 8.0
        self.mycolumn.balance_column(self.inputs, self.outputs)
        self.assertAlmostEqual(
            self.outputs["hydrostatic_force"].sum() / (self.outputs["displaced_volume"].sum() * rho_w * g),
            1.0,
            delta=1e-2,
        )

    def testCheckCost(self):
        self.outputs["column_outfitting_mass"] = 25.0
        self.outputs["column_total_mass"] = 25 * np.ones(10)
        self.mycolumn.compute_cost(self.inputs, self.outputs)

        self.assertEqual(self.outputs["column_structural_cost"], (1 + 2 + 3 + 4))
        self.assertEqual(self.outputs["column_outfitting_cost"], 1.0 * 25.0)
        self.assertEqual(self.outputs["column_total_cost"], (1 + 2 + 3 + 4) + 1.0 * (25.0) + 5)


class TestBuckle(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        onepts = np.ones((NPTS,))
        onesec = np.ones((NPTS - 1,))
        # onesec0 = np.ones((NHEIGHT,))
        self.inputs["d_full"] = 600 * onepts * in_to_si
        self.inputs["t_full"] = 0.75 * onesec * in_to_si
        self.inputs["t_web"] = 5.0 / 8.0 * onesec * in_to_si
        self.inputs["h_web"] = 14.0 * onesec * in_to_si
        self.inputs["t_flange"] = 1.0 * onesec * in_to_si
        self.inputs["w_flange"] = 10.0 * onesec * in_to_si
        self.inputs["L_stiffener"] = 5.0 * onesec * ft_to_si
        # self.inputs['section_height'] = 50.0 * onesec0 * ft_to_si
        self.inputs["pressure"] = (64.0 * lbperft3_to_si) * g * (60 * ft_to_si) * onepts
        self.inputs["E_full"] = 29e3 * ksi_to_si * onesec
        self.inputs["nu_full"] = 0.3 * onesec
        self.inputs["sigma_y_full"] = 50 * ksi_to_si * onesec
        self.inputs["wave_height"] = 0.0  # gives only static pressure
        self.inputs["stack_mass_in"] = 9000 * kip_to_si / g
        self.inputs["section_mass"] = 0.0 * np.ones((NPTS - 1,))
        self.discrete_inputs["loading"] = "radial"
        self.inputs["z_full"] = np.linspace(0, 1, NPTS)
        self.inputs["z_section"], _ = nodal2sectional(self.inputs["z_full"])
        self.inputs["z_param"] = np.linspace(0, 1, NHEIGHT)
        opt = {}
        opt["gamma_f"] = 1.0
        opt["gamma_b"] = 1.0

        self.buckle = column.ColumnBuckling(n_height=NHEIGHT, modeling_options=opt)

    def testAppliedAxial(self):
        t = self.inputs["t_full"][0]
        d = self.inputs["d_full"][0]
        kip_to_si = 4.4482216 * 1e3
        expect = 9000 * kip_to_si / (2 * np.pi * t * (0.5 * d - 0.5 * t))
        npt.assert_almost_equal(self.buckle.compute_applied_axial(self.inputs), expect, decimal=4)

    def testCheckStresses(self):
        self.buckle.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_almost_equal(self.outputs["web_compactness"], 24.1 / 22.4, decimal=3)
        npt.assert_almost_equal(self.outputs["flange_compactness"], 9.03 / 5.0, decimal=3)
        self.assertAlmostEqual(self.outputs["axial_local_api"][1], 1.07, 1)
        self.assertAlmostEqual(self.outputs["axial_general_api"][1], 0.34, 1)
        self.assertAlmostEqual(self.outputs["external_local_api"][1], 1.07, 1)
        self.assertAlmostEqual(self.outputs["external_general_api"][1], 0.59, 1)


class TestGroup(unittest.TestCase):
    def testAll(self):
        opt = {}
        opt["gamma_f"] = 1.0
        opt["gamma_b"] = 1.0
        opt["materials"] = {}
        opt["materials"]["n_mat"] = 1
        colopt = {}
        colopt["n_height"] = 3
        colopt["n_bulkhead"] = 3
        colopt["n_layers"] = 1

        prob = om.Problem()

        prob.model.add_subsystem(
            "col", column.Column(column_options=colopt, modeling_options=opt, n_mat=1), promotes=["*"]
        )

        prob.setup()
        prob["freeboard"] = 15.0
        prob["height"] = 50.0
        prob["s"] = np.array([0.0, 0.4, 1.0])
        prob["outer_diameter_in"] = 10.0 * np.ones(3)
        prob["layer_thickness"] = 0.05 * np.ones((1, 2))
        prob["stiffener_web_height"] = np.ones(2)
        prob["stiffener_web_thickness"] = 0.5 * np.ones(2)
        prob["stiffener_flange_width"] = 2.0 * np.ones(2)
        prob["stiffener_flange_thickness"] = 0.3 * np.ones(2)
        prob["stiffener_spacing"] = 0.1 * np.ones(2)
        prob["bulkhead_thickness"] = 0.05 * np.ones(3)
        prob["bulkhead_locations"] = np.array([0.0, 0.5, 1.0])
        prob["permanent_ballast_height"] = 1.0
        prob["buoyancy_tank_diameter"] = 15.0
        prob["buoyancy_tank_height"] = 0.25
        prob["buoyancy_tank_location"] = 0.3
        prob["rho_water"] = 1e3
        prob["mu_water"] = 1e-5
        prob["water_depth"] = 100.0
        prob["permanent_ballast_density"] = 2e4
        prob["beta_wave"] = 0.0
        prob["wave_z0"] = -100.0
        prob["Hsig_wave"] = 5.0
        prob["wind_z0"] = 0.0
        prob["zref"] = 100.0
        prob["Uref"] = 10.0
        prob["rho_air"] = 1.0
        prob["mu_air"] = 1e-5

        prob["Tsig_wave"] = 10.0
        prob["outfitting_factor"] = 1.05
        prob["ballast_cost_rate"] = 5.0
        prob["unit_cost_mat"] = np.array([2.0])
        prob["labor_cost_rate"] = 10.0
        prob["painting_cost_rate"] = 20.0
        prob["outfitting_cost_rate"] = 300.0
        prob["loading"] = "hydrostatic"
        prob["shearExp"] = 0.1
        prob["beta_wind"] = 0.0
        prob["cd_usr"] = -1.0
        prob["cm"] = 0.0
        prob["Uc"] = 0.0
        prob["yaw"] = 0.0
        prob["rho_mat"] = np.array([1e4])
        prob["E_mat"] = 2e9 * np.ones((1, 3))
        nu = 0.3
        prob["G_mat"] = 0.5 * prob["E_mat"] / (1 + nu)
        prob["sigma_y_mat"] = np.array([3e6])
        prob["max_draft"] = 70.0

        prob.run_model()
        self.assertTrue(True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestInputs))
    suite.addTest(unittest.makeSuite(TestBulk))
    suite.addTest(unittest.makeSuite(TestBuoyancyTank))
    suite.addTest(unittest.makeSuite(TestStiff))
    suite.addTest(unittest.makeSuite(TestBallast))
    suite.addTest(unittest.makeSuite(TestGeometry))
    suite.addTest(unittest.makeSuite(TestProperties))
    suite.addTest(unittest.makeSuite(TestBuckle))
    suite.addTest(unittest.makeSuite(TestGroup))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
