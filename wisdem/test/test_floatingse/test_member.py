import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.floatingse.member as member
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
        self.inputs["outfitting_factor_in"] = 1.05
        self.discrete_inputs["material_names"] = ["steel"]
        myobj = member.DiscretizationYAML(n_height=5, n_layers=1, n_mat=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["section_height"], 25.0 * np.ones(4))
        npt.assert_equal(self.outputs["outer_diameter"], self.inputs["outer_diameter_in"])
        npt.assert_equal(self.outputs["wall_thickness"], 0.25 * np.ones(4))
        npt.assert_equal(self.outputs["E"], 1e9 * np.ones(4))
        npt.assert_equal(self.outputs["G"], 1e8 * np.ones(4))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.ones(4))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.ones(4))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.ones(4))
        npt.assert_equal(self.outputs["outfitting_factor"], 1.05 * np.ones(4))

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
        self.inputs["outfitting_factor_in"] = 1.05
        self.discrete_inputs["material_names"] = ["steel", "other"]
        myobj = member.DiscretizationYAML(n_height=5, n_layers=1, n_mat=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["section_height"], 25.0 * np.ones(4))
        npt.assert_equal(self.outputs["outer_diameter"], self.inputs["outer_diameter_in"])
        npt.assert_equal(self.outputs["wall_thickness"], np.array([0.25, 0.25, 0.1, 0.1]))
        npt.assert_equal(self.outputs["E"], 1e9 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["G"], 1e8 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["sigma_y"], 1e7 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["rho"], 1e4 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["unit_cost"], 1e1 * np.array([1, 1, 2, 2]))
        npt.assert_equal(self.outputs["outfitting_factor"], 1.05 * np.ones(4))


class TestDiscretization(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}

        self.inputs["s"] = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        self.inputs["height"] = 1e1
        self.inputs["outer_diameter"] = 5.0 * np.ones(5)
        self.inputs["wall_thickness"] = 0.05 * np.ones(4)
        self.inputs["unit_cost"] = 1.0 * np.ones(4)
        self.inputs["E"] = 2e9 * np.ones(4)
        self.inputs["G"] = 2e7 * np.ones(4)
        self.inputs["sigma_y"] = 3e9 * np.ones(4)
        self.inputs["rho"] = 7850 * np.ones(4)
        self.inputs["outfitting_factor"] = 1.05 * np.ones(4)
        self.inputs["unit_cost"] = 7.0 * np.ones(4)

        self.mydis = member.MemberDiscretization(n_height=5, nRefine=2)

    def testRefine2(self):
        self.mydis.compute(self.inputs, self.outputs)
        npt.assert_array_equal(self.outputs["z_full"], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]))
        npt.assert_array_equal(self.outputs["d_full"], 5.0 * np.ones(9))
        npt.assert_array_equal(self.outputs["t_full"], 0.05 * np.ones(8))
        npt.assert_array_equal(self.outputs["E_full"], 2e9 * np.ones(8))
        npt.assert_array_equal(self.outputs["G_full"], 2e7 * np.ones(8))
        npt.assert_array_equal(self.outputs["nu_full"], 49 * np.ones(8))
        npt.assert_array_equal(self.outputs["sigma_y_full"], 3e9 * np.ones(8))
        npt.assert_array_equal(self.outputs["rho_full"], 7850 * np.ones(8))
        npt.assert_array_equal(self.outputs["unit_cost_full"], 7 * np.ones(8))
        npt.assert_array_equal(self.outputs["outfitting_full"], 1.05 * np.ones(8))

        npt.assert_almost_equal(self.outputs["s_full"], np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]))
        for k in self.inputs["s"]:
            self.assertIn(k, self.outputs["s_full"])

    def testRefineInterp(self):
        self.inputs["outer_diameter"] = np.array([5.0, 5.0, 6.0, 7.0, 7.0])
        self.inputs["wall_thickness"] = 1e-2 * np.array([5.0, 5.0, 6.0, 7.0])
        self.mydis.compute(self.inputs, self.outputs)
        npt.assert_almost_equal(self.outputs["s_full"], np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]))
        npt.assert_array_equal(self.outputs["z_full"], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]))
        npt.assert_array_equal(self.outputs["d_full"], np.array([5.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.0, 7.0]))
        npt.assert_array_equal(self.outputs["t_full"], 1e-2 * np.array([5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0]))
        npt.assert_array_equal(self.outputs["E_full"], 2e9 * np.ones(8))
        npt.assert_array_equal(self.outputs["G_full"], 2e7 * np.ones(8))
        npt.assert_array_equal(self.outputs["nu_full"], 49 * np.ones(8))
        npt.assert_array_equal(self.outputs["sigma_y_full"], 3e9 * np.ones(8))
        npt.assert_array_equal(self.outputs["rho_full"], 7850 * np.ones(8))
        npt.assert_array_equal(self.outputs["unit_cost_full"], 7 * np.ones(8))
        npt.assert_array_equal(self.outputs["outfitting_full"], 1.05 * np.ones(8))


class TestMemberComponent(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}

        self.inputs["s_full"] = np.linspace(0, 1, NPTS)
        self.inputs["z_full"] = 100 * np.linspace(0, 1, NPTS)
        self.inputs["height"] = 100
        self.inputs["d_full"] = 10.0 * myones
        self.inputs["t_full"] = 0.05 * secones
        self.inputs["rho_full"] = 1e3 * secones
        self.inputs["E_full"] = 1e6 * secones
        self.inputs["G_full"] = 1e5 * secones
        self.inputs["outfitting_full"] = 1.1 * secones
        self.inputs["unit_cost_full"] = 1.0 * secones
        self.inputs["painting_cost_rate"] = 10.0
        self.inputs["labor_cost_rate"] = 2.0

        self.inputs["bulkhead_grid"] = np.array([0.0, 0.22, 0.88, 1.0])
        nbulk = len(self.inputs["bulkhead_grid"])
        self.inputs["bulkhead_thickness"] = 1.0 * np.ones(nbulk)

        self.inputs["ring_stiffener_web_thickness"] = 0.2
        self.inputs["ring_stiffener_flange_thickness"] = 0.3
        self.inputs["ring_stiffener_web_height"] = 0.5
        self.inputs["ring_stiffener_flange_width"] = 1.0
        self.inputs["ring_stiffener_spacing"] = 20.0

        self.inputs["ballast_grid"] = np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.5]])
        self.inputs["ballast_density"] = np.array([2e3, 4e3, 1e2])
        self.inputs["ballast_volume"] = np.pi * np.array([10.0, 10.0, 0.0])
        self.inputs["ballast_unit_cost"] = np.array([2.0, 4.0, 0.0])

        self.inputs["grid_axial_joints"] = np.array([0.44, 0.55, 0.66])
        self.inputs["joint0"] = np.array([20.0, 10.0, -30.0])
        self.inputs["joint1"] = np.array([25.0, 10.0, 15.0])

        self.mem = member.MemberComponent()
        self.mem.options = {}
        self.mem.options["n_axial"] = 3
        self.mem.options["n_bulkhead"] = nbulk
        self.mem.options["n_ballast"] = 2
        self.mem.options["n_ring"] = 5
        self.mem.sections = member.SortedDict()

    def testSortedDict(self):
        # Test create list
        self.mem.add_section(0.0, 0.5, "sec0")
        self.mem.add_section(0.5, 1.0, "sec1")
        self.assertEqual(list(self.mem.sections.keys()), [0.0, 0.5, 1.0])
        self.assertEqual(list(self.mem.sections.values()), ["sec0", "sec1", None])

        # Test adding a node
        self.mem.add_node(0.25)
        self.assertEqual(list(self.mem.sections.keys()), [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(list(self.mem.sections.values()), ["sec0", "sec0", "sec1", None])
        self.mem.add_node(0.25)
        self.assertEqual(list(self.mem.sections.keys()), [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(list(self.mem.sections.values()), ["sec0", "sec0", "sec1", None])

        # Test inserting a section
        self.mem.insert_section(0.75, 0.8, "sec2")
        self.assertEqual(list(self.mem.sections.keys()), [0.0, 0.25, 0.5, 0.75, 0.8, 1.0])
        self.assertEqual(list(self.mem.sections.values()), ["sec0", "sec0", "sec1", "sec2", "sec1", None])
        self.mem.insert_section(0.45, 0.55, "sec3")
        self.assertEqual(list(self.mem.sections.keys()), [0.0, 0.25, 0.45, 0.5, 0.55, 0.75, 0.8, 1.0])
        self.assertEqual(
            list(self.mem.sections.values()), ["sec0", "sec0", "sec3", "sec3", "sec1", "sec2", "sec1", None]
        )

    def testMainSections(self):
        self.mem.add_main_sections(self.inputs, self.outputs)

        m = np.pi * 0.25 * (10.0 ** 2 - 9.9 ** 2) * 1e3 * 1.1 * 100.0
        Iax = 0.5 * m * 0.25 * (10.0 ** 2 + 9.9 ** 2)
        Ix = (1 / 12.0) * m * (3 * 0.25 * (10.0 ** 2 + 9.9 ** 2) + 100 ** 2) + m * 50 * 50  # parallel axis on last term
        self.assertAlmostEqual(self.outputs["shell_mass"], m)
        self.assertAlmostEqual(self.outputs["shell_z_cg"], 50.0)
        npt.assert_almost_equal(self.outputs["shell_I_base"], [Ix, Ix, Iax, 0.0, 0.0, 0.0], decimal=5)
        self.assertGreater(self.outputs["shell_cost"], 1e3)

        key = list(self.mem.sections.keys())
        self.assertEqual(key, self.inputs["s_full"].tolist())
        for k in key:
            if k == 1.0:
                self.assertEqual(self.mem.sections[k], None)
            else:
                self.assertAlmostEqual(self.mem.sections[k].A, 1.1 * np.pi * 0.25 * (10.0 ** 2 - 9.9 ** 2))
                self.assertAlmostEqual(self.mem.sections[k].Ixx, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Iyy, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Izz, 2 * 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].rho, 1e3)
                self.assertAlmostEqual(self.mem.sections[k].E, 1e6)
                self.assertAlmostEqual(self.mem.sections[k].G, 1e5)

    def testBulk(self):
        self.mem.add_main_sections(self.inputs, self.outputs)
        self.mem.add_bulkhead_sections(self.inputs, self.outputs)

        s_full = self.inputs["s_full"]
        key = list(self.mem.sections.keys())
        self.assertEqual(key, np.sort(np.r_[s_full, 0.01, 0.215, 0.225, 0.875, 0.885, 0.99]).tolist())
        for k in key:
            if k in [0.0, 0.215, 0.875, 0.99]:
                self.assertAlmostEqual(self.mem.sections[k].A, 1.1 * np.pi * 0.25 * (10.0 ** 2 - 0 ** 2))
                self.assertAlmostEqual(self.mem.sections[k].Ixx, 1.1 * np.pi * (10.0 ** 4 - 0 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Iyy, 1.1 * np.pi * (10.0 ** 4 - 0 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Izz, 2 * 1.1 * np.pi * (10.0 ** 4 - 0 ** 4) / 64)
            elif k == 1.0:
                self.assertEqual(self.mem.sections[k], None)
                continue
            else:
                self.assertAlmostEqual(self.mem.sections[k].A, 1.1 * np.pi * 0.25 * (10.0 ** 2 - 9.9 ** 2))
                self.assertAlmostEqual(self.mem.sections[k].Ixx, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Iyy, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Izz, 2 * 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)

            self.assertAlmostEqual(self.mem.sections[k].rho, 1e3)
            self.assertAlmostEqual(self.mem.sections[k].E, 1e6)
            self.assertAlmostEqual(self.mem.sections[k].G, 1e5)

        R_i = 0.5 * 10 - 0.05
        m_bulk = 1.1 * 1e3 * np.pi * R_i ** 2 * 1.0
        npt.assert_almost_equal(self.outputs["bulkhead_mass"], m_bulk * 4)
        npt.assert_almost_equal(self.outputs["bulkhead_z_cg"], 100 * (1 + 0.88 + 0.22) / 4)

        J0 = 0.50 * m_bulk * R_i ** 2
        I0 = 0.25 * m_bulk * R_i ** 2

        I = np.zeros(6)
        I[2] = 4.0 * J0
        I[0] = I0 + m_bulk * 0 ** 2
        I[0] += I0 + m_bulk * 22 ** 2
        I[0] += I0 + m_bulk * 88 ** 2
        I[0] += I0 + m_bulk * 100 ** 2
        I[1] = I[0]
        npt.assert_almost_equal(self.outputs["bulkhead_I_base"], I)

        self.assertGreater(self.outputs["bulkhead_cost"], 2e3)

    def testStiff(self):
        self.mem.add_main_sections(self.inputs, self.outputs)
        self.mem.add_ring_stiffener_sections(self.inputs, self.outputs)

        s_stiff = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        z_stiff = 100 * s_stiff

        Rwo = 0.5 * (10 - 2 * 0.05)
        Rwi = Rwo - 0.5
        Rfi = Rwi - 0.3
        self.assertEqual(self.outputs["flange_spacing_ratio"], 0.1)
        npt.assert_almost_equal(self.outputs["stiffener_radius_ratio"], 1 - Rfi / 5)

        # Test Mass
        A1 = np.pi * (Rwo ** 2 - Rwi ** 2)
        A2 = np.pi * (Rwi ** 2 - Rfi ** 2)
        V1 = A1 * 0.2
        V2 = A2 * 1.0
        m1 = V1 * 1e3
        m2 = V2 * 1e3
        m = m1 + m2
        f = 0.2
        self.assertAlmostEqual(self.outputs["stiffener_mass"], m * 5)
        self.assertAlmostEqual(self.outputs["stiffener_z_cg"], 50.0)
        self.assertGreater(self.outputs["stiffener_cost"], 1e3)

        # Test moment
        I_web = member.I_cyl(Rwi, Rwo, 0.2, m1)
        I_fl = member.I_cyl(Rfi, Rwi, 1.0, m2)
        I_sec = (I_web + I_fl).flatten()

        I = np.zeros(6)
        I[0] = np.sum(I_sec[0] + m * z_stiff ** 2.0)
        I[1] = I[0]
        I[2] = 5 * I_sec[2]
        npt.assert_almost_equal(self.outputs["stiffener_I_base"], I)

        s_full = self.inputs["s_full"]
        key = list(self.mem.sections.keys())
        npt.assert_almost_equal(
            key, np.sort(np.r_[s_full, 0.095, 0.105, 0.295, 0.305, 0.495, 0.505, 0.695, 0.705, 0.895, 0.905])
        )
        for k in key:
            if np.round(k, 3) in [0.095, 0.295, 0.495, 0.695, 0.895]:
                self.assertAlmostEqual(
                    self.mem.sections[k].A, f * A1 + A2 + 1.1 * np.pi * 0.25 * (10.0 ** 2 - 9.9 ** 2)
                )
                self.assertGreater(self.mem.sections[k].Ixx, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertGreater(self.mem.sections[k].Iyy, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertGreater(self.mem.sections[k].Izz, 2 * 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
            elif k == 1.0:
                self.assertEqual(self.mem.sections[k], None)
                continue
            else:
                self.assertAlmostEqual(self.mem.sections[k].A, 1.1 * np.pi * 0.25 * (10.0 ** 2 - 9.9 ** 2))
                self.assertAlmostEqual(self.mem.sections[k].Ixx, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Iyy, 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
                self.assertAlmostEqual(self.mem.sections[k].Izz, 2 * 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)

            self.assertAlmostEqual(self.mem.sections[k].rho, 1e3)
            self.assertAlmostEqual(self.mem.sections[k].E, 1e6)
            self.assertAlmostEqual(self.mem.sections[k].G, 1e5)

    def testBallast(self):
        self.mem.add_main_sections(self.inputs, self.outputs)
        self.mem.add_ballast_sections(self.inputs, self.outputs)

        area = 0.25 * np.pi * 9.9 ** 2
        h = 10 * np.pi / area
        cg_perm = (2 * 0.5 * h + 4 * (10 + 0.5 * h)) / 6
        m_perm = np.pi * 6e4

        I_perm = np.zeros(6)
        I_perm[2] = 0.5 * m_perm * 0.25 * 9.9 ** 2
        I_perm[0] = (
            m_perm * (3 * 0.25 * 9.9 ** 2 + h ** 2) / 12.0
            + (1 / 3) * m_perm * (0.5 * h) ** 2
            + (2 / 3) * m_perm * (10 + 0.5 * h) ** 2
        )
        I_perm[1] = I_perm[0]

        self.assertAlmostEqual(self.outputs["ballast_mass"], m_perm)
        self.assertAlmostEqual(self.outputs["ballast_cost"], np.pi * 20e4)
        self.assertAlmostEqual(self.outputs["ballast_z_cg"], cg_perm)
        npt.assert_almost_equal(self.outputs["ballast_I_base"], I_perm)
        self.assertAlmostEqual(self.outputs["variable_ballast_capacity"], area * 30)

    def testMassProp(self):
        self.mem.add_main_sections(self.inputs, self.outputs)
        self.mem.add_bulkhead_sections(self.inputs, self.outputs)
        self.mem.add_ring_stiffener_sections(self.inputs, self.outputs)
        self.mem.add_ballast_sections(self.inputs, self.outputs)
        self.mem.compute_mass_properties(self.inputs, self.outputs)

        m_shell = np.pi * 0.25 * (10.0 ** 2 - 9.9 ** 2) * 1e3 * 1.1 * 100.0
        R_i = 0.5 * 10 - 0.05
        cg_shell = 50

        m_bulk = 1.1 * 1e3 * np.pi * R_i ** 2 * 1.0
        cg_bulk = 100 * (1 + 0.88 + 0.22) / 4

        Rwo = 0.5 * (10 - 2 * 0.05)
        Rwi = Rwo - 0.5
        Rfi = Rwi - 0.3
        A1 = np.pi * (Rwo ** 2 - Rwi ** 2)
        A2 = np.pi * (Rwi ** 2 - Rfi ** 2)
        V1 = A1 * 0.2
        V2 = A2 * 1.0
        m1 = V1 * 1e3
        m2 = V2 * 1e3
        m_stiff = m1 + m2
        cg_stiff = 50.0

        area = 0.25 * np.pi * 9.9 ** 2
        h = 10 * np.pi / area
        cg_perm = (2 * 0.5 * h + 4 * (10 + 0.5 * h)) / 6
        m_perm = np.pi * 6e4

        m_tot = m_shell + 4 * m_bulk + 5 * m_stiff + m_perm
        self.assertAlmostEqual(self.outputs["total_mass"], m_tot)
        self.assertAlmostEqual(self.outputs["structural_mass"], m_tot - m_perm)
        self.assertAlmostEqual(
            self.outputs["z_cg"], (50 * (m_shell + 5 * m_stiff) + 4 * m_bulk * cg_bulk + m_perm * cg_perm) / m_tot
        )
        self.assertEqual(
            self.outputs["total_cost"],
            self.outputs["shell_cost"]
            + self.outputs["ballast_cost"]
            + self.outputs["bulkhead_cost"]
            + self.outputs["stiffener_cost"],
        )
        self.assertEqual(
            self.outputs["structural_cost"],
            self.outputs["shell_cost"] + self.outputs["bulkhead_cost"] + self.outputs["stiffener_cost"],
        )

    def testNodalFinish(self):
        self.outputs["z_cg"] = 40.0
        self.mem.add_main_sections(self.inputs, self.outputs)
        self.mem.nodal_discretization(self.inputs, self.outputs)

        s_full = self.inputs["s_full"]
        s_all = self.outputs["s_all"]
        npt.assert_almost_equal(s_all, np.sort(np.r_[s_full, 0.44, 0.55, 0.66]))

        npt.assert_almost_equal(self.outputs["center_of_mass"], np.array([22, 10, -12]))
        npt.assert_almost_equal(self.outputs["nodes_xyz"][:, 0], 20 + s_all * 5)
        npt.assert_almost_equal(self.outputs["nodes_xyz"][:, 1], 10)
        npt.assert_almost_equal(self.outputs["nodes_xyz"][:, 2], -30 + s_all * 45)

        npt.assert_almost_equal(self.outputs["section_area"], 1.1 * np.pi * 0.25 * (10.0 ** 2 - 9.9 ** 2))
        npt.assert_almost_equal(self.outputs["section_Ixx"], 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
        npt.assert_almost_equal(self.outputs["section_Iyy"], 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
        npt.assert_almost_equal(self.outputs["section_Izz"], 2 * 1.1 * np.pi * (10.0 ** 4 - 9.9 ** 4) / 64)
        npt.assert_almost_equal(self.outputs["section_rho"], 1e3)
        npt.assert_almost_equal(self.outputs["section_E"], 1e6)
        npt.assert_almost_equal(self.outputs["section_G"], 1e5)

    def testCompute(self):
        self.mem.compute(self.inputs, self.outputs)
        # 2 points added for bulkheads and stiffeners
        # Bulkheads at 0,1 only get 1 new point
        self.assertEqual(self.outputs["s_all"].size, NPTS + 3 + 2 * 4 - 2 + 2 * 5)


class TestProperties(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}

        # For Geometry call
        this_nheight = 3
        this_npts = member.get_nfull(this_nheight)
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

        self.geom = member.MemberGeometry(n_height=this_nheight)
        self.set_geometry()

        self.mymember = member.MemberProperties(n_height=this_nheight)

    def set_geometry(self):
        tempOutputs = {}
        self.geom.compute(self.inputs, tempOutputs)
        for pairs in tempOutputs.items():
            self.inputs[pairs[0]] = pairs[1]
        self.inputs["z_section"], _ = nodal2sectional(self.inputs["z_full"])

    def testMemberMassCG(self):
        self.mymember.compute_member_mass_cg(self.inputs, self.outputs)
        ibox = self.mymember.ibox

        bulk = self.inputs["bulkhead_mass"]
        bulkcg = self.inputs["bulkhead_z_cg"]
        stiff = self.inputs["stiffener_mass"]
        shell = self.inputs["shell_mass"]
        box = self.inputs["buoyancy_tank_mass"]
        boxcg = self.inputs["buoyancy_tank_cg"]
        m_ballast = self.inputs["ballast_mass"]
        cg_ballast = self.inputs["ballast_z_cg"]

        m_member = bulk.sum() + stiff.sum() + shell.sum() + box
        m_out = 0.05 * m_member
        m_expect = m_member + m_ballast.sum() + m_out

        mysec = stiff + shell + bulk
        mysec[ibox] += box
        mysec += m_ballast
        mysec += m_out / len(mysec)

        mycg = (box * boxcg + bulk.sum() * bulkcg + np.dot(stiff + shell, self.inputs["z_section"])) / m_member
        cg_system = ((m_member + m_out) * mycg + m_ballast.sum() * cg_ballast) / m_expect

        Iones = np.r_[np.ones(3), np.zeros(3)]
        I_expect = 1.05 * 5.6e6 * Iones + 2e3 * Iones
        I_expect[0] = I_expect[1] = I_expect[0] - m_expect * (cg_system - self.inputs["z_full"][0]) ** 2

        self.assertAlmostEqual(self.outputs["member_total_mass"].sum(), m_expect)
        self.assertAlmostEqual(self.outputs["z_center_of_mass"], cg_system)

        self.assertAlmostEqual(self.outputs["member_structural_mass"], m_member + m_out)
        self.assertAlmostEqual(self.outputs["member_outfitting_mass"], m_out)
        npt.assert_almost_equal(self.outputs["member_total_mass"], mysec)
        npt.assert_almost_equal(self.outputs["I_member"], I_expect)

    def testBalance(self):
        rho_w = self.inputs["rho_water"]

        self.mymember.compute_member_mass_cg(self.inputs, self.outputs)
        self.mymember.balance_member(self.inputs, self.outputs)

        V_member = np.pi * 100.0 * 35.0
        V_box = self.inputs["buoyancy_tank_displacement"]
        box_cg = self.inputs["buoyancy_tank_cg"]
        V_expect = V_member + V_box
        cb_expect = (-17.5 * V_member + V_box * box_cg) / V_expect
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
        npt.assert_almost_equal(self.outputs["member_added_mass"], m_a, decimal=-4)

        # Test if everything under water
        dz = -1.5 * self.inputs["z_full"][-1]
        self.inputs["z_section"] += dz
        self.inputs["z_full"] += dz
        self.mymember.balance_member(self.inputs, self.outputs)
        V_member = np.pi * 100.0 * 50.0
        V_expect = V_member + V_box
        cb_expect = (V_member * (-25.0 + self.inputs["z_full"][-1]) + V_box * box_cg) / V_expect
        self.assertAlmostEqual(self.outputs["displaced_volume"].sum(), V_expect)
        self.assertAlmostEqual(self.outputs["hydrostatic_force"].sum(), V_expect * rho_w * g)
        self.assertAlmostEqual(self.outputs["z_center_of_buoyancy"], cb_expect)

        # Test taper- check hydrostatic via Archimedes within 1%
        self.inputs["d_full"][5] -= 8.0
        self.mymember.balance_member(self.inputs, self.outputs)
        self.assertAlmostEqual(
            self.outputs["hydrostatic_force"].sum() / (self.outputs["displaced_volume"].sum() * rho_w * g),
            1.0,
            delta=1e-2,
        )

    def testCheckCost(self):
        self.outputs["member_outfitting_mass"] = 25.0
        self.outputs["member_total_mass"] = 25 * np.ones(10)
        self.mymember.compute_cost(self.inputs, self.outputs)

        self.assertEqual(self.outputs["member_structural_cost"], (1 + 2 + 3 + 4))
        self.assertEqual(self.outputs["member_outfitting_cost"], 1.0 * 25.0)
        self.assertEqual(self.outputs["member_total_cost"], (1 + 2 + 3 + 4) + 1.0 * (25.0) + 5)


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
            "col", member.Member(member_options=colopt, modeling_options=opt, n_mat=1), promotes=["*"]
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
    suite.addTest(unittest.makeSuite(TestDiscretization))
    suite.addTest(unittest.makeSuite(TestMemberComponent))
    # suite.addTest(unittest.makeSuite(TestProperties))
    # suite.addTest(unittest.makeSuite(TestGroup))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
