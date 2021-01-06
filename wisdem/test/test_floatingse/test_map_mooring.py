import unittest

import numpy as np
import numpy.testing as npt
import wisdem.floatingse.map_mooring as mapMooring
from wisdem.pymap import pyMAP
from wisdem.commonse import gravity as g


def myisnumber(instr):
    try:
        float(instr)
    except:
        return False
    return True


myones = np.ones((100,))
truth = [
    "---------------------- LINE DICTIONARY ---------------------------------------",
    "LineType  Diam      MassDenInAir   EA            CB   CIntDamp  Ca   Cdn    Cdt",
    "(-)       (m)       (kg/m)        (N)           (-)   (Pa-s)    (-)  (-)    (-)",
    "myline   0.1    0.1   20000   0.65   1.0E8   0.6   -1.0   0.05",
    "---------------------- NODE PROPERTIES ---------------------------------------",
    "Node Type X     Y    Z   M     V FX FY FZ",
    "(-)  (-) (m)   (m)  (m) (kg) (m^3) (kN) (kN) (kN)",
    "1   VESSEL   11.0   0.0   -10.0   0.0   0.0   #   #   #",
    "2   FIX   216.50635094610965   -125.0   depth   0.0   0.0   #   #   #",
    "3   FIX   216.50635094610965   125.0   depth   0.0   0.0   #   #   #",
    "---------------------- LINE PROPERTIES ---------------------------------------",
    "Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags",
    "(-)      (-)       (m)       (-)       (-)       (-)",
    "1   myline   270.0   2   1",
    "2   myline   270.0   3   1",
    "---------------------- SOLVER OPTIONS-----------------------------------------",
    "Option",
    "(-)",
    "help",
    " integration_dt 0",
    " kb_default 3.0e6",
    " cb_default 3.0e5",
    " wave_kinematics",
    "inner_ftol 1e-5",
    "inner_gtol 1e-5",
    "inner_xtol 1e-5",
    "outer_tol 1e-3",
    " pg_cooked 10000 1",
    " outer_fd",
    " outer_bd",
    " outer_cd",
    " inner_max_its 200",
    " outer_max_its 600",
    "repeat 120 240",
    " krylov_accelerator 3",
    " ref_position 0.0 0.0 0.0",
]


class TestMapMooring(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}

        myones = np.ones(1)
        self.inputs["fairlead"] = 10.0 * myones
        self.inputs["fairlead_radius"] = 11.0 * myones
        self.inputs["anchor_radius"] = 250.0 * myones
        self.inputs["anchor_cost"] = 10.0

        self.inputs["rho_water"] = 1e3
        self.inputs["water_depth"] = 200.0  # 100.0

        self.inputs["line_length"] = 0.6 * (self.inputs["water_depth"] + self.inputs["anchor_radius"]) * myones
        self.inputs["line_diameter"] = 0.1 * myones
        self.inputs["line_mass_density_coeff"] = 10.0 * myones
        self.inputs["line_stiffness_coeff"] = 2e6 * myones
        self.inputs["line_breaking_load_coeff"] = 30.0 * myones
        self.inputs["line_cost_rate_coeff"] = 40.0 * myones
        self.inputs["max_surge_fraction"] = 0.2
        self.inputs["survival_heel"] = 10.0
        self.inputs["operational_heel"] = 5.0

        opt = {}
        opt["n_attach"] = 3
        opt["n_anchors"] = 6

        self.mymap = mapMooring.MapMooring(options=opt, gamma=1.35)

    def testGeometry(self):
        self.mymap.set_geometry(self.inputs, self.outputs)
        self.assertFalse(self.mymap.tlpFlag)
        npt.assert_equal(self.outputs["constr_mooring_length"], 0.6 * 450 / (0.95 * 440))

        self.inputs["line_length"] = 150.0
        self.mymap.set_geometry(self.inputs, self.outputs)
        self.assertTrue(self.mymap.tlpFlag)
        npt.assert_equal(self.outputs["constr_mooring_length"], 150 / (200 - 10 - 1.35 * 11 * np.sin(np.deg2rad(10))))

    def testWriteInputAll(self):
        self.mymap.set_geometry(self.inputs, self.outputs)
        self.mymap.write_input_file(self.inputs)
        actual = self.mymap.finput[:]
        expect = truth[:]
        # for k in actual: print(k)
        self.assertEqual(len(actual), len(expect))

        for n in range(len(actual)):
            actualTok = actual[n].split()
            expectTok = expect[n].split()
            self.assertEqual(len(actualTok), len(expectTok))

            for k in range(len(actualTok)):
                if myisnumber(actualTok[k]):
                    self.assertAlmostEqual(float(actualTok[k]), float(expectTok[k]), 6)
                else:
                    self.assertEqual(actualTok[k], expectTok[k])

    def testRunMap(self):
        self.mymap.compute(self.inputs, self.outputs)
        npt.assert_almost_equal(self.outputs["mooring_neutral_load"].sum(axis=0)[:2], 0.0, decimal=2)
        self.assertEqual(self.outputs["mooring_neutral_load"].shape[0], 3)
        self.assertGreater(self.outputs["constr_axial_load"], 1.0)
        self.assertGreater(self.outputs["max_surge_restoring_force"], 1e3)

        self.assertEqual(self.outputs["operational_heel_restoring_force"].shape[0], 3)
        self.assertGreater(0.0, self.outputs["operational_heel_restoring_force"].sum(axis=0)[0])
        self.assertAlmostEqual(self.outputs["operational_heel_restoring_force"].sum(axis=0)[1], 0.0, 2)
        self.assertGreater(0.0, self.outputs["operational_heel_restoring_force"].sum(axis=0)[2])

        self.assertEqual(self.outputs["survival_heel_restoring_force"].shape[0], 3)
        self.assertGreater(0.0, self.outputs["survival_heel_restoring_force"].sum(axis=0)[0])
        self.assertAlmostEqual(self.outputs["survival_heel_restoring_force"].sum(axis=0)[1], 0.0, 2)
        self.assertGreater(0.0, self.outputs["survival_heel_restoring_force"].sum(axis=0)[2])

        self.assertAlmostEqual(self.outputs["mooring_mass"], 6 * 270 * 0.1)
        self.assertAlmostEqual(self.outputs["mooring_cost"], 6 * 270 * 0.4 + 6 * 10)

    def testListEntry(self):
        # Initiate MAP++ for this design
        mymap = pyMAP()
        # mymap.ierr = 0
        mymap.map_set_sea_depth(self.inputs["water_depth"])
        mymap.map_set_gravity(g)
        mymap.map_set_sea_density(self.inputs["rho_water"])
        mymap.read_list_input(truth)
        mymap.init()
        mymap.displace_vessel(0, 0, 0, 0, 10, 0)
        mymap.update_states(0.0, 0)
        mymap.end()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMapMooring))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
