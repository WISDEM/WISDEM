import unittest

import numpy as np
import numpy.testing as npt
import wisdem.towerse.tower_struct as tow
from wisdem.towerse import RIGID


class TestStruct(unittest.TestCase):
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

    def testPreFrame(self):

        # Test Land
        self.inputs["z_param"] = 10.0 * np.array([0.0, 3.0, 6.0])
        self.inputs["z_full"] = 10.0 * np.arange(0, 7)
        self.inputs["d_full"] = 6.0 * np.ones(self.inputs["z_full"].shape)
        self.inputs["mass"] = 1e5
        self.inputs["mI"] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        self.inputs["mrho"] = np.array([-3.0, 0.0, 1.0])
        self.inputs["transition_piece_mass"] = 0.0
        self.inputs["transition_piece_cost"] = 0.0
        self.inputs["transition_piece_height"] = 0.0
        self.inputs["transition_piece_I"] = np.zeros(6)
        self.inputs["gravity_foundation_I"] = np.zeros(6)
        self.inputs["gravity_foundation_mass"] = 0.0
        self.inputs["suctionpile_depth"] = 0.0
        self.inputs["rna_F"] = 1e5 * np.arange(2,5)
        self.inputs["rna_M"] = 1e6 * np.arange(2,5)
        self.inputs["E"] = 1e9 * np.ones(2)
        self.inputs["G"] = 1e8 * np.ones(2)
        self.inputs["sigma_y"] = 1e8 * np.ones(2)

        myobj = tow.TowerPreFrame(n_height=3, n_refine=3, monopile=False)
        myobj.compute(self.inputs, self.outputs)

        npt.assert_equal(self.outputs["kidx"], np.array([0]))
        npt.assert_equal(self.outputs["kx"], np.array([RIGID]))
        npt.assert_equal(self.outputs["ky"], np.array([RIGID]))
        npt.assert_equal(self.outputs["kz"], np.array([RIGID]))
        npt.assert_equal(self.outputs["ktx"], np.array([RIGID]))
        npt.assert_equal(self.outputs["kty"], np.array([RIGID]))
        npt.assert_equal(self.outputs["ktz"], np.array([RIGID]))

        npt.assert_equal(self.outputs["midx"], np.zeros(2))
        npt.assert_equal(self.outputs["m"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhox"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoy"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxx"], np.zeros(2))
        npt.assert_equal(self.outputs["mIyy"], np.zeros(2))
        npt.assert_equal(self.outputs["mIzz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxy"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIyz"], np.zeros(2))

        npt.assert_equal(self.outputs["plidx"], np.array([6]))
        npt.assert_equal(self.outputs["Fx"], np.array([2e5]))
        npt.assert_equal(self.outputs["Fy"], np.array([3e5]))
        npt.assert_equal(self.outputs["Fz"], np.array([4e5]))
        npt.assert_equal(self.outputs["Mxx"], np.array([2e6]))
        npt.assert_equal(self.outputs["Myy"], np.array([3e6]))
        npt.assert_equal(self.outputs["Mzz"], np.array([4e6]))

        # Test Monopile no springs, no GBF
        self.inputs["z_full"] = 10.0 * np.arange(-6, 7)
        self.inputs["d_full"] = 6.0 * np.ones(self.inputs["z_full"].shape)
        self.inputs["transition_piece_mass"] = 1e3
        self.inputs["transition_piece_cost"] = 1e4
        self.inputs["transition_piece_I"] = 1e3 * 9 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]
        self.inputs["transition_piece_height"] = 10.0
        self.inputs["gravity_foundation_mass"] = 0.0  # 1e4
        self.inputs["suctionpile_depth"] = 30.0
        self.inputs["rna_F"] = 1e5 * np.arange(2,5)
        self.inputs["rna_M"] = 1e6 * np.arange(2,5)
        self.inputs["k_soil"] = (20.0 + np.arange(6))[np.newaxis, :] * np.ones((2, 6))
        self.inputs["z_soil"] = np.r_[-30.0, 0.0]

        myobj = tow.TowerPreFrame(n_height=5, n_refine=3, monopile=True, soil_springs=False)
        myobj.compute(self.inputs, self.outputs)

        npt.assert_equal(self.outputs["kidx"], np.arange(4))
        npt.assert_equal(self.outputs["kx"], RIGID)
        npt.assert_equal(self.outputs["ky"], RIGID)
        npt.assert_equal(self.outputs["kz"], RIGID)
        npt.assert_equal(self.outputs["ktx"], RIGID)
        npt.assert_equal(self.outputs["kty"], RIGID)
        npt.assert_equal(self.outputs["ktz"], RIGID)

        npt.assert_equal(self.outputs["midx"], np.array([7, 0]))
        npt.assert_equal(self.outputs["m"], np.array([1e3, 0.0]))
        npt.assert_equal(self.outputs["mrhox"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoy"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxx"], np.array([1e3 * 9 * 0.5, 0]))
        npt.assert_equal(self.outputs["mIyy"], np.array([1e3 * 9 * 0.5, 0]))
        npt.assert_equal(self.outputs["mIzz"], np.array([1e3 * 9, 0]))
        npt.assert_equal(self.outputs["mIxy"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIyz"], np.zeros(2))

        npt.assert_equal(self.outputs["plidx"], np.array([12]))
        npt.assert_equal(self.outputs["Fx"], np.array([2e5]))
        npt.assert_equal(self.outputs["Fy"], np.array([3e5]))
        npt.assert_equal(self.outputs["Fz"], np.array([4e5]))
        npt.assert_equal(self.outputs["Mxx"], np.array([2e6]))
        npt.assert_equal(self.outputs["Myy"], np.array([3e6]))
        npt.assert_equal(self.outputs["Mzz"], np.array([4e6]))

        # Test Monopile springs, no GBF
        self.inputs["z_full"] = 10.0 * np.arange(-6, 7)
        self.inputs["d_full"] = 6.0 * np.ones(self.inputs["z_full"].shape)
        self.inputs["transition_piece_mass"] = 1e3
        self.inputs["transition_piece_cost"] = 1e4
        self.inputs["transition_piece_I"] = 1e3 * 9 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]
        self.inputs["transition_piece_height"] = 10.0
        self.inputs["gravity_foundation_mass"] = 0.0  # 1e4
        self.inputs["suctionpile_depth"] = 30.0
        self.inputs["rna_F"] = 1e5 * np.arange(2,5)
        self.inputs["rna_M"] = 1e6 * np.arange(2,5)
        self.inputs["k_soil"] = (20.0 + np.arange(6))[np.newaxis, :] * np.ones((2, 6))
        self.inputs["z_soil"] = np.r_[-30.0, 0.0]

        myobj = tow.TowerPreFrame(n_height=5, n_refine=3, monopile=True, soil_springs=True)
        myobj.compute(self.inputs, self.outputs)

        npt.assert_equal(self.outputs["kidx"], np.arange(4))
        npt.assert_equal(self.outputs["kx"], 20.0)
        npt.assert_equal(self.outputs["ky"], 22.0)
        npt.assert_equal(self.outputs["kz"], np.r_[24.0, np.zeros(3)])
        npt.assert_equal(self.outputs["ktx"], 21.0)
        npt.assert_equal(self.outputs["kty"], 23.0)
        npt.assert_equal(self.outputs["ktz"], 25.0)

        npt.assert_equal(self.outputs["midx"], np.array([7, 0]))
        npt.assert_equal(self.outputs["m"], np.array([1e3, 0.0]))
        npt.assert_equal(self.outputs["mrhox"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoy"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxx"], np.array([1e3 * 9 * 0.5, 0]))
        npt.assert_equal(self.outputs["mIyy"], np.array([1e3 * 9 * 0.5, 0]))
        npt.assert_equal(self.outputs["mIzz"], np.array([1e3 * 9, 0]))
        npt.assert_equal(self.outputs["mIxy"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIyz"], np.zeros(2))

        npt.assert_equal(self.outputs["plidx"], np.array([12]))
        npt.assert_equal(self.outputs["Fx"], np.array([2e5]))
        npt.assert_equal(self.outputs["Fy"], np.array([3e5]))
        npt.assert_equal(self.outputs["Fz"], np.array([4e5]))
        npt.assert_equal(self.outputs["Mxx"], np.array([2e6]))
        npt.assert_equal(self.outputs["Myy"], np.array([3e6]))
        npt.assert_equal(self.outputs["Mzz"], np.array([4e6]))

        # Test Monopile with GBF- TODO: THESE REACTIONS NEED THOUGHT
        self.inputs["z_full"] = 10.0 * np.arange(-6, 7)
        self.inputs["d_full"] = 6.0 * np.ones(self.inputs["z_full"].shape)
        self.inputs["transition_piece_mass"] = 1e3
        self.inputs["transition_piece_cost"] = 1e4
        self.inputs["transition_piece_height"] = 10.0
        self.inputs["transition_piece_I"] = 1e3 * 9 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]
        self.inputs["gravity_foundation_I"] = 0.5 * 1e4 * 9 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]
        self.inputs["gravity_foundation_mass"] = 1e4
        self.inputs["suctionpile_depth"] = 0.0
        self.inputs["rna_F"] = 1e5 * np.arange(2,5)
        self.inputs["rna_M"] = 1e6 * np.arange(2,5)
        self.inputs["k_soil"] = (20.0 + np.arange(6))[np.newaxis, :] * np.ones((2, 6))
        self.inputs["z_soil"] = np.r_[-30.0, 0.0]

        myobj = tow.TowerPreFrame(n_height=5, n_refine=3, monopile=True, gravity_foundation=True)
        myobj.compute(self.inputs, self.outputs)

        npt.assert_equal(self.outputs["kidx"], np.array([0]))
        npt.assert_equal(self.outputs["kx"], np.array([RIGID]))
        npt.assert_equal(self.outputs["ky"], np.array([RIGID]))
        npt.assert_equal(self.outputs["kz"], np.array([RIGID]))
        npt.assert_equal(self.outputs["ktx"], np.array([RIGID]))
        npt.assert_equal(self.outputs["kty"], np.array([RIGID]))
        npt.assert_equal(self.outputs["ktz"], np.array([RIGID]))

        npt.assert_equal(self.outputs["midx"], np.array([7, 0]))
        npt.assert_equal(self.outputs["m"], np.array([1e3, 1e4]))
        npt.assert_equal(self.outputs["mrhox"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoy"], np.zeros(2))
        npt.assert_equal(self.outputs["mrhoz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxx"], np.array([1e3 * 9 * 0.5, 1e4 * 9 * 0.25]))
        npt.assert_equal(self.outputs["mIyy"], np.array([1e3 * 9 * 0.5, 1e4 * 9 * 0.25]))
        npt.assert_equal(self.outputs["mIzz"], np.array([1e3 * 9, 1e4 * 9 * 0.5]))
        npt.assert_equal(self.outputs["mIxy"], np.zeros(2))
        npt.assert_equal(self.outputs["mIxz"], np.zeros(2))
        npt.assert_equal(self.outputs["mIyz"], np.zeros(2))

        npt.assert_equal(self.outputs["plidx"], np.array([12]))
        npt.assert_equal(self.outputs["Fx"], np.array([2e5]))
        npt.assert_equal(self.outputs["Fy"], np.array([3e5]))
        npt.assert_equal(self.outputs["Fz"], np.array([4e5]))
        npt.assert_equal(self.outputs["Mxx"], np.array([2e6]))
        npt.assert_equal(self.outputs["Myy"], np.array([3e6]))
        npt.assert_equal(self.outputs["Mzz"], np.array([4e6]))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestStruct))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
