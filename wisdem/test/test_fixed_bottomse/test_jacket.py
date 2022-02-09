import copy
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt

from wisdem.commonse import gravity as g
from wisdem.fixed_bottomse.jacket import JacketSE


class Test(unittest.TestCase):
    def testAll(self):
        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["n_dlc"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = True
        modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = 3
        modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = 2
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.0
        modeling_options["WISDEM"]["FixedBottomSE"]["material"] = "steel"
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"] = {}
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"] = 1
        modeling_options["WISDEM"]["FixedBottomSE"]["save_truss_figures"] = False
        modeling_options["materials"] = {}
        modeling_options["materials"]["n_mat"] = 1

        prob = om.Problem(
            model=JacketSE(
                modeling_options=modeling_options,
            )
        )

        prob.setup()

        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8

        prob.run_model()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
