import os
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.rotorse.rail_transport as rt


class TestRT(unittest.TestCase):
    def testRailTransport(self):
        prob = om.Problem()

        # Data structure generated with
        # np.savez('rail.npz', blade_ref_axis = inputs['blade_ref_axis'],
        # chord = inputs['chord'], theta = inputs['theta'],
        # x_sc = inputs['x_sc'], y_sc = inputs['y_sc'],
        # A = inputs['A'], rhoA = inputs['rhoA'],
        # rhoJ = inputs['rhoJ'], GJ = inputs['GJ'],
        # EA = inputs['EA'], EIxx = inputs['EIxx'],
        # EIyy = inputs['EIyy'], EIxy = inputs['EIxy'],
        # coord_xy_interp = inputs['coord_xy_interp'])

        path2npz = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "rail.npz"
        data = np.load(path2npz)

        blade_ref_axis = data["blade_ref_axis"]
        chord = data["chord"]
        theta = data["theta"]
        x_sc = data["x_sc"]
        y_sc = data["y_sc"]
        A = data["A"]
        rhoA = data["rhoA"]
        rhoJ = data["rhoJ"]
        GJ = data["GJ"]
        EA = data["EA"]
        EIxx = data["EIxx"]
        EIyy = data["EIyy"]
        EIxy = data["EIxy"]
        coord_xy_interp = data["coord_xy_interp"]

        n_span = np.size(coord_xy_interp, axis=0)
        n_xy = np.size(coord_xy_interp, axis=1)
        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_xy"] = n_xy

        prob.model.add_subsystem("powercurve", rt.RailTransport(modeling_options=modeling_options), promotes=["*"])

        prob.setup()

        prob.set_val("blade_ref_axis", blade_ref_axis, units="m")
        prob.set_val("chord", chord, units="m")
        prob.set_val("theta", theta, units="deg")
        prob.set_val("x_sc", x_sc, units="m")
        prob.set_val("y_sc", y_sc, units="m")
        prob.set_val("A", A, units="m**2")
        prob.set_val("rhoA", rhoA, units="kg/m")
        prob.set_val("rhoJ", rhoJ, units="kg*m")
        prob.set_val("GJ", GJ, units="N*m**2")
        prob.set_val("EA", EA, units="N")
        prob.set_val("EIxx", EIxx, units="N*m**2")
        prob.set_val("EIyy", EIyy, units="N*m**2")
        prob.set_val("EIxy", EIxy, units="N*m**2")
        prob.set_val("coord_xy_interp", coord_xy_interp)

        prob.run_model()

        print(prob["constr_LV_4axle_horiz"][0]) 
        print(prob["constr_LV_4axle_horiz"][1]) 
        print(prob["constr_LV_8axle_horiz"][0]) 
        print(prob["constr_LV_8axle_horiz"][1]) 

        self.assertAlmostEqual(prob["constr_LV_4axle_horiz"][0], 2.205225587616127, places=1)
        self.assertAlmostEqual(prob["constr_LV_4axle_horiz"][1], 2.5000819086864334, places=1)
        # self.assertAlmostEqual(prob["constr_LV_8axle_horiz"][0], 0.8586925171861572, places=2)
        # self.assertAlmostEqual(prob["constr_LV_8axle_horiz"][1], 1.008140486233143, places=2)
        # self.assertAlmostEqual(max(abs(prob["constr_strainPS"])), 0.39050013815507656, places=2)
        # self.assertAlmostEqual(max(abs(prob["constr_strainSS"])), 0.432439251608084, places=2)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRT))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
