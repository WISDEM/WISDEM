import os
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from wisdem.ccblade.ccblade_component import CCBladeLoads, CCBladeTwist, CCBladeEvaluate, CCBladeGeometry

np.random.seed(314)


class Test(unittest.TestCase):
    def test_ccblade_geometry(self):
        n_span = 10

        prob = om.Problem()

        comp = CCBladeGeometry(n_span=n_span)
        prob.model.add_subsystem("comp", comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("Rtip", 80.0, units="m")
        prob.set_val("precurve_in", np.random.rand(n_span), units="m")
        prob.set_val("presweep_in", np.random.rand(n_span), units="m")
        prob.set_val("precone", 2.2, units="deg")

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True, method="fd")

        assert_check_partials(check)

    def test_ccblade_loads(self):
        prob = om.Problem()

        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(
            os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "smaller_dataset.npz", allow_pickle=True
        )

        n_span = npzfile["r"].size
        n_aoa = npzfile["aoa"].size
        n_Re = npzfile["Re"].size

        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = n_aoa
        modeling_options["WISDEM"]["RotorSE"]["n_Re"] = n_Re
        modeling_options["WISDEM"]["RotorSE"]["n_tab"] = 1

        n_span, n_aoa, n_Re, n_tab = np.moveaxis(npzfile["cl"][:, :, :, np.newaxis], 0, 1).shape
        modeling_options["airfoils"] = {}
        modeling_options["airfoils"]["n_aoa"] = n_aoa
        modeling_options["airfoils"]["n_Re"] = n_Re
        modeling_options["airfoils"]["n_tab"] = n_tab

        comp = CCBladeLoads(modeling_options=modeling_options)
        prob.model.add_subsystem("comp", comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)

        # Add some arbitrary inputs
        prob.set_val("airfoils_aoa", npzfile["aoa"], units="deg")
        prob.set_val("airfoils_Re", npzfile["Re"])
        prob.set_val("airfoils_cl", np.moveaxis(npzfile["cl"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("airfoils_cd", np.moveaxis(npzfile["cd"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("airfoils_cm", np.moveaxis(npzfile["cm"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("r", npzfile["r"], units="m")
        prob.set_val("chord", npzfile["chord"], units="m")
        prob.set_val("theta", npzfile["theta"], units="deg")

        # parameters
        prob.set_val("V_load", 12.0, units="m/s")
        prob.set_val("Omega_load", 7.0, units="rpm")
        prob.set_val("pitch_load", 2.0, units="deg")
        prob.set_val("azimuth_load", 3.0, units="deg")

        prob.set_val("Rhub", 1.0, units="m")
        prob.set_val("Rtip", 70.0, units="m")
        prob.set_val("hub_height", 100.0, units="m")
        prob.set_val("precone", 0.0, units="deg")
        prob.set_val("tilt", 0.0, units="deg")
        prob.set_val("yaw", 0.0, units="deg")
        prob.set_val("precurve", np.zeros(n_span), units="m")
        prob.set_val("precurveTip", 0.0, units="m")

        prob.set_val("rho", 1.225, units="kg/m**3")
        prob.set_val("mu", 1.81206e-5, units="kg/(m*s)")
        prob.set_val("shearExp", 0.25)
        prob.set_val("nBlades", 3)
        prob.set_val("nSector", 4)
        prob.set_val("tiploss", True)
        prob.set_val("hubloss", True)
        prob.set_val("wakerotation", True)
        prob.set_val("usecd", True)

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True)

        # Manually filter some entries out of the assert_check_partials call.
        # Will want to add this functionality to OpenMDAO itself at some point.
        new_check = {}
        for comp_name in check:
            new_check[comp_name] = {}
            for (output_name, input_name) in check[comp_name]:
                if "airfoil" not in input_name and "rho" not in input_name and "mu" not in input_name:
                    new_check[comp_name][(output_name, input_name)] = check[comp_name][(output_name, input_name)]

        assert_check_partials(new_check, rtol=5e-5, atol=10.0)

    @unittest.skip("Not useful and now OpenMDAO complains")
    def test_ccblade_twist(self):
        """
        Right now this just compares fd to fd so it is not a meaningful test.
        However, it ensures that we have the derivatives set up in the component
        to actually be finite differenced.
        """
        prob = om.Problem()

        # Add some arbitrary inputs
        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(
            os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "smaller_dataset.npz", allow_pickle=True
        )
        n_span = npzfile["r"].size
        n_aoa = npzfile["aoa"].size
        n_Re = npzfile["Re"].size

        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = n_aoa
        modeling_options["WISDEM"]["RotorSE"]["n_Re"] = n_Re
        modeling_options["WISDEM"]["RotorSE"]["n_tab"] = 1

        modeling_options["assembly"] = {}
        modeling_options["assembly"]["number_of_blades"] = 3

        n_span, n_aoa, n_Re, n_tab = np.moveaxis(npzfile["cl"][:, :, :, np.newaxis], 0, 1).shape
        modeling_options["airfoils"] = {}
        modeling_options["airfoils"]["n_aoa"] = n_aoa
        modeling_options["airfoils"]["n_Re"] = n_Re
        modeling_options["airfoils"]["n_tab"] = n_tab

        opt_options = {}
        opt_options["design_variables"] = {}
        opt_options["design_variables"]["blade"] = {}
        opt_options["design_variables"]["blade"]["aero_shape"] = {}
        opt_options["design_variables"]["blade"]["aero_shape"]["chord"] = {}
        opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"] = 8
        opt_options["design_variables"]["blade"]["aero_shape"]["twist"] = {}
        opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"] = 8
        opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["inverse"] = False
        opt_options["constraints"] = {}
        opt_options["constraints"]["blade"] = {}
        opt_options["constraints"]["blade"]["stall"] = {}
        opt_options["constraints"]["blade"]["stall"]["margin"] = 0.05233

        comp = CCBladeTwist(modeling_options=modeling_options, opt_options=opt_options)
        prob.model.add_subsystem("comp", comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("airfoils_aoa", npzfile["aoa"], units="deg")
        prob.set_val("airfoils_Re", npzfile["Re"])
        prob.set_val("airfoils_cl", np.moveaxis(npzfile["cl"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("airfoils_cd", np.moveaxis(npzfile["cd"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("airfoils_cm", np.moveaxis(npzfile["cm"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("r", npzfile["r"], units="m")
        prob.set_val("chord", npzfile["chord"], units="m")

        prob.set_val("Rhub", 1.0, units="m")
        prob.set_val("Rtip", 70.0, units="m")
        prob.set_val("hub_height", 100.0, units="m")
        prob.set_val("precone", 0.0, units="deg")
        prob.set_val("tilt", 0.0, units="deg")
        prob.set_val("yaw", 0.0, units="deg")
        prob.set_val("precurve", np.zeros(n_span), units="m")
        prob.set_val("precurveTip", 0.0, units="m")

        prob.set_val("rho", 1.225, units="kg/m**3")
        prob.set_val("mu", 1.81206e-5, units="kg/(m*s)")
        prob.set_val("shearExp", 0.25)
        prob.set_val("nBlades", 3)
        prob.set_val("nSector", 4)
        prob.set_val("tiploss", True)
        prob.set_val("hubloss", True)
        prob.set_val("wakerotation", True)
        prob.set_val("usecd", True)

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True, step=1e-7)

        # Manually filter some entries out of the assert_check_partials call.
        # Will want to add this functionality to OpenMDAO itself at some point.
        new_check = {}
        for comp_name in check:
            new_check[comp_name] = {}
            for (output_name, input_name) in check[comp_name]:
                if not input_name in ["airfoil", "rho", "mu"]:
                    new_check[comp_name][(output_name, input_name)] = check[comp_name][(output_name, input_name)]

        assert_check_partials(new_check)  # , rtol=5e-5, atol=1e-4)

    def test_ccblade_standalone(self):
        """"""
        prob = om.Problem()

        # Add some arbitrary inputs
        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(
            os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "smaller_dataset.npz", allow_pickle=True
        )
        n_span = npzfile["r"].size
        n_aoa = npzfile["aoa"].size
        n_Re = npzfile["Re"].size

        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = n_aoa
        modeling_options["WISDEM"]["RotorSE"]["n_Re"] = n_Re
        modeling_options["WISDEM"]["RotorSE"]["n_tab"] = 1

        modeling_options["assembly"] = {}
        modeling_options["assembly"]["number_of_blades"] = 3

        n_span, n_aoa, n_Re, n_tab = np.moveaxis(npzfile["cl"][:, :, :, np.newaxis], 0, 1).shape
        modeling_options["airfoils"] = {}
        modeling_options["airfoils"]["n_aoa"] = n_aoa
        modeling_options["airfoils"]["n_Re"] = n_Re
        modeling_options["airfoils"]["n_tab"] = n_tab

        comp = CCBladeEvaluate(modeling_options=modeling_options)
        prob.model.add_subsystem("comp", comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("airfoils_aoa", npzfile["aoa"], units="deg")
        prob.set_val("airfoils_Re", npzfile["Re"])
        prob.set_val("airfoils_cl", np.moveaxis(npzfile["cl"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("airfoils_cd", np.moveaxis(npzfile["cd"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("airfoils_cm", np.moveaxis(npzfile["cm"][:, :, :, np.newaxis], 0, 1))
        prob.set_val("r", npzfile["r"], units="m")
        prob.set_val("chord", npzfile["chord"], units="m")
        prob.set_val("theta", npzfile["theta"], units="deg")

        # parameters
        prob.set_val("V_load", 12.0, units="m/s")
        prob.set_val("Omega_load", 7.0, units="rpm")
        prob.set_val("pitch_load", 0.5, units="deg")

        prob.set_val("Rhub", 1.0, units="m")
        prob.set_val("Rtip", 70.0, units="m")
        prob.set_val("hub_height", 100.0, units="m")
        prob.set_val("precone", 0.1, units="deg")
        prob.set_val("tilt", 0.2, units="deg")
        prob.set_val("yaw", 0.2, units="deg")
        prob.set_val("precurve", np.linspace(0.0, 0.9, n_span), units="m")
        prob.set_val("precurveTip", 0.1, units="m")
        prob.set_val("presweep", np.linspace(0.0, 0.4, n_span), units="m")
        prob.set_val("presweepTip", 0.5, units="m")

        prob.set_val("rho", 1.225, units="kg/m**3")
        prob.set_val("mu", 1.81206e-5, units="kg/(m*s)")
        prob.set_val("shearExp", 0.25)
        prob.set_val("nBlades", 3)
        prob.set_val("nSector", 4)
        prob.set_val("tiploss", True)
        prob.set_val("hubloss", True)
        prob.set_val("wakerotation", True)
        prob.set_val("usecd", True)

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True)

        # Manually filter some entries out of the assert_check_partials call.
        # Will want to add this functionality to OpenMDAO itself at some point.
        new_check = {}
        for comp_name in check:
            new_check[comp_name] = {}
            for (output_name, input_name) in check[comp_name]:
                if "airfoil" not in input_name and "rho" not in input_name and "mu" not in input_name:
                    new_check[comp_name][(output_name, input_name)] = check[comp_name][(output_name, input_name)]

        assert_check_partials(new_check, rtol=5e-4, atol=50.0)


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
