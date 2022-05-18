import os
import sys
import copy
import platform
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from wisdem.optimization_drivers.intermittent_component import IntermittentComponent

rosenbrock_size = 4
num_procs = 1


class Rosenbrock1(IntermittentComponent):
    def initialize(self):
        # This is required only if you have something to add to the initialize(),
        # such as OpenMDAO options declarations
        super().initialize()
        self.options.declare("multiplier", 1.0)

    def setup(self):
        # This is required to run the original setup() contained within IntermittentComponent
        super().setup()

        self.add_input("x", np.ones(rosenbrock_size))
        self.add_output("f1", 0.0)

    def internal_compute(self, inputs, outputs):
        """
        This is the actual method where the computations should occur.
        """
        x = inputs["x"]
        x_0 = x[:-1]
        outputs["f1"] = self.options["multiplier"] * sum((1 - x_0) ** 2)


class Rosenbrock2(IntermittentComponent):
    def initialize(self):
        # This is required only if you have something to add to the initialize(),
        # such as OpenMDAO options declarations
        super().initialize()
        self.options.declare("multiplier", 100.0)

    def setup(self):
        # This is required to run the original setup() contained within IntermittentComponent
        super().setup()

        self.add_input("x", np.ones(rosenbrock_size))
        self.add_output("f2", 0.0)

    def internal_compute(self, inputs, outputs):
        """
        This is the actual method where the computations should occur.
        """
        x = inputs["x"]
        x_0 = x[:-1]
        x_1 = x[1:]
        outputs["f2"] = self.options["multiplier"] * sum((x_1 - x_0**2) ** 2)


class TestIntermittentComponent(unittest.TestCase):
    def test_run(self):
        prob = om.Problem(model=om.Group(num_par_fd=num_procs))
        prob.model.approx_totals(method="fd")
        indeps = prob.model.add_subsystem("indeps", om.IndepVarComp(), promotes=["*"])
        indeps.add_output("x", 1.2 * np.ones(rosenbrock_size))

        prob.model.add_subsystem(
            "rosenbrock1",
            Rosenbrock1(num_iterations_between_calls=10, multiplier=1.0),
            promotes=["*"],
        )
        prob.model.add_subsystem(
            "rosenbrock2",
            Rosenbrock2(num_iterations_between_calls=1, multiplier=100.0),
            promotes=["*"],
        )
        prob.model.add_subsystem("objective_comp", om.ExecComp("f = f1 + f2"), promotes=["*"])

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"
        prob.driver.opt_settings["tol"] = 1e-9

        prob.model.add_design_var("x", lower=-1.5, upper=1.5)
        prob.model.add_objective("f")

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob["f"][0], 0.0, tolerance=1e-2)
        assert_near_equal(prob["x"], np.ones(4), tolerance=1e-1)
        self.assertEqual(prob.model.rosenbrock1.actual_compute_calls, 6)
        # Seems sensivity to backend SLSQP parameter tuning
        # self.assertEqual(prob.model.rosenbrock2.actual_compute_calls, 48)

        # minimum value
        print(f"Optimum found: {prob['f'][0]}")
        # location of the minimum
        print(f"Optimal design: {prob['x']}")
        print()
        print(
            f"Number of actual compute calls: {prob.model.rosenbrock1.actual_compute_calls} and {prob.model.rosenbrock2.actual_compute_calls}"
        )


if __name__ == "__main__":
    unittest.main()
