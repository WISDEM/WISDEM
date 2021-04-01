import unittest

import numpy as np

from openmdao.utils.assert_utils import assert_near_equal
from wisdem.optimization_drivers.dakota_driver import DakotaOptimizer

try:
    import dakota
except ImportError:
    dakota = None


@unittest.skipIf(dakota is None, "only run if Dakota is installed.")
class TestDakotaOptimization(unittest.TestCase):
    def test_2D_opt_max_iterations(self):
        bounds = {"x": np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {"x": np.array([0.0, 0.25])}
        outputs = ["y"]
        template_dir = "template_dir/"
        model_string = "from weis.multifidelity.models.testbed_components import simple_2D_high_model as model"
        output_scalers = [1.0]
        options = {"method": "coliny_cobyla", "max_function_evaluations": 3}

        opt = DakotaOptimizer(template_dir)
        results = opt.optimize(desvars, outputs, bounds, model_string, output_scalers, options)

        assert_near_equal(np.min(np.array(results["y"])), -9.5)

    def test_2D_opt_EGO(self):
        bounds = {"x": np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {"x": np.array([0.0, 0.25])}
        outputs = ["y"]
        template_dir = "template_dir/"
        model_string = "from weis.multifidelity.models.testbed_components import simple_2D_high_model as model"
        output_scalers = [1.0]
        options = {"initial_samples": 5, "method": "efficient_global", "seed": 123456}

        opt = DakotaOptimizer(template_dir)
        results = opt.optimize(desvars, outputs, bounds, model_string, output_scalers, options)

        assert_near_equal(np.min(np.array(results["y"])), -9.999996864)

    def test_two_variables(self):
        bounds = {"x": np.array([[0.0, 1.0], [0.0, 1.0]]), "z": [1.0, 2.0]}
        desvars = {"x": np.array([0.0, 0.25]), "z": 1.5}
        outputs = ["y"]
        template_dir = "template_dir/"
        model_string = "from weis.multifidelity.models.testbed_components import simple_two_variable as model"
        output_scalers = [1.0]
        options = {"method": "coliny_cobyla", "max_function_evaluations": 3}

        opt = DakotaOptimizer(template_dir)
        results = opt.optimize(desvars, outputs, bounds, model_string, output_scalers, options)

        assert_near_equal(np.min(np.array(results["y"])), 1.0)

    def test_constraint(self):
        bounds = {"x": np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {"x": np.array([0.0, 0.25])}
        outputs = ["y", "con"]
        template_dir = "template_dir/"
        model_string = "from weis.multifidelity.models.testbed_components import simple_2D_low_model as model"
        output_scalers = [1.0, 1.0]
        options = {"method": "coliny_cobyla", "max_function_evaluations": 3}

        opt = DakotaOptimizer(template_dir)
        results = opt.optimize(desvars, outputs, bounds, model_string, output_scalers, options)

        assert_near_equal(np.min(np.array(results["y"])), 0.5)
        assert_near_equal(np.min(np.array(results["con"])), 0.0)


if __name__ == "__main__":
    unittest.main()
