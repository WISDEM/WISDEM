"""Tests for the NSGA2Driver and its integration with gc_PoseOptimization schema fields."""

import unittest
import numpy as np
import openmdao.api as om

from wisdem.optimization_drivers.nsga2_driver import NSGA2Driver


def _make_two_obj_problem():
    """Build a simple 2-objective OpenMDAO problem for testing NSGA2Driver.

    Minimize f1 = x^2  and  f2 = (x - 2)^2  subject to x in [0, 3].
    Pareto front: x in [0, 2], trading off f1 vs f2.
    """

    class TwoObj(om.ExplicitComponent):
        def setup(self):
            self.add_input("x", val=1.0)
            self.add_output("f1", val=0.0)
            self.add_output("f2", val=0.0)

        def compute(self, inputs, outputs):
            x = inputs["x"][0]
            outputs["f1"] = x ** 2
            outputs["f2"] = (x - 2.0) ** 2

    prob = om.Problem()
    prob.model.add_subsystem("comp", TwoObj(), promotes=["*"])
    prob.model.add_design_var("x", lower=0.0, upper=3.0)
    prob.model.add_objective("f1")
    prob.model.add_objective("f2")
    return prob


def _make_constrained_problem():
    """Build the 2-objective problem with scaled bounds and constraints of every kind.

    Design variable x in [0, 3] uses ref=10, so scaled and unscaled bounds differ.
    Constraints, all functions of x, give a feasible region of x in [0.6, 1.4] set by g_two:
      g_up  = x         upper=1.5  (one-sided upper, ref=2 so it is scaled)
      g_lo  = x         lower=0.5  (one-sided lower)
      g_two = 2 x       in [1.2, 2.8]  (double-sided)
      g_arr = [x, x+1]  upper=[1.5, 2.5]  (array)
    """

    class TwoObjConstrained(om.ExplicitComponent):
        def setup(self):
            self.add_input("x", val=1.0)
            self.add_output("f1", val=0.0)
            self.add_output("f2", val=0.0)
            self.add_output("g_up", val=0.0)
            self.add_output("g_lo", val=0.0)
            self.add_output("g_two", val=0.0)
            self.add_output("g_arr", val=np.zeros(2))

        def compute(self, inputs, outputs):
            x = inputs["x"][0]
            outputs["f1"] = x**2
            outputs["f2"] = (x - 2.0) ** 2
            outputs["g_up"] = x
            outputs["g_lo"] = x
            outputs["g_two"] = 2.0 * x
            outputs["g_arr"] = [x, x + 1.0]

    prob = om.Problem()
    prob.model.add_subsystem("comp", TwoObjConstrained(), promotes=["*"])
    prob.model.add_design_var("x", lower=0.0, upper=3.0, ref=10.0)
    prob.model.add_objective("f1")
    prob.model.add_objective("f2")
    prob.model.add_constraint("g_up", upper=1.5, ref=2.0)
    prob.model.add_constraint("g_lo", lower=0.5)
    prob.model.add_constraint("g_two", lower=1.2, upper=2.8)
    prob.model.add_constraint("g_arr", upper=[1.5, 2.5])
    return prob


class TestNSGA2DriverRun(unittest.TestCase):
    """Integration tests: run the driver on a minimal 2-objective problem."""

    def _run(self, pop_size=10, max_gen=2, Pc=0.9, Pm=0.1):
        prob = _make_two_obj_problem()
        prob.driver = NSGA2Driver()
        prob.driver.options["pop_size"] = pop_size
        prob.driver.options["max_gen"] = max_gen
        prob.driver.options["Pc"] = Pc
        prob.driver.options["Pm"] = Pm
        prob.setup()
        prob.run_driver()
        return prob

    def test_run_completes(self):
        """Driver runs to completion without error."""
        prob = self._run()
        self.assertIsNotNone(prob["x"])

    def test_pareto_front_populated(self):
        """desvar_nd and obj_nd are always populated after run."""
        prob = self._run()
        driver = prob.driver
        self.assertTrue(hasattr(driver, "desvar_nd"), "desvar_nd not set")
        self.assertTrue(hasattr(driver, "obj_nd"), "obj_nd not set")
        self.assertGreater(len(driver.desvar_nd), 0)
        self.assertGreater(len(driver.obj_nd), 0)

    def test_pop_size_sets_population(self):
        """pop_size controls the initial population size."""
        for size in [10, 20]:
            prob = _make_two_obj_problem()
            prob.driver = NSGA2Driver()
            prob.driver.options["pop_size"] = size
            prob.driver.options["max_gen"] = 1
            prob.setup()
            prob.run_driver()
            self.assertEqual(prob.driver.population_init.shape[0], size)

    def test_odd_pop_size_raises(self):
        """Odd pop_size raises a clear ValueError before entering the algorithm."""
        prob = _make_two_obj_problem()
        prob.driver = NSGA2Driver()
        prob.driver.options["pop_size"] = 11
        prob.setup()
        with self.assertRaises(ValueError, msg="pop_size=11"):
            prob.run_driver()

    def test_pareto_front_feasible(self):
        """All points on the Pareto front satisfy the design variable bounds."""
        # pop_size must be even for NSGA2 SBX crossover
        prob = self._run(pop_size=16, max_gen=3)
        driver = prob.driver
        for row in driver.desvar_nd:
            self.assertGreaterEqual(row[0], 0.0 - 1e-8)
            self.assertLessEqual(row[0], 3.0 + 1e-8)

    def test_unbounded_design_var_raises(self):
        """A design variable without both bounds is rejected at setup."""
        prob = om.Problem()
        prob.model.add_subsystem("comp", om.ExecComp(["f1 = x**2", "f2 = (x - 2)**2"]), promotes=["*"])
        prob.model.add_design_var("x", lower=0.0)
        prob.model.add_objective("f1")
        prob.model.add_objective("f2")
        prob.driver = NSGA2Driver()
        with self.assertRaises(ValueError):
            prob.setup()
            prob.final_setup()

    def test_equality_constraint_raises(self):
        """An equality constraint is rejected at setup instead of being silently ignored."""
        prob = om.Problem()
        prob.model.add_subsystem("comp", om.ExecComp(["f1 = x**2", "f2 = (x - 2)**2", "h = x"]), promotes=["*"])
        prob.model.add_design_var("x", lower=0.0, upper=3.0)
        prob.model.add_objective("f1")
        prob.model.add_objective("f2")
        prob.model.add_constraint("h", equals=1.0)
        prob.driver = NSGA2Driver()
        with self.assertRaisesRegex(ValueError, "Equality constraint 'h'"):
            prob.setup()
            prob.final_setup()


class TestNSGA2DriverConstraints(unittest.TestCase):
    """Constraint and bound handling with scaled variables. OpenMDAO >= 3.45.1 leaves
    unbounded sides as None, keeps meta bounds unscaled, and _set_design_var no
    longer unscales."""

    @classmethod
    def setUpClass(cls):
        prob = _make_constrained_problem()
        prob.driver = NSGA2Driver()
        prob.driver.options["pop_size"] = 20
        prob.driver.options["max_gen"] = 5
        prob.setup()
        prob.run_driver()
        cls.prob = prob

    def test_initial_population_within_bounds(self):
        """The initial population is sampled within the (unscaled) design variable bounds."""
        pop = self.prob.driver.population_init
        self.assertGreaterEqual(pop.min(), 0.0 - 1e-12)
        self.assertLessEqual(pop.max(), 3.0 + 1e-12)
        self.assertGreater(pop.max(), 0.3)  # not squeezed into the scaled range

    def test_constraint_count(self):
        """Every bounded side of every constraint element reaches the algorithm:
        g_up (1) + g_lo (1) + g_two (2 sides) + g_arr (2 elements) = 6."""
        self.assertEqual(self.prob.driver.optimizer_nsga2.N_constr, 6)

    def test_pareto_values_match_designs(self):
        """Stored objectives and constraints on the Pareto front belong to their design
        points (constraints ordered upper sides then lower sides, feasible when >= 0)."""
        driver = self.prob.driver
        x = driver.desvar_nd[:, 0]
        np.testing.assert_allclose(driver.obj_nd, np.column_stack([x**2, (x - 2.0) ** 2]), atol=1e-10)
        expected_constr = np.column_stack([1.5 - x, x - 0.5, 2.8 - 2.0 * x, 2.0 * x - 1.2, 1.5 - x, 1.5 - x])
        np.testing.assert_allclose(driver.constr_nd, expected_constr, atol=1e-10)

    def test_pareto_front_feasible(self):
        """Pareto points and the final design satisfy all constraints (x in [0.6, 1.4]),
        so the design values the driver sets match the ones it evaluates."""
        x_front = self.prob.driver.desvar_nd[:, 0]
        self.assertTrue(np.all(x_front >= 0.6 - 1e-8), x_front)
        self.assertTrue(np.all(x_front <= 1.4 + 1e-8), x_front)
        self.assertGreaterEqual(self.prob["x"][0], 0.6 - 1e-8)
        self.assertLessEqual(self.prob["x"][0], 1.4 + 1e-8)

    def test_copied_designs_not_reevaluated(self):
        """Without crossover or mutation every child is a copy of an evaluated row, so its
        stored values carry over: only the initial population and the final design run."""
        prob = _make_constrained_problem()
        prob.driver = NSGA2Driver()
        prob.driver.options["pop_size"] = 20
        prob.driver.options["max_gen"] = 3
        prob.driver.options["Pc"] = 0.0
        prob.driver.options["Pm"] = 0.0
        prob.setup()
        prob.run_driver()
        driver = prob.driver
        self.assertEqual(driver.iter_count, 20 + 1)
        x = driver.desvar_nd[:, 0]
        np.testing.assert_allclose(driver.obj_nd, np.column_stack([x**2, (x - 2.0) ** 2]), atol=1e-10)
        self.assertTrue(np.all((x >= 0.6 - 1e-8) & (x <= 1.4 + 1e-8)), x)


class TestNSGA2PoseOptimizationIntegration(unittest.TestCase):
    """Verify that gc_PoseOptimization._set_optimizer_properties correctly maps
    schema fields to NSGA2Driver options."""

    def test_set_optimizer_properties(self):
        """_set_optimizer_properties transfers pop_size, Pc, Pm, run_parallel from schema to driver options."""
        from wisdem.glue_code.gc_PoseOptimization import PoseOptimization

        opt = {
            "general": {"folder_output": "outputs"},
            "driver": {
                "optimization": {
                    "flag": True,
                    "solver": "NSGA2",
                    "tol": 1e-6,
                    "max_iter": 10,
                    "step_size": 1e-3,
                    "form": "central",
                    "step_calc": "None",
                    "debug_print": False,
                    "pop_size": 30,
                    "max_gen": 2,
                    "run_parallel": False,
                    "Pc": 0.75,
                    "Pm": 0.05,
                },
                "design_of_experiments": {"flag": False},
            },
        }

        pose = PoseOptimization(wt_init={}, modeling_options={}, analysis_options=opt)

        prob = _make_two_obj_problem()
        prob.driver = NSGA2Driver()
        prob = pose._set_optimizer_properties(prob, options_keys=["pop_size", "max_gen", "run_parallel", "Pc", "Pm"])

        self.assertEqual(prob.driver.options["pop_size"], 30)
        self.assertEqual(prob.driver.options["max_gen"], 2)
        self.assertFalse(prob.driver.options["run_parallel"])
        self.assertAlmostEqual(prob.driver.options["Pc"], 0.75)
        self.assertAlmostEqual(prob.driver.options["Pm"], 0.05)


if __name__ == "__main__":
    unittest.main()
