import os
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from wisdem.commonse.distribution import WeibullCDF, RayleighCDF, WeibullWithMeanCDF

np.random.seed(314)


class Test(unittest.TestCase):
    def test_distributions(self):
        nspline = 10

        prob = om.Problem()

        prob.model.add_subsystem("comp1", WeibullCDF(nspline=nspline), promotes_inputs=["*"])
        prob.model.add_subsystem("comp2", WeibullWithMeanCDF(nspline=nspline), promotes_inputs=["*"])
        prob.model.add_subsystem("comp3", RayleighCDF(nspline=nspline), promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)

        # Add some arbitrary inputs
        prob.set_val("x", np.random.rand(nspline), units="m/s")
        prob.set_val("xbar", 1.5, units="m/s")
        prob.set_val("k", 1.5)
        prob.set_val("A", 1.2)

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True, method="fd")

        assert_check_partials(check, atol=1e-5)


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
