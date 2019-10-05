import unittest

import test_WindWaveDrag
import test_enum
import test_environment
import test_frustum
import test_tube
import test_utilities
import test_utilizationSupplement
import test_vertical_cylinder

import numpy as np
import numpy.testing as npt

def suiteAll():
    suite = unittest.TestSuite( (test_WindWaveDrag.suite(),
                                 test_enum.suite(),
                                 test_environment.suite(),
                                 test_frustum.suite(),
                                 test_tube.suite(),
                                 test_utilities.suite(),
                                 test_utilizationSupplement.suite(),
                                 test_vertical_cylinder.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
