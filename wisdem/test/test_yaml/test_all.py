import unittest

from wisdem.test.test_yaml import test_grid_derivs

import numpy as np
import numpy.testing as npt

def suite():
    suite = unittest.TestSuite( (test_grid_derivs.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
