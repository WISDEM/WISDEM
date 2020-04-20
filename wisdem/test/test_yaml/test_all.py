import unittest

from wisdem.test.test_yaml import test_monopile_derivs
from wisdem.test.test_yaml import test_tower_derivs

import numpy as np
import numpy.testing as npt

def suite():
    suite = unittest.TestSuite( (test_monopile_derivs.suite(),
                                 test_tower_derivs.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
