import unittest

from wisdem.test.test_commonse import test_WindWaveDrag
from wisdem.test.test_commonse import test_enum
from wisdem.test.test_commonse import test_environment
from wisdem.test.test_commonse import test_frustum
from wisdem.test.test_commonse import test_tube
from wisdem.test.test_commonse import test_utilities
from wisdem.test.test_commonse import test_utilizationSupplement
from wisdem.test.test_commonse import test_vertical_cylinder

import numpy as np
import numpy.testing as npt

def suite():
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
    unittest.TextTestRunner().run(suite())
        
