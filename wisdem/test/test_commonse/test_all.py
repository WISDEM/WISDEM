import unittest

from . import test_WindWaveDrag
from . import test_enum
from . import test_environment
from . import test_frustum
from . import test_tube
from . import test_utilities
from . import test_utilizationSupplement
from . import test_vertical_cylinder

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
        
