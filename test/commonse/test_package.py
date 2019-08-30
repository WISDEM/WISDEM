import unittest

import WindWaveDrag_PyU
import enum_PyU
import environment_PyU
import frustum_PyU
import tube_PyU
import utilities_PyU
import utilizationSupplement_PyU
import vertical_cylinder_PyU

import numpy as np
import numpy.testing as npt

def suiteAll():
    suite = unittest.TestSuite( (WindWaveDrag_PyU.suite(),
                                 enum_PyU.suite(),
                                 environment_PyU.suite(),
                                 frustum_PyU.suite(),
                                 tube_PyU.suite(),
                                 utilities_PyU.suite(),
                                 utilizationSupplement_PyU.suite(),
                                 vertical_cylinder_PyU.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
