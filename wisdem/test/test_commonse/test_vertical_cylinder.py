import unittest

import numpy as np
import numpy.testing as npt
import wisdem.commonse.vertical_cylinder as vc


def suite():
    suite = unittest.TestSuite()
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
