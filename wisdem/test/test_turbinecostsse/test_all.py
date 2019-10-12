import unittest

from . import test_turbine_costsse_2015


def suite():
    suite = unittest.TestSuite( (test_turbine_costsse_2015.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
