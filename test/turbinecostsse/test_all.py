import unittest

import test_turbine_costsse_2015

def suiteAll():
    suite = unittest.TestSuite( (test_turbine_costsse_2015.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
