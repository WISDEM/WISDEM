import unittest

from wisdem.test.test_ccblade import test_ccblade
from wisdem.test.test_ccblade import test_gradients
from wisdem.test.test_ccblade import test_om_gradients

def suite():
    suite = unittest.TestSuite( (test_ccblade.suite(),
                                 test_gradients.suite(),
                                 test_om_gradients.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
