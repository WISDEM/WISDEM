import unittest

from wisdem.test.test_airfoilprep import test_airfoilprep

def suite():
    suite = unittest.TestSuite( (test_airfoilprep.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
