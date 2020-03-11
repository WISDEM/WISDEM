import unittest

from wisdem.test.test_rotorse import test_rotor_aero

def suite():
    suite = unittest.TestSuite( (test_rotor_aero.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
