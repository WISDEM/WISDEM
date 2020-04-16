import unittest

from wisdem.test.test_rotorse import test_rotor_loads_defl_strains

def suite():
    suite = unittest.TestSuite( (test_rotor_loads_defl_strains.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
