import unittest

from wisdem.test.test_towerse import test_tower

def suite():
    suite = unittest.TestSuite( (test_tower.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
