import unittest

from wisdem.test.test_ccblade import test_ccblade

def suite():
    suite = unittest.TestSuite( (test_ccblade.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
