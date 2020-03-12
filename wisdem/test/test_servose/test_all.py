import unittest

from wisdem.test.test_servose import test_servose

def suite():
    suite = unittest.TestSuite( (test_servose.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
