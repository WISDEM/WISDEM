import unittest

from wisdem.test.test_pbeam import test_pybeam

def suite():
    suite = unittest.TestSuite( (test_pybeam.suite() ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
