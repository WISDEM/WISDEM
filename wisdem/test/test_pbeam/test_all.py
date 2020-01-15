import unittest

from . import test_pybeam

def suite():
    suite = unittest.TestSuite( (test_pybeam.suite() ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
