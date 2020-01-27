import unittest

from wisdem.test.test_pyframe3dd import test_frame

def suite():
    suite = unittest.TestSuite( (test_frame.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
