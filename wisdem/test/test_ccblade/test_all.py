import unittest

from wisdem.test.test_ccblade import test_ccblade, test_gradients

def suite():
    suite = unittest.TestSuite( (test_ccblade.suite(),
                                 test_gradients.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
