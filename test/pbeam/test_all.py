import unittest

import test_pybeam

def suiteAll():
    suite = unittest.TestSuite( (test_pybeam.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
