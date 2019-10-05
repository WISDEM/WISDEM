import unittest

import test_ccblade

def suiteAll():
    suite = unittest.TestSuite( (test_ccblade.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
