import unittest

import test_plantfinancese

def suiteAll():
    suite = unittest.TestSuite( (test_plantfinancese.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
