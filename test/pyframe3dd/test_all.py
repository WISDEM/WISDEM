import unittest

import test_frame

def suiteAll():
    suite = unittest.TestSuite( (test_frame.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
