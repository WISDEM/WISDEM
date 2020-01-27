import unittest

from . import test_plantfinancese

def suite():
    suite = unittest.TestSuite( (test_plantfinancese.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
