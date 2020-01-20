import unittest

from . import test_towerse

def suite():
    suite = unittest.TestSuite( (test_towerse.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
