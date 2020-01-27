import unittest

from wisdem.test.test_plant_financese import test_plantfinancese

def suite():
    suite = unittest.TestSuite( (test_plantfinancese.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
