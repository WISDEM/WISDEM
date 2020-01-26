import unittest

from wisdem.test.test_assemblies import test_monopile_nodrive

def suite():
    suite = unittest.TestSuite( (test_monopile_nodrive.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
