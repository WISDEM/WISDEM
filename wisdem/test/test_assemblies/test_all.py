import unittest

from wisdem.test.test_assemblies import test_assembly

def suite():
    suite = unittest.TestSuite( (test_assembly.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
