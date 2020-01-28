import unittest

from wisdem.test.test_floatingse import test_column
from wisdem.test.test_floatingse import test_map_mooring
from wisdem.test.test_floatingse import test_loading
from wisdem.test.test_floatingse import test_substructure
from wisdem.test.test_floatingse import test_floating

def suite():
    suite = unittest.TestSuite( (test_column.suite(),
                                 test_map_mooring.suite(),
                                 test_loading.suite(),
                                 test_substructure.suite(),
                                 test_floating.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
