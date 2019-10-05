import unittest

import test_column
import test_map_mooring
import test_loading
import test_substructure
import test_floating

def suiteAll():
    suite = unittest.TestSuite( (test_column.suite(),
                                 test_map_mooring.suite(),
                                 test_loading.suite(),
                                 test_substructure.suite(),
                                 test_floating.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
