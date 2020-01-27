import unittest

from . import test_column
from . import test_map_mooring
from . import test_loading
from . import test_substructure
from . import test_floating

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
        
