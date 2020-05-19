import unittest

from wisdem.test.test_turbinecostsse import test_turbine_costsse_2015
from wisdem.test.test_turbinecostsse import test_nrel_csm_tcc_2015


def suite():
    suite = unittest.TestSuite( (test_turbine_costsse_2015.suite(),
                                 test_nrel_csm_tcc_2015.suite(),
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
        
