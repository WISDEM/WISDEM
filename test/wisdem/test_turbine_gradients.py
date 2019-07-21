#!/usr/bin/env python
# encoding: utf-8
"""
test_turbine_gradients.py

Created by Andrew Ning on 2013-02-10.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from wisdem.commonse.utilities import check_gradient_unit_test, check_for_missing_unit_tests
from wisdem.turbinese.turbine import MaxTipDeflection


class TestMaxTipDeflection(unittest.TestCase):

    def test1(self):

        dfl = MaxTipDeflection()
        dfl.Rtip = 63.0
        dfl.precurveTip = 5.0
        dfl.presweepTip = 2.0
        dfl.precone = 2.5
        dfl.tilt = 5.0
        dfl.hub_tt = np.array([-6.29400379597, 0.0, 3.14700189798])
        dfl.tower_z = np.array([0.0, 0.5, 1.0])
        dfl.tower_d = np.array([6.0, 4.935, 3.87])
        dfl.towerHt = 77.5632866084

        check_gradient_unit_test(self, dfl)


    def test2(self):

        dfl = MaxTipDeflection()
        dfl.Rtip = 63.0
        dfl.precurveTip = 5.0
        dfl.presweepTip = 2.0
        dfl.precone = -2.5
        dfl.tilt = 5.0
        dfl.hub_tt = np.array([-6.29400379597, 0.0, 3.14700189798])
        dfl.tower_z = np.array([0.0, 0.5, 1.0])
        dfl.tower_d = np.array([6.0, 4.935, 3.87])
        dfl.towerHt = 77.5632866084

        check_gradient_unit_test(self, dfl)




if __name__ == '__main__':
    import wisdem.turbinese.turbine

    check_for_missing_unit_tests([wisdem.turbinese.turbine])
    unittest.main()

