import unittest

import wisdem.test.test_airfoilprep as test_airfoilprep
import wisdem.test.test_ccblade as test_ccblade
import wisdem.test.test_commonse as test_commonse
import wisdem.test.test_drivetrainse as test_drivetrainse
import wisdem.test.test_floatingse as test_floatingse
import wisdem.test.test_nrelcsm as test_nrelcsm
import wisdem.test.test_pbeam as test_pbeam
import wisdem.test.test_plant_financese as test_plant_financese
import wisdem.test.test_pyframe3dd as test_pyframe3dd
import wisdem.test.test_pymap as test_pymap
import wisdem.test.test_rotorse as test_rotorse
import wisdem.test.test_towerse as test_towerse
import wisdem.test.test_turbinecostsse as test_turbinecostsse
import wisdem.test.test_wisdem as test_wisdem

def suite():
    suite = unittest.TestSuite( (
        test_airfoilprep.test_all.suite(),
        test_ccblade.test_all.suite(),
        test_commonse.test_all.suite(),
        #test_drivetrainse.test_all.suite(),
        test_floatingse.test_all.suite(),
        #test_nrelcsm.test_all.suite(),
        test_pbeam.test_all.suite(),
        test_plant_financese.test_all.suite(),
        test_pyframe3dd.test_all.suite(),
        #test_pymap.test_all.suite(),
        #test_rotorse.test_all.suite(),
        #test_towerse.test_all.suite(),
        test_turbinecostsse.test_all.suite()
        #test_wisdem.test_all.suite()                                 
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
