import unittest

from airfoilprep import test_all as airfoilprep
import ccblade
import commonse
import drivetrainse
import floatingse
import nrelcsm
import pbeam
import plant_financese
import pyframe3dd
import pymap
import rotorse
import towerse
import turbinecostsse
import wisdem

def suiteAll():
    suite = unittest.TestSuite( (
        airfoilprep.test_all.suite(),
        ccblade.test_all.suite(),
        commonse.test_all.suite(),
        #drivetrainse.test_all.suite(),
        floatingse.test_all.suite(),
        #nrelcsm.test_all.suite(),
        pbeam.test_all.suite(),
        plant_financese.test_all.suite(),
        pyframe3dd.test_all.suite(),
        #pymap.test_all.suite(),
        #rotorse.test_all.suite(),
        #towerse.test_all.suite(),
        turbinecostsse.test_all.suite()
        #wisdem.test_all.suite()                                 
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
