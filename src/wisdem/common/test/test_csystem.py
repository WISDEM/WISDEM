#!/usr/bin/env python
# encoding: utf-8
"""
test_csystem.py

Created by Andrew Ning on 2013-05-10.
Copyright (c) NREL. All rights reserved.
"""

import unittest
from math import cos, sin, radians
import numpy as np

from twister.common.csystem import DirectionVector


class TestVector(unittest.TestCase):


    def setUp(self):

        self.x = np.random.sample(1)[0]*10.0
        self.y = np.random.sample(1)[0]*10.0
        self.z = np.random.sample(1)[0]*10.0

        self.v = DirectionVector(self.x, self.y, self.z)

        self.angle = np.random.sample(1)[0]*10.0


    def testInertialWind(self):

        vrot = self.v.inertialToWind(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) + self.y*sin(angle))
        self.assertEqual(vrot.y, -self.x*sin(angle) + self.y*cos(angle))
        self.assertEqual(vrot.z, self.z)


    def testWindInertial(self):

        vrot = self.v.windToInertial(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) - self.y*sin(angle))
        self.assertEqual(vrot.y, self.x*sin(angle) + self.y*cos(angle))
        self.assertEqual(vrot.z, self.z)


    def testWindYaw(self):

        vrot = self.v.windToYaw(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) + self.y*sin(angle))
        self.assertEqual(vrot.y, -self.x*sin(angle) + self.y*cos(angle))
        self.assertEqual(vrot.z, self.z)


    def testYawWind(self):

        vrot = self.v.yawToWind(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) - self.y*sin(angle))
        self.assertEqual(vrot.y, self.x*sin(angle) + self.y*cos(angle))
        self.assertEqual(vrot.z, self.z)


    def testYawHub(self):

        vrot = self.v.yawToHub(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) - self.z*sin(angle))
        self.assertEqual(vrot.y, self.y)
        self.assertEqual(vrot.z, self.x*sin(angle) + self.z*cos(angle))


    def testHubYaw(self):

        vrot = self.v.hubToYaw(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) + self.z*sin(angle))
        self.assertEqual(vrot.y, self.y)
        self.assertEqual(vrot.z, -self.x*sin(angle) + self.z*cos(angle))


    def testHubAzimuth(self):

        vrot = self.v.hubToAzimuth(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x)
        self.assertEqual(vrot.y, self.y*cos(angle) + self.z*sin(angle))
        self.assertEqual(vrot.z, -self.y*sin(angle) + self.z*cos(angle))


    def testAzimuthHub(self):

        vrot = self.v.azimuthToHub(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x)
        self.assertEqual(vrot.y, self.y*cos(angle) - self.z*sin(angle))
        self.assertEqual(vrot.z, self.y*sin(angle) + self.z*cos(angle))


    def testAzimuthBlade(self):

        vrot = self.v.azimuthToBlade(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) + self.z*sin(angle))
        self.assertEqual(vrot.y, self.y)
        self.assertEqual(vrot.z, -self.x*sin(angle) + self.z*cos(angle))


    def testBladeAzimuth(self):

        vrot = self.v.bladeToAzimuth(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) - self.z*sin(angle))
        self.assertEqual(vrot.y, self.y)
        self.assertEqual(vrot.z, self.x*sin(angle) + self.z*cos(angle))


    def testBladeAirfoil(self):

        vrot = self.v.bladeToAirfoil(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) - self.y*sin(angle))
        self.assertEqual(vrot.y, self.x*sin(angle) + self.y*cos(angle))
        self.assertEqual(vrot.z, self.z)


    def testAirfoilBlade(self):

        vrot = self.v.airfoilToBlade(self.angle)

        angle = radians(self.angle)
        self.assertEqual(vrot.x, self.x*cos(angle) + self.y*sin(angle))
        self.assertEqual(vrot.y, -self.x*sin(angle) + self.y*cos(angle))
        self.assertEqual(vrot.z, self.z)


    def testAirfoilProfile(self):

        vrot = self.v.airfoilToProfile()

        self.assertEqual(vrot.x, self.y)
        self.assertEqual(vrot.y, self.x)
        self.assertEqual(vrot.z, self.z)


    def testProfileAirfoil(self):

        vrot = self.v.profileToAirfoil()

        self.assertEqual(vrot.x, self.y)
        self.assertEqual(vrot.y, self.x)
        self.assertEqual(vrot.z, self.z)


    def testAdd(self):

        scalar = np.random.sample(1)[0]*10.0

        vout = self.v + scalar

        self.assertEqual(vout.x, self.x + scalar)
        self.assertEqual(vout.y, self.y + scalar)
        self.assertEqual(vout.z, self.z + scalar)


    def testAdd2(self):

        x = np.random.sample(1)[0]*10.0
        y = np.random.sample(1)[0]*10.0
        z = np.random.sample(1)[0]*10.0

        vout = self.v + DirectionVector(x, y, z)

        self.assertEqual(vout.x, self.x + x)
        self.assertEqual(vout.y, self.y + y)
        self.assertEqual(vout.z, self.z + z)


    def testSubtract(self):

        scalar = np.random.sample(1)[0]*10.0

        vout = self.v - scalar

        self.assertEqual(vout.x, self.x - scalar)
        self.assertEqual(vout.y, self.y - scalar)
        self.assertEqual(vout.z, self.z - scalar)


    def testSubtract2(self):

        x = np.random.sample(1)[0]*10.0
        y = np.random.sample(1)[0]*10.0
        z = np.random.sample(1)[0]*10.0

        vout = self.v - DirectionVector(x, y, z)

        self.assertEqual(vout.x, self.x - x)
        self.assertEqual(vout.y, self.y - y)
        self.assertEqual(vout.z, self.z - z)


    def testMultiply(self):

        scalar = np.random.sample(1)[0]*10.0

        vout = self.v * scalar

        self.assertEqual(vout.x, self.x * scalar)
        self.assertEqual(vout.y, self.y * scalar)
        self.assertEqual(vout.z, self.z * scalar)


    def testMultiply2(self):

        x = np.random.sample(1)[0]*10.0
        y = np.random.sample(1)[0]*10.0
        z = np.random.sample(1)[0]*10.0

        vout = self.v * DirectionVector(x, y, z)

        self.assertEqual(vout.x, self.x * x)
        self.assertEqual(vout.y, self.y * y)
        self.assertEqual(vout.z, self.z * z)


    def testDivide(self):

        scalar = np.random.sample(1)[0]*10.0

        vout = self.v / scalar

        self.assertEqual(vout.x, self.x / scalar)
        self.assertEqual(vout.y, self.y / scalar)
        self.assertEqual(vout.z, self.z / scalar)


    def testDivide2(self):

        x = np.random.sample(1)[0]*10.0
        y = np.random.sample(1)[0]*10.0
        z = np.random.sample(1)[0]*10.0

        vout = self.v / DirectionVector(x, y, z)

        self.assertEqual(vout.x, self.x / x)
        self.assertEqual(vout.y, self.y / y)
        self.assertEqual(vout.z, self.z / z)



    def testCross(self):

        v1 = DirectionVector(1.0, 2.0, 3.0)
        v2 = DirectionVector(4.0, 5.0, 7.0)

        v3 = v1.cross(v2)

        np.testing.assert_array_equal([v3.x, v3.y, v3.z], [-1.0, 5.0, -3.0])

        v1 = DirectionVector([1.0, 2.0], [2.0, 3.0], [3.0, 4.0])
        v2 = DirectionVector([4.0, 5.0], [5.0, 6.0], [7.0, 8.0])

        v3 = v1.cross(v2)

        np.testing.assert_array_equal([v3.x[0], v3.y[0], v3.z[0]], [-1.0, 5.0, -3.0])
        np.testing.assert_array_equal([v3.x[1], v3.y[1], v3.z[1]], [0.0, 4.0, -3.0])


if __name__ == '__main__':
    unittest.main()

