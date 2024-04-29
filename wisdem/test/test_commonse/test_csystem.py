import unittest

import numpy as np
import numpy.testing as npt

from wisdem.commonse.csystem import DirectionVector


class TestCSystem(unittest.TestCase):
    def test(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.3, 4.3])
        z = np.array([2.3, 2.3])
        a = DirectionVector(x, y, z)

        x = np.array([3.2, 1.5])
        y = np.array([2.1, 3.2])
        z = np.array([5.6, 7.7])
        b = DirectionVector(x, y, z)

        c = a.cross(b)
        dx, dy, dz = a.cross_deriv(b)
        dx, dy, dz = a.cross_deriv_array(b)

        d = -a
        e = a + b
        f = a - b
        f += a
        f -= a
        k = a * b
        p = a / b
        p *= b
        p /= a

        e = a + 1.2
        f = a - 1.2
        f += 2.0
        f -= 3.0
        k = a * 4.0
        p = a / 2.0
        p *= 3.0
        p /= 2.0

        T1 = 5.0
        T2 = 4.0
        T3 = -3.0
        tilt = 20.0

        F = DirectionVector(T1, T2, T3).hubToYaw(tilt)

        Fp1 = DirectionVector(T1 + 1e-6, T2, T3).hubToYaw(tilt)
        Fp2 = DirectionVector(T1, T2 + 1e-6, T3).hubToYaw(tilt)
        Fp3 = DirectionVector(T1, T2, T3 + 1e-6).hubToYaw(tilt)
        Fp4 = DirectionVector(T1, T2, T3).hubToYaw(tilt + 1e-6)

        dFx = F.dx
        dFy = F.dy
        dFz = F.dz

        npt.assert_almost_equal(dFx["dx"], (Fp1.x - F.x) / 1e-6)
        npt.assert_almost_equal(dFx["dy"], (Fp2.x - F.x) / 1e-6)
        npt.assert_almost_equal(dFx["dz"], (Fp3.x - F.x) / 1e-6)
        npt.assert_almost_equal(dFx["dtilt"], (Fp4.x - F.x) / 1e-6)

        npt.assert_almost_equal(dFy["dx"], (Fp1.y - F.y) / 1e-6)
        npt.assert_almost_equal(dFy["dy"], (Fp2.y - F.y) / 1e-6)
        npt.assert_almost_equal(dFy["dz"], (Fp3.y - F.y) / 1e-6)
        npt.assert_almost_equal(dFy["dtilt"], (Fp4.y - F.y) / 1e-6)

        npt.assert_almost_equal(dFz["dx"], (Fp1.z - F.z) / 1e-6)
        npt.assert_almost_equal(dFz["dy"], (Fp2.z - F.z) / 1e-6)
        npt.assert_almost_equal(dFz["dz"], (Fp3.z - F.z) / 1e-6)
        npt.assert_almost_equal(dFz["dtilt"], (Fp4.z - F.z) / 1e-6)

        yaw = 3.0
        F = DirectionVector(T1, T2, T3).hubToYaw(tilt).yawToWind(yaw)

        Fp1 = DirectionVector(T1 + 1e-6, T2, T3).hubToYaw(tilt).yawToWind(yaw)
        Fp2 = DirectionVector(T1, T2 + 1e-6, T3).hubToYaw(tilt).yawToWind(yaw)
        Fp3 = DirectionVector(T1, T2, T3 + 1e-6).hubToYaw(tilt).yawToWind(yaw)
        Fp4 = DirectionVector(T1, T2, T3).hubToYaw(tilt + 1e-6).yawToWind(yaw)
        Fp5 = DirectionVector(T1, T2, T3).hubToYaw(tilt).yawToWind(yaw + 1e-6)

        dFx = F.dx
        dFy = F.dy
        dFz = F.dz

        npt.assert_almost_equal(dFx["dx"], (Fp1.x - F.x) / 1e-6)
        npt.assert_almost_equal(dFx["dy"], (Fp2.x - F.x) / 1e-6)
        npt.assert_almost_equal(dFx["dz"], (Fp3.x - F.x) / 1e-6)
        npt.assert_almost_equal(dFx["dtilt"], (Fp4.x - F.x) / 1e-6)
        npt.assert_almost_equal(dFx["dyaw"], (Fp5.x - F.x) / 1e-6)

        npt.assert_almost_equal(dFy["dx"], (Fp1.y - F.y) / 1e-6)
        npt.assert_almost_equal(dFy["dy"], (Fp2.y - F.y) / 1e-6)
        npt.assert_almost_equal(dFy["dz"], (Fp3.y - F.y) / 1e-6)
        npt.assert_almost_equal(dFy["dtilt"], (Fp4.y - F.y) / 1e-6)
        npt.assert_almost_equal(dFy["dyaw"], (Fp5.y - F.y) / 1e-6)

        npt.assert_almost_equal(dFz["dx"], (Fp1.z - F.z) / 1e-6)
        npt.assert_almost_equal(dFz["dy"], (Fp2.z - F.z) / 1e-6)
        npt.assert_almost_equal(dFz["dz"], (Fp3.z - F.z) / 1e-6)
        npt.assert_almost_equal(dFz["dtilt"], (Fp4.z - F.z) / 1e-6)
        npt.assert_almost_equal(dFz["dyaw"], (Fp5.z - F.z) / 1e-6)

        inertial_F = F.windToInertial(20.0)
        new_F = inertial_F.inertialToWind(20.0)
        npt.assert_almost_equal(F.toArray(), new_F.toArray())

        wind_F = F.yawToWind(15.0)
        new_F = wind_F.windToYaw(15.0)
        npt.assert_almost_equal(F.toArray(), new_F.toArray())

        yaw_F = F.hubToYaw(12.0)
        new_F = yaw_F.yawToHub(12.0)
        npt.assert_almost_equal(F.toArray(), new_F.toArray())

        az_F = F.hubToAzimuth(12.0)
        new_F = az_F.azimuthToHub(12.0)
        npt.assert_almost_equal(F.toArray(), new_F.toArray())

        blade_F = F.azimuthToBlade(12.0)
        new_F = blade_F.bladeToAzimuth(12.0)
        npt.assert_almost_equal(F.toArray(), new_F.toArray())

        blade_F = F.airfoilToBlade(12.0)
        new_F = blade_F.bladeToAirfoil(12.0)
        npt.assert_almost_equal(F.toArray(), new_F.toArray())

        profile_F = F.airfoilToProfile()
        new_F = profile_F.profileToAirfoil()
        npt.assert_almost_equal(F.toArray(), new_F.toArray())


if __name__ == "__main__":
    unittest.main()
