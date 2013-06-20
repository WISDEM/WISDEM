#!/usr/bin/env python
# encoding: utf-8
"""
toweraero_mdao.py

Created by Andrew Ning on 2013-05-29.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.datatypes.api import Array, Float

from components import TowerAeroBase
from twister.tower import MonopileAero, WindWithPowerProfile, \
    WindWithLogProfile, LinearWaves

# TODO: add log profile option


class TowerAero(TowerAeroBase):

    z = Array(iotype='in', units='m', desc='height locations along tower starting from base')
    d = Array(iotype='in', units='m', desc='corresponding diameter along tower')
    z_surface = Float(iotype='in', units='m', desc='z-location for surface of water')

    hs = Float(iotype='in', units='m', desc='significant wave height')
    T = Float(iotype='in', units='s', desc='wave period')
    Uc = Float(iotype='in', desc='magnitue of current speed')


    def execute(self):

        # TODO: options for other wind profile, and other parameters
        wind = WindWithPowerProfile(self.atm.shearExp, self.atm.rho, self.atm.mu)
        wave = LinearWaves(self.hs, self.T)

        tower = MonopileAero(self.z, self.d, self.z_surface, wind, wave)

        wwl = self.wind_wave_loads
        wwl.z, wwl.Px, wwl.Py, wwl.Pz, wwl.q = tower.distributedLoads(self.Uhub, self.Uc, self.yaw)



if __name__ == '__main__':

    t = TowerAero()

    t.z = [0, 30.0, 73.8, 117.6]
    t.d = [6.0, 6.0, 4.935, 3.87]
    t.z_surface = 20.0

    t.hs = 7.5   # 7.5 is 10 year extreme             # significant wave height (m)
    t.T = 19.6                # wave period (s)

    t.Uhub = 50.0            # reference wind speed (m/s) - Class I IEC (0.2 X is average for AEP, Ve50 = 1.4*Vref*(z/zhub)^.11 for the blade, max thrust at rated power for tower)
    t.Uc = 1.2
    t.shearExp = 0.2

    t.rho = 1.22
    t.mu = 1.81206e-5

    t.execute()

    loads = t.wind_wave_loads
    import matplotlib.pyplot as plt
    plt.plot(loads.Px, loads.z)
    plt.plot(loads.Py, loads.z)
    plt.plot(loads.Pz, loads.z)
    plt.show()

    from myutilities import printArray
    printArray(loads.Px, 'Px')
    printArray(loads.z, 'z')
    printArray(loads.q, 'q')




