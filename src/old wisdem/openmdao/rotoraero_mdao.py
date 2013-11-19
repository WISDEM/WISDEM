#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero_mdao.py

Created by Andrew Ning on 2013-05-16.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
import os

from openmdao.main.datatypes.api import Array, Float, Int, List, Str, Enum

from components import RotorAeroBase
from wisdem.rotor import RotorAero, CCBlade, WTPerf, Airfoil, NRELCSMDrivetrain



class RotorAeroComp(RotorAeroBase):

    # geometry
    Rhub = Float(iotype='in', units='m', desc='hub radius')
    Rtip = Float(iotype='in', units='m', desc='tip radius')
    r = Array(iotype='in', units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
    chord = Array(iotype='in', units='m', desc='chord length at each section')
    theta = Array(iotype='in', units='deg', desc='twist angle at each section (positive decreases angle of attack)')
    af_file = List(Str, iotype='in', desc='names of airfoil file')
    af_path = Str(iotype='in', desc='path to directory containing airfoil files')

    nSector = Int(iotype='in', desc='number of azimuthal sectors to discretize for aero analysis')

    # analysis types
    aeroanalysis_type = Enum('ccblade', ('ccblade', 'wtperf'), iotype='in', desc='aerodynamic analysis code')
    drivetrain_type = Enum('geared', ('geared', 'single-stage', 'multi-drive', 'pm-direct-drive'), iotype='in', desc='drivetrain type')
    machine_type = Enum('VSVP', ('VSVP', 'FSVP', 'VSFP', 'FSFP'), iotype='in', desc='variable/fixed speed, variable/fixed pitch')
    wind_turbine_class = Enum('I', ('I', 'II', 'III'), iotype='in', desc='wind turbine class')

    # control parameters
    Vin = Float(iotype='in', units='m/s', desc='cut-in speed')
    Vout = Float(iotype='in', units='m/s', desc='cut-out speed')
    ratedPower = Float(iotype='in', units='W', desc='rated power')

    # if fixed speed
    Omega = Float(iotype='in', units='rpm', desc='fixed rotor rotation speed')

    # if variable speed
    minOmega = Float(iotype='in', units='rpm', desc='minimum rotor rotation speed')
    maxOmega = Float(iotype='in', units='rpm', desc='maximum rotor rotation speed')
    tsr_r2 = Float(iotype='in', desc='nominal tip-speed ratio in region 2 (assumed to be optimized externally)')

    # if fixed pitch
    pitch = Float(iotype='in', units='deg', desc='fixed pitch setting')

    # worst-case settings for extreme loading conditions
    pitch_extreme = Float(iotype='in', units='deg', desc='worst-case blade pitch setting')
    azimuth_extreme = Float(iotype='in', units='deg', desc='worst-case blade azimuth location')

    # Umean = Float(10.0, iotype='in', units='m/s', desc='average wind speed (Rayleigh distribution)')
    # V_extreme = Float(iotype='in')


    # out
    avgOmega = Float(iotype='out', units='rpm', desc='average rotation speed across wind speed distribution')



    def __init__(self):
        super(RotorAeroComp, self).__init__()


    def execute(self):

        # initialize airfoils
        afinit = Airfoil.initFromAerodynFile
        n = len(self.af_file)
        af = [0]*n

        for i in range(n):
            af[i] = afinit(self.af_path + os.path.sep + self.af_file[i])

        # add twist
        twist_struc = np.interp(self.r, self.r_structural, self.twist_structural)
        theta = self.theta + twist_struc


        # setup wind speeds based on IEC standards
        if self.wind_turbine_class == 'I':
            Vref = 50.0
        elif self.wind_turbine_class == 'II':
            Vref = 42.5
        else:
            Vref = 37.5

        Vmean = 0.2*Vref
        PDF = RotorAero.RayleighPDF(Vmean)
        CDF = RotorAero.RayleighCDF(Vmean)
        # shearExp = 0.2

        V_extreme = 1.4*Vref


        # initialize aeroanalysis
        vars = (self.r, self.chord, theta, af, self.Rhub, self.Rtip, self.B, self.atm.rho, self.atm.mu,
                self.precone, self.tilt, self.yaw, self.atm.shearExp, self.hubHt, self.nSector)


        if self.aeroanalysis_type == 'ccblade':
            aeroanalysis = CCBlade(*vars)
        elif self.aeroanalysis_type == 'wtperf':
            aeroanalysis = WTPerf(*vars)

        # initialize drivetrain
        drivetrain = NRELCSMDrivetrain(self.drivetrain_type)

        # machine type

        if self.machine_type == 'VSVP':
            opt = RotorAero.externalOpt(self.tsr_r2, pitch=0.0)  # pitch is irrelevant
            mtype = RotorAero.VSVP(self.Vin, self.Vout, self.ratedPower, self.minOmega, self.maxOmega, opt)

        elif self.machine_type == 'VSFP':
            opt = RotorAero.externalOpt(self.tsr_r2, self.pitch)
            mtype = RotorAero.VSFP(self.Vin, self.Vout, self.ratedPower, self.minOmega, self.maxOmega, self.pitch, opt)

        elif self.machine_type == 'FSVP':
            opt = RotorAero.externalOpt(tsr=7.0, pitch=0.0)  # these terms are irrelevant
            mtype = RotorAero.FSVP(self.Vin, self.Vout, self.ratedPower, self.Omega, opt)

        else:
            mtype = RotorAero.FSFP(self.Vin, self.Vout, self.ratedPower, self.Omega, self.pitch)

        # initialize rotor
        rotor = RotorAero(aeroanalysis, drivetrain, mtype)

        # power curve
        self.V = np.linspace(self.Vin, self.Vout, 200)
        self.P = rotor.powerCurve(self.V)

        # AEP
        self.AEP = rotor.AEP(CDF)

        # conditions at rated
        rc = self.rated_conditions
        rc.V, rc.Omega, pitch_rated, rc.T, rc.Q = rotor.conditionsAtRated()
        V_rated = rc.V

        # distributed loads
        azimuth_rated = 180.0  # nearest to tower
        lr = self.loads_rated
        lr.r, lr.Px, lr.Py, lr.Pz, lr.pitch = rotor.distributedAeroLoads(V_rated, azimuth_rated)
        lr.azimuth = azimuth_rated

        le = self.loads_extreme
        le.r, le.Px, le.Py, le.Pz, le.pitch = rotor.distributedAeroLoads(V_extreme, self.azimuth_extreme, self.pitch_extreme)
        le.azimuth = self.azimuth_extreme

        # hub loads
        hr = self.hub_rated
        Fx, Fy, Fz, Mx, My, Mz = rotor.hubForcesAndMoments(V_rated)
        hr.F = [Fx, Fy, Fz]
        hr.M = [Mx, My, Mz]

        # average rotation speed
        self.avgOmega = rotor.averageRotorSpeed(PDF)



if __name__ == '__main__':

    rotor = RotorAeroComp()

    rotor.Rhub = 1.5
    rotor.Rtip = 63.0
    rotor.r = np.array([2.87, 5.60, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25,
                        36.35, 40.45, 44.55, 48.65, 52.75, 56.17, 58.9, 61.63])
    rotor.chord = np.array([3.48, 3.88, 4.22, 4.50, 4.57, 4.44, 4.28, 4.08, 3.85,
                            3.58, 3.28, 2.97, 2.66, 2.35, 2.07, 1.84, 1.59])
    rotor.theta = np.array([13.28, 13.28, 13.28, 13.28, 11.77, 10.34, 8.97, 7.67,
                            6.43, 5.28, 4.20, 3.21, 2.30, 1.50, 0.91, 0.48, 0.09])
    rotor.B = 3

    # atmosphere
    rotor.atm.rho = 1.225
    rotor.atm.mu = 1.81206e-5
    rotor.atm.shearExp = 0.2

    rotor.hubHt = 90.0

    rotor.nSector = 4

    rotor.precone = 2.5
    rotor.tilt = -5
    rotor.yaw = 0.0


    rotor.af_path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = 'Cylinder1.dat'
    airfoil_types[1] = 'Cylinder2.dat'
    airfoil_types[2] = 'DU40_A17.dat'
    airfoil_types[3] = 'DU35_A17.dat'
    airfoil_types[4] = 'DU30_A17.dat'
    airfoil_types[5] = 'DU25_A17.dat'
    airfoil_types[6] = 'DU21_A17.dat'
    airfoil_types[7] = 'NACA64_A17.dat'

    # place at appropriate radial stations, and convert format
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    rotor.af_file = ['0']*len(rotor.r)
    for i in range(len(rotor.r)):
        rotor.af_file[i] = airfoil_types[af_idx[i]]


    rotor.aeroanalysis_type = 'ccblade'
    rotor.drivetrain_type = 'geared'
    rotor.wind_turbine_class = 'I'
    rotor.machine_type = 'VSVP'


    rotor.Vin = 3.0
    rotor.Vout = 25.0
    rotor.ratedPower = 5e6

    rotor.minOmega = 0.0
    rotor.maxOmega = 10.0
    rotor.tsr_r2 = 7.55


    rotor.pitch_extreme_target = 0.0
    rotor.azimuth_extreme = 90.0

    rotor.PDF = RotorAero.RayleighPDF(10.0)
    rotor.CDF = RotorAero.RayleighCDF(10.0)
    rotor.shearExp = 0.2
    rotor.V_extreme = 70.0


    rotor.execute()

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P)

    print rotor.AEP

    rated = rotor.loads_rated
    plt.figure()
    plt.plot(rated.r, rated.Px)
    plt.plot(rated.r, rated.Py)
    plt.plot(rated.r, rated.Pz)

    extreme = rotor.loads_extreme
    plt.figure()
    plt.plot(extreme.r, extreme.Px)
    plt.plot(extreme.r, extreme.Py)
    plt.plot(extreme.r, extreme.Pz)
    plt.show()

    # from myutilities import printArray
    # printArray(rotor.r_loads, 'r_loads')
    # printArray(rotor.Px_extreme, 'Px_extreme')
    # printArray(rotor.Py_extreme, 'Py_extreme')
    # printArray(rotor.Pz_extreme, 'Pz_extreme')
    # print rotor.pitch_extreme
    # print rotor.azimuth_extreme

    # print rotor.avgOmega





