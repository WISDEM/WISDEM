#!/usr/bin/env python
# encoding: utf-8
"""
wtperf_wrapper.py

Created by Andrew Ning on 2012-05-11.
Copyright (c) NREL. All rights reserved.
"""

import os
from subprocess import Popen
import numpy as np
from math import pi

from airfoilprep import Airfoil
from rotoraero import RotorAeroAnalysisBase
from wisdem.common.utilities import exe_path, mktmpdir


RPM2RS = pi/30.0
SCRATCH_DIR = 'wtp_scratch' + os.path.sep
basefilename = SCRATCH_DIR + 'wtperf'





class WTPerf(RotorAeroAnalysisBase):
    """This class represents an anerodynamic model of a wind turbine rotor.
    It uses the WTPerf code developed at the NWTC.
    SI base units only.

    """


    def __init__(self, r, chord, theta, af, Rhub, Rtip, B=3, rho=1.225, mu=1.81206e-5,
                 precone=0.0, tilt=0.0, yaw=0.0, shearExp=0.2, hubHt=80.0, nSector=8,
                 tiploss=True, hubloss=True, wakerotation=True, usecd=True, pathToWTPerf=None, DEBUG=False):
        """Constructor for aerodynamic rotor analysis

        Parameters
        ----------
        r : array_like (m)
            radial locations where blade is defined (should be increasing)
        chord : array_like (m)
            corresponding chord length at each section
        theta : array_like (deg)
            corresponding twist angle at each section.  positive twist decreases angle of attack.
        af : list(airfoilprep.Airfoil)
            list of airfoilprep.Airfoil objects at each section
        Rhub : float (m)
            radial location of hub
        Rtip : float (m)
            radial location of tip
        B : int, optional
            number of blades.  3 assumed is not input.
        precone = float, optional (deg)
            :ref:`precone angle <azimuth_blade_coord>`, positive tilts blades away from tower
        tilt = float, optional (deg)
            :ref:`tilt angle <yaw_rotor_coord>` about axis parallel to ground. positive tilts rotor up
        yaw = float, optional (deg)
            :ref:`yaw angle <wind_yaw_coord>` about axis through tower.  positive CCW rotation when viewed from above
        rho : float, optional (kg/m^3)
            freestream fluid density.  standard atmosphere value at sea level used for default.
        mu : float, optional (kg/m/s)
            dynamic viscosity of fluid.  standard atmosphere value at sea level used for default.
        shearExp : float, optional
            shear exponent for a power low profile
        hubHt : float (m)
            used in conjunction with shearExp to estimate vertical wind gradient across rotor plane
        tiploss: boolean, optional
            if True include Prandtl tip loss model
        hubloss: boolean, optional
            if True include Prandtl hub loss model
        wakerotation: boolean, optional
            if True include effect of wake rotation (i.e., tangential induction factor is nonzero)
        usecd : boolean, optional
            if True use drag coefficient in computing induction factors
        pathToWTPerf : str, optional
            location of WTPerf binary

        """

        # geometry
        self.r = np.array(r)
        self.chord = np.array(chord)
        self.theta = np.array(theta)
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.af = af
        self.pathToWTPerf = exe_path(pathToWTPerf, 'wtperf', os.path.dirname(__file__))

        # atmosphere
        self.mu = mu
        self.shearExp = shearExp
        self.hubHt = hubHt

        # interface variables
        self.rotorR = Rtip
        self.rho = rho
        self.yaw = yaw
        self.tilt = -tilt  # opposite sign convention
        self.precone = precone
        self.nBlade = B
        nAzimuth = 1  # not used.  WTPerf handles discretization internally
        super(WTPerf, self).__init__(nAzimuth)

        # azimuthal discretization
        if self.tilt == 0.0 and self.yaw == 0.0 and self.shearExp == 0.0:
            self.nSector = 1  # no more are necessary
        else:
            self.nSector = max(4, nSector)  # at least 4 are used by WTPerf


        # options.  most should not be changed.
        self.misc = dict(Echo=False, DimenInp=True, Metric=True, MaxIter=5,
                         ATol=1.0e-5, SWTol=1.0e-5, TipLoss=True, HubLoss=True, Swirl=True,
                         SkewWake=True, IndType=True, AIDrag=True, TIDrag=True,
                         TISingularity=True, UnfPower=False, TabDel=True, OutNines=True,
                         Beep=False, WriteBED=True, InputTSR=False, OutMaxCp=True, SpdUnits='mps')
        self.misc['NumSect'] = self.nSector
        self.misc['TipLoss'] = tiploss
        self.misc['HubLoss'] = hubloss
        self.misc['Swirl'] = wakerotation
        self.misc['AIDrag'] = usecd
        self.misc['TIDrag'] = usecd

        # save some of the variables we need for post-processing
        self.nSeg = len(self.r)


        # create working directory
        mktmpdir(SCRATCH_DIR, DEBUG)


        # ------- setup airfoil data -------------

        self.af_idx = [0]*self.nSeg
        self.af_files = [0]*self.nSeg
        self.useCm = False
        self.useCpmin = False

        # write to new files (avoid issues with different file formats and window/unix paths)
        for i, a in enumerate(af):

            fname = SCRATCH_DIR + str(i) + '.af'
            a.writeToAerodynFile(fname)
            self.af_idx[i] = i+1
            self.af_files[i] = fname
        # ------------------------------------------




    @classmethod
    def initWithWTPerfFile(cls, filename, pathToWTPerf='./wtperf'):
        """Constructor using WTPerf input file.

        Parameters
        ----------
        filename : str
            WTPerf input file
        pathToWTPerf : str, optional
            location of WTPerf binary

        Returns
        -------
        wtperf : WTPerf
            an instantiation of a WTPerf object

        Notes
        -----
        Only uses the turbine data and aerodynamic data sections from the input file

        """

        # from airfoil import PolarByRe

        f = open(filename)

        # skip lines and check if nondimensional
        for i in range(5):
            f.readline()
        nondimensional = f.readline().split()[0] == 'False'
        for i in range(16):
            f.readline()

        # read turbine data
        nBlade = int(f.readline().split()[0])
        Rtip = float(f.readline().split()[0])
        Rhub = float(f.readline().split()[0])
        preCone = float(f.readline().split()[0])
        tilt = float(f.readline().split()[0])
        yaw = float(f.readline().split()[0])
        hubHt = float(f.readline().split()[0])
        nseg = int(f.readline().split()[0])

        f.readline()

        r = np.zeros(nseg)
        theta = np.zeros(nseg)
        chord = np.zeros(nseg)
        affileidx = [0]*nseg

        for i in range(nseg):
            data = f.readline().split()
            r[i] = data[0]
            theta[i] = data[1]
            chord[i] = data[2]
            affileidx[i] = int(data[3]) - 1

        # un-normalize if necessary
        if nondimensional:
            Rhub *= Rtip
            hubHt *= Rtip
            r *= Rtip
            chord *= Rtip

        f.readline()

        # read aerodynamic data
        rho = float(f.readline().split()[0])
        nu = float(f.readline().split()[0])
        shearExp = float(f.readline().split()[0])
        useCm = bool(f.readline().split()[0])
        useCpmin = bool(f.readline().split()[0])
        numaf = int(f.readline().split()[0])

        aflist = [0]*numaf

        for i in range(numaf):
            aflist[i] = f.readline().split()[0].strip().strip('\"')

        f.close()

        # create airfoil objects
        airfoilArray = [0]*nseg
        for i in range(nseg):
            airfoilArray[i] = Airfoil.initFromAerodynFile(aflist[affileidx[i]])

        return cls(r, chord, theta, af, Rhub, Rtip, nBlade, preCone, tilt, yaw,
                   rho, nu*rho, shearExp, hubHt, pathToWTPerf)




    def distributedAeroLoads(self, Uinf, Omega, pitch, azimuth):
        """see :meth:`RotorAeroAnalysisInterface:distributedAeroLoads`"""

        # azimuth ignored

        if Omega == 0: Omega = 0.01  # some small value to prevent WT_Perf error

        Np = np.zeros(self.nSeg)
        Tp = np.zeros(self.nSeg)
        cosPrecone = np.cos(np.radians(self.precone))

        # run WTPerf
        self.misc['InputTSR'] = False
        self.misc['WriteBED'] = True
        self.runWTPerf([Uinf], [Omega], [pitch])

        # read in data from file
        outfile = basefilename + '.bed'
        try:
            f = open(outfile)
            lines = f.readlines()
            f.close()
        except IOError:
            print 'WTPerf failed to run correctly.  See output:'
            print open(SCRATCH_DIR+'output').read()
            exit()

        # check to see if this is axisymmetric case or not
        numSect = 1
        if len(lines[13].strip()) == 0:
            numSect = self.nSector

        # for i in range(0, self.nCases):
        i = 0
        for j in range(0, self.nSeg):

            # add up across azimuth
            norm = 0.0
            tang = 0.0
            for k in range(0, numSect):

                # horribly obtuse line parsing
                data = lines[13 + i*(5+self.nSeg) + j
                    + (numSect > 1)*(1 + i*self.nSeg*numSect + j*numSect + k)].split()

                #convert thrust and torque to normal and tangential forces
                #TODO: add more significant figures to output
                norm += float(data[14])/cosPrecone
                tang += float(data[15])/cosPrecone/self.r[j]

            Np[j] = norm
            Tp[j] = tang


        # extend to root and hub
        r = np.concatenate(([self.Rhub], self.r, [self.Rtip]))
        theta = np.concatenate(([self.theta[0]], self.theta, [self.theta[-1]]))
        Np = np.concatenate(([0.0], Np, [0.0]))
        Tp = np.concatenate(([0.0], Tp, [0.0]))

        # conform to coordinate system
        Px = Np
        Py = -Tp
        Pz = 0*Np

        return r, Px, Py, Pz, theta  #FIXME



    def evaluate(self, Uinf, Omega, pitch, coefficient=False):
        """see :meth:`RotorAeroAnalysisInterface:evaluate`

        Notes
        -----
        Overrides method in base class becuase it is faster
        to call WT_Perf in batch

        """

        self.misc['InputTSR'] = False
        self.misc['WriteBED'] = False
        self.runWTPerf(Uinf, Omega, pitch)

        Q, T = self.__readOutput(5, 6, len(Uinf))
        Q = np.array(Q)
        T = np.array(T)
        P = Q * Omega * RPM2RS

        if coefficient:
            Uinf = np.array(Uinf)
            R = self.Rtip
            q = 0.5*self.rho*Uinf**2
            A = pi*R**2
            P /= (q * Uinf * A)
            T /= (q * A)
            Q /= (q * A * R)

        return P, T, Q






# ---------------------
# Running WTPerf
# ---------------------

    @staticmethod
    def __file3col(f, data, name, description):
        print >> f, '{0!s:15}\t{1:15}\t{2:40}'.format(data, name, description)


    def runWTPerf(self, Uinf, Omega, pitch):
        """Runs WTPerf at the specified conditions

        Parameters
        ----------
        Uinf : array_like (m/s)
            freestream velocity
        Omega : array_like (RPM)
            rotation speed
        pitch : array_like (deg)
            blade pitch angle

        """

        # rename variables for convenience
        misc = self.misc

        # create input file
        inputfilename = basefilename + '.wtp'
        f = open(inputfilename, 'w')

        print >> f, '-----  WT_Perf Input File  -----------------------------------------------------'
        print >> f, 'Auto-generated WT_Perf input file.'
        print >> f, 'Compatible with WT_Perf v3.04.00a-mlb.'

        print >> f, '-----  Input Configuration  ----------------------------------------------------'
        WTPerf.__file3col(f, misc['Echo'], 'Echo:', 'Echo input parameters to "echo.out"?')
        WTPerf.__file3col(f, misc['DimenInp'], 'DimenInp:', 'Turbine parameters are dimensional?')
        WTPerf.__file3col(f, misc['Metric'], 'Metric:', 'Turbine parameters are Metric (MKS vs FPS)?')

        print >> f, '-----  Model Configuration  ----------------------------------------------------'
        WTPerf.__file3col(f, misc['NumSect'], 'NumSect:', 'Number of circumferential sectors.')
        WTPerf.__file3col(f, misc['MaxIter'], 'MaxIter:', 'Max number of iterations for induction factor.')
        WTPerf.__file3col(f, misc['ATol'], 'ATol:', 'Error tolerance for induction iteration.')
        WTPerf.__file3col(f, misc['SWTol'], 'SWTol:', 'Error tolerance for skewed-wake iteration.')

        print >> f, '-----  Algorithm Configuration  ------------------------------------------------'
        WTPerf.__file3col(f, misc['TipLoss'], 'TipLoss:', 'Use the Prandtl tip-loss model?')
        WTPerf.__file3col(f, misc['HubLoss'], 'HubLoss:', 'Use the Prandtl hub-loss model?')
        WTPerf.__file3col(f, misc['Swirl'], 'Swirl:', 'Include Swirl effects?')
        WTPerf.__file3col(f, misc['SkewWake'], 'SkewWake:', 'Apply skewed-wake correction?')
        WTPerf.__file3col(f, misc['IndType'], 'IndType:', 'Use BEM induction algorithm?')
        WTPerf.__file3col(f, misc['AIDrag'], 'AIDrag:', 'Use the drag term in the axial induction calculation.')
        WTPerf.__file3col(f, misc['TIDrag'], 'TIDrag:', 'Use the drag term in the tangential induction calculation.')
        WTPerf.__file3col(f, misc['TISingularity'], 'TISingularity:', 'Use the singularity avoidance method in the tangential-induction calculation?')

        print >> f, '-----  Turbine Data  -----------------------------------------------------------'
        WTPerf.__file3col(f, self.nBlade, 'NumBlade:', 'Number of blades.')
        WTPerf.__file3col(f, self.Rtip, 'RotorRad:', 'Rotor radius [length].')
        WTPerf.__file3col(f, self.Rhub, 'HubRad:', 'Hub radius [length or div by radius].')
        WTPerf.__file3col(f, self.precone, 'PreCone:', 'Precone angle, positive downwind [deg].')
        WTPerf.__file3col(f, self.tilt, 'Tilt:', 'Shaft tilt [deg].')
        WTPerf.__file3col(f, self.yaw, 'Yaw:', 'Yaw error [deg].')
        WTPerf.__file3col(f, self.hubHt, 'HubHt:', 'Hub height [length or div by radius].')
        WTPerf.__file3col(f, self.nSeg, 'NumSeg:', 'Number of blade segments (entire rotor radius).')
        print >> f, '   RElm    Twist   Chord  AFfile  PrntElem'
        for r, t, c, a in zip(self.r, self.theta, self.chord, self.af_idx):
            print >> f, '{0!s:9} {1!s:9} {2!s:9} {3!s:5} {4:9}'.format(r, t, c, a, 'True')

        print >> f, '-----  Aerodynamic Data  -------------------------------------------------------'
        WTPerf.__file3col(f, self.rho, 'Rho:', 'Air density [mass/volume]')
        WTPerf.__file3col(f, self.mu/self.rho, 'KinVisc:', 'Kinematic air viscosity')
        WTPerf.__file3col(f, self.shearExp, 'ShearExp:', 'Wind shear exponent (1/7 law = 0.143)')
        WTPerf.__file3col(f, self.useCm, 'UseCm:', 'Are Cm data included in the airfoil tables?')
        WTPerf.__file3col(f, self.useCpmin, 'UseCpmin:', 'Are Cp,min data included in the airfoil tables?')
        WTPerf.__file3col(f, len(self.af_files), 'NumAF:', 'Number of airfoil files.')
        WTPerf.__file3col(f, '"' + str(self.af_files[0]) + '"', 'AF_File:', 'List of NumAF airfoil files.')
        for i in range(1, len(self.af_files)):
            print >> f, '"' + self.af_files[i] + '"'

        print >> f, '-----  I/O Settings  -----------------------------------------------------------'
        WTPerf.__file3col(f, misc['UnfPower'], 'UnfPower:', 'Write parametric power to an unformatted file?')
        WTPerf.__file3col(f, misc['TabDel'], 'TabDel:', 'Make output tab-delimited (fixed-width otherwise).')
        WTPerf.__file3col(f, misc['OutNines'], 'OutNines:', 'Output nines if the solution doesn''t fully converge to the specified tolerences.')
        WTPerf.__file3col(f, misc['Beep'], 'Beep:', 'Beep if errors occur.')
        WTPerf.__file3col(f, 'False', 'KFact:', 'Output dimensional parameters in K (e.g., kN instead on N)')
        WTPerf.__file3col(f, misc['WriteBED'], 'WriteBED:', 'Write out blade element data to "<rootname>.bed"?')
        WTPerf.__file3col(f, misc['InputTSR'], 'InputTSR:', 'Input speeds as TSRs?')
        WTPerf.__file3col(f, misc['OutMaxCp'], 'OutMaxCp:', 'Output conditions for the maximum Cp?')
        WTPerf.__file3col(f, '"' + misc['SpdUnits'] + '"', 'SpdUnits:', 'Wind-speed units (mps, fps, mph).')

        print >> f, '-----  Combined-Case Analysis  -------------------------------------------------'
        WTPerf.__file3col(f, len(Uinf), 'NumCases:', 'Number of cases to run.  Enter zero for parametric analysis.')
        WTPerf.__file3col(f, 'WS or TSR   RotSpd', 'Pitch', 'Remove following block of lines if NumCases is zero.')
        for u, o, p in zip(Uinf, Omega, pitch):
            print >> f, '{0!s:9} {1!s:9} {2!s:9}'.format(u, o, p)

        f.close()

        # remove output files (just to be sure run was successful)
        try:
            os.remove(basefilename + '.oup')
        except OSError:
            pass
        try:
            os.remove(basefilename + '.bed')
        except OSError:
            pass


        # run wtperf
        f = open(SCRATCH_DIR + 'output', 'w')
        process = Popen([self.pathToWTPerf, inputfilename], shell=False, stdout=f)
        process.communicate()  # clear buffer and wait for process to terminate
        f.close()


    def __readOutput(self, idx1, idx2, n):
        """
        private method
        reads two columns from WTPerf standard *.oup file
        reads n lines from columns idx1 and idx2
        """

        v1 = np.zeros(n)
        v2 = np.zeros(n)

        # read main output data file
        outfile = basefilename + '.oup'
        try:
            f = open(outfile)
        except IOError:
            print 'WTPerf failed to run correctly.  See output:'
            print open(SCRATCH_DIR+'output').read()
            exit()

        # skip header
        for i in range(8):
            f.readline()

        for idx, line in enumerate(f):
            array = line.split()

            if '***' in array[idx1]:
                entry1 = lastGoodEntry1  # TODO: this will fail if first line contains ****
            else:
                entry1 = float(array[idx1])
                lastGoodEntry1 = entry1

            if '***' in array[idx2]:
                entry2 = lastGoodEntry2
            else:
                entry2 = float(array[idx2])
                lastGoodEntry2 = entry2

            v1[idx] = entry1
            v2[idx] = entry2

            # data = [float(s) for s in line.split()] # FIXME: sometimes WTPerf returns a ***********
            # v1[idx] = data[idx1]
            # v2[idx] = data[idx2]



        f.close()
        return (v1, v2)






if __name__ == '__main__':


    # geometry
    Rhub = 1.5
    Rtip = 63.0
    r = np.array([2.87, 5.60, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25,
                  36.35, 40.45, 44.55, 48.65, 52.75, 56.17, 58.9, 61.63])
    chord = np.array([3.48, 3.88, 4.22, 4.50, 4.57, 4.44, 4.28, 4.08, 3.85,
                      3.58, 3.28, 2.97, 2.66, 2.35, 2.07, 1.84, 1.59])
    theta = np.array([13.28, 13.28, 13.28, 13.28, 11.77, 10.34, 8.97, 7.67,
                      6.43, 5.28, 4.20, 3.21, 2.30, 1.50, 0.91, 0.48, 0.09])
    B = 3

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5



    basepath = os.path.join('5MW_files', '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'Cylinder1.dat'))
    airfoil_types[1] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'Cylinder2.dat'))
    airfoil_types[2] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'DU40_A17.dat'))
    airfoil_types[3] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'DU35_A17.dat'))
    airfoil_types[4] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'DU30_A17.dat'))
    airfoil_types[5] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'DU25_A17.dat'))
    airfoil_types[6] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'DU21_A17.dat'))
    airfoil_types[7] = Airfoil.initFromAerodynFile(os.path.join(basepath, 'NACA64_A17.dat'))

    # place at appropriate radial stations, and convert format
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    af = [0]*len(r)
    for i in range(len(r)):
        af[i] = airfoil_types[af_idx[i]]


    rotor = WTPerf(r, chord, theta, af, Rhub, Rtip, B,
                   precone=0.0, tilt=0.0, yaw=0.0, rho=rho, mu=mu,
                   shearExp=1.0/7, hubHt=80.0)



    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 0.0

    # evaluate distributed loads
    r, theta, Px, Py, Pz = rotor.distributedAeroLoads(Uinf, Omega, pitch, azimuth)



    # plot
    import matplotlib.pyplot as plt
    rstar = (r - r[0]) / (r[-1] - r[0])
    plt.plot(rstar, -Py/1e3, 'k', label='edgewise')
    plt.plot(rstar, Px/1e3, 'r', label='flapwise')
    plt.xlabel('blade fraction')
    plt.ylabel('distributed aerodynamic loads (kN)')
    plt.legend(loc='upper left')
    plt.show()


    P, T, Q = rotor.evaluate([Uinf], [Omega], [pitch])

    CP, CT, CQ = rotor.evaluate([Uinf], [Omega], [pitch], coefficient=True)

    print CP, CT, CQ

    Fx, Fy, Fz, Mx, My, Mz = rotor.hubLoads(Uinf, Omega, pitch)
    print Fx, Fy, Fz, Mx, My, Mz

    tsr = np.linspace(2, 14, 50)
    Omega = 10.0 * np.ones_like(tsr)
    Uinf = Omega*pi/30.0 * Rtip/tsr
    pitch = np.zeros_like(tsr)

    CP, CT, CQ = rotor.evaluate(Uinf, Omega, pitch, coefficient=True)



    import matplotlib.pyplot as plt
    plt.plot(tsr, CP, 'k')
    plt.xlabel('$\lambda$')
    plt.ylabel('$c_p$')
    # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/CCBlade/docs-dev/images/cp.pdf')
    plt.show()


