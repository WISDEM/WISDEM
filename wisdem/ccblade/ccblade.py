#!/usr/bin/env python
# encoding: utf-8
"""
ccblade.py
Created by S. Andrew Ning on 5/11/2012
Copyright (c) NREL. All rights reserved.
A blade element momentum method using theory detailed in [1]_.  Has the
advantages of guaranteed convergence and at a superlinear rate, and
continuously differentiable output.
.. [1] S. Andrew Ning, "A simple solution method for the blade element momentum
equations with guaranteed convergence", Wind Energy, 2013.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import warnings
import multiprocessing as mp

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline, bisplev

import wisdem.ccblade._bem as _bem

# ------------------
#  Airfoil Class
# ------------------


class CCAirfoil(object):
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""

    def __init__(self, alpha, Re, cl, cd, cm=[], x=[], y=[], AFName="DEFAULTAF"):
        """Setup CCAirfoil from raw airfoil data on a grid.
        Parameters
        ----------
        alpha : array_like (deg)
            angles of attack where airfoil data are defined
            (should be defined from -180 to +180 degrees)
        Re : array_like
            Reynolds numbers where airfoil data are defined
            (can be empty or of length one if not Reynolds number dependent)
        cl : array_like
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : array_like
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]
        """

        alpha = np.deg2rad(alpha)
        self.x = x
        self.y = y
        self.AFName = AFName
        self.one_Re = False

        if len(cm) > 0:
            self.use_cm = True
        else:
            self.use_cm = False

        # special case if zero or one Reynolds number (need at least two for bivariate spline)
        if len(Re) < 2:
            Re = [1e1, 1e15]
            cl = np.c_[cl, cl]
            cd = np.c_[cd, cd]
            if self.use_cm:
                cm = np.c_[cm, cm]

            self.one_Re = True

        if len(alpha) < 2:
            raise ValueError(f"Need at least 2 angles of attack, but found {alpha}")

        kx = min(len(alpha) - 1, 3)
        ky = min(len(Re) - 1, 3)

        # a small amount of smoothing is used to prevent spurious multiple solutions
        self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky, s=0.01)
        self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky, s=0.001)
        self.alpha = alpha

        if self.use_cm > 0:
            self.cm_spline = RectBivariateSpline(alpha, Re, cm, kx=kx, ky=ky, s=0.0001)

    def max_eff(self, Re):
        # Get the angle of attack, cl and cd at max airfoil efficiency. For a cylinder, set the angle of attack to 0

        Eff = np.zeros_like(self.alpha)

        # Check efficiency only between -20 and +40 deg
        aoa_start = -20.0
        aoa_end = 40

        alpha = np.deg2rad(np.linspace(aoa_start, aoa_end, num=201))
        cl = np.array([self.cl_spline.ev(aoa, Re) for aoa in alpha])
        cd = np.array([self.cd_spline.ev(aoa, Re) for aoa in alpha])

        if np.max(cl) < 1.e-3:  # Cylinder
            alpha_Emax = 0.
            cl_Emax = self.cl_spline.ev(alpha_Emax, Re)
            cd_Emax = self.cd_spline.ev(alpha_Emax, Re)
            Emax = cl_Emax / cd_Emax
        else:
            Eff = [cli / cdi for cli, cdi, in zip(cl, cd)]
            i_max = np.argmax(Eff)
            alpha_Emax = alpha[i_max]
            cl_Emax = cl[i_max]
            cd_Emax = cd[i_max]
            Emax = Eff[i_max]

        # print Emax, np.deg2rad(alpha_Emax), cl_Emax, cd_Emax

        return Emax, alpha_Emax, cl_Emax, cd_Emax

    def awayfromstall(self, Re, margin):
        # Get the angle of attack, cl and cd with a margin (in degrees) from the stall point. For a cylinder, set the angle of attack to 0 deg

        # Eff         = np.zeros_like(self.alpha)

        # Look for stall only between -20 and +40 deg
        aoa_start = -20.0
        aoa_end = 40
        i_start = np.argmin(abs(self.alpha - np.deg2rad(aoa_start)))
        i_end = np.argmin(abs(self.alpha - np.deg2rad(aoa_end)))

        if len(self.alpha[i_start:i_end]) == 0:  # Cylinder
            alpha_op = 0.0

        else:
            alpha = np.deg2rad(np.linspace(aoa_start, aoa_end, num=201))
            cl = [self.cl_spline.ev(aoa, Re) for aoa in alpha]
            cd = [self.cd_spline.ev(aoa, Re) for aoa in alpha]

            i_stall = np.argmax(cl)
            alpha_stall = alpha[i_stall]
            alpha_op = alpha_stall - np.deg2rad(margin)

        cl_op = self.cl_spline.ev(alpha_op, Re)
        cd_op = self.cd_spline.ev(alpha_op, Re)
        Eff_op = cl_op / cd_op

        # print Emax, np.deg2rad(alpha_Emax), cl_Emax, cd_Emax

        return Eff_op, alpha_op, cl_op, cd_op

    def evaluate(self, alpha, Re, return_cm=False):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number.
        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        This method uses a spline so that the output is continuously differentiable, and
        also uses a small amount of smoothing to help remove spurious multiple solutions.
        """

        cl = self.cl_spline.ev(alpha, Re)
        cd = self.cd_spline.ev(alpha, Re)

        if self.use_cm and return_cm:
            cm = self.cm_spline.ev(alpha, Re)
            return cl, cd, cm
        else:
            return cl, cd

    def derivatives(self, alpha, Re):
        # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
        tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
        tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees

        dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
        dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

        if self.one_Re:
            dcl_dRe = 0.0
            dcd_dRe = 0.0
        else:
            try:
                dcl_dRe = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
                dcd_dRe = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
            except:
                dcl_dRe = 0.0
                dcd_dRe = 0.0
        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    def eval_unsteady(self, alpha, cl, cd, cm):
        # calculate unsteady coefficients from polars for OpenFAST's Aerodyn

        unsteady = {}

        alpha_rad = np.deg2rad(alpha)
        cn = cl * np.cos(alpha_rad) + cd * np.sin(alpha_rad)

        # alpha0, Cd0, Cm0
        aoa_l = [-30.0]
        aoa_h = [30.0]
        idx_low = np.argmin(abs(alpha - aoa_l))
        idx_high = np.argmin(abs(alpha - aoa_h))

        if max(np.abs(np.gradient(cl))) > 0.0:
            unsteady["alpha0"] = np.interp(0.0, cl[idx_low:idx_high], alpha[idx_low:idx_high])
            unsteady["Cd0"] = np.interp(0.0, cl[idx_low:idx_high], cd[idx_low:idx_high])
            unsteady["Cm0"] = np.interp(0.0, cl[idx_low:idx_high], cm[idx_low:idx_high])
        else:
            unsteady["alpha0"] = 0.0
            unsteady["Cd0"] = cd[np.argmin(abs(alpha - 0.0))]
            unsteady["Cm0"] = 0.0

        unsteady["eta_e"] = 1
        unsteady["T_f0"] = "Default"
        unsteady["T_V0"] = "Default"
        unsteady["T_p"] = "Default"
        unsteady["T_VL"] = "Default"
        unsteady["b1"] = "Default"
        unsteady["b2"] = "Default"
        unsteady["b5"] = "Default"
        unsteady["A1"] = "Default"
        unsteady["A2"] = "Default"
        unsteady["A5"] = "Default"
        unsteady["S1"] = 0
        unsteady["S2"] = 0
        unsteady["S3"] = 0
        unsteady["S4"] = 0

        def find_breakpoint(x, y, idx_low, idx_high, multi=1.0):
            lin_fit = np.interp(x[idx_low:idx_high], [x[idx_low], x[idx_high]], [y[idx_low], y[idx_high]])
            idx_break = 0
            lin_diff = 0
            for i, (fit, yi) in enumerate(zip(lin_fit, y[idx_low:idx_high])):
                if multi == 0:
                    diff_i = np.abs(yi - fit)
                else:
                    diff_i = multi * (yi - fit)
                if diff_i > lin_diff:
                    lin_diff = diff_i
                    idx_break = i
            idx_break += idx_low
            return idx_break

        # Cn1
        idx_alpha0 = np.argmin(abs(alpha - unsteady["alpha0"]))

        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            aoa_h = alpha[idx_alpha0] + 35.0
            idx_high = np.argmin(abs(alpha - aoa_h))

            cm_temp = cm[idx_low:idx_high]
            idx_cm_min = [
                i
                for i, local_min in enumerate(
                    np.r_[True, cm_temp[1:] < cm_temp[:-1]] & np.r_[cm_temp[:-1] < cm_temp[1:], True]
                )
                if local_min
            ] + idx_low
            idx_high = idx_cm_min[-1]

            idx_Cn1 = find_breakpoint(alpha, cm, idx_alpha0, idx_high)
            unsteady["Cn1"] = cn[idx_Cn1]
        else:
            idx_Cn1 = np.argmin(abs(alpha - 0.0))
            unsteady["Cn1"] = 0.0

        # Cn2
        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            aoa_l = np.mean([alpha[idx_alpha0], alpha[idx_Cn1]]) - 30.0
            idx_low = np.argmin(abs(alpha - aoa_l))

            cm_temp = cm[idx_low:idx_high]
            idx_cm_min = [
                i
                for i, local_min in enumerate(
                    np.r_[True, cm_temp[1:] < cm_temp[:-1]] & np.r_[cm_temp[:-1] < cm_temp[1:], True]
                )
                if local_min
            ] + idx_low
            idx_high = idx_cm_min[-1]

            idx_Cn2 = find_breakpoint(alpha, cm, idx_low, idx_alpha0, multi=0.0)
            unsteady["Cn2"] = cn[idx_Cn2]
        else:
            idx_Cn2 = np.argmin(abs(alpha - 0.0))
            unsteady["Cn2"] = 0.0

        # C_nalpha
        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            if idx_alpha0 < idx_Cn1:
                unsteady["C_nalpha"] = max(np.gradient(cn[idx_alpha0:idx_Cn1], alpha_rad[idx_alpha0:idx_Cn1]))
            else:
                unsteady['C_nalpha'] = np.gradient(cn, alpha_rad)[idx_alpha0]

        else:
            unsteady["C_nalpha"] = 0.0

        # alpha1, alpha2
        # finding the break point in drag as a proxy for Trailing Edge separation, f=0.7
        # 3d stall corrections cause erroneous f calculations
        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            aoa_l = [0.0]
            idx_low = np.argmin(abs(alpha - aoa_l))
            idx_alpha1 = find_breakpoint(alpha, cd, idx_low, idx_Cn1, multi=-1.0)
            unsteady["alpha1"] = alpha[idx_alpha1]
        else:
            idx_alpha1 = np.argmin(abs(alpha - 0.0))
            unsteady["alpha1"] = 0.0
        unsteady["alpha2"] = -1.0 * unsteady["alpha1"]

        unsteady["St_sh"] = "Default"
        unsteady["k0"] = 0
        unsteady["k1"] = 0
        unsteady["k2"] = 0
        unsteady["k3"] = 0
        unsteady["k1_hat"] = 0
        unsteady["x_cp_bar"] = "Default"
        unsteady["UACutout"] = "Default"
        unsteady["filtCutOff"] = "Default"

        unsteady["Alpha"] = alpha
        unsteady["Cl"] = cl
        unsteady["Cd"] = cd
        unsteady["Cm"] = cm

        self.unsteady = unsteady

    def af_flap_coords(
        self, xfoil_path, delta_flap=12.0, xc_hinge=0.8, yt_hinge=0.5, numNodes=250, multi_run=False, MPI_run=False
    ):
        # This function is used to create and run xfoil to get airfoil coordinates for a given flap deflection
        # Set Needed parameter values
        AFName = self.AFName
        df = str(delta_flap)  # Flap deflection angle in deg
        numNodes = str(numNodes)  # number of panels to use (will be number of points in profile)
        dist_param = "0.5"  # TE/LE panel density ratio
        # Set filenames
        if multi_run or MPI_run:
            pid = mp.current_process().pid
            # Temporary file name for coordinates...will be deleted at the end
            CoordsFlnmAF = "profilecoords_p{}.dat".format(pid)
            saveFlnmAF = "{}_{}_Airfoil_p{}.txt".format(AFName, df, pid)
            saveFlnmPolar = "Polar_p{}.txt".format(pid)
            xfoilFlnm = "xfoil_input_p{}.txt".format(pid)
            NUL_fname = "NUL_{}".format(pid)
        else:
            CoordsFlnmAF = "profilecoords.dat"  # Temporary file name for coordinates...will be deleted at the end
            saveFlnmPolar = "Polar.txt"
            saveFlnmAF = "{}_{}_Airfoil.txt".format(AFName, df)
            xfoilFlnm = "xfoil_input.txt"  # Xfoil run script that will be deleted after it is no longer needed
            NUL_fname = "NUL"

        # Cleaning up old files to prevent replacement issues
        if os.path.exists(saveFlnmAF):
            os.remove(saveFlnmAF)
        if os.path.exists(CoordsFlnmAF):
            os.remove(CoordsFlnmAF)
        if os.path.exists(xfoilFlnm):
            os.remove(xfoilFlnm)
        if os.path.exists(NUL_fname):
            os.remove(NUL_fname)

        # Saving origional profile data temporatily to a txt file so xfoil can load it in
        dat = np.array([self.x, self.y])
        np.savetxt(CoordsFlnmAF, dat.T, fmt=["%f", "%f"])

        # %% Writes the Xfoil run script to read in coordinates, create flap, re-pannel, and save coordinates to a .txt file
        # Create the airfoil with flap
        fid = open(xfoilFlnm, "w")
        fid.write("PLOP \n G \n\n")  # turn off graphics
        fid.write("LOAD \n")
        fid.write(CoordsFlnmAF + "\n")  # name of file where coordinates are stored
        fid.write(AFName + "\n")  # name given to airfoil geometry (internal to xfoil)
        fid.write("GDES \n")  # enter geometric change options
        fid.write("FLAP \n")  # add in flap
        fid.write(str(xc_hinge) + "\n")  # specify x/c location of flap hinge
        fid.write("999\n")  # to specify y/t instead of actual distance
        fid.write(str(yt_hinge) + "\n")  # specify y/t value for flap hinge point
        fid.write(df + "\n")  # set flap deflection in deg
        fid.write("NAME \n")
        fid.write(AFName + "_" + df + "\n")  # name new airfoil
        fid.write("EXEC \n \n")  # move airfoil from buffer into current airfoil

        # Re-panel with specified number of panes and LE/TE panel density ratio (to possibly smooth out points)
        fid.write("PPAR\n")
        fid.write("N \n")
        fid.write(numNodes + "\n")
        fid.write("T \n")
        fid.write(dist_param + "\n")
        fid.write("\n\n")

        # Save airfoil coordinates with designation flap deflection
        fid.write("PSAV \n")
        fid.write(saveFlnmAF + " \n \n")  # the extra \n may not be needed

        # Quit xfoil and close xfoil script file
        fid.write("QUIT \n")
        fid.close()

        # Run the XFoil calling command
        os.system(xfoil_path + " < " + xfoilFlnm + " > " + NUL_fname)  # <<< runs XFoil !

        # Load in saved airfoil coordinates (with flap) from xfoil and save to instance variables
        flap_coords = np.loadtxt(saveFlnmAF)
        self.af_flap_xcoords = flap_coords[:, 0]
        self.af_flap_ycoords = flap_coords[:, 1]
        self.ctrl = delta_flap  # bem: the way that this function is called in rotor_geometry_yaml, this instance is not going to be used when calculating polars

        # Delete uneeded txt files script file
        if os.path.exists(CoordsFlnmAF):
            os.remove(CoordsFlnmAF)
        if os.path.exists(xfoilFlnm):
            os.remove(xfoilFlnm)
        if os.path.exists(saveFlnmAF):
            os.remove(saveFlnmAF)
        if os.path.exists(NUL_fname):
            os.remove(NUL_fname)


# ------------------
#  Main Class: CCBlade
# ------------------


class CCBlade(object):
    def __init__(
        self,
        r,
        chord,
        theta,
        af,
        Rhub,
        Rtip,
        B=3,
        rho=1.225,
        mu=1.81206e-5,
        precone=0.0,
        tilt=0.0,
        yaw=0.0,
        shearExp=0.2,
        hubHt=80.0,
        nSector=8,
        precurve=None,
        precurveTip=0.0,
        presweep=None,
        presweepTip=0.0,
        tiploss=True,
        hubloss=True,
        wakerotation=True,
        usecd=True,
        iterRe=1,
        derivatives=False,
    ):
        """Constructor for aerodynamic rotor analysis

        Parameters
        ----------
        r : array_like (m)
            locations defining the blade along z-axis of :ref:`blade coordinate system <azimuth_blade_coord>`
            (values should be increasing).
        chord : array_like (m)
            corresponding chord length at each section
        theta : array_like (deg)
            corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---
            positive twist decreases angle of attack.
        Rhub : float (m)
            location of hub
        Rtip : float (m)
            distance between rotor center and blade tip along z axis of blade root c.s.
        B : int, optional
            number of blades
        rho : float, optional (kg/m^3)
            freestream fluid density
        mu : float, optional (kg/m/s)
            dynamic viscosity of fluid
        precone : float, optional (deg)
            :ref:`hub precone angle <azimuth_blade_coord>`
        tilt : float, optional (deg)
            nacelle :ref:`tilt angle <yaw_hub_coord>`
        yaw : float, optional (deg)
            nacelle :ref:`yaw angle<wind_yaw_coord>`
        shearExp : float, optional
            shear exponent for a power-law wind profile across hub
        hubHt : float, optional
            hub height used for power-law wind profile.
            U = Uref*(z/hubHt)**shearExp
        nSector : int, optional
            number of azimuthal sectors to descretize aerodynamic calculation.  automatically set to
            ``1`` if tilt, yaw, and shearExp are all 0.0.  Otherwise set to a minimum of 4.
        precurve : array_like, optional (m)
            location of blade pitch axis in x-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
        precurveTip : float, optional (m)
            location of blade pitch axis in x-direction at the tip (analogous to Rtip)
        presweep : array_like, optional (m)
            location of blade pitch axis in y-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
        presweepTip : float, optional (m)
            location of blade pitch axis in y-direction at the tip (analogous to Rtip)
        tiploss : boolean, optional
            if True, include Prandtl tip loss model
        hubloss : boolean, optional
            if True, include Prandtl hub loss model
        wakerotation : boolean, optional
            if True, include effect of wake rotation (i.e., tangential induction factor is nonzero)
        usecd : boolean, optional
            If True, use drag coefficient in computing induction factors
            (always used in evaluating distributed loads from the induction factors).
            Note that the default implementation may fail at certain points if drag is not included
            (see Section 4.2 in :cite:`Ning2013A-simple-soluti`).  This can be worked around, but has
            not been implemented.
        iterRe : int, optional
            The number of iterations to use to converge Reynolds number.  Generally iterRe=1 is sufficient,
            but for high accuracy in Reynolds number effects, iterRe=2 iterations can be used.  More than that
            should not be necessary.  Gradients have only been implemented for the case iterRe=1.
        derivatives : boolean, optional
            if True, derivatives along with function values will be returned for the various methods
        """
        r = np.array(r)
        self.r = r.copy()
        self.chord = np.array(chord)
        self.theta = np.deg2rad(theta)
        self.af = af
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.B = B
        self.rho = rho
        self.mu = mu
        self.precone = float(np.deg2rad(precone))
        self.tilt = float(np.deg2rad(tilt))
        self.yaw = float(np.deg2rad(yaw))
        self.shearExp = shearExp
        self.hubHt = hubHt
        self.bemoptions = dict(usecd=usecd, tiploss=tiploss, hubloss=hubloss, wakerotation=wakerotation)
        self.iterRe = iterRe
        self.derivatives = derivatives

        # check if no precurve / presweep
        if precurve is None:
            precurve = np.zeros(len(r))
            precurveTip = 0.0

        if presweep is None:
            presweep = np.zeros(len(r))
            presweepTip = 0.0

        self.precurve = precurve.copy()
        self.precurveTip = precurveTip
        self.presweep = presweep.copy()
        self.presweepTip = presweepTip

        # Enfore unique points at tip and hub
        r_nd = (np.array(r) - r[0]) / (r[-1] - r[0])
        nd_hub = np.minimum(0.01, r_nd[:2].mean())
        nd_tip = np.maximum(0.99, r_nd[-2:].mean())
        if Rhub == r[0]:
            self.r[0] = np.interp(nd_hub, r_nd, r)
        if Rtip == r[-1]:
            self.r[-1] = np.interp(nd_tip, r_nd, r)
        if precurveTip == precurve[-1]:
            self.precurve[-1] = np.interp(nd_tip, r_nd, precurve)
        if presweepTip == presweep[-1]:
            self.presweep[-1] = np.interp(nd_tip, r_nd, presweep)

        # # rotor radius is defined as Rtip * cos(cone). This definition is the 
        # most common across commercial aeroelastic solvers. Rtip = D_hub/2 + (blade length along z)
        self.rotorR = Rtip * np.cos(self.precone)

        # azimuthal discretization
        if self.tilt == 0.0 and self.yaw == 0.0 and self.shearExp == 0.0:
            self.nSector = 1  # no more are necessary
        else:
            self.nSector = max(4, nSector)  # at least 4 are necessary

        self.inverse_analysis = False
        self.induction = False
        self.induction_inflow = False

    # residual
    def __runBEM(self, phi, r, chord, theta, af, Vx, Vy):
        """residual of BEM method and other corresponding variables"""

        a = 0.0
        ap = 0.0

        for i in range(self.iterRe):
            alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, self.pitch, chord, theta, self.rho, self.mu)
            cl, cd = af.evaluate(alpha, Re)

            fzero, a, ap = _bem.inductionfactors(
                r, chord, self.Rhub, self.Rtip, phi, cl, cd, self.B, Vx, Vy, **self.bemoptions
            )

        return fzero, a, ap, cl, cd

    def __errorFunction(self, phi, r, chord, theta, af, Vx, Vy):
        """strip other outputs leaving only residual for Brent's method
        Standard use case, geometry is known
        """

        fzero, a, ap, _, _ = self.__runBEM(phi, r, chord, theta, af, Vx, Vy)

        return fzero

    def __runBEM_inverse(self, phi, r, chord, cl, cd, af, Vx, Vy):
        """residual of BEM method and other corresponding variables"""

        a = 0.0
        ap = 0.0
        for i in range(self.iterRe):
            fzero, a, ap = _bem.inductionfactors(
                r, chord, self.Rhub, self.Rtip, phi, cl, cd, self.B, Vx, Vy, **self.bemoptions
            )

        return fzero, a, ap

    def __errorFunction_inverse(self, phi, r, chord, cl, cd, af, Vx, Vy):
        """strip other outputs leaving only residual for Brent's method
        Parametric optimization use case, desired Cl/Cd is known, solve for twist
        """

        fzero, a, ap = self.__runBEM_inverse(phi, r, chord, cl, cd, af, Vx, Vy)

        return fzero

    def __residualDerivatives(self, phi, r, chord, theta, af, Vx, Vy):
        """derivatives of fzero, a, ap"""

        if self.iterRe != 1:
            ValueError("Analytic derivatives not supplied for case with iterRe > 1")

        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)

        # alpha, Re (analytic derivaives)
        a = 0.0
        ap = 0.0
        alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, self.pitch, chord, theta, self.rho, self.mu)

        dalpha_dx = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        dRe_dx = np.array([0.0, Re / chord, 0.0, Re * Vx / W**2, Re * Vy / W**2, 0.0, 0.0, 0.0, 0.0])

        # cl, cd (spline derivatives)
        cl, cd = af.evaluate(alpha, Re)
        dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af.derivatives(alpha, Re)

        # chain rule
        dcl_dx = dcl_dalpha * dalpha_dx + dcl_dRe * dRe_dx
        dcd_dx = dcd_dalpha * dalpha_dx + dcd_dRe * dRe_dx

        # residual, a, ap (Tapenade)
        dx_dx = np.eye(9)

        fzero, dR_dx, a, da_dx, ap, dap_dx = _bem.inductionfactors_dv(
            r,
            dx_dx[5, :],
            chord,
            dx_dx[1, :],
            self.Rhub,
            dx_dx[6, :],
            self.Rtip,
            dx_dx[7, :],
            phi,
            dx_dx[0, :],
            cl,
            dcl_dx,
            cd,
            dcd_dx,
            self.B,
            Vx,
            dx_dx[3, :],
            Vy,
            dx_dx[4, :],
            **self.bemoptions,
        )

        return dR_dx, da_dx, dap_dx

    def __loads(self, phi, rotating, r, chord, theta, af, Vx, Vy):
        """normal and tangential loads at one section (and optionally derivatives)"""
        if Vx != 0.0 and Vy != 0.0:
            cphi = np.cos(phi)
            sphi = np.sin(phi)

            if rotating:
                _, a, ap, cl, cd = self.__runBEM(phi, r, chord, theta, af, Vx, Vy)
            else:
                a = 0.0
                ap = 0.0

            alpha_rad, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, self.pitch, chord, theta, self.rho, self.mu)
            if not rotating:
                cl, cd = af.evaluate(alpha_rad, Re)

            cn = cl * cphi + cd * sphi  # these expressions should always contain drag
            ct = cl * sphi - cd * cphi

            q = 0.5 * self.rho * W**2
            Np = cn * q * chord
            Tp = ct * q * chord

            alpha_deg = np.rad2deg(alpha_rad)

            if self.derivatives:
                # derivative of residual function
                if rotating:
                    dR_dx, da_dx, dap_dx = self.__residualDerivatives(phi, r, chord, theta, af, Vx, Vy)
                    dphi_dx = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    dR_dx = np.zeros(9)
                    dR_dx[0] = 1.0  # just to prevent divide by zero
                    da_dx = np.zeros(9)
                    dap_dx = np.zeros(9)
                    dphi_dx = np.zeros(9)

                # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
                dx_dx = np.eye(9)
                dchord_dx = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                # alpha, W, Re (Tapenade)

                alpha_rad, dalpha_dx, W, dW_dx, Re, dRe_dx = _bem.relativewind_dv(
                    phi,
                    dx_dx[0, :],
                    a,
                    da_dx,
                    ap,
                    dap_dx,
                    Vx,
                    dx_dx[3, :],
                    Vy,
                    dx_dx[4, :],
                    self.pitch,
                    dx_dx[8, :],
                    chord,
                    dx_dx[1, :],
                    theta,
                    dx_dx[2, :],
                    self.rho,
                    self.mu,
                )

                # cl, cd (spline derivatives)
                dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af.derivatives(alpha_rad, Re)

                # chain rule
                dcl_dx = dcl_dalpha * dalpha_dx + dcl_dRe * dRe_dx
                dcd_dx = dcd_dalpha * dalpha_dx + dcd_dRe * dRe_dx

                # cn, cd
                dcn_dx = dcl_dx * cphi - cl * sphi * dphi_dx + dcd_dx * sphi + cd * cphi * dphi_dx
                dct_dx = dcl_dx * sphi + cl * cphi * dphi_dx - dcd_dx * cphi + cd * sphi * dphi_dx

                # Np, Tp
                if cn == 0.0 or W == 0.0 or chord == 0.0:
                    print(phi, rotating, r, chord, theta, af, Vx, Vy)
                    raise ValueError()
                dNp_dx = Np * (1.0 / cn * dcn_dx + 2.0 / W * dW_dx + 1.0 / chord * dchord_dx)
                dTp_dx = Tp * (1.0 / ct * dct_dx + 2.0 / W * dW_dx + 1.0 / chord * dchord_dx)

                alpha_deg = np.rad2deg(alpha_rad)

            else:
                dNp_dx = None
                dTp_dx = None
                dR_dx = None

            return a, ap, Np, Tp, alpha_deg, cl, cd, cn, ct, q, W, Re, dNp_dx, dTp_dx, dR_dx

        else:
            # print('Warning: CCBlade.__loads: Wind Velocities, Vx=0, Vy=0. If unexpected, check assigned load cases, connections, and/or workflow order.')
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros(9), np.zeros(9), np.zeros(9)

    def __windComponents(self, Uinf, Omega, azimuth):
        """x, y components of wind in blade-aligned coordinate system"""

        Vx, Vy = _bem.windcomponents(
            self.r,
            self.precurve,
            self.presweep,
            self.precone,
            self.yaw,
            self.tilt,
            azimuth,
            Uinf,
            Omega,
            self.hubHt,
            self.shearExp,
        )

        if not self.derivatives:
            return Vx, Vy, 0.0, 0.0, 0.0, 0.0

        # y = [r, precurve, presweep, precone, tilt, hubHt, yaw, shear, azimuth, Uinf, Omega]  (derivative order)
        n = len(self.r)
        dy_dy = np.eye(3 * n + 8)

        _, Vxd, _, Vyd = _bem.windcomponents_dv(
            self.r,
            dy_dy[:, :n],
            self.precurve,
            dy_dy[:, n : 2 * n],
            self.presweep,
            dy_dy[:, 2 * n : 3 * n],
            self.precone,
            dy_dy[:, 3 * n],
            self.yaw,
            dy_dy[:, 3 * n + 3],
            self.tilt,
            dy_dy[:, 3 * n + 1],
            azimuth,
            dy_dy[:, 3 * n + 5],
            Uinf,
            dy_dy[:, 3 * n + 6],
            Omega,
            dy_dy[:, 3 * n + 7],
            self.hubHt,
            dy_dy[:, 3 * n + 2],
            self.shearExp,
            dy_dy[:, 3 * n + 4],
        )

        dVx_dr = np.diag(Vxd[:n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dr = np.diag(Vyd[:n, :])

        dVx_dcurve = Vxd[n : 2 * n, :]  # tri-diagonal  (note: dVx_j / dcurve_i  i==row)
        dVy_dcurve = Vyd[n : 2 * n, :]  # off-diagonal are actually all zero, but leave for convenience

        dVx_dsweep = np.diag(Vxd[2 * n : 3 * n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dsweep = np.diag(Vyd[2 * n : 3 * n, :])

        # w = [r, presweep, precone, tilt, hubHt, yaw, shear, azimuth, Uinf, Omega]
        dVx_dw = np.vstack((dVx_dr, dVx_dsweep, Vxd[3 * n :, :]))
        dVy_dw = np.vstack((dVy_dr, dVy_dsweep, Vyd[3 * n :, :]))

        return Vx, Vy, dVx_dw, dVy_dw, dVx_dcurve, dVy_dcurve

    def distributedAeroLoads(self, Uinf, Omega, pitch, azimuth):
        """Compute distributed aerodynamic loads along blade.

        Parameters
        ----------
        Uinf : float or array_like (m/s)
            hub height wind speed (float).  If desired, an array can be input which specifies
            the velocity at each radial location along the blade (useful for analyzing loads
            behind tower shadow for example).  In either case shear corrections will be applied.
        Omega : float (RPM)
            rotor rotation speed
        pitch : float (deg)
            blade pitch in same direction as :ref:`twist <blade_airfoil_coord>`
            (positive decreases angle of attack)
        azimuth : float (deg)
            the :ref:`azimuth angle <hub_azimuth_coord>` where aerodynamic loads should be computed at

        Returns
        -------
        loads : dict
            Dictionary of distributed aerodynamic loads (and other useful quantities). Keys include:

            - 'Np' : force per unit length normal to the section on downwind side (N/m)
            - 'Tp' : force per unit length tangential to the section in the direction of rotation (N/m)
            - 'a' : axial induction factor
            - 'ap' : tangential induction factor
            - 'alpha' : airfoil angle of attack (degrees)
            - 'Cl' : lift coefficient
            - 'Cd' : drag coefficient
            - 'Cn' : normal force coefficient
            - 'Ct' : tangential force coefficient
            - 'W' : airfoil relative velocity (m/s)
            - 'Re' : chord Reynolds number

        derivs : dict
            Dictionary of derivatives of distributed aerodynamic loads with respect to inputs. Keys include:

            - dNp : dictionary containing arrays (present if ``self.derivatives = True``)
                derivatives of normal loads.  Each item in the dictionary a 2D Jacobian.
                The array sizes and keys are (where n = number of stations along blade):
                n x n (diagonal): 'dr', 'dchord', 'dtheta', 'dpresweep'
                n x n (tridiagonal): 'dprecurve'
                n x 1: 'dRhub', 'dRtip', 'dprecone', 'dtilt', 'dhubHt', 'dyaw', 'dshear', 'dazimuth', 'dUinf', 'dOmega', 'dpitch'
                for example dNp_dr = dNp['dr']  (where dNp_dr is an n x n array)
                and dNp_dr[i, j] = dNp_i / dr_j
            - dTp : dictionary (present if ``self.derivatives = True``)
                derivatives of tangential loads.  Same keys as dNp.
        """

        self.pitch = np.deg2rad(pitch)
        azimuth = np.deg2rad(azimuth)

        # component of velocity at each radial station
        Vx, Vy, dVx_dw, dVy_dw, dVx_dcurve, dVy_dcurve = self.__windComponents(Uinf, Omega, azimuth)

        # initialize
        n = len(self.r)
        a = np.zeros(n)
        ap = np.zeros(n)
        alpha = np.zeros(n)
        cl = np.zeros(n)
        cd = np.zeros(n)
        Np = np.zeros(n)
        Tp = np.zeros(n)
        cn = np.zeros(n)
        ct = np.zeros(n)
        q = np.zeros(n)
        Re = np.zeros(n)
        W = np.zeros(n)

        dNp_dVx = np.zeros(n)
        dTp_dVx = np.zeros(n)
        dNp_dVy = np.zeros(n)
        dTp_dVy = np.zeros(n)
        dNp_dz = np.zeros((6, n))
        dTp_dz = np.zeros((6, n))

        if self.inverse_analysis == True:
            errf = self.__errorFunction_inverse
            self.theta = np.zeros_like(self.r)
        else:
            errf = self.__errorFunction
        rotating = Omega != 0.0

        # ---------------- loop across blade ------------------
        for i in range(n):
            # index dependent arguments
            if self.inverse_analysis == True:
                args = (self.r[i], self.chord[i], self.cl[i], self.cd[i], self.af[i], Vx[i], Vy[i])
            else:
                args = (self.r[i], self.chord[i], self.theta[i], self.af[i], Vx[i], Vy[i])

            if not rotating:  # non-rotating
                phi_star = np.pi / 2.0

            else:
                # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------

                # set standard limits
                epsilon = 1e-6
                phi_lower = epsilon
                phi_upper = np.pi / 2

                if errf(phi_lower, *args) * errf(phi_upper, *args) > 0:  # an uncommon but possible case
                    if errf(-np.pi / 4, *args) < 0 and errf(-epsilon, *args) > 0:
                        phi_lower = -np.pi / 4
                        phi_upper = -epsilon
                    else:
                        phi_lower = np.pi / 2
                        phi_upper = np.pi - epsilon

                try:
                    phi_star = brentq(errf, phi_lower, phi_upper, args=args, disp=False)

                except ValueError:
                    warnings.warn("error.  check input values.")
                    phi_star = 0.0

                # ----------------------------------------------------------------

            if self.inverse_analysis == True:
                self.theta[i] = phi_star - self.alpha[i] - self.pitch  # rad
                args = (self.r[i], self.chord[i], self.theta[i], self.af[i], Vx[i], Vy[i])

            # derivatives of residual

            (
                a[i],
                ap[i],
                Np[i],
                Tp[i],
                alpha[i],
                cl[i],
                cd[i],
                cn[i],
                ct[i],
                q[i],
                W[i],
                Re[i],
                dNp_dx,
                dTp_dx,
                dR_dx,
            ) = self.__loads(phi_star, rotating, *args)

            if np.isnan(Np[i]):
                print(f"NaNs at {i}/{n}: {phi_lower} {phi_star} {phi_upper}")
                a[i] = 0.0
                ap[i] = 0.0
                Np[i] = 0.0
                Tp[i] = 0.0
                alpha[i] = 0.0
                # print('warning, BEM convergence error, setting Np[%d] = Tp[%d] = 0.' % (i,i))

            if self.derivatives:
                # separate state vars from design vars
                # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
                dNp_dy = dNp_dx[0]
                dNp_dx = dNp_dx[1:]
                dTp_dy = dTp_dx[0]
                dTp_dx = dTp_dx[1:]
                dR_dy = dR_dx[0]
                dR_dx = dR_dx[1:]

                # direct (or adjoint) total derivatives
                DNp_Dx = dNp_dx - dNp_dy / dR_dy * dR_dx
                DTp_Dx = dTp_dx - dTp_dy / dR_dy * dR_dx

                # parse components
                # z = [r, chord, theta, Rhub, Rtip, pitch]
                zidx = [4, 0, 1, 5, 6, 7]
                dNp_dz[:, i] = DNp_Dx[zidx]
                dTp_dz[:, i] = DTp_Dx[zidx]

                dNp_dVx[i] = DNp_Dx[2]
                dTp_dVx[i] = DTp_Dx[2]

                dNp_dVy[i] = DNp_Dx[3]
                dTp_dVy[i] = DTp_Dx[3]

        derivs = {}
        if self.derivatives:
            # chain rule
            dNp_dw = dNp_dVx * dVx_dw + dNp_dVy * dVy_dw
            dTp_dw = dTp_dVx * dVx_dw + dTp_dVy * dVy_dw

            dNp_dprecurve = dNp_dVx * dVx_dcurve + dNp_dVy * dVy_dcurve
            dTp_dprecurve = dTp_dVx * dVx_dcurve + dTp_dVy * dVy_dcurve

            # stack
            # z = [r, chord, theta, Rhub, Rtip, pitch]
            # w = [r, presweep, precone, tilt, hubHt, yaw, shear, azimuth, Uinf, Omega]
            # X = [r, chord, theta, Rhub, Rtip, presweep, precone, tilt, hubHt, yaw, shear, azimuth, Uinf, Omega, pitch]
            dNp_dz[0, :] += dNp_dw[0, :]  # add partial w.r.t. r
            dTp_dz[0, :] += dTp_dw[0, :]

            dNp_dX = np.vstack((dNp_dz[:-1, :], dNp_dw[1:, :], dNp_dz[-1, :]))
            dTp_dX = np.vstack((dTp_dz[:-1, :], dTp_dw[1:, :], dTp_dz[-1, :]))

            # add chain rule for conversion to radians
            ridx = [2, 6, 7, 9, 11, 14]
            dNp_dX[ridx, :] *= np.pi / 180.0
            dTp_dX[ridx, :] *= np.pi / 180.0

            # save these values as the packing in one matrix is convenient for evaluate
            # (intended for internal use only.  not to be accessed by user)
            self._dNp_dX = dNp_dX
            self._dTp_dX = dTp_dX
            self._dNp_dprecurve = dNp_dprecurve
            self._dTp_dprecurve = dTp_dprecurve

            # pack derivatives into dictionary
            dNp = {}
            dTp = {}

            # n x n (diagonal)
            dNp["dr"] = np.diag(dNp_dX[0, :])
            dTp["dr"] = np.diag(dTp_dX[0, :])
            dNp["dchord"] = np.diag(dNp_dX[1, :])
            dTp["dchord"] = np.diag(dTp_dX[1, :])
            dNp["dtheta"] = np.diag(dNp_dX[2, :])
            dTp["dtheta"] = np.diag(dTp_dX[2, :])
            dNp["dpresweep"] = np.diag(dNp_dX[5, :])
            dTp["dpresweep"] = np.diag(dTp_dX[5, :])

            # n x n (tridiagonal)
            dNp["dprecurve"] = dNp_dprecurve.T
            dTp["dprecurve"] = dTp_dprecurve.T

            # n x 1
            dNp["dRhub"] = dNp_dX[3, :].reshape(n, 1)
            dTp["dRhub"] = dTp_dX[3, :].reshape(n, 1)
            dNp["dRtip"] = dNp_dX[4, :].reshape(n, 1)
            dTp["dRtip"] = dTp_dX[4, :].reshape(n, 1)
            dNp["dprecone"] = dNp_dX[6, :].reshape(n, 1)
            dTp["dprecone"] = dTp_dX[6, :].reshape(n, 1)
            dNp["dtilt"] = dNp_dX[7, :].reshape(n, 1)
            dTp["dtilt"] = dTp_dX[7, :].reshape(n, 1)
            dNp["dhubHt"] = dNp_dX[8, :].reshape(n, 1)
            dTp["dhubHt"] = dTp_dX[8, :].reshape(n, 1)
            dNp["dyaw"] = dNp_dX[9, :].reshape(n, 1)
            dTp["dyaw"] = dTp_dX[9, :].reshape(n, 1)
            dNp["dshear"] = dNp_dX[10, :].reshape(n, 1)
            dTp["dshear"] = dTp_dX[10, :].reshape(n, 1)
            dNp["dazimuth"] = dNp_dX[11, :].reshape(n, 1)
            dTp["dazimuth"] = dTp_dX[11, :].reshape(n, 1)
            dNp["dUinf"] = dNp_dX[12, :].reshape(n, 1)
            dTp["dUinf"] = dTp_dX[12, :].reshape(n, 1)
            dNp["dOmega"] = dNp_dX[13, :].reshape(n, 1)
            dTp["dOmega"] = dTp_dX[13, :].reshape(n, 1)
            dNp["dpitch"] = dNp_dX[14, :].reshape(n, 1)
            dTp["dpitch"] = dTp_dX[14, :].reshape(n, 1)

            derivs["dNp"] = dNp
            derivs["dTp"] = dTp

        loads = {
            "Np": Np,
            "Tp": Tp,
            "a": a,
            "ap": ap,
            "alpha": alpha,
            "Cl": cl,
            "Cd": cd,
            "Cn": cn,
            "Ct": ct,
            "W": W,
            "Re": Re,
        }

        return loads, derivs

    def evaluate(self, Uinf, Omega, pitch, coefficients=False):
        """Run the aerodynamic analysis at the specified conditions.

        Parameters
        ----------
        Uinf : array_like (m/s)
            hub height wind speed
        Omega : array_like (RPM)
            rotor rotation speed
        pitch : array_like (deg)
            blade pitch setting
        coefficients : bool, optional
            if True, results are returned in nondimensional form

        Returns
        -------
        outputs : dict
            Dictionary of integrated rotor quantities with the following keys:

            - 'P' (W) or 'CP' : Rotor power or coefficient of
            - 'T' (N) or 'CT' : Rotor thrust or coefficient of
            - 'Y' (N) or 'CY' : Rotor side force or coefficient of
            - 'Z' (N) or 'CZ' : Rotor vertical force or coefficient of
            - 'Q' (N*m) or 'CQ' : Rotor torque or coefficient of
            - 'Mb' (N*m) or 'CMb' : Blade root flap moment or coefficient of
            - 'My' (N*m) or 'CMy' : Rotor y-axis moment or coefficient of
            - 'Mz' (N*m) or 'CMz' : Rotor z-axis moment or coefficient of
        derivs: dict
            Dictionary of partial derivatives of rotor quantities with the following keys:

            - dP or dCP : dictionary of arrays (present only if derivatives==True)
                derivatives of power or power coefficient.  Each item in dictionary is a Jacobian.
                The array sizes and keys are below
                npts is the number of conditions (len(Uinf)),
                n = number of stations along blade (len(r))
                npts x 1: 'dprecone', 'dtilt', 'dhubHt', 'dRhub', 'dRtip', 'dprecurveTip', 'dpresweepTip', 'dyaw', 'dshear'
                npts x npts: 'dUinf', 'dOmega', 'dpitch'
                npts x n: 'dr', 'dchord', 'dtheta', 'dprecurve', 'dpresweep'
                for example dP_dr = dP['dr']  (where dP_dr is an npts x n array)
                and dP_dr[i, j] = dP_i / dr_j
            - dT or dCT : dictionary of arrays (present only if derivatives==True)
                derivative of thrust or thrust coefficient.  Same format as dP and dCP
            - dY or dCY : dictionary of arrays (present only if derivatives==True)
                derivative of side force or coefficient.  Same format as dP and dCP
            - dZ or dCZ : dictionary of arrays (present only if derivatives==True)
                derivative of vert force or coefficient.  Same format as dP and dCP
            - dQ or dCQ : dictionary of arrays (present only if derivatives==True)
                derivative of torque or torque coefficient.  Same format as dP and dCP
            - dMy or dCMy : dictionary of arrays (present only if derivatives==True)
                derivative of y-axis moment or coefficient.  Same format as dP and dCP
            - dMz or dCMz : dictionary of arrays (present only if derivatives==True)
                derivative of z-axis moment or coefficient.  Same format as dP and dCP
            - dMb or dCMb : dictionary of arrays (present only if derivatives==True)
                derivative of blade root flap moment or coefficient.  Same format as dP and dCP

        Notes
        -----
        CP = P / (q * Uinf * A)
        CT = T / (q * A)
        CY = Y / (q * A)
        CZ = Z / (q * A)
        CQ = Q / (q * A * R)
        CMy = My / (q * A * R)
        CMz = Mz / (q * A * R)
        CMb = Mb / (q * A * R)
        The rotor radius R, may not actually be Rtip if precone is nonzero
        ``R = Rtip*cos(precone)``
        """

        # rename
        args = (
            self.r,
            self.precurve,
            self.presweep,
            self.precone,
            self.Rhub,
            self.Rtip,
            self.precurveTip,
            self.presweepTip,
        )
        nsec = self.nSector

        # initialize
        Uinf = np.array(Uinf).flatten()
        Omega = np.array(Omega).flatten()
        pitch = np.array(pitch).flatten()

        npts = len(Uinf)
        T = np.zeros(npts)
        Y = np.zeros(npts)
        Z = np.zeros(npts)
        Q = np.zeros(npts)
        My = np.zeros(npts)
        Mz = np.zeros(npts)
        Mb = np.zeros(npts)
        P = np.zeros(npts)

        if self.derivatives:
            dT_ds = np.zeros((npts, 12))
            dY_ds = np.zeros((npts, 12))
            dZ_ds = np.zeros((npts, 12))
            dQ_ds = np.zeros((npts, 12))
            dMy_ds = np.zeros((npts, 12))
            dMz_ds = np.zeros((npts, 12))
            dMb_ds = np.zeros((npts, 12))

            nr = len(self.r)
            dT_dv = np.zeros((npts, 5, nr))
            dY_dv = np.zeros((npts, 5, nr))
            dZ_dv = np.zeros((npts, 5, nr))
            dQ_dv = np.zeros((npts, 5, nr))
            dMy_dv = np.zeros((npts, 5, nr))
            dMz_dv = np.zeros((npts, 5, nr))
            dMb_dv = np.zeros((npts, 5, nr))

        azimuth_angles = np.linspace(0.0, 2 * np.pi, nsec + 1)[:-1]
        for i in range(npts):  # iterate across conditions
            for azimuth in azimuth_angles:  # integrate across azimuth
                ca = np.cos(azimuth)
                sa = np.sin(azimuth)

                # contribution from this azimuthal location
                loads, derivs = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], np.rad2deg(azimuth))
                Np, Tp, W = (loads["Np"], loads["Tp"], loads["W"])

                Tsub, Ysub, Zsub, Qsub, Msub = _bem.thrusttorque(Np, Tp, *args)

                # Scale rotor quantities (thrust & torque) by num blades.  Keep blade root moment as is
                T[i] += self.B * Tsub / nsec
                Y[i] += self.B * (Ysub * ca - Zsub * sa) / nsec
                Z[i] += self.B * (Zsub * ca + Ysub * sa) / nsec
                Q[i] += self.B * Qsub / nsec
                My[i] += self.B * Msub * ca / nsec
                Mz[i] += self.B * Msub * sa / nsec
                Mb[i] += Msub / nsec

                if self.derivatives:
                    # dNp = derivs["dNp"]
                    # dTp = derivs["dTp"]

                    (
                        dT_ds_sub,
                        dY_ds_sub,
                        dZ_ds_sub,
                        dQ_ds_sub,
                        dM_ds_sub,
                        dT_dv_sub,
                        dY_dv_sub,
                        dZ_dv_sub,
                        dQ_dv_sub,
                        dM_dv_sub,
                    ) = self.__thrustTorqueDeriv(
                        Np, Tp, self._dNp_dX, self._dTp_dX, self._dNp_dprecurve, self._dTp_dprecurve, *args
                    )

                    dT_ds[i, :] += self.B * dT_ds_sub / nsec
                    dY_ds[i, :] += self.B * (dY_ds_sub * ca - dZ_ds_sub * sa) / nsec
                    dZ_ds[i, :] += self.B * (dZ_ds_sub * ca + dY_ds_sub * sa) / nsec
                    dQ_ds[i, :] += self.B * dQ_ds_sub / nsec
                    dMy_ds[i, :] += self.B * dM_ds_sub * ca / nsec
                    dMz_ds[i, :] += self.B * dM_ds_sub * sa / nsec
                    dMb_ds[i, :] += dM_ds_sub / nsec

                    dT_dv[i, :, :] += self.B * dT_dv_sub / nsec
                    dY_dv[i, :, :] += self.B * (dY_dv_sub * ca - dZ_dv_sub * sa) / nsec
                    dZ_dv[i, :, :] += self.B * (dZ_dv_sub * ca + dY_dv_sub * sa) / nsec
                    dQ_dv[i, :, :] += self.B * dQ_dv_sub / nsec
                    dMy_dv[i, :, :] += self.B * dM_dv_sub * ca / nsec
                    dMz_dv[i, :, :] += self.B * dM_dv_sub * sa / nsec
                    dMb_dv[i, :, :] += dM_dv_sub / nsec

        # Power
        P = Q * Omega * np.pi / 30.0  # RPM to rad/s

        if self.derivatives:
            # scalars = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, shear, Uinf, Omega, pitch]
            # vectors = [r, chord, theta, precurve, presweep]

            dP_ds = (dQ_ds.T * Omega * np.pi / 30.0).T
            dP_ds[:, 10] += Q * np.pi / 30.0
            dP_dv = (dQ_dv.T * Omega * np.pi / 30.0).T

            # pack derivatives into dictionary
            dT, dY, dZ, dQ, dMy, dMz, dMb, dP = self.__thrustTorqueDictionary(
                dT_ds,
                dY_ds,
                dZ_ds,
                dQ_ds,
                dMy_ds,
                dMz_ds,
                dMb_ds,
                dP_ds,
                dT_dv,
                dY_dv,
                dZ_dv,
                dQ_dv,
                dMy_dv,
                dMz_dv,
                dMb_dv,
                dP_dv,
                npts,
            )

        # normalize if necessary
        if coefficients:
            q = 0.5 * self.rho * Uinf**2
            A = np.pi * self.rotorR**2
            CP = P / (q * A * Uinf)
            CT = T / (q * A)
            CY = Y / (q * A)
            CZ = Z / (q * A)
            CQ = Q / (q * self.rotorR * A)
            CMy = My / (q * self.rotorR * A)
            CMz = Mz / (q * self.rotorR * A)
            CMb = Mb / (q * self.rotorR * A)

            if self.derivatives:
                # s = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, shear, Uinf, Omega, pitch]

                dR_ds = np.r_[
                    np.deg2rad(-self.Rtip * np.sin(self.precone)),
                    0.0,
                    0.0,
                    0.0,
                    np.cos(self.precone),
                    np.sin(self.precone),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                dR_ds = np.dot(np.ones((npts, 1)), np.array([dR_ds]))  # same for each operating condition

                dA_ds = 2 * np.pi * self.rotorR * dR_ds

                dU_ds = np.zeros((npts, 12))
                dU_ds[:, 9] = 1.0

                dOmega_ds = np.zeros((npts, 12))
                dOmega_ds[:, 10] = 1.0

                dq_ds = (self.rho * Uinf * dU_ds.T).T

                dCT_ds = (CT * (dT_ds.T / T - dA_ds.T / A - dq_ds.T / q)).T
                dCT_dv = (dT_dv.T / (q * A)).T

                dCY_ds = (CY * (dY_ds.T / Y - dA_ds.T / A - dq_ds.T / q)).T
                dCY_dv = (dY_dv.T / (q * A)).T

                dCZ_ds = (CZ * (dZ_ds.T / Z - dA_ds.T / A - dq_ds.T / q)).T
                dCZ_dv = (dZ_dv.T / (q * A)).T

                dCQ_ds = (CQ * (dQ_ds.T / Q - dA_ds.T / A - dq_ds.T / q - dR_ds.T / self.rotorR)).T
                dCQ_dv = (dQ_dv.T / (q * self.rotorR * A)).T

                dCMy_ds = (CMy * (dMy_ds.T / My - dA_ds.T / A - dq_ds.T / q - dR_ds.T / self.rotorR)).T
                dCMy_dv = (dMy_dv.T / (q * self.rotorR * A)).T

                dCMz_ds = (CMz * (dMz_ds.T / Mz - dA_ds.T / A - dq_ds.T / q - dR_ds.T / self.rotorR)).T
                dCMz_dv = (dMz_dv.T / (q * self.rotorR * A)).T

                dCMb_ds = (CMb * (dMb_ds.T / Mb - dA_ds.T / A - dq_ds.T / q - dR_ds.T / self.rotorR)).T
                dCMb_dv = (dMb_dv.T / (q * self.rotorR * A)).T

                dCP_ds = (CP * (dQ_ds.T / Q + dOmega_ds.T / Omega - dA_ds.T / A - dq_ds.T / q - dU_ds.T / Uinf)).T
                dCP_dv = (dQ_dv.T * CP / Q).T

                # pack derivatives into dictionary
                dCT, dCY, dCZ, dCQ, dCMy, dCMz, dCMb, dCP = self.__thrustTorqueDictionary(
                    dCT_ds,
                    dCY_ds,
                    dCZ_ds,
                    dCQ_ds,
                    dCMy_ds,
                    dCMz_ds,
                    dCMb_ds,
                    dCP_ds,
                    dCT_dv,
                    dCY_dv,
                    dCZ_dv,
                    dCQ_dv,
                    dCMy_dv,
                    dCMz_dv,
                    dCMb_dv,
                    dCP_dv,
                    npts,
                )

        outputs = {}
        derivs = {}

        outputs["P"] = P
        outputs["T"] = T
        outputs["Y"] = Y
        outputs["Z"] = Z
        outputs["Q"] = Q
        outputs["My"] = My
        outputs["Mz"] = Mz
        outputs["Mb"] = Mb
        outputs["W"] = W
        if self.derivatives:
            derivs["dP"] = dP
            derivs["dT"] = dT
            derivs["dY"] = dY
            derivs["dZ"] = dZ
            derivs["dQ"] = dQ
            derivs["dMy"] = dMy
            derivs["dMz"] = dMz
            derivs["dMb"] = dMb

        if coefficients:
            outputs["CP"] = CP
            outputs["CT"] = CT
            outputs["CY"] = CY
            outputs["CZ"] = CZ
            outputs["CQ"] = CQ
            outputs["CMy"] = CMy
            outputs["CMz"] = CMz
            outputs["CMb"] = CMb
            if self.derivatives:
                derivs["dCP"] = dCP
                derivs["dCT"] = dCT
                derivs["dCY"] = dCY
                derivs["dCZ"] = dCZ
                derivs["dCQ"] = dCQ
                derivs["dCMy"] = dCMy
                derivs["dCMz"] = dCMz
                derivs["dCMb"] = dCMb

        return outputs, derivs

    def __thrustTorqueDeriv(
        self,
        Np,
        Tp,
        dNp_dX,
        dTp_dX,
        dNp_dprecurve,
        dTp_dprecurve,
        r,
        precurve,
        presweep,
        precone,
        Rhub,
        Rtip,
        precurveTip,
        presweepTip,
    ):
        """derivatives of thrust and torque"""

        Tb = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        Yb = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        Zb = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        Qb = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        Mb = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        Npb, Tpb, rb, precurveb, presweepb, preconeb, Rhubb, Rtipb, precurvetipb, presweeptipb = _bem.thrusttorque_bv(
            Np, Tp, r, precurve, presweep, precone, Rhub, Rtip, precurveTip, presweepTip, Tb, Yb, Zb, Qb, Mb
        )

        # X = [r, chord, theta, Rhub, Rtip, presweep, precone, tilt, hubHt, yaw, shear, azimuth, Uinf, Omega, pitch]
        dT_dNp = Npb[0, :]
        dY_dNp = Npb[1, :]
        dZ_dNp = Npb[2, :]
        dQ_dNp = Npb[3, :]
        dM_dNp = Npb[4, :]

        dT_dTp = Tpb[0, :]
        dY_dTp = Tpb[1, :]
        dZ_dTp = Tpb[2, :]
        dQ_dTp = Tpb[3, :]
        dM_dTp = Tpb[4, :]

        # chain rule
        dT_dX = dT_dNp * dNp_dX + dT_dTp * dTp_dX
        dY_dX = dY_dNp * dNp_dX + dY_dTp * dTp_dX
        dZ_dX = dZ_dNp * dNp_dX + dZ_dTp * dTp_dX
        dQ_dX = dQ_dNp * dNp_dX + dQ_dTp * dTp_dX
        dM_dX = dM_dNp * dNp_dX + dM_dTp * dTp_dX

        dT_dr = dT_dX[0, :] + rb[0, :]
        dY_dr = dY_dX[0, :] + rb[1, :]
        dZ_dr = dZ_dX[0, :] + rb[2, :]
        dQ_dr = dQ_dX[0, :] + rb[3, :]
        dM_dr = dM_dX[0, :] + rb[4, :]

        dT_dchord = dT_dX[1, :]
        dY_dchord = dY_dX[1, :]
        dZ_dchord = dZ_dX[1, :]
        dQ_dchord = dQ_dX[1, :]
        dM_dchord = dM_dX[1, :]

        dT_dtheta = dT_dX[2, :]
        dY_dtheta = dY_dX[2, :]
        dZ_dtheta = dZ_dX[2, :]
        dQ_dtheta = dQ_dX[2, :]
        dM_dtheta = dM_dX[2, :]

        dT_dRhub = np.sum(dT_dX[3, :]) + Rhubb[0]
        dY_dRhub = np.sum(dY_dX[3, :]) + Rhubb[1]
        dZ_dRhub = np.sum(dZ_dX[3, :]) + Rhubb[2]
        dQ_dRhub = np.sum(dQ_dX[3, :]) + Rhubb[3]
        dM_dRhub = np.sum(dM_dX[3, :]) + Rhubb[4]

        dT_dRtip = np.sum(dT_dX[4, :]) + Rtipb[0]
        dY_dRtip = np.sum(dY_dX[4, :]) + Rtipb[1]
        dZ_dRtip = np.sum(dZ_dX[4, :]) + Rtipb[2]
        dQ_dRtip = np.sum(dQ_dX[4, :]) + Rtipb[3]
        dM_dRtip = np.sum(dM_dX[4, :]) + Rtipb[4]

        dT_dpresweep = dT_dX[5, :] + presweepb[0, :]
        dY_dpresweep = dY_dX[5, :] + presweepb[1, :]
        dZ_dpresweep = dZ_dX[5, :] + presweepb[2, :]
        dQ_dpresweep = dQ_dX[5, :] + presweepb[3, :]
        dM_dpresweep = dM_dX[5, :] + presweepb[4, :]

        preconeb_d = np.deg2rad(preconeb)
        dT_dprecone = np.sum(dT_dX[6, :]) + preconeb_d[0]
        dY_dprecone = np.sum(dY_dX[6, :]) + preconeb_d[1]
        dZ_dprecone = np.sum(dZ_dX[6, :]) + preconeb_d[2]
        dQ_dprecone = np.sum(dQ_dX[6, :]) + preconeb_d[3]
        dM_dprecone = np.sum(dM_dX[6, :]) + preconeb_d[4]

        dT_dtilt = np.sum(dT_dX[7, :])
        dY_dtilt = np.sum(dY_dX[7, :])
        dZ_dtilt = np.sum(dZ_dX[7, :])
        dQ_dtilt = np.sum(dQ_dX[7, :])
        dM_dtilt = np.sum(dM_dX[7, :])

        dT_dhubht = np.sum(dT_dX[8, :])
        dY_dhubht = np.sum(dY_dX[8, :])
        dZ_dhubht = np.sum(dZ_dX[8, :])
        dQ_dhubht = np.sum(dQ_dX[8, :])
        dM_dhubht = np.sum(dM_dX[8, :])

        dT_dprecurvetip = precurvetipb[0]
        dY_dprecurvetip = precurvetipb[1]
        dZ_dprecurvetip = precurvetipb[2]
        dQ_dprecurvetip = precurvetipb[3]
        dM_dprecurvetip = precurvetipb[4]

        dT_dpresweeptip = presweeptipb[0]
        dY_dpresweeptip = presweeptipb[1]
        dZ_dpresweeptip = presweeptipb[2]
        dQ_dpresweeptip = presweeptipb[3]
        dM_dpresweeptip = presweeptipb[4]

        dT_dprecurve = np.sum(dT_dNp * dNp_dprecurve + dT_dTp * dTp_dprecurve, axis=1) + precurveb[0, :]
        dY_dprecurve = np.sum(dY_dNp * dNp_dprecurve + dY_dTp * dTp_dprecurve, axis=1) + precurveb[1, :]
        dZ_dprecurve = np.sum(dZ_dNp * dNp_dprecurve + dZ_dTp * dTp_dprecurve, axis=1) + precurveb[2, :]
        dQ_dprecurve = np.sum(dQ_dNp * dNp_dprecurve + dQ_dTp * dTp_dprecurve, axis=1) + precurveb[3, :]
        dM_dprecurve = np.sum(dM_dNp * dNp_dprecurve + dM_dTp * dTp_dprecurve, axis=1) + precurveb[4, :]

        dT_dyaw = np.sum(dT_dX[9, :])
        dY_dyaw = np.sum(dY_dX[9, :])
        dZ_dyaw = np.sum(dZ_dX[9, :])
        dQ_dyaw = np.sum(dQ_dX[9, :])
        dM_dyaw = np.sum(dM_dX[9, :])

        dT_dshear = np.sum(dT_dX[10, :])
        dY_dshear = np.sum(dY_dX[10, :])
        dZ_dshear = np.sum(dZ_dX[10, :])
        dQ_dshear = np.sum(dQ_dX[10, :])
        dM_dshear = np.sum(dM_dX[10, :])

        dT_dUinf = np.sum(dT_dX[12, :])
        dY_dUinf = np.sum(dY_dX[12, :])
        dZ_dUinf = np.sum(dZ_dX[12, :])
        dQ_dUinf = np.sum(dQ_dX[12, :])
        dM_dUinf = np.sum(dM_dX[12, :])

        dT_dOmega = np.sum(dT_dX[13, :])
        dY_dOmega = np.sum(dY_dX[13, :])
        dZ_dOmega = np.sum(dZ_dX[13, :])
        dQ_dOmega = np.sum(dQ_dX[13, :])
        dM_dOmega = np.sum(dM_dX[13, :])

        dT_dpitch = np.sum(dT_dX[14, :])
        dY_dpitch = np.sum(dY_dX[14, :])
        dZ_dpitch = np.sum(dZ_dX[14, :])
        dQ_dpitch = np.sum(dQ_dX[14, :])
        dM_dpitch = np.sum(dM_dX[14, :])

        # scalars = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, shear, Uinf, Omega, pitch]
        dT_ds = np.array(
            [
                dT_dprecone,
                dT_dtilt,
                dT_dhubht,
                dT_dRhub,
                dT_dRtip,
                dT_dprecurvetip,
                dT_dpresweeptip,
                dT_dyaw,
                dT_dshear,
                dT_dUinf,
                dT_dOmega,
                dT_dpitch,
            ]
        )

        dY_ds = np.array(
            [
                dY_dprecone,
                dY_dtilt,
                dY_dhubht,
                dY_dRhub,
                dY_dRtip,
                dY_dprecurvetip,
                dY_dpresweeptip,
                dY_dyaw,
                dY_dshear,
                dY_dUinf,
                dY_dOmega,
                dY_dpitch,
            ]
        )

        dZ_ds = np.array(
            [
                dZ_dprecone,
                dZ_dtilt,
                dZ_dhubht,
                dZ_dRhub,
                dZ_dRtip,
                dZ_dprecurvetip,
                dZ_dpresweeptip,
                dZ_dyaw,
                dZ_dshear,
                dZ_dUinf,
                dZ_dOmega,
                dZ_dpitch,
            ]
        )

        dQ_ds = np.array(
            [
                dQ_dprecone,
                dQ_dtilt,
                dQ_dhubht,
                dQ_dRhub,
                dQ_dRtip,
                dQ_dprecurvetip,
                dQ_dpresweeptip,
                dQ_dyaw,
                dQ_dshear,
                dQ_dUinf,
                dQ_dOmega,
                dQ_dpitch,
            ]
        )

        dM_ds = np.array(
            [
                dM_dprecone,
                dM_dtilt,
                dM_dhubht,
                dM_dRhub,
                dM_dRtip,
                dM_dprecurvetip,
                dM_dpresweeptip,
                dM_dyaw,
                dM_dshear,
                dM_dUinf,
                dM_dOmega,
                dM_dpitch,
            ]
        )

        # vectors = [r, chord, theta, precurve, presweep]
        dT_dv = np.vstack((dT_dr, dT_dchord, dT_dtheta, dT_dprecurve, dT_dpresweep))
        dY_dv = np.vstack((dY_dr, dY_dchord, dY_dtheta, dY_dprecurve, dY_dpresweep))
        dZ_dv = np.vstack((dZ_dr, dZ_dchord, dZ_dtheta, dZ_dprecurve, dZ_dpresweep))
        dQ_dv = np.vstack((dQ_dr, dQ_dchord, dQ_dtheta, dQ_dprecurve, dQ_dpresweep))
        dM_dv = np.vstack((dM_dr, dM_dchord, dM_dtheta, dM_dprecurve, dM_dpresweep))

        return dT_ds, dY_ds, dZ_ds, dQ_ds, dM_ds, dT_dv, dY_dv, dZ_dv, dQ_dv, dM_dv

    def __thrustTorqueDictionary(
        self,
        dT_ds,
        dY_ds,
        dZ_ds,
        dQ_ds,
        dMy_ds,
        dMz_ds,
        dMb_ds,
        dP_ds,
        dT_dv,
        dY_dv,
        dZ_dv,
        dQ_dv,
        dMy_dv,
        dMz_dv,
        dMb_dv,
        dP_dv,
        npts,
    ):
        # pack derivatives into dictionary
        dT = {}
        dY = {}
        dZ = {}
        dQ = {}
        dMy = {}
        dMz = {}
        dMb = {}
        dP = {}

        # npts x 1
        dT["dprecone"] = dT_ds[:, 0].reshape(npts, 1)
        dY["dprecone"] = dY_ds[:, 0].reshape(npts, 1)
        dZ["dprecone"] = dZ_ds[:, 0].reshape(npts, 1)
        dQ["dprecone"] = dQ_ds[:, 0].reshape(npts, 1)
        dMy["dprecone"] = dMy_ds[:, 0].reshape(npts, 1)
        dMz["dprecone"] = dMz_ds[:, 0].reshape(npts, 1)
        dMb["dprecone"] = dMb_ds[:, 0].reshape(npts, 1)
        dP["dprecone"] = dP_ds[:, 0].reshape(npts, 1)

        dT["dtilt"] = dT_ds[:, 1].reshape(npts, 1)
        dY["dtilt"] = dY_ds[:, 1].reshape(npts, 1)
        dZ["dtilt"] = dZ_ds[:, 1].reshape(npts, 1)
        dQ["dtilt"] = dQ_ds[:, 1].reshape(npts, 1)
        dMy["dtilt"] = dMy_ds[:, 1].reshape(npts, 1)
        dMz["dtilt"] = dMz_ds[:, 1].reshape(npts, 1)
        dMb["dtilt"] = dMb_ds[:, 1].reshape(npts, 1)
        dP["dtilt"] = dP_ds[:, 1].reshape(npts, 1)

        dT["dhubHt"] = dT_ds[:, 2].reshape(npts, 1)
        dY["dhubHt"] = dY_ds[:, 2].reshape(npts, 1)
        dZ["dhubHt"] = dZ_ds[:, 2].reshape(npts, 1)
        dQ["dhubHt"] = dQ_ds[:, 2].reshape(npts, 1)
        dMy["dhubHt"] = dMy_ds[:, 2].reshape(npts, 1)
        dMz["dhubHt"] = dMz_ds[:, 2].reshape(npts, 1)
        dMb["dhubHt"] = dMb_ds[:, 2].reshape(npts, 1)
        dP["dhubHt"] = dP_ds[:, 2].reshape(npts, 1)

        dT["dRhub"] = dT_ds[:, 3].reshape(npts, 1)
        dY["dRhub"] = dY_ds[:, 3].reshape(npts, 1)
        dZ["dRhub"] = dZ_ds[:, 3].reshape(npts, 1)
        dQ["dRhub"] = dQ_ds[:, 3].reshape(npts, 1)
        dMy["dRhub"] = dMy_ds[:, 3].reshape(npts, 1)
        dMz["dRhub"] = dMz_ds[:, 3].reshape(npts, 1)
        dMb["dRhub"] = dMb_ds[:, 3].reshape(npts, 1)
        dP["dRhub"] = dP_ds[:, 3].reshape(npts, 1)

        dT["dRtip"] = dT_ds[:, 4].reshape(npts, 1)
        dY["dRtip"] = dY_ds[:, 4].reshape(npts, 1)
        dZ["dRtip"] = dZ_ds[:, 4].reshape(npts, 1)
        dQ["dRtip"] = dQ_ds[:, 4].reshape(npts, 1)
        dMy["dRtip"] = dMy_ds[:, 4].reshape(npts, 1)
        dMz["dRtip"] = dMz_ds[:, 4].reshape(npts, 1)
        dMb["dRtip"] = dMb_ds[:, 4].reshape(npts, 1)
        dP["dRtip"] = dP_ds[:, 4].reshape(npts, 1)

        dT["dprecurveTip"] = dT_ds[:, 5].reshape(npts, 1)
        dY["dprecurveTip"] = dY_ds[:, 5].reshape(npts, 1)
        dZ["dprecurveTip"] = dZ_ds[:, 5].reshape(npts, 1)
        dQ["dprecurveTip"] = dQ_ds[:, 5].reshape(npts, 1)
        dMy["dprecurveTip"] = dMy_ds[:, 5].reshape(npts, 1)
        dMz["dprecurveTip"] = dMz_ds[:, 5].reshape(npts, 1)
        dMb["dprecurveTip"] = dMb_ds[:, 5].reshape(npts, 1)
        dP["dprecurveTip"] = dP_ds[:, 5].reshape(npts, 1)

        dT["dpresweepTip"] = dT_ds[:, 6].reshape(npts, 1)
        dY["dpresweepTip"] = dY_ds[:, 6].reshape(npts, 1)
        dZ["dpresweepTip"] = dZ_ds[:, 6].reshape(npts, 1)
        dQ["dpresweepTip"] = dQ_ds[:, 6].reshape(npts, 1)
        dMy["dpresweepTip"] = dMy_ds[:, 6].reshape(npts, 1)
        dMz["dpresweepTip"] = dMz_ds[:, 6].reshape(npts, 1)
        dMb["dpresweepTip"] = dMb_ds[:, 6].reshape(npts, 1)
        dP["dpresweepTip"] = dP_ds[:, 6].reshape(npts, 1)

        dT["dyaw"] = dT_ds[:, 7].reshape(npts, 1)
        dY["dyaw"] = dY_ds[:, 7].reshape(npts, 1)
        dZ["dyaw"] = dZ_ds[:, 7].reshape(npts, 1)
        dQ["dyaw"] = dQ_ds[:, 7].reshape(npts, 1)
        dMy["dyaw"] = dMy_ds[:, 7].reshape(npts, 1)
        dMz["dyaw"] = dMz_ds[:, 7].reshape(npts, 1)
        dMb["dyaw"] = dMb_ds[:, 7].reshape(npts, 1)
        dP["dyaw"] = dP_ds[:, 7].reshape(npts, 1)

        dT["dshear"] = dT_ds[:, 8].reshape(npts, 1)
        dY["dshear"] = dY_ds[:, 8].reshape(npts, 1)
        dZ["dshear"] = dZ_ds[:, 8].reshape(npts, 1)
        dQ["dshear"] = dQ_ds[:, 8].reshape(npts, 1)
        dMy["dshear"] = dMy_ds[:, 8].reshape(npts, 1)
        dMz["dshear"] = dMz_ds[:, 8].reshape(npts, 1)
        dMb["dshear"] = dMb_ds[:, 8].reshape(npts, 1)
        dP["dshear"] = dP_ds[:, 8].reshape(npts, 1)

        # npts x npts (diagonal)
        dT["dUinf"] = np.diag(dT_ds[:, 9])
        dY["dUinf"] = np.diag(dY_ds[:, 9])
        dZ["dUinf"] = np.diag(dZ_ds[:, 9])
        dQ["dUinf"] = np.diag(dQ_ds[:, 9])
        dMy["dUinf"] = np.diag(dMy_ds[:, 9])
        dMz["dUinf"] = np.diag(dMz_ds[:, 9])
        dMb["dUinf"] = np.diag(dMb_ds[:, 9])
        dP["dUinf"] = np.diag(dP_ds[:, 9])

        dT["dOmega"] = np.diag(dT_ds[:, 10])
        dY["dOmega"] = np.diag(dY_ds[:, 10])
        dZ["dOmega"] = np.diag(dZ_ds[:, 10])
        dQ["dOmega"] = np.diag(dQ_ds[:, 10])
        dMy["dOmega"] = np.diag(dMy_ds[:, 10])
        dMz["dOmega"] = np.diag(dMz_ds[:, 10])
        dMb["dOmega"] = np.diag(dMb_ds[:, 10])
        dP["dOmega"] = np.diag(dP_ds[:, 10])

        dT["dpitch"] = np.diag(dT_ds[:, 11])
        dY["dpitch"] = np.diag(dY_ds[:, 11])
        dZ["dpitch"] = np.diag(dZ_ds[:, 11])
        dQ["dpitch"] = np.diag(dQ_ds[:, 11])
        dMy["dpitch"] = np.diag(dMy_ds[:, 11])
        dMz["dpitch"] = np.diag(dMz_ds[:, 11])
        dMb["dpitch"] = np.diag(dMb_ds[:, 11])
        dP["dpitch"] = np.diag(dP_ds[:, 11])

        # npts x n
        dT["dr"] = dT_dv[:, 0, :]
        dY["dr"] = dY_dv[:, 0, :]
        dZ["dr"] = dZ_dv[:, 0, :]
        dQ["dr"] = dQ_dv[:, 0, :]
        dMy["dr"] = dMy_dv[:, 0, :]
        dMz["dr"] = dMz_dv[:, 0, :]
        dMb["dr"] = dMb_dv[:, 0, :]
        dP["dr"] = dP_dv[:, 0, :]

        dT["dchord"] = dT_dv[:, 1, :]
        dY["dchord"] = dY_dv[:, 1, :]
        dZ["dchord"] = dZ_dv[:, 1, :]
        dQ["dchord"] = dQ_dv[:, 1, :]
        dMy["dchord"] = dMy_dv[:, 1, :]
        dMz["dchord"] = dMz_dv[:, 1, :]
        dMb["dchord"] = dMb_dv[:, 1, :]
        dP["dchord"] = dP_dv[:, 1, :]

        dT["dtheta"] = dT_dv[:, 2, :]
        dY["dtheta"] = dY_dv[:, 2, :]
        dZ["dtheta"] = dZ_dv[:, 2, :]
        dQ["dtheta"] = dQ_dv[:, 2, :]
        dMy["dtheta"] = dMy_dv[:, 2, :]
        dMz["dtheta"] = dMz_dv[:, 2, :]
        dMb["dtheta"] = dMb_dv[:, 2, :]
        dP["dtheta"] = dP_dv[:, 2, :]

        dT["dprecurve"] = dT_dv[:, 3, :]
        dY["dprecurve"] = dY_dv[:, 3, :]
        dZ["dprecurve"] = dZ_dv[:, 3, :]
        dQ["dprecurve"] = dQ_dv[:, 3, :]
        dMy["dprecurve"] = dMy_dv[:, 3, :]
        dMz["dprecurve"] = dMz_dv[:, 3, :]
        dMb["dprecurve"] = dMb_dv[:, 3, :]
        dP["dprecurve"] = dP_dv[:, 3, :]

        dT["dpresweep"] = dT_dv[:, 4, :]
        dY["dpresweep"] = dY_dv[:, 4, :]
        dZ["dpresweep"] = dZ_dv[:, 4, :]
        dQ["dpresweep"] = dQ_dv[:, 4, :]
        dMy["dpresweep"] = dMy_dv[:, 4, :]
        dMz["dpresweep"] = dMz_dv[:, 4, :]
        dMb["dpresweep"] = dMb_dv[:, 4, :]
        dP["dpresweep"] = dP_dv[:, 4, :]

        return dT, dY, dZ, dQ, dMy, dMz, dMb, dP
