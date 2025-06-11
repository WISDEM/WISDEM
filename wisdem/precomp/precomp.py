#!/usr/bin/env python
# encoding: utf-8
"""
precomp.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

import os
import numpy as np

#from wisdem.precomp._precomp import precomp as _precomp
import wisdem.precomp.properties as properties


def web_loc(r, chord, le, ib_idx, ob_idx, ib_webc, ob_webc):
    n = len(r)
    loc = np.zeros(n)

    for i in range(n):
        if i < ib_idx or i > ob_idx:
            loc[i] = -1
        else:
            xn = (r[i] - r[ib_idx]) / (r[ob_idx] - r[ib_idx])
            loc[i] = (
                le[i]
                - (le[ib_idx] - ib_webc) * chord[ib_idx] / chord[i] * (1 - xn)
                - (le[ob_idx] - ob_webc) * chord[ob_idx] / chord[i] * xn
            )

    return loc


class PreComp:
    def __init__(
        self,
        r,
        chord,
        theta,
        leLoc,
        precurve,
        presweep,
        profile,
        materials,
        upperCS,
        lowerCS,
        websCS,
        sector_idx_strain_spar_ps,
        sector_idx_strain_spar_ss,
        sector_idx_strain_te_ps,
        sector_idx_strain_te_ss,
    ):
        """Constructor

        Parameters
        ----------
        r : ndarray (m)
            radial positions. r[0] should be the hub location
            while r[-1] should be the blade tip. Any number
            of locations can be specified between these in ascending order.
        chord : ndarray (m)
            array of chord lengths at corresponding radial positions
        theta : ndarray (deg)
            array of twist angles at corresponding radial positions.
            (positive twist decreases angle of attack)
        leLoc : ndarray(float)
            array of leading-edge positions from a reference blade axis (usually blade pitch axis).
            locations are normalized by the local chord length.  e.g. leLoc[i] = 0.2 means leading edge
            is 0.2*chord[i] from reference axis.   positive in -x direction for airfoil-aligned coordinate system
        profile : list(:class:`Profile`)
            airfoil shape at each radial position
        materials : list(:class:`Orthotropic2DMaterial`), optional
            list of all Orthotropic2DMaterial objects used in defining the geometry
        upperCS, lowerCS, websCS : list(:class:`CompositeSection`)
            list of CompositeSection objections defining the properties for upper surface, lower surface,
            and shear webs (if any) for each section

        """

        self.r = np.array(r)
        self.chord = np.array(chord)
        self.theta = np.array(theta)
        self.leLoc = np.array(leLoc)
        self.precurve = np.array(precurve)
        self.presweep = np.array(presweep)

        self.profile = profile
        self.materials = materials
        self.upperCS = upperCS
        self.lowerCS = lowerCS
        self.websCS = websCS

        self.sector_idx_strain_spar_ps = sector_idx_strain_spar_ps
        self.sector_idx_strain_spar_ss = sector_idx_strain_spar_ss
        self.sector_idx_strain_te_ps = sector_idx_strain_te_ps
        self.sector_idx_strain_te_ss = sector_idx_strain_te_ss

        # twist rate
        #self.th_prime = _precomp.tw_rate(self.r, self.theta)
        self.th_prime = properties.tw_rate(self.r, self.theta)

    def sectionProperties(self):
        """see meth:`SectionStrucInterface.sectionProperties`"""

        # radial discretization
        nsec = len(self.r)

        # initialize variables
        beam_z = self.r
        beam_EA = np.zeros(nsec)
        beam_EIxx = np.zeros(nsec)
        beam_EIyy = np.zeros(nsec)
        beam_EIxy = np.zeros(nsec)
        beam_GJ = np.zeros(nsec)
        beam_EA_EIxx = np.zeros(nsec)
        beam_EA_EIyy = np.zeros(nsec)
        beam_EIxx_GJ = np.zeros(nsec)
        beam_EIyy_GJ = np.zeros(nsec)
        beam_EA_GJ = np.zeros(nsec)
        beam_rhoA = np.zeros(nsec)
        beam_A = np.zeros(nsec)
        beam_rhoJ = np.zeros(nsec)
        beam_Tw_iner = np.zeros(nsec)

        beam_flap_iner = np.zeros(nsec)
        beam_edge_iner = np.zeros(nsec)

        beam_x_sc = np.zeros(nsec)
        beam_y_sc = np.zeros(nsec)
        beam_x_tc = np.zeros(nsec)
        beam_y_tc = np.zeros(nsec)
        beam_x_cg = np.zeros(nsec)
        beam_y_cg = np.zeros(nsec)

        # distance to elastic center from airfoil nose
        # using profile coordinate system
        self.x_ec_nose = np.zeros(nsec)
        self.y_ec_nose = np.zeros(nsec)

        profile = self.profile
        mat = self.materials
        csU = self.upperCS
        csL = self.lowerCS
        csW = self.websCS

        # arrange materials into array
        n = len(mat)
        E1 = [0] * n
        E2 = [0] * n
        G12 = [0] * n
        nu12 = [0] * n
        rho = [0] * n

        for i in range(n):
            E1[i] = mat[i].E1
            E2[i] = mat[i].E2
            G12[i] = mat[i].G12
            nu12[i] = mat[i].nu12
            rho[i] = mat[i].rho

        for i in range(nsec):
            # print(i)

            xnode, ynode = profile[i]._preCompFormat()
            locU, n_laminaU, n_pliesU, tU, thetaU, mat_idxU = csU[i]._preCompFormat()
            locL, n_laminaL, n_pliesL, tL, thetaL, mat_idxL = csL[i]._preCompFormat()
            locW, n_laminaW, n_pliesW, tW, thetaW, mat_idxW = csW[i]._preCompFormat()

            nwebs = len(locW)

            # address a bug in f2py (need to pass in length 1 arrays even though they are not used)
            if nwebs == 0:
                locW = [0]
                n_laminaW = [0]
                n_pliesW = [0]
                tW = [0]
                thetaW = [0]
                mat_idxW = [0]

            (eifbar,eilbar,gjbar,eabar,eiflbar,
             sfbar,slbar,sftbar,sltbar,satbar,
             z_sc,y_sc,ztc_ref,ytc_ref,
             mass,area,iflap_eta,ilag_zeta,tw_iner,
             zcm_ref,ycm_ref) = properties.properties(
                self.chord[i],
                self.theta[i],
                self.th_prime[i],
                self.leLoc[i],
                xnode,
                ynode,
                E1,
                E2,
                G12,
                nu12,
                rho,
                locU,
                n_laminaU,
                n_pliesU,
                tU,
                thetaU,
                mat_idxU,
                locL,
                n_laminaL,
                n_pliesL,
                tL,
                thetaL,
                mat_idxL,
                nwebs,
                locW,
                n_laminaW,
                n_pliesW,
                tW,
                thetaW,
                mat_idxW,
            )
            
            beam_EIxx[i] = eilbar  # EI_lag, Section lag (edgewise) bending stiffness about the XE axis (Nm2)
            beam_EIyy[i] = eifbar  # EI_flap, Section flap bending stiffness about the YE axis (Nm2)
            beam_GJ[i] = gjbar #  Section torsion stiffness (Nm2)
            beam_EA[i] = eabar # Section axial stiffness (N)
            beam_EIxy[i] = eiflbar # Coupled flap-lag stiffness with respect to the XE-YE frame (Nm2)
            beam_EA_EIxx[i] = slbar # Coupled axial-lag stiffness with respect to the XE-YE frame (Nm.)
            beam_EA_EIyy[i] = sfbar # Coupled axial-flap stiffness with respect to the XE-YE frame (Nm)
            beam_EIxx_GJ[i] = sltbar # Coupled lag-torsion stiffness with respect to the XE-YE frame (Nm2)
            beam_EIyy_GJ[i] = sftbar # Coupled flap-torsion stiffness with respect to the XE-YE frame (Nm2)
            beam_EA_GJ[i] = satbar # Coupled axial-torsion stiffness (Nm)
            beam_x_sc[i] = z_sc # X-coordinate of the shear-center offset with respect to the ref axes (m)
            beam_y_sc[i] = y_sc # Chordwise offset of the section shear-center with respect to the reference frame, XR-YR (m)
            beam_x_tc[i] = ztc_ref # X-coordinate of the tension-center offset with respect to the XR-YR axes (m)
            beam_y_tc[i] = ytc_ref # Chordwise offset of the section tension-center with respect to the XR-YR axes (m)
            beam_rhoA[i] = mass # Section mass per unit length (kg/m)
            beam_A[i] = area # Cross-Sectional area (m)
            beam_flap_iner[i] = iflap_eta # Section flap inertia about the YG axis per unit length (kg-m)
            beam_edge_iner[i] = ilag_zeta # Section lag inertia about the XG axis per unit length (kg-m)
            beam_Tw_iner[i] = tw_iner # Orientation of the section principal inertia axes with respect the blade reference plane, theta (deg)
            beam_x_cg[i] = zcm_ref # X-coordinate of the center-of-mass offset with respect to the XR-YR axes (m)
            beam_y_cg[i] = ycm_ref # Chordwise offset of the section center of mass with respect to the XR-YR axes (m)


        beam_rhoJ = beam_flap_iner + beam_edge_iner  # perpendicular axis theorem

        self.x_ec_nose = beam_y_tc + self.leLoc * self.chord
        self.y_ec_nose = beam_x_tc  # switch b.c of coordinate system used
            
        return (
            beam_EIxx,
            beam_EIyy,
            beam_GJ,
            beam_EA,
            beam_EIxy,
            beam_EA_EIxx,
            beam_EA_EIyy,
            beam_EIxx_GJ,
            beam_EIyy_GJ,
            beam_EA_GJ,
            beam_rhoA,
            beam_A,
            beam_rhoJ,
            beam_Tw_iner,
            beam_flap_iner,
            beam_edge_iner,
            beam_x_tc,
            beam_y_tc,
            beam_x_sc,
            beam_y_sc,
            beam_x_cg,
            beam_y_cg,
        )

    def criticalStrainLocations(self, sector_idx_strain_ss, sector_idx_strain_ps):
        n = len(self.r)

        # find location of max thickness on airfoil
        xun = np.zeros(n)
        xln = np.zeros(n)
        yun = np.zeros(n)
        yln = np.zeros(n)

        # for i, p in enumerate(self.profile):
        #     xun[i], yun[i], yln[i] = p.locationOfMaxThickness()
        # xln = xun

        for i in range(n):
            csU = self.upperCS[i]
            csL = self.lowerCS[i]
            pf = self.profile[i]
            idx_ss = sector_idx_strain_ss[i]
            idx_ps = sector_idx_strain_ps[i]

            if idx_ps is None:
                xln[i] = 0.0
                yln[i] = 0.0
            else:
                xln[i] = 0.5 * (csL.loc[idx_ps] + csL.loc[idx_ps + 1])
                yln[i] = np.interp(xln[i], pf.x, pf.yl)

            if idx_ss is None:
                xun[i] = 0.0
                yun[i] = 0.0
            else:
                xun[i] = 0.5 * (csU.loc[idx_ss] + csU.loc[idx_ss + 1])
                yun[i] = np.interp(xun[i], pf.x, pf.yu)

        # make dimensional and define relative to elastic center
        xu = xun * self.chord - self.x_ec_nose
        xl = xln * self.chord - self.x_ec_nose
        yu = yun * self.chord - self.y_ec_nose
        yl = yln * self.chord - self.y_ec_nose

        # switch to airfoil coordinate system
        xu, yu = yu, xu
        xl, yl = yl, xl

        return xu, xl, yu, yl

    def panelBucklingStrain(self, sector_idx_array):
        """
        see chapter on Structural Component Design Techniques from Alastair Johnson
        section 6.2: Design of composite panels

        assumes: large aspect ratio, simply supported, uniaxial compression, flat rectangular plate

        """
        chord = self.chord
        nsec = len(self.r)

        eps_crit = np.zeros(nsec)

        for i in range(nsec):
            cs = self.upperCS[i]  # TODO: lower surface may be the compression one
            sector_idx = sector_idx_array[i]

            if sector_idx is None:
                eps_crit[i] = 0.0

            else:
                # chord-wise length of sector
                sector_length = chord[i] * (cs.loc[sector_idx + 1] - cs.loc[sector_idx])

                # get matrices
                A, B, D, totalHeight = cs.compositeMatrices(sector_idx)
                E = cs.effectiveEAxial(sector_idx)
                D1 = D[0, 0]
                D2 = D[1, 1]
                D3 = D[0, 1] + 2 * D[2, 2]

                # use empirical formula
                Nxx = 2 * (np.pi / sector_length) ** 2 * (np.sqrt(D1 * D2) + D3)
                # Nxx = 3.6 * (np.pi/sector_length)**2 * D1

                eps_crit[i] = -Nxx / totalHeight / E

        return eps_crit


if __name__ == "__main__":
    import os

    # geometry
    r_str = [
        1.5,
        1.80135,
        1.89975,
        1.99815,
        2.1027,
        2.2011,
        2.2995,
        2.87145,
        3.0006,
        3.099,
        5.60205,
        6.9981,
        8.33265,
        10.49745,
        11.75205,
        13.49865,
        15.84795,
        18.4986,
        19.95,
        21.99795,
        24.05205,
        26.1,
        28.14795,
        32.25,
        33.49845,
        36.35205,
        38.4984,
        40.44795,
        42.50205,
        43.49835,
        44.55,
        46.49955,
        48.65205,
        52.74795,
        56.16735,
        58.89795,
        61.62855,
        63.0,
    ]
    chord_str = [
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.387,
        3.39,
        3.741,
        4.035,
        4.25,
        4.478,
        4.557,
        4.616,
        4.652,
        4.543,
        4.458,
        4.356,
        4.249,
        4.131,
        4.007,
        3.748,
        3.672,
        3.502,
        3.373,
        3.256,
        3.133,
        3.073,
        3.01,
        2.893,
        2.764,
        2.518,
        2.313,
        2.086,
        1.419,
        1.085,
    ]
    theta_str = [
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        12.53,
        11.48,
        10.63,
        10.16,
        9.59,
        9.01,
        8.4,
        7.79,
        6.54,
        6.18,
        5.36,
        4.75,
        4.19,
        3.66,
        3.4,
        3.13,
        2.74,
        2.32,
        1.53,
        0.86,
        0.37,
        0.11,
        0.0,
    ]
    le_str = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.498,
        0.497,
        0.465,
        0.447,
        0.43,
        0.411,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
    ]
    web1 = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.4114,
            0.4102,
            0.4094,
            0.3876,
            0.3755,
            0.3639,
            0.345,
            0.3342,
            0.3313,
            0.3274,
            0.323,
            0.3206,
            0.3172,
            0.3138,
            0.3104,
            0.307,
            0.3003,
            0.2982,
            0.2935,
            0.2899,
            0.2867,
            0.2833,
            0.2817,
            0.2799,
            0.2767,
            0.2731,
            0.2664,
            0.2607,
            0.2562,
            0.1886,
            -1.0,
        ]
    )
    web2 = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.5886,
            0.5868,
            0.5854,
            0.5508,
            0.5315,
            0.5131,
            0.4831,
            0.4658,
            0.4687,
            0.4726,
            0.477,
            0.4794,
            0.4828,
            0.4862,
            0.4896,
            0.493,
            0.4997,
            0.5018,
            0.5065,
            0.5101,
            0.5133,
            0.5167,
            0.5183,
            0.5201,
            0.5233,
            0.5269,
            0.5336,
            0.5393,
            0.5438,
            0.6114,
            -1.0,
        ]
    )
    web3 = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        ]
    )
    precurve_str = np.zeros_like(r_str)
    presweep_str = np.zeros_like(r_str)

    # -------- materials and composite layup  -----------------
    basepath = os.path.join("5MW_files", "5MW_PreCompFiles")

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, "materials.inp"))

    ncomp = len(r_str)
    upper = [0] * ncomp
    lower = [0] * ncomp
    webs = [0] * ncomp
    profile = [0] * ncomp

    # # web 1
    # ib_idx = 7
    # ob_idx = 36
    # ib_webc = 0.4114
    # ob_webc = 0.1886

    # web1 = web_loc(r_str, chord_str, le_str, ib_idx, ob_idx, ib_webc, ob_webc)

    # # web 2
    # ib_idx = 7
    # ob_idx = 36
    # ib_webc = 0.5886
    # ob_webc = 0.6114

    # web2 = web_loc(r_str, chord_str, le_str, ib_idx, ob_idx, ib_webc, ob_webc)

    for i in range(ncomp):
        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(
            os.path.join(basepath, "layup_" + str(i + 1) + ".inp"), webLoc, materials
        )
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, "shape_" + str(i + 1) + ".inp"))
    # --------------------------------------

    precomp = PreComp(
        r_str, chord_str, theta_str, le_str, precurve_str, presweep_str, profile, materials, upper, lower, webs
    )

    # evalute section properties
    EA, EIxx, EIyy, EIxy, GJ, rhoA, rhoJ, x_ec_str, y_ec_str = precomp.sectionProperties()

    import matplotlib.pyplot as plt

    r_str = np.array(r_str)
    rstar = (r_str - r_str[0]) / (r_str[-1] - r_str[0])

    plt.figure(1)
    plt.semilogy(rstar, EIxx)
    plt.xlabel("blade fraction")
    plt.ylabel("Edgewise Stiffness ($N m^2$)")

    plt.figure(2)
    plt.semilogy(rstar, EIyy)
    plt.xlabel("blade fraction")
    plt.ylabel("Flapwise Stiffness ($N m^2$)")

    plt.figure(3)
    plt.semilogy(rstar, EA)
    plt.figure(4)
    plt.semilogy(rstar, EIxy)
    plt.figure(5)
    plt.semilogy(rstar, GJ)
    plt.figure(6)
    plt.semilogy(rstar, rhoA)
    plt.figure(7)
    plt.semilogy(rstar, rhoJ)
    plt.figure(8)
    plt.plot(rstar, x_ec_str)
    plt.figure(9)
    plt.plot(rstar, y_ec_str)
    plt.figure(10)
    plt.plot(rstar, precomp.x_ec_nose)
    plt.figure(11)
    plt.plot(rstar, precomp.y_ec_nose)

    plt.show()
