#!/usr/bin/env python
# encoding: utf-8
"""
rotorstruc.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

from zope.interface import implements
import os
import subprocess
import shutil
import platform
import numpy as np
import math

from rotorstruc import SectionStrucInterface
from twister.common.utilities import exe_path, mkdir, rmdir


isWindows = (platform.system() == 'Windows')
SCRATCH_DIR='pc_scratch' + os.path.sep


# --- convenience methods for reading/writing Precomp files ----

def skipLines(f, n):
    for i in range(n):
        f.readline()


def file3col(f, data, name, description):
    print >> f, '{0!s:20} {1:20} {2:40}'.format(data, name, description)


def file2col(f, data, description):
    print >> f, '{0!s:20} {1:<40}'.format(data, description)


def file5col(f, s1, s2, s3, s4, s5):
    print >> f, '{0!s:10} {1!s:10} {2!s:15} {3!s:20} {4!s:20}'.format(s1, s2, s3, s4, s5)


def file7col(f, s1, s2, s3, s4, s5, s6, s7):
    print >> f, '{0!s:10} {1!s:10} {2!s:10} {3!s:10} {4!s:10} {5!s:10} {6!s:20}'.format(s1, s2, s3, s4, s5, s6, s7)


def file7colnum(f, s1, s2, s3, s4, s5, s6, s7):
    print >> f, '{0:<10d} {1:<10.6g} {2:<10.6g} {3:<10.6g} {4:<10.6g} {5:<10.6g} {6:20}'.format(s1, s2, s3, s4, s5, s6, s7)

# ------------------------------------




class PreComp:
    implements(SectionStrucInterface)



    def __init__(self, r, chord, theta, profile, compSec, leLoc, materials=None,
                 precompPath=None, DEBUG=False):
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
        profile : ndarray(:class:`Profile`)
            airfoil shape at each radial position
        compSec : ndarray(:class:`CompositeSection`)
            composite section definition at each radial position
        leLoc : ndarray(float)
            array of leading-edge positions from a reference blade axis (usually blade pitch axis).
            locations are normalized by the local chord length.  e.g. leLoc[i] = 0.2 means leading edge
            is 0.2*chord[i] from reference axis.   positive in -x direction for airfoil-aligned coordinate system
        materials : list(:class:`Orthotropic2DMaterial`), optional
            list of all Orthotropic2DMaterial objects used in defining the geometry
            if not included it can be figured out from the provided CompositeSection objects
        precompPath : str
            path to preComp executable.  If not provided assumed to be in default location
            which is in same directory as this file.

        """
        self.r = np.array(r)
        self.chord = np.array(chord)
        self.theta = np.array(theta)
        self.profile = profile
        self.compSec = compSec
        self.leLoc = leLoc
        self.materials = materials

        self.precompPath = exe_path(precompPath, 'precomp', os.path.dirname(__file__))
        self.DEBUG = DEBUG

        # create working directory
        mkdir(SCRATCH_DIR)

        # create a symlink if necessary (only for *nix, windows must copy)
        if not isWindows:
            mfile = 'materials.inp'

            if not os.path.exists(SCRATCH_DIR+mfile):
                f = open(SCRATCH_DIR+mfile, 'w')
                f.close()

            if not os.path.exists(mfile):
                os.symlink(SCRATCH_DIR+mfile, mfile)

            elif not os.path.islink(mfile) or not os.readlink(mfile) == SCRATCH_DIR+mfile:
                os.remove(mfile)
                os.symlink(SCRATCH_DIR+mfile, mfile)


    def __del__(self):
        """remove all PreComp input/output files unless in debug mode"""
        if not self.DEBUG:
            rmdir(SCRATCH_DIR)
            os.remove('materials.inp')



    def sectionProperties(self):
        """see meth:`SectionStrucInterface.sectionProperties`"""

        # radial discretization
        nsec = len(self.r)

        # write materials file
        if self.materials is None:
            self.materials = set()
            for cs in self.compSec:
                self.materials = self.materials.union(cs.getSetOfMaterials())
        Orthotropic2DMaterial.writeToPreCompFile(SCRATCH_DIR+'materials.inp', self.materials)

        if isWindows:  # because symlinks don't work
            shutil.copy(SCRATCH_DIR+'materials.inp', 'materials.inp')

        # compute web locations
        webIdxI = -1
        webIdxO = -1
        webLocI = []
        webLocO = []
        for i, cs in enumerate(self.compSec):
            if len(cs.webLoc) > 0:
                if webIdxI == -1:
                    webIdxI = i
                    webLocI = cs.webLoc
                webIdxO = i
                webLocO = cs.webLoc


        # write layup files
        str_files = [0]*nsec

        for i, cs in enumerate(self.compSec):
            str_files[i] = SCRATCH_DIR + 'lay' + str(i) + '.inp'
            cs.writeToPreCompLayupFile(str_files[i])

        # write profile files
        profile_files = [0]*nsec

        for i, pf in enumerate(self.profile):
            profile_files[i] = SCRATCH_DIR + 'af' + str(i) + '.inp'
            pf.writeToPrecompFile(profile_files[i])


        # normalize spam location
        rnorm = self.r - self.r[0]
        rnorm /= rnorm[-1]

        # create main input file
        inputfile = SCRATCH_DIR + 'precomp.pci'
        f = open(inputfile, 'w')

        print >> f, '*****************  main input file for PreComp *****************************'
        print >> f, 'Auto-generated PreComp input file.'
        print >> f, ''

        print >> f, 'General information -----------------------------------------------'
        file3col(f, self.r[-1] - self.r[0], 'Bl_length:', ': blade length (m)')
        file3col(f, nsec, 'N_sections:', ': no of blade sections (-)')
        file3col(f, len(self.materials), 'N_materials:', ': no of materials listed in the materials table (material.inp)')
        file3col(f, 1, 'Out_format:', ': output file   (1: general format, 2: BModes-format, 3: both)')
        file3col(f, False, 'TabDelim:', ': (true: tab-delimited table; false: space-delimited table)')
        print >> f

        print >> f, 'Blade-sections-specific data --------------------------------------'
        print >> f, 'Sec span         l.e.         chord       aerodynamic       af_shape        int str layup'
        print >> f, 'location       position       length        twist             file              file'
        print >> f, 'Span_loc        Le_loc        Chord        Tw_aero       Af_shape_file      Int_str_file'
        print >> f, '  (-)            (-)           (m)        (degrees)           (-)               (-)'
        print >> f
        for r, le, c, t, af, lay in zip(rnorm, self.leLoc, self.chord, self.theta, profile_files, str_files):
            print >> f, '{0:<15f} {1:<15f} {2:<15f} {3:<15f} \'{4:<15}\' \'{5:<15}\''.format(r, le, c, t, af, lay)
        print >> f

        print >> f, 'Webs (spars) data  --------------------------------------------------'
        print >> f
        file3col(f, len(webLocI), 'Nweb:', ': number of webs (-)  ! enter 0 if the blade has no webs')
        file3col(f, webIdxI+1, 'Ib_sp_stn:', ': blade station number where inner-most end of webs is located (-)')
        file3col(f, webIdxO+1, 'Ob_sp_stn:', ': blade station number where outer-most end of webs is located (-)')
        print >> f
        print >> f, 'Web_num   Inb_end_ch_loc   Oub_end_ch_loc (fraction of chord length)'
        for idx, (inb, oub) in enumerate(zip(webLocI, webLocO)):
            print >> f, '{0:<15} {1:<15f} {2:<15f}'.format(idx+1, inb, oub)

        f.close()

        # remove any existing output file just to be sure precomp run successfully
        outfile = SCRATCH_DIR + 'precomp.out_gen'
        try:
            os.remove(outfile)
        except OSError:
            pass

        # run precomp
        f = open(SCRATCH_DIR + 'output', 'w')
        process = subprocess.Popen([self.precompPath, inputfile], shell=False, stdout=f)
        process.communicate()  # clear buffer and wait for process to terminate
        f.close()

        # open output file
        f = open(outfile)

        # skip header
        skipLines(f, 9)

        # initialize variables
        EA = np.zeros(nsec)
        EIxx = np.zeros(nsec)
        EIyy = np.zeros(nsec)
        EIxy = np.zeros(nsec)
        GJ = np.zeros(nsec)
        rhoA = np.zeros(nsec)
        rhoJ = np.zeros(nsec)

        # distance to elastic center from point about which structural properties are computed
        # using airfoil coordinate system
        x_ec_str = np.zeros(nsec)
        y_ec_str = np.zeros(nsec)

        # distance to elastic center from airfoil nose
        # using profile coordinate system
        self.x_ec_nose = np.zeros(nsec)
        self.y_ec_nose = np.zeros(nsec)


        # read data
        for idx, line in enumerate(f):
            values = [float(s) for s in line.split()]

            EIxx[idx] = values[4]  # EIedge
            EIyy[idx] = values[3]  # EIflat
            GJ[idx] = values[5]
            EA[idx] = values[6]
            EIxy[idx] = values[7]  # EIflapedge
            x_ec_str[idx] = values[15] - values[13]
            y_ec_str[idx] = values[16] - values[14]
            rhoA[idx] = values[17]
            rhoJ[idx] = values[18] + values[19]  # perpindicular axis theorem

            self.x_ec_nose[idx] = values[16] + self.leLoc[idx]*self.chord[idx]
            self.y_ec_nose[idx] = values[15]  # switch b.c of coordinate system used

        f.close()

        return self.r, EA, EIxx, EIyy, EIxy, GJ, rhoA, rhoJ, x_ec_str, y_ec_str



    def criticalStrainLocations(self):

        n = len(self.r)

        # find location of max thickness on airfoil
        xn = np.zeros(n)
        yun = np.zeros(n)
        yln = np.zeros(n)

        for i, p in enumerate(self.profile):
            xn[i], yun[i], yln[i] = p.locationOfMaxThickness()

        # evaluate on both upper and lower surface
        xu = np.zeros(n)
        yu = np.zeros(n)
        xl = np.zeros(n)
        yl = np.zeros(n)

        xu = xn*self.chord - self.x_ec_nose  # define relative to elastic center
        xl = xn*self.chord - self.x_ec_nose
        yu = yun*self.chord - self.y_ec_nose
        yl = yln*self.chord - self.y_ec_nose

        # switch to airfoil coordinate system
        xu, yu = yu, xu
        xl, yl = yl, xl

        return xu, yu, xl, yl



    def panelBucklingStrain(self, sector_idx_array):
        """
        see chapter on Structural Component Design Techniques from Alastair Johnson
        section 6.2: Design of composite panels

        assumes: large aspect ratio, simply supported, uniaxial compression, flat rectangular plate

        """

        cs = self.compSec
        chord = self.chord
        nsec = len(self.r)

        eps_crit = np.zeros(nsec)

        for i in range(nsec):

            # get sector
            sector_idx = sector_idx_array[i]
            sector = cs[i].secListUpper[sector_idx]  # TODO: lower surface may be the compression one

            # chord-wise length of sector
            locations = np.concatenate(([0.0], cs[i].sectorLocU, [1.0]))
            sector_length = chord[i] * (locations[sector_idx+1] - locations[sector_idx])

            # get matrices
            A, B, D, totalHeight = sector.compositeMatrices()
            E = sector.effectiveEAxial()
            D1 = D[0, 0]
            D2 = D[1, 1]
            D3 = D[0, 1] + 2*D[2, 2]

            # use empirical formula
            Nxx = 2 * (math.pi/sector_length)**2 * (math.sqrt(D1*D2) + D3)
            # Nxx = 3.6 * (math.pi/sector_length)**2 * D1

            Nxx *= 3.9  # a fudge factor that gives good agreement with other simple method
            # need some ANSYS tests to pick a better factor.

            eps_crit[i] = - Nxx / totalHeight / E


        return eps_crit








class CompositeSection:
    """A CompositeSection defines the layup of the entire
    airfoil cross-section

    """

    def __init__(self, sectorLocU, secListUpper, sectorLocL, secListLower, webLoc, secListWeb):
        """Constructor

        Parameters
        ----------
        webLoc : ndarray
            array of web locations (i.e. [0.15, 0.5] has two webs
            one located at 15% chord from the leading edge and
            the second located at 50% chord)
        secListUpper : list(:class:`Sector`)
            a list of sectors for the upper surface defined between the points in webLoc
            should be of length: len(webLoc) + 1
        secListLower : list(:class:`Sector`)
            similar sector list for the lower surface
        secListWeb : list(:class:`Sector`)
            array of sectors for the webs starting from first
            web. should be of length: len(webLoc)

        """

        self.sectorLocU = sectorLocU
        self.secListUpper = secListUpper
        self.sectorLocL = sectorLocL
        self.secListLower = secListLower
        self.webLoc = webLoc
        self.secListWeb = secListWeb


    @classmethod
    def initFromPreCompLayupFile(cls, fname, webLoc, materials):
        """Construct CompositeSection object from a PreComp input file

        .. TODO:: can't remember why I did this.  why can't I just get webLoc from the main input file.

        Parameters
        ----------
        fname : str
            name of input file
        webLoc : ndarray
            array of web locations (i.e. [0.15, 0.5] has two webs
            one located at 15% chord from the leading edge and
            the second located at 50% chord)
        materials : list(:class:`Orthotropic2DMaterial`)
            material objects defined in same order as used in the input file
            can use :meth:`Orthotropic2DMaterial.initFromPreCompFile`

        Returns
        -------
        compSec : CompositeSection
            an initialized CompositeSection object

        """

        f = open(fname)

        skipLines(f, 3)

        # number of sectors
        n_sector = int(f.readline().split()[0])

        skipLines(f, 2)

        # read normalized chord locations
        sectorLocU = [float(x) for x in f.readline().split()]
        sectorLocU = sectorLocU[1:-1]  # don't need first and last

        sectorU = CompositeSection.__readSectorsFromFile(f, n_sector, materials)

        skipLines(f, 3)

        # number of sectors
        n_sector = int(f.readline().split()[0])

        skipLines(f, 2)

        sectorLocL = [float(x) for x in f.readline().split()]
        sectorLocL = sectorLocL[1:-1]

        sectorL = CompositeSection.__readSectorsFromFile(f, n_sector, materials)

        skipLines(f, 4)

        n_sector = len(webLoc)

        sectorW = CompositeSection.__readSectorsFromFile(f, n_sector, materials)

        f.close()

        return cls(sectorLocU, sectorU, sectorLocL, sectorL, webLoc, sectorW)


    @staticmethod
    def __readSectorsFromFile(f, n_sector, materials):
        """private method"""

        sectors = [0]*n_sector

        for i in range(n_sector):
            skipLines(f, 2)

            line = f.readline()
            if line == '':
                return []  # no webs
            n_lamina = int(line.split()[1])

            skipLines(f, 4)

            lam = [0]*n_lamina

            for j in range(n_lamina):
                array = f.readline().split()
                n_ply = int(array[1])
                t_ply = float(array[2])
                direction = float(array[3])
                n_mat = int(array[4])

                lam[j] = Lamina(n_ply, t_ply, direction, materials[n_mat-1])

            sectors[i] = Sector(*lam)

        return sectors


    def writeToPreCompLayupFile(self, fname):
        """Write CompositeSection data to PreComp file

        Parameters
        ----------
        fname : str
            name/path of output file

        """

        locationU = '0.0'
        for loc in self.sectorLocU:
            locationU += '   ' + str(loc)
        locationU += '   1.0'

        locationL = '0.0'
        for loc in self.sectorLocL:
            locationL += '   ' + str(loc)
        locationL += '   1.0'


        f = open(fname, 'w')

        print >> f, 'Composite laminae lay-up inside the blade section'
        print >> f
        print >> f, '*************************** TOP SURFACE ****************************'
        file2col(f, len(self.secListUpper), 'N_scts(1):  no of sectors on top surface')
        print >> f
        print >> f, 'normalized chord location of nodes defining airfoil sectors boundaries (xsec_node)'
        print >> f, locationU

        CompositeSection.__writeSectorsToFile(f, self.secListUpper)
        print >> f
        print >> f

        print >> f, '*************************** BOTTOM SURFACE ****************************'
        file2col(f, len(self.secListLower), 'N_scts(2):  no of sectors on bottom surface')
        print >> f
        print >> f, 'normalized chord location of nodes defining airfoil sectors boundaries (xsec_node)'
        print >> f, locationL

        CompositeSection.__writeSectorsToFile(f, self.secListLower)
        print >> f
        print >> f

        print >> f, '*************************** WEBS ****************************'
        print >> f, 'Laminae schedule for webs (input required only if webs exist at this section):'
        CompositeSection.__writeSectorsToFile(f, self.secListWeb)

        f.close()


    @staticmethod
    def __writeSectorsToFile(f, sectorList):
        """private method"""

        for i, sector in enumerate(sectorList):
            print >> f, '..................................................................'
            file2col(f, 'Sect_num', 'no of laminae (N_laminas)')
            file2col(f, i+1, sector.nLamina)
            print >> f
            file5col(f, 'lamina', 'num of', 'thickness', 'fibers_direction', 'composite_material ID')
            file5col(f, 'number', 'plies', 'of ply (m)', '(deg)', '(-)')
            file5col(f, 'lam_num', 'N_plies', 'Tply', 'Tht_lam', 'Mat_id')
            for j, lam in enumerate(sector.laminaList):
                file5col(f, j+1, lam.n_plies, lam.t, lam.direction,
                         str(lam.material.mat_id) + ' (' + lam.material.name + ')')





    def getSetOfMaterials(self):
        """Returns a set of all :class:`Orthotropic2DMaterial` objects used in this section

        Returns
        -------
        materialSet : set
            a set of all :class:`Orthotropic2DMaterial` objects used in this CompositeSection

        """

        materialSet = set()
        for sector in self.secListUpper + self.secListLower + self.secListWeb:
            for lamina in sector.laminaList:
                materialSet.add(lamina.material)

        return materialSet



class Sector:
    """A Sector is a sequence of lamina that applies over
    some section of the airfoil cross-section

    """

    def __init__(self, *laminas):
        """Constructor

        Parameters
        ----------
        laminas : variable length argument list of Lamina objects
            ordered list of the laminate sequence

        Notes
        -----
        Available parameters include

        laminaList : list(:class:`Lamina`)
            the laminate sequence
        nLamina : int
            the number of Lamina

        """

        self.laminaList = laminas
        self.nLamina = len(self.laminaList)


    def compositeMatrices(self):
        """Computes the matrix components defining the constituitive equations
        of the complete laminate stack

        Returns
        -------
        A : ndarray, shape(3, 3)
            upper left portion of constitutive matrix
        B : ndarray, shape(3, 3)
            off-diagonal portion of constitutive matrix
        D : ndarray, shape(3, 3)
            lower right portion of constitutive matrix
        totalHeight : float (m)
            total height of the laminate stack

        Notes
        -----
        | The constitutive equations are arranged in the format
        | [N; M] = [A B; B D] * [epsilon; k]
        | where N = [N_x, N_y, N_xy] are the normal force resultants for the laminate
        | M = [M_x, M_y, M_xy] are the moment resultants
        | epsilon = [epsilon_x, epsilon_y, gamma_xy] are the midplane strains
        | k = [k_x, k_y, k_xy] are the midplane curvates

        See [1]_ for further details, and this :ref:`equation <ABBD>` in the user guide.

        References
        ----------
        .. [1] J. C. Halpin. Primer on Composite Materials Analysis. Technomic, 2nd edition, 1992.

        """

        n = self.nLamina
        laminas = self.laminaList

        # heights (z - absolute, h - relative to mid-plane)
        z = np.zeros(n+1)
        for i, lam in enumerate(laminas):
            z[i+1] = z[i] + lam.t*lam.n_plies

        z_mid = (z[-1] - z[0]) / 2.0
        h = z - z_mid

        # ABD matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for i in range(n):
            Qbar = self.laminaList[i].computeQbar()
            A += Qbar*(h[i+1] - h[i])
            B += 0.5*Qbar*(h[i+1]**2 - h[i]**2)
            D += 1.0/3.0*Qbar*(h[i+1]**3 - h[i]**3)

        totalHeight = z[-1] - z[0]

        return A, B, D, totalHeight



    def effectiveEAxial(self):
        """Estimates the effective axial modulus of elasticity for the laminate

        Returns
        -------
        E : float (N/m^2)
            effective axial modulus of elasticity

        Notes
        -----
        see user guide for a :ref:`derivation <ABBD>`

        """

        A, B, D, totalHeight = self.compositeMatrices()

        # S = [A B; B D]

        S = np.vstack((np.hstack((A, B)), np.hstack((B, D))))

        # E_eff_x = N_x/h/eps_xx and eps_xx = S^{-1}(0,0)*N_x (approximately)
        detS = np.linalg.det(S)
        Eaxial = detS/np.linalg.det(S[1:, 1:])/totalHeight

        return Eaxial



class Lamina:
    """Represents a composite lamina

    """
    def __init__(self, n_plies, t, direction, material):
        """
        A struct-like object.  All inputs are also fields.

        Parameters
        ----------
        n_plies : int
            number of plies
        t : float (m)
            thickness of one ply
        direction : float (deg)
            ply orientation.  positive rotation is about the 3 axis
            where 1 and 2 come from the material directions E1 and E2
            and direction 3 is defined by the right hand rule.  The
            angle is defined between the x-axis and the 1-axis.
            see :ref:`here <ply>` for a picture of ply angle.
        material : Orthotropic2DMaterial
            material used in this lamina

        """

        self.n_plies = n_plies
        self.t = t
        self.direction = direction
        self.material = material


    def computeQbar(self):
        """Computes the lamina stiffness matrix

        Returns
        -------
        Qbar : numpy matrix
            the lamina stifness matrix

        Notes
        -----
        Transforms a specially orthotropic lamina from principal axis to
        an arbitrary axis defined by the ply orientation.
        [sigma_x; sigma_y; tau_xy]^T = Qbar * [epsilon_x; epsilon_y, gamma_xy]^T
        See [1]_ for further details.

        References
        ----------
        .. [1] J. C. Halpin. Primer on Composite Materials Analysis. Technomic, 2nd edition, 1992.


        """

        E11 = self.material.E1
        E22 = self.material.E2
        nu12 = self.material.nu12
        nu21 = nu12*E22/E11
        G12 = self.material.G12
        denom = (1 - nu12*nu21)

        theta = math.radians(self.direction)
        c = math.cos(theta)
        s = math.sin(theta)
        c2 = c*c
        s2 = s*s
        cs = c*s

        Q = np.mat([[E11/denom, nu12*E22/denom, 0],
                    [nu12*E22/denom, E22/denom, 0],
                    [0, 0, G12]])
        T12 = np.mat([[c2, s2, cs],
                      [s2, c2, -cs],
                      [-cs, cs, 0.5*(c2-s2)]])
        Tinv = np.mat([[c2, s2, -2*cs],
                       [s2, c2, 2*cs],
                       [cs, -cs, c2-s2]])

        return Tinv*Q*T12



class Orthotropic2DMaterial:
    """Represents a homogeneous orthotropic material in a
    plane stress state.

    """

    next_id = 1

    def __init__(self, E1, E2, G12, nu12, rho, name=''):
        """a struct-like object.  all inputs are also fields.
        The object also has an identification
        number *.mat_id so unique materials can be identified.

        Parameters
        ----------
        E1 : float (N/m^2)
            Young's modulus in first principal direction
        E2 : float (N/m^2)
            Young's modulus in second principal direction
        G12 : float (N/m^2)
            shear modulus
        nu12 : float
            Poisson's ratio  (nu12*E22 = nu21*E11)
        rho : float (kg/m^3)
            density
        name : str
            an optional identifier

        """
        self.mat_id = Orthotropic2DMaterial.next_id
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.rho = rho
        self.name = name

        Orthotropic2DMaterial.next_id += 1


    def __hash__(self):
        # return self.mat_id
        return hash((self.E1, self.E1, self.G12, self.nu12, self.rho))


    @classmethod
    def initFromPreCompFile(cls, fname):
        """initialize the object by extracting materials from a PreComp materials file

        Parameters
        ----------
        fname : str
            name of input file

        Returns
        -------
        materials : List(:class:`Orthotropic2DMaterial`)
            a list of all materials gathered from the file.

        """

        f = open(fname)

        skipLines(f, 3)

        materials = []
        for line in f:
            array = line.split()
            mat = cls(float(array[1]), float(array[2]), float(array[3]),
                      float(array[4]), float(array[5]), array[6])
            mat.mat_id = int(array[0])

            materials.append(mat)

        return materials

        f.close()


    @staticmethod
    def writeToPreCompFile(fname, materials):
        """Write object data to file in PreComp format

        Parameters
        ----------
        fname : str
            name/path of output file
        materials : list(:class:`Orthotropic2DMaterial`)
            a list of materials

        """

        f = open(fname, 'w')

        file7col(f, 'Mat_Id', 'E1', 'E2', 'G12', 'Nu12', 'Density', 'Mat_Name')
        file7col(f, '(-)', '(Pa)', '(Pa)', '(Pa)', '(-)', '(Kg/m^3)', '(-)')
        print >> f
        for i, mat in enumerate(materials):
            file7colnum(f, mat.mat_id, mat.E1, mat.E2, mat.G12, mat.nu12, mat.rho, mat.name)
        print >> f

        f.close()




class Profile:
    """Defines the shape of an airfoil"""

    def __init__(self, xu, yu, xl, yl):
        """Constructor

        Parameters
        ----------
        xu : ndarray
            x coordinates for upper surface of airfoil
        yu : ndarray
            y coordinates for upper surface of airfoil
        xl : ndarray
            x coordinates for lower surface of airfoil
        yl : ndarray
            y coordinates for lower surface of airfoil

        Notes
        -----
        uses :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`.
        Nodes should be ordered from the leading edge toward the trailing edge.
        The leading edge can be located at any position and the
        chord may be any size, however the airfoil should be untwisted.
        Normalization to unit chord will happen internally.

        """

        # parse airfoil data
        xu = np.array(xu)
        yu = np.array(yu)
        xl = np.array(xl)
        yl = np.array(yl)

        # ensure leading edge at zero
        xu -= xu[0]
        xl -= xl[0]
        yu -= yu[0]
        yl -= yl[0]

        # ensure unit chord
        c = xu[-1] - xu[0]
        xu /= c
        xl /= c
        yu /= c
        yl /= c

        # interpolate onto common grid
        arc = np.linspace(0, math.pi, 100)
        self.x = 0.5*(1-np.cos(arc))  # cosine spacing
        self.yu = np.interp(self.x, xu, yu)
        self.yl = np.interp(self.x, xl, yl)

        # compute thickness to chord ratio
        self.tc = max(self.yu - self.yl)



    @classmethod
    def initWithTEtoTEdata(cls, x, y):
        """Factory constructor for data points ordered from trailing edge to trailing edge.

        Parameters
        ----------
        x, y : ndarray, ndarray
            airfoil coordinates starting at trailing edge and
            ending at trailing edge, traversing airfoil in either direction.

        Returns
        -------
        profile : Profile
            initialized Profile object

        Notes
        -----
        It is not necessary to start and end at the same point
        for an airfoil with trailing edge thickness.
        Although, one point should be right at the nose.
        see also notes for :meth:`__init__`

        """

        # parse airfoil data
        x = np.array(x)
        y = np.array(y)

        # separate into 2 halves
        for i in range(len(x)):
            if x[i+1] >= x[i]:
                break

        xu = x[i::-1]
        yu = y[i::-1]
        xl = x[i:]
        yl = y[i:]


        # check if coordinates were input in other direction
        if y[1] < y[0]:
            temp = yu
            yu = yl
            yl = temp

            temp = xu
            xu = xl
            xl = temp

        return cls(xu, yu, xl, yl)


    @classmethod
    def initWithLEtoLEdata(cls, x, y):
        """Factory constructor for data points ordered from leading edge to leading edge.

        Parameters
        ----------
        x, y : ndarray, ndarray
            airfoil coordinates starting at leading edge and
            ending at leading edge, traversing airfoil in either direction.

        Returns
        -------
        profile : Profile
            initialized Profile object

        Notes
        -----
        x,y data must start and end at the same point.
        see also notes for :meth:`__init__`

        """

        # parse airfoil data
        x = np.array(x)
        y = np.array(y)

        # separate into 2 halves
        for i in range(len(x)):
            if x[i+1] <= x[i]:
                iuLast = i
                ilLast = i
                if x[i+1] == x[i]:  # blunt t.e.
                    ilLast = i+1  # stop at i+1
                break

        xu = x[:iuLast+1]
        yu = y[:iuLast+1]
        xl = x[-1:ilLast-1:-1]
        yl = y[-1:ilLast-1:-1]

        # check if coordinates were input in other direction
        if y[1] < y[0]:
            temp = yu
            yu = yl
            yl = temp

            temp = xu
            xu = xl
            xl = temp

        return cls(xu, yu, xl, yl)


    @staticmethod
    def initFromPreCompFile(precompProfileFile):
        """Construct profile from PreComp formatted file

        Parameters
        ----------
        precompProfileFile : str
            path/name of file

        Returns
        -------
        profile : Profile
            initialized Profile object

        """

        return Profile.initFromFile(precompProfileFile, 4, True)




    @staticmethod
    def initFromFile(filename, numHeaderlines, LEtoLE):
        """Construct profile from a generic form text file (see Notes)

        Parameters
        ----------
        filename : str
            name/path of input file
        numHeaderlines : int
            number of header rows in input file
        LEtoLE : boolean
            True if data is ordered from leading-edge to leading-edge
            False if from trailing-edge to trailing-edge

        Returns
        -------
        profile : Profile
            initialized Profile object

        Notes
        -----
        file should be of the form:

        header row
        header row
        x1 y1
        x2 y2
        x3 y3
        .  .
        .  .
        .  .

        where any number of header rows can be used.

        """

        # open file
        f = open(filename, 'r')

        # skip through header
        for i in range(numHeaderlines):
            f.readline()

        # loop through
        x = []
        y = []

        for line in f:
            if not line.strip():
                break  # break if empty line
            data = line.split()
            x.append(float(data[0]))
            y.append(float(data[1]))

        f.close()

        # close nose if LE to LE
        if LEtoLE:
            x.append(x[0])
            y.append(y[0])
            return Profile.initWithLEtoLEdata(x, y)

        else:
            return Profile.initWithTEtoTEdata(x, y)




    def writeToPrecompFile(self, filename):
        """Write the airfoil section data to a file using PreComp input file style.

        Parameters
        ----------
        filename : str
            name/path of output file

        """

        # check if they share a common trailing edge point
        te_same = self.yu[-1] == self.yl[-1]

        # count number of points
        nu = len(self.x)
        if (te_same):
            nu -= 1
        nl = len(self.x) - 1  # they do share common leading-edge
        n = nu + nl

        # initialize
        x = np.zeros(n)
        y = np.zeros(n)

        # leading edge round to leading edge
        x[0:nu] = self.x[0:nu]
        y[0:nu] = self.yu[0:nu]
        x[nu:] = self.x[:0:-1]
        y[nu:] = self.yl[:0:-1]


        f = open(filename, 'w')
        print >> f, '{0:<10d} {1:40}'.format(n, 'n_af_nodes: # of airfoil nodes.')
        print >> f, '{0:<10} {1:40}'.format('', 'clockwise starting at leading edge')
        print >> f
        print >> f, 'xnode \t ynode'
        for xpt, ypt in zip(x, y):
            print >> f, '{0:<15f} {1:15f}'.format(xpt, ypt)

        f.close()


    def locationOfMaxThickness(self):
        """Find location of max airfoil thickness

        Returns
        -------
        x : float
            x location of maximum thickness
        yu : float
            upper surface y location of maximum thickness
        yl : float
            lower surface y location of maximum thickness

        Notes
        -----
        uses :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`

        """

        idx = np.argmax(self.yu - self.yl)
        return (self.x[idx], self.yu[idx], self.yl[idx])


    def blend(self, other, weight):
        """Blend this profile with another one with the specified weighting.

        Parameters
        ----------
        other : Profile
            another Profile to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        profile : Profile
            a blended profile

        """


        # blend coordinates
        yu = self.yu + weight*(other.yu - self.yu)
        yl = self.yl + weight*(other.yl - self.yl)

        return Profile(self.x, yu, self.x, yl)


    @property
    def tc(self):
        """thickness to chord ratio of the Profile"""
        return self.tc





if __name__ == '__main__':

    # geometry
    r_str = [1.5, 1.80135, 1.89975, 1.99815, 2.1027, 2.2011, 2.2995, 2.87145, 3.0006, 3.099, 5.60205, 6.9981, 8.33265, 10.49745, 11.75205, 13.49865, 15.84795, 18.4986, 19.95, 21.99795, 24.05205, 26.1, 28.14795, 32.25, 33.49845, 36.35205, 38.4984, 40.44795, 42.50205, 43.49835, 44.55, 46.49955, 48.65205, 52.74795, 56.16735, 58.89795, 61.62855, 63.]
    chord_str = [3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.387, 3.39, 3.741, 4.035, 4.25, 4.478, 4.557, 4.616, 4.652, 4.543, 4.458, 4.356, 4.249, 4.131, 4.007, 3.748, 3.672, 3.502, 3.373, 3.256, 3.133, 3.073, 3.01, 2.893, 2.764, 2.518, 2.313, 2.086, 1.419, 1.085]
    theta_str = [13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 12.53, 11.48, 10.63, 10.16, 9.59, 9.01, 8.4, 7.79, 6.54, 6.18, 5.36, 4.75, 4.19, 3.66, 3.4, 3.13, 2.74, 2.32, 1.53, 0.86, 0.37, 0.11, 0.0]
    le_str = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    nweb_str = [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]

    # -------- materials and composite layup  -----------------
    basepath = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', 'wese', 'aning', 'twister', 'src', 'twister', 'examples', '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.initFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(r_str)
    compSec = [0]*ncomp
    profile = [0]*ncomp

    for i in range(ncomp):

        if nweb_str[i] == 3:
            webLoc = [0.3, 0.6]  # the last "shear web" is just a flat trailing edge - negligible
        elif nweb_str[i] == 2:
            webLoc = [0.3, 0.6]
        else:
            webLoc = []

        compSec[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))
    # --------------------------------------

    # create object
    precomp = PreComp(r_str, chord_str, theta_str, profile, compSec, le_str, materials)

    # evalute section properties
    r_str, EA, EIxx, EIyy, EIxy, GJ, rhoA, rhoJ, x_ec_str, y_ec_str, x_ec_nose, y_ec_nose = precomp.sectionProperties()

    # cleanup
    precomp.cleanup()

    # plot
    r_str = np.array(r_str)
    rstar = (r_str - r_str[0])/(r_str[-1] - r_str[0])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(rstar, EIxx)
    plt.xlabel('blade fraction')
    plt.ylabel('flapwise stifness ($N m^2$)')
    plt.show()
