import os
import copy
import numpy as np


def skipLines(f, n):
    for i in range(n):
        f.readline()

class CompositeSection:
    """A CompositeSection defines the layup of the entire
    airfoil cross-section

    """

    def __init__(self, loc, n_plies, t, theta, mat_idx, materials):
        """Constructor

        Parameters
        ----------


        """

        self.loc = np.array(loc)  # np.array([0.0, 0.15, 0.50, 1.00])

        # should be list of numpy arrays
        self.n_plies = n_plies  # [ [1, 1, 33],  [1, 1, 17, 38, 0, 37, 16], [1, 1, 17, 0, 16] ]
        self.t = t  # [ [0.000381, 0.00051, 0.00053], [0.000381, 0.00051, 0.00053, 0.00053, 0.003125, 0.00053, 0.00053], [0.000381, 0.00051, 0.00053, 0.003125, 0.00053] ]
        self.theta = theta  # [ [0, 0, 20], [0, 0, 20, 30, 0, 30, 20], [0, 0, 20, 0, 0] ]
        self.mat_idx = mat_idx  # [ [3, 4, 2], [3, 4, 2, 1, 5, 1, 2], [3, 4, 2, 5, 2] ]

        self.materials = materials

    def mycopy(self):
        return CompositeSection(
            copy.deepcopy(self.loc),
            copy.deepcopy(self.n_plies),
            copy.deepcopy(self.t),
            copy.deepcopy(self.theta),
            copy.deepcopy(self.mat_idx),
            self.materials,
        )  # TODO: copy materials (for now it never changes so I'm not looking at it)

    @classmethod
    def initFromPreCompLayupFile(cls, fname, locW, materials, readLocW=False):
        """Construct CompositeSection object from a PreComp input file

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
        readLocW : optionally read web location from main input file rather than
            have the user provide it

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
        locU = [float(x) for x in f.readline().split()]

        n_pliesU, tU, thetaU, mat_idxU = CompositeSection.__readSectorsFromFile(f, n_sector)
        upper = cls(locU, n_pliesU, tU, thetaU, mat_idxU, materials)

        skipLines(f, 3)

        # number of sectors
        n_sector = int(f.readline().split()[0])

        skipLines(f, 2)

        locL = [float(x) for x in f.readline().split()]

        n_pliesL, tL, thetaL, mat_idxL = CompositeSection.__readSectorsFromFile(f, n_sector)
        lower = cls(locL, n_pliesL, tL, thetaL, mat_idxL, materials)

        skipLines(f, 4)

        if readLocW:
            locW = CompositeSection.__readWebLocFromFile(fname)
        n_sector = len(locW)

        n_pliesW, tW, thetaW, mat_idxW = CompositeSection.__readSectorsFromFile(f, n_sector)
        webs = cls(locW, n_pliesW, tW, thetaW, mat_idxW, materials)

        f.close()

        return upper, lower, webs

    @staticmethod
    def __readSectorsFromFile(f, n_sector):
        """private method"""

        n_plies = [0] * n_sector
        t = [0] * n_sector
        theta = [0] * n_sector
        mat_idx = [0] * n_sector

        for i in range(n_sector):
            skipLines(f, 2)

            line = f.readline()
            if line == "":
                return []  # no webs
            n_lamina = int(line.split()[1])

            skipLines(f, 4)

            n_plies_S = np.zeros(n_lamina)
            t_S = np.zeros(n_lamina)
            theta_S = np.zeros(n_lamina)
            mat_idx_S = np.zeros(n_lamina)

            for j in range(n_lamina):
                array = f.readline().split()
                n_plies_S[j] = int(array[1])
                t_S[j] = float(array[2])
                theta_S[j] = float(array[3])
                mat_idx_S[j] = int(array[4]) - 1

            n_plies[i] = n_plies_S
            t[i] = t_S
            theta[i] = theta_S
            mat_idx[i] = mat_idx_S

        return n_plies, t, theta, mat_idx

    @staticmethod
    def __readWebLocFromFile(fname):
        # Get web locations from main input file
        f_main = os.path.join(os.path.split(fname)[0], os.path.split(fname)[1].replace("layup", "input"))

        # Error handling for different file extensions
        if not os.path.isfile(f_main):
            extensions = ["dat", "inp", "pci"]
            for ext in extensions:
                f_main = f_main[:-3] + ext
                if os.path.isfile(f_main):
                    break

        fid = open(f_main)

        var = fid.readline().split()[0]
        while var != "Web_num":
            text = fid.readline().split()
            if len(text) > 0:
                var = text[0]
            else:
                var = None

        web_loc = []
        line = fid.readline().split()
        while line:
            web_loc.append(float(line[1]))
            line = fid.readline().split()

        return web_loc

    def compositeMatrices(self, sector):
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

        t = self.t[sector]
        n_plies = self.n_plies[sector]
        mat_idx = self.mat_idx[sector]
        theta = self.theta[sector]

        mat_idx = mat_idx.astype(int)  # convert to integers if actually stored as floats

        n = len(theta)

        # heights (z - absolute, h - relative to mid-plane)
        z = np.zeros(n + 1)
        for i in range(n):
            z[i + 1] = z[i] + t[i] * n_plies[i]

        z_mid = (z[-1] - z[0]) / 2.0
        h = z - z_mid

        # ABD matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for i in range(n):
            Qbar = self.__Qbar(self.materials[mat_idx[i]], theta[i])
            A += Qbar * (h[i + 1] - h[i])
            B += 0.5 * Qbar * (h[i + 1] ** 2 - h[i] ** 2)
            D += 1.0 / 3.0 * Qbar * (h[i + 1] ** 3 - h[i] ** 3)

        totalHeight = z[-1] - z[0]

        return A, B, D, totalHeight

    def effectiveEAxial(self, sector):
        """Estimates the effective axial modulus of elasticity for the laminate

        Returns
        -------
        E : float (N/m^2)
            effective axial modulus of elasticity

        Notes
        -----
        see user guide for a :ref:`derivation <ABBD>`

        """

        A, B, D, totalHeight = self.compositeMatrices(sector)

        # S = [A B; B D]

        S = np.vstack((np.hstack((A, B)), np.hstack((B, D))))

        # E_eff_x = N_x/h/eps_xx and eps_xx = S^{-1}(0,0)*N_x (approximately)
        detS = np.linalg.det(S)
        Eaxial = detS / np.linalg.det(S[1:, 1:]) / totalHeight

        return Eaxial

    def __Qbar(self, material, theta):
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

        E11 = material.E1
        E22 = material.E2
        nu12 = material.nu12
        nu21 = nu12 * E22 / E11
        G12 = material.G12
        denom = 1 - nu12 * nu21

        c = np.cos(theta * np.pi / 180.0)
        s = np.sin(theta * np.pi / 180.0)
        c2 = c * c
        s2 = s * s
        cs = c * s

        Q = np.array([[E11 / denom, nu12 * E22 / denom, 0], [nu12 * E22 / denom, E22 / denom, 0], [0, 0, G12]])
        T12 = np.array([[c2, s2, cs], [s2, c2, -cs], [-cs, cs, 0.5 * (c2 - s2)]])
        Tinv = np.array([[c2, s2, -2 * cs], [s2, c2, 2 * cs], [cs, -cs, c2 - s2]])

        return Tinv @ Q @ T12

    def _preCompFormat(self):
        n = len(self.theta)
        n_lamina = np.zeros(n)

        if n == 0:
            return self.loc, n_lamina, self.n_plies, self.t, self.theta, self.mat_idx

        for i in range(n):
            n_lamina[i] = len(self.theta[i])

        mat = np.concatenate(self.mat_idx)
        for i in range(len(mat)):
            mat[i] += 1  # 1-based indexing in Fortran

        return self.loc, n_lamina, np.concatenate(self.n_plies), np.concatenate(self.t), np.concatenate(self.theta), mat


