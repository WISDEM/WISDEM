import numpy as np
from wisdem.moorpy.helpers import (
    getH,
    printVec,
    rotatePosition,
    rotationMatrix,
    transformPosition,
    translateForce3to6DOF,
)


class Body:
    """A class for any object in the mooring system that will have its own reference frame"""

    def __init__(self, mooringSys, num, type, r6, m=0, v=0, rCG=np.zeros(3), AWP=0, rM=np.zeros(3), f6Ext=np.zeros(6)):
        """Initialize Body attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the body object
        num : int
            indentifier number
        type : int
            the body type: 0 free to move, 1 fixed, -1 coupled externally
        r6 : array
            6DOF position and orientation vector [m, rad]
        m : float, optional
            mass, centered at CG [kg]. The default is 0.
        v : float, optional
            volume, centered at reference point [m^3]. The default is 0.
        rCG : array, optional
            center of gravity position in body reference frame [m]. The default is np.zeros(3).
        AWP : float, optional
            waterplane area - used for hydrostatic heave stiffness if nonzero [m^2]. The default is 0.
        rM : float or array, optional
            coorindates or height of metacenter relative to body reference frame [m]. The default is np.zeros(3).
        f6Ext : array, optional
            applied external forces and moments vector in global orientation (not including weight/buoyancy) [N]. The default is np.zeros(6).
        attachedP: list, int
            list of ID numbers of any Points attached to the Body
        rPointRel: list, float
            list of coordinates of each attached Point relative to the Body reference frame [m]

        Returns
        -------
        None.

        """

        self.sys = mooringSys  # store a reference to the overall mooring system (instance of System class)

        self.number = num
        self.type = type  # 0 free to move, or -1 coupled externally
        self.r6 = np.array(r6, dtype=np.float_)  # 6DOF position and orientation vector [m, rad]
        self.m = m  # mass, centered at CG [kg]
        self.v = v  # volume, assumed centered at reference point [m^3]
        self.rCG = np.array(rCG, dtype=np.float_)  # center of gravity position in body reference frame [m]
        self.AWP = AWP  # waterplane area - used for hydrostatic heave stiffness if nonzero [m^2]
        if np.isscalar(rM):
            self.rM = np.array(
                [0, 0, rM], dtype=np.float_
            )  # coordinates of body metacenter relative to body reference frame [m]
        else:
            self.rM = np.array(rM, dtype=np.float_)

        self.f6Ext = np.array(
            f6Ext, dtype=np.float_
        )  # for adding external forces and moments in global orientation (not including weight/buoyancy)

        self.attachedP = []  # ID numbers of any Points attached to the Body
        self.rPointRel = []  # coordinates of each attached Point relative to the Body reference frame

        self.attachedR = []  # ID numbers of any Rods attached to the Body (not yet implemented)

        self.R = np.eye(3)  # body orientation rotation matrix
        # print("Created Body "+str(self.number))

    def attachPoint(self, pointID, rAttach):
        """Adds a Point to the Body, at the specified relative position on the body.

        Parameters
        ----------
        pointID : int
            The identifier ID number of a point
        rAttach : array
            The position of the point relative to the body's frame [m]

        Returns
        -------
        None.

        """

        self.attachedP.append(pointID)
        self.rPointRel.append(np.array(rAttach))

        # print("attached Point "+str(pointID)+" to Body "+str(self.number))

    def setPosition(self, r6):
        """Sets the position of the Body, along with that of any dependent objects.

        Parameters
        ----------
        r6 : array
            6DOF position and orientation vector of the body [m, rad]

        Raises
        ------
        ValueError
            If the length of the input r6 array is not of length 6

        Returns
        -------
        None.

        """

        if len(r6) == 6:
            self.r6 = np.array(r6, dtype=np.float_)  # update the position of the Body itself
        else:
            raise ValueError(
                f"Body setPosition method requires an argument of size 6, but size {len(r6):d} was provided"
            )

        self.R = rotationMatrix(self.r6[3], self.r6[4], self.r6[5])  # update body rotation matrix

        # update the position of any attached Points
        for PointID, rPointRel in zip(self.attachedP, self.rPointRel):
            rPoint = np.matmul(self.R, rPointRel) + self.r6[:3]  # rPoint = transformPosition(rPointRel, r6)
            self.sys.pointList[PointID - 1].setPosition(rPoint)

        if self.sys.display > 3:
            printVec(rPoint)
            breakpoint()

    def getForces(self, lines_only=False):
        """Sums the forces and moments on the Body, including its own plus those from any attached objects.

        Parameters
        ----------
        lines_only : boolean, optional
            An option for calculating forces from just the mooring lines or not. The default is False.

        Returns
        -------
        f6 : array
            The 6DOF forces and moments applied to the body in its current position [N, Nm]

        """

        f6 = np.zeros(6)

        # TODO: could save time in below by storing the body's rotation matrix when it's position is set rather than
        #       recalculating it in each of the following function calls.

        if lines_only == False:

            # add weight, which may result in moments as well as a force
            rCG_rotated = rotatePosition(
                self.rCG, self.r6[3:]
            )  # relative position of CG about body ref point in unrotated reference frame
            f6 += translateForce3to6DOF(
                rCG_rotated, np.array([0, 0, -self.m * self.sys.g])
            )  # add to net forces/moments

            # add buoyancy force and moments if applicable (this can include hydrostatic restoring moments
            # if rM is considered the metacenter location rather than the center of buoyancy)
            rM_rotated = rotatePosition(
                self.rM, self.r6[3:]
            )  # relative position of metacenter about body ref point in unrotated reference frame
            f6 += translateForce3to6DOF(
                rM_rotated, np.array([0, 0, self.sys.rho * self.sys.g * self.v])
            )  # add to net forces/moments

            # add hydrostatic heave stiffness (if AWP is nonzero)
            f6[2] -= self.sys.rho * self.sys.g * self.AWP * self.r6[2]

            # add any externally applied forces/moments (in global orientation)
            f6 += self.f6Ext

        # add forces from any attached Points (and their attached lines)
        for PointID, rPointRel in zip(self.attachedP, self.rPointRel):

            fPoint = self.sys.pointList[PointID - 1].getForces(lines_only=lines_only)  # get net force on attached Point
            rPoint_rotated = rotatePosition(
                rPointRel, self.r6[3:]
            )  # relative position of Point about body ref point in unrotated reference frame
            f6 += translateForce3to6DOF(
                rPoint_rotated, fPoint
            )  # add net force and moment resulting from its position to the Body

        # All forces and moments on the body should now be summed, and are in global/unrotated orientations.

        # For application to the body DOFs, convert the moments to be about the body's local/rotated x/y/z axes <<< do we want this in all cases?
        rotMat = rotationMatrix(*self.r6[3:])  # get rotation matrix for body
        moment_about_body_ref = np.matmul(
            rotMat.T, f6[3:]
        )  # transform moments so that they are about the body's local/rotated axes
        f6[3:] = moment_about_body_ref  # use these moments

        return f6

    def getStiffness(self, X=[], tol=0.0001, dx=0.1):
        """Gets the stiffness matrix of a Body due only to mooring lines with all other objects free to equilibriate.
        The rotational indicies of the stiffness matrix correspond to the local/rotated axes of the body rather than
        the global x/y/z directions.

        Parameters
        ----------
        X1 : array
            The position vector (6DOF) of the main axes of the Body at which the stiffness matrix is to be calculated.
        dx : float, optional
            The change in displacement to be used for calculating the change in force. The default is 0.01.

        Returns
        -------
        K : matrix
            The stiffness matrix of the body at the given position X1.

        """

        # print("Getting Body "+str(self.number)+" stiffness matrix...")

        if len(X) == 6:
            X1 = np.array(X)
        elif len(X) == 0:
            X1 = self.r6
        else:
            raise ValueError("Body.getStiffness expects the optional X parameter to be size 6")

        # set this Body's type to fixed so mooring system equilibrium response to its displacements can be found
        type0 = self.type  # store original type to restore later
        self.type = 1  # set type to 1 (not free) so that it won't be adjusted when finding equilibrium

        # ensure this Body is positioned at the desired linearization point
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)  # find equilibrium of mooring system given this Body in current position
        f6 = self.getForces(lines_only=True)  # get the net 6DOF forces/moments from any attached lines

        # Build a stiffness matrix by perturbing each DOF in turn
        K = np.zeros([6, 6])

        for i in range(len(K)):
            X2 = X1 + np.insert(np.zeros(5), i, dx)  # calculate perturbed Body position by adding dx to DOF in question
            self.setPosition(X2)  # perturb this Body's position
            self.sys.solveEquilibrium3(tol=tol)  # find equilibrium of mooring system given this Body's new position
            f6_2 = self.getForces(lines_only=True)  # get the net 6DOF forces/moments from any attached lines

            K[:, i] = -(f6_2 - f6) / dx  # get stiffness in this DOF via finite difference and add to matrix column

        # ----------------- restore the system back to previous positions ------------------
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)  # find equilibrium of mooring system given this Body in current position
        self.type = type0  # restore the Body's type to its original value

        return K

    def getStiffnessA(self, lines_only=False):
        """Gets the analytical stiffness matrix of the Body with other objects fixed.

        Returns
        -------
        K : matrix
            6x6 analytic stiffness matrix.

        """

        # print("Getting Body "+str(self.number)+" stiffness matrix...")

        K = np.zeros([6, 6])

        for PointID, rPointRel in zip(self.attachedP, self.rPointRel):

            r = rotatePosition(
                rPointRel, self.r6[3:]
            )  # relative position of Point about body ref point in unrotated reference frame
            f3 = self.sys.pointList[
                PointID - 1
            ].getForces()  # total force on point (for additional rotational stiffness term due to change in moment arm)
            K3 = self.sys.pointList[PointID - 1].getStiffnessA()  # local 3D stiffness matrix of the point

            # following are from functions translateMatrix3to6
            H = getH(r)
            K[:3, :3] += K3
            K[:3, 3:] += np.matmul(
                K3, H
            )  # only add up one off-diagonal sub-matrix for now, then we'll mirror at the end
            K[3:, 3:] += np.matmul(np.matmul(H, K3), H.T) + np.matmul(getH(f3), H.T)
            # K[3:,3:] += np.matmul(np.matmul(H, K3), H.T) - np.matmul( getH(f3), H)  # <<< should be the same

        K[3:, :3] = K[:3, 3:].T  # copy over other off-diagonal sub-matrix

        if lines_only == False:

            # rotational stiffness effect of weight
            rCG_rotated = rotatePosition(
                self.rCG, self.r6[3:]
            )  # relative position of CG about body ref point in unrotated reference frame
            Kw = -np.matmul(getH([0, 0, -self.m * self.sys.g]), getH(rCG_rotated))

            # rotational stiffness effect of buoyancy at metacenter
            rM_rotated = rotatePosition(
                self.rM, self.r6[3:]
            )  # relative position of metacenter about body ref point in unrotated reference frame
            Kb = -np.matmul(getH([0, 0, self.sys.rho * self.sys.g * self.v]), getH(rM_rotated))

            # hydrostatic heave stiffness (if AWP is nonzero)
            Kwp = self.sys.rho * self.sys.g * self.AWP

            K[3:, 3:] += Kw + Kb
            K[2, 2] += Kwp

        return K

    def draw(self, ax):
        """Draws the reference axis of the body

        Parameters
        ----------
        ax : axes
            matplotlib.pyplot axes to be used for drawing and plotting.

        Returns
        -------
        linebit : list
            a list to hold plotted lines of the body's frame axes.

        """

        linebit = []  # make empty list to hold plotted lines, however many there are

        rx = transformPosition(np.array([5, 0, 0]), self.r6)
        ry = transformPosition(np.array([0, 5, 0]), self.r6)
        rz = transformPosition(np.array([0, 0, 5]), self.r6)

        linebit.append(ax.plot([self.r6[0], rx[0]], [self.r6[1], rx[1]], [self.r6[2], rx[2]], color="r"))
        linebit.append(ax.plot([self.r6[0], ry[0]], [self.r6[1], ry[1]], [self.r6[2], ry[2]], color="g"))
        linebit.append(ax.plot([self.r6[0], rz[0]], [self.r6[1], rz[1]], [self.r6[2], rz[2]], color="b"))

        self.linebit = linebit

        return linebit

    def redraw(self):
        """Redraws the reference axis of the body

        Returns
        -------
        linebit : list
            a list to hold redrawn lines of the body's frame axes.

        """

        linebit = self.linebit

        rx = transformPosition(np.array([5, 0, 0]), self.r6)
        ry = transformPosition(np.array([0, 5, 0]), self.r6)
        rz = transformPosition(np.array([0, 0, 5]), self.r6)

        linebit[0][0].set_data_3d([self.r6[0], rx[0]], [self.r6[1], rx[1]], [self.r6[2], rx[2]])
        linebit[1][0].set_data_3d([self.r6[0], ry[0]], [self.r6[1], ry[1]], [self.r6[2], ry[2]])
        linebit[2][0].set_data_3d([self.r6[0], rz[0]], [self.r6[1], rz[1]], [self.r6[2], rz[2]])
        """
        linebit[0][0].set_data([self.r6[0], rx[0]], [self.r6[1], rx[1]])
        linebit[0][0].set_3d_properties([self.r6[2], rx[2]])
        linebit[1][0].set_data([self.r6[0], ry[0]], [self.r6[1], ry[1]])
        linebit[1][0].set_3d_properties([self.r6[2], ry[2]])
        linebit[2][0].set_data([self.r6[0], rz[0]], [self.r6[1], rz[1]])
        linebit[2][0].set_3d_properties([self.r6[2], rz[2]])
        """
        return linebit


#
