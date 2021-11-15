import numpy as np


class Point:
    """A class for any object in the mooring system that can be described by three translational coorindates"""

    def __init__(self, mooringSys, num, type, r, m=0, v=0, fExt=np.zeros(3), DOFs=[0, 1, 2]):
        """Initialize Point attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the point object
        num : int
            indentifier number
        type : int
            the point type: 0 free to move, 1 fixed, -1 coupled externally
        r : array
            x,y,z coorindate position vector [m].
        m : float, optional
            mass [kg]. The default is 0.
        v : float, optional
            volume [m^3]. The default is 0.
        fExt : array, optional
            applied external force vector in global orientation (not including weight/buoyancy). The default is np.zeros(3).
        DOFs: list
            list of which coordinate directions are DOFs for this point (default 0,1,2=x,y,z). E.g. set [2] for vertical motion only.

        Returns
        -------
        None.

        """

        self.sys = mooringSys  # store a reference to the overall mooring system (instance of System class)

        self.number = num
        self.type = type  # 1: fixed/attached to something, 0 free to move, or -1 coupled externally
        self.r = np.array(r, dtype=np.float_)

        self.m = np.float_(m)
        self.v = np.float_(v)
        self.fExt = fExt  # external forces plus weight/buoyancy
        self.fBot = 10.0  # this is a seabed contact force that will be added if a point is specified below the seabed
        self.zSub = 0.0  # this is the depth that the point is positioned below the seabed (since r[2] will be capped at the depth)
        self.zTol = 2.0  # depth tolerance to be used when updating the point's position relative to the seabed

        self.DOFs = DOFs
        self.nDOF = len(DOFs)

        self.attached = []  # ID numbers of any Lines attached to the Point
        self.attachedEndB = []  # specifies which end of the line is attached (1: end B, 0: end A)

        # print("Created Point "+str(self.number))

    def attachLine(self, lineID, endB):
        """Adds a Line end to the Point

        Parameters
        ----------
        lineID : int
            The identifier ID number of a line
        endB : boolean
            Determines which end of the line is attached to the point

        Returns
        -------
        None.

        """

        self.attached.append(lineID)
        self.attachedEndB.append(endB)
        # print("attached Line "+str(lineID)+" to Point "+str(self.number))

    def detachLine(self, lineID, endB):
        """Detaches a Line end from the Point

        Parameters
        ----------
        lineID : int
            The identifier ID number of a line
        endB : boolean
            Determines which end of the line is to be detached from the point

        Returns
        -------
        None.

        """

        self.attached.pop(self.attached.index(lineID))
        self.attachedEndB.pop(self.attachedEndB.index(endB))
        print("detached Line " + str(lineID) + " from Point " + str(self.number))

    def setPosition(self, r):
        """Sets the position of the Point, along with that of any dependent objects.

        Parameters
        ----------
        r : array
            x,y,z coordinate position vector of the point [m]

        Raises
        ------
        ValueError
            If the length of the input r array is not of length 3

        Returns
        -------
        None.

        """

        # update the position of the Point itself
        if len(r) == 3:  # original case, setting all three coordinates as normal, asuming x,y,z
            self.r = np.array(r)
        elif len(r) == self.nDOF:
            self.r[
                self.DOFs
            ] = r  # this does a mapping based on self.DOFs, to support points with e.g. only a z DOF or only x and z DOFs
        else:
            raise ValueError(
                f"Point setPosition method requires an argument of size 3 or nDOF, but size {len(r):d} was provided"
            )

        # update the point's depth and position based on relation to seabed
        self.zSub = np.max([-self.zTol, -self.r[2] - self.sys.depth])  # depth of submergence in seabed if > -zTol
        self.r = np.array(
            [self.r[0], self.r[1], np.max([self.r[2], -self.sys.depth])]
        )  # don't let it sink below the seabed

        # update the position of any attached Line ends
        for LineID, endB in zip(self.attached, self.attachedEndB):
            self.sys.lineList[LineID - 1].setEndPosition(self.r, endB)

        if len(self.r) < 3:
            print("Double check how this point's position vector is calculated")
            breakpoint()

    def getForces(self, lines_only=False, seabed=True, xyz=False):
        """Sums the forces on the Point, including its own plus those of any attached Lines.

        Parameters
        ----------
        lines_only : boolean, optional
            An option for calculating forces from just the mooring lines or not. The default is False.
        seabed : bool, optional
            if False, will not include the effect of the seabed pushing the point up
        xyz : boolean, optional
            if False, returns only forces corresponding to enabled DOFs. If true, returns forces in x,y,z regardless of DOFs.

        Returns
        -------
        f : array
            The force vector applied to the point in its current position [N]

        """

        f = np.zeros(3)  # create empty force vector on the point

        if lines_only == False:

            f[2] += -self.m * self.sys.g  # add weight

            if self.r[2] < -1:  # add buoyancy if fully submerged
                f[2] += self.v * self.sys.rho * self.sys.g
            elif self.r[2] < 1:  # imagine a +/-1 m band at z=0 where buoyancy tapers to zero
                f[2] += self.v * self.sys.rho * self.sys.g * (0.5 - 0.5 * self.r[2])

            f += np.array(self.fExt)  # add external forces

            # handle case of Point resting on or below the seabed, to provide a restoring force
            # add smooth transition to fz=0 at seabed (starts at zTol above seabed)
            f[2] += max(self.m - self.v * self.sys.rho, 0) * self.sys.g * (self.zSub + self.zTol) / self.zTol

        # add forces from attached lines
        for LineID, endB in zip(self.attached, self.attachedEndB):
            f += self.sys.lineList[LineID - 1].getEndForce(endB)

        if xyz:
            return f
        else:
            return f[self.DOFs]  # return only the force(s) in the enable DOFs

    def getStiffness(self, X=[], tol=0.0001, dx=0.01):
        """Gets the stiffness matrix of the point due only to mooring lines with all other objects free to equilibrate.
        NOTE: This method currently isn't set up to worry about nDOF and DOFs settings of the Point. It only works for DOFs=[0,1,2].

        Parameters
        ----------
        X1 : array
            The position vector of the Point at which the stiffness matrix is to be calculated.
        dx : float, optional
            The change in displacement to be used for calculating the change in force. The default is 0.01.

        Returns
        -------
        K : matrix
            The stiffness matrix of the point at the given position X1.

        """

        # print("Getting Point "+str(self.number)+" stiffness matrix...")

        if len(X) == 3:
            X1 = np.array(X)
        elif len(X) == 0:
            X1 = self.r
        else:
            raise ValueError("Point.getStiffness expects the optional X parameter to be size 3")

        # set this Point's type to fixed so mooring system equilibrium response to its displacements can be found
        type0 = self.type  # store original type to restore later
        self.type = 1  # set type to 1 (not free) so that it won't be adjusted when finding equilibrium

        # if this Point is attached to a Body, set that Body's type to fixed so equilibrium can be found
        for body in self.sys.bodyList:  # search through all the bodies in the mooring system
            if self.number in body.attachedP:  # find the one that this Point is attached to (if at all)
                num = body.number  # store body number to index later
                Btype0 = body.type  # store original body type to restore later
                body.type = 1  # set body type to 1 (not free) so that it won't be adjusted when finding equilibrium

        # ensure this Point is positioned at the desired linearization point
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)  # find equilibrium of mooring system given this Point in current position
        f = self.getForces(lines_only=True)  # get the net 6DOF forces/moments from any attached lines

        # Build a stiffness matrix by perturbing each DOF in turn
        K = np.zeros([3, 3])

        for i in range(len(K)):
            X2 = X1 + np.insert(
                np.zeros(2), i, dx
            )  # calculate perturbed Point position by adding dx to DOF in question
            self.setPosition(X2)  # perturb this Point's position
            self.sys.solveEquilibrium3(tol=tol)  # find equilibrium of mooring system given this Point's new position
            f_2 = self.getForces(lines_only=True)  # get the net 3DOF forces/moments from any attached lines

            K[:, i] = -(f_2 - f) / dx  # get stiffness in this DOF via finite difference and add to matrix column

        # ----------------- restore the system back to previous positions ------------------
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)  # find equilibrium of mooring system given this Point in current position
        self.type = type0  # restore the Point's type to its original value
        for body in self.sys.bodyList:
            if self.number in body.attachedP:
                num = body.number
                self.sys.bodyList[
                    num - 1
                ].type = Btype0  # restore the type of the Body that the Point is attached to back to original value

        return K

    def getStiffnessA(self, lines_only=False, xyz=False):
        """Gets analytical stiffness matrix of Point due only to mooring lines with other objects fixed.

        Returns
        -------
        K : matrix
            3x3 analytic stiffness matrix.

        """

        # print("Getting Point "+str(self.number)+" analytic stiffness matrix...")

        K = np.zeros([3, 3])  # create an empty 3x3 stiffness matrix

        # append the stiffness matrix of each line attached to the point
        for lineID, endB in zip(self.attached, self.attachedEndB):
            line = self.sys.lineList[lineID - 1]
            # KA, KB = line.getStiffnessMatrix()

            if endB == 1:  # assuming convention of end A is attached to the point, so if not,
                # KA, KB = KB, KA            # swap matrices of ends A and B
                K += line.KB
            else:
                K += line.KA

        # NOTE: can rotate the line's stiffness matrix in either Line.getStiffnessMatrix() or here in Point.getStiffnessA()

        # add seabed or hydrostatic terms if needed
        if lines_only == False:

            # if within a +/-1 m band from z=0, apply a hydrostatic stiffness based on buoyancy
            if abs(self.r[2]) < 1:
                K[2, 2] += self.v * self.sys.rho * self.sys.g * 0.5

            # if less than zTol above the seabed (could even be below the seabed), apply a stiffness (should bring wet weight to zero at seabed)
            if self.r[2] < self.zTol - self.sys.depth:
                K[2, 2] += max(self.m - self.v * self.sys.rho, 0) * self.sys.g / self.zTol

            # if on seabed, apply a large stiffness to help out system equilibrium solve (if it's transitioning off, keep it a small step to start with)
            if self.r[2] == -self.sys.depth:
                K[2, 2] += 1.0e12

        if xyz:  # if asked to output all DOFs, do it
            return K
        else:  # otherwise only return rows/columns of active DOFs
            return K[:, self.DOFs][self.DOFs, :]
