import numpy as np
from wisdem.moorpy.helpers import LineError, CatenaryError, rotationMatrix
from wisdem.moorpy.Catenary import catenary


class Line:
    """A class for any mooring line that consists of a single material"""

    def __init__(self, mooringSys, num, L, lineTypeName, nSegs=100, cb=0, isRod=0, attachments=[0, 0]):
        """Initialize Line attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the point object
        num : int
            indentifier number
        L : float
            line unstretched length [m]
        lineTypeName : string
            string identifier of LineType object that this Line is to be
        nSegs : int, optional
            number of segments to split the line into. Used in MoorPy just for plotting. The default is 100.
        cb : float, optional
            line seabed friction coefficient (will be set negative if line is fully suspended). The default is 0.
        isRod : boolean, optional
            determines whether the line is a rod or not. The default is 0.
        attachments : TYPE, optional
            ID numbers of any Points attached to the Line. The default is [0,0].

        Returns
        -------
        None.

        """

        self.sys = mooringSys  # store a reference to the overall mooring system (instance of System class)

        self.number = num
        self.isRod = isRod

        self.L = L  # line unstretched length
        self.type = lineTypeName  # string that should match a lineTypes dict entry

        self.nNodes = int(nSegs) + 1
        self.cb = float(cb)  # friction coefficient (will automatically be set negative if line is fully suspended)

        self.rA = np.zeros(3)  # end coordinates
        self.rB = np.zeros(3)
        self.fA = np.zeros(3)  # end forces
        self.fB = np.zeros(3)

        # Perhaps this could be made less intrusive by defining it using a line.addpoint() method instead, similar to point.attachline().
        self.attached = (
            attachments  # ID numbers of the Points at the Line ends [a,b] >>> NOTE: not fully supported <<<<
        )
        self.th = 0  # heading of line from end A to B
        self.HF = 0  # fairlead horizontal force saved for next solve
        self.VF = 0  # fairlead vertical force saved for next solve
        self.jacobian = []  # to be filled with the 2x2 Jacobian from catenary
        self.info = {}  # to hold all info provided by catenary

        self.qs = 1  # flag indicating quasi-static analysis (1). Set to 0 for time series data

        # print("Created Line "+str(self.number))

    def loadData(self, dirname):
        """Loads line-specific time series data from a MoorDyn output file"""

        self.qs = 0  # signals time series data

        # load time series data
        if self.isRod > 0:
            data, ch, channels, units = read_mooring_file(
                dirname, "Rod" + str(self.number) + ".out"
            )  # remember number starts on 1 rather than 0
        else:
            data, ch, channels, units = read_mooring_file(
                dirname, "Line" + str(self.number) + ".out"
            )  # remember number starts on 1 rather than 0

        # get time info
        if "Time" in ch:
            self.Tdata = data[:, ch["Time"]]
            self.dt = self.Tdata[1] - self.Tdata[0]
        else:
            raise LineError("loadData: could not find Time channel for mooring line " + str(self.number))

        nT = len(self.Tdata)  # number of time steps

        # check for position data <<<<<<

        self.xp = np.zeros([nT, self.nNodes])
        self.yp = np.zeros([nT, self.nNodes])
        self.zp = np.zeros([nT, self.nNodes])

        for i in range(self.nNodes):
            self.xp[:, i] = data[:, ch["Node" + str(i) + "px"]]
            self.yp[:, i] = data[:, ch["Node" + str(i) + "py"]]
            self.zp[:, i] = data[:, ch["Node" + str(i) + "pz"]]

        if self.isRod == 0:
            self.Te = np.zeros([nT, self.nNodes - 1])  # read in tension data if available
            if "Seg1Te" in ch:
                for i in range(self.nNodes - 1):
                    self.Te[:, i] = data[:, ch["Seg" + str(i + 1) + "Te"]]

            self.Ku = np.zeros([nT, self.nNodes])  # read in curvature data if available
            if "Node0Ku" in ch:
                for i in range(self.nNodes):
                    self.Ku[:, i] = data[:, ch["Node" + str(i) + "Ku"]]

        self.Ux = np.zeros([nT, self.nNodes])  # read in fluid velocity data if available
        self.Uy = np.zeros([nT, self.nNodes])
        self.Uz = np.zeros([nT, self.nNodes])
        if "Node0Ux" in ch:
            for i in range(self.nNodes):
                self.Ux[:, i] = data[:, ch["Node" + str(i) + "Ux"]]
                self.Uy[:, i] = data[:, ch["Node" + str(i) + "Uy"]]
                self.Uz[:, i] = data[:, ch["Node" + str(i) + "Uz"]]

        self.xpi = self.xp[0, :]
        self.ypi = self.yp[0, :]
        self.zpi = self.zp[0, :]

        # get length (constant)
        self.L = np.sqrt(
            (self.xpi[-1] - self.xpi[0]) ** 2 + (self.ypi[-1] - self.ypi[0]) ** 2 + (self.zpi[-1] - self.zpi[0]) ** 2
        )

        # check for tension data <<<<<<<

    def getTimestep(self, Time):
        """Get the time step to use for showing time series data"""

        if Time < 0:
            ts = np.int(-Time)  # negative value indicates passing a time step index
        else:  # otherwise it's a time in s, so find closest time step
            for index, item in enumerate(self.Tdata):
                # print "index is "+str(index)+" and item is "+str(item)
                ts = -1
                if item > Time:
                    ts = index
                    break
            if ts == -1:
                raise LineError("getTimestep: requested time likely out of range")

        return ts

    def getLineCoords(self, Time):  # formerly UpdateLine
        """Gets the updated line coordinates for drawing and plotting purposes."""

        # if a quasi-static analysis, just call the catenary function to return the line coordinates
        if self.qs == 1:

            depth = self.sys.depth

            dr = self.rB - self.rA
            LH = np.hypot(dr[0], dr[1])  # horizontal spacing of line ends
            LV = dr[2]  # vertical offset from end A to end B

            if np.min([self.rA[2], self.rB[2]]) > -depth:
                self.cb = -depth - np.min(
                    [self.rA[2], self.rB[2]]
                )  # if this line's lower end is off the seabed, set cb negative and to the distance off the seabed
            elif (
                self.cb < 0
            ):  # if a line end is at the seabed, but the cb is still set negative to indicate off the seabed
                self.cb = 0.0  # set to zero so that the line includes seabed interaction.

            try:
                (fAH, fAV, fBH, fBV, info) = catenary(
                    LH,
                    LV,
                    self.L,
                    self.sys.lineTypes[self.type].EA,
                    self.sys.lineTypes[self.type].w,
                    self.cb,
                    HF0=self.HF,
                    VF0=self.VF,
                    nNodes=self.nNodes,
                    plots=1,
                )
            except CatenaryError as error:
                raise LineError(self.number, error.message)

            Xs = self.rA[0] + info["X"] * dr[0] / LH
            Ys = self.rA[1] + info["X"] * dr[1] / LH
            Zs = self.rA[2] + info["Z"]

            return Xs, Ys, Zs

        # otherwise, count on read-in time-series data
        else:
            # figure out what time step to use
            ts = self.getTimestep(Time)

            # drawing rods
            if self.isRod > 0:

                k1 = (
                    np.array(
                        [
                            self.xp[ts, -1] - self.xp[ts, 0],
                            self.yp[ts, -1] - self.yp[ts, 0],
                            self.zp[ts, -1] - self.zp[ts, 0],
                        ]
                    )
                    / self.length
                )  # unit vector

                k = np.array(k1)  # make copy

                Rmat = np.array(
                    rotationMatrix(0, np.arctan2(np.hypot(k[0], k[1]), k[2]), np.arctan2(k[1], k[0]))
                )  # <<< should fix this up at some point, MattLib func may be wrong

                # make points for appropriately sized cylinder
                d = self.sys.lineTypes[self.type].d
                Xs, Ys, Zs = makeTower(self.length, np.array([d, d]))

                # translate and rotate into proper position for Rod
                coords = np.vstack([Xs, Ys, Zs])
                newcoords = np.matmul(Rmat, coords)
                Xs = newcoords[0, :] + self.xp[ts, 0]
                Ys = newcoords[1, :] + self.yp[ts, 0]
                Zs = newcoords[2, :] + self.zp[ts, 0]

                return Xs, Ys, Zs

            # drawing lines
            else:

                return self.xp[ts, :], self.yp[ts, :], self.zp[ts, :]

    def drawLine2d(self, Time, ax, color="k", Xuvec=[1, 0, 0], Yuvec=[0, 0, 1]):
        """Draw the line on 2D plot (ax must be 2D)

        Parameters
        ----------
        Time : float
            time value at which to draw the line
        ax : axis
            the axis on which the line is to be drawn
        color : string, optional
            color identifier in one letter (k=black, b=blue,...). The default is "k".
        Xuvec : list, optional
            plane at which the x-axis is desired. The default is [1,0,0].
        Yuvec : lsit, optional
            plane at which the y-axis is desired. The default is [0,0,1].

        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted

        """

        linebit = []  # make empty list to hold plotted lines, however many there are

        if self.isRod > 0:

            Xs, Ys, Zs = self.getLineCoords(Time)

            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs * Xuvec[0] + Ys * Xuvec[1] + Zs * Xuvec[2]
            Ys2d = Xs * Yuvec[0] + Ys * Yuvec[1] + Zs * Yuvec[2]

            for i in range(int(len(Xs) / 2 - 1)):
                linebit.append(
                    ax.plot(Xs2d[2 * i : 2 * i + 2], Ys2d[2 * i : 2 * i + 2], lw=0.5, color=color)
                )  # side edges
                linebit.append(
                    ax.plot(Xs2d[[2 * i, 2 * i + 2]], Ys2d[[2 * i, 2 * i + 2]], lw=0.5, color=color)
                )  # end A edges
                linebit.append(
                    ax.plot(Xs2d[[2 * i + 1, 2 * i + 3]], Ys2d[[2 * i + 1, 2 * i + 3]], lw=0.5, color=color)
                )  # end B edges

        # drawing lines...
        else:

            Xs, Ys, Zs = self.getLineCoords(Time)

            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs * Xuvec[0] + Ys * Xuvec[1] + Zs * Xuvec[2]
            Ys2d = Xs * Yuvec[0] + Ys * Yuvec[1] + Zs * Yuvec[2]

            linebit.append(ax.plot(Xs2d, Ys2d, lw=1, color=color))

        self.linebit = linebit  # can we store this internally?

        return linebit

    def drawLine(self, Time, ax, color="k"):
        """Draw the line in 3D

        Parameters
        ----------
        Time : float
            time value at which to draw the line
        ax : axis
            the axis on which the line is to be drawn
        color : string, optional
            color identifier in one letter (k=black, b=blue,...). The default is "k".
        Xuvec : list, optional
            plane at which the x-axis is desired. The default is [1,0,0].
        Yuvec : lsit, optional
            plane at which the y-axis is desired. The default is [0,0,1].

        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted

        """

        linebit = []  # make empty list to hold plotted lines, however many there are

        if self.isRod > 0:

            Xs, Ys, Zs = self.getLineCoords(Time)

            for i in range(int(len(Xs) / 2 - 1)):
                linebit.append(
                    ax.plot(Xs[2 * i : 2 * i + 2], Ys[2 * i : 2 * i + 2], Zs[2 * i : 2 * i + 2], color=color)
                )  # side edges
                linebit.append(
                    ax.plot(Xs[[2 * i, 2 * i + 2]], Ys[[2 * i, 2 * i + 2]], Zs[[2 * i, 2 * i + 2]], color=color)
                )  # end A edges
                linebit.append(
                    ax.plot(
                        Xs[[2 * i + 1, 2 * i + 3]], Ys[[2 * i + 1, 2 * i + 3]], Zs[[2 * i + 1, 2 * i + 3]], color=color
                    )
                )  # end B edges

        # drawing lines...
        else:

            Xs, Ys, Zs = self.getLineCoords(Time)

            linebit.append(ax.plot(Xs, Ys, Zs, color=color))

            # drawing water velocity vectors (not for Rods for now) <<< should handle this better (like in getLineCoords) <<<
            if self.qs == 0:
                ts = self.getTimestep(Time)
                Ux = self.Ux[ts, :]
                Uy = self.Uy[ts, :]
                Uz = self.Uz[ts, :]
                self.Ubits = ax.quiver(Xs, Ys, Zs, Ux, Uy, Uz)  # make quiver plot and save handle to line object

        self.linebit = linebit  # can we store this internally?

        self.X = np.array([Xs, Ys, Zs])

        return linebit

    def redrawLine(self, Time):  # , linebit):
        """Update 3D line drawing based on instantaneous position"""

        linebit = self.linebit

        if self.isRod > 0:

            Xs, Ys, Zs = self.getLineCoords(Time)

            for i in range(int(len(Xs) / 2 - 1)):

                linebit[3 * i][0].set_data(
                    Xs[2 * i : 2 * i + 2], Ys[2 * i : 2 * i + 2]
                )  # side edges (x and y coordinates)
                linebit[3 * i][0].set_3d_properties(Zs[2 * i : 2 * i + 2])  #            (z coordinates)
                linebit[3 * i + 1][0].set_data(Xs[[2 * i, 2 * i + 2]], Ys[[2 * i, 2 * i + 2]])  # end A edges
                linebit[3 * i + 1][0].set_3d_properties(Zs[[2 * i, 2 * i + 2]])
                linebit[3 * i + 2][0].set_data(Xs[[2 * i + 1, 2 * i + 3]], Ys[[2 * i + 1, 2 * i + 3]])  # end B edges
                linebit[3 * i + 2][0].set_3d_properties(Zs[[2 * i + 1, 2 * i + 3]])

        # drawing lines...
        else:

            Xs, Ys, Zs = self.getLineCoords(Time)
            linebit[0][0].set_data(Xs, Ys)  # (x and y coordinates)
            linebit[0][0].set_3d_properties(Zs)  # (z coordinates)

            # drawing water velocity vectors (not for Rods for now)
            if self.qs == 0:
                ts = self.getTimestep(Time)
                Ux = self.Ux[ts, :]
                Uy = self.Uy[ts, :]
                Uz = self.Uz[ts, :]
                segments = quiver_data_to_segments(Xs, Ys, Zs, Ux, Uy, Uz, scale=2)
                self.Ubits.set_segments(segments)

        return linebit

    def setEndPosition(self, r, endB):
        """Sets the end position of the line based on the input endB value.

        Parameters
        ----------
        r : array
            x,y,z coorindate position vector of the line end [m].
        endB : boolean
            An indicator of whether the r array is at the end or beginning of the line

        Raises
        ------
        LineError
            If the given endB value is not a 1 or 0

        Returns
        -------
        None.

        """

        if endB == 1:
            self.rB = np.array(r, dtype=np.float_)
        elif endB == 0:
            self.rA = np.array(r, dtype=np.float_)
        else:
            raise LineError("setEndPosition: endB value has to be either 1 or 0")

    def staticSolve(self, reset=False):
        """Solves static equilibrium of line. Sets the end forces of the line based on the end points' positions.

        Parameters
        ----------
        reset : boolean, optional
            Determines if the previous fairlead force values will be used for the catenary iteration. The default is False.

        Raises
        ------
        LineError
            If the horizontal force at the fairlead (HF) is less than 0

        Returns
        -------
        None.

        """

        depth = self.sys.depth

        dr = self.rB - self.rA
        LH = np.hypot(dr[0], dr[1])  # horizontal spacing of line ends
        LV = dr[2]  # vertical offset from end A to end B

        if self.rA[2] < -depth:
            raise LineError("Line {} end A is lower than the seabed.".format(self.number))
        elif self.rB[2] < -depth:
            raise LineError("Line {} end B is lower than the seabed.".format(self.number))
        elif np.min([self.rA[2], self.rB[2]]) > -depth:
            self.cb = -depth - np.min(
                [self.rA[2], self.rB[2]]
            )  # if this line's lower end is off the seabed, set cb negative and to the distance off the seabed
        elif self.cb < 0:  # if a line end is at the seabed, but the cb is still set negative to indicate off the seabed
            self.cb = 0.0  # set to zero so that the line includes seabed interaction.

        if (
            self.HF < 0
        ):  # or self.VF < 0:  <<<<<<<<<<< it shouldn't matter if VF is negative - this could happen for buoyant lines, etc.
            raise LineError("Line HF cannot be negative")  # this could be a ValueError too...

        if reset == True:  # Indicates not to use previous fairlead force values to start catenary
            self.HF = 0  # iteration with, and insteady use the default values.

        try:
            (fAH, fAV, fBH, fBV, info) = catenary(
                LH,
                LV,
                self.L,
                self.sys.lineTypes[self.type].EA,
                self.sys.lineTypes[self.type].w,
                CB=self.cb,
                HF0=self.HF,
                VF0=self.VF,
            )  # call line model
        except CatenaryError as error:
            raise LineError(self.number, error.message)

        self.th = np.arctan2(dr[1], dr[0])  # probably a more efficient way to handle this <<<
        self.HF = info["HF"]
        self.VF = info["VF"]
        self.jacobian = info["jacobian"]
        self.LBot = info["LBot"]
        self.info = info

        self.fA[0] = fAH * dr[0] / LH
        self.fA[1] = fAH * dr[1] / LH
        self.fA[2] = fAV
        self.fB[0] = fBH * dr[0] / LH
        self.fB[1] = fBH * dr[1] / LH
        self.fB[2] = fBV

    def getEndForce(self, endB):
        """Returns the force of the line at the specified end based on the endB value

        Parameters
        ----------
        endB : boolean
            An indicator of which end of the line is the force wanted

        Raises
        ------
        LineError
            If the given endB value is not a 1 or 0

        Returns
        -------
        fA or fB: array
            The force vector at the end of the line

        """

        if endB == 1:
            return self.fB
        elif endB == 0:
            return self.fA
        else:
            raise LineError("getEndForce: endB value has to be either 1 or 0")

    def getStiffnessMatrix(self):
        """Returns the stiffness matrix of a line derived from analytic terms in the jacobian of catenary

        Raises
        ------
        LineError
            If a singluar matrix error occurs while taking the inverse of the Line's Jacobian matrix.

        Returns
        -------
        K2_rot : matrix
            the analytic stiffness matrix of the line in the rotated frame.

        """

        # take the inverse of the Jacobian to get the starting analytic stiffness matrix
        try:
            K = np.linalg.inv(self.jacobian)
        except:
            raise LineError(
                self.number, f"Check Line Length ({self.L}), it might be too long, or check catenary ProfileType"
            )

        # solve for required variables to set up the perpendicular stiffness. Keep it horizontal
        L_xy = np.linalg.norm(self.rB[:2] - self.rA[:2])
        T_xy = np.linalg.norm(self.fB[:2])
        Kt = T_xy / L_xy

        # initialize the line's analytic stiffness matrix in the "in-line" plane
        K2 = np.array([[K[0, 0], 0, K[0, 1]], [0, Kt, 0], [K[1, 0], 0, K[1, 1]]])

        # create the rotation matrix based on the heading angle that the line is from the horizontal
        R = rotationMatrix(0, 0, self.th)

        # rotate the matrix to be about the global frame [K'] = [R][K][R]^T
        K2_rot = np.matmul(np.matmul(R, K2), R.T)

        # need to make sign changes if the end fairlead (B) of the line is lower than the starting point (A)
        if self.rB[2] < self.rA[2]:
            K2_rot[2, 0] *= -1
            K2_rot[0, 2] *= -1
            K2_rot[2, 1] *= -1
            K2_rot[1, 2] *= -1

        return K2_rot
