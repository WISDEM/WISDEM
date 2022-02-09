import numpy as np
from matplotlib import cm
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
        self.KA = []  # to be filled with the 2x2 end stiffness matrix from catenary
        self.KB = []  # to be filled with the 2x2 end stiffness matrix from catenary
        self.info = {}  # to hold all info provided by catenary

        self.qs = 1  # flag indicating quasi-static analysis (1). Set to 0 for time series data

        # print("Created Line "+str(self.number))

    def loadData(self, dirname, rootname):
        """Loads line-specific time series data from a MoorDyn output file"""

        self.qs = 0  # signals time series data

        # load time series data
        if self.isRod > 0:
            data, ch, channels, units = self.read_mooring_file(
                dirname + rootname + ".MD.", "Rod" + str(self.number) + ".out"
            )  # remember number starts on 1 rather than 0
        else:
            data, ch, channels, units = self.read_mooring_file(
                dirname + rootname + ".MD.", "Line" + str(self.number) + ".out"
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

        # calculate the dynamic LBot !!!!!!! doesn't work for sloped bathymetry yet !!!!!!!!!!
        for i in range(len(self.zp[0])):
            if np.max(self.zp[:, i]) > self.zp[0, 0]:
                inode = i
                break
            else:
                inode = i
        self.LBotDyn = (inode - 1) * self.L / (self.nNodes - 1)

        # get length (constant)
        self.L = np.sqrt(
            (self.xpi[-1] - self.xpi[0]) ** 2 + (self.ypi[-1] - self.ypi[0]) ** 2 + (self.zpi[-1] - self.zpi[0]) ** 2
        )
        # ^^^^^^^ why are we changing the self.L value to not the unstretched length specified in MoorDyn?
        # moved this below the dynamic LBot calculation because I wanted to use the original self.L

        # check for tension data <<<<<<<

    def read_mooring_file(self, dirName, fileName):

        # load data from time series for single mooring line

        print("attempting to load " + dirName + fileName)

        f = open(dirName + fileName, "r")

        channels = []
        units = []
        data = []
        i = 0

        for line in f:  # loop through lines in file

            if i == 0:
                for entry in line.split():  # loop over the elemets, split by whitespace
                    channels.append(entry)  # append to the last element of the list

            elif i == 1:
                for entry in line.split():  # loop over the elemets, split by whitespace
                    units.append(entry)  # append to the last element of the list

            elif len(line.split()) > 0:
                data.append([])  # add a new sublist to the data matrix
                import re

                r = re.compile(
                    r"(?<=\d)\-(?=\d)"
                )  # catch any instances where a large negative exponent has been written with the "E"
                line2 = r.sub("E-", line)  # and add in the E

                for entry in line2.split():  # loop over the elemets, split by whitespace
                    data[-1].append(entry)  # append to the last element of the list

            else:
                break

            i += 1

        f.close()  # close data file

        # use a dictionary for convenient access of channel columns (eg. data[t][ch['PtfmPitch'] )
        ch = dict(zip(channels, range(len(channels))))

        data2 = np.array(data)

        data3 = data2.astype(float)

        return data3, ch, channels, units

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

    def getLineCoords(self, Time, n=0):  # formerly UpdateLine
        """Gets the updated line coordinates for drawing and plotting purposes."""

        if n == 0:
            n = self.nNodes

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
                    nNodes=n,
                    plots=1,
                )
            except CatenaryError as error:
                raise LineError(self.number, error.message)

            Xs = self.rA[0] + info["X"] * dr[0] / LH
            Ys = self.rA[1] + info["X"] * dr[1] / LH
            Zs = self.rA[2] + info["Z"]
            Ts = info["Te"]
            return Xs, Ys, Zs, Ts

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
                Xs, Ys, Zs = makeTower(
                    self.length, np.array([d, d])
                )  # add in makeTower method once you start using Rods

                # translate and rotate into proper position for Rod
                coords = np.vstack([Xs, Ys, Zs])
                newcoords = np.matmul(Rmat, coords)
                Xs = newcoords[0, :] + self.xp[ts, 0]
                Ys = newcoords[1, :] + self.yp[ts, 0]
                Zs = newcoords[2, :] + self.zp[ts, 0]

                return Xs, Ys, Zs, None

            # drawing lines
            else:

                return self.xp[ts, :], self.yp[ts, :], self.zp[ts, :], self.Tdata[ts:]

    def getCoordinate(self, s, n=100):
        """Returns position and tension at a specific point along the line's unstretched length"""

        dr = self.rB - self.rA
        LH = np.hypot(dr[0], dr[1])

        Ss = np.linspace(0, self.L, n)
        Xs, Ys, Zs, Ts = self.getLineCoords(0.0, n=n)

        X = np.interp(s, Ss, Xs) * dr[0] / LH
        Y = np.interp(s, Ss, Ys) * dr[1] / LH
        Z = np.interp(s, Ss, Zs)
        T = np.interp(s, Ss, Ts)

        return X, Y, Z, T

    def drawLine2d(self, Time, ax, color="k", Xuvec=[1, 0, 0], Yuvec=[0, 0, 1], colortension=False, cmap="rainbow"):
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
        Yuvec : list, optional
            plane at which the y-axis is desired. The default is [0,0,1].
        colortension : bool, optional
            toggle to plot the lines in a colormap based on node tensions. The default is False
        cmap : string, optional
            colormap string type to plot tensions when colortension=True. The default is 'rainbow'

        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted

        """

        linebit = []  # make empty list to hold plotted lines, however many there are

        if self.isRod > 0:

            Xs, Ys, Zs, Te = self.getLineCoords(Time)

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

            if self.qs == 1:
                Xs, Ys, Zs, tensions = self.getLineCoords(Time)
            elif self.qs == 0:
                Xs, Ys, Zs, Ts = self.getLineCoords(Time)
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
                tensions = self.getLineTens()

            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs * Xuvec[0] + Ys * Xuvec[1] + Zs * Xuvec[2]
            Ys2d = Xs * Yuvec[0] + Ys * Yuvec[1] + Zs * Yuvec[2]

            if colortension:  # if the mooring lines want to be plotted with colors based on node tensions
                maxt = np.max(tensions)
                mint = np.min(tensions)
                for i in range(len(Xs) - 1):  # for each node in the line
                    color_ratio = ((tensions[i] + tensions[i + 1]) / 2 - mint) / (
                        maxt - mint
                    )  # ratio of the node tension in relation to the max and min tension
                    cmap_obj = cm.get_cmap(cmap)  # create a cmap object based on the desired colormap
                    rgba = cmap_obj(color_ratio)  # return the rbga values of the colormap of where the node tension is
                    linebit.append(ax.plot(Xs2d[i : i + 2], Ys2d[i : i + 2], color=rgba))
            else:
                linebit.append(ax.plot(Xs2d, Ys2d, lw=1, color=color))  # previously had lw=1 (linewidth)

        self.linebit = linebit  # can we store this internally?

        self.X = np.array([Xs, Ys, Zs])

        return linebit

    def drawLine(self, Time, ax, color="k", endpoints=False, shadow=True, colortension=False, cmap_tension="rainbow"):
        """Draw the line in 3D

        Parameters
        ----------
        Time : float
            time value at which to draw the line
        ax : axis
            the axis on which the line is to be drawn
        color : string, optional
            color identifier in one letter (k=black, b=blue,...). The default is "k".
        endpoints : bool, optional
            toggle to plot the end points of the lines. The default is False
        shadow : bool, optional
            toggle to plot the mooring line shadow on the seabed. The default is True
        colortension : bool, optional
            toggle to plot the lines in a colormap based on node tensions. The default is False
        cmap : string, optional
            colormap string type to plot tensions when colortension=True. The default is 'rainbow'

        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted
        """

        if color == "self":
            color = self.color  # attempt to allow custom colors
            lw = self.lw
        else:
            lw = 1

        linebit = []  # make empty list to hold plotted lines, however many there are

        if self.isRod > 0:

            Xs, Ys, Zs, Ts = self.getLineCoords(Time)

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

            # scatter points for line ends
            if endpoints == True:
                linebit.append(ax.scatter([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], [Zs[0], Zs[-1]], color=color))

        # drawing lines...
        else:

            if self.qs == 1:  # returns the node positions and tensions of the line, doesn't matter what time
                Xs, Ys, Zs, tensions = self.getLineCoords(Time)
            elif self.qs == 0:  # returns the node positions and time data at the given time
                Xs, Ys, Zs, Ts = self.getLineCoords(Time)
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
                tensions = self.getLineTens()

            if colortension:  # if the mooring lines want to be plotted with colors based on node tensions
                maxt = np.max(tensions)
                mint = np.min(tensions)
                for i in range(len(Xs) - 1):  # for each node in the line
                    color_ratio = ((tensions[i] + tensions[i + 1]) / 2 - mint) / (
                        maxt - mint
                    )  # ratio of the node tension in relation to the max and min tension
                    cmap_obj = cm.get_cmap(cmap_tension)  # create a cmap object based on the desired colormap
                    rgba = cmap_obj(color_ratio)  # return the rbga values of the colormap of where the node tension is
                    linebit.append(ax.plot(Xs[i : i + 2], Ys[i : i + 2], Zs[i : i + 2], color=rgba, zorder=100))
            else:
                linebit.append(ax.plot(Xs, Ys, Zs, color=color, lw=lw, zorder=100))

            if shadow:
                ax.plot(
                    Xs, Ys, np.zeros_like(Xs) - self.sys.depth, color=[0.5, 0.5, 0.5, 0.2], lw=lw, zorder=1.5
                )  # draw shadow

            if endpoints == True:
                linebit.append(ax.scatter([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], [Zs[0], Zs[-1]], color=color))

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

    def redrawLine(self, Time, colortension=False, cmap_tension="rainbow"):  # , linebit):
        """Update 3D line drawing based on instantaneous position"""

        linebit = self.linebit

        if self.isRod > 0:

            Xs, Ys, Zs, Ts = self.getLineCoords(Time)

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

            Xs, Ys, Zs, Ts = self.getLineCoords(Time)

            if colortension:
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])  # update the line ends based on the MoorDyn data
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
                tensions = self.getLineTens()  # get the tensions of the line calculated quasi-statically
                maxt = np.max(tensions)
                mint = np.min(tensions)
                cmap_obj = cm.get_cmap(cmap_tension)  # create the colormap object

                for i in range(
                    len(Xs) - 1
                ):  # for each node in the line, find the relative tension of the segment based on the max and min tensions
                    color_ratio = ((tensions[i] + tensions[i + 1]) / 2 - mint) / (maxt - mint)
                    rgba = cmap_obj(color_ratio)
                    linebit[i][
                        0
                    ]._color = rgba  # set the color of the segment to a new color based on its updated tension
                    linebit[i][0].set_data(Xs[i : i + 2], Ys[i : i + 2])  # set the x and y coordinates
                    linebit[i][0].set_3d_properties(Zs[i : i + 2])  # set the z coorindates

            else:
                linebit[0][0].set_data(Xs, Ys)  # (x and y coordinates)
                linebit[0][0].set_3d_properties(Zs)  # (z coordinates)

            # drawing water velocity vectors (not for Rods for now)
            if self.qs == 0:
                ts = self.getTimestep(Time)
                Ux = self.Ux[ts, :]
                Uy = self.Uy[ts, :]
                Uz = self.Uz[ts, :]
                # segments = quiver_data_to_segments(Xs, Ys, Zs, Ux, Uy, Uz, scale=2)
                # self.Ubits.set_segments(segments)

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

    def staticSolve(self, reset=False, tol=0.0001, profiles=0):
        """Solves static equilibrium of line. Sets the end forces of the line based on the end points' positions.

        Parameters
        ----------
        reset : boolean, optional
            Determines if the previous fairlead force values will be used for the catenary iteration. The default is False.

        tol : float
            Convergence tolerance for catenary solver measured as absolute error of x and z values in m.

        profiles : int
            Values greater than 0 signal for line profile data to be saved (used for plotting, getting distributed tensions, etc).

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
                Tol=tol,
                HF0=self.HF,
                VF0=self.VF,
                plots=profiles,
            )  # call line model
        except CatenaryError as error:
            raise LineError(self.number, error.message)

        self.th = np.arctan2(dr[1], dr[0])  # probably a more efficient way to handle this <<<
        self.HF = info["HF"]
        self.VF = info["VF"]
        self.KA2 = info["stiffnessA"]
        self.KB2 = info["stiffnessB"]
        self.LBot = info["LBot"]
        self.info = info

        self.fA[0] = fAH * dr[0] / LH
        self.fA[1] = fAH * dr[1] / LH
        self.fA[2] = fAV
        self.fB[0] = fBH * dr[0] / LH
        self.fB[1] = fBH * dr[1] / LH
        self.fB[2] = fBV
        self.TA = np.sqrt(fAH * fAH + fAV * fAV)  # end tensions
        self.TB = np.sqrt(fBH * fBH + fBV * fBV)

        # ----- compute 3d stiffness matrix for both line ends (3 DOF + 3 DOF) -----

        # solve for required variables to set up the perpendicular stiffness. Keep it horizontal
        # L_xy = np.linalg.norm(self.rB[:2] - self.rA[:2])
        # T_xy = np.linalg.norm(self.fB[:2])

        # create the rotation matrix based on the heading angle that the line is from the horizontal
        R = rotationMatrix(0, 0, self.th)

        # initialize the line's analytic stiffness matrix in the "in-line" plane then rotate the matrix to be about the global frame [K'] = [R][K][R]^T
        def from2Dto3Drotated(K2D, Kt):
            K2 = np.array([[K2D[0, 0], 0, K2D[0, 1]], [0, Kt, 0], [K2D[1, 0], 0, K2D[1, 1]]])
            return np.matmul(np.matmul(R, K2), R.T)

        self.KA = from2Dto3Drotated(
            info["stiffnessA"], -fBH / LH
        )  # stiffness matrix describing reaction force on end A due to motion of end A
        self.KB = from2Dto3Drotated(
            info["stiffnessB"], -fBH / LH
        )  # stiffness matrix describing reaction force on end B due to motion of end B
        self.KAB = from2Dto3Drotated(
            info["stiffnessAB"], fBH / LH
        )  # stiffness matrix describing reaction force on end B due to motion of end A

        # self.K6 = np.block([[ from2Dto3Drotated(self.KA),  from2Dto3Drotated(self.KAB.T)],
        #                    [ from2Dto3Drotated(self.KAB), from2Dto3Drotated(self.KB)  ]])

        if profiles > 1:
            import matplotlib.pyplot as plt

            plt.plot(info["X"], info["Z"])
            plt.show()

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
        """
        if np.isnan(self.jacobian[0,0]): #if self.LBot >= self.L and self.HF==0. and self.VF==0.  << handle tricky cases here?
            K = np.array([[0., 0.], [0., 1.0/self.jacobian[1,1] ]])
        else:
            try:
                K = np.linalg.inv(self.jacobian)
            except:
                raise LineError(self.number, f"Check Line Length ({self.L}), it might be too long, or check catenary ProfileType")
        """

        # solve for required variables to set up the perpendicular stiffness. Keep it horizontal
        L_xy = np.linalg.norm(self.rB[:2] - self.rA[:2])
        T_xy = np.linalg.norm(self.fB[:2])
        Kt = T_xy / L_xy

        # initialize the line's analytic stiffness matrix in the "in-line" plane
        KA = np.array([[self.KA2[0, 0], 0, self.KA2[0, 1]], [0, Kt, 0], [self.KA2[1, 0], 0, self.KA2[1, 1]]])

        KB = np.array([[self.KB2[0, 0], 0, self.KB2[0, 1]], [0, Kt, 0], [self.KB2[1, 0], 0, self.KB2[1, 1]]])

        # create the rotation matrix based on the heading angle that the line is from the horizontal
        R = rotationMatrix(0, 0, self.th)

        # rotate the matrix to be about the global frame [K'] = [R][K][R]^T
        KA_rot = np.matmul(np.matmul(R, KA), R.T)
        KB_rot = np.matmul(np.matmul(R, KB), R.T)

        return KA_rot, KB_rot

    def getLineTens(self):
        """Calls the catenary function to return the tensions of the Line for a quasi-static analysis"""

        # >>> this can probably be done using data already generated by static Solve <<<

        depth = self.sys.depth

        dr = self.rB - self.rA
        LH = np.hypot(dr[0], dr[1])  # horizontal spacing of line ends
        LV = dr[2]  # vertical offset from end A to end B

        if np.min([self.rA[2], self.rB[2]]) > -depth:
            self.cb = -depth - np.min(
                [self.rA[2], self.rB[2]]
            )  # if this line's lower end is off the seabed, set cb negative and to the distance off the seabed
        elif self.cb < 0:  # if a line end is at the seabed, but the cb is still set negative to indicate off the seabed
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

        Ts = info["Te"]
        return Ts

    def getTension(self, s):
        """Returns tension at a given point along the line

        Parameters
        ----------

        s : scalar or array-like
            Value or array of values for the arc length along the line from end A to end B at which
            the information is desired. Positive values are arc length in m, negative values are a
            relative location where 0 is end A, -1 is end B, and -0.5 is the midpoint.

        Returns
        -------

        tension value(s)

        """
        # if s < 0:
        #    s = -s*self.L
        # if s > self.L:
        #    raise ValueError('Specified arc length is larger than the line unstretched length.')

        Te = np.interp(s, self.info["s"], self.info["Te"])

        return Te

    def getPosition(self, s):
        """Returns position at a given point along the line

        Parameters
        ----------

        s : scalar or array-like
            Value or array of values for the arc length along the line from end A to end B at which
            the information is desired. Positive values are arc length in m, negative values are a
            relative location where 0 is end A, -1 is end B, and -0.5 is the midpoint.

        Returns
        -------

        position vector(s)

        """

        # >>> should be merged with getLineCoords and getCoordinate functionality <<<

        x = np.interp(s, self.info["s"], self.info["X"])
        z = np.interp(s, self.info["s"], self.info["Z"])

        dr = self.rB - self.rA
        LH = np.hypot(dr[0], dr[1])
        Xs = self.rA[0] + x * dr[0] / LH
        Ys = self.rA[1] + x * dr[1] / LH
        Zs = self.rA[2] + z

        return np.vstack([Xs, Ys, Zs])
