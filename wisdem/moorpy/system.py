import warnings

import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from scipy.sparse import csr_matrix
from wisdem.moorpy.body import Body
from wisdem.moorpy.line import Line
from scipy.sparse.linalg import spsolve
from wisdem.moorpy.point import Point
from mpl_toolkits.mplot3d import Axes3D

# import wisdem.moorpy.MoorSolve as msolve
from wisdem.moorpy.helpers import (
    SolveError,
    MoorPyError,
    getH,
    dsolve2,
    printVec,
    rotatePosition,
    rotationMatrix,
    set_axes_equal,
)
from wisdem.moorpy.lineType import LineType
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse.linalg.dsolve import MatrixRankWarning


class System:
    """A class for the whole mooring system"""

    # >>> note: system module will need to import Line, Point, Body for its add/creation routines
    #     (but line/point/body modules shouldn't import system) <<<

    def __init__(self, file="", dirname="", rootname="", depth=0, rho=1025, g=9.81, qs=1):
        """Creates an empty MoorPy mooring system data structure and will read an input file if provided.

        Parameters
        ----------
        file : string, optional
            An input file, usually a MoorDyn input file, that can be read into a MoorPy system. The default is "".
        depth : float, optional
            Water depth of the system. The default is 0.
        rho : float, optional
            Water density of the system. The default is 1025.
        g : float, optional
            Gravity of the system. The default is 9.81.

        Returns
        -------
        None.

        """

        # lists to hold mooring system objects
        self.bodyList = []
        # self.RodList = []    <<< TODO: add support for Rods eventually, for compatability with MoorDyn systems
        self.pointList = []
        self.lineList = []
        self.lineTypes = {}

        # the ground body (number 0, type 1[fixed]) never moves but is the parent of all anchored things
        self.groundBody = Body(
            self, 0, 1, np.zeros(6)
        )  # <<< implementation not complete <<<< be careful here if/when MoorPy is split up

        # constants used in the analysis
        self.depth = depth  # water depth [m]
        self.rho = rho  # water density [kg/m^3]
        self.g = g  # gravitational acceleration [m/s^2]

        self.nDOF = 0  # number of (free) degrees of freedom of the mooring system (needs to be set elsewhere)
        self.freeDOFs = []  # array of the values of the free DOFs of the system at different instants (2D list)

        self.nCpldDOF = 0  # number of (coupled) degrees of freedom of the mooring system (needs to be set elsewhere)
        self.cpldDOFs = []  # array of the values of the coupled DOFs of the system at different instants (2D list)

        self.display = 0  # a flag that controls how much printing occurs in methods within the System (Set manually. Values > 0 cause increasing output.)

        # read in data from an input file if a filename was provided
        if len(file) > 0:
            self.load(file)

        # set the quasi-static/dynamic toggle for the entire mooring system
        self.qs = qs
        if self.qs == 0:  # if the mooring system is desired to be used as a portrayal of MoorDyn data
            if len(file) == 0 or len(dirname) == 0 or len(rootname) == 0:
                raise ValueError(
                    "The directory location of the MoorDyn output files needs to be given OR the name of the .fst file needs to be given, without the .fst"
                )
            # load in the MoorDyn data for each line to set the xp,yp,zp positions of each node in the line
            # Each row in the xp matrix is a time step and each column is a node in the line
            for line in self.lineList:
                try:
                    line.loadData(dirname, rootname)
                except:
                    raise ValueError(
                        "There is likely not a .MD.Line#.out file in the directory. Make sure Line outputs are set to 'p' in the MoorDyn input file"
                    )

    def addBody(self, mytype, r6, m=0, v=0, rCG=np.zeros(3), AWP=0, rM=np.zeros(3), f6Ext=np.zeros(6)):
        """Convenience function to add a Body to a mooring system

        Parameters
        ----------
        type : int
            the body type: 0 free to move, 1 fixed, -1 coupled externally
        r6 : array
            6DOF position and orientation vector [m, rad].
        m : float, optional
            mass, centered at CG [kg]. The default is 0.
        v : float, optional
            volume, centered at reference point [m^3]. The default is 0.
        rCG : array, optional
            center of gravity position in body reference frame [m]. The default is np.zeros(3).
        AWP : float, optional
            waterplane area - used for hydrostatic heave stiffness if nonzero [m^2]. The default is 0.
        rM : float or array, optional
            coordinates or height of metacenter relative to body reference frame [m]. The default is np.zeros(3).
        f6Ext : array, optional
            applied external forces and moments vector in global orientation (not including weight/buoyancy) [N]. The default is np.zeros(6).

        Returns
        -------
        None.

        """

        self.bodyList.append(
            Body(self, len(self.bodyList) + 1, mytype, r6, m=m, v=v, rCG=rCG, AWP=AWP, rM=rM, f6Ext=f6Ext)
        )

        # handle display message if/when MoorPy is reorganized by classes

    def addPoint(self, mytype, r, m=0, v=0, fExt=np.zeros(3), DOFs=[0, 1, 2]):
        """Convenience function to add a Point to a mooring system

        Parameters
        ----------
        type : int
            the point type: 0 free to move, 1 fixed, -1 coupled externally
        r : array
            x,y,z coordate position vector [m].
        m : float, optional
            mass [kg]. The default is 0.
        v : float, optional
            volume [m^3]. The default is 0.
        fExt : array, optional
            applied external force vector in global orientation (not including weight/buoyancy) [N]. The default is np.zeros(3).
        DOFs : list, optional
            list of which coordinate directions are DOFs for this point (default 0,1,2=x,y,z). E.g. set [2] for vertical motion only.. The default is [0,1,2].

        Returns
        -------
        None.

        """

        self.pointList.append(Point(self, len(self.pointList) + 1, mytype, r, m=m, v=v, fExt=fExt, DOFs=DOFs))

        # print("Created Point "+str(self.pointList[-1].number))
        # handle display message if/when MoorPy is reorganized by classes

    def addLine(self, lUnstr, type_string, nSegs=40, pointA=0, pointB=0, cb=0):
        """Convenience function to add a Line to a mooring system

        Parameters
        ----------
        lUnstr : float
            unstretched line length [m].
        type_string : string
            string identifier of LineType object that this Line is to be.
        nSegs : int, optional
            number of segments to split the line into. The default is 20.
        pointA int, optional
            Point number to attach end A of the line to.
        pointB int, optional
            Point number to attach end B of the line to.

        Returns
        -------
        None.

        """

        self.lineList.append(
            Line(self, len(self.lineList) + 1, lUnstr, self.lineTypes[type_string].name, nSegs=nSegs, cb=cb)
        )

        if pointA > 0:
            if pointA <= len(self.pointList):
                self.pointList[pointA - 1].attachLine(self.lineList[-1].number, 0)
            else:
                raise Exception(f"Provided pointA of {pointA} exceeds number of points.")
        if pointB > 0:
            if pointB <= len(self.pointList):
                self.pointList[pointB - 1].attachLine(self.lineList[-1].number, 1)
            else:
                raise Exception(f"Provided pointB of {pointB} exceeds number of points.")

        # print("Created Line "+str(self.lineList[-1].number))
        # handle display message if/when MoorPy is reorganized by classes

    """
    def removeLine(self, lineID):
        '''Removes a line from the system.'''

        if lineID > 0 and lineID <= len(self.lineList):

            # detach line from Points
            for point in self.pointList:
                if lineID in point.attached:
                    endB = point.attachedEndB[point.attached.index(lineID)] # get whether it's end B of the line attached to this ponit
                    point.detachLine(lineID, endB)

            # remove line from list
            self.lineList.pop(lineID-1)
            >>> This doesn't currently work because it would required adjusting indexing of all references to lines in the system  <<<

        else:
            raise Exception("Invalid line number")

    """

    def addLineType(self, type_string, d, massden, EA):
        """Convenience function to add a LineType to a mooring system

        Parameters
        ----------
        type_string : string
            string identifier of the LineType object that is to be added.
        d : float
            volume-equivalent diameter [m].
        massden : float
            mass of line per length, or mass density [kg/m], used to calculate weight density (w) [N/m]
        EA : float
            extensional stiffness [N].

        Returns
        -------
        None.

        """

        self.lineTypes[type_string] = LineType(type_string, d, massden, EA)

        # handle display message if/when MoorPy is reorganized by classes

    def load(self, filename):
        """Loads a MoorPy System from a MoorDyn-style input file

        Parameters
        ----------
        filename : string
            the file name of a MoorDyn-style input file.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # create/empty the lists to start with

        RodDict = {}  # create empty dictionary for rod types
        self.lineTypes = {}  # create empty dictionary for line types

        # ensure the mooring system's object lists are empty before adding to them
        self.bodyList = []
        # self.RodList  = []
        self.pointList = []
        self.lineList = []

        # figure out if it's a YAML file or MoorDyn-style file based on the extension, then open and process
        print("attempting to read " + filename)

        # assuming YAML format
        if ".yaml" in filename.lower() or ".yml" in filename.lower():

            with open(filename) as file:
                mooring = yaml.load(file, Loader=yaml.FullLoader)  # get dict from YAML file

            self.parseYAML(mooring)

        # assuming normal form
        else:
            f = open(filename, "r")

            # read in the data

            for line in f:  # loop through each line in the file

                # get line type property sets
                if line.count("---") > 0 and (
                    line.upper().count("LINE DICTIONARY") > 0 or line.upper().count("LINE TYPES") > 0
                ):
                    line = next(f)  # skip this header line, plus channel names and units lines
                    line = next(f)
                    line = next(f)
                    while line.count("---") == 0:
                        entries = line.split()
                        self.lineTypes[entries[0]] = LineType(
                            entries[0], np.float_(entries[1]), np.float_(entries[2]), np.float_(entries[3])
                        )
                        line = next(f)

                # get line type property sets
                if line.count("---") > 0 and (
                    line.upper().count("ROD DICTIONARY") > 0 or line.upper().count("ROD TYPES") > 0
                ):
                    line = next(f)  # skip this header line, plus channel names and units lines
                    line = next(f)
                    line = next(f)
                    while line.count("---") == 0:
                        entries = line.split()
                        # RodTypesName.append(entries[0]) # name string
                        # RodTypesD.append(   entries[1]) # diameter
                        # RodDict[entries[0]] = entries[1] # add dictionary entry with name and diameter
                        line = next(f)

                # get properties of each Body
                if line.count("---") > 0 and (
                    line.upper().count("BODIES") > 0
                    or line.upper().count("BODY LIST") > 0
                    or line.upper().count("BODY PROPERTIES") > 0
                ):
                    line = next(f)  # skip this header line, plus channel names and units lines
                    line = next(f)
                    line = next(f)
                    while line.count("---") == 0:
                        entries = line.split()
                        entry0 = entries[0].lower()

                        num = np.int(
                            "".join(c for c in entry0 if not c.isalpha())
                        )  # remove alpha characters to identify Body #

                        if ("fair" in entry0) or ("coupled" in entry0) or ("ves" in entry0):  # coupled case
                            bodyType = -1
                        elif ("con" in entry0) or ("free" in entry0):  # free case
                            bodyType = 0
                        else:  # for now assuming unlabeled free case
                            bodyType = 0
                            # if we detected there were unrecognized chars here, could: raise ValueError(f"Body type not recognized for Body {num}")
                        # bodyType = -1   # manually setting the body type as -1 for FAST.Farm SM investigation

                        r6 = np.array(entries[1:7], dtype=float)  # initial position and orientation [m, rad]
                        r6[3:] = r6[3:] * np.pi / 180.0  # convert from deg to rad
                        rCG = np.array(entries[7:10], dtype=float)  # location of body CG in body reference frame [m]
                        m = np.float_(entries[10])  # mass, centered at CG [kg]
                        v = np.float_(entries[11])  # volume, assumed centered at reference point [m^3]

                        self.bodyList.append(Body(self, num, bodyType, r6, m=m, v=v, rCG=rCG))

                        line = next(f)

                # get properties of each rod
                if line.count("---") > 0 and (
                    line.upper().count("RODS") > 0
                    or line.upper().count("ROD LIST") > 0
                    or line.upper().count("ROD PROPERTIES") > 0
                ):
                    line = next(f)  # skip this header line, plus channel names and units lines
                    line = next(f)
                    line = next(f)
                    while line.count("---") == 0:
                        entries = line.split()
                        entry0 = entries[0].lower()

                        num = np.int(
                            "".join(c for c in entry0 if not c.isalpha())
                        )  # remove alpha characters to identify Rod #
                        lUnstr = 0  # not specified directly so skip for now
                        dia = RodDict[entries[2]]  # find diameter based on specified rod type string
                        nSegs = np.int(entries[9])

                        # additional things likely missing here <<<

                        # RodList.append( Line(dirName, num, lUnstr, dia, nSegs, isRod=1) )
                        line = next(f)

                # get properties of each Point
                if line.count("---") > 0 and (
                    line.upper().count("POINTS") > 0
                    or line.upper().count("POINT LIST") > 0
                    or line.upper().count("POINT PROPERTIES") > 0
                ):
                    line = next(f)  # skip this header line, plus channel names and units lines
                    line = next(f)
                    line = next(f)
                    while line.count("---") == 0:
                        entries = line.split()
                        entry0 = entries[0].lower()
                        entry1 = entries[1].lower()

                        num = np.int(
                            "".join(c for c in entry0 if not c.isalpha())
                        )  # remove alpha characters to identify Point #

                        if ("anch" in entry1) or ("fix" in entry1):
                            pointType = 1
                            # attach to ground body for ease of identifying anchors
                            self.groundBody.attachPoint(num, entries[2:5])

                        elif ("body" in entry1) or ("turbine" in entry1):
                            pointType = 1
                            # attach to body here
                            BodyID = int("".join(filter(str.isdigit, entry1)))
                            if len(self.bodyList) < BodyID:
                                self.bodyList.append(Body(self, 1, 0, np.zeros(6)))

                            rRel = np.array(entries[2:5], dtype=float)
                            self.bodyList[BodyID - 1].attachPoint(num, rRel)

                        elif ("fair" in entry1) or ("ves" in entry1):
                            pointType = -1
                            # attach to a generic platform body (and make it if it doesn't exist)
                            if len(self.bodyList) > 1:
                                raise ValueError(
                                    "Generic Fairlead/Vessel-type points aren't supported when bodies are defined."
                                )
                            if len(self.bodyList) == 0:
                                # print("Adding a body to attach fairlead points to.")
                                self.bodyList.append(Body(self, 1, 0, np.zeros(6)))  # , m=m, v=v, rCG=rCG) )

                            rRel = np.array(entries[2:5], dtype=float)
                            self.bodyList[0].attachPoint(num, rRel)

                        elif ("con" in entry1) or ("free" in entry1):
                            pointType = 0
                        else:
                            print("Point type not recognized")

                        if "seabed" in entries[4]:
                            entries[4] = -self.depth
                        r = np.array(entries[2:5], dtype=float)
                        m = np.float_(entries[5])
                        v = np.float_(entries[6])
                        fExt = np.array(entries[7:10], dtype=float)
                        self.pointList.append(Point(self, num, pointType, r, m=m, v=v, fExt=fExt))
                        line = next(f)

                # get properties of each line
                if line.count("---") > 0 and (
                    line.upper().count("LINES") > 0
                    or line.upper().count("LINE LIST") > 0
                    or line.upper().count("LINE PROPERTIES") > 0
                ):
                    line = next(f)  # skip this header line, plus channel names and units lines
                    line = next(f)
                    line = next(f)
                    while line.count("---") == 0:
                        entries = line.split()

                        # print(entries)

                        num = np.int(entries[0])
                        # dia    = lineTypes[entries[1]].d # find diameter based on specified rod type string
                        lUnstr = np.float_(entries[2])
                        nSegs = np.int(entries[3])
                        # w = lineTypes[entries[1]].w  # line wet weight per unit length
                        # EA= lineTypes[entries[1]].EA

                        # lineList.append( Line(dirName, num, lUnstr, dia, nSegs) )
                        self.lineList.append(
                            Line(
                                self,
                                num,
                                lUnstr,
                                self.lineTypes[entries[1]].name,
                                nSegs=nSegs,
                                attachments=[np.int(entries[4]), np.int(entries[5])],
                            )
                        )

                        # attach ends
                        self.pointList[np.int(entries[4]) - 1].attachLine(num, 0)
                        self.pointList[np.int(entries[5]) - 1].attachLine(num, 1)

                        line = next(f)

                # get options entries
                if line.count("---") > 0 and "options" in line.lower():
                    # print("READING OPTIONS")
                    line = next(f)  # skip this header line
                    while line.count("---") == 0:
                        entries = line.split()
                        entry0 = entries[0].lower()
                        entry1 = entries[1].lower()

                        # print(entries)

                        if entry1 == "g" or entry1 == "gravity":
                            self.g = np.float_(entry0)
                        elif entries[1] == "WtrDpth" or entries[1] == "depth":
                            self.depth = np.float_(entry0)
                        elif entry1 == "rho" or entry1 == "wtrdnsty":
                            self.rho = np.float_(entry0)

                        line = next(f)

            f.close()  # close data file

        # any error check? <<<

        print(f"Mooring input file '{filename}' loaded successfully.")

    def parseYAML(self, data):
        """Creates a MoorPy System from a YAML dictionary
        >>> work in progress <<<

        Parameters
        ----------
        data : dictionary
            YAML dictionary to be parsed through.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # line types
        for d in data["line_types"]:
            dia = float(d["diameter"])
            w = float(d["mass_density"])
            EA = float(d["stiffness"])
            if d["breaking_load"]:
                MBL = float(d["breaking_load"])
            else:
                MBL = 0
            self.lineTypes[d["name"]] = LineType(d["name"], dia, w, EA, MBL=MBL)

        # rod types TBD

        # bodies TBDish
        if "bodies" in data:
            pass  # <<<<<<<<<< need to fill this in once the YAML format is full defined

        # rods TBD

        # points
        pointDict = dict()
        for i, d in enumerate(data["points"]):

            pointDict[d["name"]] = i  # make dictionary based on names pointing to point indices, for name-based linking

            entry0 = d["name"].lower()
            entry1 = d["type"].lower()

            # num = np.int("".join(c for c in entry0 if not c.isalpha()))  # remove alpha characters to identify Point #
            num = i + 1  # not counting on things being numbered in YAML files

            if ("anch" in entry1) or ("fix" in entry1):
                pointType = 1
                # attach to ground body for ease of identifying anchors
                self.groundBody.attachPoint(num, d["location"])

            elif "body" in entry1:
                pointType = 1
                # attach to body here
                BodyID = int("".join(filter(str.isdigit, entry1)))
                rRel = np.array(d["location"], dtype=float)
                self.bodyList[BodyID - 1].attachPoint(num, rRel)

            elif ("fair" in entry1) or ("ves" in entry1):
                pointType = 1  # <<< this used to be -1.  I need to figure out a better way to deal with this for different uses! <<<<<<
                # attach to a generic platform body (and make it if it doesn't exist)
                if len(self.bodyList) > 1:
                    raise ValueError("Generic Fairlead/Vessel-type points aren't supported when bodies are defined.")
                if len(self.bodyList) == 0:
                    # print("Adding a body to attach fairlead points to.")
                    self.bodyList.append(Body(self, 1, 0, np.zeros(6)))  # , m=m, v=v, rCG=rCG) )

                rRel = np.array(d["location"], dtype=float)
                self.bodyList[0].attachPoint(num, rRel)

            elif ("con" in entry1) or ("free" in entry1):
                pointType = 0
            else:
                print("Point type not recognized")

            r = np.array(d["location"], dtype=float)

            if "mass" in d:
                m = np.float_(d["mass"])
            else:
                m = 0.0

            if "volume" in d:
                v = np.float_(d["volume"])
            else:
                v = 0.0

            self.pointList.append(Point(self, num, pointType, r, m=m, v=v))

        # lines
        for i, d in enumerate(data["lines"]):

            num = i + 1

            lUnstr = np.float_(d["length"])

            self.lineList.append(Line(self, num, lUnstr, self.lineTypes[d["type"]].name))

            # attach ends (name matching here)
            self.pointList[pointDict[d["endA"]]].attachLine(num, 0)
            self.pointList[pointDict[d["endB"]]].attachLine(num, 1)

        # get options entries
        if "water_depth" in data:
            self.depth = data["water_depth"]

        if "rho" in data:
            self.rho = data["rho"]
        elif "water_density" in data:
            self.rho = data["water_density"]

    def readBathymetryFile(self, filename):
        f = open(filename, "r")

        # skip the header
        line = next(f)
        # collect the number of grid values in the x and y directions from the second and third lines
        line = next(f)
        nGridX = int(line.split()[1])
        line = next(f)
        nGridY = int(line.split()[1])
        # allocate the Xs, Ys, and main bathymetry grid arrays
        bathGrid_Xs = np.zeros(nGridX)
        bathGrid_Ys = np.zeros(nGridY)
        bathGrid = np.zeros([nGridX, nGridY])
        # read in the fourth line to the Xs array
        line = next(f)
        bathGrid_Xs = [float(line.split()[i]) for i in range(nGridX)]
        # read in the remaining lines in the file into the Ys array (first entry) and the main bathymetry grid
        for i in range(nGridY):
            line = next(f)
            entries = line.split()
            bathGrid_Ys[i] = entries[0]
            bathGrid[i, :] = entries[1:]

        return bathGrid_Xs, bathGrid_Ys, bathGrid

    def unload(self, fileName, MDversion=2, **kwargs):
        """Unloads a MoorPy system into a MoorDyn-style input file

        Parameters
        ----------
        fileName : string
            file name of output file to hold MoorPy System.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if MDversion == 1:
            # For version MoorDyn v1

            # Collection of default values, each can be customized when the method is called

            # Settings
            Echo = False  # Echo input data to <RootName>.ech (flag)
            dtm = 0.001
            kbot = 3e6
            cbot = 3e5
            dtIC = 2
            TmaxIC = 600
            CdScaleIC = 10
            threshIC = 0.01

            # Line Properties
            cIntDamp = -0.8
            EI = 0.0
            Can = 1.0
            Cat = 0.0
            Cdn = 1.6
            Cdt = 0.05

            # Body Properties (for each body in bodyList)
            #! Add Comments
            IX = 0
            IY = 0
            IZ = 0
            CdA_xyz = [0, 0, 0]
            Ca_xyz = [0, 0, 0]

            # Rod List Properties

            # Point Properties (for each point in pointList)
            #! Add Comments
            CdA = 0.0
            Ca = 0.0

            # Line Properties
            flag = "p"  # "-"

            # If a custom value was given, use that instead of the default value(For some reason this doesnt work)
            # The exec method isn't working and isn't encouraged. perhaps we have to save all the above variables in a dictionary, and update that dictioanry with kwargs.
            for key in kwargs:
                print("Using Custom value for", key, kwargs[key])
                # vars()[key] = kwargs[key]
                # exec(key + ' = ' + str(kwargs[key]))
                # eval(key + ' = ' + str(kwargs[key]))

            # Outputs List
            # Outputs = ["FairTen1","FairTen2","FairTen3","FairTen4","FairTen5","FairTen6","FairTen7","FairTen8","FairTen9","FairTen10","FairTen11","FairTen12"]
            # Outputs = ["FairTen1","FairTen2","FairTen3"];
            Outputs = [
                "FairTen1",
                "FairTen2",
                "FairTen3",
                "FairTen4",
                "FairTen5",
                "FairTen6",
                "FairTen7",
                "FairTen8",
                "FairTen9",
                "Con2Fz",
                "Con3Fz",
                "Con6Fz",
                "Con7Fz",
                "Con10Fz",
                "Con11Fz",
                "L3N20T",
                "L6N20T",
                "L9N20T",
            ]

            #! Standard Option (Fairing Tenstion for num of lines)

            print("attempting to write " + fileName + " for MoorDyn v" + str(MDversion))
            # Array to add strings to for each line of moordyn input file
            L = []

            # Input File Header
            L.append(f"---------------- MoorDyn v{MDversion} Input File ------------------")
            if "description" in locals():
                L.append("MoorDyn input for " + description)
            else:
                L.append("Generated by MoorPy")

            L.append("{:5}    Echo      - echo the input file data (flag)".format(str(Echo).upper()))

            # Line Dictionary Header
            L.append("---------------------- LINE TYPES -----------------------------------------------------")
            L.append(f"{len(self.lineTypes)}    NTypes   - number of LineTypes")
            L.append("LineType         Diam     MassDen   EA        cIntDamp     EI     Can    Cat    Cdn    Cdt")
            L.append("   (-)           (m)      (kg/m)    (N)        (Pa-s)    (N-m^2)  (-)    (-)    (-)    (-)")

            # Line Dicationary Table
            for key in self.lineTypes:
                # for key,value in self.lineTypes.items(): (Another way to iterate through dictionary)
                L.append(
                    "{:<15} {:7.3f} {:8.3f} {:8.1f} ".format(
                        key, self.lineTypes[key].d, self.lineTypes[key].mlin, self.lineTypes[key].EA
                    )
                    + "{:7.1f} {:<7.1f} {:<7.1f} {:<7.1f} {:<7.2f}".format(cIntDamp, Can, Cat, Cdn, Cdt)
                )

            # Point Properties Header
            L.append("---------------------- POINTS ---------------------------------------------------------")
            L.append(f"{len(self.pointList)}    NConnects   - number of connections including anchors and fairleads")
            L.append("Node    Type         X        Y        Z        M      V      FX     FY     FZ    CdA    Ca ")
            L.append("(-)     (-)         (m)      (m)      (m)      (kg)   (m^3)  (kN)   (kN)   (kN)   (m2)   ()")

            # Point Properties Table
            for point in self.pointList:
                point_pos = point.r  # Define point position in global reference frame
                if point.type == 1:  # point is Fized or attached (anch, body, fix)
                    point_type = "Fixed"

                    # import pdb
                    # pdb.set_trace()
                    # Check if the point is attached to body
                    for body in self.bodyList:
                        for attached_Point in body.attachedP:
                            if attached_Point == point.number:
                                # point_type = "Body" + str(body.number)
                                point_type = "Vessel"
                                point_pos = body.rPointRel[
                                    body.attachedP.index(attached_Point)
                                ]  # Redefine point position in the body reference frame

                if point.type == 0:  # point is Coupled Externally (con, free)
                    point_type = "Connect"

                if point.type == -1:  # point is free to move (fair, ves)
                    point_type = "Vessel"

                L.append(
                    "{:<4d} {:12} {:8.2f} {:8.2f} {:8.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(
                        point.number,
                        point_type,
                        point_pos[0],
                        point_pos[1],
                        point_pos[2],
                        point.m,
                        point.v,
                        point.fExt[0],
                        point.fExt[1],
                        point.fExt[2],
                        CdA,
                        Ca,
                    )
                )

            # Line Properties Header
            L.append("---------------------- LINES -----------------------------------------------------")
            L.append(f"{len(self.lineList)}    NLines   - number of line objects")
            L.append("Line      LineType   UnstrLen  NumSegs  AttachA  AttachB  Outputs")
            L.append("(-)         (-)       (m)        (-)     (-)      (-)     (-)")

            # Line Properties Table
            # (Create a ix2 array of connection points from a list of m points)
            connection_points = np.empty(
                [len(self.lineList), 2]
            )  # First column is Anchor Node, second is Fairlead node
            for point_ind, point in enumerate(self.pointList, start=1):  # Loop through all the points
                for (line, line_pos) in zip(
                    point.attached, point.attachedEndB
                ):  # Loop through all the lines #s connected to this point
                    if line_pos == 0:  # If the A side of this line is connected to the point
                        connection_points[line - 1, 0] = point_ind  # Save as as an Anchor Node
                        # connection_points[line -1,0] = self.pointList.index(point) + 1
                    elif line_pos == 1:  # If the B side of this line is connected to the point
                        connection_points[line - 1, 1] = point_ind  # Save as a Fairlead node
                        # connection_points[line -1,1] = self.pointList.index(point) + 1
            # Populate text
            for i in range(len(self.lineList)):
                L.append(
                    "{:<4d} {:<15} {:8.3f} {:5d} {:7d} {:8d}      {}".format(
                        self.lineList[i].number,
                        self.lineList[i].type,
                        self.lineList[i].L,
                        self.lineList[i].nNodes - 1,
                        int(connection_points[i, 0]),
                        int(connection_points[i, 1]),
                        flag,
                    )
                )

            # Solver Options Header
            L.append("---------------------- OPTIONS ----------------------------------------")

            # Solver Options
            L.append("{:<9.3f}dtM          - time step to use in mooring integration (s)".format(float(dtm)))
            L.append("{:<9.0e}kbot           - bottom stiffness (Pa/m)".format(kbot))
            L.append("{:<9.0e}cbot           - bottom damping (Pa-s/m)".format(cbot))
            L.append("{:<9.0f}dtIC      - time interval for analyzing convergence during IC gen (s)".format(int(dtIC)))
            L.append("{:<9.0f}TmaxIC      - max time for ic gen (s)".format(int(TmaxIC)))
            L.append(
                "{:<9.0f}CdScaleIC      - factor by which to scale drag coefficients during dynamic relaxation (-)".format(
                    int(CdScaleIC)
                )
            )
            L.append("{:<9.2f}threshIC      - threshold for IC convergence (-)".format(threshIC))

            """
            #Failure Header
            #Failure Table
            """

            # Outputs Header
            L.append("----------------------------OUTPUTS--------------------------------------------")

            # Outputs List
            for Output in Outputs:
                L.append(Output)
            L.append("END")

            # Final Line
            L.append("--------------------- need this line ------------------")

            # Write the text file
            with open(fileName, "w") as out:
                for x in range(len(L)):
                    out.write(L[x])
                    out.write("\n")

            print("Successfully written " + fileName + " input file using MoorDyn v1")

        elif MDversion == 2:
            # For version MoorDyn v?.??

            # Collection of default values, each can be customized when the method is called

            # Header
            # version =
            # description =

            # Settings
            Echo = False  # Echo input data to <RootName>.ech (flag)
            dtm = 0.0002  # time step to use in mooring integration
            WaveKin = 3  # wave kinematics flag (1=include(unsupported), 0=neglect, 3=currentprofile.txt
            kb = 3.0e6  # bottom stiffness
            cb = 3.0e5  # bottom damping
            ICDfac = 2.0  # factor by which to scale drag coefficients during dynamic relaxation IC gen
            ICthresh = 0.01  # threshold for IC convergence
            ICTmax = 10  # threshold for IC convergence

            # Line Properties
            #! Add Comments
            cIntDamp = -1.0
            EI = 0.0
            Can = 1.0
            Cat = 0.0
            Cdn = 1.0
            Cdt = 0.0

            # Body Properties (for each body in bodyList)
            #! Add Comments
            IX = 0
            IY = 0
            IZ = 0
            CdA_xyz = [0, 0, 0]
            Ca_xyz = [0, 0, 0]

            # Rod List Properties

            # Point Properties (for each point in pointList)
            #! Add Comments
            CdA = 0.0
            Ca = 0.0

            # Line Properties
            flag = "p"  # "-"

            # If a custom value was given, use that instead of the default value(For some reason this doesnt work)
            # The exec method isn't working and isn't encouraged. perhaps we have to save all the above variables in a dictionary, and update that dictioanry with kwargs.
            for key in kwargs:
                print("Using Custom value for", key, kwargs[key])
                # vars()[key] = kwargs[key]
                # exec(key + ' = ' + str(kwargs[key]))
                # eval(key + ' = ' + str(kwargs[key]))

            # Outputs List
            # Outputs = ["FairTen1","FairTen2","FairTen3","FairTen4","FairTen5","FairTen6","FairTen7","FairTen8","FairTen9","FairTen10","FairTen11","FairTen12"]
            Outputs = ["FairTen1", "FairTen2", "FairTen3"]
            #! Standard Option (Fairing Tenstion for num of lines)

            print("attempting to write " + fileName + " for MoorDyn v" + str(MDversion))
            # Array to add strings to for each line of moordyn input file
            L = []

            # Input File Header
            L.append(f" MoorDyn v{MDversion} Input File ")
            if "description" in locals():
                L.append("MoorDyn input for " + description)
            else:
                L.append("Generated by MoorPy")

            # L.append("{:5}    Echo      - echo the input file data (flag)"
            #          .format(str(Echo).upper()))

            # Line Dictionary Header
            L.append("---------------------- LINE TYPES -----------------------------------------------------")
            L.append("LineType         Diam     MassDen   EA        cIntDamp     EI     Can    Cat    Cdn    Cdt")
            L.append("   (-)           (m)      (kg/m)    (N)        (Pa-s)    (N-m^2)  (-)    (-)    (-)    (-)")

            # Line Dicationary Table
            for key in self.lineTypes:
                # for key,value in self.lineTypes.items(): (Another way to iterate through dictionary)
                L.append(
                    "{:<15} {:7.4f} {:8.2f} {:7.3e} ".format(
                        key, self.lineTypes[key].d, self.lineTypes[key].mlin, self.lineTypes[key].EA
                    )
                    + "{:7.3e} {:7.3e} {:<7.3f} {:<7.3f} {:<7.3f} {:<7.3f}".format(cIntDamp, EI, Can, Cat, Cdn, Cdt)
                )

            # Rod Dictionary Header
            L.append("--------------------- ROD TYPES -----------------------------------------------------")
            L.append("RodType  Diam    MassDenInAir   Can     Cat    Cdn     Cdt ")
            L.append("(-)       (m)       (kg/m)      (-)     (-)    (-)     (-)  ")

            """
            # Rod Dictionary Table
            for i, rod_type in enumerate(self.lineTypes,start=1):
            """

            # Body List Header
            L.append("----------------------- BODIES -----------------------------------")
            L.append(
                "BodyID      X0   Y0   Z0    r0    p0    y0    Xcg   Ycg   Zcg     M      V        IX       IY       IZ     CdA-x,y,z Ca-x,y,z"
            )
            L.append(
                "   (-)      (m)  (m)  (m)  (deg) (deg) (deg)  (m)   (m)   (m)    (kg)   (m^3)  (kg-m^2) (kg-m^2) (kg-m^2)   (m^2)      (-)"
            )

            # Body List Table
            for body in self.bodyList:
                L.append(
                    "    {:<4d} {:<5.2f} {:<5.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<8.2f} {:<7.2f} {:<11.2f}".format(
                        body.number,
                        body.r6[0],
                        body.r6[1],
                        body.r6[2],
                        np.rad2deg(body.r6[3]),
                        np.rad2deg(body.r6[4]),
                        np.rad2deg(body.r6[5]),
                        body.rCG[0],
                        body.rCG[1],
                        body.rCG[2],
                        body.m,
                        body.v,
                    )
                    + "{:<9d} {:<9d} {:<7d} {:<2d} {:<2d} {:<8d} {:<1d}".format(
                        IX, IY, IZ, CdA_xyz[0], CdA_xyz[1], CdA_xyz[2], Ca_xyz[0], Ca_xyz[1], Ca_xyz[2]
                    )
                )

            # Rod Properties Header
            L.append("---------------------- RODS --------------------")
            L.append("RodID  Type/BodyID  RodType   Xa   Ya   Za   Xb   Yb   Zb  NumSegs  Flags/Outputs")
            L.append("(-)      (-)         (-)      (m)  (m)  (m)  (m)  (m)  (m)    (-)      (-)   ")

            """
            #Rod Properties Table
            """

            # Point Properties Header
            L.append("---------------------- POINTS ---------------------------------------------------------")
            L.append("Node    Type         X        Y        Z        M      V      FX     FY     FZ    CdA    Ca ")
            L.append("(-)     (-)         (m)      (m)      (m)      (kg)   (m^3)  (kN)   (kN)   (kN)   (m2)   ()")

            # Point Properties Table
            for point in self.pointList:
                point_pos = point.r  # Define point position in global reference frame
                if point.type == 1:  # point is Fized or attached (anch, body, fix)
                    point_type = "Fixed"

                    # import pdb
                    # pdb.set_trace()
                    # Check if the point is attached to body
                    for body in self.bodyList:
                        for attached_Point in body.attachedP:
                            if attached_Point == point.number:
                                point_type = "Body" + str(body.number)
                                point_pos = body.rPointRel[
                                    body.attachedP.index(attached_Point)
                                ]  # Redefine point position in the body reference frame

                if point.type == 0:  # point is Coupled Externally (con, free)
                    point_type = "Connect"

                if point.type == -1:  # point is free to move (fair, ves)
                    point_type = "Vessel"

                L.append(
                    "{:<4d} {:12} {:8.2f} {:8.2f} {:8.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(
                        point.number,
                        point_type,
                        point_pos[0],
                        point_pos[1],
                        point_pos[2],
                        point.m,
                        point.v,
                        point.fExt[0],
                        point.fExt[1],
                        point.fExt[2],
                        CdA,
                        Ca,
                    )
                )

            # Line Properties Header
            L.append("---------------------- LINES -----------------------------------------------------")
            L.append("Line      LineType   UnstrLen  NumSegs  AttachA  AttachB  Outputs")
            L.append("(-)         (-)       (m)        (-)     (-)      (-)     (-)")

            # Line Properties Table
            # (Create a ix2 array of connection points from a list of m points)
            connection_points = np.empty(
                [len(self.lineList), 2]
            )  # First column is Anchor Node, second is Fairlead node
            for point_ind, point in enumerate(self.pointList, start=1):  # Loop through all the points
                for (line, line_pos) in zip(
                    point.attached, point.attachedEndB
                ):  # Loop through all the lines #s connected to this point
                    if line_pos == 0:  # If the A side of this line is connected to the point
                        connection_points[line - 1, 0] = point_ind  # Save as as an Anchor Node
                        # connection_points[line -1,0] = self.pointList.index(point) + 1
                    elif line_pos == 1:  # If the B side of this line is connected to the point
                        connection_points[line - 1, 1] = point_ind  # Save as a Fairlead node
                        # connection_points[line -1,1] = self.pointList.index(point) + 1
            # Populate text
            for i in range(len(self.lineList)):
                L.append(
                    "{:<4d} {:<15} {:8.3f} {:5d} {:7d} {:8d}      {}".format(
                        self.lineList[i].number,
                        self.lineList[i].type,
                        self.lineList[i].L,
                        self.lineList[i].nNodes - 1,
                        int(connection_points[i, 0]),
                        int(connection_points[i, 1]),
                        flag,
                    )
                )

            # Solver Options Header
            L.append("---------------------- OPTIONS ----------------------------------------")

            # Solver Options
            L.append("{:<9.4f}dtM          - time step to use in mooring integration".format(float(dtm)))
            L.append(
                "{:<9d}WaveKin      - wave kinematics flag (1=include(unsupported), 0=neglect, 3=currentprofile.txt)".format(
                    int(WaveKin)
                )
            )
            L.append("{:<9.1e}kb           - bottom stiffness".format(kb))
            L.append("{:<9.1e}cb           - bottom damping".format(cb))
            L.append("{:<9.2f}WtrDpth      - water depth".format(self.depth))
            L.append(
                "{:<9.1f}ICDfac       - factor by which to scale drag coefficients during dynamic relaxation IC gen".format(
                    int(ICDfac)
                )
            )
            L.append("{:<9.2f}ICthresh     - threshold for IC convergence".format(ICthresh))
            L.append("{:<9d}ICTmax       - threshold for IC convergence".format(int(ICTmax)))

            """
            #Failure Header
            #Failure Table
            """

            # Outputs Header
            L.append("----------------------------OUTPUTS--------------------------------------------")

            # Outputs List
            for Output in Outputs:
                L.append(Output)
            L.append("END")

            # Final Line
            L.append("--------------------- need this line ------------------")

            # Write the text file
            with open(fileName, "w") as out:
                for x in range(len(L)):
                    out.write(L[x])
                    out.write("\n")

            print("Successfully written " + fileName + " input file using MoorDyn v2")

    def unload_farm(self, fileName, depth=600):
        """Unloads a MoorPy system into a MoorDyn FAST.Farm input file"""

        # Settings
        Echo = False  # Echo input data to <RootName>.ech (flag)
        dtm = 0.001
        kbot = 3e6
        cbot = 3e5
        dtIC = 2
        TmaxIC = 10
        CdScaleIC = 4.0
        threshIC = 0.01

        # Line Type Properties
        BA = -1.0
        Can = 0.8
        Cat = 0.25
        Cdn = 2.0
        Cdt = 0.4

        # Point Properties
        CdA = 0
        Ca = 0

        # Line Properties
        flag = "-"

        # For when we want to specify the above settings using kwargs
        # for key in kwargs:
        # print('Using Custom value for', key,kwargs[key])

        # Outputs List
        Outputs = ["FairTen1", "FairTen2", "FairTen3"]

        print("attempting to write " + fileName + " for MoorDyn FAST.Farm input file")

        # Array to add strings to for each line of moordyn input file
        L = []

        # Input File Header
        L.append(f"---------------- MoorDyn FAST.Farm Input File ------------------")
        L.append("Generated by MoorDesign")

        L.append("{:5}    Echo      - echo the input file data (flag)".format(str(Echo).upper()))

        # Line Dictionary Header
        L.append("---------------------- LINE TYPES -----------------------------------------------------")
        L.append(f"{len(self.lineTypes)}    NTypes   - number of LineTypes")
        L.append("LineType         Diam     MassDen   EA        BA/-zeta     Can    Cat    Cdn    Cdt")
        L.append("   (-)           (m)      (kg/m)    (N)       (N-s/-)      (-)    (-)    (-)    (-)")

        # Line Dictionary Table
        for key in self.lineTypes:
            L.append(
                "{:<15} {:7.4f} {:8.3f} {:<10.1f} ".format(
                    key, self.lineTypes[key].d, self.lineTypes[key].mlin, self.lineTypes[key].EA
                )
                + "{:<7.1f} {:<7.1f} {:<7.2f} {:<7.1f} {:<7.2f}".format(BA, Can, Cat, Cdn, Cdt)
            )

        # Point Properties Header
        L.append("---------------------- POINTS ---------------------------------------------------------")
        L.append(f"{len(self.pointList)}    NConnects   - number of connections including anchors and fairleads")
        L.append("Node    Type         X        Y        Z        M      V      FX     FY     FZ    CdA    Ca ")
        L.append("(-)     (-)         (m)      (m)      (m)      (kg)   (m^3)  (kN)   (kN)   (kN)  (m^2)   ()")

        # Point Properties Table
        for point in self.pointList:
            point_pos = point.r  # Define point position in global reference frame
            if point.type == 1:  # point is Fized or attached (anch, body, fix)
                point_type = "Fixed"

                for body in self.bodyList:
                    if point.number in body.attachedP:
                        point_type = "Turbine" + str(body.number)
                        point_pos = body.rPointRel[body.attachedP.index(point.number)]

            if point.type == 0:  # point is Coupled Externally (con, free)
                point_type = "Connect"

            # if point.type == -1:       # I haven't seen a point.type = -1 yet

            L.append(
                "{:<4d} {:12} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(
                    point.number,
                    point_type,
                    point_pos[0],
                    point_pos[1],
                    point_pos[2],
                    point.m,
                    point.v,
                    point.fExt[0],
                    point.fExt[1],
                    point.fExt[2],
                    CdA,
                    Ca,
                )
            )

        # Line Properties Header
        L.append("---------------------- LINES -----------------------------------------------------")
        L.append(f"{len(self.lineList)}    NLines   - number of line objects")
        L.append("Line      LineType   UnstrLen  NumSegs  NodeAnch  NodeFair  Outputs  CtrlChan")
        L.append("(-)         (-)       (m)        (-)      (-)       (-)       (-)       (-)")

        for line in self.lineList:
            if line.L < 100:
                line.nNodes = 4
        # Line Properties Table
        # (Create a ix2 array of connection points from a list of m points)
        connection_points = np.empty([len(self.lineList), 2])  # First column is Anchor Node, second is Fairlead node
        for point_ind, point in enumerate(self.pointList, start=1):  # Loop through all the points
            for (line, line_pos) in zip(
                point.attached, point.attachedEndB
            ):  # Loop through all the lines #s connected to this point
                if line_pos == 0:  # If the A side of this line is connected to the point
                    connection_points[line - 1, 0] = point_ind  # Save as as an Anchor Node
                    # connection_points[line -1,0] = self.pointList.index(point) + 1
                elif line_pos == 1:  # If the B side of this line is connected to the point
                    connection_points[line - 1, 1] = point_ind  # Save as a Fairlead node
                    # connection_points[line -1,1] = self.pointList.index(point) + 1
        # Populate text
        for i in range(len(self.lineList)):
            L.append(
                "{:<4d} {:<15} {:8.3f} {:5d} {:8d} {:9d}        {}   {}".format(
                    self.lineList[i].number,
                    self.lineList[i].type,
                    self.lineList[i].L,
                    self.lineList[i].nNodes - 1,
                    int(connection_points[i, 0]),
                    int(connection_points[i, 1]),
                    flag,
                    0,
                )
            )

        # Solver Options Header
        L.append("---------------------- OPTIONS ----------------------------------------")

        # Solver Options
        L.append("{:<9.3f}dtM          - time step to use in mooring integration (s)".format(float(dtm)))
        L.append(
            "{:<9.1f}wtrdpth        - water depth (m) <<< must be specified for farm-level mooring".format(float(depth))
        )
        L.append("{:<9.1e}kbot         - bottom stiffness (Pa/m)".format(kbot))
        L.append("{:<9.1e}cbot         - bottom damping (Pa-s/m)".format(cbot))
        L.append("{:<9.1f}dtIC         - time interval for analyzing convergence during IC gen (s)".format(int(dtIC)))
        L.append("{:<9.1f}TmaxIC       - max time for ic gen (s)".format(int(TmaxIC)))
        L.append(
            "{:<9.1f}CdScaleIC    - factor by which to scale drag coefficients during dynamic relaxation (-)".format(
                int(CdScaleIC)
            )
        )
        L.append("{:<9.2f}threshIC     - threshold for IC convergence (-)".format(threshIC))

        # Outputs Header
        L.append("----------------------------OUTPUTS--------------------------------------------")

        # Outputs List
        for Output in Outputs:
            L.append(Output)
        L.append("END")

        # Final Line
        L.append("--------------------- need this line ------------------")

        # Write the text file
        with open(fileName, "w") as out:
            for x in range(len(L)):
                out.write(L[x])
                out.write("\n")

        print("Successfully written " + fileName + " MoorDyn FAST.Farm input file")

    def getDOFs(self):
        """returns updated nDOFs and nCpldDOFs if the body and point types ever change

        Returns
        -------
        nDOF : int
            number of free degrees of freedom based on body and point types.
        nCpldDOF : int
            number of coupled degrees of freedom based on body ad point types.

        """

        nDOF = 0
        nCpldDOF = 0

        for body in self.bodyList:
            if body.type == 0:
                nDOF += 6
            if body.type == -1:
                nCpldDOF += 6

        for point in self.pointList:
            if point.type == 0:
                nDOF += point.nDOF
            if point.type == -1:
                nCpldDOF += point.nDOF

        return nDOF, nCpldDOF

    def initialize(self, plots=0):
        """Initializes the mooring system objects to their initial positions

        Parameters
        ----------
        plots : bool, optional
            toggle to plot the system at initialization or not. The default is 0.

        Returns
        -------
        None.

        """

        self.nDOF, self.nCpldDOF = self.getDOFs()

        for body in self.bodyList:
            body.setPosition(body.r6)

        for point in self.pointList:
            point.setPosition(point.r)

        for line in self.lineList:
            line.staticSolve(
                profiles=1
            )  # flag to enable additional line outputs used for plotting, tension results, etc.

        for point in self.pointList:
            point.getForces()

        for body in self.bodyList:
            body.getForces()

        # draw initial mooring system if desired
        if plots == 1:
            self.plot(title="Mooring system at initialization")

    def transform(self, trans=[0, 0], rot=0, scale=[1, 1, 1]):
        """Applies scaling (can flip if negative), rotation, and translations (in that order) to the mooring system positions

        Parameters
        ----------
        trans : array, optional
            how far to shift the whole mooring system in x and y directions [m]. The default is [0,0].
        rot : float, optional
            how much to rotate the entire mooring system in the yaw direction [degrees]. The default is 0.
        scale : array, optional
            how much to scale the mooring system x and y dimensions by (relative) (NOT IMPLEMENTED). The default is [1,1] (unity).

        """

        if len(scale) == 3:
            scale = np.array(scale)
        else:
            raise ValueError("scale parameter must be of length 3")

        rotMat = rotationMatrix(0, 0, rot * np.pi / 180.0)
        tVec = np.array([trans[0], trans[1], 0.0])

        # little functions to transform r or r6 vectors in place
        def transform3(X):
            Xrot = np.matmul(rotMat, X * scale)
            X = Xrot + tVec
            return X

        def transform6(X):
            Xrot = np.matmul(rotMat, X[:3] * scale)
            # X = np.hstack([Xrot + tVec, X[3], X[4], X[5]+rot*np.pi/180.0])  # this line would be to also rotate the body, but that's double counting if we also rotate the fairlead positions
            X = np.hstack([Xrot + tVec, X[3], X[4], X[5]])
            return X

        # update positions of all objects
        for body in self.bodyList:
            body.r6 = transform6(body.r6)
        for point in self.pointList:
            point.r = transform3(point.r)
            for body in self.bodyList:
                if point.number in body.attachedP:
                    i = body.attachedP.index(point.number)
                    rRel = body.rPointRel[i]  # get relative location of point on body
                    body.rPointRel[i] = np.matmul(rotMat, rRel * scale)  # apply rotation to relative location

    def getPositions(self, DOFtype="free", dXvals=[]):
        """Returns a vector with the DOF values of objects in the System. DOFs can be of 'free' objects,
        'coupled' objects, or 'both'.

        Parameters
        ----------
        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.

        dXvals : list or array, optional
            If provided, a second vector filled with a value coresponding to each DOF type returned.
            If dXvals is size 2, it contains the values for translational and rotational DOFs, respectively.
            If dXvals is size 3, it expects the values for point DOFs, body translational DOFs, and body
            rotational DOFs, respectively.

        Returns
        -------
        X : array
            The DOF values - bodies, then points.

        dX : array, if dXvals is provided
            A vector with the corresponding dXvals value for each returned DOF value.

        """

        if DOFtype == "free":
            types = [0]
        elif DOFtype == "coupled":
            types = [-1]
        elif DOFtype == "both":
            types = [0, -1]
        else:
            raise ValueError("getPositions called with invalid DOFtype input. Must be free, coupled, or both")

        X = np.array([], dtype=np.float_)
        if len(dXvals) > 0:
            dX = []

        # gather DOFs from bodies
        for body in self.bodyList:
            if body.type in types:
                X = np.append(X, body.r6)
                if len(dXvals) > 0:
                    dX += 3 * [dXvals[-2]] + 3 * [dXvals[-1]]

        # gather DOFs from points
        for point in self.pointList:
            if point.type in types:
                X = np.append(
                    X, point.r[point.DOFs]
                )  # note: only including active DOFs of the point (may be less than all 3)
                if len(dXvals) > 0:
                    dX += point.nDOF * [dXvals[0]]

        if len(dXvals) > 0:
            dX = np.array(dX, dtype=np.float_)
            return X, dX
        else:
            return X

    def setPositions(self, X, DOFtype="free"):
        """Sets the DOF values of some objects in the System - 'free' objects,
        'coupled' objects, or 'both'.

        Parameters
        ----------
        X : array
            A list or array containing the values of all relevant DOFs -- for bodies first, then for points.
            If type is 'both', X provides the free DOFs followed by the coupled DOFs.
        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.

        """

        i = 0  # index used to split off input positions X for each free object

        # check to ensure len(X) matches nDOF, nCpldDOF or nDOF+nCpldDOF
        nDOF, nCpldDOF = self.getDOFs()
        if DOFtype == "free":
            types = [0]
            if len(X) != nDOF:
                raise ValueError("Inconsistency between the vector of positions and the free DOFs")
        elif DOFtype == "coupled":
            types = [-1]
            if len(X) != nCpldDOF:
                raise ValueError("Inconsistency between the vector of positions and the coupled DOFs")
        elif DOFtype == "both":
            types = [0, -1]
            if len(X) != nDOF + nCpldDOF:
                raise ValueError("Inconsistency between the vector of positions and the free and coupled DOFs")
        else:
            raise ValueError("setPositions called with invalid DOFtype input. Must be free, coupled, or both")

        # update positions of bodies
        for body in self.bodyList:
            if body.type in types:
                body.setPosition(X[i : i + 6])
                i += 6

        # update position of Points
        for point in self.pointList:
            if point.type in types:
                point.setPosition(
                    X[i : i + point.nDOF]
                )  # note: only including active DOFs of the point (may be less than all 3)
                i += point.nDOF

    def getForces(self, DOFtype="free", lines_only=False):
        """Returns a vector with the net forces/moments along DOFs in the System.
        DOFs can be of 'free' objects, 'coupled' objects, or 'both'.

        Parameters
        ----------
        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.
        lines_only : boolean, optional
            An option for calculating forces from just the mooring lines or not. The default is False,
            meaning forces will include weight, buoyancy, and any external loads assigned to bodies or points.

        Returns
        -------
        f : array
            The force values.

        """

        nDOF, nCpldDOF = self.getDOFs()

        # initialize force array based on DOFtype specified
        if DOFtype == "free":
            types = [0]
            f = np.zeros(nDOF)
        elif DOFtype == "coupled":
            types = [-1]
            f = np.zeros(nCpldDOF)
        elif DOFtype == "both":
            types = [0, -1]
            f = np.zeros(nDOF + nCpldDOF)
        else:
            raise ValueError("getForces called with invalid DOFtype input. Must be free, coupled, or both")

        i = 0  # index used in assigning outputs in output vector

        # gather net loads from bodies
        for body in self.bodyList:
            if body.type in types:
                f[i : i + 6] = body.getForces(lines_only=lines_only)
                i += 6

        # gather net loads from points
        for point in self.pointList:
            if point.type in types:
                f[i : i + point.nDOF] = point.getForces(
                    lines_only=lines_only
                )  # note: only including active DOFs of the point (may be less than all 3)
                i += point.nDOF

        return np.array(f)

    def getTensions(self):
        """Returns a vector with the line end tensions for all lines in the system.

        Returns
        -------
        T : array
            The tension values for all line ends A then all line ends B [N].

        """

        n = len(self.lineList)

        T = np.zeros(n * 2)

        for i in range(n):
            T[i] = self.lineList[i].TA
            T[n + i] = self.lineList[i].TB

        return T

    def mooringEq(self, X, DOFtype="free", lines_only=False, tol=0.001, profiles=0):
        """Error function used in solving static equilibrium by calculating the forces on free objects

        Parameters
        ----------
        X : array
            A list or array containing the values of all relevant DOFs -- for bodies first, then for points.
            If type is 'both', X provides the free DOFs followed by the coupled DOFs.
        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.
        tol : float, optional
            Tolerance to use in catenary calculations [m].

        Returns
        -------
        f : array
            The forces (and moments) on all applicable DOFs in the system. f is the same size as X.

        """

        if self.display > 3:
            print(f" mooringEq X={X}")

        # update DOFs
        self.setPositions(X, DOFtype=DOFtype)

        # solve profile and forces of all lines
        for line in self.lineList:
            line.staticSolve(tol=tol, profiles=profiles)

        # get reactions in DOFs
        f = self.getForces(DOFtype=DOFtype, lines_only=lines_only)

        if self.display > 3:
            print(f" mooringEq f={f}")

        return f

    """
    def solveEquilibrium(self, DOFtype="free", plots=0, rmsTol=10, maxIter=200):
        '''Solves for the static equilibrium of the system using the stiffness matrix, while updating positions of all free objects.

        Parameters
        ----------
        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.
        plots : int, optional
            Determines whether to plot the equilibrium process or not. The default is 0.
        rmsTol : float, optional
            The maximum RMS tolerance that the calculated forces and moments should be from 0. The default is 10 (units of N and/or N-m).
        maxIter : int, optional
            The maximum number of interations to try to solve for equilibrium. The default is 200.

        Raises
        ------
        SolveError
            If the system fails to solve for equilirbium in the given tolerance and iteration number

        Returns
        -------
        None.


        NOTE: this does not do anything intelligent with catenary tolerances.

        '''

        # Note: this approach appears to be reliable since it has some safeguards.

        # TO DO: make an option so these functions can find equilibrium of free AND coupled objects <<<<

        # fill in some arrays for each DOF
        '''
        X0 = []  # vector of current DOFs
        db = []  # step size bound

        for body in self.bodyList:
            if body.type==0:
                X0  += [*Body.r6 ]               # add free Body position and orientation to vector
                db  += [ 5, 5, 5, 0.3,0.3,0.3]   # put a strict bound on how quickly rotations can occur

        for point in self.pointList:
            if point.type==0:
                X0  += [*Point.r ]               # add free Point position to vector
                db  += [ 5., 5., 5.]                # specify maximum step size for point positions

        X0 = np.array(X0, dtype=np.float_)
        db = np.array(db, dtype=np.float_)
        '''
        X0, db = self.getPositions(DOFtype=DOFtype, dXvals=[5.0, 0.3])

        n = len(X0)

        # clear some arrays to log iteration progress
        self.freeDOFs.clear()    # clear stored list of positions, so it can be refilled for this solve process
        self.Xs = [] # for storing iterations from callback fn
        self.Fs = []


        if self.display > 1:
            print(f" solveEquilibrium called for {n} DOFs (DOFtype={DOFtype})")

        # do the main iteration, using Newton's method to find the zeros of the system's net force/moment functions
        for iter in range(maxIter):
            #print('X0 is', X0)

            # first get net force vector from current position
            F0 = self.mooringEq(X0, DOFtype=DOFtype)

            # if there are no DOFs to solve equilibrium, exit (doing this after calling mooringEq so that lines are solved)
            if n == 0:
                if self.display > 1:
                    print(f'  0 DOFs to equilibrate so exiting')
                break

            if self.display > 1:
                print(f'  i{iter}, X0 {X0}, F0 {F0}')

            # log current position and resulting force (note: this may need updating based on DOFtype
            self.freeDOFs.append(X0)
            self.Xs.append(X0)
            self.Fs.append(F0)

            # check for equilibrium, and finish if condition is met
            rmse = np.linalg.norm(F0)       # root mean square of all force/moment errors
            if np.linalg.norm(F0) < rmsTol:
                #print("Equilibrium solution completed after "+str(iter)+" iterations with RMS error of "+str(rmse))
                break
            elif iter==maxIter-1:

                if self.display > 1:
                    print("Failed to find equilibrium after "+str(iter)+" iterations, with RMS error of "+str(rmse))


                if self.display > 2:
                    for i in range(iter+1):
                        print(f" i={i}, RMSE={np.linalg.norm(self.Fs[i]):6.2e}, X={self.Xs[i]}")

                    K = self.getSystemStiffness(DOFtype=DOFtype)

                    print("===========================")
                    print("current system stiffness:")
                    print(K)

                    print("\n Current force ")
                    print(F0)


                raise SolveError(f"solveEquilibrium failed to find equilibrium after {iter} iterations, with RMS error of {rmse}")

            # get stiffness matrix
            K = self.getSystemStiffness(DOFtype=DOFtype)


            # adjust positions according to stiffness matrix to move toward net zero forces (but only a fraction of the way!)
            dX = np.matmul(np.linalg.inv(K), F0)   # calculate position adjustment according to Newton's method

            # >>> TODO: handle singular matrix error <<<

            #dXd= np.zeros(len(F0))   # a diagonal-only approach in each DOF in isolation


            for i in range(n):             # but limit adjustment to keep things under control

                #dX[i] = 0.5*dX[i] + 0.5*F0[i]/K[i,i] # alternative approach using diagonal stiffness only

                if dX[i] > db[i]:
                    dX[i] = db[i]
                elif dX[i] < -db[i]:
                    dX[i] = -db[i]

            #if iter == 6:
            #    breakpoint()

            X0 = X0 + 1.0*dX
            #print(X0)
            #print(self.mooringEq(X0))
            # features to add:
            # - reduce catenary error tolerance in proportion to how close we are to the solution
            # - also adapt stiffness solver perturbation size as we approach the solution



        if self.display > 1:
            print(F0)

        # show an animation of the equilibrium solve if applicable
        if plots > 0:
            self.animateSolution()
    """

    def solveEquilibrium3(
        self,
        DOFtype="free",
        plots=0,
        tol=0.05,
        rmsTol=0.0,
        maxIter=500,
        display=0,
        no_fail=False,
        finite_difference=False,
    ):
        self.solveEquilibrium(
            DOFtype=DOFtype,
            plots=plots,
            tol=tol,
            rmsTol=rmsTol,
            maxIter=maxIter,
            display=display,
            no_fail=no_fail,
            finite_difference=finite_difference,
        )

    def solveEquilibrium(
        self,
        DOFtype="free",
        plots=0,
        tol=0.05,
        rmsTol=0.0,
        maxIter=500,
        display=0,
        no_fail=False,
        finite_difference=False,
    ):
        """Solves for the static equilibrium of the system using the dsolve function approach in MoorSolve

        Parameters
        ----------

        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.
        plots : int, optional
            Determines whether to plot the equilibrium process or not. The default is 0.
        tol : float, optional
            The absolute tolerance on positions when calculating equilibriumk [m]
        maxIter : int, optional
            The maximum number of interations to try to solve for equilibrium. The default is 200.
        no_fail : bool
            False raises an error if convergence fails. True doesn't.
        finite_difference : bool
            False uses the analytical methods for system stiffness, true uses original finite difference methods.

        Raises
        ------
        SolveError
            If the system fails to solve for equilibrium in the given tolerance and iteration number

        Returns
        -------
        success : bool
            True/False whether converged to within tolerance.

        """

        self.DOFtype_solve_for = DOFtype
        # create arrays for the initial positions of the objects that need to find equilibrium, and the max step sizes
        X0, db = self.getPositions(DOFtype=DOFtype, dXvals=[100, 0.3])

        # temporary for backwards compatibility <<<<<<<<<<
        """
        if rmsTol != 0.0:
            tols = np.zeros(len(X0)) + rmsTol
            print("WHAT IS PASSING rmsTol in to solveEquilibrium?")
            breakpoint()
        elif np.isscalar(tol):
            if tol < 0:
                tols = -tol*db    # tolerances set relative to max step size
                lineTol = 0.05*tols[0]  # hard coding a tolerance for catenary calcs <<<<<<<<<<<
            else:
                tols = 1.0*tol   # normal case, passing dsovle(2) a scalar for tol
        else:
            tols = np.array(tol)  # assuming tolerances are passed in for each free variable
        """

        # store z indices for later seabed contact handling, and create vector of tolerances
        zInds = []
        tols = []
        i = 0  # index to go through system DOF vector
        if DOFtype == "free":
            types = [0]
        elif DOFtype == "coupled":
            types = [-1]
        elif DOFtype == "both":
            types = [0, -1]

        for body in self.bodyList:
            if body.type in types:
                zInds.append(i + 2)
                i += 6
                rtol = tol / max(
                    [np.linalg.norm(rpr) for rpr in body.rPointRel]
                )  # estimate appropriate body rotational tolerance based on attachment point radii
                tols += 3 * [tol] + 3 * [rtol]

        for point in self.pointList:
            if point.type in types:
                if 2 in point.DOFs:
                    zInds.append(i + point.DOFs.index(2))  # may need to check this bit <<<<
                i += point.nDOF  # note: only including active DOFs of the point (z may not be one of them)

                tols += point.nDOF * [tol]

        tols = np.array(tols)
        lineTol = 0.01 * tol
        n = len(X0)

        # if there are no DOFs, just update the mooring system force calculations then exit
        if n == 0:
            self.mooringEq(X0, DOFtype=DOFtype, tol=lineTol, profiles=1)
            if display > 0:
                print("There are no DOFs so solveEquilibrium is returning without adjustment.")
            return True

        # clear some arrays to log iteration progress
        self.freeDOFs.clear()  # clear stored list of positions, so it can be refilled for this solve process
        self.Xs = []  # for storing iterations from callback fn
        self.Es = []

        def eval_func_equil(X, args):

            Y = self.mooringEq(X, DOFtype=DOFtype, tol=lineTol)
            oths = dict(status=1)  # other outputs - returned as dict for easy use

            self.Xs.append(X)  # temporary
            self.Es.append(Y)

            return Y, oths, False

        def step_func_equil(X, args, Y, oths, Ytarget, err, tol_, iter, maxIter):

            # get stiffness matrix
            if finite_difference:
                K = self.getSystemStiffness(DOFtype=DOFtype)
            else:
                K = self.getSystemStiffnessA(DOFtype=DOFtype)

            # adjust positions according to stiffness matrix to move toward net zero forces

            # else:                                       # Normal case where all DOFs are adjusted
            try:  # try the normal solve first to avoid calculating the determinant every time
                if n > 20:  # if huge, count on the system being sparse and use a sparse solver
                    # with warnings.catch_warnings():
                    #    warnings.simplefilter("error", category=MatrixRankWarning)
                    Kcsr = csr_matrix(K)
                    dX = spsolve(Kcsr, Y)
                else:
                    dX = np.linalg.solve(K, Y)  # calculate position adjustment according to Newton's method
            except:

                if np.linalg.det(K) == 0.0:  # if the stiffness matrix is singular, we will modify the approach

                    # first try ignoring any DOFs with zero stiffness
                    indices = list(range(n))  # list of DOF indices that will remain active for this step
                    mask = [True] * n  # this is a mask to be applied to the array K indices

                    for i in range(n - 1, -1, -1):  # go through DOFs and flag any with zero stiffness for exclusion
                        if K[i, i] == 0:
                            mask[i] = False
                            del indices[i]

                    K_select = K[mask, :][:, mask]
                    Y_select = Y[mask]

                    dX = np.zeros(n)

                    if np.linalg.det(K_select) == 0.0:
                        dX_select = Y_select / np.diag(
                            K_select
                        )  # last-ditch attempt to get a step despite matrix singularity
                    else:
                        dX_select = np.linalg.solve(K_select, Y_select)
                    dX[indices] = dX_select  # assign active step DOFs, other DOFs will be zero

                else:
                    raise Exception("why did it fail even though det isn't zero?")

            # but limit adjustment magnitude (still preserve direction) to keep things under control
            overratio = np.max(np.abs(dX) / db)
            if overratio > 1.0:
                dX = dX / overratio
            """
            for i in range(n):
                if dX[i] > db[i]:
                    dX[i] = db[i]
                elif dX[i] < -db[i]:
                    dX[i] = -db[i]
            """

            # avoid oscillations about the seabed
            for i in zInds:
                if X[i] + dX[i] <= -self.depth or (X[i] <= -self.depth and Y[i] <= 0.0):
                    dX[i] = -self.depth - X[i]

            # if iter > 100:
            #    print(iter)
            #    breakpoint()

            return dX

        # Call dsolve function
        # X, Y, info = msolve.dsolve(eval_func_equil, X0, step_func=step_func_equil, tol=tol, maxIter=maxIter)
        # try:
        X, Y, info = dsolve2(
            eval_func_equil,
            X0,
            step_func=step_func_equil,
            tol=tols,
            a_max=1.4,
            maxIter=maxIter,
            display=display,
            dodamping=True,
        )  # <<<<
        # except Exception as e:
        #    raise MoorPyError(e)
        # Don't need to call Ytarget in dsolve because it's already set to be zeros

        if display > 1:
            print(X)
            print(Y)

        self.Xs2 = info["Xs"]  # List of positions as it finds equilibrium for every iteration
        self.Es2 = info[
            "Es"
        ]  # List of errors that the forces are away from 0, which in this case, is the same as the forces

        # Update equilibrium position at converged X values
        F = self.mooringEq(X, DOFtype=DOFtype, tol=lineTol, profiles=1)

        # Print statements if it ever reaches the maximum number of iterations
        if info["iter"] == maxIter - 1:
            if display > 1:

                if finite_difference:
                    K = self.getSystemStiffness(DOFtype=DOFtype)
                else:
                    K = self.getSystemStiffnessA(DOFtype=DOFtype)

                print("solveEquilibrium did not converge!")
                print(f"current system stiffness: {K}")
                print(f"\n Current force {F}")

                # plot the convergence failure
                self.plotEQsolve(iter=info["iter"] + 1)

            # breakpoint()

            if no_fail:
                return False
            else:
                raise SolveError(
                    f"solveEquilibrium failed to find equilibrium after {info['iter']} iterations, with residual forces of {F}"
                )

        # show an animation of the equilibrium solve if applicable
        if plots > 0:
            self.animateSolution()

        return True

    def plotEQsolve(self, iter=-1):
        """Plots trajectories of solving equilibrium from solveEquilibrium."""

        n = self.Xs2.shape[1]

        if n < 8:
            fig, ax = plt.subplots(2 * n, 1, sharex=True)
            for i in range(n):
                ax[i].plot(self.Xs2[:iter, i])
                ax[n + i].plot(self.Es2[:iter, i])
            ax[-1].set_xlabel("iteration")
        else:
            fig, ax = plt.subplots(n, 2, sharex=True)
            for i in range(n):
                ax[i, 0].plot(self.Xs2[:iter, i])
                ax[i, 1].plot(self.Es2[:iter, i])
            ax[-1, 0].set_xlabel("iteration, X")
            ax[-1, 1].set_xlabel("iteration, Error")
        plt.show()

    def getSystemStiffness(self, DOFtype="free", dx=0.1, dth=0.1, solveOption=1, lines_only=False, plots=0):
        """Calculates the stiffness matrix for all selected degrees of freedom of a mooring system
        whether free, coupled, or both (other DOFs are considered fixed).

        Parameters
        ----------

        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.
        dx : float, optional
            The change in displacement to be used for calculating the change in force. The default is 0.1.
        dth : float, optional
            The change in rotation to be used for calculating the change in force. The default is 0.1.
        solveOption : boolean, optional
            Indicator of which solve option to use. The default is 1.
        plots : boolean, optional
            Determines whether the stiffness calculation process is plotted and/or animated or not. The default is 0.

        Raises
        ------
        ValueError
            If the solveOption is not a 1 or 0

        Returns
        -------
        K : matrix
            nDOF x nDOF stiffness matrix of the system

        """

        if not DOFtype in ["free", "coupled", "both"]:
            raise ValueError("setPositions called with invalid DOFtype input. Must be free, coupled, or both")

        if self.display > 2:
            print("Getting mooring system stiffness matrix...")

        lineTol = 0.05 * dx  # manually specify an adaptive catenary solve tolerance <<<<

        # ------------------ get the positions to linearize about -----------------------

        X1, dX = self.getPositions(DOFtype=DOFtype, dXvals=[dx, dth])

        n = len(X1)

        # F1 = self.getForces(DOFtype=DOFtype)                # get mooring forces/moments about linearization point
        F1 = self.mooringEq(
            X1, DOFtype=DOFtype, lines_only=lines_only, tol=lineTol
        )  # get mooring forces/moments about linearization point (call this ensures point forces reflect current reported positions (even if on seabed)

        K = np.zeros([n, n])  # allocate stiffness matrix

        if plots > 0:
            self.freeDOFs.clear()  # clear the positions history to refill if animating this process  <<<< needs updating for DOFtype

        # ------------------------- perform linearization --------------------------------

        if solveOption == 0:  # ::: forward difference approach :::

            for i in range(n):  # loop through each DOF

                X2 = np.array(X1, dtype=np.float_)
                X2[i] += dX[i]  # perturb positions by dx in each DOF in turn
                F2p = self.mooringEq(
                    X2, DOFtype=DOFtype, lines_only=lines_only, tol=lineTol
                )  # system net force/moment vector from positive perturbation

                if self.display > 2:
                    print(F2p)
                if plots > 0:
                    self.freeDOFs.append(X2)

                K[:, i] = -(F2p - F1) / dX[i]  # take finite difference of force w.r.t perturbation

        elif solveOption == 1:  # ::: adaptive central difference approach :::

            nTries = 3  # number of refinements to allow -1

            for i in range(n):  # loop through each DOF

                dXi = 1.0 * dX[i]

                # potentially iterate with smaller step sizes if we're at a taut-slack transition (but don't get too small, or else numerical errors)
                for j in range(nTries):
                    if self.display > 2:
                        print(" ")
                    X2 = np.array(X1, dtype=np.float_)
                    X2[i] += dXi  # perturb positions by dx in each DOF in turn
                    F2p = self.mooringEq(
                        X2, DOFtype=DOFtype, lines_only=lines_only, tol=lineTol
                    )  # system net force/moment vector from positive perturbation
                    if self.display > 2:
                        printVec(self.pointList[0].r)
                        printVec(self.lineList[0].rB)
                    if plots > 0:
                        self.freeDOFs.append(X2.copy())

                    X2[i] -= 2.0 * dXi
                    F2m = self.mooringEq(
                        X2, DOFtype=DOFtype, lines_only=lines_only, tol=lineTol
                    )  # system net force/moment vector from negative perturbation
                    if self.display > 2:
                        printVec(self.pointList[0].r)
                        printVec(self.lineList[0].rB)
                    if plots > 0:
                        self.freeDOFs.append(X2.copy())

                    if self.display > 2:
                        print(
                            f"i={i}, j={j} and dXi={dXi}.   F2m, F1, and F2p are {F2m[i]:6.2f} {F1[i]:6.2f} {F2p[i]:6.2f}"
                        )
                        printVec(F2p - F1)
                        printVec(F1 - F2m)
                        printVec(F2m)
                        printVec(F1)
                        printVec(F2p)

                    # Break if the force is zero or the change in the first derivative is small
                    if abs(F1[i]) == 0 or abs(F2m[i] - 2.0 * F1[i] + F2p[i]) < 0.1 * np.abs(
                        F1[i]
                    ):  # note: the 0.1 is the adjustable tolerance
                        break
                    elif j == nTries - 1:
                        if self.display > 2:
                            print("giving up on refinement")
                    else:
                        # Otherwise, we're at a tricky point and should stay in the loop to keep narrowing the step size
                        # until the derivatives agree better. Decrease the step size by 10X.
                        dXi = 0.1 * dXi

                K[:, i] = -0.5 * (F2p - F2m) / dXi  # take finite difference of force w.r.t perturbation

        else:
            raise ValueError("getSystemStiffness was called with an invalid solveOption (only 0 and 1 are supported)")

        # ----------------- restore the system back to previous positions ------------------
        self.setPositions(X1, DOFtype=DOFtype)

        # show an animation of the stiffness perturbations if applicable
        if plots > 0:
            self.animateSolution()

        return K

    def getCoupledStiffness(self, dx=0.1, dth=0.1, solveOption=1, lines_only=False, tensions=False, nTries=3, plots=0):
        """Calculates the stiffness matrix for coupled degrees of freedom of a mooring system
        with free uncoupled degrees of freedom equilibrated.

        Parameters
        ----------
        dx : float, optional
            The change in displacement to be used for calculating the change in force. The default is 0.1.
        dth : float, optional
            The change in rotation to be used for calculating the change in force. The default is 0.1.
        solveOption : boolean, optional
            Indicator of which solve option to use. The default is 1.
        plots : boolean, optional
            Determines whether the stiffness calculation process is plotted and/or animated or not. The default is 0.
        lines_only : boolean
            Whether to consider only line forces and ignore body/point properties.
        tensions : boolean
            Whether to also compute and return mooring line tension jacobians

        Raises
        ------
        ValueError
            If the solveOption is not a 1 or 0

        Returns
        -------
        K : matrix
            nCpldDOF x nCpldDOF stiffness matrix of the system

        """
        self.nDOF, self.nCpldDOF = self.getDOFs()

        if self.display > 2:
            print("Getting mooring system stiffness matrix...")

        lineTol = 0.05 * dx  # manually specify an adaptive catenary solve tolerance <<<<
        eqTol = 0.05 * dx  # manually specify an adaptive tolerance for when calling solveEquilibrium

        # ------------------ get the positions to linearize about -----------------------

        # get the positions about which the system is linearized, and an array containting
        # the perturbation size in each coupled DOF of the system
        X1, dX = self.getPositions(DOFtype="coupled", dXvals=[dx, dth])

        self.solveEquilibrium(tol=eqTol)  # let the system settle into equilibrium

        F1 = self.getForces(
            DOFtype="coupled", lines_only=lines_only
        )  # get mooring forces/moments about linearization point
        K = np.zeros([self.nCpldDOF, self.nCpldDOF])  # allocate stiffness matrix

        if tensions:
            T1 = self.getTensions()
            J = np.zeros([len(T1), self.nCpldDOF])  # allocate Jacobian of tensions w.r.t. coupled DOFs

        if plots > 0:
            self.cpldDOFs.clear()  # clear the positions history to refill if animating this process

        # ------------------------- perform linearization --------------------------------

        if solveOption == 0:  # ::: forward difference approach :::

            for i in range(self.nCpldDOF):  # loop through each DOF

                X2 = np.array(X1, dtype=np.float_)
                X2[i] += dX[i]  # perturb positions by dx in each DOF in turn
                self.setPositions(X2, DOFtype="coupled")  # set the perturbed coupled DOFs
                self.solveEquilibrium()  # let the system settle into equilibrium
                F2p = self.getForces(
                    DOFtype="coupled", lines_only=lines_only
                )  # get resulting coupled DOF net force/moment response
                if tensions:
                    T2p = self.getTensions()

                if self.display > 2:
                    print(F2p)
                if plots > 0:
                    self.cpldDOFs.append(X2)

                K[:, i] = -(F2p - F1) / dX[i]  # take finite difference of force w.r.t perturbation
                if tensions:
                    J[:, i] = (T2p - T1) / dX[i]

        elif solveOption == 1:  # ::: adaptive central difference approach :::

            # nTries = 1  # number of refinements to allow -1

            for i in range(self.nCpldDOF):  # loop through each DOF

                dXi = 1.0 * dX[i]
                # print(f'__________ nCpldDOF = {i+1} __________')
                # potentially iterate with smaller step sizes if we're at a taut-slack transition (but don't get too small, or else numerical errors)
                for j in range(nTries):

                    # print(f'-------- nTries = {j+1} --------')
                    X2 = np.array(X1, dtype=np.float_)
                    X2[i] += dXi  # perturb positions by dx in each DOF in turn
                    self.setPositions(X2, DOFtype="coupled")  # set the perturbed coupled DOFs
                    # print(f'solving equilibrium {i+1}+_{self.nCpldDOF}')
                    self.solveEquilibrium(tol=eqTol)  # let the system settle into equilibrium
                    F2p = self.getForces(
                        DOFtype="coupled", lines_only=lines_only
                    )  # get resulting coupled DOF net force/moment response
                    if tensions:
                        T2p = self.getTensions()

                    if plots > 0:
                        self.cpldDOFs.append(X2.copy())

                    X2[i] -= 2.0 * dXi  # now perturb from original to -dx
                    self.setPositions(X2, DOFtype="coupled")  # set the perturbed coupled DOFs
                    # print(f'solving equilibrium {i+1}-_{self.nCpldDOF}')
                    self.solveEquilibrium(tol=eqTol)  # let the system settle into equilibrium
                    F2m = self.getForces(
                        DOFtype="coupled", lines_only=lines_only
                    )  # get resulting coupled DOF net force/moment response
                    if tensions:
                        T2m = self.getTensions()

                    if plots > 0:
                        self.cpldDOFs.append(X2.copy())

                    if j == 0:
                        F2pi = F2p
                        F2mi = F2m

                    if self.display > 2:
                        print(
                            f"j = {j}  and dXi = {dXi}.   F2m, F1, and F2p are {F2m[i]:6.2f} {F1[i]:6.2f} {F2p[i]:6.2f}"
                        )
                        print(abs(F2m[i] - 2.0 * F1[i] + F2p[i]))
                        # print(0.1*np.abs(F1[i]))
                        print(0.1 * np.abs(F2pi[i] - F2mi[i]))
                        print(abs(F1[i]) < 1)
                        # print(abs(F2m[i]-2.0*F1[i]+F2p[i]) < 0.1*np.abs(F1[i]))
                        print(abs(F2m[i] - 2.0 * F1[i] + F2p[i]) < 0.1 * np.abs(F2pi[i] - F2mi[i]))

                    # Break if the force is zero or the change in the first derivative is small
                    # if abs(F1[i]) < 1 or abs(F2m[i]-2.0*F1[i]+F2p[i]) < 0.1*np.abs(F1[i]):  # note: the 0.1 is the adjustable tolerance
                    if abs(F1[i]) < 1 or abs(F2m[i] - 2.0 * F1[i] + F2p[i]) < 0.1 * np.abs(
                        F2pi[i] - F2mi[i]
                    ):  # note: the 0.1 is the adjustable tolerance
                        break
                    elif j == nTries - 1:
                        if self.display > 2:
                            print("giving up on refinement")
                    else:
                        # Otherwise, we're at a tricky point and should stay in the loop to keep narrowing the step size
                        # untill the derivatives agree better. Decrease the step size by 10X.
                        dXi = 0.1 * dXi

                K[:, i] = -0.5 * (F2p - F2m) / dXi  # take finite difference of force w.r.t perturbation
                if tensions:
                    J[:, i] = 0.5 * (T2p - T2m) / dX[i]

        else:
            raise ValueError("getSystemStiffness was called with an invalid solveOption (only 0 and 1 are supported)")

        # ----------------- restore the system back to previous positions ------------------
        self.mooringEq(X1, DOFtype="coupled", tol=lineTol)
        self.solveEquilibrium(tol=eqTol)

        # show an animation of the stiffness perturbations if applicable
        if plots > 0:
            self.animateSolution()

        if tensions:
            return K, J
        else:
            return K

    def getSystemStiffnessA(self, DOFtype="free", lines_only=False, rho=1025, g=9.81):
        """A method to calculate the system's stiffness matrix based entirely on analytic gradients from catenary

        Parameters
        ----------
        DOFtype : string, optional
            specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is "free".
        hydro : bool, optional   <<<<< replaced this with lines_only for now for consistency with others. Could reconsider (for all functions)
            specifies whether to include hydrostatic stiffness components of bodies. The default is 0 (to not include).
        rho : float, optional
            DESCRIPTION. The default is 1025.
        g : float, optional
            DESCRIPTION. The default is 9.81.

        Raises
        ------
        ValueError
            Raised if an invalid DOFtype is used.

        Returns
        -------
        K : matrix
            complete analytic stiffness matrix of the system for the specified DOFs.

        """

        # note: This is missing some pieces, and needs to check more.
        # So far this seems to not capture yaw stiffness for non-bridle configs...
        # it would require proper use of chain rule for the derivatives

        # find the total number of free and coupled DOFs in case any object types changed
        self.nDOF, self.nCpldDOF = self.getDOFs()

        # self.solveEquilibrium()   # should we make sure the system is in equilibrium?

        # allocate stiffness matrix according to the DOFtype specified
        if DOFtype == "free":
            K = np.zeros([self.nDOF, self.nDOF])
            d = [0]
        elif DOFtype == "coupled":
            K = np.zeros([self.nCpldDOF, self.nCpldDOF])
            d = [-1]
        elif DOFtype == "both":
            K = np.zeros([self.nDOF + self.nCpldDOF, self.nDOF + self.nCpldDOF])
            d = [0, -1]
        else:
            raise ValueError("getSystemStiffnessA called with invalid DOFtype input. Must be free, coupled, or both")

        # The following will go through and get the lower-triangular stiffness terms,
        # calculated as the force/moment on Body/Point 2 from translation/rotation of Body/Point 1.

        # go through DOFs, looking for lines that couple to anchors or other DOFs

        i = 0  # start counting number of DOFs at zero

        # go through each movable body in the system
        for body1 in self.bodyList:
            if (
                body1.type in d
            ):  # >>>> when DOFtype==both, this approach gives different indexing than what is in setPositions/getForces and getSystemStiffness <<<<<

                # i = (body1.number-1)*6      # start counting index for body DOFs based on body number to keep indexing consistent

                # get body's self-stiffness matrix (now only cross-coupling terms will be handled on a line-by-line basis)
                K6 = body1.getStiffnessA(lines_only=lines_only)
                K[i : i + 6, i : i + 6] += K6

                # go through each attached point
                for pointID1, rPointRel1 in zip(body1.attachedP, body1.rPointRel):
                    point1 = self.pointList[pointID1 - 1]

                    r1 = rotatePosition(
                        rPointRel1, body1.r6[3:]
                    )  # relative position of Point about body ref point in unrotated reference frame
                    H1 = getH(r1)  # produce alternator matrix of current point's relative position to current body

                    for (
                        lineID
                    ) in (
                        point1.attached
                    ):  # go through each attached line to the Point, looking for when its other end is attached to something that moves

                        endFound = 0  # simple flag to indicate when the other end's attachment has been found
                        j = (
                            i + 6
                        )  # first index of the DOFs this line is attached to. Start it off at the next spot after body1's DOFs

                        # get cross-coupling stiffness of line: force on end attached to body1 due to motion of other end
                        if point1.attachedEndB == 1:
                            KB = self.lineList[lineID - 1].KAB
                        else:
                            KB = self.lineList[lineID - 1].KAB.T
                        """
                        KA, KB = self.lineList[lineID-1].getStiffnessMatrix()
                        # flip sign for coupling
                        if point1.attachedEndB == 1:    # assuming convention of end A is attached to the first point, so if not,
                            KB = -KA                    # swap matrices of ends A and B
                        else:
                            KB = -KB
                        """
                        # look through Bodies further on in the list (coupling with earlier Bodies will already have been taken care of)
                        for body2 in self.bodyList[self.bodyList.index(body1) + 1 :]:
                            if body2.type in d:

                                # go through each attached Point
                                for pointID2, rPointRel2 in zip(body2.attachedP, body2.rPointRel):
                                    point2 = self.pointList[pointID2 - 1]

                                    if (
                                        lineID in point2.attached
                                    ):  # if the line is also attached to this Point2 in Body2

                                        # following are analagous to what's in functions getH and translateMatrix3to6 except for cross coupling between two bodies
                                        r2 = rotatePosition(
                                            rPointRel2, body2.r6[3:]
                                        )  # relative position of Point about body ref point in unrotated reference frame
                                        H2 = getH(r2)

                                        # loads on body1 due to motions of body2
                                        K66 = np.block(
                                            [
                                                [KB, np.matmul(KB, H1)],
                                                [np.matmul(H2.T, KB), np.matmul(np.matmul(H2, KB), H1.T)],
                                            ]
                                        )

                                        K[i : i + 6, j : j + 6] += K66
                                        K[j : j + 6, i : i + 6] += K66.T  # mirror

                                        # note: the additional rotational stiffness due to change in moment arm does not apply to this cross-coupling case

                                        endFound = 1  # signal that the line has been handled so we can move on to the next thing
                                        break

                                j += 6  # if this body has DOFs we're considering, then count them

                        # look through free Points
                        if endFound == 0:  #  if the end of this line hasn't already been found attached to a body
                            for point2 in self.pointList:
                                if point2.type in d:  # if it's a free point and
                                    if lineID in point2.attached:  # the line is also attached to it

                                        # only add up one off-diagonal sub-matrix for now, then we'll mirror at the end
                                        # K[i  :i+3, j:j+3] += K3
                                        # K[i+3:i+6, j:j+3] += np.matmul(H1.T, K3)
                                        K63 = np.vstack([KB, np.matmul(H1.T, KB)])
                                        K63 = K63[
                                            :, point2.DOFs
                                        ]  # trim the matrix to only use the enabled DOFs of each point

                                        K[i : i + 6, j : j + point2.nDOF] += K63
                                        K[j : j + point2.nDOF, i : i + 6] += K63.T  # mirror

                                        break

                                    j += point2.nDOF  # if this point has DOFs we're considering, then count them

                                # note: No cross-coupling with fixed points. The body's own stiffness matrix is now calculated at the start.

                i += 6  # moving along to the next body...

        # go through each movable point in the system
        for point in self.pointList:
            if point.type in d:

                n = point.nDOF

                # >>> TODO: handle case of free end point resting on seabed <<<

                # get point's self-stiffness matrix
                K1 = point.getStiffnessA(lines_only=lines_only)
                K[i : i + n, i : i + n] += K1

                # go through attached lines and add cross-coupling terms
                for lineID in point.attached:

                    j = i + n

                    # go through movable points to see if one is attached
                    for point2 in self.pointList[self.pointList.index(point) + 1 :]:
                        if point2.type in d:
                            if lineID in point2.attached:  # if this point is at the other end of the line
                                """
                                KA, KB = self.lineList[lineID-1].getStiffnessMatrix()       # get full 3x3 stiffness matrix of the line that attaches them
                                # flip sign for coupling
                                if point.attachedEndB == 1:     # assuming convention of end A is attached to the first point, so if not,
                                    KB = -KA                    # swap matrices of ends A and B
                                else:
                                    KB = -KB
                                """
                                # get cross-coupling stiffness of line: force on end attached to point1 due to motion of other end
                                if point.attachedEndB == 1:
                                    KB = self.lineList[lineID - 1].KAB
                                else:
                                    KB = self.lineList[lineID - 1].KAB.T

                                KB = KB[point.DOFs, :][
                                    :, point2.DOFs
                                ]  # trim the matrix to only use the enabled DOFs of each point

                                K[i : i + n, j : j + point2.nDOF] += KB
                                K[j : j + point2.nDOF, i : i + n] += KB.T  # mirror

                            j += point2.nDOF  # if this point has DOFs we're considering, then count them

                i += n

        return K

    def plot(self, ax=None, bounds="default", rbound=0, color=None, **kwargs):
        """Plots the mooring system objects in their current positions
        Parameters
        ----------
        bounds : string, optional
            signifier for the type of bounds desired in the plot. The default is "default".
        ax : axes, optional
            Plot on an existing set of axes
        color : string, optional
            Some way to control the color of the plot ... TBD <<<
        hidebox : bool, optional
            If true, hides the axes and box so just the plotted lines are visible.
        rbound : float, optional
            A bound to be placed on each axis of the plot. If 0, the bounds will be the max values on each axis. The default is 0.
        title : string, optional
            A title of the plot. The default is "".
        linelabels : bool, optional
            Adds line numbers to plot in text. Default is False.
        pointlabels: bool, optional
            Adds point numbers to plot in text. Default is False.
        endpoints: bool, optional
            Adds visible end points to lines. Default is False.
        bathymetry: bool, optional
            Creates a bathymetry map of the seabed based on an input file. Default is False.

        Returns
        -------
        fig : figure object
            To hold the axes of the plot
        ax: axis object
            To hold the points and drawing of the plot

        """

        hidebox = kwargs.get("hidebox", False)  # toggles whether to show the axes or not
        title = kwargs.get("title", "")  # optional title for the plot
        time = kwargs.get("time", 0)  # the time in seconds of when you want to plot
        linelabels = kwargs.get("linelabels", False)  # toggle to include line number labels in the plot
        pointlabels = kwargs.get("pointlabels", False)  # toggle to include point number labels in the plot
        endpoints = kwargs.get("endpoints", False)  # toggle to include the line end points in the plot
        bathymetry = kwargs.get(
            "bathymetry", False
        )  # toggle (and string) to include bathymetry or not. Can do full map based on text file, or simple squares
        water = kwargs.get("water", 0)  # option to plot water surface (if > 0)
        cmap_bath = kwargs.get("cmap", "ocean")  # matplotlib colormap specification
        alpha = kwargs.get("opacity", 1.0)  # the transparency of the bathymetry plot_surface
        draw_body = kwargs.get("draw_body", True)  # toggle to draw the Bodies or not
        shadow = kwargs.get("shadow", True)  # toggle to draw the mooring line shadows or not
        rang = kwargs.get(
            "rang", "hold"
        )  # colorbar range: if range not used, set it as a placeholder, it will get adjusted later
        cbar_bath = kwargs.get("cbar_bath", False)  # toggle to include a colorbar for a plot or not
        cbar_bath_size = kwargs.get(
            "colorbar_size", 1.0
        )  # the scale of the colorbar. Not the same as aspect. Aspect adjusts proportions
        colortension = kwargs.get(
            "colortension", False
        )  # toggle to draw the mooring lines in colors based on node tensions
        cmap_tension = kwargs.get("cmap_tension", "rainbow")  # the type of color spectrum desired for colortensions
        cbar_tension = kwargs.get(
            "cbar_tension", False
        )  # toggle to include a colorbar of the tensions when colortension=True

        # sort out bounds
        xs = []
        ys = []
        zs = [0, -self.depth]

        for point in self.pointList:
            xs.append(point.r[0])
            ys.append(point.r[1])
            zs.append(point.r[2])

        # if axes not passed in, make a new figure
        if ax == None:
            fig = plt.figure()
            # fig = plt.figure(figsize=(20/2.54,12/2.54), dpi=300)
            ax = plt.axes(projection="3d")
        else:
            fig = ax.get_figure()

        # set bounds
        if rbound == 0:
            rbound = max([max(xs), max(ys), -min(xs), -min(ys)])  # this is the most extreme coordinate

        if bounds == "default":
            ax.set_zlim([-self.depth, 0])
        elif bounds == "rbound":
            ax.set_xlim([-rbound, rbound])
            ax.set_ylim([-rbound, rbound])
            ax.set_zlim([-rbound, rbound])
        elif bounds == "mooring":
            ax.set_xlim([-rbound, 0])
            ax.set_ylim([-rbound / 2, rbound / 2])
            ax.set_zlim([-self.depth, 0])

        # draw things
        if draw_body:
            for body in self.bodyList:
                body.draw(ax)

        j = 0
        for line in self.lineList:
            j = j + 1
            if color == None and isinstance(line.type, str):
                if "chain" in line.type:
                    line.drawLine(
                        time,
                        ax,
                        color=[0.1, 0, 0],
                        endpoints=endpoints,
                        shadow=shadow,
                        colortension=colortension,
                        cmap_tension=cmap_tension,
                    )
                elif "rope" in line.type or "polyester" in line.type:
                    line.drawLine(
                        time,
                        ax,
                        color=[0.3, 0.5, 0.5],
                        endpoints=endpoints,
                        shadow=shadow,
                        colortension=colortension,
                        cmap_tension=cmap_tension,
                    )
                else:
                    line.drawLine(
                        time,
                        ax,
                        color=[0.2, 0.2, 0.2],
                        endpoints=endpoints,
                        shadow=shadow,
                        colortension=colortension,
                        cmap_tension=cmap_tension,
                    )
            else:
                line.drawLine(
                    time,
                    ax,
                    color=color,
                    endpoints=endpoints,
                    shadow=shadow,
                    colortension=colortension,
                    cmap_tension=cmap_tension,
                )

            # Add line labels
            if linelabels == True:
                ax.text((line.rA[0] + line.rB[0]) / 2, (line.rA[1] + line.rB[1]) / 2, (line.rA[2] + line.rB[2]) / 2, j)

        if cbar_tension:
            maxten = max([max(line.getLineTens()) for line in self.lineList])  # find the max tension in the System
            minten = min([min(line.getLineTens()) for line in self.lineList])  # find the min tension in the System
            bounds = range(int(minten), int(maxten), int((maxten - minten) / 256))
            norm = mpl.colors.BoundaryNorm(
                bounds, 256
            )  # set the bounds in a norm object, with 256 being the length of all colorbar strings
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_tension), label="Tension (N)")  # add the colorbar
            fig.tight_layout()

        # Add point labels
        i = 0
        for point in self.pointList:
            points = []
            i = i + 1
            if pointlabels == True:
                ax.text(point.r[0], point.r[1], point.r[2], i, c="r")

            if bathymetry == True:  # if bathymetry is true, then make squares at each anchor point
                if point.attachedEndB[0] == 0 and point.r[2] < -400:
                    points.append([point.r[0] + 250, point.r[1] + 250, point.r[2]])
                    points.append([point.r[0] + 250, point.r[1] - 250, point.r[2]])
                    points.append([point.r[0] - 250, point.r[1] - 250, point.r[2]])
                    points.append([point.r[0] - 250, point.r[1] + 250, point.r[2]])

                    Z = np.array(points)
                    verts = [[Z[0], Z[1], Z[2], Z[3]]]
                    ax.add_collection3d(
                        Poly3DCollection(verts, facecolors="limegreen", linewidths=1, edgecolors="g", alpha=alpha)
                    )

        if isinstance(bathymetry, str):  # or, if it's a string, load in the bathymetry file

            # parse through the MoorDyn bathymetry file
            bathGrid_Xs, bathGrid_Ys, bathGrid = self.readBathymetryFile(bathymetry)
            if rang == "hold":
                rang = (np.min(-bathGrid), np.max(-bathGrid))
            """
            # First method: plot nice 2D squares using Poly3DCollection
            nX = len(bathGrid_Xs)
            nY = len(bathGrid_Ys)
            # store a list of points in the grid
            Z = [[bathGrid_Xs[j],bathGrid_Ys[i],-bathGrid[i,j]] for i in range(nY) for j in range(nX)]
            # plot every square in the grid (e.g. 16 point grid yields 9 squares)
            verts = []
            for i in range(nY-1):
                for j in range(nX-1):
                    verts.append([Z[j+nX*i],Z[(j+1)+nX*i],Z[(j+1)+nX*(i+1)],Z[j+nX*(i+1)]])
                    ax.add_collection3d(Poly3DCollection(verts, facecolors='limegreen', linewidths=1, edgecolors='g', alpha=0.5))
                    verts = []
            """
            # Second method: plot a 3D surface, plot_surface
            X, Y = np.meshgrid(bathGrid_Xs, bathGrid_Ys)
            bath = ax.plot_surface(X, Y, -bathGrid, cmap=cmap_bath, vmin=rang[0], vmax=rang[1], alpha=alpha)

            if (
                cbar_bath_size != 1.0
            ):  # make sure the colorbar is turned on just in case it isn't when the other colorbar inputs are used
                cbar_bath = True
            if cbar_bath:
                fig.colorbar(bath, shrink=cbar_bath_size, label="depth (m)")

        # draw water surface if requested
        # if water > 0:

        fig.suptitle(title)

        set_axes_equal(ax)

        ax.set_zticks([-self.depth, 0])  # set z ticks to just 0 and seabed

        if hidebox:
            ax.axis("off")

        return fig, ax  # return the figure and axis object in case it will be used later to update the plot

    def plot2d(self, Xuvec=[1, 0, 0], Yuvec=[0, 0, 1], ax=None, color=None, **kwargs):
        """Makes a 2D plot of the mooring system objects in their current positions

        Parameters
        ----------
        Xuvec : list, optional
            plane at which the x-axis is desired. The default is [1,0,0].
        Yuvec : lsit, optional
            plane at which the y-axis is desired. The default is [0,0,1].
        ax : axes, optional
            Plot on an existing set of axes
        color : string, optional
            Some way to control the color of the plot ... TBD <<<
        title : string, optional
            A title of the plot. The default is "".

        Returns
        -------
        fig : figure object
            To hold the axes of the plot
        ax: axis object
            To hold the points and drawing of the plot

        """

        title = kwargs.get("title", "")  # optional title for the plot
        time = kwargs.get("time", 0)  # the time in seconds of when you want to plot
        linelabels = kwargs.get("linelabels", False)  # toggle to include line number labels in the plot
        pointlabels = kwargs.get("pointlabels", False)  # toggle to include point number labels in the plot
        bathymetry = kwargs.get(
            "bathymetry", False
        )  # toggle (and string) to include bathymetry contours or not based on text file
        draw_body = kwargs.get("draw_body", False)  # toggle to draw the Bodies or not
        cmap_bath = kwargs.get("cmap_bath", "ocean")  # matplotlib colormap specification
        alpha = kwargs.get("opacity", 1.0)  # the transparency of the bathymetry plot_surface
        levels = kwargs.get("levels", 7)  # the number (or array) of levels in the contour plot
        rang = kwargs.get(
            "rang", "hold"
        )  # colorbar range: if range not used, set it as a placeholder, it will get adjusted later
        cbar_bath = kwargs.get("colorbar", False)  # toggle to include a colorbar for a plot or not
        cbar_bath_aspect = kwargs.get(
            "cbar_bath_aspect", 20
        )  # the proportion of the colorbar. Default is 20 height x 1 width
        cbar_bath_ticks = kwargs.get(
            "cbar_bath_ticks", None
        )  # the desired tick labels on the colorbar (can be an array)
        colortension = kwargs.get(
            "colortension", False
        )  # toggle to draw the mooring lines in colors based on node tensions
        cmap_tension = kwargs.get("cmap_tension", "rainbow")  # the type of color spectrum desired for colortensions
        cbar_tension = kwargs.get(
            "cbar_tension", False
        )  # toggle to include a colorbar of the tensions when colortension=True

        # if axes not passed in, make a new figure
        if ax == None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()  # will this work like this? <<<

        if draw_body:
            for body in self.bodyList:
                # body.draw(ax)
                plt.plot(body.r6[0], body.r6[1], "ko", markersize=5)

        j = 0
        for line in self.lineList:
            j = j + 1
            if color == None and isinstance(line.type, str):
                if "chain" in line.type:
                    line.drawLine2d(
                        time,
                        ax,
                        color=[0.1, 0, 0],
                        Xuvec=Xuvec,
                        Yuvec=Yuvec,
                        colortension=colortension,
                        cmap=cmap_tension,
                    )
                elif "rope" in line.type or "polyester" in line.type:
                    line.drawLine2d(
                        time,
                        ax,
                        color=[0.3, 0.5, 0.5],
                        Xuvec=Xuvec,
                        Yuvec=Yuvec,
                        colortension=colortension,
                        cmap=cmap_tension,
                    )
                else:
                    line.drawLine2d(
                        time,
                        ax,
                        color=[0.3, 0.3, 0.3],
                        Xuvec=Xuvec,
                        Yuvec=Yuvec,
                        colortension=colortension,
                        cmap=cmap_tension,
                    )
            else:
                line.drawLine2d(
                    time, ax, color=color, Xuvec=Xuvec, Yuvec=Yuvec, colortension=colortension, cmap=cmap_tension
                )

            # Add Line labels
            if linelabels == True:
                xloc = np.dot(
                    [(line.rA[0] + line.rB[0]) / 2, (line.rA[1] + line.rB[1]) / 2, (line.rA[2] + line.rB[2]) / 2], Xuvec
                )
                yloc = np.dot(
                    [(line.rA[0] + line.rB[0]) / 2, (line.rA[1] + line.rB[1]) / 2, (line.rA[2] + line.rB[2]) / 2], Yuvec
                )
                ax.text(xloc, yloc, j)

        if cbar_tension:
            maxten = max([max(line.getLineTens()) for line in self.lineList])  # find the max tension in the System
            minten = min([min(line.getLineTens()) for line in self.lineList])  # find the min tension in the System
            bounds = range(int(minten), int(maxten), int((maxten - minten) / 256))
            norm = mpl.colors.BoundaryNorm(
                bounds, 256
            )  # set the bounds in a norm object, with 256 being the length of all colorbar strings
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_tension), label="Tension (N)")  # add the colorbar
            fig.tight_layout()

        # Add point labels
        i = 0
        for point in self.pointList:
            i = i + 1
            if pointlabels == True:
                xloc = np.dot([point.r[0], point.r[1], point.r[2]], Xuvec)
                yloc = np.dot([point.r[0], point.r[1], point.r[2]], Yuvec)
                ax.text(xloc, yloc, i, c="r")

        if isinstance(bathymetry, str):  # or, if it's a string, load in the bathymetry file

            # parse through the MoorDyn bathymetry file
            bathGrid_Xs, bathGrid_Ys, bathGrid = self.readBathymetryFile(bathymetry)

            X, Y = np.meshgrid(bathGrid_Xs, bathGrid_Ys)
            Z = -bathGrid
            if rang == "hold":
                rang = (np.min(Z), np.max(Z))

            Xind = Xuvec.index(1)
            Yind = Yuvec.index(1)
            Zind = int(3 - Xind - Yind)
            W = [X, Y, Z]

            # plot a contour profile of the bathymetry
            bath = ax.contourf(
                W[Xind], W[Yind], W[Zind], cmap=cmap_bath, levels=levels, alpha=alpha, vmin=rang[0], vmax=rang[1]
            )

            if (
                cbar_bath_aspect != 20 or cbar_bath_ticks != None
            ):  # make sure the colorbar is turned on just in case it isn't when the other colorbar inputs are used
                cbar_bath = True
            if cbar_bath:
                fig.colorbar(bath, label="depth (m)", aspect=cbar_bath_aspect, ticks=cbar_bath_ticks)

        ax.axis("equal")
        ax.set_title(title)

        return fig, ax  # return the figure and axis object in case it will be used later to update the plot

    def animateSolution(self, DOFtype="free"):
        """Creates an animation of the system

        Returns
        -------
        None.

        """

        # first draw a plot of DOFs and forces
        x = np.array(self.Xs)
        f = np.array(self.Es)
        fig, ax = plt.subplots(2, 1, sharex=True)
        for i in range(len(self.Es[0])):
            ax[0].plot(x[:, i])  # <<< warning this is before scale and offset!
            ax[1].plot(f[:, i], label=i + 1)
        ax[1].legend()

        # self.mooringEq(self.freeDOFs[0])   # set positions back to the first ones of the iteration process
        self.mooringEq(
            self.Xs[0], DOFtype=self.DOFtype_solve_for
        )  # set positions back to the first ones of the iteration process
        # ^^^^^^^ this only works for free DOF animation cases (not coupled DOF ones) <<<<< ...should be good now

        fig, ax = self.plot()  # make the initial plot to then animate

        nFreeDOF, nCpldDOF = self.getDOFs()
        if DOFtype == "free":
            nDOF = nFreeDOF
        elif DOFtype == "coupled":
            nDOF = nCpldDOF
        elif DOFtype == "both":
            nDOF = nFreeDOF + nCpldDOF

        # ms_delay = 10000/len(self.freeDOFs)  # time things so the animation takes 10 seconds

        line_ani = animation.FuncAnimation(
            fig,
            self.animate,
            np.arange(0, len(self.Xs), 1),  # fargs=(ax),
            interval=1000,
            blit=False,
            repeat_delay=2000,
            repeat=True,
        )

        return line_ani

    def animate(self, ts):
        """Redraws mooring system positions at step ts. Currently set up in a hack-ish way to work for animations
        involving movement of either free DOFs or coupled DOFs (but not both)
        """

        # following sets positions of all objects and may eventually be made into self.setPositions(self.positions[i])

        X = self.Xs[ts]  # Xs are already specified in solveEquilibrium's DOFtype

        if self.DOFtype_solve_for == "free":
            types = [0]
        elif self.DOFtype_solve_for == "coupled":
            types = [-1]
        elif self.DOFtype_solve_for == "both":
            types = [0, -1]
        else:
            raise ValueError("System.animate called but there is an invalid DOFtype being used")
        """
        if len(self.freeDOFs) > 0:
            X = self.freeDOFs[ts]   # get freeDOFs of current instant
            type = 0
        elif len(self.cpldDOFs) > 0:
            X = self.cpldDOFs[ts]   # get freeDOFs of current instant
            type = -1
        else:
            raise ValueError("System.animate called but no animation data is saved in freeDOFs or cpldDOFs")
        """

        # print(ts)

        i = 0  # index used to split off input positions X for each free object

        # update position of free Bodies
        for body in self.bodyList:
            if body.type in types:
                body.setPosition(X[i : i + 6])  # update position of free Body
                i += 6
            body.redraw()  # redraw Body

        # update position of free Points
        for point in self.pointList:
            if point.type in types:
                point.setPosition(X[i : i + 3])  # update position of free Point
                i += 3
                # redraw Point?

        # redraw all lines
        for line in self.lineList:
            line.redrawLine(0)

        # ax.set_title("iteration "+str(ts))
        # eventually could show net forces too? <<< if using a non MINPACK method, use callback and do this

        pass  # I added this line to get the above commented lines (^^^) to be included in the animate method

    def animatelines(self, interval=200, repeat=True, delay=0, runtime=-1, **kwargs):
        """
        Parameters
        ----------
        dirname : string
            The name of the directory folder you are in.
        rootname : string
            The name of the front portion of the main file name, like spar_WT1, or DTU_10MW_NAUTILUS_GoM.
        interval : int, optional
            The time between animation frames in milliseconds. The default is 200.
        repeat : bool, optional
            Whether or not to repeat the animation. The default is True.
        delay : int, optional
            The time between consecutive animation runs in milliseconds. The default is 0.

        Returns
        -------
        line_ani : animation
            an animation of the mooring lines based off of MoorDyn data.
            Needs to be stored, returned, and referenced in a variable
        """

        bathymetry = kwargs.get("bathymetry", False)  # toggles whether to show the axes or not
        opacity = kwargs.get("opacity", 1.0)  # the transparency of the bathymetry plot_surface
        hidebox = kwargs.get("hidebox", False)  # toggles whether to show the axes or not
        rang = kwargs.get(
            "rang", "hold"
        )  # colorbar range: if range not used, set it as a placeholder, it will get adjusted later
        res = kwargs.get(
            "res", 10
        )  # the resolution of the animation; how fluid the animation is. Higher res means spottier animation. counter-intuitive
        colortension = kwargs.get(
            "colortension", False
        )  # toggle to draw the mooring lines in colors based on node tensions
        cmap_tension = kwargs.get("cmap_tension", "rainbow")  # the type of color spectrum desired for colortensions

        # not adding cbar_tension colorbar yet since the tension magnitudes might change in the animation and the colorbar won't reflect that
        # can use any other kwargs that go into self.plot()

        if self.qs == 1:
            raise ValueError(
                "This System is set to be quasi-static. Import MoorDyn data and make qs=0 to use this method"
            )

        # update animation function. This gets called every iteration of the animation and redraws the line in its next position
        def update_Coords(
            tStep, tempLineList, tempax, colortension, cmap_tension
        ):  # not sure why it needs a 'tempax' input but it works better with it

            for imooring in tempLineList:
                imooring.redrawLine(-tStep, colortension=colortension, cmap_tension=cmap_tension)

            return

        # create the figure and axes to draw the animation
        fig, ax = self.plot(
            bathymetry=bathymetry, opacity=opacity, hidebox=hidebox, rang=rang, colortension=colortension
        )
        """
        # can do this section instead of self.plot(). They do the same thing
        fig = plt.figure(figsize=(20/2.54,12/2.54))
        ax = Axes3D(fig)
        for imooring in self.lineList:
            imooring.drawLine(0, ax)
        """
        # set figure x/y/z bounds
        d = 1600  # can make this an input later
        ax.set_xlim((-d, d))
        ax.set_ylim((-d, d))
        ax.set_zlim((-self.depth, 300))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # make the axes scaling equal
        rangex = np.diff(ax.get_xlim3d())[0]
        rangey = np.diff(ax.get_ylim3d())[0]
        rangez = np.diff(ax.get_zlim3d())[0]
        ax.set_box_aspect([rangex, rangey, rangez])

        if runtime == -1:
            nFrames = len(self.lineList[0].Tdata)
        else:
            itime = int(np.where(self.lineList[0].Tdata == runtime)[0])
            nFrames = len(self.lineList[0].Tdata[0:itime])

        # Animation: update the figure with the updated coordinates from update_Coords function
        # NOTE: the animation needs to be stored in a variable, return out of the method, and referenced when calling self.animatelines()
        line_ani = animation.FuncAnimation(
            fig,
            update_Coords,
            np.arange(1, nFrames - 1, res),
            fargs=(self.lineList, ax, colortension, cmap_tension),
            interval=1,
            repeat=repeat,
            repeat_delay=delay,
            blit=False,
        )
        # works well when np.arange(...nFrames...) is used. Others iterable ways to do this

        return line_ani
