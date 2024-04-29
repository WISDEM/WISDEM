import numpy as np

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
        arc = np.linspace(0, np.pi, 100)
        self.x = 0.5 * (1 - np.cos(arc))  # cosine spacing
        self.yu = np.interp(self.x, xu, yu)
        self.yl = np.interp(self.x, xl, yl)

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
        i = np.argmin(x)

        xu = x[i::-1]
        yu = y[i::-1]
        xl = x[i:]
        yl = y[i:]

        # check if coordinates were input in other direction
        if np.mean(y[0:i]) < np.mean(y[i:]):
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
            if x[i + 1] <= x[i]:
                iuLast = i
                ilLast = i
                if x[i + 1] == x[i]:  # blunt t.e.
                    ilLast = i + 1  # stop at i+1
                break

        xu = x[: iuLast + 1]
        yu = y[: iuLast + 1]
        xl = x[-1 : ilLast - 1 : -1]
        yl = y[-1 : ilLast - 1 : -1]

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
        f = open(filename, "r")

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

    def _preCompFormat(self):
        """
        docstring
        """

        # check if they share a common trailing edge point
        te_same = self.yu[-1] == self.yl[-1]

        # count number of points
        nu = len(self.x)
        if te_same:
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

        return x, y

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
        yu = self.yu + weight * (other.yu - self.yu)
        yl = self.yl + weight * (other.yl - self.yl)

        return Profile(self.x, yu, self.x, yl)

    @property
    def tc(self):
        """thickness to chord ratio of the Profile"""
        return max(self.yu - self.yl)

    def set_tc(self, new_tc):
        factor = new_tc / self.tc

        self.yu *= factor
        self.yl *= factor
