import numpy as np


class LineType:
    """A class to hold the various properties of a mooring line type"""

    def __init__(self, name, d, massden, EA, MBL=0.0, cost=0.0, notes="", input_d=0.0, input_type=""):
        """Initialize LineType attributes

        Parameters
        ----------
        name : string
            identifier string
        d : float
            volume-equivalent diameter [m]
        massden : float
            linear mass density [kg/m] used to calculate weight density (w) [N/m]
        EA : float
            extensional stiffness [N]
        MBL : float, optional
            Minimum breaking load [N]. The default is 0.0.
        cost : float, optional
            material cost per unit length [$/m]. The default is 0.0.
        notes : string, optional
            optional notes/description string. The default is "".
        input_d : float, optional
            the given input diameter that has not been adjusted for the line's volume [m]. The default is 0.0.
        input_type : string, optional
            the type of the line (e.g. chain, polyester), different from the name. The default is "".
        Returns
        -------
        None.

        """
        self.name = name  # identifier string
        self.d = d  # volume-equivalent diameter [m]
        self.m = massden  # linear desnity [kg/m]
        self.w = (massden - np.pi / 4 * d * d * 1025) * 9.81  # wet weight [N/m]
        self.EA = EA  # extensional stiffness [N]
        self.MBL = MBL  # minimum breaking load [N]
        self.cost = cost  # material cost of line per unit length [$/m]
        self.notes = notes  # optional notes/description string
        self.input_d = input_d  # the non-volume-equivalent, input diameter [m]
        self.input_type = input_type  # line type string (e.g. chain, polyester)
