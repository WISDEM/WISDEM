import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from wisdem.moorpy.helpers import CatenaryError


def nonlinear(XF, ZF, L, Str, Ten, W, nNodes=20, plots=0):
    """
    nonlinear mooring line solver.  The nonlinear module is assumed to be off of the seafloor, and have no bending.  The nonlinear module is called by
    putting a tension-strain file in the place of linestiffness (EA).  Unlike the catenary solver no iterative method is needed, but this function is heavily
    borrows from the catenary function.

    Parameters
    ----------
    XF : float
        Horizontal distance from end 1 to end 2 [m]
    ZF : float
        Vertical distance from end 1 to end 2 [m] (positive up)
    L  : float
        Unstretched length of line [m]
    Str : dictionary
        vector of strains from the tension-strain input file [-]
    Ten : dictionary
        vector of  of tensions from the tension-strain input file [N]
    W  : float
        Weight of line in fluid per unit length [N/m]
    nNodes : int, optional
        Number of nodes to describe the line
    plots  : int, optional
        1: plot output, 0: don't


    Returns
    -------
    : tuple
        (end 1 horizontal tension, end 1 vertical tension, end 2 horizontal tension, end 2 vertical tension, info dictionary) [N] (positive up)

    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The following is originally from the catenary.py function but applies to this one aswell
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make info dict to contain any additional outputs
    info = dict(error=False)

    # make some arrays if needed for plotting each node
    if plots > 0:
        s = np.linspace(
            0, L, nNodes
        )  #  Unstretched arc distance along line from anchor to each node where the line position and tension can be output (meters)
        Xs = np.zeros(nNodes)  #  Horizontal locations of each line node relative to the anchor (meters)
        Zs = np.zeros(nNodes)  #  Vertical   locations of each line node relative to the anchor (meters)
        Te = np.zeros(nNodes)  #  Effective line tensions at each node (N)

    # flip line in the solver if it is buoyant
    if W < 0:
        W = -W
        ZF = -ZF
        CB = -10000.0  # <<< TODO: set this to the distance to sea surface <<<
        flipFlag = True
    else:
        flipFlag = False

    # reverse line in the solver if end A is above end B
    if ZF < 0:
        ZF = -ZF
        reverseFlag = True
    else:
        reverseFlag = False

    # ensure the input variables are realistic
    if XF < 0.0:
        raise CatenaryError("XF is negative!")
    if L <= 0.0:
        breakpoint()
        raise CatenaryError("L is zero or negative!")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    New Code
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Determine the stretched length of the nonlinear element based on the horizontal and vertical fairlead excursions
    str_L = np.sqrt(XF * XF + ZF * ZF)

    # Based on the stretched length and the horizontal fairlead excursions define cos/sin and products for transformation matrix
    # Define cosine and sine
    c = XF / str_L
    s = ZF / str_L
    # Define all the products of sine and cosine
    c2 = c * c
    s2 = s * s
    cs = c * s

    # Determine the strain in the segment with stretched length and original length
    str_seg = str_L / L - 1

    # Make the input dictionarys for str and ten a vector of floats
    Str = [float(x) for x in Str]
    Ten = [float(x) for x in Ten]

    # Strain in the line cant be less than zero (cant push a rope so to speak)
    if str_seg < 0:
        str_seg = 0

    ten_seg = np.interp(str_seg, Str, Ten)
    # small change is strain for a finite difference
    d_str = 1e-5
    # finite difference to take derivative of ten-strain curve (dT/dstr = EA)
    EA_est = (np.interp(str_seg + d_str, Str, Ten) - np.interp(str_seg, Str, Ten)) / d_str

    # Breakdown the horizontal components of the tension
    FxA = ten_seg * c
    FxB = -FxA

    # Breakdown the vertical components of the tension and subtract the weight of the nonlinear element
    FzA = ten_seg * s - 0.5 * W * L
    FzB = -ten_seg * s - 0.5 * W * L

    # Tension at both ends of the line
    TA = np.sqrt(FxA * FxA + FzA * FzA)
    TB = np.sqrt(FxB * FxB + FzB * FzB)

    # Estimate Stiffness Numerically (Potentially find something better)
    # could also be hard to do because the input tension strain files are arbitrary

    # Stiffness matrcies for the ends of the line (In this case since we already assume the element can only deform axially
    # this is essentially just a truss element with the nonlinear stiffness that we calculated
    # Ka = (EA_est/L)*np.array([[c2, cs],[cs, s2]])
    # Kb = (EA_est/L)*np.array([[c2, cs],[cs, s2]])
    # Kab = -(EA_est/L)*np.array([[c2, cs],[cs, s2]])

    Ka = np.array(
        [
            [
                c2 * (EA_est / str_L) + s2 * ten_seg / (L * (1 + str_seg)),
                cs * (EA_est / str_L) - cs * ten_seg / (L * (1 + str_seg)),
            ],
            [
                cs * (EA_est / str_L) - cs * ten_seg / (L * (1 + str_seg)),
                s2 * (EA_est / str_L) + c2 * ten_seg / (L * (1 + str_seg)),
            ],
        ]
    )
    Kb = Ka
    Kab = -Ka

    # Assign values to info
    # Fairlead forces
    info["HF"] = -FxB
    info["VF"] = -FzB
    info["HA"] = FxA
    info["VA"] = FzA
    # Line stiffnesses
    info["stiffnessA"] = Ka
    info["stiffnessB"] = Kb
    info["stiffnessAB"] = Kab
    # Length on the bottom is zero as we assumed earlier
    info["LBot"] = 0
    # self.info = info
    # For plotting (assumed straight line so no issue and tension varies linearly
    info["X"] = np.linspace(0, XF, nNodes)
    info["Z"] = np.linspace(0, ZF, nNodes)
    info["s"] = np.linspace(0, str_L, nNodes)
    info["Te"] = np.linspace(TA, TB, nNodes)

    print(info["stiffnessA"])
    print(info["stiffnessB"])
    print(info["stiffnessAB"])

    return (FxA, FzA, FxB, FzB, info)
