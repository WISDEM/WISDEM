import os
import time

import yaml
import numpy as np


# base class for MoorPy exceptions
class Error(Exception):
    """Base class for MoorPy exceptions"""

    pass


# Catenary error class
class CatenaryError(Error):
    """Derived error class for catenary function errors. Contains an error message."""

    def __init__(self, message):
        self.message = message


# Line Object error class
class LineError(Error):
    """Derived error class for Line object errors. Contains an error message and the line number with the error."""

    def __init__(self, num, message):
        self.line_num = num
        self.message = message


# Solve error class for any solver process
class SolveError(Error):
    """Derived error class for various solver errors. Contains an error message"""

    def __init__(self, message):
        self.message = message


# Generic MoorPy error
class MoorPyError(Error):
    """Derived error class for MoorPy. Contains an error message"""

    def __init__(self, message):
        self.message = str(message)


def printMat(mat):
    """Prints a matrix to a format that is specified

    Parameters
    ----------
    mat : array
        Any matrix that is to be printed.

    Returns
    -------
    None.

    """
    for i in range(mat.shape[0]):
        print("\t".join(["{:+8.3e}"] * mat.shape[1]).format(*mat[i, :]))


def printVec(vec):
    """Prints a vector to a format that is specified

    Parameters
    ----------
    vec : array
        Any vector that is to be printed.

    Returns
    -------
    None.

    """
    print("\t".join(["{:+9.4e}"] * len(vec)).format(*vec))


def getH(r):
    """function gets the alternator matrix, H, that when multiplied with a vector,
    returns the cross product of r and that vector

    Parameters
    ----------
    r : array
        the position vector that another vector is from a point of interest.

    Returns
    -------
    H : matrix
        the alternator matrix for the size-3 vector, r.

    """

    H = np.array([[0, r[2], -r[1]], [-r[2], 0, r[0]], [r[1], -r[0], 0]])
    return H


def rotationMatrix(x3, x2, x1):
    """Calculates a rotation matrix based on order-z,y,x instrinsic (tait-bryan?) angles, meaning
    they are about the ROTATED axes. (rotation about z-axis would be (0,0,theta) )

    Parameters
    ----------
    x3, x2, x1: floats
        The angles that the rotated axes are from the nonrotated axes. Normally roll,pitch,yaw respectively. [rad]

    Returns
    -------
    R : matrix
        The rotation matrix
    """
    # initialize the sines and cosines
    s1 = np.sin(x1)
    c1 = np.cos(x1)
    s2 = np.sin(x2)
    c2 = np.cos(x2)
    s3 = np.sin(x3)
    c3 = np.cos(x3)

    # create the rotation matrix
    R = np.array(
        [
            [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
            [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
            [-s2, c2 * s3, c2 * c3],
        ]
    )

    return R


def rotatePosition(rRelPoint, rot3):
    """Calculates the new position of a point by applying a rotation (rotates a vector by three angles)

    Parameters
    ----------
    rRelPoint : array
        x,y,z coordinates of a point relative to a local frame [m]
    rot3 : array
        Three angles that describe the difference between the local frame and the global frame/ Normally roll,pitch,yaw. [rad]

    Returns
    -------
    rRel : array
        The relative rotated position of the point about the local frame [m]
    """

    # get rotation matrix from three provided angles
    RotMat = rotationMatrix(rot3[0], rot3[1], rot3[2])

    # find location of point in unrotated reference frame about reference point
    rRel = np.matmul(RotMat, rRelPoint)

    return rRel


def transformPosition(rRelPoint, r6):
    """Calculates the position of a point based on its position relative to translated and rotated 6DOF body

    Parameters
    ----------
    rRelPoint : array
        x,y,z coordinates of a point relative to a local frame [m]
    r6 : array
        6DOF position vector of the origin of the local frame, in the global frame coorindates [m]

    Returns
    -------
    rAbs : array
        The absolute position of the point about the global frame [m]
    """
    # note: r6 should be in global orientation frame

    # absolute location = rotation of relative position + absolute position of reference point
    rAbs = rotatePosition(rRelPoint, r6[3:]) + r6[:3]

    return rAbs


def translateForce3to6DOF(r, Fin):
    """Takes in a position vector and a force vector (applied at the positon), and calculates
    the resulting 6-DOF force and moment vector.

    Parameters
    ----------
    r : array
        x,y,z coordinates at which force is acting [m]
    Fin : array
        x,y,z components of force [N]

    Returns
    -------
    Fout : array
        The resulting force and moment vector [N, Nm]
    """

    # initialize output vector as same dtype as input vector (to support both real and complex inputs)
    Fout = np.zeros(6, dtype=Fin.dtype)

    # set the first three elements of the output vector the same as the input vector
    Fout[:3] = Fin

    # set the last three elements of the output vector as the cross product of r and Fin
    Fout[3:] = np.cross(r, Fin)

    return Fout


def set_axes_equal(ax):
    """Sets 3D plot axes to equal scale

    Parameters
    ----------
    ax : matplotlib.pyplot axes
        the axes that are to be set equal in scale to each other.

    Returns
    -------
    None.

    """

    rangex = np.diff(ax.get_xlim3d())[0]
    rangey = np.diff(ax.get_ylim3d())[0]
    rangez = np.diff(ax.get_zlim3d())[0]

    ax.set_box_aspect([rangex, rangey, rangez])  # note: this may require a matplotlib update


def dsolve2(
    eval_func,
    X0,
    Ytarget=[],
    step_func=None,
    args=[],
    tol=0.0001,
    ytol=0,
    maxIter=20,
    Xmin=[],
    Xmax=[],
    a_max=2.0,
    dX_last=[],
    stepfac=4,
    display=0,
    dodamping=False,
):
    """
    PARAMETERS
    ----------
    eval_func : function
        function to solve (will be passed array X, and must return array Y of same size)
    X0 : array
        initial guess of X
    Ytarget : array (optional)
        target function results (Y), assumed zero if not provided
    stp_func : function (optional)
        function use for adjusting the variables (computing dX) each step.
        If not provided, Netwon's method with finite differencing is used.
    args : list
        A list of variables (e.g. the system object) to be passed to both the eval_func and step_func
    tol : float or array
        If scalar, the*relative* convergence tolerance (applied to step size components, dX).
        If an array, must be same size as X, and specifies an absolute convergence threshold for each variable.
    ytol: float, optional
        If specified, this is the absolute error tolerance that must be satisfied. This overrides the tol setting which otherwise works based on x values.
    Xmin, Xmax
        Bounds. by default start bounds at infinity
    a_max
        maximum step size acceleration allowed
    dX_last
        Used if you want to dictate the initial step size/direction based on a previous attempt
    """
    success = False
    start_time = time.time()
    # process inputs and format as arrays in case they aren't already

    X = np.array(X0, dtype=np.float_)  # start off design variable
    N = len(X)

    Xs = np.zeros([maxIter, N])  # make arrays to store X and error results of the solve
    Es = np.zeros([maxIter, N])
    dXlist = np.zeros([maxIter, N])
    dXlist2 = np.zeros([maxIter, N])

    damper = 1.0  # used to add a relaxation/damping factor to reduce the step size and combat instability

    # check the target Y value input
    if len(Ytarget) == N:
        Ytarget = np.array(Ytarget, dtype=np.float_)
    elif len(Ytarget) == 0:
        Ytarget = np.zeros(N, dtype=np.float_)
    else:
        raise TypeError("Ytarget must be of same length as X0")

    # ensure all tolerances are positive
    if ytol == 0:  # if not using ytol
        if np.isscalar(tol) and tol <= 0.0:
            raise ValueError("tol value passed to dsovle2 must be positive")
        elif not np.isscalar(tol) and any(tol <= 0):
            raise ValueError("every tol entry passed to dsovle2 must be positive")

    # if a step function wasn't provided, provide a default one
    if step_func == None:
        if display > 1:
            print("Using default finite difference step func")

        def step_func(X, args, Y, oths, Ytarget, err, tols, iter, maxIter):
            """this now assumes tols passed in is a vector and are absolute quantities"""
            J = np.zeros([N, N])  # Initialize the Jacobian matrix that has to be a square matrix with nRows = len(X)

            for i in range(
                N
            ):  # Newton's method: perturb each element of the X variable by a little, calculate the outputs from the
                X2 = np.array(
                    X
                )  # minimizing function, find the difference and divide by the perturbation (finding dForce/d change in design variable)
                deltaX = (
                    stepfac * tols[i]
                )  # note: this function uses the tols variable that is computed in dsolve based on the tol input
                X2[i] += deltaX
                Y2, _, _ = eval_func(X2, args)  # here we use the provided eval_func

                J[:, i] = (Y2 - Y) / deltaX  # and append that column to each respective column of the Jacobian matrix

            if N > 1:
                dX = -np.matmul(
                    np.linalg.inv(J), Y - Ytarget
                )  # Take this nth output from the minimizing function and divide it by the jacobian (derivative)
            else:
                if J[0, 0] == 0.0:
                    raise ValueError("dsolve2 found a zero gradient")

                dX = np.array([-(Y[0] - Ytarget[0]) / J[0, 0]])

                if display > 1:
                    print(
                        f" step_func iter {iter} X={X[0]:9.2e}, error={Y[0]-Ytarget[0]:9.2e}, slope={J[0,0]:9.2e}, dX={dX[0]:9.2e}"
                    )

            return dX  # returns dX (step to make)

    # handle bounds
    if len(Xmin) == 0:
        Xmin = np.zeros(N) - np.inf
    elif len(Xmin) == N:
        Xmin = np.array(Xmin, dtype=np.float_)
    else:
        raise TypeError("Xmin must be of same length as X0")

    if len(Xmax) == 0:
        Xmax = np.zeros(N) + np.inf
    elif len(Xmax) == N:
        Xmax = np.array(Xmax, dtype=np.float_)
    else:
        raise TypeError("Xmax must be of same length as X0")

    if len(dX_last) == 0:
        dX_last = np.zeros(N)
    else:
        dX_last = np.array(dX_last, dtype=np.float_)

    if display > 0:
        print(f"Starting dsolve iterations>>>   aiming for Y={Ytarget}")

    for iter in range(maxIter):

        # call evaluation function
        Y, oths, stop = eval_func(X, args)

        # compute error
        err = Y - Ytarget

        if display == 2:
            print(f"  new iteration #{iter} with RMS error {np.linalg.norm(err):8.3e}")
        if display > 2:
            print(f"  new iteration #{iter} with X={X} and Y={Y}")

        Xs[iter, :] = X
        Es[iter, :] = err

        # stop if commanded by objective function
        if stop:
            break

        # handle tolerances input
        if np.isscalar(tol):
            tols = tol * (np.abs(X) + tol)
        else:
            tols = np.array(tol)

        # check maximum iteration
        if iter == maxIter - 1:
            if display > 0:
                print("Failed to find solution after " + str(iter) + " iterations, with error of " + str(err))

            # looks like things didn't converge, so if N=1 do a linear fit on the last 30% of points to estimate the soln
            if N == 1:

                m, b = np.polyfit(Es[int(0.7 * iter) : iter, 0], Xs[int(0.7 * iter) : iter, 0], 1)
                X = np.array([b])
                Y = np.array([0.0])
                print(f"Using linaer fit to estimate solution at X={b}")

            break

        # >>>> COULD ALSO HAVE AN ITERATION RESTART FUNCTION? >>>
        #  that returns a restart boolean, as well as what values to use to restart things if true. How?

        else:
            dX = step_func(X, args, Y, oths, Ytarget, err, tols, iter, maxIter)

        # if display>2:
        #    breakpoint()

        # Make sure we're not diverging by keeping things from reversing too much.
        # Track the previous step (dX_last) and if the current step reverses too much, stop it part way.
        # Stop it at a plane part way between the current X value and the previous X value (using golden ratio, why not).

        # get the point along the previous step vector where we'll draw the bounding hyperplane (could be a line, plane, or more in higher dimensions)
        Xlim = X - 0.62 * dX_last

        # the equation for the plane we don't want to recross is then sum(X*dX_last) = sum(Xlim*dX_last)
        if np.sum((X + dX) * dX_last) < np.sum(Xlim * dX_last):  # if we cross are going to cross it

            alpha = np.sum((Xlim - X) * dX_last) / np.sum(
                dX * dX_last
            )  # this is how much we need to scale down dX to land on it rather than cross it

            if display > 2:
                print("  limiting oscillation with alpha=" + str(alpha))
                print(f"   dX_last was {dX_last}, dX was going to be {dX}, now it'll be {alpha*dX}")
                print(f"   dX_last was {dX_last/1000}, dX was going to be {dX/1000}, now it'll be {alpha*dX/1000}")

            dX = alpha * dX  # scale down dX

        # also avoid extreme accelerations in the same direction
        for i in range(N):

            # should update the following for ytol >>>
            if abs(dX_last[i]) > tols[i]:  # only worry about accelerations if the last step was non-negligible

                dX_max = (
                    a_max * dX_last[i]
                )  # set the maximum permissible dx in each direction based an an acceleration limit

                if dX_max == 0.0:  # avoid a divide-by-zero case (if dX[i] was zero to start with)
                    breakpoint()
                    dX[i] = 0.0
                else:
                    a_i = dX[i] / dX_max  # calculate ratio of desired dx to max dx

                    if a_i > 1.0:

                        if display > 2:
                            print(f"    limiting acceleration ({1.0/a_i:6.4f}) for axis {i}")
                            print(f"     dX_last was {dX_last}, dX was going to be {dX}")

                        # dX = dX*a_max/a_i  # scale it down to the maximum value
                        dX[i] = dX[i] / a_i  # scale it down to the maximum value (treat each DOF individually)

                        if display > 2:
                            print(f"     now dX will be {dX}")

        dXlist[iter, :] = dX
        # if iter==196:
        # breakpoint()

        # add damping if cyclic behavior is detected at the halfway point
        if dodamping and iter == int(0.5 * maxIter):
            if display > 2:
                print(f"dsolve2 is at iteration {iter} (50% of maxIter)")

            for j in range(2, iter - 1):
                iterc = iter - j
                if all(np.abs(X - Xs[iterc, :]) < tols):
                    print(f"dsolve2 is going in circles detected at iteration {iter}")
                    print(f"last similar point was at iteration {iterc}")
                    damper = damper * 0.9
                    break

        dX = damper * dX

        # enforce bounds
        for i in range(N):

            if X[i] + dX[i] < Xmin[i]:
                dX[i] = Xmin[i] - X[i]

            elif X[i] + dX[i] > Xmax[i]:
                dX[i] = Xmax[i] - X[i]

        dXlist2[iter, :] = dX
        # check for convergence
        if (ytol == 0 and all(np.abs(dX) < tols)) or (ytol > 0 and all(np.abs(err) < ytol)):

            if display > 0:
                print(
                    "Iteration converged after "
                    + str(iter)
                    + " iterations with error of "
                    + str(err)
                    + " and dX of "
                    + str(dX)
                )
                print("Solution X is " + str(X))

                # if abs(err) > 10:
                #    breakpoint()

                if display > 0:
                    print(
                        "Total run time: {:8.2f} seconds = {:8.2f} minutes".format(
                            (time.time() - start_time), ((time.time() - start_time) / 60)
                        )
                    )

            if any(X == Xmin) or any(X == Xmax):
                success = False
                print("Warning: dsolve ended on a bound.")
            else:
                success = True

            break

        dX_last = 1.0 * dX  # remember this current value

        X = X + dX

    return (
        X,
        Y,
        dict(iter=iter, err=err, dX=dX_last, oths=oths, Xs=Xs, Es=Es, success=success, dXlist=dXlist, dXlist2=dXlist2),
    )


def getLineProps(dnommm, material, source=None, name="", rho=1025.0, g=9.81, **kwargs):
    """Sets up a dictionary that represents a mooring line type based on the
    specified diameter and material type. The

    - This function requires at least one input: the line diameter in millimeters.
    - The rest of the inputs are optional: describe the desired type of line (chain, polyester, wire, etc.),
    the type of chain (studless or studlink), the source of data (Orcaflex-original or altered), or a name identifier
    - The function will output a MoorPy linetype object

    # support options for :

    # 1. dictionary passed in (this is what will be done when called from a System method)

        (a) by default, System will load in MoorPy's default line property information YAML

        (b) or user specify's alternate YAML file when making the system


    # 2. yaml filename pass in or use default moorpy yaml and load yaml (used when called independently)

    System.linePropsDatabase will be a dictionary of stored YAML coefficients

    System.lineTypes will become a dictionary of LineType DICTIONARIES


    Parameters
    ----------
    dnommm : float
        nominal diameter [mm].
    material : string
        string identifier of the material type be used.
    source : dict or filename (optional)
        YAML file name or dictionary containing line property scaling coefficients
    name : any dict index (optional)
        Identifier for the line type (otherwise will be generated automatically).
    """

    # deal with the source (is it a dictionary, or reading in a new yaml?)
    lineProps = loadLineProps(source)

    # raise an error if the material isn't in the source dictionary
    if not material in lineProps:
        raise ValueError(f"Specified mooring line material, {material}, is not in the database.")

    # calculate the relevant properties for this specific line type
    mat = lineProps[material]  # shorthand for the sub-dictionary of properties for the material in question
    d = dnommm * 0.001  # convert nominal diameter from mm to m
    d_vol = mat["dvol_dnom"] * d
    mass = mat["mass_0"] + mat["mass_d"] * d + mat["mass_d2"] * d**2 + mat["mass_d3"] * d**3
    EA = mat["EA_0"] + mat["EA_d"] * d + mat["EA_d2"] * d**2 + mat["EA_d3"] * d**3
    MBL = mat["MBL_0"] + mat["MBL_d"] * d + mat["MBL_d2"] * d**2 + mat["MBL_d3"] * d**3
    cost = (
        mat["cost_0"]
        + mat["cost_d"] * d
        + mat["cost_d2"] * d**2
        + mat["cost_d3"] * d**3
        + mat["cost_mass"] * mass
        + mat["cost_EA"] * EA
        + mat["cost_MBL"] * MBL
    )
    w = (mass - np.pi / 4 * d_vol**2 * rho) * g

    # Set up a main identifier for the linetype unless one is provided
    if name == "":
        typestring = f"{type}{dnommm:.0f}"
    else:
        typestring = name

    notes = f"made with getLineProps"

    lineType = dict(
        name=typestring,
        d_vol=d_vol,
        m=mass,
        EA=EA,
        w=w,
        MBL=MBL,
        cost=cost,
        notes=notes,
        input_type=type,
        input_d=d,
        material=material,
    )

    lineType.update(kwargs)  # add any custom arguments provided in the call to the lineType's dictionary

    return lineType


def loadLineProps(source):
    """Loads a set of MoorPy mooring line property scaling coefficients from
    a specified YAML file or passed dictionary. Any coefficients not included
    will take a default value (zero for everything except diameter ratio,
    which is 1). It returns a dictionary containing the complete mooring line
    property scaling coefficient set to use for any provided mooring line types.

    Parameters
    ----------

    source : dict or filename
        YAML file name or dictionary containing line property scaling coefficients

    """

    if type(source) is dict:
        source = source

    elif source is None or source == "default":
        import os

        mpdir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(mpdir, "MoorProps_default.yaml")) as file:
            source = yaml.load(file, Loader=yaml.FullLoader)

    elif type(source) is string:
        with open(source) as file:
            source = yaml.load(file, Loader=yaml.FullLoader)

    else:
        raise Exception("loadLineProps supplied with invalid source")

    if "lineProps" in source:
        lineProps = source["lineProps"]
    else:
        raise Exception("YAML file or dictionary must have a 'lineProps' field containing the data")

    output = dict()  # output dictionary combining default values with loaded coefficients

    # combine loaded coefficients and default values into dictionary that will be saved for each material
    for mat, props in lineProps.items():
        output[mat] = {}
        output[mat]["mass_0"] = getFromDict(props, "mass_0", default=0.0)
        output[mat]["mass_d"] = getFromDict(props, "mass_d", default=0.0)
        output[mat]["mass_d2"] = getFromDict(props, "mass_d2", default=0.0)
        output[mat]["mass_d3"] = getFromDict(props, "mass_d3", default=0.0)
        output[mat]["EA_0"] = getFromDict(props, "EA_0", default=0.0)
        output[mat]["EA_d"] = getFromDict(props, "EA_d", default=0.0)
        output[mat]["EA_d2"] = getFromDict(props, "EA_d2", default=0.0)
        output[mat]["EA_d3"] = getFromDict(props, "EA_d3", default=0.0)
        output[mat]["MBL_0"] = getFromDict(props, "MBL_0", default=0.0)
        output[mat]["MBL_d"] = getFromDict(props, "MBL_d", default=0.0)
        output[mat]["MBL_d2"] = getFromDict(props, "MBL_d2", default=0.0)
        output[mat]["MBL_d3"] = getFromDict(props, "MBL_d3", default=0.0)
        output[mat]["dvol_dnom"] = getFromDict(props, "dvol_dnom", default=1.0)
        output[mat]["cost_0"] = getFromDict(props, "cost_0", default=0.0)
        output[mat]["cost_d"] = getFromDict(props, "cost_d", default=0.0)
        output[mat]["cost_d2"] = getFromDict(props, "cost_d2", default=0.0)
        output[mat]["cost_d3"] = getFromDict(props, "cost_d3", default=0.0)
        output[mat]["cost_mass"] = getFromDict(props, "cost_mass", default=0.0)
        output[mat]["cost_EA"] = getFromDict(props, "cost_EA", default=0.0)
        output[mat]["cost_MBL"] = getFromDict(props, "cost_MBL", default=0.0)

    return output


def getFromDict(dict, key, shape=0, dtype=float, default=None):
    """
    Function to streamline getting values from design dictionary from YAML file, including error checking.

    Parameters
    ----------
    dict : dict
        the dictionary
    key : string
        the key in the dictionary
    shape : list, optional
        The desired shape of the output. If not provided, assuming scalar output. If -1, any input shape is used.
    dtype : type
        Must be a python type than can serve as a function to format the input value to the right type.
    default : number, optional
        The default value to fill in if the item isn't in the dictionary. Otherwise will raise error if the key doesn't exist.
    """
    # in future could support nested keys   if type(key)==list: ...

    if key in dict:
        val = dict[key]  # get the value from the dictionary
        if shape == 0:  # scalar input expected
            if np.isscalar(val):
                return dtype(val)
            else:
                raise ValueError(f"Value for key '{key}' is expected to be a scalar but instead is: {val}")
        elif shape == -1:  # any input shape accepted
            if np.isscalar(val):
                return dtype(val)
            else:
                return np.array(val, dtype=dtype)
        else:
            if np.isscalar(val):  # if a scalar value is provided and we need to produce an array (of any shape)
                return np.tile(dtype(val), shape)

            elif np.isscalar(shape):  # if expecting a 1D array
                if len(val) == shape:
                    return np.array([dtype(v) for v in val])
                else:
                    raise ValueError(f"Value for key '{key}' is not the expected size of {shape} and is instead: {val}")

            else:  # must be expecting a multi-D array
                vala = np.array(val, dtype=dtype)  # make array

                if list(vala.shape) == shape:  # if provided with the right shape
                    return vala
                elif len(shape) > 2:
                    raise ValueError("Function getFromDict isn't set up for shapes larger than 2 dimensions")
                elif (
                    vala.ndim == 1 and len(vala) == shape[1]
                ):  # if we expect an MxN array, and an array of size N is provided, tile it M times
                    return np.tile(vala, [shape[0], 1])
                else:
                    raise ValueError(
                        f"Value for key '{key}' is not a compatible size for target size of {shape} and is instead: {val}"
                    )

    else:
        if default == None:
            raise ValueError(f"Key '{key}' not found in input file...")
        else:
            if shape == 0 or shape == -1:
                return default
            else:
                return np.tile(default, shape)


def makeTower(twrH, twrRad):
    """Sets up mesh points for visualizing a cylindrical structure (should align with RAFT eventually."""

    n = 8
    X = []
    Y = []
    Z = []
    ax = np.zeros(n + 1)
    ay = np.zeros(n + 1)
    for jj in range(n + 1):
        ax[jj] = np.cos(float(jj) / float(n) * 2.0 * np.pi)
        ay[jj] = np.sin(float(jj) / float(n) * 2.0 * np.pi)

    for ii in range(int(len(twrRad) - 1)):
        z0 = twrH * float(ii) / float(len(twrRad) - 1)
        z1 = twrH * float(ii + 1) / float(len(twrRad) - 1)
        for jj in range(n + 1):
            X.append(twrRad[ii] * ax[jj])
            Y.append(twrRad[ii] * ay[jj])
            Z.append(z0)
            X.append(twrRad[ii + 1] * ax[jj])
            Y.append(twrRad[ii + 1] * ay[jj])
            Z.append(z1)

    Xs = np.array(X)
    Ys = np.array(Y)
    Zs = np.array(Z)

    return Xs, Ys, Zs


def read_mooring_file(dirName, fileName):
    # Taken from line system.... maybe should be a helper function?
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
