# --------------------------------------------------------------------------------------------
#                                  MoorPy
#
#       A mooring system visualizer and quasi-static modeler in Python.
#                         Matt Hall and Stein Housner
#
# --------------------------------------------------------------------------------------------
# 2018-08-14: playing around with making a QS shared-mooring simulation tool, to replace what's in Patrick's work
# 2020-06-17: Trying to create a new quasi-static mooring system solver based on my Catenary function adapted from FAST v7, and using MoorDyn architecture

import numpy as np
import wisdem.moorpy.MoorSolve as msolve

# import scipy.optimize

# Import plot libraries if need be, but don't error if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
except:
    pass


# base class for MoorPy exceptions
class Error(Exception):
    """ Base class for MoorPy exceptions"""

    pass


# Catenary error class
class CatenaryError(Error):
    """Derived error class for Catenary function errors. Contains an error message."""

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


def Catenary(XF, ZF, L, EA, W, CB=0, HF0=0, VF0=0, Tol=0.000001, nNodes=20, MaxIter=50, plots=0):
    """
    The quasi-static mooring line solver. Adapted from Catenary subroutine in FAST v7 by J. Jonkman.
    Note: this version is updated Oct 7 2020 to use the dsolve solver.

    Parameters
    ----------
    XF : float
        Horizontal distance from end 1 to end 2 [m]
    ZF : float
        Vertical distance from end 1 to end 2 [m] (positive up)
    L  : float
        Unstretched length of line [m]
    EA : float
        Extensional stiffness of line [N]
    W  : float
        Weight of line in fluid per unit length [N/m]
    CB : float, optional
        If positive, coefficient of seabed static friction drag. If negative, no seabed contact and the value is the distance down from end A to the seabed in m\
            NOTE: for lines between floating bodies, there must be no seabed contact (set CB < 0)
    HF0 : float, optional
        Horizontal fairlead tension. If zero or not provided, a guess will be calculated.
    VF0 : float, optional
        Vertical fairlead tension. If zero or not provided, a guess will be calculated.

    Tol    :  int, optional
        Convergence tolerance within Newton-Raphson iteration specified as a fraction of tension
    nNodes : int, optional
        Number of nodes to describe the line
    MaxIter:  int, optional
        Maximum number of iterations to try before resetting to default ICs and then trying again
    plots  : int, optional
        1: plot output, 0: don't


    Returns
    -------
    : tuple
        (end 1 horizontal tension, end 1 vertical tension, end 2 horizontal tension, end 2 vertical tension, info dictionary) [N] (positive up)

    """

    # make info dict to contain any additional outputs
    info = dict(error=False)

    # flip line in the solver if end A is above end B
    if ZF < 0:
        ZF = -ZF
        reverseFlag = 1
    else:
        reverseFlag = 0

    # ensure the input variables are realistic
    if XF <= 0.0:
        raise CatenaryError("XF is zero or negative!")
    if L <= 0.0:
        raise CatenaryError("L is zero or negative!")
    if EA <= 0.0:
        raise CatenaryError("EA is zero or negative!")

    # Solve for the horizontal and vertical forces at the fairlead (HF, VF) and at the anchor (HA, VA)

    # There are many "ProfileTypes" of a mooring line and each must be analyzed separately
    # ProfileType=1: No portion of the line rests on the seabed
    # ProfileType=2: A portion of the line rests on the seabed and the anchor tension is nonzero
    # ProfileType=3: A portion of the line must rest on the seabed and the anchor tension is zero
    # ProfileType=4: Entire line is on seabed
    # ProfileType=0: The line is negatively buoyant, seabed interaction is enabled, and the line
    # is longer than a full L between end points (including stretching) i.e. it is horizontal
    # along the seabed from the anchor, then vertical to the fairlaed. Computes the maximum
    # stretched length of the line with seabed interaction beyond which the line would have to
    # double-back on itself; the line forms an "L" between the anchor and fairlead. Then it
    # models it as bunched up on the seabed (instead of throwing an error)

    EA_W = EA / W

    # ProfileType 4 case - entirely along seabed
    if ZF == 0.0 and CB >= 0.0 and W > 0:

        ProfileType = 4
        # this is a special case that requires no iteration

        HF = np.max([0, (XF / L - 1.0) * EA])  # calculate fairlead tension based purely on elasticity
        VF = 0.0
        HA = np.max([0.0, HF - CB * W * L])  # calculate anchor tension by subtracting any seabed friction
        VA = 0.0

        dZFdVF = np.sqrt(2.0 * ZF * EA_W + EA_W * EA_W) / EA_W  # inverse of vertical stiffness

        info[
            "HF"
        ] = HF  # solution to be used to start next call (these are the solved variables, may be for anchor if line is reversed)
        info["VF"] = 0.0
        info["jacobian"] = np.array([[0.0, 0.0], [0.0, dZFdVF]])
        info["LBot"] = L

    # ProfileType 0 case - slack
    elif (W > 0.0) and (CB >= 0.0) and (L >= XF - EA_W + np.sqrt(2.0 * ZF * EA_W + EA_W * EA_W)):

        ProfileType = 0
        # this is a special case that requires no iteration

        LHanging = (
            np.sqrt(2.0 * ZF * EA_W + EA_W * EA_W) - EA_W
        )  # unstretched length of line hanging vertically to seabed

        HF = 0.0
        VF = W * LHanging
        HA = 0.0
        VA = 0.0

        dZFdVF = np.sqrt(2.0 * ZF * EA_W + EA_W * EA_W) / EA_W  # inverse of vertical stiffness

        info[
            "HF"
        ] = HF  # solution to be used to start next call (these are the solved variables, may be for anchor if line is reversed)
        info["VF"] = VF
        info["jacobian"] = np.array([[0.0, 0.0], [0.0, dZFdVF]])
        info["LBot"] = L - LHanging

    # Use an iterable solver function to solve for the forces on the line
    else:

        # Initialize some commonly used terms that don't depend on the iteration:

        WL = W * L
        WEA = W * EA
        L_EA = L / EA
        CB_EA = CB / EA
        # MaxIter = 50 #int(1.0/Tol)   # Smaller tolerances may take more iterations, so choose a maximum inversely proportional to the tolerance

        # more initialization
        I = 1  # Initialize iteration counter
        FirstIter = 1  # 1 means first attempt (can be retried), 0 means it's alread been retried, -1 triggers a retry

        # make HF and VF initial guesses if either was provided as zero <<<<<<<<<<<< why does it matter if VF0 is zero??
        if HF0 <= 0 or VF0 <= 0:

            XF2 = XF * XF
            ZF2 = ZF * ZF

            if L <= np.sqrt(XF2 + ZF2):  # if the current mooring line is taut
                Lamda0 = 0.2
            else:  # The current mooring line must be slack and not vertical
                Lamda0 = np.sqrt(3.0 * ((L * L - ZF2) / XF2 - 1.0))

            HF = np.max([abs(0.5 * W * XF / Lamda0), Tol])
            # ! As above, set the lower limit of the guess value of HF to the tolerance
            VF = 0.5 * W * (ZF / np.tanh(Lamda0) + L)
        else:
            HF = 1.0 * HF0
            VF = 1.0 * VF0

        # make sure required values are non-zero
        HF = np.max([HF, Tol])
        XF = np.max([XF, Tol])
        ZF = np.max([ZF, Tol])

        # some initial values just for printing before they're filled in
        EXF = 0
        EZF = 0

        # Solve the analytical, static equilibrium equations for a catenary (or taut) mooring line with seabed interaction:
        X0 = [HF, VF]
        Ytarget = [0, 0]
        args = dict(cat=[XF, ZF, L, EA, W, CB, WL, WEA, L_EA, CB_EA], step=[0.15, 1.0, 1.5])
        # call the master solver function
        X, Y, info2 = msolve.dsolve(
            msolve.eval_func_cat,
            X0,
            Ytarget=Ytarget,
            step_func=msolve.step_func_cat,
            args=args,
            tol=Tol,
            maxIter=MaxIter,
            a_max=1.2,
        )

        # retry if it failed
        if info2["iter"] >= MaxIter - 1 or info2["oths"]["error"] == True:
            #  ! Perhaps we failed to converge because our initial guess was too far off.
            #   (This could happen, for example, while linearizing a model via large
            #   pertubations in the DOFs.)  Instead, use starting values documented in:
            #   Peyrot, Alain H. and Goulois, A. M., "Analysis Of Cable Structures,"
            #   Computers & Structures, Vol. 10, 1979, pp. 805-813:
            # NOTE: We don't need to check if the current mooring line is exactly
            #       vertical (i.e., we don't need to check if XF == 0.0), because XF is
            #       limited by the tolerance above. */

            XF2 = XF * XF
            ZF2 = ZF * ZF

            if L <= np.sqrt(XF2 + ZF2):  # if the current mooring line is taut
                Lamda0 = 0.2
            else:  # The current mooring line must be slack and not vertical
                Lamda0 = np.sqrt(3.0 * ((L * L - ZF2) / XF2 - 1.0))

            HF = np.max(
                [abs(0.5 * W * XF / Lamda0), Tol]
            )  # As above, set the lower limit of the guess value of HF to the tolerance
            VF = 0.5 * W * (ZF / np.tanh(Lamda0) + L)

            X0 = [HF, VF]
            Ytarget = [0, 0]
            args = dict(
                cat=[XF, ZF, L, EA, W, CB, WL, WEA, L_EA, CB_EA], step=[0.1, 0.8, 1.5]
            )  # step: alpha_min, alpha0, alphaR
            # call the master solver function
            X, Y, info3 = msolve.dsolve(
                msolve.eval_func_cat,
                X0,
                Ytarget=Ytarget,
                step_func=msolve.step_func_cat,
                args=args,
                tol=Tol,
                maxIter=MaxIter,
                a_max=1.1,
            )  # , dX_last=info2['dX'])

            # retry if it failed
            if info3["iter"] >= MaxIter - 1 or info3["oths"]["error"] == True:

                X0 = X
                Ytarget = [0, 0]
                args = dict(cat=[XF, ZF, L, EA, W, CB, WL, WEA, L_EA, CB_EA], step=[0.1, 1.0, 2.0])
                # call the master solver function
                X, Y, info4 = msolve.dsolve(
                    msolve.eval_func_cat,
                    X0,
                    Ytarget=Ytarget,
                    step_func=msolve.step_func_cat,
                    args=args,
                    tol=Tol,
                    maxIter=10 * MaxIter,
                    a_max=1.15,
                )  # , dX_last=info3['dX'])

                # check if it failed
                if info4["iter"] >= 10 * MaxIter - 1 or info4["oths"]["error"] == True:

                    print("Catenary solve failed on all 3 attempts.")
                    print(
                        f"Catenary({XF}, {ZF}, {L}, {EA}, {W}, CB={CB}, HF0={HF0}, VF0={VF0}, Tol={Tol}, MaxIter={MaxIter}, plots=1)"
                    )

                    print("First attempt's iterations are as follows:")
                    for i in range(info2["iter"] + 1):
                        print(
                            f" Iteration {i}: HF={info2['Xs'][i,0]: 8.4e}, VF={info2['Xs'][i,1]: 8.4e}, EX={info2['Es'][i,0]: 6.2e}, EZ={info2['Es'][i,1]: 6.2e}"
                        )

                    print("Second attempt's iterations are as follows:")
                    for i in range(info3["iter"] + 1):
                        print(
                            f" Iteration {i}: HF={info3['Xs'][i,0]: 8.4e}, VF={info3['Xs'][i,1]: 8.4e}, EX={info3['Es'][i,0]: 6.2e}, EZ={info3['Es'][i,1]: 6.2e}"
                        )

                    print("Last attempt's iterations are as follows:")
                    for i in range(info4["iter"] + 1):
                        print(
                            f" Iteration {i}: HF={info4['Xs'][i,0]: 8.4e}, VF={info4['Xs'][i,1]: 8.4e}, EX={info4['Es'][i,0]: 6.2e}, EZ={info4['Es'][i,1]: 6.2e}"
                        )

                    """
                    # plot solve performance
                    fig, ax = plt.subplots(4,1, sharex=True)
                    ax[0].plot(np.hstack([info2['Xs'][:,0], info3['Xs'][:,0], info4['Xs'][:,0]]))
                    ax[1].plot(np.hstack([info2['Xs'][:,1], info3['Xs'][:,1], info4['Xs'][:,1]]))
                    ax[2].plot(np.hstack([info2['Es'][:,0], info3['Es'][:,0], info4['Es'][:,0]]))
                    ax[3].plot(np.hstack([info2['Es'][:,1], info3['Es'][:,1], info4['Es'][:,1]]))
                    ax[0].set_ylabel("HF")
                    ax[1].set_ylabel("VF")
                    ax[2].set_ylabel("X err")
                    ax[3].set_ylabel("Z err")

                    # plot solve path
                    plt.figure()

                    #c = np.hypot(info2['Es'][:,0], info2['Es'][:,1])


                    c = np.arange(info2['iter']+1)
                    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))

                    for i in np.arange(info2['iter']):
                        plt.plot(info2['Xs'][i:i+2,0], info2['Xs'][i:i+2,1],":", c=c[i])
                    plt.plot(info2['Xs'][0,0], info2['Xs'][0,1],"o")

                    c = np.arange(info3['iter']+1)
                    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))

                    for i in np.arange(info3['iter']):
                        plt.plot(info3['Xs'][i:i+2,0], info3['Xs'][i:i+2,1], c=c[i])
                    plt.plot(info3['Xs'][0,0], info3['Xs'][0,1],"*")

                    c = np.arange(info4['iter']+1)
                    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))

                    for i in np.arange(info4['iter']):
                        plt.plot(info4['Xs'][i:i+2,0], info4['Xs'][i:i+2,1], c=c[i])
                    plt.plot(info4['Xs'][0,0], info4['Xs'][0,1],"*")

                    plt.title("Catenary solve path for troubleshooting")
                    plt.show()

                    #breakpoint()
                    """

                    raise CatenaryError("Catenary solver failed.")

                else:  # if the solve was successful,
                    info.update(info4["oths"])  # copy info from last solve into existing info dictionary

            else:  # if the solve was successful,
                info.update(info3["oths"])  # copy info from last solve into existing info dictionary

        else:  # if the solve was successful,
            info.update(info2["oths"])  # copy info from last solve into existing info dictionary

        # check for errors ( WOULD SOME NOT ALREADY HAVE BEEN CAUGHT AND RAISED ALREADY?)
        if info["error"] == True:
            breakpoint()
            # >>>> what about errors for which we can first plot the line profile?? <<<<
            raise CatenaryError("Error in Catenary computations: " + info["message"])

        if info["Zextreme"] < CB:
            info["warning"] = "Line is suspended from both ends but hits the seabed (this isn't allowed in MoorPy)"

        ProfileType = info["ProfileType"]
        HF = X[0]
        VF = X[1]
        HA = info["HA"]
        VA = info["VA"]

    # do plotting-related calculations (plots=1: show plots; plots=2: just return values)
    if plots > 0 or info["error"] == True:

        # some arrays only used for plotting each node
        s = np.linspace(
            0, L, nNodes
        )  #  Unstretched arc distance along line from anchor to each node where the line position and tension can be output (meters)
        X = np.zeros(nNodes)  #  Horizontal locations of each line node relative to the anchor (meters)
        Z = np.zeros(nNodes)  #  Vertical   locations of each line node relative to the anchor (meters)
        Te = np.zeros(nNodes)  #  Effective line tensions at each node (N)

        # ------------------------ compute line position and tension at each node -----------------------------

        for I in range(nNodes):

            # check s values?
            if (s[I] < 0.0) or (s[I] > L):
                raise CatenaryError(
                    "Warning from Catenary:: All line nodes must be located between the anchor and fairlead (inclusive) in routine Catenary()"
                )
                # cout << "        s[I] = " << s[I] << " and L = " << L << endl;
                # return -1;

            # fully along seabed
            if ProfileType == 4:

                if (L - s[I]) * CB * W > HF:  # if this node is in the zero tension range

                    X[I] = s[I]
                    Z[I] = 0.0
                    Te[I] = 0.0

                else:  # this node rests on the seabed and the tension is nonzero

                    if L * CB * W > HF:  # zero anchor tension case
                        X[I] = s[I] - 1.0 / EA * (
                            HF * (s[I] - L)
                            - CB * W * (L * s[I] - 0.5 * s[I] * s[I] - 0.5 * L * L)
                            + 0.5 * HF * HF / (CB * W)
                        )
                    else:
                        X[I] = s[I] + s[I] / EA * (HF - CB * W * (L - 0.5 * s[I]))

                    Z[I] = 0.0
                    Te[I] = HF - CB * W * (L - s[I])

            # Freely hanging line with no horizontal tension
            elif ProfileType == 0:

                if s[I] > L - LHanging:  # this node is on the suspended/hanging portion of the line

                    X[I] = XF
                    Z[I] = ZF - (L - s[I] + 0.5 * W / EA * (L - s[I]) ** 2)
                    Te[I] = W * (L - s[I])

                else:  # this node is on the seabed

                    X[I] = np.min([s[I], XF])
                    Z[I] = 0.0
                    Te[I] = 0.0

            # the other profile types are more involved
            else:

                # calculate some commonly used terms that depend on HF and VF:  AGAIN
                VFMinWL = VF - WL
                LBot = L - VF / W
                # unstretched length of line resting on seabed (Jonkman's PhD eqn 2-38), LMinVFOVrW
                HF_W = HF / W
                HF_WEA = HF / WEA
                VF_WEA = VF / WEA
                VF_HF = VF / HF
                VFMinWL_HF = VFMinWL / HF
                VF_HF2 = VF_HF * VF_HF
                VFMinWL_HF2 = VFMinWL_HF * VFMinWL_HF
                SQRT1VF_HF2 = np.sqrt(1.0 + VF_HF2)
                SQRT1VFMinWL_HF2 = np.sqrt(1.0 + VFMinWL_HF2)

                # calculate some values for the current node
                Ws = W * s[I]
                VFMinWLs = VFMinWL + Ws  # = VF - W*(L-s[I])
                VFMinWLs_HF = VFMinWLs / HF
                s_EA = s[I] / EA
                SQRT1VFMinWLs_HF2 = np.sqrt(1.0 + VFMinWLs_HF * VFMinWLs_HF)

                # No portion of the line rests on the seabed
                if ProfileType == 1:

                    X[I] = (
                        np.log(VFMinWLs_HF + SQRT1VFMinWLs_HF2) - np.log(VFMinWL_HF + SQRT1VFMinWL_HF2)
                    ) * HF_W + s_EA * HF
                    Z[I] = (SQRT1VFMinWLs_HF2 - SQRT1VFMinWL_HF2) * HF_W + s_EA * (VFMinWL + 0.5 * Ws)
                    Te[I] = np.sqrt(HF * HF + VFMinWLs * VFMinWLs)

                # A portion of the line rests on the seabed and the anchor tension is nonzero
                elif ProfileType == 2:

                    if s[I] <= LBot:  # // .TRUE. if this node rests on the seabed and the tension is nonzero

                        X[I] = s[I] + s_EA * (HF + CB * VFMinWL + 0.5 * Ws * CB)
                        Z[I] = 0.0
                        Te[I] = HF + CB * VFMinWLs

                    else:  # // LBot < s <= L:  ! This node must be above the seabed

                        X[I] = (
                            np.log(VFMinWLs_HF + SQRT1VFMinWLs_HF2) * HF_W
                            + s_EA * HF
                            + LBot
                            - 0.5 * CB * VFMinWL * VFMinWL / WEA
                        )
                        Z[I] = (
                            (-1.0 + SQRT1VFMinWLs_HF2) * HF_W
                            + s_EA * (VFMinWL + 0.5 * Ws)
                            + 0.5 * VFMinWL * VFMinWL / WEA
                        )
                        Te[I] = np.sqrt(HF * HF + VFMinWLs * VFMinWLs)

                # A portion of the line must rest on the seabed and the anchor tension is zero
                elif ProfileType == 3:

                    if (
                        s[I] <= LBot - HF_W / CB
                    ):  # (aka Lbot - s > HF/(CB*W) ) if this node rests on the seabed and the tension is zero

                        X[I] = s[I]
                        Z[I] = 0.0
                        Te[I] = 0.0

                    elif s[I] <= LBot:  # // .TRUE. if this node rests on the seabed and the tension is nonzero

                        X[I] = (
                            s[I]
                            - (LBot - 0.5 * HF_W / CB) * HF / EA
                            + s_EA * (HF + CB * VFMinWL + 0.5 * Ws * CB)
                            + 0.5 * CB * VFMinWL * VFMinWL / WEA
                        )
                        Z[I] = 0.0
                        Te[I] = HF + CB * VFMinWLs

                    else:  #  // LBot < s <= L ! This node must be above the seabed

                        X[I] = (
                            np.log(VFMinWLs_HF + SQRT1VFMinWLs_HF2) * HF_W
                            + s_EA * HF
                            + LBot
                            - (LBot - 0.5 * HF_W / CB) * HF / EA
                        )
                        Z[I] = (
                            (-1.0 + SQRT1VFMinWLs_HF2) * HF_W
                            + s_EA * (VFMinWL + 0.5 * Ws)
                            + 0.5 * VFMinWL * VFMinWL / WEA
                        )
                        Te[I] = np.sqrt(HF * HF + VFMinWLs * VFMinWLs)

        # re-reverse line distributed data back to normal if applicable
        if reverseFlag == 1:
            s = L - s[::-1]
            X = XF - X[::-1]
            Z = Z[::-1] - ZF  # remember ZF still has a flipped sign right now
            Te = Te[::-1]

        #   print("End 1 Fx "+str(HA))
        #   print("End 1 Fy "+str(VA))
        #   print("End 2 Fx "+str(-HF))
        #   print("End 2 Fy "+str(-VF))
        #   print("Scope is "+str(XF-LBot))

        if plots == 2 or info["error"] == True:  # also show the profile plot

            plt.figure()
            plt.plot(X, Z)

        # save data to info dict
        info["X"] = X
        info["Z"] = Z
        info["s"] = s
        info["Te"] = Te

    # un-swap line ends if they've been previously swapped, and apply global sign convention
    # (vertical force positive-up, horizontal force positive from A to B)
    if reverseFlag == 1:
        ZF = -ZF  # put height rise from end A to B back to negative

        FxA = HF
        FzA = -VF  # VF is positive-down convention so flip sign
        FxB = -HA
        FzB = VA
    else:
        FxA = HA
        FzA = VA
        FxB = -HF
        FzB = -VF

    # return horizontal and vertical (positive-up) tension components at each end, and length along seabed
    return (FxA, FzA, FxB, FzB, info)


def printMat(mat):
    """Print a matrix"""
    for i in range(mat.shape[0]):
        print("\t".join(["{:+8.3e}"] * mat.shape[1]).format(*mat[i, :]))


def printVec(vec):
    """Print a vector"""
    print("\t".join(["{:+8.3e}"] * len(vec)).format(*vec))


def RotationMatrix(
    x3, x2, x1
):  # this is order-z,y,x intrinsic (tait-bryan?) angles, meaning that order about the ROTATED axes
    """Calculates a rotation matrix based on order-z,y,x instrinsic angles that are about a rotated axis

    Parameters
    ----------
    x3, x2, x1: floats
        The angles that the rotated axes are from the nonrotated axes [rad]

    Returns
    -------
    R : matrix
        The rotation matrix
    """

    s1 = np.sin(x1)
    c1 = np.cos(x1)
    s2 = np.sin(x2)
    c2 = np.cos(x2)
    s3 = np.sin(x3)
    c3 = np.cos(x3)

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
        Three angles that describe the difference between the local frame and the global frame [rad]

    Returns
    -------
    rRel : array
        The relative rotated position of the point about the local frame [m]
    """

    # get rotation matrix from three provided angles
    RotMat = RotationMatrix(rot3[0], rot3[1], rot3[2])

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

    Fout = np.zeros(
        6, dtype=Fin.dtype
    )  # initialize output vector as same dtype as input vector (to support both real and complex inputs)

    Fout[:3] = Fin

    Fout[3:] = np.cross(r, Fin)

    return Fout


def set_axes_equal(ax):
    """Sets 3D plot axes to equal scale"""

    rangex = np.diff(ax.get_xlim3d())[0]
    rangey = np.diff(ax.get_ylim3d())[0]
    rangez = np.diff(ax.get_zlim3d())[0]

    ax.set_box_aspect([rangex, rangey, rangez])  # note: this may require a matplotlib update

    # ax.set_xlim3d([x - radius, x + radius])
    # ax.set_ylim3d([y - radius, y + radius])
    # ax.set_zlim3d([z - radius*0.5, z + radius*0.5])

    """
    ax.set_box_aspect([1,1,0.5])  # note: this may require a matplotlib update

    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
    x, y, z = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius*0.5, z + radius*0.5])
    """


# <<<< should make separate class for Rods
# self.RodType = RodType # 0: free to move; 1: pinned; 2: attached rigidly (positive if to something, negative if coupled)


class Line:
    """A class for any mooring line that consists of a single material"""

    def __init__(self, mooringSys, num, L, LineType, nSegs=20, cb=0, isRod=0, attachments=[0, 0]):
        """Initialize Line attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the point object
        num : int
            indentifier number
        L : float
            line unstretched length [m]
        LineType : LineType object
            LineType object that holds all the line properties
        nSegs : int, optional
            Number of segments to split the line into. The default is 40.
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

        # TODO: replace LineType input with just the name

        self.sys = mooringSys  # store a reference to the overall mooring system (instance of System class)

        self.number = num
        self.isRod = isRod

        self.L = L  # line unstretched length
        self.type = LineType.name  # string that should match a LineTypes dict entry

        self.nNodes = int(nSegs) + 1
        self.cb = float(cb)  # friction coefficient (will automatically be set negative if line is fully suspended)

        self.rA = np.zeros(3)  # end coordinates
        self.rB = np.zeros(3)
        self.fA = np.zeros(3)  # end forces
        self.fB = np.zeros(3)

        # Perhaps this could be made less intrusive by defining it using a line.addpoint() method instead, similar to pointladdline().
        self.attached = (
            attachments  # ID numbers of the Points at the Line ends [a,b] >>> NOTE: not fully supported <<<<
        )
        self.th = 0  # heading of line from end A to B
        self.HF = 0  # fairlead horizontal force saved for next solve
        self.VF = 0  # fairlead vertical force saved for next solve
        self.jacobian = []  # to be filled with the 2x2 Jacobian from Catenary
        self.info = {}  # to hold all info provided by Catenary

        self.qs = 1  # flag indicating quasi-static analysis (1). Set to 0 for time series data

        # print("Created Line "+str(self.number))

    # Load line-specific time series data from a MoorDyn output file
    def loadData(dirname):
        """Loads line-specific time series data from a MoorDyn input file"""

        self.qs = 0  # signals time series data

        # load time series data
        if isRod > 0:
            data, ch, channels, units = read_mooring_file(
                dirname, "Rod" + str(number) + ".out"
            )  # remember number starts on 1 rather than 0
        else:
            data, ch, channels, units = read_mooring_file(
                dirname, "Line" + str(number) + ".out"
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

        if isRod == 0:
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

    # figure out what time step to use for showing time series data
    def GetTimestep(self, Time):
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
                raise LineError("GetTimestep: requested time likely out of range")

        return ts

    # updates line coordinates for drawing
    def GetLineCoords(self, Time):  # formerly UpdateLine
        """Updates the line coordinates for drawing"""

        # if a quasi-static analysis, just call the Catenary code
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
                (fAH, fAV, fBH, fBV, info) = Catenary(
                    LH,
                    LV,
                    self.L,
                    self.sys.LineTypes[self.type].EA,
                    self.sys.LineTypes[self.type].w,
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
            ts = self.GetTimestep(Time)

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
                    RotationMatrix(0, np.arctan2(np.hypot(k[0], k[1]), k[2]), np.arctan2(k[1], k[0]))
                )  # <<< should fix this up at some point, MattLib func may be wrong

                # make points for appropriately sized cylinder
                d = self.sys.LineTypes[self.type].d
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

    def DrawLine2d(self, Time, ax, color="k", Xuvec=[1, 0, 0], Yuvec=[0, 0, 1]):
        """Draw the line in 2D

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

        # draw line on a 2d plot (ax must be 2d)

        linebit = []  # make empty list to hold plotted lines, however many there are

        if self.isRod > 0:

            Xs, Ys, Zs = self.GetLineCoords(Time)

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

            Xs, Ys, Zs = self.GetLineCoords(Time)

            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs * Xuvec[0] + Ys * Xuvec[1] + Zs * Xuvec[2]
            Ys2d = Xs * Yuvec[0] + Ys * Yuvec[1] + Zs * Yuvec[2]

            linebit.append(ax.plot(Xs2d, Ys2d, lw=1, color=color))

        self.linebit = linebit  # can we store this internally?

        return linebit

    def DrawLine(self, Time, ax, color="k"):
        """Draw the line

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

        # draw line in 3d for first time (ax must be 2d)

        linebit = []  # make empty list to hold plotted lines, however many there are

        if self.isRod > 0:

            Xs, Ys, Zs = self.GetLineCoords(Time)

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

            Xs, Ys, Zs = self.GetLineCoords(Time)

            linebit.append(ax.plot(Xs, Ys, Zs, color=color))

            # drawing water velocity vectors (not for Rods for now) <<< should handle this better (like in GetLineCoords) <<<
            if self.qs == 0:
                ts = self.GetTimestep(Time)
                Ux = self.Ux[ts, :]
                Uy = self.Uy[ts, :]
                Uz = self.Uz[ts, :]
                self.Ubits = ax.quiver(Xs, Ys, Zs, Ux, Uy, Uz)  # make quiver plot and save handle to line object

        self.linebit = linebit  # can we store this internally?

        self.X = np.array([Xs, Ys, Zs])

        return linebit

    def RedrawLine(self, Time):  # , linebit):
        """Update 3D line drawing based on instantaneous position"""

        linebit = self.linebit

        if self.isRod > 0:

            Xs, Ys, Zs = self.GetLineCoords(Time)

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

            Xs, Ys, Zs = self.GetLineCoords(Time)
            linebit[0][0].set_data(Xs, Ys)  # (x and y coordinates)
            linebit[0][0].set_3d_properties(Zs)  # (z coordinates)

            # drawing water velocity vectors (not for Rods for now)
            if self.qs == 0:
                ts = self.GetTimestep(Time)
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
            Determines if the previous fairlead force values will be used for the Catenary iteration. The default is False.

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

        if reset == True:  # Indicates not to use previous fairlead force values to start Catenary
            self.HF = 0  # iteration with, and insteady use the default values.

        try:
            (fAH, fAV, fBH, fBV, info) = Catenary(
                LH,
                LV,
                self.L,
                self.sys.LineTypes[self.type].EA,
                self.sys.LineTypes[self.type].w,
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

    """ Analytical Stiffness Method - In Progress - Write documentation
    def getStiffnessMatrix(self):
        '''Returns the 3 by 3 stiffness matrix of the line, corresponding to stiffness it imposes between relative motions of its two ends.'''

        K2 = np.linalg.inv(self.jacobian)              # get line's 2D stiffness matrix, which is inverse of jacobian from Catenary
        #Theta = self.LineList[LineID-1].th             # heading of line from end A to B
        # is there any need to handle directionality here, i.e. depending on which Body is attached to end A vs B? <<<
        sinTh = np.sin(self.th)
        cosTh = np.cos(self.th)

        # calculate the stiffness matrix in 3 dimensions accounting for line heading
        K3 = np.array([[ K2[0,0]*cosTh*cosTh, K2[0,0]*cosTh*sinTh, K2[0,1]*cosTh ],
                       [ K2[0,0]*cosTh*sinTh, K2[0,0]*sinTh*sinTh, K2[0,1]*sinTh ],
                       [ K2[1,0]*cosTh      , K2[1,0]*sinTh      , K2[1,1]       ]])

        return K3
    """

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

    def getMemberStiffness(self):
        """Give Member Stiffness Matrix for Line for simple model.
        In complex model, this code can be included in staticSolve()"""

        # 1 Solve for theta
        theta = np.arctan2((self.rB[1] - self.rA[1]), (self.rB[0] - self.rA[0]))
        c = np.cos(theta)
        s = np.sin(theta)

        # 2 Lookup inline and perpendicular stiffness values for this line type (assumes a certain line spacing, etc.)
        K = self.sys.LineTypes[self.type].k
        Kt = self.sys.LineTypes[self.type].kt_over_k * K

        # 3 Multiply stiffness values by transformation matrix
        K_inline = K * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ]
        )

        # Force in y direction from displacement in y direction caused by tension in x direction
        K_perpendicular = Kt * np.array(
            [
                [s * s, -c * s, -s * s, c * s],
                [-c * s, c * c, c * s, -c * c],
                [-s * s, c * s, s * s, -c * s],
                [c * s, -c * c, -c * s, c * c],
            ]
        )

        # self.MemberStiffness2 = self.kt + np.array([[-c*s, -s**2, c*s, s**2], [c**2, c*s, -c**2, -c*s], [c*s, s**2, -c*s, -s**2], [-c**2, -c*s, c**2, c*s]])
        # Note: Force in x direction from displacement in y direction caused by tension in x direction is neglected as second-order

        return K_inline + K_perpendicular

    """
    def getTensionValues(self): <<<<<< rename this!!!
        '''Define Line values for k, kt, kt_over_k, L_xy, and t)'''

        self.k = self.sys.LineTypes[self.type].k                    #in-line stiffness
        self.kt_over_k = self.sys.LineTypes[self.type].kt_over_k    #assumed ratio between in-line stiffness and perpendicular stiffness
        self.kt = self.k * self.kt_over_k                           #perpendicular stiffness

        self.L_xy = np.linalg.norm(self.rB[:2] - self.rA[:2])       #x-y spacing of line
        self.t = self.kt * self.L_xy                                #Tension Calculated fromm perpendicular stiffness
    """


class Point:
    """A class for any object in the mooring system that can be described by three translational coorindates"""

    def __init__(self, mooringSys, num, type, r, m=0, v=0, fExt=np.zeros(3)):
        """Initialize Point attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the point object
        num : int
            indentifier number
        type : int
            the body type: 0 free to move, 1 fixed, -1 coupled externally
        r : array
            x,y,z coorindate position vector [m].
        m : float, optional
            mass [kg]. The default is 0.
        v : float, optional
            volume. The default is 0.
        fExt : array, optional
            applied external force vector in global orientation (not including weight/buoyancy). The default is np.zeros(3).
        attached: list, int
            list of ID numbers of any Lines attached to the Point
        attachedEndB: list, int
            list that specifies which end of the Line is attached (1: end B, 0: end A)

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

        self.attached = []  # ID numbers of any Lines attached to the Point
        self.attachedEndB = []  # specifies which end of the line is attached (1: end B, 0: end A)

        # print("Created Point "+str(self.number))

    def addLine(self, lineID, endB):  # <<< should rename to attachLine
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
        if len(r) == 3:
            self.r = np.array(r, dtype=np.float_)
        else:
            raise ValueError(
                f"Point setPosition method requires an argument of size 3, but size {len(r):d} was provided"
            )

        # handle case of Point resting on seabed
        if self.r[2] <= -self.sys.depth:
            self.r[2] = -self.sys.depth  # don't let it sink below the seabed

        # update the position of any attached Line ends
        for LineID, endB in zip(self.attached, self.attachedEndB):
            self.sys.LineList[LineID - 1].setEndPosition(self.r, endB)

    def getForces(self, lines_only=False):
        """Sums the forces on the Point, including its own plus those of any attached Lines.

        Parameters
        ----------
        lines_only : boolean, optional
            An option for calculating forces from just the mooring lines or not. The default is False.

        Returns
        -------
        f : array
            The force vector applied to the point in its current position [N]

        """

        f = np.zeros(3)

        if lines_only == False:

            f[2] += -self.m * self.sys.g  # add weight

            if self.r[2] < -1:  # add buoyancy if fully submerged
                f[2] += self.v * self.sys.rho * self.sys.g
            elif self.r[2] < 1:  # imagine a +/-1 m band at z=0 where buoyancy tapers to zero
                f[2] += self.v * self.sys.rho * self.sys.g * (0.5 - 0.5 * self.r[2])

            f += np.array(self.fExt)  # add external forces

        # add forces from attached lines
        for LineID, endB in zip(self.attached, self.attachedEndB):
            f += self.sys.LineList[LineID - 1].getEndForce(endB)

        # handle case of Point resting on seabed
        if self.r[2] <= -self.sys.depth and f[2] < 0:  # if the net force is downward
            f[2] = 0.0  # assume the seabed supports this weight, so zero net vertical force

        return f

    def getStiffness(self, X1, dx=0.01):
        """Gets the stiffness matrix of Point due only to mooring lines with all other objects free to equilibrate.

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

        # set this Point's type to fixed so mooring system equilibrium response to its displacements can be found
        type0 = self.type  # store original type to restore later
        self.type = 1  # set type to 1 (not free) so that it won't be adjusted when finding equilibrium

        # ensure this Point is positioned at the desired linearization point
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium()  # find equilibrium of mooring system given this Point in current position
        f = self.getForces(lines_only=True)  # get the net 6DOF forces/moments from any attached lines

        # Build a stiffness matrix by perturbing each DOF in turn
        K = np.zeros([3, 3])

        for i in range(len(K)):
            X2 = X1 + np.insert(
                np.zeros(2), i, dx
            )  # calculate perturbed Point position by adding dx to DOF in question
            self.setPosition(X2)  # perturb this Point's position
            self.sys.solveEquilibrium()  # find equilibrium of mooring system given this Point's new position
            f_2 = self.getForces(lines_only=True)  # get the net 3DOF forces/moments from any attached lines

            K[:, i] = -(f_2 - f) / dx  # get stiffness in this DOF via finite difference and add to matrix column

        # ----------------- restore the system back to previous positions ------------------
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium()  # find equilibrium of mooring system given this Point in current position
        self.type = type0  # restore the Point's type to its original value

        return K

    """ Analytical Stiffness Method - In Progress - Write documentation
    def getStiffnessA(self):
        '''Get analytical stiffness matrix of Point only due to mooring lines with other objects fixed.
        '''

        print("Getting Point "+str(self.number)+" stiffness matrix...")

        K = np.zeros([3,3])

        # add forces from attached lines
        for LineID,endB in zip(self.attached,self.attachedEndB):

            # get the key properties from the line object
            Line = self.sys.LineList[LineID-1]
            jac = np.linalg.inv(Line.jacobian)
            sinTh = np.sin(Line.th)
            cosTh = np.cos(Line.th)

            # calculate the stiffness contribution (this should be the same on either line end except for anchors)
            K += np.array([[ jac[0,0]*cosTh*cosTh, jac[0,0]*cosTh*sinTh, jac[0,1]*cosTh ],
                           [ jac[0,0]*cosTh*sinTh, jac[0,0]*sinTh*sinTh, jac[0,1]*sinTh ],
                           [ jac[1,0]*cosTh      , jac[1,0]*sinTh      , jac[1,1]       ]])

        return K
    """


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
            coorindates of metacenter relative to body reference frame [m]. The default is np.zeros(3).
        f6Ext : array, optional
            applied external forces and moments vector in global orientation (not including weight/buoyancy). The default is np.zeros(6).
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
        self.AWP = AWP  # waterplane area - used for hydrostatic heave stiffness if nonzero
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

        self.sharedLineTheta = []
        self.fairR = 0.0

        # print("Created Body "+str(self.number))

    def addPoint(self, pointID, rAttach):
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

        # update the position of any attached Points
        for PointID, rPointRel in zip(self.attachedP, self.rPointRel):

            rPoint = transformPosition(rPointRel, r6)
            self.sys.PointList[PointID - 1].setPosition(rPoint)

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

            fPoint = self.sys.PointList[PointID - 1].getForces(lines_only=lines_only)  # get net force on attached Point
            rPoint_rotated = rotatePosition(
                rPointRel, self.r6[3:]
            )  # relative position of Point about body ref point in unrotated reference frame
            f6 += translateForce3to6DOF(
                rPoint_rotated, fPoint
            )  # add net force and moment resulting from its position to the Body

        # All forces and moments on the body should now be summed, and are in global/unrotated orientations.

        # For application to the body DOFs, convert the moments to be about the body's local/rotated x/y/z axes
        rotMat = RotationMatrix(*self.r6[3:])  # get rotation matrix for body
        moment_about_body_ref = np.matmul(
            rotMat.T, f6[3:]
        )  # transform moments so that they are about the body's local/rotated axes
        f6[3:] = moment_about_body_ref  # use these moments

        return f6

    def getStiffness(self, X1, dx=0.01):
        """
        Gets the stiffness matrix of a Body due only to mooring lines with all other objects free to equilibriate.
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

        # set this Body's type to fixed so mooring system equilibrium response to its displacements can be found
        type0 = self.type  # store original type to restore later
        self.type = 1  # set type to 1 (not free) so that it won't be adjusted when finding equilibrium

        # ensure this Body is positioned at the desired linearization point
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium()  # find equilibrium of mooring system given this Body in current position
        f6 = self.getForces(lines_only=True)  # get the net 6DOF forces/moments from any attached lines

        # Build a stiffness matrix by perturbing each DOF in turn
        K = np.zeros([6, 6])

        for i in range(len(K)):
            X2 = X1 + np.insert(np.zeros(5), i, dx)  # calculate perturbed Body position by adding dx to DOF in question
            self.setPosition(X2)  # perturb this Body's position
            self.sys.solveEquilibrium()  # find equilibrium of mooring system given this Body's new position
            f6_2 = self.getForces(lines_only=True)  # get the net 6DOF forces/moments from any attached lines

            K[:, i] = -(f6_2 - f6) / dx  # get stiffness in this DOF via finite difference and add to matrix column

        # ----------------- restore the system back to previous positions ------------------
        self.setPosition(X1)  # set position to linearization point
        self.sys.solveEquilibrium()  # find equilibrium of mooring system given this Body in current position
        self.type = type0  # restore the Body's type to its original value

        return K

    """ Analytical Stiffness Method - In Progress - Write documentation
    def getStiffnessA(self):
        '''Get analytical stiffness matrix of Body only due to mooring lines with other objects fixed.
        '''

        print("Getting Body "+str(self.number)+" stiffness matrix...")

        K = np.zeros([6,6])

        for PointID,rPointRel in zip(self.attachedP,self.rPointRel):

            K3 = self.sys.PointList[PointID-1].getStiffnessA()
            r = rotatePosition(rPointRel, self.r6[3:])    # relative position of Point about body ref point in unrotated reference frame

            # following are from functions getH and translateMatrix3to6
            H = np.array([[    0, r[2],-r[1]],
                          [-r[2],    0, r[0]],
                          [ r[1],-r[0],    0]])
            K[:3,:3] += K3
            K[:3,3:] += np.matmul(K3, H)          # only add up one off-diagonal sub-matrix for now, then we'll mirror at the end
            K[3:,3:] += np.matmul(np.matmul(H, K3), H.T)


        # copy over other off-diagonal sub-matrix
        K[3:,:3] = K[:3,3:].T

        return K
    """

    def draw(self, ax):
        """Draws the reference axis of the body

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.

        Returns
        -------
        linebit : TYPE
            DESCRIPTION.

        """

        linebit = []  # make empty list to hold plotted lines, however many there are

        rx = transformPosition(np.array([5, 0, 0]), self.r6)
        ry = transformPosition(np.array([0, 5, 0]), self.r6)
        rz = transformPosition(np.array([0, 0, 5]), self.r6)

        linebit.append(ax.plot([self.r6[0], rx[0]], [self.r6[1], rx[1]], [self.r6[2], rx[2]], color="r"))
        linebit.append(ax.plot([self.r6[0], ry[0]], [self.r6[1], ry[1]], [self.r6[2], ry[2]], color="g"))
        linebit.append(ax.plot([self.r6[0], rz[0]], [self.r6[1], rz[1]], [self.r6[2], rz[2]], color="b"))

        self.linebit = linebit  # can we store this internally?

        return linebit

    def redraw(self):
        """Redraws the reference axis of the body

        Returns
        -------
        linebit : TYPE
            DESCRIPTION.

        """

        linebit = self.linebit

        rx = transformPosition(np.array([5, 0, 0]), self.r6)
        ry = transformPosition(np.array([0, 5, 0]), self.r6)
        rz = transformPosition(np.array([0, 0, 5]), self.r6)

        linebit[0][0].set_data([self.r6[0], rx[0]], [self.r6[1], rx[1]])
        linebit[0][0].set_3d_properties([self.r6[2], rx[2]])
        linebit[1][0].set_data([self.r6[0], ry[0]], [self.r6[1], ry[1]])
        linebit[1][0].set_3d_properties([self.r6[2], ry[2]])
        linebit[2][0].set_data([self.r6[0], rz[0]], [self.r6[1], rz[1]])
        linebit[2][0].set_3d_properties([self.r6[2], rz[2]])

        return linebit


class LineType:
    """A class to hold the various properties of a mooring line type"""

    def __init__(self, name, d, massden, EA, MBL=0.0, cost=0.0, notes="", shared=False, k=0, kt_over_k=0):
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
        k : float, optional
            simple-system line stiffness. The default is 0.
        kt_over_k : float, optional
            simple-system line stiffness from tension to inline stiffness ratio. The default is 0.

        Returns
        -------
        None.

        """
        self.name = name  # identifier string
        self.d = d  # volume-equivalent diameter [m]
        self.mlin = massden  # linear desnity [kg/m]
        self.w = (massden - np.pi / 4 * d * d * 1025) * 9.81  # wet weight [N]  <<<<<<<<< should this be [N/m]? >>>>>>>>
        self.EA = EA  # stiffness [N]
        self.MBL = MBL  # minimum breaking load [N]
        self.cost = cost  # material cost of line per unit length [$/m]
        self.notes = notes  # optional notes/description string

        # simple system additional variables (living here temporarily)
        self.k = k  # effective inline horizontal stiffness [N/m]
        self.kt_over_k = kt_over_k  # ratio of perpendicular to inline effective stiffness coefficients [-] Kt = t/L
        self.t = 0  # horizontal tension at undisplaced position [N]
        self.k_over_w = 0  # effective inline horizontal stiffness per unit wet weight [N/m/(N/m)]
        self.t_over_w = 0  # effective inline horizontal stiffness per unit wet weight [N/m/(N/m)]
        self.Lxy = 0.0  # expected horizontal spacing of each line of this type [m]
        self.shared = shared  # set to True for shared line, False for anchored line


class System:
    """A class for the whole mooring system"""

    def __init__(self, file="", depth=0, rho=1025, g=9.81):
        """Creates an empty MoorPy mooring system data structure, and will read an input file if provided.

        Parameters
        ----------
        file : string, optional
            An input file, usually a MoorDyn input file, that can be read into a MoorPy system. The default is "".
        depth : float, optional
            Water depth of the system. The default is 0.
        rho : float, optional
            Water density of the system. The default is 1025.
        g : TYPE, optional
            Gravity of the system. The default is 9.81.

        Returns
        -------
        None.

        """

        # lists to hold mooring system objects
        self.BodyList = []
        # self.RodList = []    <<< TODO: add support for Rods eventually, for compatability with MoorDyn systems
        self.PointList = []
        self.LineList = []
        self.LineTypes = {}

        # the ground body (number 0, type 1[fixed]) never moves but is the parent of all anchored things
        self.groundBody = Body(self, 0, 1, np.zeros(6))  # <<< implementation not complete

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

    def addBody(self, mytype, r6, m=0, v=0, rCG=np.zeros(3), AWP=0, rM=np.zeros(3), f6Ext=np.zeros(6)):
        """Convenience function to add a Body to a mooring system."""
        self.BodyList.append(
            Body(self, len(self.BodyList) + 1, mytype, r6, m=m, v=v, rCG=rCG, AWP=AWP, rM=rM, f6Ext=f6Ext)
        )

        # TODO: display output message

    def addPoint(self, mytype, r, m=0, v=0, fExt=np.zeros(3)):
        """Convenience function to add a Point to a mooring system."""

        self.PointList.append(Point(self, len(self.PointList) + 1, mytype, r, m=m, v=v, fExt=fExt))

        # print("Created Point "+str(self.PointList[-1].number))
        # Including this print line, prints out the same statement twice. A print statement is already called in the Point init method

    def addLine(self, lUnstr, type_string, nSegs=20):
        """Convenience function to add a Line to a mooring system."""
        self.LineList.append(Line(self, len(self.LineList) + 1, lUnstr, self.LineTypes[type_string], nSegs=nSegs))

        # print("Created Line "+str(self.LineList[-1].number))
        # Including this print line, prints out the same statement twice. A print statement is already called in the Line init method

    def addLineType(self, type_string, d, massden, EA):
        """Convenience function to add a LineType to a mooring system."""
        self.LineTypes[type_string] = LineType(type_string, d, massden, EA)

    def load(self, filename):
        """Loads a MoorPy System from a MoorDyn-style input file"""

        # create/empty the lists to start with

        RodDict = {}  # create empty dictionary for rod types
        self.LineTypes = {}  # create empty dictionary for line types

        # ensure the mooring system's object lists are empty before adding to them
        self.BodyList = []
        # self.RodList  = []
        self.PointList = []
        self.LineList = []

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
                        self.LineTypes[entries[0]] = LineType(
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
                    line.upper().count("BODY LIST") > 0 or line.upper().count("BODY PROPERTIES") > 0
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

                        r6 = np.array(entries[1:7], dtype=float)  # initial position and orientation [m, rad]
                        r6[3:] = r6[3:] * np.pi / 180.0  # convert from deg to rad
                        rCG = np.array(entries[7:10], dtype=float)  # location of body CG in body reference frame [m]
                        m = np.float_(entries[10])  # mass, centered at CG [kg]
                        v = np.float_(entries[11])  # volume, assumed centered at reference point [m^3]

                        self.BodyList.append(Body(self, num, bodyType, r6, m=m, v=v, rCG=rCG))

                        line = next(f)

                # get properties of each rod
                if line.count("---") > 0 and (
                    line.upper().count("ROD LIST") > 0 or line.upper().count("ROD PROPERTIES") > 0
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
                    line.upper().count("POINT LIST") > 0 or line.upper().count("POINT PROPERTIES") > 0
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
                            self.groundBody.addPoint(num, entries[2:5])

                        elif "body" in entry1:
                            pointType = 1
                            # attach to body here
                            BodyID = int("".join(filter(str.isdigit, entry1)))
                            rRel = np.array(entries[2:5], dtype=float)
                            self.BodyList[BodyID - 1].addPoint(num, rRel)

                        elif ("fair" in entry1) or ("ves" in entry1):
                            pointType = -1
                            # attach to a generic platform body (and make it if it doesn't exist)
                            if len(self.BodyList) > 1:
                                raise ValueError(
                                    "Generic Fairlead/Vessel-type points aren't supported when bodies are defined."
                                )
                            if len(self.BodyList) == 0:
                                # print("Adding a body to attach fairlead points to.")
                                self.BodyList.append(Body(self, 1, 0, np.zeros(6)))  # , m=m, v=v, rCG=rCG) )

                            rRel = np.array(entries[2:5], dtype=float)
                            self.BodyList[0].addPoint(num, rRel)

                        elif ("con" in entry1) or ("free" in entry1):
                            pointType = 0
                        else:
                            print("Point type not recognized")

                        r = np.array(entries[2:5], dtype=float)
                        m = np.float_(entries[5])
                        v = np.float_(entries[6])
                        fExt = np.array(entries[7:10], dtype=float)

                        self.PointList.append(Point(self, num, pointType, r, m=m, v=v, fExt=fExt))

                        line = next(f)

                # get properties of each line
                if line.count("---") > 0 and (
                    line.upper().count("LINE LIST") > 0 or line.upper().count("LINE PROPERTIES") > 0
                ):
                    line = next(f)  # skip this header line, plus channel names and units lines
                    line = next(f)
                    line = next(f)
                    while line.count("---") == 0:
                        entries = line.split()

                        # print(entries)

                        num = np.int(entries[0])
                        # dia    = LineTypes[entries[1]].d # find diameter based on specified rod type string
                        lUnstr = np.float_(entries[2])
                        nSegs = np.int(entries[3])
                        # w = LineTypes[entries[1]].w  # line wet weight per unit length
                        # EA= LineTypes[entries[1]].EA

                        # LineList.append( Line(dirName, num, lUnstr, dia, nSegs) )
                        self.LineList.append(
                            Line(
                                self,
                                num,
                                lUnstr,
                                self.LineTypes[entries[1]],
                                nSegs=nSegs,
                                attachments=[np.int(entries[4]), np.int(entries[5])],
                            )
                        )

                        # attach ends
                        self.PointList[np.int(entries[4]) - 1].addLine(num, 0)
                        self.PointList[np.int(entries[5]) - 1].addLine(num, 1)

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
                        elif entries[1] == "WtrDpth":
                            self.depth = np.float_(entry0)
                        elif entry1 == "rho" or entry1 == "wtrdnsty":
                            self.rho = np.float_(entry0)

                        line = next(f)

            f.close()  # close data file

        # any error check? <<<

        print(f"Mooring input file '{filename}' loaded successfully.")

    def parseYAML(self, data):
        """Creates a MoorPy System from a YAML dict >>> work in progress <<<"""

        # line types
        for d in data["line_types"]:
            dia = float(d["diameter"])
            w = float(d["mass_density"])
            EA = float(d["stiffness"])
            self.LineTypes[d["name"]] = LineType(d["name"], dia, w, EA)

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
                self.groundBody.addPoint(num, d["location"])

            elif "body" in entry1:
                pointType = 1
                # attach to body here
                BodyID = int("".join(filter(str.isdigit, entry1)))
                rRel = np.array(d["location"], dtype=float)
                self.BodyList[BodyID - 1].addPoint(num, rRel)

            elif ("fair" in entry1) or ("ves" in entry1):
                pointType = 1  # <<< this used to be -1.  I need to figure out a better way to deal with this for different uses! <<<<<<
                # attach to a generic platform body (and make it if it doesn't exist)
                if len(self.BodyList) > 1:
                    raise ValueError("Generic Fairlead/Vessel-type points aren't supported when bodies are defined.")
                if len(self.BodyList) == 0:
                    # print("Adding a body to attach fairlead points to.")
                    self.BodyList.append(Body(self, 1, 0, np.zeros(6)))  # , m=m, v=v, rCG=rCG) )

                rRel = np.array(d["location"], dtype=float)
                self.BodyList[0].addPoint(num, rRel)

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

            self.PointList.append(Point(self, num, pointType, r, m=m, v=v))

        # lines
        for i, d in enumerate(data["lines"]):

            num = i + 1

            lUnstr = np.float_(d["length"])

            self.LineList.append(Line(self, num, lUnstr, self.LineTypes[d["type"]]))

            # attach ends (name matching here)
            self.PointList[pointDict[d["endA"]]].addLine(num, 0)
            self.PointList[pointDict[d["endB"]]].addLine(num, 1)

        # get options entries
        if "water_depth" in data:
            self.depth = data["water_depth"]

        if "rho" in data:
            self.rho = data["rho"]
        elif "water_density" in data:
            self.rho = data["water_density"]

    def unload(self, fileName, **kwargs):
        """Unloads a MoorPy system into a MoorDyn-style input file"""
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

        # Body Properties (for each body in bodylist)
        #! Add Comments
        IX = 0
        IY = 0
        IZ = 0
        CdA_xyz = [0, 0, 0]
        Ca_xyz = [0, 0, 0]

        # Rod List Properties

        # Point Properties (for each point in pointlist)
        #! Add Comments
        CdA = 0
        Ca = 0

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

        print("attempting to write " + fileName + " using MoorDyn v?.??")
        # Array to add strings to for each line of moordyn input file
        L = []

        # Input File Header
        L.append("------------------- MoorDyn v?.??. Input File ------------------------------------")
        if "description" in locals():
            L.append("MoorDyn input for " + description)
        else:
            L.append("MoorDyn input")
        #!
        L.append("{:5}    Echo      - echo the input file data (flag)".format(str(Echo).upper()))

        # Line Dictionary Header
        L.append("---------------------- LINE DICTIONARY -----------------------------------------------------")
        L.append("LineType  Diam    MassDenInAir   EA        cIntDamp     EI     Can     Cat    Cdn     Cdt")
        L.append("(-)       (m)       (kg/m)       (N)        (Pa-s)    (N-m^2)  (-)     (-)    (-)     (-)")

        # Line Dicationary Table
        for key in self.LineTypes:
            # for key,value in self.LineTypes.items(): (Another way to iterate through dictionary)
            L.append(
                "{:<10}{:<10.4f}{:<10.4f} {:<10.5e} ".format(
                    key, self.LineTypes[key].d, self.LineTypes[key].mlin, self.LineTypes[key].EA
                )
                + "{:<11.4f}{:<7.3f}{:<8.3f}{:<7.3f}{:<8.3f}{:<8.3f}".format(cIntDamp, EI, Can, Cat, Cdn, Cdt)
            )

        # Rod Dictionary Header
        L.append("--------------------- ROD DICTIONARY -----------------------------------------------------")
        L.append("RodType  Diam    MassDenInAir   Can     Cat    Cdn     Cdt ")
        L.append("(-)       (m)       (kg/m)      (-)     (-)    (-)     (-)  ")

        """
        # Rod Dictionary Table
        for i, rod_type in enumerate(self.LineTypes,start=1):
        """

        # Body List Header
        L.append("----------------------- BODY LIST -----------------------------------")
        L.append(
            "BodyID      X0   Y0   Z0    r0    p0    y0    Xcg   Ycg   Zcg     M      V        IX       IY       IZ     CdA-x,y,z Ca-x,y,z"
        )
        L.append(
            "   (-)      (m)  (m)  (m)  (deg) (deg) (deg)  (m)   (m)   (m)    (kg)   (m^3)  (kg-m^2) (kg-m^2) (kg-m^2)   (m^2)      (-)"
        )

        # Body List Table
        for Body in self.BodyList:
            L.append(
                "    {:<7d} {:<5.2f} {:<5.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<8.2f} {:<7.2f} {:<11.2f}".format(
                    Body.number,
                    Body.r6[0],
                    Body.r6[1],
                    Body.r6[2],
                    np.rad2deg(Body.r6[3]),
                    np.rad2deg(Body.r6[4]),
                    np.rad2deg(Body.r6[5]),
                    Body.rCG[0],
                    Body.rCG[1],
                    Body.rCG[2],
                    Body.m,
                    Body.v,
                )
                + "{:<9d}{:<9d}{:<7d}{:<2d}{:<2d}{:<8d}{:<1d}".format(
                    IX, IY, IZ, CdA_xyz[0], CdA_xyz[1], CdA_xyz[2], Ca_xyz[0], Ca_xyz[1], Ca_xyz[2]
                )
            )

        # Rod Properties Header
        L.append("---------------------- ROD LIST --------------------")
        L.append("RodID  Type/BodyID  RodType   Xa   Ya   Za   Xb   Yb   Zb  NumSegs  Flags/Outputs")
        L.append("(-)      (-)         (-)      (m)  (m)  (m)  (m)  (m)  (m)    (-)      (-)   ")

        """
        #Rod Properties Table
        """

        # Point Properties Header
        L.append("---------------------- POINT LIST -----------------------------------------------------")
        L.append("Node      Type      X        Y        Z        M      V      FX   FY   FZ   CdA   Ca ")
        L.append("(-)       (-)      (m)      (m)      (m)      (kg)   (m^3)  (kN) (kN) (kN)  (m2)  ()")

        # Point Properties Table
        for Point in self.PointList:
            point_pos = Point.r  # Define point position in global reference frame
            if Point.type == 1:  # Point is Fized or attached (anch, body, fix)
                point_type = "Fixed"

                # import pdb
                # pdb.set_trace()
                # Check if the point is attached to body
                for Body in self.BodyList:
                    for Attached_Point in Body.attachedP:
                        if Attached_Point == Point.number:
                            point_type = "Body" + str(Body.number)
                            point_pos = Body.rPointRel[
                                Body.attachedP.index(Attached_Point)
                            ]  # Redefine point position in the body reference frame

            if Point.type == 0:  # Point is Coupled Externally (con, free)
                point_type = "Connect"

            if Point.type == -1:  # Point is free to move (fair, ves)
                point_type = "Vessel"

            L.append(
                "{:<9d} {:8} {:<20.15f} {:<20.15f} {:<11.4f} {:<7.2f} {:<7.2f} {:<5.2f} {:<5.2f} {:<7.2f}".format(
                    Point.number,
                    point_type,
                    point_pos[0],
                    point_pos[1],
                    point_pos[2],
                    Point.m,
                    Point.v,
                    Point.fExt[0],
                    Point.fExt[1],
                    Point.fExt[2],
                )
                + " {:<6d} {:<1d}".format(CdA, Ca)
            )

        # Line Properties Header
        L.append("---------------------- LINE LIST -----------------------------------------------------")
        L.append("Line     LineType  UnstrLen  NumSegs   NodeAnch  NodeFair  Flags/Outputs")
        L.append("(-)       (-)       (m)        (-)       (-)       (-)       (-)")

        # Line Properties Table
        # (Create a ix2 array of connection points from a list of m points)
        connection_points = np.empty([len(self.LineList), 2])  # First column is Anchor Node, second is Fairlead node
        for point_ind, point in enumerate(self.PointList, start=1):  # Loop through all the points
            for (line, line_pos) in zip(
                point.attached, point.attachedEndB
            ):  # Loop through all the lines #s connected to this point
                if line_pos == 0:  # If the A side of this line is connected to the point
                    connection_points[line - 1, 0] = point_ind  # Save as as an Anchor Node
                    # connection_points[line -1,0] = self.PointList.index(point) + 1
                elif line_pos == 1:  # If the B side of this line is connected to the point
                    connection_points[line - 1, 1] = point_ind  # Save as a Fairlead node
                    # connection_points[line -1,1] = self.PointList.index(point) + 1
        # Populate text
        for i in range(len(self.LineList)):
            L.append(
                "{:<9d}{:<11}{:<11.2f}{:<11d}{:<10d}{:<10d}{}".format(
                    self.LineList[i].number,
                    self.LineList[i].type,
                    self.LineList[i].L,
                    self.LineList[i].nNodes - 1,
                    int(connection_points[i, 0]),
                    int(connection_points[i, 1]),
                    flag,
                )
            )

        # Solver Options Header
        L.append("---------------------- SOLVER OPTIONS ----------------------------------------")

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

        print("Successfully written " + fileName + " input file using MoorDyn v?.??")

    def initialize(self, plots=0):
        """Initializes the mooring system objects to their initial positions"""

        self.nDOF = 0
        self.nCpldDOF = 0

        for Body in self.BodyList:
            Body.setPosition(Body.r6)
            if Body.type == 0:
                self.nDOF += 6
            if Body.type == -1:
                self.nCpldDOF += 6

        for Point in self.PointList:
            Point.setPosition(Point.r)
            if Point.type == 0:
                self.nDOF += 3
            if Point.type == -1:
                self.nCpldDOF += 3

        for Line in self.LineList:
            Line.staticSolve()

        for Point in self.PointList:
            f = Point.getForces()

        for Body in self.BodyList:
            f = Body.getForces()

        # draw initial mooring system if desired
        if plots == 1:
            self.plot(title="Mooring system at initialization")

    def transform(self, trans=[0, 0], rot=0, scale=[1, 1]):
        """Applies translations, rotations, and/or stretching to the mooring system positions
        Parameters
        ----------
        trans : array
            how far to shift the whole mooring system in x and y directions (m)
        rot : float
            how much to rotate the entire mooring system in the yaw direction (degrees)
        scale : array
            how much to scale the mooring system x and y dimensions by (relative). Default is unity. (NOT IMPLEMENTED)
        """

        rotMat = RotationMatrix(0, 0, rot * np.pi / 180.0)
        tVec = np.array([trans[0], trans[1], 0.0])

        # little functions to transform r or r6 vectors in place
        def transform3(X):
            Xrot = np.matmul(rotMat, X)
            X = Xrot + tVec
            return X

        def transform6(X):
            Xrot = np.matmul(rotMat, X[:3])
            X = np.hstack([Xrot + tVec, X[3], X[4], X[5] + rot * np.pi / 180.0])
            return X

        # update positions of all objects
        for body in self.BodyList:
            body.r6 = transform6(body.r6)
        for point in self.PointList:
            point.r = transform3(point.r)

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
            The DOF values.

        dX : array, if dXvals is provided
            A vector with the corresponding dXvals value for each returned DOF value.

        """

        if not DOFtype in ["free", "coupled", "both"]:
            raise ValueError("getPositions called with invalid DOFtype input. Must be free, coupled, or both")

        X = np.array([], dtype=np.float_)
        if len(dXvals) > 0:
            dX = []

        # gather DOFs from free bodies or points
        if DOFtype == "free" or DOFtype == "both":

            for body in self.BodyList:
                if body.type == 0:
                    X = np.append(X, body.r6)
                    if len(dXvals) > 0:
                        dX += 3 * [dXvals[-2]] + 3 * [dXvals[-1]]

            for point in self.PointList:
                if point.type == 0:
                    X = np.append(X, point.r)
                    if len(dXvals) > 0:
                        dX += 3 * [dXvals[0]]

        # self.nDOF = len(X)

        # gather DOFs from coupled bodies or points
        if DOFtype == "coupled" or DOFtype == "both":

            for body in self.BodyList:
                if body.type == -1:
                    X = np.append(X, body.r6)
                    if len(dXvals) > 0:
                        dX += 3 * [dXvals[-2]] + 3 * [dXvals[-1]]

            for point in self.PointList:
                if point.type == -1:
                    X = np.append(X, point.r)
                    if len(dXvals) > 0:
                        dX += 3 * [dXvals[0]]

        # self.nCpldDOF = len(X) - self.nDOF

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

        f = np.zeros(len(X))  # blank array to hold net reactions in all applicable DOFs
        # <<< could add a check to ensure len(X) matches nDOF, nCpldDOF or nDOF+nCpldDOF

        if not DOFtype in ["free", "coupled", "both"]:
            raise ValueError("setPositions called with invalid DOFtype input. Must be free, coupled, or both")

        # free bodies or points
        if DOFtype == "free" or DOFtype == "both":

            for body in self.BodyList:
                if body.type == 0:
                    body.setPosition(X[i : i + 6])
                    i += 6

            # update position of free Points
            for point in self.PointList:
                if point.type == 0:
                    point.setPosition(X[i : i + 3])
                    i += 3

        # coupled bodies or points
        if DOFtype == "coupled" or DOFtype == "both":

            for body in self.BodyList:
                if body.type == -1:
                    body.setPosition(X[i : i + 6])
                    i += 6

            # update position of free Points
            for point in self.PointList:
                if point.type == -1:
                    point.setPosition(X[i : i + 3])
                    i += 3

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

        # initialize force array based on DOFtype specified
        if DOFtype == "free":
            f = np.zeros(self.nDOF)
        elif DOFtype == "coupled":
            f = np.zeros(self.nCpldDOF)
        elif DOFtype == "both":
            f = np.zeros(self.nDOF + self.nCpldDOF)
        else:
            raise ValueError("getForces called with invalid DOFtype input. Must be free, coupled, or both")

        i = 0  # index used in assigning outputs in output vector

        # gather net loads from free bodies or points
        if DOFtype == "free" or DOFtype == "both":

            for body in self.BodyList:
                if body.type == 0:
                    f[i : i + 6] = body.getForces(lines_only=lines_only)
                    i += 6

            for point in self.PointList:
                if point.type == 0:
                    f[i : i + 3] = point.getForces(lines_only=lines_only)
                    i += 3

        # gather net loads from coupled bodies or points
        if DOFtype == "coupled" or DOFtype == "both":

            for body in self.BodyList:
                if body.type == -1:
                    f[i : i + 6] = body.getForces(lines_only=lines_only)
                    i += 6

            for point in self.PointList:
                if point.type == -1:
                    f[i : i + 3] = point.getForces(lines_only=lines_only)
                    i += 3

        return np.array(f)

    def mooringEq(self, X, DOFtype="free"):
        """Error function used in solving static equilibrium by calculating the forces on free objects

        Parameters
        ----------
        X : array
            A list or array containing the values of all relevant DOFs -- for bodies first, then for points.
            If type is 'both', X provides the free DOFs followed by the coupled DOFs.
        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.

        Returns
        -------
        f : array
            The forces (and moments) on all applicable DOFs in the system. f is the same size as X.

        """

        # update DOFs
        self.setPositions(X, DOFtype)

        # solve profile and forces of all lines
        for Line in self.LineList:
            Line.staticSolve()

        # get reactions in DOFs
        f = self.getForces(DOFtype)

        return f

    def solveEquilibrium(self, DOFtype="free", plots=0, rmsTol=10, maxIter=200):
        """Solves for the static equilibrium of the system using the stiffness matrix, while updating positions of all free objects.

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

        """

        # Note: this approach appears to be reliable since it has some safeguards.

        # TO DO: make an option so these functions can find equilibrium of free AND coupled objects <<<<

        # fill in some arrays for each DOF
        """
        X0 = []  # vector of current DOFs
        db = []  # step size bound

        for Body in self.BodyList:
            if Body.type==0:
                X0  += [*Body.r6 ]               # add free Body position and orientation to vector
                db  += [ 5, 5, 5, 0.3,0.3,0.3]   # put a strict bound on how quickly rotations can occur

        for Point in self.PointList:
            if Point.type==0:
                X0  += [*Point.r ]               # add free Point position to vector
                db  += [ 5., 5., 5.]                # specify maximum step size for point positions

        X0 = np.array(X0, dtype=np.float_)
        db = np.array(db, dtype=np.float_)
        """
        X0, db = self.getPositions(DOFtype=DOFtype, dXvals=[5.0, 0.3])

        n = len(X0)

        # clear some arrays to log iteration progress
        self.freeDOFs.clear()  # clear stored list of positions, so it can be refilled for this solve process
        self.Xs = []  # for storing iterations from callback fn
        self.Fs = []

        if self.display > 1:
            print(f" solveEquilibrium called for {n} DOFs (DOFtype={DOFtype})")

        # do the main iteration, using Newton's method to find the zeros of the system's not force/moment functions
        for iter in range(maxIter):
            # print('X0 is', X0)

            # first get net force vector from current position
            F0 = self.mooringEq(X0, DOFtype=DOFtype)

            # if there are no DOFs to solve equilibrium, exit (doing this after calling mooringEq so that lines are solved)
            if n == 0:
                if self.display > 1:
                    print(f"  0 DOFs to equilibrate so exiting")
                break

            if self.display > 1:
                print(f"  i{iter}, X0 {X0}, F0 {F0}")

            # log current position and resulting force (note: this may need updating based on DOFtype
            self.freeDOFs.append(X0)
            self.Xs.append(X0)
            self.Fs.append(F0)

            # check for equilibrium, and finish if condition is met
            rmse = np.linalg.norm(F0)  # root mean square of all force/moment errors
            if np.linalg.norm(F0) < rmsTol:
                # print("Equilibrium solution completed after "+str(iter)+" iterations with RMS error of "+str(rmse))
                break
            elif iter == maxIter - 1:

                if self.display > 1:
                    print(
                        "Failed to find equilibrium after " + str(iter) + " iterations, with RMS error of " + str(rmse)
                    )

                if self.display > 2:
                    for i in range(iter + 1):
                        print(f" i={i}, RMSE={np.linalg.norm(self.Fs[i]):6.2e}, X={self.Xs[i]}")

                    K = self.getSystemStiffness(DOFtype=DOFtype)

                    print("===========================")
                    print("current system stiffness:")
                    print(K)

                    print("\n Current force ")
                    print(F0)

                raise SolveError(
                    f"solveEquilibrium failed to fine equilibrium after {iter} iterations, with RMS error of {rmse}"
                )

            # get stiffness matrix
            K = self.getSystemStiffness(DOFtype=DOFtype)

            # adjust positions according to stiffness matrix to move toward net zero forces (but only a fraction of the way!)
            dX = np.matmul(np.linalg.inv(K), F0)  # calculate position adjustment according to Newton's method

            # >>> TODO: handle singular matrix error <<<

            # dXd= np.zeros(len(F0))   # a diagonal-only approach in each DOF in isolation

            for i in range(n):  # but limit adjustment to keep things under control

                # dX[i] = 0.5*dX[i] + 0.5*F0[i]/K[i,i] # alternative approach using diagonal stiffness only

                if dX[i] > db[i]:
                    dX[i] = db[i]
                elif dX[i] < -db[i]:
                    dX[i] = -db[i]

            # if iter == 6:
            #    breakpoint()

            X0 = X0 + 1.0 * dX
            # print(X0)
            # print(self.mooringEq(X0))
            # features to add:
            # - reduce Catenary error tolerance in proportion to how close we are to the solution
            # - also adapt stiffness solver perturbation size as we approach the solution

        if self.display > 1:
            print(F0)

        # show an animation of the equilibrium solve if applicable
        if plots > 0:
            self.animateSolution()

    def solveEquilibrium3(self, DOFtype="free", plots=0, rmsTol=1.0e-5, maxIter=200):
        """Solves for the static equilibrium of the system using the dsolve function approach in MoorSolve

        Parameters
        ----------

        DOFtype : string, optional
            Specifies whether to consider 'free' DOFs, 'coupled' DOFs, or 'both'. The default is 'free'.
        plots : int, optional
            Determines whether to plot the equilibrium process or not. The default is 0.
        rmsTol : float, optional
            The *relative* maximum tolerance in the calculated forces and moments.
        maxIter : int, optional
            The maximum number of interations to try to solve for equilibrium. The default is 200.

        Raises
        ------
        SolveError
            If the system fails to solve for equilirbium in the given tolerance and iteration number

        Returns
        -------
        None.

        """

        # create arrays for the positions of the objects that need to find equilibrium
        X0, db = self.getPositions(DOFtype=DOFtype, dXvals=[5.0, 0.3])

        n = len(X0)

        # clear some arrays to log iteration progress
        self.freeDOFs.clear()  # clear stored list of positions, so it can be refilled for this solve process
        self.Xs = []  # for storing iterations from callback fn
        self.Es = []

        def eval_func_equil(X, args):

            Y = self.mooringEq(X, DOFtype=DOFtype)
            oths = dict(status=1)  # other outputs - returned as dict for easy use

            return Y, oths, False

        def step_func_equil(X, args, Y, oths, Ytarget, err, tol, iter, maxIter):

            # get stiffness matrix
            K = self.getSystemStiffness(DOFtype=DOFtype)

            # adjust positions according to stiffness matrix to move toward net zero forces (but only a fraction of the way!)
            dX = np.matmul(np.linalg.inv(K), Y)  # calculate position adjustment according to Newton's method

            # but limit adjustment to keep things under control
            for i in range(n):
                if dX[i] > db[i]:
                    dX[i] = db[i]
                elif dX[i] < -db[i]:
                    dX[i] = -db[i]

            return dX

        # Call dsolve function
        X, Y, info = msolve.dsolve(eval_func_equil, X0, step_func=step_func_equil, tol=rmsTol, maxIter=maxIter)
        # Don't need to call Ytarget in dsolve because it's already set to be zeros

        if self.display > 1:
            print(X)
            print(Y)

        self.Xs = info["Xs"]  # List of positions as it finds equilibrium for every iteration
        self.Es = info[
            "Es"
        ]  # List of errors that the forces are away from 0, which in this case, is the same as the forces

        # Update equilibrium position at converged X values
        F = self.mooringEq(X, DOFtype=DOFtype)

        # Print statements if it ever reaches the maximum number of iterations
        if info["iter"] == maxIter - 1:
            if self.display > 1:

                K = self.getSystemStiffness(DOFtype=DOFtype)

                print("solveEquilibrium3 did not converge!")
                print(f"current system stiffness: {K}")
                print(f"\n Current force {F}")

            raise SolveError(
                f"solveEquilibrium3 failed to fine equilibrium after {iter} iterations, with residual forces of {F}"
            )

        # show an animation of the equilibrium solve if applicable
        if plots > 0:
            self.animateSolution()

    def getSystemStiffness(self, DOFtype="free", dx=0.1, dth=0.1, solveOption=1, plots=0):
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

        # ------------------ get the positions to linearize about -----------------------

        X1, dX = self.getPositions(DOFtype=DOFtype, dXvals=[dx, dth])

        # solve profile and forces of all lines (ensure lines are up to date)
        for Line in self.LineList:
            Line.staticSolve()

        n = len(X1)

        F1 = self.getForces(DOFtype=DOFtype)  # get mooring forces/moments about linearization point

        K = np.zeros([n, n])  # allocate stiffness matrix

        if plots > 0:
            self.freeDOFs.clear()  # clear the positions history to refill if animating this process  <<<< needs updating for DOFtype

        # ------------------------- perform linearization --------------------------------

        if solveOption == 0:  # ::: forward difference approach :::

            for i in range(n):  # loop through each DOF

                X2 = np.array(X1, dtype=np.float_)
                X2[i] += dX[i]  # perturb positions by dx in each DOF in turn
                F2p = self.mooringEq(X2, DOFtype=DOFtype)  # system net force/moment vector from positive perturbation

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

                    X2 = np.array(X1, dtype=np.float_)
                    X2[i] += dXi  # perturb positions by dx in each DOF in turn
                    F2p = self.mooringEq(
                        X2, DOFtype=DOFtype
                    )  # system net force/moment vector from positive perturbation

                    if plots > 0:
                        self.freeDOFs.append(X2.copy())

                    X2[i] -= 2.0 * dXi
                    F2m = self.mooringEq(
                        X2, DOFtype=DOFtype
                    )  # system net force/moment vector from negative perturbation

                    if plots > 0:
                        self.freeDOFs.append(X2.copy())

                    if self.display > 2:
                        print(
                            f"j = {j}  and dXi = {dXi}.   F2m, F1, and F2p are {F2m[i]:6.2f} {F1[i]:6.2f} {F2p[i]:6.2f}"
                        )

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

    def getCoupledStiffness(self, dx=0.1, dth=0.1, solveOption=1, lines_only=False, plots=0):
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

        Raises
        ------
        ValueError
            If the solveOption is not a 1 or 0

        Returns
        -------
        K : matrix
            nCpldDOF x nCpldDOF stiffness matrix of the system

        """

        if self.display > 2:
            print("Getting mooring system stiffness matrix...")

        # ------------------ get the positions to linearize about -----------------------

        # get the positions about which the system is linearized, and an array containting
        # the perturbation size in each coupled DOF of the system
        X1, dX = self.getPositions(DOFtype="coupled", dXvals=[dx, dth])

        self.solveEquilibrium()  # let the system settle into equilibrium

        F1 = self.getForces(
            DOFtype="coupled", lines_only=lines_only
        )  # get mooring forces/moments about linearization point

        K = np.zeros([self.nCpldDOF, self.nCpldDOF])  # allocate stiffness matrix

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

                if self.display > 2:
                    print(F2p)
                if plots > 0:
                    self.cpldDOFs.append(X2)

                K[:, i] = -(F2p - F1) / dX[i]  # take finite difference of force w.r.t perturbation

        elif solveOption == 1:  # ::: adaptive central difference approach :::

            nTries = 3  # number of refinements to allow -1

            for i in range(self.nCpldDOF):  # loop through each DOF

                dXi = 1.0 * dX[i]

                # potentially iterate with smaller step sizes if we're at a taut-slack transition (but don't get too small, or else numerical errors)
                for j in range(nTries):

                    X2 = np.array(X1, dtype=np.float_)
                    X2[i] += dXi  # perturb positions by dx in each DOF in turn
                    self.setPositions(X2, DOFtype="coupled")  # set the perturbed coupled DOFs
                    self.solveEquilibrium()  # let the system settle into equilibrium
                    F2p = self.getForces(
                        DOFtype="coupled", lines_only=lines_only
                    )  # get resulting coupled DOF net force/moment response

                    if plots > 0:
                        self.cpldDOFs.append(X2.copy())

                    X2[i] -= 2.0 * dXi  # now perturb from original to -dx
                    self.setPositions(X2, DOFtype="coupled")  # set the perturbed coupled DOFs
                    self.solveEquilibrium()  # let the system settle into equilibrium
                    F2m = self.getForces(
                        DOFtype="coupled", lines_only=lines_only
                    )  # get resulting coupled DOF net force/moment response

                    if plots > 0:
                        self.cpldDOFs.append(X2.copy())

                    if self.display > 2:
                        print(
                            f"j = {j}  and dXi = {dXi}.   F2m, F1, and F2p are {F2m[i]:6.2f} {F1[i]:6.2f} {F2p[i]:6.2f}"
                        )

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
                        # untill the derivatives agree better. Decrease the step size by 10X.
                        dXi = 0.1 * dXi

                K[:, i] = -0.5 * (F2p - F2m) / dXi  # take finite difference of force w.r.t perturbation
        else:
            raise ValueError("getSystemStiffness was called with an invalid solveOption (only 0 and 1 are supported)")

        # ----------------- restore the system back to previous positions ------------------
        self.mooringEq(X1, DOFtype="coupled")

        # show an animation of the stiffness perturbations if applicable
        if plots > 0:
            self.animateSolution()

        return K

    """ Analytical Stiffness Method - In Progress - Write documentation
    def getSystemStiffnessA(self):
        '''TODO: a method to calculate the system's stiffness matrix based entirely on analytic gradients from Catenary.'''

        # note: This is missing some pieces, and needs to check more.
        # So far this seems to not capture yaw stiffness for non-bridle configs...
        # it would require proper use of chain rule for the derivatives


        K = np.zeros([self.nDOF,self.nDOF])                 # allocate stiffness matrix


        # The following will go through and get the lower-triangular stiffness terms,
        # calculated as the force/moment on Body/Point 2 from translation/rotation of Body/Point 1.


        # go through DOFs, looking for lines that couple to anchors or other DOFs

        i = 0                               # start counting number of DOFs at zero

        for Body1 in self.BodyList:
            if Body1.type==0:

                # go through each attached point
                for PointID1,rPointRel1 in zip(Body1.attachedP,Body1.rPointRel):
                    Point1 = self.PointList[PointID1-1]

                    r1 = rotatePosition(rPointRel1, Body1.r6[3:])                 # relative position of Point about body ref point in unrotated reference frame
                    H1 = np.array([[     0, r1[2],-r1[1]],
                                   [-r1[2],     0, r1[0]],
                                   [ r1[1],-r1[0],     0]])

                    for LineID in Point1.attached:       # go through each attached line to the Point, looking for what its other end attached to

                        K3 = self.LineList[LineID-1].getStiffnessMatrix()

                        endFound = 0                    # simple flag to indicate when the other end's attachment has been found

                        # look through Bodies further on in the list (coupling with earlier Bodies will already have been taken care of)
                        j = i+6

                        for Body2 in self.BodyList[self.BodyList.index(Body1)+1: ]:
                            if Body2.type==0:

                                # go through each attached Point
                                for PointID2,rPointRe2l in zip(Body2.attachedP,Body2.rPointRel):
                                    Point2 = self.PointList[PointID2-1]

                                    if LineID in Point2.attached:     # if the line is also attached to this Point2 in Body2

                                        # following are analagous to what's in functions getH and translateMatrix3to6 except for cross coupling between two bodies
                                        r2 = rotatePosition(rPointRel2, Body2.r6[3:])                 # relative position of Point about body ref point in unrotated reference frame

                                        H2 = np.array([[     0, r2[2],-r2[1]],
                                                       [-r2[2],     0, r2[0]],
                                                       [ r2[1],-r2[0],     0]])

                                        K[i  :i+3, j  :j+3] += K3
                                        K[i  :i+3, j+3:j+6] += np.matmul(K3, H1)
                                        K[i+3:i+6, j  :j+3] += np.matmul(H2.T, K3)
                                        K[i+3:i+6, j+3:j+6] += np.matmul(np.matmul(H2, K3), H1.T)

                                        endFound = 1  # signal that the line has been handled so we can move on to the next thing
                                        break

                                J += 6


                        # look through free Points
                        if endFound==0:                              #  if the end hasn't already been found
                            for Point2 in self.PointList:

                                if Point2.type==0:                   # if it's a free point and
                                    if LineID in Point2.attached:    # the line is also attached to it

                                        # only add up one off-diagonal sub-matrix for now, then we'll mirror at the end
                                        K[i  :i+3, j:j+3] += K3
                                        K[i+3:i+6, j:j+3] += np.matmul(H1.T, K3)

                                        break

                                    j += 3

                                elif Point2.type==1:                 # if it's an anchor point
                                    if LineID in Point2.attached:    # the line is also attached to it

                                        # only add up one off-diagonal sub-matrix for now, then we'll mirror at the end
                                        K[i  :i+3, i  :i+3] += K3
                                        K[i+3:i+6, i  :i+3] += np.matmul(H1.T, K3)
                                        K[i+3:i+6, i+3:i+6] += np.matmul(np.matmul(H1, K3), H1.T)

                                        break

                i += 6    # moving along to the next body...


        for Point in self.PointList:
            if Point.type==0:

                for LineID in Point.attached:       # go through each attached line, looking for what it's attached to

                    # look through Points further on in the list
                    # (couplings with earlier Points or Bodies will already have been dealt with)
                    j = i+3

                    for Point2 in self.PointList[self.PointList.index(Point)+1: ]:

                        if Point2.type==0:                   # if it's a free point and
                            if LineID in Point2.attached:     # the line is also attached to it

                                K3 = self.LineList[LineID-1].getStiffnessMatrix()

                                K[i:i+3, j:j+3] += K3

                            j += 3



                #X1 += [*Point.r]            # add free Point position to vector <<< ?
                #dX += [ dx,dx,dx ]

                i += 3




        #  TODO! <<<<

        # copy over other off-diagonal sub-matrix

        # also need to add in body hydrostatic terms!



        return K
    """

    def plot(self, rbound=0, ax=None, color="k", title=""):
        """Plots the mooring system objects in their current positions

        Parameters
        ----------
        rbound : float, optional
            A bound to be placed on each axis of the plot. If 0, the bounds will be the max values on each axis. The default is 0.
        title : string, optional
            A title of the plot. The default is "".
        ax : axes, optional
            Plot on an existing set of axes
        color : string, optional
            Some way to control the color of the plot ... TBD <<<

        Returns
        -------
        fig : figure object
            To hold the axes of the plot
        ax: axis object
            To hold the points and drawing of the plot

        """

        # sort out bounds
        xs = []
        ys = []
        zs = [0, -self.depth]

        for Point in self.PointList:
            xs.append(Point.r[0])
            ys.append(Point.r[1])
            zs.append(Point.r[2])

        if rbound == 0:
            rbound = max([max(xs), max(ys), -min(xs), -min(ys)])

        # if axes not passed in, make a new figure
        if ax == None:
            fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
            ax = Axes3D(fig)
        else:
            fig = plt.gcf()  # will this work like this? <<<

        # draw things
        for body in self.BodyList:
            body.draw(ax)

        for line in self.LineList:
            line.DrawLine(0, ax, color=color)
            """
            #if (line.number % 2) == 0:
            if line.rA[2] == -self.depth:
                line.DrawLine(0, ax, color="b")
            else:
                line.DrawLine(0, ax)
            """

        # TODO @Sam - change plot limits so they look good for all example files.
        # ax.set_xlim([-rbound,rbound])
        # ax.set_ylim([-rbound,rbound])
        # ax.set_xlim([-rbound,0])
        # ax.set_ylim([-rbound/2,rbound/2])
        ax.set_zlim([-self.depth, 0])
        fig.suptitle(title)

        set_axes_equal(ax)

        # plt.show()

        return fig, ax  # return the figure and axis object in case it will be used later to update the plot

    def plot2d(self, Xuvec=[1, 0, 0], Yuvec=[0, 0, 1], ax=None, color="k", title=""):
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

        # if axes not passed in, make a new figure
        if ax == None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()  # will this work like this? <<<

        for Body in self.BodyList:
            # Body.draw(ax)
            plt.plot(Body.r6[0], Body.r6[1], "ko", markersize=2)

        for Line in self.LineList:
            Line.DrawLine2d(0, ax, color=color, Xuvec=Xuvec, Yuvec=Yuvec)

        ax.axis("equal")
        ax.set_title(title)
        # plt.show()

        return fig, ax  # return the figure and axis object in case it will be used later to update the plot

    def animateSolution(self):
        """Creates an animation of the system"""

        # first draw a plot of DOFs and forces
        x = np.array(self.Xs)
        f = np.array(self.Fs)
        fig, ax = plt.subplots(2, 1, sharex=True)
        for i in range(len(self.Fs[0])):
            ax[0].plot(x[:, i])  # <<< warning this is before scale and offset!
            ax[1].plot(f[:, i], label=i + 1)
        ax[1].legend()

        self.mooringEq(self.freeDOFs[0])  # set positions back to the first ones of the iteration process
        # ^^^^^^^ this only works for free DOF animation cases (not coupled DOF ones) <<<<<

        fig, ax = self.plot()  # make the initial plot to then animate

        ms_delay = 10000 / len(self.freeDOFs)  # time things so the animation takes 10 seconds

        line_ani = animation.FuncAnimation(
            fig, self.animate, len(self.freeDOFs), interval=ms_delay, blit=False, repeat_delay=2000
        )

        plt.show()

    def animate(self, ts):
        """Redraws mooring system positions at step ts. Currently set up in a hack-ish way to work for animations
        involving movement of either free DOFs or coupled DOFs (but not both)."""

        # following sets positions of all objects and may eventually be made into self.setPositions(self.positions[i])

        if len(self.freeDOFs) > 0:
            X = self.freeDOFs[ts]  # get freeDOFs of current instant
            type = 0
        elif len(self.cpldDOFs) > 0:
            X = self.cpldDOFs[ts]  # get freeDOFs of current instant
            type = -1
        else:
            raise ValueError("System.animate called but no animation data is saved in freeDOFs or cpldDOFs")

        # print(ts)

        i = 0  # index used to split off input positions X for each free object

        # update position of free Bodies
        for Body in self.BodyList:
            if Body.type == type:
                Body.setPosition(X[i : i + 6])  # update position of free Body
                i += 6
            Body.redraw()  # redraw Body

        # update position of free Points
        for Point in self.PointList:
            if Point.type == type:
                Point.setPosition(X[i : i + 3])  # update position of free Point
                i += 3
                # redraw Point?

        # redraw all lines
        for Line in self.LineList:
            Line.RedrawLine(0)

        # ax.set_title("iteration "+str(ts))
        # eventually could show net forces too? <<< if using a non MINPACK method, use callback and do this

        pass  # I added this line to get the above commented lines (^^^) to be included in the animate method


if __name__ == "__main__":

    # some test functions

    Catenary(
        576.2346666666667,
        514.6666666666666,
        800,
        4809884.623076923,
        -2.6132152062554828,
        CB=-64.33333333333337,
        HF0=0,
        VF0=0,
        Tol=1e-05,
        MaxIter=50,
        plots=2,
    )
    print("\nTEST 2")
    Catenary(
        88.91360441490338,
        44.99537159734132,
        100.0,
        854000000.0000001,
        1707.0544275185273,
        CB=0.0,
        HF0=912082.6820817506,
        VF0=603513.100376363,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 3")
    Catenary(
        99.81149090002897,
        0.8459770263789324,
        100.0,
        854000000.0000001,
        1707.0544275185273,
        CB=0.0,
        HF0=323638.97834178555,
        VF0=30602.023233123222,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 4")
    Catenary(
        99.81520776134033,
        0.872357398602503,
        100.0,
        854000000.0000001,
        1707.0544275185273,
        CB=0.0,
        HF0=355255.0943810993,
        VF0=32555.18285808794,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 5")
    Catenary(
        99.81149195956499,
        0.8459747131565791,
        100.0,
        854000000.0000001,
        1707.0544275185273,
        CB=0.0,
        HF0=323645.55876751675,
        VF0=30602.27072107738,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 6")
    Catenary(
        88.91360650151807,
        44.99537139684605,
        100.0,
        854000000.0000001,
        1707.0544275185273,
        CB=0.0,
        HF0=912082.6820817146,
        VF0=603513.100376342,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 7")
    Catenary(
        9.516786788834565,
        2.601777402222183,
        10.0,
        213500000.00000003,
        426.86336920488003,
        CB=0.0,
        HF0=1218627.2292202935,
        VF0=328435.58512892434,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 8")
    Catenary(
        9.897879983411258,
        0.3124565409495972,
        10.0,
        213500000.00000003,
        426.86336920488003,
        CB=0.0,
        HF0=2191904.191415531,
        VF0=69957.98566771008,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 9")
    Catenary(
        107.77260514238083,
        7.381234307499085,
        112.08021179445676,
        6784339692.139625,
        13559.120871401587,
        CB=0.0,
        HF0=672316.8532881762,
        VF0=-552499.1313868811,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 9.1")
    Catenary(
        107.67265157943795,
        7.381234307499085,
        112.08021179445676,
        6784339692.139625,
        13559.120871401587,
        CB=0.0,
        HF0=3752787.759641461,
        VF0=-1678302.5929179655,
        Tol=1e-06,
        MaxIter=50,
        plots=1,
    )
    print("\nTEST 9.2")
    Catenary(
        107.77260514238083,
        7.381234307499085,
        112.08021179445676,
        6784339692.139625,
        13559.120871401587,
        CB=0.0,
        Tol=1e-06,
        MaxIter=50,
        plots=2,
        HF0=1.35e05,
        VF0=1.13e02,
    )
    print("\nTEST 9.3")
    Catenary(
        98.6712173965359,
        8.515909042185399,
        102.7903150736787,
        5737939634.533289,
        11467.878219531065,
        CB=0.0,
        HF0=118208621.36075467,
        VF0=-12806834.457078349,
        Tol=1e-07,
        MaxIter=50,
        plots=2,
    )


"""
# test script

test = System()

test.depth = 100

# Create the LineType of the line for the system
test.addLineType("main", 0.1, 100.0, 1e8)

# add points and lines
test.addPoint(1, [ 200, 0, -100])
test.addPoint(0, [ 100, 0,  -50])
test.addPoint(1, [   0, 0,    0])

test.addLine(80, "main")
test.addLine(170, "main")

# attach
test.PointList[0].addLine(1, 0)
test.PointList[1].addLine(1, 1)
test.PointList[1].addLine(2, 0)
test.PointList[2].addLine(2, 1)

test.initialize(plots=1)

test.solveEquilibrium(plots=1)

"""
