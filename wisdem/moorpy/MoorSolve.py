# a file to hold the custom solvers used in MoorPy

import numpy as np

# ================================ original above / modified below ===========================================


"""
def eval_func1(X, args):
    '''returns target outputs and also secondary outputs for constraint checks etc.'''

    # Step 1. break out design variables and arguments into nice names

    # Step 2. do the evaluation (this may change mutable things in args)

    # Step 3. group the outputs into objective function value and others

    return Y, oths



def step_func1(X, args, Y, oths, Ytarget, err, tol, iter, maxIter):
    '''General stepping functions, which can also contain special condition checks or other adjustments to the process

    '''

    # step 1. break out variables as needed

    # do stepping, as well as any conditional checks

    return dL   # returns dX (step to make)
"""


def dsolve1D(eval_func, step_func, X0, Ytarget, args, tol=0.0001, maxIter=20, Xmin=-np.inf, Xmax=np.inf):
    """
    Assumes the function is positive sloped (so use -X if negative-sloped)

    tol        - relative convergence tolerance (relative to step size, dX)
    Xmin, Xmax - bounds. by default start bounds at infinity
    """

    X = 1 * X0  # start off design variable

    print(f"Starting dsolve1D iterations>>>   aiming for Y={Ytarget}")

    for iter in range(maxIter):

        # call evaluation function
        Y, oths = eval_func(X, args)

        # compute error
        err = Y - Ytarget

        print(f"  new iteration with X={X:6.2f} and Y={Y:6.2f}")

        # update/narrow the bounds (currently this part assumes that the function is positively sloped)  << any N-D equivalent?
        if err > 0:  # and L < LUpper:       #
            Xmax = 1.0 * X
        elif err < 0:  # and L > LLower:       #
            Xmin = 1.0 * X

        if iter == maxIter - 1:
            print("Failed to find solution after " + str(iter) + " iterations, with error of " + str(err))
            break

        # >>>> COULD ALSO HAVE AN ITERATION RESTART FUNCTION? >>>
        #  that returns a restart boolean, as well as what values to use to restart things if true. How?

        else:
            dX = step_func(X, args, Y, oths, Ytarget, err, tol, iter, maxIter)

        # check for convergence
        if np.abs(dX) < tol * (np.abs(X) + tol):
            print(
                "Equilibrium solution completed after "
                + str(iter)
                + " iterations with error of "
                + str(err)
                + " and dX of "
                + str(dX)
            )
            print("solution X is " + str(X))
            break

        # Make sure we're not diverging by keeping things within narrowing bounds that span the solution.
        #        I.e. detect potential for oscillation and avoid bouncing out and then back in to semi-taut config
        #        Use previous values to bound where the correct soln is, and if an iteration moves beyond that,
        #        stop it and put it between the last value and where the bound is (using golden ratio, why not).
        if dX > 0 and X + dX >= Xmax:  # if moving up and about to go beyond previous too-high value
            X = X + 0.62 * (
                Xmax - X
            )  # move to mid point between current value and previous too-high value, rather than overshooting
            print("<--|")
        elif dX < 0 and X + dX <= Xmin:  # if moving down and about to go beyond previous too-low value
            X = X + 0.62 * (
                Xmin - X
            )  # 0.5*(L+LLower)               # move to mid point between current value and previous too-low value, rather than overshooting
            print("|-->")
        else:
            X = X + dX

    return X, Y, dict(iter=iter, err=err)


#    X, Y, info = dsolve1D(eval_func1, step_func1, X0, Ytarget, args, tol=tol, maxIter=maxIter)


# TODO: add default step_func (finite differencer), Ytarget, and args


def dsolve(
    eval_func,
    X0,
    Ytarget=[],
    step_func=None,
    args=[],
    tol=0.0001,
    maxIter=20,
    Xmin=[],
    Xmax=[],
    a_max=1.15,
    dX_last=[],
    plots=0,
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
    tol : float
        *relative* convergence tolerance (applied to step size components, dX)
    Xmin, Xmax
        Bounds. by default start bounds at infinity
    a_max
        maximum step size acceleration allowed
    dX_last
        Used if you want to dictate the initial step size/direction based on a previous attempt
    """
    success = False

    # process inputs and format as arrays in case they aren't already

    X = np.array(X0, dtype=np.float_)  # start off design variable
    N = len(X)

    Xs = np.zeros([maxIter, N])  # make arrays to store X and error results of the solve
    Es = np.zeros([maxIter, N])

    # check the target Y value input
    if len(Ytarget) == N:
        Ytarget = np.array(Ytarget, dtype=np.float_)
    elif len(Ytarget) == 0:
        Ytarget = np.zeros(N, dtype=np.float_)
    else:
        raise TypeError("Ytarget must be of same length as X0")

    # if a step function wasn't provided, provide a default one
    if step_func == None:
        if plots > 2:
            print("Using default finite difference step func")

        def step_func(X, args, Y, oths, Ytarget, err, tol, iter, maxIter):

            J = np.zeros([N, N])  # Initialize the Jacobian matrix that has to be a square matrix with nRows = len(X)

            for i in range(
                N
            ):  # Newton's method: perturb each element of the X variable by a little, calculate the outputs from the
                X2 = np.array(
                    X
                )  # minimizing function, find the difference and divide by the perturbation (finding dForce/d change in design variable)
                deltaX = tol * (np.abs(X[i]) + tol)
                X2[i] += deltaX
                Y2, _, _ = eval_func(X2, args)  # here we use the provided eval_func

                J[:, i] = (Y2 - Y) / deltaX  # and append that column to each respective column of the Jacobian matrix

            if N > 1:
                dX = -np.matmul(
                    np.linalg.inv(J), Y - Ytarget
                )  # Take this nth output from the minimizing function and divide it by the jacobian (derivative)
            else:
                dX = np.array([-(Y[0] - Ytarget[0]) / J[0, 0]])

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

    if plots > 2:
        print(f"Starting dsolve iterations>>>   aiming for Y={Ytarget}")

    for iter in range(maxIter):

        # call evaluation function
        Y, oths, stop = eval_func(X, args)

        # compute error
        err = Y - Ytarget

        if plots > 2:
            print(f"  new iteration #{iter} with X={X} and Y={Y}")

        Xs[iter, :] = X
        Es[iter, :] = err

        # stop if commanded by objective function
        if stop:
            break

        if iter == maxIter - 1:
            if plots > 2:
                print("Failed to find solution after " + str(iter) + " iterations, with error of " + str(err))
            break

        # >>>> COULD ALSO HAVE AN ITERATION RESTART FUNCTION? >>>
        #  that returns a restart boolean, as well as what values to use to restart things if true. How?

        else:
            dX = step_func(X, args, Y, oths, Ytarget, err, tol, iter, maxIter)

        # if plots>2:
        #    breakpoint()

        ## Make sure we're not diverging by keeping things from reversing too much.
        # Track the previous step (dX_last) and if the current step reverses too much, stop it part way.
        # Stop it at a plane part way between the current X value and the previous X value (using golden ratio, why not).

        # get the point along the previous step vector where we'll draw the bounding hyperplane (could be a line, plane, or more in higher dimensions)
        Xlim = X - 0.62 * dX_last

        # the equation for the plane we don't want to recross is then sum(X*dX_last) = sum(Xlim*dX_last)

        if np.sum((X + dX) * dX_last) < np.sum(Xlim * dX_last):  # if we cross are going to cross it

            alpha = np.sum((Xlim - X) * dX_last) / np.sum(
                dX * dX_last
            )  # this is how much we need to scale down dX to land on it rather than cross it

            # print("limiting oscillation with alpha="+str(alpha))
            # print(f"dX_last was {dX_last}, dX was going to be {dX}, now it'll be {alpha*dX}")
            # print(f"dX_last was {dX_last/1000}, dX was going to be {dX/1000}, now it'll be {alpha*dX/1000}")

            dX = alpha * dX  # scale down dX

        ## also avoid extreme accelerations in the same direction

        if np.linalg.norm(dX_last) > tol:
            for i in range(N):

                if abs(dX_last[i]) < tol:
                    dX_max = a_max * 10 * tol * np.sign(dX[i])
                else:
                    dX_max = a_max * dX_last[i]

                if dX_max == 0.0:
                    dX[i] = 0.0
                else:
                    a_i = dX[i] / dX_max  # a_i = dX[i]/(dX_last[i]+tol*np.sign(dX_last[i]))

                    if a_i > a_max:

                        # print(f"limiting accelerations with alpha={a_max/a_i} for axis {i}")
                        # print(f"dX_last was {dX_last}, dX was going to be {dX}, now it'll be {dX*a_max/a_i}")

                        # dX = dX*a_max/a_i  # scale it down to the maximum value
                        dX[i] = dX[i] * a_max / a_i  # scale it down to the maximum value (treat each DOF individually)

        """
        if np.linalg.norm(X) < 1000:
            print('X',X)
            print('Y',Y)
            print('dX',dX)
            print('tol',tol)
            print(np.abs(dX), tol*(np.abs(X)+tol))
        """

        # enforce bounds
        for i in range(N):

            if X[i] + dX[i] < Xmin[i]:
                dX[i] = Xmin[i] - X[i]

            elif X[i] + dX[i] > Xmax[i]:
                dX[i] = Xmax[i] - X[i]

        # check for convergence
        if all(np.abs(dX) < tol * (np.abs(X) + tol)):

            if plots > 2:
                print(
                    "Iteration converged after "
                    + str(iter)
                    + " iterations with error of "
                    + str(err)
                    + " and dX of "
                    + str(dX)
                )
                print("Solution X is " + str(X))

            if any(X == Xmin) or any(X == Xmax):
                success = False
                print("Warning: dsolve ended on a bound.")
            else:
                success = True

            break

        dX_last = 1.0 * dX  # remember this current value

        X = X + dX
        # print(X)

    return X, Y, dict(iter=iter, err=err, dX=dX_last, oths=oths, Xs=Xs, Es=Es, success=success)


def dopt(eval_func, X0, tol=0.0001, maxIter=20, Xmin=[], Xmax=[], a_max=1.15, dX_last=[], display=0):
    """
    Multi-direction Newton's method solver.

    tol        - *relative* convergence tolerance (applied to step size components, dX)
    Xmin, Xmax - bounds. by default start bounds at infinity
    a_max      - maximum step size acceleration allowed
    """

    # process inputs and format as arrays in case they aren't already
    if len(X0) == 0:
        raise ValueError("X0 cannot be empty")

    X = np.array(X0, dtype=np.float_)  # start off design variable (optimized)

    # do a test call to see what size the results are
    f, g, Xextra, Yextra, oths, stop = eval_func(X)  # , XtLast, Ytarget, args)

    N = len(X)  # number of design variables
    Nextra = len(Xextra)  # additional relevant variables calculated internally and passed out, for tracking
    m = len(g)  # number of constraints

    Xs = np.zeros([maxIter, N + Nextra])  # make arrays to store X and error results of the solve
    Fs = np.zeros([maxIter])  # make arrays to store objective function values
    Gs = np.zeros([maxIter, m])  # make arrays to store constraint function values

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

    if len(dX_last) == N:
        dX_last = np.array(dX_last, dtype=np.float_)
    elif len(dX_last) == 0:
        dX_last = np.zeros(N)
    else:
        raise ValueError("dX_last input must be of same size as design vector, if provided")
    # XtLast = 1.0*Xt0

    if display > 0:
        print("Starting dopt iterations>>>")

    for iter in range(maxIter):

        # call evaluation function (returns objective val, constrain vals, tuned variables, tuning results)
        f, g, Xextra, Yextra, oths, stop = eval_func(X)  # , XtLast, Ytarget, args)

        if display > 0:
            print(f" >> Iteration {iter}: X={X}  Xe={Xextra}  f={f}")

        if display > 1:
            print(f"    Constraint values: {g}")

        Xs[iter, :] = np.hstack([X, Xextra])
        Fs[iter] = f
        Gs[iter, :] = g

        # stop if commanded by objective function
        if stop:
            break

        # temporarily display output
        # print(np.hstack([X,Y]))

        if iter == maxIter - 1:
            print("Failed to find solution after " + str(iter) + " iterations")
            break

        # >>>> COULD ALSO HAVE AN ITERATION RESTART FUNCTION? >>>
        #  that returns a restart boolean, as well as what values to use to restart things if true. How?

        else:  # this is where we get derivatives and then take a step

            # dX = step_func(X, args, Y, oths, Ytarget, err, tol, iter, maxIter)
            # hard coding a generic approach for now

            dX = np.zeros(N)  # optimization step size to take

            X2 = np.array(X, dtype=np.float_)

            Jf = np.zeros([N])
            Jg = np.zeros([N, m])
            Hf = np.zeros([N])  # this is ust the diagonal of the Hessians
            Hg = np.zeros([N, m])

            for i in range(N):  # loop through each variable

                dXi = 4.0  # 0.5# 1.0*dX[i]  # this is gradient finite difference step size, not opto step size

                # could do repetition to hone in when second derivative is large, but not going to for now

                X2[i] += dXi  # perturb
                fp, gp, Xtp, Yp, othsp, stopp = eval_func(X2)

                X2[i] -= 2.0 * dXi
                fm, gm, Xtm, Ym, othsm, stopm = eval_func(X2)

                # for objective function and constraints (note that g may be multidimensional),
                # fill in diagonal values of Jacobian and Hession (not using off-diagonals for now)
                Jf[i] = (fp - fm) / (2 * dXi)
                Jg[i, :] = (gp - gm) / (2 * dXi)
                Hf[i] = (fm - 2.0 * f + fp) / dXi ** 2
                Hg[i, :] = (gm - 2.0 * g + gp) / dXi ** 2

            # If we're currently violating a constraint, fix it rather than worrying about the objective function
            # This step is when new gradients need to be calculated at the violating point
            # e.g. in cases where the constraint functions are flat when not violated
            if any(g < 0.0):

                # handle each dimension individually
                for j in range(m):  # go through each constraint
                    if g[j] < 0:  # if a constraint will be violated
                        if np.sum(np.abs(Jg[:, j])) == 0.0:
                            breakpoint()
                            print(f"Error, zero Jacobian for constraint {j}")

                        alpha = (0.0 - g[j]) / np.sum(
                            Jg[:, j] * Jg[:, j]
                        )  # assuming we follow the gradient, finding how far to move to get to zero
                        dX += (
                            Jg[:, j] * alpha * 1.1
                        )  # step size is gradient times alpha (plus a little extra for margin)

                        if display > 1:
                            print(f"    Constraint {j} violated ({g[j]}). Correction: {Jg[:,j]*alpha *1.1}")

                # if the above fails, we could try backtracking along dX_last until the constriant is no longer violated...

                # at the end of this, the step will be a summation of the steps estimated to resolve each constraint

                if display > 1:
                    print(f"     Constraint solution step, dX={dX}")

            # otherwise make an optimization step
            else:

                # figure out step size in each dimension
                for i in range(N):
                    if Hf[i] <= 0.0:  # if the hessian is zero or negative, just move a fixed step size
                        dX[i] = -Jf[i] / np.linalg.norm(Jf) * np.abs(dX_last[i]) * a_max
                    else:
                        dX[i] = -Jf[i] / Hf[i]

                # breakpoint()

                # respect bounds (handle each dimension individually)
                for i in range(N):
                    if X[i] + dX[i] < Xmin[i]:
                        dX[i] = Xmin[i] - X[i]
                    elif X[i] + dX[i] > Xmax[i]:
                        dX[i] = Xmax[i] - X[i]

                # deal with potential constraint violations in making the step (based on existing gradients)
                # respect constraints approximately (handle each dimension individually...for now)
                X2 = X + dX  # save jump before constrain correction
                for j in range(m):  # go through each constraint
                    g2j = g[j] + np.sum(Jg[:, j] * dX)  # get constraint value at jump
                    if g2j < 0:  # if a constraint will be violated
                        alpha = -(0.0 - g2j) / np.sum(
                            Jg[:, j] * Jg[:, j]
                        )  # assuming we follow the gradient, finding how far to move to get to zero <<< double check signs
                        dX = (
                            Jg[:, j] * alpha
                        )  # step size is gradient times alpha (NOT adding a little extra for margin)

                        if display > 1:
                            print(f"    Cutting down dX to {dX} to avoid potential violation of constraint {j}")

                    # if iter > 20:
                    #    breakpoint()

                # this is how to stop the dX vector at the approximate constraint boundary (not good for navigation)
                # for j in len(g):                           # go through each constraint
                #    if g[j] + np.sum(Jg[:,j]*dX) < 0:        # if a constraint will be violated
                #        alpha = -g[j]/np.sum(Jg[:,j]*dX)     # find where the constraint boundary is (linear approximation)
                #        dX = dX*alpha                        # shrink the step size accordingly (to stop at edge of constraint)

                if display > 1:
                    print(f"     Minimization step, dX={dX}    J={Jf}   H={Hf}")

        ## Make sure we're not diverging by keeping things from reversing too much.
        # Track the previous step (dX_last) and if the current step reverses too much, stop it part way.
        # Stop it at a plane part way between the current X value and the previous X value (using golden ratio, why not).

        # get the point along the previous step vector where we'll draw the bounding hyperplane (could be a line, plane, or more in higher dimensions)
        Xlim = X - 0.62 * dX_last

        # the equation for the plane we don't want to recross is then sum(X*dX_last) = sum(Xlim*dX_last)

        if np.sum((X + dX) * dX_last) < np.sum(Xlim * dX_last):  # if we cross are going to cross it

            alpha = np.sum((Xlim - X) * dX_last) / np.sum(
                dX * dX_last
            )  # this is how much we need to scale down dX to land on it rather than cross it

            if display > 1:
                print("    limiting oscillation with alpha=" + str(alpha))
                print(f"    dX_last was {dX_last}, dX was going to be {dX}, now it'll be {alpha*dX}")
                print(f"    dX_last was {dX_last/1000}, dX was going to be {dX/1000}, now it'll be {alpha*dX/1000}")

            dX = alpha * dX  # scale down dX

        ## also avoid extreme accelerations in the same direction

        if np.linalg.norm(dX_last) > tol:
            for i in range(N):

                if abs(dX_last[i]) < tol:
                    dX_max = a_max * 10 * tol * np.sign(dX[i])
                else:
                    dX_max = a_max * dX_last[i]

                a_i = dX[i] / dX_max  # a_i = dX[i]/(dX_last[i]+tol*np.sign(dX_last[i]))

                if a_i > a_max:

                    if display > 1:
                        print(f"    limiting accelerations with alpha={a_max/a_i} for axis {i}")
                        print(f"    dX_last was {dX_last}, dX was going to be {dX}, now it'll be {dX*a_max/a_i}")

                    dX = dX * a_max / a_i  # scale it down to the maximum value

        # enforce bounds
        for i in range(N):
            if X[i] + dX[i] < Xmin[i]:
                dX[i] = Xmin[i] - X[i]
            elif X[i] + dX[i] > Xmax[i]:
                dX[i] = Xmax[i] - X[i]

        # check for convergence
        if all(np.abs(dX) < tol * (np.abs(X) + tol)):
            print("Solution found after " + str(iter) + " iterations. Solution X is " + str(X))

            break

        # breakpoint()

        dX_last = 1.0 * dX  # remember this current value
        # XtLast = 1.0*Xt

        X = X + dX

    return X, f, dict(iter=iter, dX=dX_last, oths=oths, Xs=Xs, Fs=Fs, Gs=Gs, Xextra=Xextra, g=g, Yextra=Yextra)


# ------------------------------ sample functions ----------------------------


def eval_func1(X, args):
    """returns target outputs and also secondary outputs for constraint checks etc."""

    # Step 1. break out design variables and arguments into nice names

    # Step 2. do the evaluation (this may change mutable things in args)
    y1 = (X[0] - 2) ** 2 + X[1]
    y2 = X[0] + X[1]

    # Step 3. group the outputs into objective function value and others
    Y = np.array([y1, y2])  # objective function
    oths = dict(status=1)  # other outputs - returned as dict for easy use

    return Y, oths, False


def step_func1(X, args, Y, oths, Ytarget, err, tol, iter, maxIter):
    """General stepping functions, which can also contain special condition checks or other adjustments to the process"""

    # get numerical derivative
    J = np.zeros([len(X), len(X)])  # Initialize the Jacobian matrix that has to be a square matrix with nRows = len(X)

    for i in range(
        len(X)
    ):  # Newton's method: perturb each element of the X variable by a little, calculate the outputs from the
        X2 = np.array(
            X
        )  # minimizing function, find the difference and divide by the perturbation (finding dForce/d change in design variable)
        deltaX = tol * (np.abs(X[i]) + tol)
        X2[i] += deltaX
        Y2, extra = eval_func1(X2, args)

        J[:, i] = (Y2 - Y) / deltaX  # and append that column to each respective column of the Jacobian matrix

    dX = -np.matmul(
        np.linalg.inv(J), Y
    )  # Take this nth output from the minimizing function and divide it by the jacobian (derivative)

    return dX  # returns dX (step to make)


## ============================== below is a new attempt at the Catenary solve ======================================


def eval_func_cat(X, args):
    """returns target outputs and also secondary outputs for constraint checks etc."""

    info = dict(error=False)  # a dict of extra outputs to be returned

    ## Step 1. break out design variables and arguments into nice names
    HF = X[0]
    VF = X[1]

    [XF, ZF, L, EA, W, CB, WL, WEA, L_EA, CB_EA] = args["cat"]

    ## Step 2. do the evaluation (this may change mutable things in args)

    # print("Catenary iteration HF={:8.4e}, VF={:8.4e}".format(HF,VF))

    # calculate some commonly used terms that depend on HF and VF:

    VFMinWL = VF - WL
    # = VA, the vertical anchor load (positive-up, but VF is positive-down)
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

    # determine line profile type
    if (CB < 0.0) or (W < 0.0) or (VFMinWL > 0.0):  # no portion of the line rests on the seabed
        ProfileType = 1
    elif -CB * VFMinWL < HF:  # a portion of the line rests on the seabed and the anchor tension is nonzero
        ProfileType = 2
    else:  # must be 0.0 < HF <= -CB*VFMinWL, meaning a portion of the line must rest on the seabed and the anchor tension is zero
        ProfileType = 3

    # Compute the error functions (to be zeroed) and the Jacobian matrix
    #   (these depend on the anticipated configuration of the mooring line):

    # <<< could eliminate frequent division by W below, (make 1/W variable) >>>>>

    # No portion of the line rests on the seabed
    if ProfileType == 1:

        if VF_HF + SQRT1VF_HF2 <= 0:
            info["error"] = True
            info["message"] = "ProfileType 1: VF_HF + SQRT1VF_HF2 <= 0"
        elif VFMinWL_HF + SQRT1VFMinWL_HF2 <= 0:
            info["error"] = True
            info["message"] = "ProfileType 1: VFMinWL_HF + SQRT1VFMinWL_HF2 <= 0"
            # note: these errors seemed to occur when a buoyant line got to an HF=0 iteration (hopefully avoided now)

        else:

            LBot = 0  # note that there is no seabed contact (for clarity in outputs)

            EXF = (
                (np.log(VF_HF + SQRT1VF_HF2) - np.log(VFMinWL_HF + SQRT1VFMinWL_HF2)) * HF_W + L_EA * HF - XF
            )  # error in horizontal distance

            EZF = (SQRT1VF_HF2 - SQRT1VFMinWL_HF2) * HF_W + L_EA * (VF - 0.5 * WL) - ZF  # error in vertical distance

            dXFdHF = (
                (np.log(VF_HF + SQRT1VF_HF2) - np.log(VFMinWL_HF + SQRT1VFMinWL_HF2)) / W
                - (
                    (VF_HF + VF_HF2 / SQRT1VF_HF2) / (VF_HF + SQRT1VF_HF2)
                    - (VFMinWL_HF + VFMinWL_HF2 / SQRT1VFMinWL_HF2) / (VFMinWL_HF + SQRT1VFMinWL_HF2)
                )
                / W
                + L_EA
            )

            dXFdVF = (
                (1.0 + VF_HF / SQRT1VF_HF2) / (VF_HF + SQRT1VF_HF2)
                - (1.0 + VFMinWL_HF / SQRT1VFMinWL_HF2) / (VFMinWL_HF + SQRT1VFMinWL_HF2)
            ) / W

            dZFdHF = (SQRT1VF_HF2 - SQRT1VFMinWL_HF2) / W - (VF_HF2 / SQRT1VF_HF2 - VFMinWL_HF2 / SQRT1VFMinWL_HF2) / W

            dZFdVF = (VF_HF / SQRT1VF_HF2 - VFMinWL_HF / SQRT1VFMinWL_HF2) / W + L_EA

    # A portion of the line rests on the seabed and the anchor tension is nonzero
    elif ProfileType == 2:

        if VF_HF + SQRT1VF_HF2 <= 0:
            info["error"] = True
            info["message"] = "ProfileType 2: VF_HF + SQRT1VF_HF2 <= 0"

        else:
            EXF = np.log(VF_HF + SQRT1VF_HF2) * HF_W - 0.5 * CB_EA * W * LBot * LBot + L_EA * HF + LBot - XF

            EZF = (SQRT1VF_HF2 - 1.0) * HF_W + 0.5 * VF * VF_WEA - ZF

            dXFdHF = (
                np.log(VF_HF + SQRT1VF_HF2) / W - ((VF_HF + VF_HF2 / SQRT1VF_HF2) / (VF_HF + SQRT1VF_HF2)) / W + L_EA
            )

            dXFdVF = ((1.0 + VF_HF / SQRT1VF_HF2) / (VF_HF + SQRT1VF_HF2)) / W + CB_EA * LBot - 1.0 / W

            dZFdHF = (SQRT1VF_HF2 - 1.0 - VF_HF2 / SQRT1VF_HF2) / W

            dZFdVF = (VF_HF / SQRT1VF_HF2) / W + VF_WEA

            # print(" {:6.2e} {:6.2e}  {:6.2e} {:6.2e}   {:6.2e} {:6.2e} {:6.2e} {:6.2e}".format(HF,VF,EXF,EZF,dXFdHF, dXFdVF, dZFdHF, dZFdVF))
            # if abs( ( SQRT1VF_HF2 - 1.0 )*HF_W + 0.5*VF*VF_WEA ) < 0.0001:
            #    breakpoint()

    # A portion of the line must rest on the seabed and the anchor tension is zero
    elif ProfileType == 3:

        if VF_HF + SQRT1VF_HF2 <= 0:
            info["error"] = True
            info["message"] = "ProfileType 3: VF_HF + SQRT1VF_HF2 <= 0"

        else:
            EXF = (
                np.log(VF_HF + SQRT1VF_HF2) * HF_W
                - 0.5 * CB_EA * W * (LBot * LBot - (LBot - HF_W / CB) * (LBot - HF_W / CB))
                + L_EA * HF
                + LBot
                - XF
            )

            EZF = (SQRT1VF_HF2 - 1.0) * HF_W + 0.5 * VF * VF_WEA - ZF

            dXFdHF = (
                np.log(VF_HF + SQRT1VF_HF2) / W
                - ((VF_HF + VF_HF2 / SQRT1VF_HF2) / (VF_HF + SQRT1VF_HF2)) / W
                + L_EA
                - (LBot - HF_W / CB) / EA
            )

            dXFdVF = ((1.0 + VF_HF / SQRT1VF_HF2) / (VF_HF + SQRT1VF_HF2)) / W + HF_WEA - 1.0 / W

            dZFdHF = (SQRT1VF_HF2 - 1.0 - VF_HF2 / SQRT1VF_HF2) / W

            dZFdVF = (VF_HF / SQRT1VF_HF2) / W + VF_WEA

    # Now compute the tensions at the anchor

    Zextreme = 0.0

    if ProfileType == 1:  # No portion of the line rests on the seabed
        HA = HF
        VA = VFMinWL  # note: VF is defined positive when tension pulls downward, while VA is defined positive when tension pulls up

        # for a freely suspended line, if necessary, check to ensure the line doesn't droop and hit the seabed
        if CB < 0 and VFMinWL < 0.0:  # only need to do this if the line is slack (has zero slope somewhere)
            # this is indicated by the anchor force having a positive value, meaning it's helping hold up the line

            Zextreme = (
                1 - SQRT1VFMinWL_HF2
            ) * HF_W - 0.5 * VFMinWL ** 2 / WEA  # max or min line elevation (where slope=0)

    elif ProfileType == 2:  # A portion of the line rests on the seabed and the anchor tension is nonzero
        HA = (
            HF + CB * VFMinWL
        )  # note: -VFMinWL = -(VF-W*L) is the negative of line weight NOT supported by the fairlead; i.e. the weight on the seabed
        VA = 0.0

    elif ProfileType == 3:  # A portion of the line must rest on the seabed and the anchor tension is zero
        HA = 0.0
        VA = 0.0

    # if there was an error, send the stop signal
    if info["error"] == True:
        return np.zeros(2), info, True

    ## Step 3. group the outputs into objective function value and others
    Y = np.array([EXF, EZF])  # objective function

    # info is a dict of other outputs to be returned
    info[
        "HF"
    ] = HF  # solution to be used to start next call (these are the solved variables, may be for anchor if line is reversed)
    info["VF"] = VF
    info["jacobian"] = np.array([[dXFdHF, dXFdVF], [dZFdHF, dZFdVF]])
    info["LBot"] = LBot
    info["HA"] = HA
    info["VA"] = VA
    info["Zextreme"] = Zextreme
    info["ProfileType"] = ProfileType

    # print("EX={:5.2e}, EZ={:5.2e}".format(EXF, EZF))

    return Y, info, False


def step_func_cat(X, args, Y, info, Ytarget, err, tol, iter, maxIter):
    """General stepping functions, which can also contain special condition checks or other adjustments to the process

    info - the info dict created by the main catenary function

    """
    [XF, ZF, L, EA, W, CB, WL, WEA, L_EA, CB_EA] = args["cat"]

    # if abs( err[1] + ZF ) < 0.0001:
    #    breakpoint()

    [alpha_min, alpha0, alphaR] = args[
        "step"
    ]  # get minimum alpha, initial alpha, and alpha reduction rate from passed arguments

    J = info["jacobian"]

    dX = -np.matmul(np.linalg.inv(J), err)

    # ! Reduce dHF by factor (between 1 at I = 1 and 0 at I = MaxIter) that reduces linearly with iteration count
    # to ensure that we converge on a solution even in the case were we obtain a nonconvergent cycle about the
    # correct solution (this happens, for example, if we jump to quickly between a taut and slack catenary)

    alpha = np.max([alpha_min, alpha0 * (1.0 - alphaR * iter / maxIter)])

    # exponential approach       alpha = alpha0 * np.exp( iter/maxIter * np.log(alpha_min/alpha0 ) )

    dX[0] = dX[0] * alpha  # dHF*( 1.0 - Tol*I )
    dX[1] = dX[1] * alpha  # dVF*( 1.0 - Tol*I )

    # To avoid an ill-conditioned situation, make sure HF does not go less than or equal to zero by having a lower limit of Tol*HF
    # [NOTE: the value of dHF = ( Tol - 1.0 )*HF comes from: HF = HF + dHF = Tol*HF when dHF = ( Tol - 1.0 )*HF]
    # dX[0] = max( dX[0], ( tol - 1.0 )*info['HF']);

    # To avoid an ill-conditioned situation, make sure HF does not get too close to zero, by forcing HF >= Tol*abs(VF)
    if info["HF"] + dX[0] <= tol * abs(info["VF"] + dX[1]):
        dX[0] = tol * abs(info["VF"] + dX[1]) - info["HF"]

    # To avoid an ill-conditioned situation where the line is nearly all on the seabed but the solver gets stuck,
    # if np.abs(err[1] + ZF)/ZF < tol:
    #    breakpoint()
    # deltaHFVF = info['HF'] - info['VF']
    # dX[0] = dX[0] - 0.5*deltaHFVF
    # dX[1] = dX[1] + 0.5*deltaHFVF

    # prevent silly situation where a line with weight and positive ZF considers a negative VF
    if info["ProfileType"] == 2:
        if X[1] + dX[1] <= tol:
            VFtarget = (L - info["LBot"]) * W  # set next VF to weight of portion of line that's suspended
            dX[1] = VFtarget - X[1]

    return dX  # returns dX (step to make)


"""

# test run


#Catenary2(100, 50, 130, 1e8, 100, plots=1)

print("\nTEST 1")
Catenary(576.2346666666667, 514.6666666666666, 800, 4809884.623076923, -2.6132152062554828, CB=-64.33333333333337, HF0=0, VF0=0, Tol=1e-05, MaxIter=50, plots=2)
print("\nTEST 2")
Catenary(88.91360441490338, 44.99537159734132, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=912082.6820817506, VF0=603513.100376363, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 3")
Catenary(99.81149090002897, 0.8459770263789324, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=323638.97834178555, VF0=30602.023233123222, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 4")
Catenary(99.81520776134033, 0.872357398602503, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=355255.0943810993, VF0=32555.18285808794, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 5")
Catenary(99.81149195956499, 0.8459747131565791, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=323645.55876751675, VF0=30602.27072107738, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 6")
Catenary(88.91360650151807, 44.99537139684605, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=912082.6820817146, VF0=603513.100376342, Tol=1e-06, MaxIter=50, plots=1)
"""
"""
maxIter = 10
# call the master solver function
X0      = [2,2]
Ytarget = [0,0]
args    = []
X, Y, info = dsolve(eval_func1, step_func1, X0, Ytarget, args, maxIter=maxIter)
"""
