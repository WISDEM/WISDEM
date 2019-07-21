""" testing utilities should be moved to CommonSE 
"""

import numpy as np
from openmdao.api import IndepVarComp, Component, Problem, Group, SqliteRecorder, BaseRecorder



##### Test Setup Utilities #####
def init_IndepVar_add(prob, data):
    """ Add component independent vars from a dictionary of inputs """
    for var in data.keys():
        if type(data[var]) is np.ndarray or type(data[var]) is np.array:
            print var
            prob.root.add(var, IndepVarComp(var, np.zeros_like(data[var])), promotes=['*'])
        else:
            prob.root.add(var, IndepVarComp(var, data[var]), promotes=['*'])
    return prob

def init_IndepVar_set(prob, data):
    """ Set component independent vars from a dictionary of inputs """
    for var in data.keys():
        prob[var] = data[var]
    return prob




##### Check Gradients ##### <- from test_rotor_aeropower_gradients.py
def check_gradient_unit_test(prob, fd='central', step_size=1e-6, tol=1e-6, display=False,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6, comp=None):
    """compare provided analytic gradients to finite-difference gradients with unit testing.
    Same as check_gradient, but provides a unit test for each gradient for convenience.
    the unit tests checks that the error for each gradient is less than tol.
    Parameters
    ----------
    comp : obj
        An OpenMDAO component that provides analytic gradients through provideJ()
    fd : str
        the type of finite difference to use.  options are central or forward
    step_size : float
        step size to use in finite differencing
    tol : float
        tolerance for how close the gradients should agree to
    display : boolean
        if True, display results for each gradient
    show_missing_warnings: boolean
        if True, warn for gradients that were not provided
        (they may be ones that are unecessary, but may be ones that were accidentally skipped)
    show_scaling_warnings: boolean
        if True, warn for gradients that are either very small or very large which may lead
        to challenges in solving the full linear system
    min_grad/max_grad : float
        quantifies what "very small" or "very large" means when using show_scaling_warnings
    """

    J_fd, J_fwd, J_rev = check_gradient(prob, fd, step_size, tol, display, show_missing_warnings,
        show_scaling_warnings, min_grad, max_grad)
    if comp == None:
        comp = prob.root.comp
    if "list_deriv_vars" in dir(comp): #  callable(getattr(comp, 'list_deriv_vars')):
        inputs, outputs = comp.list_deriv_vars()
        for output in outputs:
            for input in inputs:
                J = J_fwd[output, input]
                JFD = J_fd[output, input]
                m, n = J.shape
                # print '---------', output, input, '---------'
                # print 'J', J
                # print 'JFD', JFD
                for i in range(m):
                    for j in range(n):
                        if np.abs(J[i, j]) <= tol:
                            errortype = 'absolute'
                            error = J[i, j] - JFD[i, j]
                        else:
                            errortype = 'relative'
                            error = 1.0 - JFD[i, j]/J[i, j]
                        error = np.abs(error)

                        # # display
                        # if error > tol:
                        #     star = ' ***** '
                        # else:
                        #     star = ''
                        #
                        # if display:
                        #     output = '{}{:<20} ({}) {}: ({}, {})'.format(star, error, errortype, name, J[i, j], JFD[i, j])
                        #     print output
                        #
                        # if show_scaling_warnings and J[i, j] != 0 and np.abs(J[i, j]) < min_grad:
                        #     print '*** Warning: The following analytic gradient is very small and may need to be scaled:'
                        #     print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
                        #
                        # if show_scaling_warnings and np.abs(J[i, j]) > max_grad:
                        #     print '*** Warning: The following analytic gradient is very large and may need to be scaled:'
                        #     print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
                        #

                        try:
                            # unittest.assertLessEqual(error, tol)
                            assert error <= tol
                        except AssertionError, e:
                            print '*** error in:', "\n\tOutput: ", output, "\n\tInput: ", input, "\n\tPosition: ", i, j
                            print JFD[i, j], J[i, j]
                            print error, tol
                            raise e
    else:
        for key, value in J_fd.iteritems():
                J = J_fwd[key]
                JFD = J_fd[key]

                m, n = J.shape
                for i in range(m):
                    for j in range(n):
                        if np.abs(J[i, j]) <= tol:
                            errortype = 'absolute'
                            error = J[i, j] - JFD[i, j]
                        else:
                            errortype = 'relative'
                            error = 1.0 - JFD[i, j]/J[i, j]
                        error = np.abs(error)
                        try:
                            # unittest.assertLessEqual(error, tol)
                            assert error <= tol
                        except AssertionError, e:
                            print '*** error in:', "\n\tKey: ", key, "\n\tPosition: ", i, j
                            raise e


def check_gradient(prob, fd='central', step_size=1e-6, tol=1e-6, display=False,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6):
    """compare provided analytic gradients to finite-difference gradients
    Parameters
    ----------
    comp : obj
        An OpenMDAO component that provides analytic gradients through provideJ()
    fd : str
        the type of finite difference to use.  options are central or forward
    step_size : float
        step size to use in finite differencing
    tol : float
        tolerance for how close the gradients should agree to
    display : boolean
        if True, display results for each gradient
    show_missing_warnings: boolean
        if True, warn for gradients that were not provided
        (they may be ones that are unecessary, but may be ones that were accidentally skipped)
    show_scaling_warnings: boolean
        if True, warn for gradients that are either very small or very large which may lead
        to challenges in solving the full linear system
    min_grad/max_grad : float
        quantifies what "very small" or "very large" means when using show_scaling_warnings
    Returns
    -------
    names : array(str)
        list of the names of all the gradients
    errorvec : array(float)
        list of all the errors for the gradients.  If the magnitude of the gradient is less than
        tol, then an absolute error is used, otherwise a relative error is used.
    """
    # inputs = comp.list_deriv_vars
    # inputs, outputs = comp.list_deriv_vars()

    # show_missing_warnings = False

    # if show_missing_warnings:
    #     all_inputs = _explodeall(comp, vtype='inputs')
    #     all_outputs = _explodeall(comp, vtype='outputs')
    #     reserved_inputs = ['missing_deriv_policy', 'directory', 'force_fd', 'force_execute', 'eval_only']
    #     reserved_outputs = ['derivative_exec_count', 'itername', 'exec_count']
    #     potential_missed_inputs = list(set(all_inputs) - set(reserved_inputs) - set(inputs))
    #     potential_missed_outputs = list(set(all_outputs) - set(reserved_outputs) - set(outputs))
    #
    #     if len(potential_missed_inputs) > 0 or len(potential_missed_outputs) > 0:
    #         print
    #         print '*** Warning: ' + comp.__class__.__name__ + ' does not supply derivatives for the following'
    #         print '\tinputs:', potential_missed_inputs
    #         print '\toutputs:', potential_missed_outputs
    #         print

    # prob = Problem()
    # prob.root = Group()
    # prob.root.add('comp', comp, promotes=['*'])
    # prob.setup()
    #
    # for i in range(len(inputs)):
    #     prob[inputs[i]] = comp

    prob.run()
    root = prob.root

    # Linearize the model
    root._sys_linearize(root.params, root.unknowns, root.resids)

    data = {}

    # Derivatives should just be checked without parallel adjoint for now.
    voi = None

    jac_fwd = {}
    jac_rev = {}
    jac_fd = {}

    # Check derivative calculations for all comps at every level of the
    # system hierarchy.
    for comp in root.components(recurse=True):
        cname = comp.pathname

        # No need to check comps that don't have any derivs.
        if comp.deriv_options['type'] == 'fd':
            continue

        # IndepVarComps are just clutter too.
        if isinstance(comp, IndepVarComp):
            continue

        data[cname] = {}
        jac_fwd = {}
        jac_rev = {}
        jac_fd = {}

        # try:
        #     params, unknowns = comp.list_deriv_vars()
        # except:
        #     pass
        params = comp.params
        unknowns = comp.unknowns
        resids = comp.resids
        dparams = comp.dpmat[voi]
        dunknowns = comp.dumat[voi]
        dresids = comp.drmat[voi]

        # Skip if all of our inputs are unconnected.
        # if len(dparams) == 0:
        #     continue

        # if out_stream is not None:
        #     out_stream.write('-'*(len(cname)+15) + '\n')
        #     out_stream.write("Component: '%s'\n" % cname)
        #     out_stream.write('-'*(len(cname)+15) + '\n')

        states = comp.states

        param_list = [item for item in dparams if not \
                      dparams.metadata(item).get('pass_by_obj')]
        param_list.extend(states)

        # Create all our keys and allocate Jacs
        for p_name in param_list:

            dinputs = dunknowns if p_name in states else dparams
            p_size = np.size(dinputs[p_name])

            # Check dimensions of user-supplied Jacobian
            for u_name in unknowns:

                u_size = np.size(dunknowns[u_name])
                if comp._jacobian_cache:

                    # We can perform some additional helpful checks.
                    if (u_name, p_name) in comp._jacobian_cache:

                        user = comp._jacobian_cache[(u_name, p_name)].shape

                        # User may use floats for scalar jacobians
                        if len(user) < 2:
                            user = (user[0], 1)

                        if user[0] != u_size or user[1] != p_size:
                            msg = "derivative in component '{}' of '{}' wrt '{}' is the wrong size. " + \
                                  "It should be {}, but got {}"
                            msg = msg.format(cname, u_name, p_name, (u_size, p_size), user)
                            raise ValueError(msg)

                jac_fwd[(u_name, p_name)] = np.zeros((u_size, p_size))
                jac_rev[(u_name, p_name)] = np.zeros((u_size, p_size))

        # Reverse derivatives first
        for u_name in dresids:
            u_size = np.size(dunknowns[u_name])

            # Send columns of identity
            for idx in range(u_size):
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0

                dresids._dat[u_name].val[idx] = 1.0
                try:
                    comp.apply_linear(params, unknowns, dparams,
                                      dunknowns, dresids, 'rev')
                finally:
                    dparams._apply_unit_derivatives()

                for p_name in param_list:

                    dinputs = dunknowns if p_name in states else dparams
                    # try:
                    jac_rev[(u_name, p_name)][idx, :] = dinputs._dat[p_name].val
                    # except:
                    #     pass
        # Forward derivatives second
        for p_name in param_list:

            dinputs = dunknowns if p_name in states else dparams
            p_size = np.size(dinputs[p_name])

            # Send columns of identity
            for idx in range(p_size):
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0

                dinputs._dat[p_name].val[idx] = 1.0
                dparams._apply_unit_derivatives()
                comp.apply_linear(params, unknowns, dparams,
                                  dunknowns, dresids, 'fwd')

                for u_name, u_val in dresids.vec_val_iter():
                    jac_fwd[(u_name, p_name)][:, idx] = u_val

        # Finite Difference goes last
        dresids.vec[:] = 0.0
        root.clear_dparams()
        dunknowns.vec[:] = 0.0

        # Component can request to use complex step.
        if comp.deriv_options['form'] == 'complex_step':
            fd_func = comp.complex_step_jacobian
        else:
            fd_func = comp.fd_jacobian
        jac_fd = fd_func(params, unknowns, resids, option_overrides=comp.deriv_options) #EMG: derv_options were not being passed

        # # Assemble and Return all metrics.
        # _assemble_deriv_data(chain(dparams, states), resids, data[cname],
        #                      jac_fwd, jac_rev, jac_fd, out_stream,
        #                      c_name=cname)

    return jac_fd, jac_fwd, jac_rev