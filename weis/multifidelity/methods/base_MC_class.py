from __future__ import print_function
import numpy as np
from scipy.stats import truncnorm

from time import time
import dill
import pyDOE


class BaseMCClass:
    def __init__(
        self,
        funcs,
        Ex_stdx,
        eta,
        J_star,
        output_filename,
        X_bounds=None,
        nbXsamp=5,
        num_timing_samples=101,
        useHFonly=True,
        scaleVar_option=True,
        rad_IR=0,
    ):

        self.Ex_stdx = Ex_stdx
        self.eta = eta
        self.J_star = J_star
        self.output_filename = output_filename
        self.X_bounds = X_bounds
        self.nbXsamp = nbXsamp
        self.num_timing_samples = num_timing_samples
        self.useHFonly = useHFonly
        self.varFailCount = (
            0  # Counter for cases when MFMC variance fails and only HF is used
        )
        self.scaleVar_option = scaleVar_option  # scale the variance of variance using a scaling factor if scaleVar_option = True
        self.rad_IR = (
            rad_IR  # radius of hypersphere to consider using information reuse
        )
        # self.X_fixed = X_fixed  # fixed set of samples used for all designs for information reuse

        self.funcs = funcs
        self.num_fidelities = len(funcs)

        self.fB = np.zeros((0, self.num_fidelities))
        self.Con = np.zeros((0, self.num_fidelities))

        # Store history parameters
        self.D_all = []
        self.mfB = []
        self.vfB = []
        self.mCon = []
        self.vCon = []
        self.m_star_all = []
        self.rhofB_all = []
        self.qfB_all = []
        self.p_all = []
        self.fB_all = []

        self.scaler = 1e3
        self.maxbudget = 500 * 0.47416563  # 500 HF evaluations

    def getFPTimings(self):
        """
        I don't think this is set up correctly as-is
        """

        Ex_stdx = self.Ex_stdx

        num_keys = len(Ex_stdx)
        Ex = np.zeros((num_keys))
        stdx = np.zeros((num_keys))

        for i, key in enumerate(Ex_stdx):
            Ex[i] = Ex_stdx[key][0]
            stdx[i] = Ex_stdx[key][1]

        X_lb = (
            Ex - 2 * stdx - Ex
        ) / stdx  # Lower bound for truncated normal distribution
        X_ub = (
            Ex + 2 * stdx - Ex
        ) / stdx  # Upper bound for truncated normal distribution
        nt = self.num_timing_samples

        # Generate input samples to get sample estimates of correlation and variance for each fidelity
        # Truncated normal distribution
        X = np.zeros((nt, Ex.shape[0]))
        for i in range(Ex.shape[0]):
            X[:, i] = truncnorm.rvs(X_lb[i], X_ub[i], loc=Ex[i], scale=stdx[i], size=nt)

        Din = np.zeros((6, nt))

        # lb = np.array([-10., -10., -10., 0.5, 0.5, 0.5, -5.])
        # ub = np.array([10., 10., 10., 3., 3., 3., 10.])

        lb = np.array([-1.0, -1.0, 1.7, 1.7, 1.7, 4.0])
        ub = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 5.0])

        Din = np.random.uniform(lb, ub, (nt, 7))
        m_star = np.ones(1, dtype=int) * nt

        self.time_diff = np.zeros((self.num_fidelities, 2))
        for j, m_star_version in enumerate([m_star, np.ones((1), dtype=int)]):
            for i, FP in enumerate(self.FPtot):
                FP = np.atleast_2d(FP)
                self.fB = np.zeros((nt, self.num_fidelities))
                self.SF = np.zeros((nt, self.num_fidelities))
                self.LC = np.zeros((nt, self.num_fidelities))
                self.CM = np.zeros((nt, self.num_fidelities))
                s = time()
                self.query_functions(X, FP, Din[0, :], m_star_version)
                self.time_diff[i, j] = time() - s

        timings = (self.time_diff[:, 0] - self.time_diff[:, 1]) / (nt - 1)
        self.t_DinT = timings

        print("times:", timings)

    def sampleLHS(self):

        if self.X_bounds is None:
            Exception("Need to supply bounds to do LHS!")

        Ex_stdx = self.Ex_stdx

        num_keys = len(Ex_stdx)
        Ex = np.zeros((num_keys))
        stdx = np.zeros((num_keys))

        for i, key in enumerate(Ex_stdx):
            Ex[i] = Ex_stdx[key][0]
            stdx[i] = Ex_stdx[key][1]

        X_lb = (
            Ex - 2 * stdx - Ex
        ) / stdx  # Lower bound for truncated normal distribution
        X_ub = (
            Ex + 2 * stdx - Ex
        ) / stdx  # Upper bound for truncated normal distribution
        nt = 2

        # Generate input samples to get sample estimates of correlation and variance for each fidelity
        # Truncated normal distribution
        X = np.zeros((nt, Ex.shape[0]))
        for i in range(Ex.shape[0]):
            X[:, i] = truncnorm.rvs(X_lb[i], X_ub[i], loc=Ex[i], scale=stdx[i], size=nt)

        lb = self.X_bounds[0]
        ub = self.X_bounds[1]

        mean = (ub + lb) / 2.0
        stdx = (ub - mean) / 2.0

        D_list = []
        fB_list = []

        lhs = pyDOE.lhs(len(lb), samples=100, criterion="center")

        for k, Din in enumerate(lhs):

            if 1:  # this is LHS
                Din = Din * (ub - lb) + lb
            else:  # this is truncated normal
                Din = np.zeros((len(lb)))
                for i in range(len(lb)):
                    Din[i] = truncnorm.rvs(
                        lb[i], ub[i], loc=mean[i], scale=stdx[i], size=1
                    )

            m_star = np.ones(self.num_fidelities, dtype=int) * nt

            self.fB = np.zeros((0, self.num_fidelities))
            self.SF = np.zeros((0, self.num_fidelities))
            self.LC = np.zeros((0, self.num_fidelities))
            self.CM = np.zeros((0, self.num_fidelities))
            self.query_functions(X, self.FPtot, Din, m_star)

            D_list.append(Din)
            fB_list.append(self.fB)

        print()
        print()

    def query_functions(self, X, funcs, DX, m_star):
        new_fB = np.zeros((np.max(m_star), self.num_fidelities))
        new_Con = np.zeros((np.max(m_star), self.num_fidelities))

        for j, m in enumerate(m_star):
            func = funcs[j]
            for i in range(m):
                if m > 0:
                    new_fB[i, j] = func(DX) + func(X[i]) / 2. + (np.random.random() - 0.5) / 2.
                    new_Con[i, j] = DX - 0.4 - X[i]

        self.fB = np.vstack((self.fB, new_fB))
        self.Con = np.vstack((self.Con, new_Con))
        
        print()
