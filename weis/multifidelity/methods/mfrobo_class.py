from __future__ import print_function
import numpy as np
from scipy.stats import truncnorm

from openmdao.api import AnalysisError

from time import time
import dill
import pyDOE

from base_MC_class import BaseMCClass

np.random.seed(31415)


class MFROBO(BaseMCClass):
    def MFMC(self, Din):

        # Output all current results
        with open(self.output_filename, 'wb') as f:
            dill.dump(self, f)

        # Save the current design to the all-design list
        self.D_all.append(Din)

        funcs = self.funcs.copy()

        # Define time taken to run each fidelity
        t_DinT = self.t_DinT
        t_Din = t_DinT.copy()

        Ex_stdx = self.Ex_stdx

        num_keys = len(Ex_stdx)
        Ex = np.zeros((num_keys))
        stdx = np.zeros((num_keys))

        for i, key in enumerate(Ex_stdx):
            Ex[i] = Ex_stdx[key][0]
            stdx[i] = Ex_stdx[key][1]

        # Uncertain parameters
        #  2D: E,G, Mach Number, CT, mrho
        X_lb = (
            Ex - 2 * stdx - Ex
        ) / stdx  # Lower bound for truncated normal distribution
        X_ub = (
            Ex + 2 * stdx - Ex
        ) / stdx  # Upper bound for truncated normal distribution
        nbXsamp = self.nbXsamp  # Number of initial X samples

        # Generate input samples to get sample estimates of correlation and variance for each fidelity
        # Truncated normal distribution
        X = np.zeros((nbXsamp, Ex.shape[0]))
        for i in range(Ex.shape[0]):
            X[:, i] = truncnorm.rvs(
                X_lb[i], X_ub[i], loc=Ex[i], scale=stdx[i], size=nbXsamp
            )

        rhofB = np.zeros(self.num_fidelities + 1)
        sig2fB = np.zeros(self.num_fidelities)
        qfB = np.zeros(self.num_fidelities + 1)
        tau2fB = np.zeros(self.num_fidelities)
        
        rhoCon = np.zeros(self.num_fidelities + 1)
        sig2Con = np.zeros(self.num_fidelities)
        qCon = np.zeros(self.num_fidelities + 1)
        tau2Con = np.zeros(self.num_fidelities)

        m_star = np.ones((self.num_fidelities), dtype=int) * nbXsamp

        self.query_functions(X, funcs, Din, m_star)

        for i in range(self.num_fidelities):
            # Sample correlation coefficient wrt highest fidelity
            rhofB[i] = np.corrcoef(self.fB[:, 0], self.fB[:, i])[0, 1]
            # Sample Variance of each fidelity
            sig2fB[i] = np.var(self.fB[:, i])

            # Sample correlation coefficient for variance samples (change of variable) wrt highest fidelity
            qfB[i] = np.corrcoef(
                (nbXsamp / (nbXsamp - 1))
                * (self.fB[:, 0] - np.mean(self.fB[:, 0])) ** 2,
                (nbXsamp / (nbXsamp - 1))
                * (self.fB[:, i] - np.mean(self.fB[:, i])) ** 2,
            )[0, 1]
            # Sample variance for variance samples (change of variable)
            tau2fB[i] = np.var(
                (nbXsamp / (nbXsamp - 1))
                * (self.fB[:, i] - np.mean(self.fB[:, i])) ** 2
            )


            # Sample correlation coefficient wrt highest fidelity
            rhoCon[i] = np.corrcoef(self.Con[:,0], self.Con[:,i])[0,1]
            # Sample Variance of each fidelity
            sig2Con[i] = np.var(self.Con[:,i])

            # Sample correlation coefficient for variance samples (change of variable) wrt highest fidelity
            qCon[i] = np.corrcoef((nbXsamp/(nbXsamp-1))*(self.Con[:,0]-np.mean(self.Con[:,0]))**2, (nbXsamp/(nbXsamp-1))*(self.Con[:,i]-np.mean(self.Con[:,i]))**2)[0,1]
            # Sample variance for variance samples (change of variable)
            tau2Con[i] = np.var((nbXsamp/(nbXsamp-1))*(self.Con[:,i]-np.mean(self.Con[:,i]))**2)
        print("correlation coeffs:")
        print(rhofB)
        self.rhofB_all.append(rhofB)
        self.qfB_all.append(qfB)

        # Correlation coefficients for models > self.num_fidelities are given a value of 0
        rhofB[self.num_fidelities] = 0
        qfB[self.num_fidelities] = 0

        relevant_funcs = funcs.copy()

        # Average time taken by each fidelity
        t_Din = t_DinT[: self.num_fidelities].copy()

        r = np.ones((self.num_fidelities))
        w = t_Din.copy()  # Cost of each fidelity

        s_p = 0  # Sum of values to be used in calculation of budget p from given J_star

        for i in np.arange(1, self.num_fidelities):
            # Optimal Allocation
            term1 = w[0] * (
                sig2fB[0] * (rhofB[i] ** 2 - rhofB[i + 1] ** 2)
                + tau2fB[0] * (qfB[i] ** 2 - qfB[i + 1] ** 2)
            )
            term2 = w[i] * (
                sig2fB[0] * (1 - rhofB[1] ** 2) + tau2fB[0] * (1 - qfB[1] ** 2)
            )
            r[i] = np.divide(term1, term2, out=np.zeros_like(term1), where=term2!=0)
            if r[i] < 0.:
                r[i] = 0.01  # TODO : this probably isn't correct
            else:
                r[i] = r[i] ** 0.5
                
                s_p = s_p + (1 / r[i] - 1 / r[i - 1]) * (
                    rhofB[i] ** 2 * sig2fB[0] + qfB[i] ** 2 * tau2fB[0]
                )

        ############################################################################
        # Find mean and variance using MFMC
        # Analytic solutions for optimal allocation and weights
        # Optimal control variate coefficients
        alpha_s = rhofB[:-1] * sig2fB[0] ** 0.5 / sig2fB ** 0.5  # For mean estimate
        beta_s = qfB[:-1] * tau2fB[0] ** 0.5 / tau2fB ** 0.5  # For variance estimate
        
        # For constraints
        alpha_Con = rhoCon[:-1] * sig2Con[0]**0.5 / sig2Con**0.5 # For mean estimate
        beta_Con = qCon[:-1] * tau2Con[0]**0.5 / tau2Con**0.5    # For variance estimate

        # Given tolerance for total variance Jstar, find required budget
        p = (np.sum(w * r) * (sig2fB[0] + tau2fB[0] + s_p)) / self.J_star
        self.p_all.append(p)

        # Limit the value of the budget to 500 HF evaluations
        if p > self.maxbudget:
            p = self.maxbudget

        # Optimal number of samples for each fidelity
        m_star = np.ones((self.num_fidelities))
        m_star[0] = p / np.sum(w * r)
        m_star[1:] = m_star[0] * r[1:]

        # Take out the initial number of samples already used for sample estimates of rho, sig, q, tau
        m_star = np.ceil(m_star) - nbXsamp
        m_star[m_star < 0] = 0

        m_star = np.array(m_star, dtype=int)

        # Sample the uncertain parameters (m_star(end) number of samples)
        # Truncated normal distribution
        X = np.zeros((np.max(m_star), Ex.shape[0]))
        for i in range(Ex.shape[0]):
            X[:, i] = truncnorm.rvs(
                X_lb[i], X_ub[i], loc=Ex[i], scale=stdx[i], size=np.max(m_star)
            )

        ##################
        self.query_functions(X, relevant_funcs, Din, m_star)

        m_star = m_star + nbXsamp  # Adding back the initial samples
        self.m_star_all.append(np.array(np.ceil(m_star), dtype=int))

        # Finding the mean and variance estimates using MFMC
        # For highest fidelity
        mfB = np.mean(self.fB[: m_star[0], 0])
        mCon = np.mean(self.Con[:m_star[0],0])

        # Change of variable for variance estimate
        var_samp = (m_star[0] / (m_star[0] - 1)) * (
            self.fB[: m_star[0], 0] - np.mean(self.fB[: m_star[0], 0])
        ) ** 2
        vfB = np.mean(var_samp)
        
        var_sampCon = (m_star[0]/(m_star[0]-1))*(self.Con[:m_star[0],0]-np.mean(self.Con[:m_star[0],0]))**2
        vCon = np.mean(var_sampCon)

        for i in np.arange(1, self.num_fidelities):
            # For lower fidelities: control variates
            mfB = mfB + alpha_s[i] * (
                np.mean(self.fB[: m_star[i], i]) - np.mean(self.fB[: m_star[i - 1], i])
            )
            # Change of variable for variance estimate
            var_samp1 = (m_star[i] / (m_star[i] - 1)) * (
                self.fB[: m_star[i], i] - np.mean(self.fB[: m_star[i], i])
            ) ** 2
            var_samp2 = (m_star[i - 1] / (m_star[i - 1] - 1)) * (
                self.fB[: m_star[i - 1], i] - np.mean(self.fB[: m_star[i - 1], i])
            ) ** 2
            vfB = vfB + beta_s[i] * (np.mean(var_samp1) - np.mean(var_samp2))
            
            mCon = mCon + alpha_Con[i]*(np.mean(self.Con[:m_star[i],i]) - np.mean(self.Con[:m_star[i-1],i]))
            # Change of variable for variance estimate
            var_samp1Con = (m_star[i]/(m_star[i]-1))*(self.Con[:m_star[i],i] - np.mean(self.Con[:m_star[i],i]))**2
            var_samp2Con = (m_star[i-1]/(m_star[i-1]-1))*(self.Con[:m_star[i-1],i] - np.mean(self.Con[:m_star[i-1]+1,i]))**2
            vCon = vCon + beta_Con[i]*(np.mean(var_samp1Con) - np.mean(var_samp2Con))

        self.mfB.append(mfB)
        self.vfB.append(vfB)
        self.mCon.append(mCon)
        self.vCon.append(vCon)
        self.fB_all.append(self.fB)

    def master_func(self, Din):
        """
        Master function that takes in the design vector.
        It runs the MFMC on the design point if we haven't already.
        It then returns an index, -1, to refernece the most recent results generated.

        If we already have results for the design point, this function
        returns the index corresponding to the correct results.

        This function is called by the objective and constraint functions.
        """

        Din = np.array(Din)

        if len(self.D_all) == 0:
            run_case = True
        else:
            summ = np.where(np.all(Din == np.array(self.D_all), axis=1))

            if len(summ[0]):
                run_case = False
                idx = summ[0][0]
            else:
                run_case = True

        if run_case:
            # Clear out the results from a previous design point
            self.fB = np.zeros((0, self.num_fidelities))
            self.Con = np.zeros((0, self.num_fidelities))

            self.MFMC(Din)

            # Robust optimization objective
            idx = -1
            
        else:
            print('Skipping; case already ran')

        return idx

    def obj_func(self, Din):

        idx = self.master_func(Din)
        RO_obj = self.mfB[idx] + self.eta * self.vfB[idx] ** 0.5

        print("Design", Din)
        print("ObjFun", RO_obj)
        
        if np.isnan(RO_obj):
            RO_obj = 1e10
            
        return RO_obj
        
    def fake_con(self, Din):

        idx = self.master_func(Din)
        Con_con = self.mCon[idx] + self.eta*self.vCon[idx]**0.5

        print('Con con', Con_con)
        return Con_con
