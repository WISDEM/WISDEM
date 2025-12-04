import time
from itertools import islice

import numpy as np
from numpy.typing import ArrayLike

from openmdao.utils.mpi import MPI

from wisdem.optimization_drivers.nsga2.fast_nondom_sort import fast_nondom_sort
from wisdem.optimization_drivers.nsga2.crowding_distance_assignment import crowding_distance_assignment
from wisdem.optimization_drivers.nsga2.genetic_functions import (
    binary_tournament_selection,
    polynomial_mutation,
    simulated_binary_crossover,
)


class NSGA2:

    N_DV: int = 0  # number of design variables
    N_constr: int = 0  # number of constraints
    N_obj: int = 0  # number of objective functions
    N_population: int = 0  # number of members in the population

    design_vars_population: ArrayLike = np.array([])  # design variables
    objs_population: ArrayLike = np.array([])  # objective values at DV points
    constrs_population: ArrayLike = np.array([])  # constraint values at DV points
    feasibility_population: ArrayLike = np.array([])  # feasibility values at DV points
    fun_combined: callable = None  # combined objective+constraint function storage

    # front organization
    idx_fronts: list[list[int]] = None  # list of indices into raw DVs to ID fronts
    design_vars_fronts: list[np.ndarray] = None  # list of views into DVs by front
    objs_fronts: list[np.ndarray] = None  # list of views into objectives by front
    constrs_fronts: list[np.ndarray] = None  # list of views into constraints by front
    feasibility_fronts: list[bool] = None  # list of feasibility values for fronts

    rate_crossover: float = 0.9  # crossover probability
    eta_c: float = 20  # distribution index
    rate_mutation: float = 0.1  # mutation probability
    eta_m: float = 20  # distribution index

    feasibility_dominates: bool = True  # use feasibility in the sorting process?

    if MPI:
        comm_mpi: MPI.Comm = None  # MPI communicator for parallel evaluation
    model_mpi: tuple[int, int] = None  # parallelization model: size, color
    # follows the format used by openmdao/openmdao/utils/concurrent_utils.py

    rng_seed: int = None  # a random number generator seed
    accelerated: bool = True  # should we use numba acceleration

    verbose: bool = False  # verbosity switch

    def __init__(
        self,
        design_vars_init: ArrayLike,  # initial population
        fun_combined: callable,  # callable combo objective+constraint functions: DVs -> objs+constrs
        N_obj: int,  # manual counts of objectives
        N_constr: int = 0,  # manual counts of constraints
        design_vars_l: ArrayLike = None,  # the lower bound of the DVs
        design_vars_u: ArrayLike = None,  # the upper bound of the DVs
        params_override=(None, None, None, None),  # override params for NSGA-II
        comm_mpi = None,  # communicator for parallel implementation, comm_mpi should be MPI.Comm
        model_mpi: tuple[int, int] = None,  # model for spreading work across processes
        verbose: bool = False,  # verbose outputs
        rng_seed: int = None,  # rng seed
    ):
        """
        initialize NSGA2 optimizer and its population

        Parameters
        ----------
        design_vars_init : ArrayLike
            initial population for design varibales
        fun_combined : callable
            a function that, given any row in design_vars_init, returns the objectives and constraints
        ... TODO
        N_obj : int, optional
            the number of objectives for an expensive evaluation function, by default None
        design_vars_l : _type_, optional
            lower bounds on the design variables, if None, defaults to -inf, by default None
        design_vars_u : _type_, optional
            upper bounds on the design variables, if None, defaults to inf, by default None
        """

        # install provided settings
        if params_override[0] is not None:
            self.rate_crossover = params_override[0]
        if params_override[1] is not None:
            self.eta_c = params_override[1]
        if params_override[2] is not None:
            self.rate_mutation = params_override[2]
        if params_override[3] is not None:
            self.eta_m = params_override[3]
        self.comm_mpi = comm_mpi
        self.model_mpi = model_mpi
        self.verbose = verbose

        # take in an initial population of design variables
        design_vars_init = np.atleast_2d(design_vars_init)  # convert to numpy if necessary
        self.N_population, self.N_DV = design_vars_init.shape  # extract sizes
        self.design_vars_population = design_vars_init.copy()  # hold on to a copy

        # broadcast the initial population to all MPI ranks
        if self.comm_mpi is not None:
            self.design_vars_population = self.comm_mpi.bcast(self.design_vars_population, root=0)

        # set up the counts
        self.N_obj = N_obj
        self.N_constr = N_constr
        # use feasibility-based domination when a constraint is passed in
        if self.N_constr:
            self.feasibility_dominates = True

        # initialize objectives and constraints to all nan
        self.objs_population = np.nan * np.ones((self.N_population, self.N_obj))
        self.constrs_population = np.nan * np.ones((self.N_population, self.N_constr))
        self.needs_recompute = [True for _ in range(self.N_population)]  # initialize recompute to yes
        self.feasibility_population = np.ones((self.N_population,), dtype=bool)  # initialize feasibility to ones

        # install evaluation functions for objectives and constraints
        self.fun_combined = fun_combined
        self.update_data()  # now that there are functions, update values

        design_vars_l = np.array(design_vars_l)  # convert to np.array
        design_vars_u = np.array(design_vars_u)  # convert to np.array
        # validate the sizes of the design variables
        if design_vars_l.size != self.N_DV:
            raise ValueError(f"Lower bounds size mismatch: expected {self.N_DV}, got {design_vars_l.size}.")
        if design_vars_u.size != self.N_DV:
            raise ValueError(f"Upper bounds size mismatch: expected {self.N_DV}, got {design_vars_u.size}.")
        self.design_vars_l = design_vars_l  # save the lower bounds on design variables
        self.design_vars_u = design_vars_u  # save the upper bounds on design variables

        # set numpy random number generator if specified
        if rng_seed is not None:
            self.rng_seed = rng_seed
        self._rng_seed_generator = np.random.default_rng(self.rng_seed)
        if self.comm_mpi is not None:
            self._rng_seed_generator = self.comm_mpi.bcast(self._rng_seed_generator, root=0)

    def next_seed(self):
        """
        Returns repeatable pseudo-random integers to use as seed using the class RNG
        """
        return self._rng_seed_generator.integers(100000000)

    def update_feasibility(self):
        """
        Update the internal feasibility information
        """

        if self.N_constr:
            self.feasibility_population = np.all(self.constrs_population >= 0.0, axis=1)

    def update_data_external(
        self,
        design_vars_p: np.ndarray,
        objs_p: np.ndarray,
        needs_recompute: list[bool],
        constrs_p: np.ndarray = None,
    ):
        """
        Update external objectives (and constraints, if applicable) that are
        flagged for recomputation from a provided dataset.

        Parameters
        ----------
        design_vars_p : np.ndarray
            incoming population's design variable specifications
        objs_p : np.ndarray
            array of objective outcomes
        needs_recompute_objs : list[bool]
            list of objective evaluations that need re-computation
        constrs_p : np.ndarray, optional
            array of constraint outcomes

        Returns
        -------
        np.ndarray
            returns a reference to objs_p
        np.ndarray, optional
            returns a reference to constrs_p

        Raises
        ------
        ValueError
            if the sizes of DV and objective/constraint populations are mismatched
        """

        # initialize some counters
        neval_obj_update = 0
        neval_constr_update = 0

        if self.verbose:
            print("UPDATING DATA...", end="", flush=True)
            tm_st = time.time()

        # size and validate the input data
        N_pop, N_DV = design_vars_p.shape
        xN_pop, N_obj = objs_p.shape
        if N_pop != xN_pop:
            raise ValueError(
                f"Dimension mismatch: design_vars_p has {N_pop} individuals, objs_p has {xN_pop} individuals."
            )
        if constrs_p is not None:
            xN_pop, N_constr = constrs_p.shape
            if N_pop != xN_pop:
                raise ValueError(
                    f"Dimension mismatch: design_vars_p has {N_pop} individuals, constrs_p has {xN_pop} individuals."
                )
        else:
            N_constr = 0

        # gather indices that need recomputation
        indices_to_update = [i for i, flag in enumerate(needs_recompute) if flag]
        args_to_eval = [design_vars_p[i, :] for i in indices_to_update]

        # evaluate in batch if possible, else fallback to single
        if args_to_eval:
            if self.comm_mpi is None:
                results_combo = [self.fun_combined(arg) for arg in args_to_eval]
                results_obj = [v[:N_obj] for v in results_combo]
                if N_constr:
                    results_constr = [v[N_obj : (N_obj + N_constr)] for v in results_combo]
            else:
                # distribute the evaluation across MPI processes
                comm = self.comm_mpi
                rank = comm.rank

                if self.model_mpi is not None:  # i.e.: parallelization model is specified
                    size, color = self.model_mpi
                    local_args = islice(args_to_eval, color, None, size)  # slice by color
                else:
                    size = comm.size
                    local_args = islice(args_to_eval, rank, None, size)  # slice by rank

                # scatter the indices to all processes
                local_results_combo = [self.fun_combined(arg) for arg in local_args]
                local_results_obj = [v[:N_obj] for v in local_results_combo]
                if self.N_constr:
                    local_results_constr = [v[N_obj : (N_obj + N_constr)] for v in local_results_combo]

                # allgather all results
                gathered_results_obj = comm.allgather(local_results_obj)
                if self.N_constr:
                    gathered_results_constr = comm.allgather(local_results_constr)

                # flatten the gathered results on rank 0
                if rank == 0:
                    results_obj = [item for sublist in gathered_results_obj for item in sublist]
                    if self.N_constr:
                        results_constr = [item for sublist in gathered_results_constr for item in sublist]
                else:
                    results_obj = None
                    if self.N_constr:
                        results_constr = None

                # broadcast results to all processes
                results_obj = comm.bcast(results_obj, root=0)
                if self.N_constr:
                    results_constr = comm.bcast(results_constr, root=0)

            # assign results across processors
            v2w = [results_obj]
            if self.N_constr:
                v2w.append(results_constr)
            for zipval in zip(indices_to_update, *v2w):
                idx = zipval[0]  # unpack the index
                objs_p[idx, :] = zipval[1]  # assign the objective
                if self.N_constr:
                    constrs_p[idx, :] = zipval[2]  # assign the constraint val
                    neval_constr_update += 1  # update the counter
                needs_recompute[idx] = False  # update the recompute vector
                neval_obj_update += 1  # update the counter

        if self.verbose:
            tm_end = time.time()
            print(
                f" DONE. USING {neval_obj_update} OBJECTIVE FUNCTION CALLS IN {tm_end-tm_st:.4f}s.",
                flush=True,
            )

        # put together the return values
        rv = [objs_p]
        if self.N_constr:
            rv.append(constrs_p)

        return tuple(rv)

    def update_data(self):
        """
        Update the internal objectives.

        Returns
        -------
        np.ndarray
            a reference to objs_p
        np.ndarray, optional
            a reference to constrs_p if constraints exist
        """

        if np.any(self.needs_recompute):
            # any time there's an update, the front data becomes stale
            self.idx_fronts = None
            self.design_vars_fronts = None
            self.objs_fronts = None
            self.constrs_fronts = None
            self.feasibility_fronts = None

        # update the data that needs updating
        ude_ret = self.update_data_external(
            self.design_vars_population,
            self.objs_population,
            self.needs_recompute,
            constrs_p=self.constrs_population,
        )

        # update the feasibility with the new constraint values
        self.update_feasibility()

        return ude_ret

    def compute_fronts_external(
        self,
        design_vars_in: np.ndarray,
        objs_in: np.ndarray,
        needs_recompute: list[bool],
        constrs_in: np.ndarray = None,
        feasibility_dominates: bool = True,
    ):
        """
        Compute the non-dominated fronts data from the provided data.

        This function coputes the non-dominated fronts from the provided design
        variables (design_vars_in) and objective values (objs_in). It updates the
        objectives if necessary and returns the indices of the fronts.

        Parameters
        ----------
        design_vars_in : np.ndarray
            population design varible input
        objs_in : np.ndarray
            population objective input
        needs_recompute : list[bool]
            flags for recomputation of the individual objectives and constraints
        ...

        Returns
        -------
        list[list[int]]
            indices of the non-dominated fronts in the population
        """

        # first, update any objectives and constraints
        self.update_data_external(design_vars_in, objs_in, needs_recompute, constrs_p=constrs_in)

        # create a list to pass to the sorting algorithm
        objs_tosort = list(map(tuple, objs_in))
        # TODO: figure out how to pass the map object and still have it work

        if self.verbose:  # if verbose, print and start timer
            print("COMPUTING THE PARETO FRONTS...", end="", flush=True)
            tm_st = time.time()
        # do the fast, non-dominated sort algorithm to sort
        fns_function = fast_nondom_sort if self.accelerated else fast_nondom_sort.function_nojit
        idx_fronts = fns_function(objs_tosort)  # indices of the fronts
        if self.verbose:  # if verbose, stop timer and print
            tm_end = time.time()
            print(f" DONE. TIME: {tm_end-tm_st:.4f}s", flush=True)

        if (constrs_in is not None) and feasibility_dominates:
            if self.verbose:  # if verbose, print and start timer
                print("RESORTING THE PARETO FRONTS FOR FEASIBILITY...", end="", flush=True)
                tm_st = time.time()

            feasibility_fronts = [constrs_in[idx_dv, :] >= 0.0 for idx_dv in idx_fronts]

            idx_fronts_feasible = []
            idx_fronts_infeasible = []

            for idx_f, f in enumerate(idx_fronts):
                is_feasible = np.all(feasibility_fronts[idx_f], axis=1)
                points_added = 0
                if np.sum(is_feasible):
                    idx_fronts_feasible.append(np.array(f)[is_feasible].tolist())
                    points_added += len(idx_fronts_feasible[-1])
                if np.sum(~is_feasible):
                    idx_fronts_infeasible.append(np.array(f)[~is_feasible].tolist())
                    points_added += len(idx_fronts_infeasible[-1])
                assert len(f) == points_added

            idx_fronts = idx_fronts_feasible + idx_fronts_infeasible

            if self.verbose:  # if verbose, stop timer and print
                tm_end = time.time()
                print(f" DONE. TIME: {tm_end-tm_st:.4f}s", flush=True)

        return idx_fronts

    def compute_fronts(self):
        """
        Get the non-dominated front map for the current population.

        This function computes the non-dominated fronts from the optimizer's
        internal design variables (design_vars_population) and objective values
        (objs_population). It updates the objectives if necessary and returns the
        indices of the fronts.

        Returns
        -------
        list[list[int]]
            list of maps into self.design_vars_population and self.objs_population
            representing each front
        """

        # pass internal data to the external front computation pieces
        self.idx_fronts = self.compute_fronts_external(
            design_vars_in=self.design_vars_population,
            objs_in=self.objs_population,
            needs_recompute=self.needs_recompute,
            constrs_in=self.constrs_population,
            feasibility_dominates=self.feasibility_dominates,
        )

        return self.idx_fronts

    def get_Nfronts(self):
        """get the number of non-dominated fronts we're dealing with"""
        if self.idx_fronts is None:
            return 0  # no data, no fronts
        return len(self.idx_fronts)

    def get_fronts_external(
        self,
        design_vars_in: np.ndarray,
        objs_in: np.ndarray,
        needs_recompute: list[bool],
        idx_fronts_in: list[list[int]] = None,
        compute_design_vars: bool = True,
        compute_objs: bool = True,
        constrs_in: np.ndarray = None,
        compute_constrs: bool = False,
        feasibility_dominates: bool = False,
    ):
        """
        Get the non-dominated front data from the provided data.

        This function returns the data for the non-dominated fronts from the
        provided design variables (design_vars_in) and objective values (objs_in). It
        updates the objectives if necessary and returns the indices of the
        fronts, along with the corresponding design variables and objective
        values if requested.

        Parameters
        ----------
        design_vars_in : np.ndarray
            population design varible input
        objs_in : np.ndarray
            population objective input
        needs_recompute : list[bool]
            flags for recomputation of the individual objectives or constraints
        compute_design_vars : bool, optional
            should the design var. fronts be computed and returned, by default True
        compute_objs : bool, optional
            should the objective fronts be computed and returned, by default True
        ... TODO

        Returns
        -------
        list[list[int]]
            indices of the non-dominated fronts in the population
        list[np.ndarray], optional
            design variables for the fronts, if compute_design_vars is True
        list[np.ndarray], optional
            objective values for the fronts, if compute_objs is True
        """

        # compute the fronts data
        idx_fronts = (
            self.compute_fronts_external(
                design_vars_in,
                objs_in,
                needs_recompute,
                constrs_in=constrs_in,
                feasibility_dominates=feasibility_dominates,
            )
            if idx_fronts_in is None
            else idx_fronts_in
        )

        # front re-computation
        design_vars_fronts = []
        objs_fronts = []
        constrs_fronts = []
        for idx_f, f in enumerate(idx_fronts):
            if compute_design_vars:
                design_vars_f = design_vars_in[f, :]  # slice index in to create views
                design_vars_fronts.append(design_vars_f)
            if compute_objs:
                objs_f = objs_in[f, :]  # slice index in to create views
                objs_fronts.append(objs_f)
            if compute_constrs:
                if constrs_in is None:
                    raise ValueError("Cannot compute constraints fronts without constrs_in being provided.")
                constrs_f = np.zeros(shape=(len(f), 0)) if len(constrs_in) == 0 else constrs_in[f, :]  # slice index in to create views
                constrs_fronts.append(constrs_f)

        # compile returns and ship
        to_return = [idx_fronts]
        if compute_design_vars:
            to_return.append([np.array(v) for v in design_vars_fronts])
        if compute_objs:
            to_return.append([np.array(v) for v in objs_fronts])
        if compute_constrs:
            to_return.append([np.array(v) for v in constrs_fronts])

        return tuple(to_return)

    def get_fronts(
        self,
        compute_design_vars: bool = True,
        compute_objs: bool = True,
        compute_constrs: bool = True,
        feasibility_dominates: bool = True,
    ):
        """
        Get the non-dominated fronts from the current population.

        This function computes the non-dominated fronts from the optimizer's
        internal design variables (design_vars_population) and objective values
        (objs_population). It updates the objectives if necessary and returns the
        indices of the fronts, along with the corresponding design variables and
        objective values if requested.

        Parameters
        ----------
        compute_design_vars : bool, optional
            should the design var. fronts be computed and returned, by default True
        compute_objs : bool, optional
            should the objective fronts be computed and returned, by default True
        ... TODO

        Returns
        -------
        list[list[int]]
            indices of the non-dominated fronts in the population
        list[np.ndarray], optional
            design variables for the fronts, if compute_design_vars is True
        list[np.ndarray], optional
            objective values for the fronts, if compute_objs is True
        """

        values_return = self.get_fronts_external(
            self.design_vars_population,
            self.objs_population,
            self.needs_recompute,
            self.idx_fronts,
            compute_design_vars,
            compute_objs,
            constrs_in=self.constrs_population,
            compute_constrs=compute_constrs,
            feasibility_dominates=feasibility_dominates,
        )

        # deal with the return values
        rvi = 0
        self.idx_fronts = values_return[rvi]
        rvi += 1
        if compute_design_vars:
            self.design_vars_fronts = values_return[rvi]
            rvi += 1
        if compute_objs:
            self.objs_fronts = values_return[rvi]
            rvi += 1
        if compute_constrs:
            self.constrs_fronts = values_return[rvi]
            rvi += 1

        return values_return

    def get_crowding_distance_data(
        self,
        objs_front_in: list[np.ndarray],
    ):

        # return the crowding distances for an input set of fronts

        # get the front
        if self.verbose:
            print("COMPUTING CROWDING DISTANCE...", end="", flush=True)
            tm_st = time.time()
        cda_function = crowding_distance_assignment if self.accelerated else crowding_distance_assignment.function_nojit
        D_front = [cda_function(f) for f in objs_front_in]
        if self.verbose:
            tm_end = time.time()
            print(f" DONE. TIME: {tm_end-tm_st:.4f}s", flush=True)

        # return the crowding distances
        return D_front

    def get_rank_data(
        self,
        objs_front_in: np.ndarray,
        local: bool = False,
    ):
        """
        get the rankings of the fronts on a set of data supplied

        Parameters
        ----------
        objs_front_in : np.ndarray
            a list of fronts represented by numpy arrays w/ values of the objectives
        constrs_front_in : np.ndarray, optional
            a list of fronts represented by numpy arrays w/ values of the constraints, by default None
        local : bool, optional
            return a local (i.e. within-front) ranking, by default False

        Returns
        -------
        list[list[int]]
            a list of the requested index rankings of the fronts
        """

        # get the global/local ranking of the points
        D_front = self.get_crowding_distance_data(objs_front_in)

        count_front = ([0] + [len(f) for f in D_front])[:-1]  # get the front sizes
        localR_front = [np.argsort(-D).tolist() for D in D_front]  # sort on crowding distance within fronts

        if local:
            return localR_front  # we're done if we just want intra-front ranking

        # continue on and return global ranking
        R_front = [(R + np.cumsum(count_front)[i]).tolist() for i, R in enumerate(localR_front)]
        return R_front

    def get_rank(
        self,
        local: bool = False,
    ):
        """
        get the rank
        """

        return self.get_rank_data(
            objs_front_in=self.objs_fronts,
            local=local,
        )

    def sort_data(self):
        """
        re-sort the raw data so it's in rank order
        """

        # get the fronts and unpack
        rv = self.get_fronts(
            feasibility_dominates=self.feasibility_dominates,
            compute_constrs=True if self.N_constr else False,
        )
        design_vars_fronts = rv[1]
        objs_fronts = rv[2]
        if self.N_constr:
            constrs_fronts = rv[3]

        D_fronts = self.get_crowding_distance_data(self.objs_fronts)  # crowding distances
        idx_argsort_D = [np.argsort(Df)[::-1] for Df in D_fronts]  # get argsort in the front

        # new placeholders
        idx_fronts_new = []
        design_vars_fronts_new = []
        objs_fronts_new = []
        if self.N_constr:
            constrs_fronts_new = []

        front_accumulation = 0
        for idx_f, argsort_Df in enumerate(idx_argsort_D):  # loop over the fronts
            # populate our new fronts
            idx_fronts_new.append([front_accumulation + v for v in list(range(len(argsort_Df)))])
            front_accumulation += len(argsort_Df)
            design_vars_fronts_new.append(design_vars_fronts[idx_f][argsort_Df, :])
            objs_fronts_new.append(objs_fronts[idx_f][argsort_Df, :])
            if self.N_constr:
                constrs_fronts_new.append(constrs_fronts[idx_f][argsort_Df, :])

        # store all the results
        self.design_vars_population = np.vstack(design_vars_fronts_new)
        self.objs_population = np.vstack(objs_fronts_new)
        self.idx_fronts = idx_fronts_new
        self.design_vars_fronts = design_vars_fronts_new
        self.objs_fronts = objs_fronts_new
        if self.N_constr:
            self.constrs_fronts = constrs_fronts_new

        # broadcast all the results
        if self.comm_mpi is not None:
            self.design_vars_population = self.comm_mpi.bcast(self.design_vars_population, root=0)
            self.objs_population = self.comm_mpi.bcast(self.objs_population, root=0)
            self.idx_fronts = self.comm_mpi.bcast(self.idx_fronts, root=0)
            self.design_vars_fronts = self.comm_mpi.bcast(self.design_vars_fronts, root=0)
            self.objs_fronts = self.comm_mpi.bcast(self.objs_fronts, root=0)
            if self.N_constr:
                self.constrs_fronts = self.comm_mpi.bcast(self.constrs_fronts, root=0)

        # sanity check the re-sorting
        if not np.allclose(self.design_vars_population.shape, np.array([self.N_population, self.N_DV])):
            raise ValueError(
                f"new design_vars_population size mismatch: expected ({self.N_population}, {self.N_DV}), got {self.design_vars_population.shape}."
            )
        if not np.allclose(self.objs_population.shape, np.array([self.N_population, self.N_obj])):
            raise ValueError(
                f"new objs_population size mismatch: expected ({self.N_population}, {self.N_obj}), got {self.objs_population.shape}."
            )

    def get_binary_tournament_selection(self, ratio_keep=0.5):
        """run a binary tournament selection"""

        rank = np.hstack(self.get_rank())  # get the overall ranking
        # run, return binary tournament selection on ranking
        bts_function = (
            binary_tournament_selection
            if self.accelerated and (self.rng_seed is None)
            else binary_tournament_selection.function_nojit
        )
        idx_select = bts_function(
            rank,
            ratio_keep=ratio_keep,
            rng_seed=self.next_seed(),
        )
        return idx_select

    def _get_default_limits(self):
        design_vars_l = (
            -np.inf * np.ones_like(self.design_vars_raw[0]) if (self.design_vars_l is None) else self.design_vars_l
        )
        design_vars_u = (
            np.inf * np.ones_like(self.design_vars_raw[0]) if (self.design_vars_u is None) else self.design_vars_u
        )
        return design_vars_l, design_vars_u

    def propose_new_generation(self):

        # get the limits
        design_vars_l, design_vars_u = self._get_default_limits()  # get the default limits

        # use the selection to generate a new (elitist) population
        idx_selection = self.get_binary_tournament_selection(ratio_keep=1.0)

        # create new proposed populations
        design_vars_proposal = self.design_vars_population[idx_selection, :].copy()
        objs_proposal = self.objs_population[idx_selection, :].copy()
        if self.N_constr:
            constrs_proposal = self.constrs_population[idx_selection, :].copy()

        # array of changes
        changed = np.array([False for _ in design_vars_proposal])

        # now, try a crossover
        sbx_function = (
            simulated_binary_crossover
            if self.accelerated and (self.rng_seed is None)
            else simulated_binary_crossover.function_nojit
        )
        design_vars_proposal, changed_crossover = sbx_function(
            design_vars_proposal,
            design_vars_l=design_vars_l,
            design_vars_u=design_vars_u,
            rate_crossover=self.rate_crossover,
            eta_c=self.eta_c,
            rng_seed=self.next_seed(),
        )
        assert len(design_vars_proposal) == self.N_population
        changed = np.logical_or(changed, changed_crossover)

        # now, do a mutation
        pm_function = (
            polynomial_mutation if self.accelerated and (self.rng_seed is None) else polynomial_mutation.function_nojit
        )
        design_vars_proposal, changed_mutation = pm_function(
            design_vars_proposal,
            design_vars_l=design_vars_l,
            design_vars_u=design_vars_u,
            rate_mutation=self.rate_mutation,
            eta_m=self.eta_m,
            rng_seed=self.next_seed(),
        )
        assert len(design_vars_proposal) == self.N_population
        changed = np.logical_or(changed, changed_mutation)

        # check if DVs are violating the bounds
        if np.any(design_vars_proposal < design_vars_l) or np.any(design_vars_proposal > design_vars_u):
            raise ValueError(
                "Proposed design variables violate the bounds. "
                f"Lower bounds: {design_vars_l}, Upper bounds: {design_vars_u}, "
                f"Proposed: {design_vars_proposal}"
            )

        # update the objectives that have changed
        self.update_data_external(
            design_vars_proposal,
            objs_proposal,
            needs_recompute=changed,
            constrs_p=constrs_proposal if self.N_constr else None,
        )

        # return the results
        rv = [design_vars_proposal, objs_proposal, changed]
        if self.N_constr:
            rv.append(constrs_proposal)
        return rv

    def iterate_population(self):

        # get previous, proposed next populations
        design_vars_prev, objs_prev, constrs_prev = (
            self.design_vars_population,
            self.objs_population,
            self.constrs_population,
        )

        # run on root
        if self.verbose:
            print("PROPOSING NEW GENERATION...", end="", flush=True)
        rv = self.propose_new_generation()

        # run this code in serial or on root
        if (not self.comm_mpi) or self.comm_mpi.Get_rank() == 0:

            design_vars_next = rv[0]
            objs_next = rv[1]
            changed_next = rv[2]
            if self.N_constr:
                constrs_next = rv[3]

            # combine the populations and compute the fronts
            design_vars_combo = np.vstack([design_vars_prev, design_vars_next])
            objs_combo = np.vstack([objs_prev, objs_next])
            changed_combo = np.hstack([self.needs_recompute, changed_next])
            if self.N_constr:
                constrs_combo = np.vstack([constrs_prev, constrs_next])

            # compute the fronts of the combined dataset
            rv = self.get_fronts_external(
                design_vars_combo,
                objs_combo,
                changed_combo,
                constrs_in=constrs_combo if self.N_constr else None,
                compute_constrs=True if self.N_constr else False,
                feasibility_dominates=self.feasibility_dominates,
            )
            idx_fronts = rv[0]
            design_vars_fronts = rv[1]
            objs_fronts = rv[2]
            if self.N_constr:
                constrs_fronts = rv[3]
            R_fronts = self.get_rank_data(objs_fronts, local=True)

            # new data
            self.design_vars_population = []
            self.objs_population = []
            self.constrs_population = []
            self.idx_fronts = []

            idx_counter = 0  # count how many we get in there
            for idx_f, f in enumerate(R_fronts):
                if idx_counter >= self.N_population:
                    break
                self.idx_fronts.append([])  # add a new front to the map
                for idx_v in f:  # for each index in the front
                    if idx_counter >= self.N_population:
                        break
                    self.design_vars_population.append(design_vars_fronts[idx_f][idx_v])  # add to the re-sort
                    self.objs_population.append(objs_fronts[idx_f][idx_v])  # add to the re-sort
                    if self.N_constr:
                        self.constrs_population.append(constrs_fronts[idx_f][idx_v])  # add to the re-sort
                    self.idx_fronts[idx_f].append(idx_counter)  # put the new index in the map
                    idx_counter += 1  # increment counter

            if len(self.design_vars_population) != self.N_population:
                raise ValueError(
                    f"Population size mismatch: expected {self.N_population}, got {len(self.design_vars_population)}"
                )

            self.design_vars_population = np.array(self.design_vars_population)
            self.objs_population = np.array(self.objs_population)
            if self.N_constr:
                self.constrs_population = np.array(self.constrs_population)

        if self.comm_mpi:
            # broadcast the new populations to all ranks
            self.design_vars_population = self.comm_mpi.bcast(self.design_vars_population, root=0)
            self.objs_population = self.comm_mpi.bcast(self.objs_population, root=0)
            if self.N_constr:
                self.constrs_population = self.comm_mpi.bcast(self.constrs_population, root=0)
            self.idx_fronts = self.comm_mpi.bcast(self.idx_fronts, root=0)
