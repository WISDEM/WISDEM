import numpy as np

try:
    import numba

    compile_numba = True
except ImportError:
    compile_numba = False
compile_numba = False  # DEBUG!!!!!


def binary_tournament_selection_python(
    fitness,
    ratio_keep=0.5,
    rng_seed: int = None,  # rng seed
):
    rng = np.random  # default to numpy random
    if rng_seed is not None:
        rng.seed(rng_seed)

    indices_selected = []

    for i in range(int(len(fitness) // (1 / ratio_keep))):
        # select two random individuals
        idx1 = rng.randint(0, len(fitness))
        idx2 = rng.randint(0, len(fitness))

        # compare their fitness
        if fitness[idx1] < fitness[idx2]:
            indices_selected.append(idx1)
        else:
            indices_selected.append(idx2)

    return indices_selected


if compile_numba:
    binary_tournament_selection = numba.njit(binary_tournament_selection_python)
    binary_tournament_selection.is_numba = True
    binary_tournament_selection.function_nojit = binary_tournament_selection_python
else:
    binary_tournament_selection = binary_tournament_selection_python
    binary_tournament_selection.is_numba = False
    binary_tournament_selection.function_nojit = binary_tournament_selection_python


def unit_simulated_binary_crossover_python(
    design_vars_1: np.array,
    design_vars_2: np.array,
    design_vars_l: np.array,
    design_vars_u: np.array,
    rate_crossover: float = 0.9,  # crossover probability
    eta_c: float = 20,  # distribution index
    rng_seed: int = None,  # rng seed
):
    rng = np.random  # default to numpy random
    if rng_seed is not None:
        rng.seed(rng_seed)

    is_changed = False

    # preliminaries
    N_DV = len(design_vars_1)
    assert N_DV == len(design_vars_2)  # should be the same number of entries in both

    # should we crossover? if not, return parents
    if rng.rand() >= rate_crossover:
        return design_vars_1.copy(), design_vars_2.copy(), is_changed

    # if we get this far... apply the crossover

    design_vars_a = np.zeros_like(design_vars_1)  # create the potential children
    design_vars_b = np.zeros_like(design_vars_2)  # create the potential children

    for i_DV in range(N_DV):  # loop over the design variables

        if rng.rand() > 0.5:  # coinflip to see if this DV gets touched

            if np.isclose(design_vars_1[i_DV], design_vars_2[i_DV]):
                # when they're effectively the same, don't overcomplicate it
                design_vars_a[i_DV] = design_vars_1[i_DV]
                design_vars_b[i_DV] = design_vars_2[i_DV]
                continue
            else:
                # child A gets greater of 1 and 2, vice versa
                V1 = design_vars_1[i_DV] if design_vars_1[i_DV] <= design_vars_2[i_DV] else design_vars_2[i_DV]
                V2 = design_vars_1[i_DV] if design_vars_1[i_DV] > design_vars_2[i_DV] else design_vars_2[i_DV]

            rv = rng.rand()  # a random uniform real

            # do the calculations of the child candidate 1
            beta = 1.0 + 2.0 * (V1 - design_vars_l[i_DV]) / (V2 - V1)
            alpha = 2.0 - np.power(beta, -(eta_c + 1.0))
            beta_q = (
                np.power(rv * alpha, 1.0 / (eta_c + 1.0))
                if rng.rand() <= 1 / alpha
                else np.power(1.0 / (2.0 - rv * alpha), 1.0 / (eta_c + 1.0))
            )
            c1 = 0.5 * (V1 + V2 - beta_q * (V2 - V1))

            # do the calculations of the child candidate 2
            beta = 1.0 + 2.0 * (design_vars_u[i_DV] - V2) / (V2 - V1)
            alpha = 2.0 - np.power(beta, -(eta_c + 1.0))
            beta_q = (
                np.power(rv * alpha, 1.0 / (eta_c + 1.0))
                if rng.rand() < 1.0 / alpha
                else np.power(1.0 / (2.0 - rv * alpha), 1.0 / (eta_c + 1.0))
            )
            c2 = 0.5 * (V1 + V2 + beta_q * (V2 - V1))

            # check the limits
            if c1 < design_vars_l[i_DV]:
                c1 = design_vars_l[i_DV]
            if c2 < design_vars_l[i_DV]:
                c2 = design_vars_l[i_DV]
            if c1 > design_vars_u[i_DV]:
                c1 = design_vars_u[i_DV]
            if c2 > design_vars_u[i_DV]:
                c2 = design_vars_u[i_DV]

            is_changed = True

            # coinflip to determine who gets which kid... the only fair way
            if rng.rand() > 0.5:
                design_vars_a[i_DV] = c2
                design_vars_b[i_DV] = c1
            else:
                design_vars_a[i_DV] = c1
                design_vars_b[i_DV] = c2

        else:
            design_vars_a[i_DV] = design_vars_1[i_DV]
            design_vars_b[i_DV] = design_vars_2[i_DV]

    return design_vars_a, design_vars_b, is_changed


if compile_numba:
    unit_simulated_binary_crossover = numba.njit(unit_simulated_binary_crossover_python)
    unit_simulated_binary_crossover.is_numba = True
    unit_simulated_binary_crossover.function_nojit = unit_simulated_binary_crossover_python
else:
    unit_simulated_binary_crossover = unit_simulated_binary_crossover_python
    unit_simulated_binary_crossover.is_numba = False
    unit_simulated_binary_crossover.function_nojit = unit_simulated_binary_crossover_python


def simulated_binary_crossover_python(
    P_in: np.array,  # population to cross over
    design_vars_l: np.array,  # DV lower limits
    design_vars_u: np.array,  # DV upper limits
    rate_crossover: float = 0.9,  # crossover probability
    eta_c: float = 20,  # distribution index
    rng_seed: int = None,  # rng seed
):
    rng = np.random  # default to numpy random
    if rng_seed is not None:
        rng.seed(rng_seed)

    N = len(P_in)
    N_pairs = N // 2
    indices = np.empty((N_pairs, 2), dtype=np.int64)
    for i in numba.prange(N_pairs) if compile_numba else range(N_pairs):
        indices[i, 0] = rng.randint(0, N)
        indices[i, 1] = rng.randint(0, N)

    N_DV = P_in.shape[1]
    Q_out = np.empty((N_pairs * 2, N_DV), dtype=P_in.dtype)
    changed_out = np.empty(
        N_pairs * 2,
        dtype=numba.boolean if compile_numba else bool,
    )

    for i in numba.prange(N_pairs) if compile_numba else range(N_pairs):
        idx0 = indices[i, 0]
        idx1 = indices[i, 1]
        c0, c1, is_changed = unit_simulated_binary_crossover(
            P_in[idx0],
            P_in[idx1],
            design_vars_l=design_vars_l,
            design_vars_u=design_vars_u,
            rate_crossover=rate_crossover,
            eta_c=eta_c,
            rng_seed=None if rng_seed is None else rng_seed + i,
        )
        Q_out[2 * i] = c0
        Q_out[2 * i + 1] = c1
        changed_out[2 * i] = is_changed
        changed_out[2 * i + 1] = is_changed

    return Q_out, changed_out


if compile_numba:
    simulated_binary_crossover = numba.njit(
        simulated_binary_crossover_python,
        parallel=True,
    )
    simulated_binary_crossover.is_numba = True
    simulated_binary_crossover.function_nojit = simulated_binary_crossover_python
else:
    simulated_binary_crossover = simulated_binary_crossover_python
    simulated_binary_crossover.is_numba = False
    simulated_binary_crossover.function_nojit = simulated_binary_crossover_python


def unit_polynomial_mutation_python(
    design_vars_1: np.array,
    design_vars_l: np.array,
    design_vars_u: np.array,
    rate_mutation: float = 0.1,  # mutation probability
    eta_m: float = 20,  # distribution index
    rng_seed: int = None,  # rng seed
):
    rng = np.random  # default to numpy random
    if rng_seed is not None:
        rng.seed(rng_seed)

    is_changed = False

    # preliminaries
    N_DV = len(design_vars_1)

    design_vars_a = design_vars_1.copy()  # create the potential children

    for i_DV in range(N_DV):  # loop over the design variables

        # determine if mutation should occur
        if rng.rand() >= rate_mutation:
            continue  # to next DV index

        is_changed = True

        # get values on this DV
        Y = design_vars_1[i_DV]
        YL = design_vars_l[i_DV]
        YU = design_vars_u[i_DV]
        # handle infinite bounds safely
        if np.isinf(-YL) and np.isinf(YU):
            DELTA1 = 0.5
            DELTA2 = 0.5
        elif np.isinf(-YL):
            DELTA1 = 0.5
            DELTA2 = 0.0
        elif np.isinf(YU):
            DELTA1 = 0.0
            DELTA2 = 0.5
        else:
            # relative coordinates
            DELTA1 = (Y - YL) / (YU - YL)
            DELTA2 = (YU - Y) / (YU - YL)

        RAND = rng.rand()  # unit random
        MUT_POW = 1.0 / (eta_m + 1.0)  # utility variable

        # relative mutation value
        DELTA_Q = (
            (
                np.power(
                    2.0 * RAND + (1.0 - 2.0 * RAND) * np.power(1.0 - DELTA1, eta_m + 1.0),
                    MUT_POW,
                )
                - 1.0
            )
            if rng.rand() <= 0.5  # if coinflip is heads
            else (
                1.0
                - np.power(
                    2.0 * (1.0 - RAND) + 2.0 * (RAND - 0.5) * np.power(1.0 - DELTA2, eta_m + 1.0),
                    MUT_POW,
                )
            )
        )

        # apply mutation
        Y = Y + DELTA_Q * (YU - YL)

        # set into the outgoing variable
        design_vars_a[i_DV] = YL if Y < YL else YU if Y > YU else Y

    return design_vars_a, is_changed


if compile_numba:
    unit_polynomial_mutation = numba.njit(unit_polynomial_mutation_python)
    unit_polynomial_mutation.is_numba = True
    unit_polynomial_mutation.function_nojit = unit_polynomial_mutation_python
else:
    unit_polynomial_mutation = unit_polynomial_mutation_python
    unit_polynomial_mutation.is_numba = False
    unit_polynomial_mutation.function_nojit = unit_polynomial_mutation_python


def polynomial_mutation_python(
    P_in: np.array,  # population to mutate
    design_vars_l: np.array,  # DV lower limits
    design_vars_u: np.array,  # DV upper limits
    rate_mutation: float = 0.1,  # mutation probability
    eta_m: float = 20,  # distribution index
    rng_seed: int = None,  # rng seed
):
    rng = np.random  # default to numpy random
    if rng_seed is not None:
        rng.seed(rng_seed)

    N = P_in.shape[0]
    N_DV = P_in.shape[1]
    Q_out = np.empty_like(P_in)
    changed_out = np.empty(N, dtype=numba.boolean if compile_numba else bool)
    for i in numba.prange(N) if compile_numba else range(N):
        xPi, is_changed = unit_polynomial_mutation(
            P_in[i],
            design_vars_l=design_vars_l,
            design_vars_u=design_vars_u,
            rate_mutation=rate_mutation,
            eta_m=eta_m,
            rng_seed=None if rng_seed is None else rng_seed + i,
        )
        Q_out[i] = xPi
        changed_out[i] = is_changed
    return Q_out, changed_out


if compile_numba:
    polynomial_mutation = numba.njit(polynomial_mutation_python, parallel=True)
    polynomial_mutation.is_numba = True
    polynomial_mutation.function_nojit = polynomial_mutation_python
else:
    polynomial_mutation = polynomial_mutation_python
    polynomial_mutation.is_numba = False
    polynomial_mutation.function_nojit = polynomial_mutation_python
