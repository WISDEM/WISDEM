import numpy as np

try:
    import numba

    compile_numba = True
except ImportError:
    compile_numba = False


def crowding_distance_assignment_python(I):

    ### algorithm 3 from Deb et al. (2002)

    l = len(I)  # number of solutions
    N_obj = len(I[0])  # number of objectives

    d = np.zeros(l)  # crowding distance of each solution

    if np.any(np.isinf(I)):
        raise Exception("there's problem! infinite objective function in I:", I)

    for m in range(N_obj):
        idx_m = np.argsort(I[:, m])[::-1]
        # I = sorted(I, key=lambda x: x[m])  # sort the solutions to obj. m
        d[idx_m[0]] += np.inf  # set first and ...
        d[idx_m[-1]] += np.inf  # ... last solution to infinity
        for i in range(1, l - 1):
            if np.isclose(I[idx_m[l - 1]][m], I[idx_m[0]][m]):
                d[idx_m[i]] = 0.0
            else:
                d[idx_m[i]] += (I[idx_m[i + 1]][m] - I[idx_m[i - 1]][m]) / (
                    I[idx_m[l - 1]][m] - I[idx_m[0]][m]
                )  # compute the crowding distance

    return d  # return the crowding distance of each solution


if compile_numba:
    crowding_distance_assignment = numba.njit(crowding_distance_assignment_python)
    crowding_distance_assignment.is_numba = True
    crowding_distance_assignment.function_nojit = crowding_distance_assignment_python
else:
    crowding_distance_assignment = crowding_distance_assignment_python
    crowding_distance_assignment.is_numba = False
    crowding_distance_assignment.function_nojit = crowding_distance_assignment_python
