import numpy as np

try:
    import numba

    compile_numba = True
except ImportError:
    compile_numba = False


def fast_nondom_sort_ranks_python(P):
    """
    Perform fast non-dominated sorting on population P.

    Args:
      P (list or np.ndarray): Population, each element is a list/array of objectives.

    Returns:
        ranks (list): List of front ranks for each solution in P.
    """
    N = len(P)
    M = len(P[0])
    # Use lists of lists instead of sets for numba compatibility
    S = [numba.typed.List.empty_list(numba.types.int64) for _ in range(N)] if compile_numba else [[] for _ in range(N)]
    n = np.zeros(N, dtype=np.int64)
    ranks = -1 * np.ones(N, dtype=np.int64)
    fronts = [numba.typed.List.empty_list(numba.types.int64)] if compile_numba else [[]]
    fronts[0].clear()

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            p_better = False
            q_better = False
            for m in range(M):
                if P[p][m] < P[q][m]:
                    p_better = True
                elif P[q][m] < P[p][m]:
                    q_better = True
            if (not q_better) and p_better:
                S[p].append(q)
            elif (not p_better) and q_better:
                n[p] += 1
        if n[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = numba.typed.List.empty_list(numba.types.int64) if compile_numba else []
        for idx in range(len(fronts[i])):
            p = fronts[i][idx]
            for j in range(len(S[p])):
                q = S[p][j]
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        fronts.append(next_front)
        i += 1

    # Convert -1 to None for compatibility with original code
    result = [None if ranks[i] == -1 else int(ranks[i]) for i in range(N)]
    return result


if compile_numba:
    _fast_nondom_sort_ranks = numba.njit(fast_nondom_sort_ranks_python)
    def fast_nondom_sort_ranks(P):
        P = numba.typed.List(P)  # type protection to avoid deprecation
        return _fast_nondom_sort_ranks(P)
    fast_nondom_sort_ranks.is_numba = True
    fast_nondom_sort_ranks.function_nojit = fast_nondom_sort_ranks_python
else:
    fast_nondom_sort_ranks = fast_nondom_sort_ranks_python
    fast_nondom_sort_ranks.is_numba = False
    fast_nondom_sort_ranks.function_nojit = fast_nondom_sort_ranks_python


def _fast_nondom_sort(P, compile_numba_local=False):
    """
    Wrapper for fast_nondom_sort_ranks that returns the list-of-lists of indices for each front.

    Args:
      P (list or np.ndarray): Population, each element is a list/array of objectives.

    Returns:
        fronts (list of lists): Each sublist contains indices of solutions in that front.
    """
    fnsr = fast_nondom_sort_ranks if compile_numba_local else fast_nondom_sort_ranks.function_nojit
    ranks = fnsr(P)
    max_rank = max(r for r in ranks if r is not None)
    fronts = [[] for _ in range(max_rank + 1)]
    for idx, rank in enumerate(ranks):
        if rank is not None:
            fronts[rank].append(idx)
    return fronts


fast_nondom_sort = lambda P: _fast_nondom_sort(P, compile_numba_local=compile_numba)
fast_nondom_sort.is_numba = compile_numba
fast_nondom_sort.function_nojit = lambda P: _fast_nondom_sort(P, compile_numba_local=False)
