import os
import sys

try:
    from mpi4py import MPI
except:
    MPI = False


def under_mpirun():
    """Return True if we're being executed under mpirun."""
    # this is a bit of a hack, but there appears to be
    # no consistent set of environment vars between MPI
    # implementations.
    for name in os.environ.keys():
        if (
            name == "OMPI_COMM_WORLD_RANK"
            or name == "MPIEXEC_HOSTNAME"
            or name.startswith("MPIR_")
            or name.startswith("MPICH_")
        ):
            return True
    return False


if under_mpirun():
    from mpi4py import MPI

    def debug(*msg):  # pragma: no cover
        newmsg = ["%d: " % MPI.COMM_WORLD.rank] + list(msg)
        for m in newmsg:
            sys.stdout.write("%s " % m)
        sys.stdout.write("\n")
        sys.stdout.flush()


else:
    MPI = None


def map_comm_heirarchical(K, K2):
    """
    Heirarchical parallelization communicator mapping.  Assumes K top level processes with K2 subprocessors each.
    Requires comm_world_size >= K + K*K2.  Noninclusive, Ki not included in K2i execution.
    (TODO, this is not the most efficient architecture, could be achieve with K fewer processors, but this was easier to generalize)
    """
    N = K + K * K2
    comm_map_down = {}
    comm_map_up = {}
    color_map = [0] * K

    for i in range(K):
        comm_map_down[i] = [K + j + i * K2 for j in range(K2)]
        color_map.extend([i + 1] * K2)

        for j in comm_map_down[i]:
            comm_map_up[j] = i

    return comm_map_down, comm_map_up, color_map


def subprocessor_loop(comm_map_up):
    """
    Subprocessors loop, waiting to receive a function and its arguements to evaluate.
    Output of the function is returned.  Loops until a stop signal is received

    Input data format:
    data[0] = function to be evaluated
    data[1] = [list of arguments]
    If the function to be evaluated does not fit this format, then a wrapper function
    should be created and passed, that handles the setup, argument assignment, etc
    for the actual function.

    Stop sigal:
    data[0] = False
    """
    # comm        = impl.world_comm()
    rank = MPI.COMM_WORLD.Get_rank()
    rank_target = comm_map_up[rank]

    keep_running = True
    while keep_running == True:
        data = MPI.COMM_WORLD.recv(source=(rank_target), tag=0)
        if data[0] == False:
            break
        else:
            func_execution = data[0]
            args = data[1]
            output = func_execution(args)
            MPI.COMM_WORLD.send(output, dest=(rank_target), tag=1)


def subprocessor_stop(comm_map_down):
    """
    Send stop signal to subprocessors
    """
    # comm = MPI.COMM_WORLD
    for rank in comm_map_down.keys():
        subranks = comm_map_down[rank]
        for subrank_i in subranks:
            MPI.COMM_WORLD.send([False], dest=subrank_i, tag=0)
        print("All MPI subranks closed.")


if __name__ == "__main__":
    from mpi4py import MPI

    (
        _,
        _,
        _,
    ) = map_comm_heirarchical(2, 4)
