import os
import sys

try:
    from mpi4py import MPI
except ImportError:
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
            or name.startswith("INTEL_ONEAPI_MPI_")
            or name.startswith("I_MPI_")
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


def map_comm_heirarchical(n_DV, n_OF, openmp=False):
    """
    Heirarchical parallelization communicator mapping.  Assumes a number of top level processes
    equal to the number of design variables (x2 if central finite differencing is used), each
    with its associated number of openfast simulations.
    When openmp flag is turned on, the code spreads the openfast simulations across nodes to
    lavereage the opnemp parallelization of OpenFAST. The cores that will run under openmp, are marked
    in the color map as 1000000. The ones handling python and the DV are marked as 0, and
    finally the master ones for each openfast run are marked with a 1.
    """
    if openmp:
        n_procs_per_node = 36  # Number of
        num_procs = MPI.COMM_WORLD.Get_size()
        n_nodes = num_procs / n_procs_per_node

        comm_map_down = {}
        comm_map_up = {}
        color_map = [1000000] * num_procs

        n_DV_per_node = n_DV / n_nodes

        # for m in range(n_DV_per_node):
        for nn in range(int(n_nodes)):
            for n_dv in range(int(n_DV_per_node)):
                comm_map_down[nn * n_procs_per_node + n_dv] = [
                    int(n_DV_per_node) + n_dv * n_OF + nn * (n_procs_per_node) + j for j in range(n_OF)
                ]

                # This core handles python, so in the colormap the entry is 0
                color_map[nn * n_procs_per_node + n_dv] = int(0)
                # These cores handles openfast, so in the colormap the entry is 1
                for k in comm_map_down[nn * n_procs_per_node + n_dv]:
                    color_map[k] = int(1)

                for j in comm_map_down[nn * n_procs_per_node + n_dv]:
                    comm_map_up[j] = nn * n_procs_per_node + n_dv
    else:
        N = n_DV + n_DV * n_OF
        comm_map_down = {}
        comm_map_up = {}
        color_map = [0] * n_DV

        for i in range(n_DV):
            comm_map_down[i] = [n_DV + j + i * n_OF for j in range(n_OF)]
            color_map.extend([i + 1] * n_OF)

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
    while keep_running:
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
