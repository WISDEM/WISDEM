import os
import sys

def under_mpirun():
    """Return True if we're being executed under mpirun."""
    # this is a bit of a hack, but there appears to be
    # no consistent set of environment vars between MPI
    # implementations.
    for name in os.environ.keys():
        if name == 'OMPI_COMM_WORLD_RANK' or \
           name == 'MPIEXEC_HOSTNAME' or \
           name.startswith('MPIR_') or \
           name.startswith('MPICH_'):
            return True
    return False

if under_mpirun():
    from mpi4py import MPI

    def debug(*msg):  # pragma: no cover
        newmsg = ["%d: " % MPI.COMM_WORLD.rank] + list(msg)
        for m in newmsg:
            sys.stdout.write("%s " % m)
        sys.stdout.write('\n')
        sys.stdout.flush()
else:
    MPI = None