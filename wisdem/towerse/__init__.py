from wisdem.commonse import NFREQ, eps, gravity

RIGID = 1e30
NREFINE = 3
NPTS_SOIL = 10


def get_nfull(npts, nref=NREFINE):
    n_full = int(1 + nref * (npts - 1))
    return n_full


def get_npts(nFull, nref=NREFINE):
    npts = int(1 + (nFull - 1) / nref)
    return npts
