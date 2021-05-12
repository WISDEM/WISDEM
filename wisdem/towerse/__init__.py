from wisdem.commonse import NFREQ, eps, gravity

RIGID = 1e30
NREFINE = 3
NPTS_SOIL = 10


def get_nfull(npts):
    nFull = int(1 + NREFINE * (npts - 1))
    return nFull


def get_npts(nFull):
    npts = int(1 + (nFull - 1) / NREFINE)
    return npts
