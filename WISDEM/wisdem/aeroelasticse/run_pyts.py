import sys
from pyts.runInput.main import readInput, run, write
import time

import pyts
import pyts.runInput.main as ptsm
import pyts.io.input as ptsin
from pyts.base import tsGrid
from pyts.phaseModels.main import Rinker, Uniform
import numpy as np

fname = sys.argv[1]
outname = sys.argv[2]



def run1():
    config = readInput(fname)
    tm0 = time.time()
    tsdat = run(config)
    write(tsdat, config, outname)
    print('TurbSim exited normally, runtime was %g seconds' % (time.time() - tm0))


def run2():
    tsinput = ptsin.read(fname)
    tsinput['URef'] = 16.0
    tsr = ptsm.cfg2tsrun(tsinput)

    tsr.cohere = pyts.cohereModels.main.nwtc()
    tsr.stress = pyts.stressModels.main.uniform(0,0,0)
    rho = 0.5
    tmax = 50.0
    tsr.phase = Rinker(rho, np.pi)  ### pgraf turned it off for testing!
    cg = tsr.grid
    tsr.grid = tsGrid(center=cg.center, ny=cg.n_y, nz=cg.n_z,
                      height=cg.height, width=cg.width,
                      time_sec=tmax, dt=cg.dt)
    tm0 = time.time()
    tsdata = tsr()  ## actually runs turbsim
    ptsm.write(tsdata, tsinput, fname=outname+"2")
    print('TurbSim exited normally, runtime was %g seconds' % (time.time() - tm0))


run1()

run2()
