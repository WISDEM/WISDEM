from wisdem.floating.floating_turbine_instance import FloatingTurbineInstance, NSECTIONS, NPTS, vecOption
import numpy as np


class TurbineSemiInstance(FloatingTurbineInstance):
    def __init__(self):
        super(TurbineSemiInstance, self).__init__()


    
if __name__ == '__main__':
    mysemi=optimize_semi('psqp')
    #example()
