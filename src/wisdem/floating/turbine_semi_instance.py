from floatingse.floatingInstance import NSECTIONS, NPTS, vecOption
from floatingse.semiInstance import SemiInstance
from floating_turbine_assembly import FloatingTurbine
from commonse import eps
from rotorse import TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE, r_aero
import numpy as np
import offshorebos.wind_obos as wind_obos
import time

NDEL = 0

class TurbineSemiInstance(SemiInstance):
    def __init__(self):
        super(TurbineSemiInstance, self).__init__()

    def get_constraints(self):
        conList = super(TurbineSemiInstance, self).get_constraints()
        for con in conList:
            con[0] = 'sm.' + con[0]

        conList.extend( [['rotor.Pn_margin', None, 1.0, None],
                         ['rotor.P1_margin', None, 1.0, None],
                         ['rotor.Pn_margin_cfem', None, 1.0, None],
                         ['rotor.P1_margin_cfem', None, 1.0, None],
                         ['rotor.rotor_strain_sparU', -1.0, None, None],
                         ['rotor.rotor_strain_sparL', None, 1.0, None],
                         ['rotor.rotor_strain_teU', -1.0, None, None],
                         ['rotor.rotor_strain_teL', None, 1.0, None],
                         ['rotor.rotor_buckling_sparU', None, 1.0, None],
                         ['rotor.rotor_buckling_sparL', None, 1.0, None],
                         ['rotor.rotor_buckling_teU', None, 1.0, None],
                         ['rotor.rotor_buckling_teL', None, 1.0, None],
                         ['rotor.rotor_damage_sparU', None, 0.0, None],
                         ['rotor.rotor_damage_sparL', None, 0.0, None],
                         ['rotor.rotor_damage_teU', None, 0.0, None],
                         ['rotor.rotor_damage_teL', None, 0.0, None],
                         ['tcons.frequency_ratio', None, 1.0, None],
                         ['tcons.tip_deflection_ratio', None, 1.0, None],
                         ['tcons.ground_clearance', 30.0, None, None],
        ])
        return conList


    
if __name__ == '__main__':
    mysemi=optimize_semi('psqp')
    #example()
