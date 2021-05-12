# --------------------------------------------------------------------------------------------
#                                  MoorPy
#
#       A mooring system visualizer and quasi-static modeler in Python.
#                         Matt Hall and Stein Housner
#
# --------------------------------------------------------------------------------------------
# 2018-08-14: playing around with making a QS shared-mooring simulation tool, to replace what's in Patrick's work
# 2020-06-17: Trying to create a new quasi-static mooring system solver based on my Catenary function adapted from FAST v7, and using MoorDyn architecture


import numpy as np
import wisdem.moorpy as mp

ms = mp.System("lines.txt")
ms.initialize(plots=1)
# ms.plot()


"""
# catenary testing
mp.catenary(576.2346666666667, 514.6666666666666, 800, 4809884.623076923, -2.6132152062554828, CB=-64.33333333333337, HF0=0, VF0=0, Tol=1e-05, MaxIter=50, plots=2)
print("\nTEST 2")
mp.catenary(88.91360441490338, 44.99537159734132, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=912082.6820817506, VF0=603513.100376363, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 3")
mp.catenary(99.81149090002897, 0.8459770263789324, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=323638.97834178555, VF0=30602.023233123222, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 4")
mp.catenary(99.81520776134033, 0.872357398602503, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=355255.0943810993, VF0=32555.18285808794, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 5")
mp.catenary(99.81149195956499, 0.8459747131565791, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=323645.55876751675, VF0=30602.27072107738, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 6")
mp.catenary(88.91360650151807, 44.99537139684605, 100.0, 854000000.0000001, 1707.0544275185273, CB=0.0, HF0=912082.6820817146, VF0=603513.100376342, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 7")
mp.catenary(9.516786788834565, 2.601777402222183, 10.0, 213500000.00000003, 426.86336920488003, CB=0.0, HF0=1218627.2292202935, VF0=328435.58512892434, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 8")
mp.catenary(9.897879983411258, 0.3124565409495972, 10.0, 213500000.00000003, 426.86336920488003, CB=0.0, HF0=2191904.191415531, VF0=69957.98566771008, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 9")
mp.catenary(107.77260514238083, 7.381234307499085, 112.08021179445676, 6784339692.139625, 13559.120871401587, CB=0.0, HF0=672316.8532881762, VF0=-552499.1313868811, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 9.1")
mp.catenary(107.67265157943795, 7.381234307499085, 112.08021179445676, 6784339692.139625, 13559.120871401587, CB=0.0, HF0=3752787.759641461, VF0=-1678302.5929179655, Tol=1e-06, MaxIter=50, plots=1)
print("\nTEST 9.2")
mp.catenary(107.77260514238083, 7.381234307499085, 112.08021179445676, 6784339692.139625, 13559.120871401587, CB=0.0, Tol=1e-06, MaxIter=50, plots=2, HF0= 1.35e+05,VF0= 1.13e+02)
print("\nTEST 9.3")
mp.catenary(98.6712173965359, 8.515909042185399, 102.7903150736787, 5737939634.533289, 11467.878219531065, CB=0.0, HF0=118208621.36075467, VF0=-12806834.457078349, Tol=1e-07, MaxIter=50, plots=2)
"""


"""
test = mp.System()

test.depth = 100

# Create the LineType of the line for the system
test.addLineType("main", 0.1, 100.0, 1e8)

# add points and lines
test.addPoint(1, [   0, 0,    0])
test.addPoint(0, [ 100, 0,  -50], DOFs=[2])

test.addLine(120, "main")

# attach
test.pointList[0].attachLine(1, 1)
test.pointList[1].attachLine(1, 0)

test.initialize(plots=1)

test.solveEquilibrium3()

test.plot()
"""


ms = mp.System()

# --- diagonal scenario ---
"""
ms.depth = 600
r6 = np.zeros(6)
#r6 = np.array([1,1,0,1,1,1])
ms.addBody(0, r6)

r = np.zeros(3)
#r = np.array([-2,1,3])
ms.addPoint(1, r)
ms.bodyList[0].attachPoint(len(ms.pointList),r-r6[:3])
#ms.addPoint(1, np.array([-10,0,0]))
#ms.bodyList[0].attachPoint(len(ms.pointList),np.array([-10,0,0]))

ms.addPoint(1, np.array([-1000,-1000,-600]))

ms.addLineType('main', 0.10, 40, 1e9)

ms.addLine(1600, 'main')

ms.pointList[0].attachLine(1, 1)
ms.pointList[1].attachLine(1, 0)


"""

# --- orthogonal scenario ---
ms.depth = 100
ms.addBody(0, np.zeros(6))

ms.addPoint(1, np.zeros(3))  # fairlead point
# ms.bodyList[0].attachPoint(len(ms.pointList), [0, 10, 0])  # translations good but some rotations are different
# ms.bodyList[0].attachPoint(len(ms.pointList), [10, 0, 0])   # looks good except 6,4 and 6,6 terms (just first-order vs. nonlinear?)
ms.bodyList[0].attachPoint(
    len(ms.pointList), [0, 0, 10]
)  # looks good except 4,4 and 4,6 terms (just first-order vs. nonlinear?) but 4,4 has sign flipped!

ms.addPoint(1, [700, -380, -10])  # anchor point

ms.addLineType("main", 0.10, 10, 1e10)

ms.addLine(815, "main")

ms.pointList[0].attachLine(1, 1)
ms.pointList[1].attachLine(1, 0)

# --- end of scenario choices ---


ms.initialize()
ms.plot()

a = np.array([0.1, 0, 0, 0, 0, 0])


Kbody = ms.bodyList[0].getStiffness()
Kpoint = ms.pointList[0].getStiffness()
KlineA = ms.lineList[0].getStiffnessMatrix()
KpointA = ms.pointList[0].getStiffnessA()
KbodyA = ms.bodyList[0].getStiffnessA()

# Ksystem = ms.getSystemStiffness()
KsystemA = ms.getSystemStiffnessA()

ms.display = 3
Ksystem = ms.getSystemStiffness(dth=0.05)

print(ms.pointList[0].r)
print("line stiffness A")
mp.printMat(KlineA)
print("body stiffness A")
mp.printMat(KbodyA)
print("system stiffness A")
mp.printMat(KsystemA)
print("system stiffness nonlinear")
mp.printMat(Ksystem)
