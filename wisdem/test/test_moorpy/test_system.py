# tests MoorPy System functionality and results

import numpy as np
import pytest
import wisdem.moorpy as mp
from numpy.testing import assert_allclose
from wisdem.moorpy.MoorProps import getLineProps

# import matplotlib.pyplot as plt


inCBs = [0, 1.0, 10.0]  # friction coefficients as inputs for test_seabed


def test_tensions_swap():
    """Compares two equivalent catenary mooring lines that are defined in opposite directions."""

    ms = mp.System(depth=60)

    ms.lineTypes["chain"] = getLineProps(120, name="chain")  # add a line type

    ms.addPoint(1, [0, 0, -60])
    ms.addPoint(1, [100, 10, -30])

    # line sloping up from A to B, and another in the opposite order
    ms.addLine(120, "chain", pointA=1, pointB=2)
    ms.addLine(120, "chain", pointA=2, pointB=1)

    ms.initialize()

    # compare tensions
    assert_allclose(
        np.hstack([ms.lineList[0].fA, ms.lineList[0].fB]),
        np.hstack([ms.lineList[1].fB, ms.lineList[1].fA]),
        rtol=0,
        atol=10.0,
        verbose=True,
    )


def test_stiffnesses_swap():
    """Compares two equivalent catenary mooring lines that are defined in opposite directions."""

    ms = mp.System(depth=60)

    ms.lineTypes["chain"] = getLineProps(120, name="chain")  # add a line type

    ms.addPoint(1, [0, 0, -60])
    ms.addPoint(1, [100, 10, -30])

    # line sloping up from A to B, and another in the opposite order
    ms.addLine(120, "chain", pointA=1, pointB=2)
    ms.addLine(120, "chain", pointA=2, pointB=1)

    ms.initialize()

    # compare stiffnesses
    assert_allclose(
        np.hstack([ms.lineList[0].KA, ms.lineList[0].KB, ms.lineList[0].KAB]),
        np.hstack([ms.lineList[1].KB, ms.lineList[1].KA, ms.lineList[1].KAB]),
        rtol=0,
        atol=10.0,
        verbose=True,
    )


def test_stiffness_body():
    """Tests that the mooring stiffness on a body has the expected relationship to stiffness on points"""

    ms = mp.System(depth=60)

    ms.lineTypes["chain"] = getLineProps(120, name="chain")  # add a line type

    ms.addPoint(1, [0, 0, -60])
    ms.addPoint(1, [100, 10, -30])

    # line sloping up from A to B, and another in the opposite order
    ms.addLine(120, "chain", pointA=1, pointB=2)
    ms.addLine(120, "chain", pointA=2, pointB=1)

    # create body and attach lines to it
    ms.addBody(1, [0, 0, 0, 0, 0, 0])
    ms.bodyList[0].attachPoint(2, [0, 10, 0])

    ms.initialize()

    # compare stiffnesses
    assert_allclose(
        np.hstack([ms.lineList[0].KA, ms.lineList[0].KB, ms.lineList[0].KAB]),
        np.hstack([ms.lineList[1].KB, ms.lineList[1].KA, ms.lineList[1].KAB]),
        rtol=0,
        atol=10.0,
        verbose=True,
    )


def test_basic():

    depth = 600
    angle = np.arange(3) * np.pi * 2 / 3  # line headings list
    anchorR = 1600  # anchor radius/spacing
    fair_depth = 21
    fairR = 20
    LineLength = 1800
    typeName = "chain"  # identifier string for line type

    # --------------- set up mooring system ---------------------

    # Create blank system object
    ms = mp.System()

    # Set the depth of the system to the depth of the input value
    ms.depth = depth

    # add a line type
    ms.lineTypes[typeName] = getLineProps(120, name=typeName)

    # Add a free, body at [0,0,0] to the system (including some properties to make it hydrostatically stiff)
    ms.addBody(0, np.zeros(6), m=1e6, v=1e3, rM=100, AWP=1e6)

    # Set the anchor points of the system
    anchors = []
    for i in range(len(angle)):
        ms.addPoint(1, np.array([anchorR * np.cos(angle[i]), anchorR * np.sin(angle[i]), -ms.depth], dtype=float))
        anchors.append(len(ms.pointList))

    # Set the points that are attached to the body to the system
    bodypts = []
    for i in range(len(angle)):
        ms.addPoint(1, np.array([fairR * np.cos(angle[i]), fairR * np.sin(angle[i]), -fair_depth], dtype=float))
        bodypts.append(len(ms.pointList))
        ms.bodyList[0].attachPoint(
            ms.pointList[bodypts[i] - 1].number, ms.pointList[bodypts[i] - 1].r - ms.bodyList[0].r6[:3]
        )

    # Add and attach lines to go from the anchor points to the body points
    for i in range(len(angle)):
        ms.addLine(LineLength, typeName)
        line = len(ms.lineList)
        ms.pointList[anchors[i] - 1].attachLine(ms.lineList[line - 1].number, 0)
        ms.pointList[bodypts[i] - 1].attachLine(ms.lineList[line - 1].number, 1)

    # ------- simulate it ----------
    ms.initialize()  # make sure everything's connected
    ms.bodyList[0].setPosition([10, 10, 1, 0, 0, 0])  # apply an offset
    ms.solveEquilibrium3()  # equilibrate - see if it goes back to zero!

    # check
    assert_allclose(ms.bodyList[0].r6, np.zeros(6), rtol=0, atol=0.01, verbose=True)


def test_multiseg():
    """Compares a single catenary mooring line with a two-line system."""

    # single line
    ms1 = mp.System()
    ms1.depth = 200
    ms1.lineTypes["chain"] = getLineProps(120, name="chain")
    ms1.addPoint(1, [-800, 0, -200])  # anchor point
    ms1.addPoint(1, [0, 0, 0])  # fairlead
    ms1.addLine(860, "chain", pointA=1, pointB=2)
    ms1.initialize()
    ms1.solveEquilibrium3(tol=0.0001)

    # two line
    ms2 = mp.System()
    ms2.depth = 200
    ms2.lineTypes["chain"] = getLineProps(120, name="chain")
    ms2.addPoint(1, [-800, 0, -200])  # anchor point
    ms2.addPoint(0, [-205, 0, -100])  # point along line
    ms2.addPoint(1, [0, 0, 0])  # fairlead
    ms2.addLine(630, "chain", pointA=1, pointB=2)
    ms2.addLine(230, "chain", pointA=2, pointB=3)
    ms2.initialize()
    ms2.solveEquilibrium3(tol=0.0001)

    # compare tensions
    assert_allclose(
        np.hstack([ms1.pointList[0].getForces(), ms1.pointList[-1].getForces()]),
        np.hstack([ms2.pointList[0].getForces(), ms2.pointList[-1].getForces()]),
        rtol=0,
        atol=10.0,
        verbose=True,
    )


def test_multiseg_seabed():
    """Compares a single catenary mooring line with a three-line system where two of the segments are fully along the seabed."""

    # single line
    ms1 = mp.System()
    ms1.depth = 200
    ms1.lineTypes["chain"] = getLineProps(120, name="chain")
    ms1.addPoint(1, [-800, 0, -200])  # anchor point
    ms1.addPoint(1, [0, 0, 0])  # fairlead
    ms1.addLine(860, "chain", pointA=1, pointB=2)
    ms1.initialize()
    ms1.solveEquilibrium3(tol=0.0001)

    # three line
    ms2 = mp.System()
    ms2.depth = 200
    ms2.lineTypes["chain"] = getLineProps(120, name="chain")
    ms2.addPoint(1, [-800, 0, -200])  # anchor point
    ms2.addPoint(0, [-700, 0, -200])  # point along line
    ms2.addPoint(0, [-600, 0, -200])  # point along line
    ms2.addPoint(1, [0, 0, 0])  # fairlead
    ms2.addLine(100, "chain", pointA=1, pointB=2)
    ms2.addLine(100, "chain", pointA=2, pointB=3)
    ms2.addLine(660, "chain", pointA=3, pointB=4)
    ms2.initialize()
    ms2.solveEquilibrium3(tol=0.0001)

    # compare tensions
    assert_allclose(
        np.hstack([ms1.pointList[0].getForces(), ms1.pointList[-1].getForces()]),
        np.hstack([ms2.pointList[0].getForces(), ms2.pointList[-1].getForces()]),
        rtol=0,
        atol=10.0,
        verbose=True,
    )


def test_basicU():
    """Compares a system with a U-shape line with seabed contact with an equivalent case
    that has a node along the U."""

    # a seabed contact case with 2 lines to form a U
    ms1 = mp.System()

    ms1.depth = 100

    ms1.lineTypes["chain"] = getLineProps(120, name="chain")

    ms1.addPoint(1, [-200, 0, -100])  # anchor point
    ms1.addPoint(0, [-100, 0, -50], m=0, v=50)  # float
    ms1.addPoint(0, [0, 0, -100])  # midpoint
    ms1.addPoint(0, [100, 0, -40], m=0, v=50)  # float
    ms1.addPoint(1, [200, 0, -100])  # anchor point

    ms1.addLine(120, "chain", pointA=1, pointB=2)
    ms1.addLine(125, "chain", pointA=2, pointB=3)
    ms1.addLine(125, "chain", pointA=3, pointB=4)
    ms1.addLine(120, "chain", pointA=4, pointB=5)

    # a seabed contact case with single U line
    msU = mp.System()

    msU.depth = 100

    msU.lineTypes["chain"] = getLineProps(120, name="chain")

    msU.addPoint(1, [-200, 0, -100])  # anchor point
    msU.addPoint(0, [-100, 0, -50], m=0, v=50)  # float
    msU.addPoint(0, [100, 0, -40], m=0, v=50)  # float
    msU.addPoint(1, [200, 0, -100])  # anchor point

    msU.addLine(120, "chain", pointA=1, pointB=2)
    msU.addLine(250, "chain", pointA=2, pointB=3)
    msU.addLine(120, "chain", pointA=3, pointB=4)

    # ------- simulate it ----------
    ms1.initialize()  # make sure everything's connected
    msU.initialize()  # make sure everything's connected

    ms1.solveEquilibrium3(tol=0.0001)  # equilibrate - see if it goes back to zero!
    msU.solveEquilibrium3(tol=0.0001)  # equilibrate - see if it goes back to zero!

    # compare floating point positions
    assert_allclose(
        np.hstack([ms1.pointList[1].r, ms1.pointList[3].r]),
        np.hstack([msU.pointList[1].r, msU.pointList[2].r]),
        rtol=0,
        atol=0.001,
        verbose=True,
    )


@pytest.mark.parametrize("CB", inCBs)
def test_seabed(CB):
    """Compares a single catenary mooring line along the seabed with a two-line system
    where the point is on the seabed, with different friction settings."""

    # single line
    ms1 = mp.System()
    ms1.depth = 200
    ms1.lineTypes["chain"] = getLineProps(120, name="chain")
    ms1.addPoint(1, [-800, 0, -200])  # anchor point
    ms1.addPoint(1, [0, 0, 0])  # fairlead
    ms1.addLine(860, "chain", pointA=1, pointB=2, cb=CB)
    ms1.initialize()
    ms1.solveEquilibrium3(tol=0.0001)

    # two line
    ms2 = mp.System()
    ms2.depth = 200
    ms2.lineTypes["chain"] = getLineProps(120, name="chain")
    ms2.addPoint(1, [-800, 0, -200])  # anchor point
    ms2.addPoint(0, [-405, 0, -150])  # midpoint
    ms2.addPoint(1, [0, 0, 0])  # fairlead
    ms2.addLine(430, "chain", pointA=1, pointB=2, cb=CB)
    ms2.addLine(430, "chain", pointA=2, pointB=3, cb=CB)
    ms2.initialize()
    ms2.solveEquilibrium3(tol=0.0001)

    # compare tensions
    assert_allclose(
        np.hstack([ms1.pointList[0].getForces(), ms1.pointList[-1].getForces()]),
        np.hstack([ms2.pointList[0].getForces(), ms2.pointList[-1].getForces()]),
        rtol=0,
        atol=10.0,
        verbose=True,
    )


if __name__ == "__main__":

    # test_basic()

    """
      # a seabed contact case with 2 lines to form a U
    ms1 = mp.System()

    ms1.depth = 100

    ms1.lineTypes['chain'] = getLineProps(120, name='chain')

    ms1.addPoint(1, [-200, 0 , -100])                 # anchor point
    ms1.addPoint(1, [-100, 0 ,  -50], m=0, v=50)      # float
    ms1.addPoint(0, [   0, 0 , -100])                 # midpoint
    ms1.addPoint(1, [ 100, 0 ,  -40], m=0, v=50)      # float
    ms1.addPoint(1, [ 200, 0 , -100])                 # anchor point

    ms1.addLine(120, 'chain', pointA=1, pointB=2)
    ms1.addLine(125, 'chain', pointA=2, pointB=3)
    ms1.addLine(125, 'chain', pointA=3, pointB=4)
    ms1.addLine(120, 'chain', pointA=4, pointB=5)


    # a seabed contact case with single U line
    ms = mp.System()

    ms.depth = 100

    ms.lineTypes['chain'] = getLineProps(120, name='chain')

    ms.addPoint(1, [-200, 0 , -100])                 # anchor point
    ms.addPoint(0, [-100, 0 ,  -50], m=0, v=50)      # float
    ms.addPoint(0, [ 100, 0 ,  -40], m=0, v=50)      # float
    ms.addPoint(1, [ 200, 0 , -100])                 # anchor point

    ms.addLine(120, 'chain', pointA=1, pointB=2)
    ms.addLine(250, 'chain', pointA=2, pointB=3)
    ms.addLine(120, 'chain', pointA=3, pointB=4)


    # ------- simulate it ----------
    ms1.initialize()                                             # make sure everything's connected
    ms.initialize()                                             # make sure everything's connected

    fig, ax = ms1.plot(color='b')
    ms.plot(ax=ax, color='k')

    #ms.display=2

    ms1.solveEquilibrium3(maxIter=20)                                      # equilibrate - see if it goes back to zero!
    #ms.solveEquilibrium3(maxIter=20)                                      # equilibrate - see if it goes back to zero!

    ms1.plot(ax=ax, color='g')
    #ms.plot(ax=ax, color='r')

    # compare
    print(ms1.pointList[1].getForces())
    print(ms.pointList[1].getForces())
    print(ms1.pointList[3].getForces())
    print(ms.pointList[2].getForces())

    print(ms1.pointList[1].getStiffnessA())
    print(ms.pointList[1].getStiffnessA())
    print(ms1.pointList[3].getStiffnessA())
    print(ms.pointList[2].getStiffnessA())
    """

    """
    # a seabed contact case with single U line
    ms = mp.System()
    ms.depth = 100
    ms.lineTypes['chain'] = getLineProps(120, name='chain')
    # a suspension bridge shape
    ms.addPoint(1, [-200, 0 , -100])                 # 1 anchor point
    ms.addPoint(0, [-100, 0 ,  -50], m=0, v=50)      # 2 float
    ms.addPoint(0, [ 100, 0 ,  -40], m=0, v=90)      # 3 float
    ms.addPoint(1, [ 200, 0 , -100])                 # 4 anchor point
    ms.addLine(120, 'chain', pointA=1, pointB=2)
    ms.addLine(250, 'chain', pointA=2, pointB=3)
    ms.addLine(120, 'chain', pointA=3, pointB=4)
    # something in a new direction
    ms.addPoint(1, [ 100,  200, -100])               # 5 north anchor
    ms.addPoint(0, [ 100,  100,  -50])               # 6 point along seabed
    ms.addPoint(1, [ 100, -100,  -50])               # 7 south anchor in midair
    ms.addLine(120, 'chain', pointA=5, pointB=6)
    ms.addLine(120, 'chain', pointA=6, pointB=3)
    ms.addLine(250, 'chain', pointA=7, pointB=3)
    # ------- simulate it ----------
    ms.initialize()
    fig,ax = ms.plot(color='g')
    ms.solveEquilibrium3(tol=0.0001)
    ms.plot(color=[1,0,0,0.5], ax=ax)

    #>>> add a test for stiffnesses at each end of lines that have seaed contact, comparing U shape to two catenary lines. Also look at system-level stiffnes.

    plt.show()
    """
