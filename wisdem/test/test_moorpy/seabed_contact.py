# playing around with seabed contact scenarios

# import pytest
# from numpy.testing import assert_allclose

import numpy as np
import wisdem.moorpy as mp
import matplotlib.pyplot as plt

ms = mp.System(depth=200)

ms.addPoint(1, [-100, 0, -50])
ms.addPoint(1, [100, 0, 0])
ms.addPoint(0, [0, 0, -90])

ms.addLineType("chain", 0.1, 50.0, 1e12)

ms.addLine(120, "chain")
ms.addLine(200, "chain")

ms.pointList[2].attachLine(1, 0)
ms.pointList[2].attachLine(2, 0)
ms.pointList[0].attachLine(1, 1)
ms.pointList[1].attachLine(2, 1)

ms.initialize()

ms.solveEquilibrium3()
fig, ax = ms.plot2d(color="r")

for depth in [100, 110, 120, 130, 140]:
    ms.depth = depth
    ms.solveEquilibrium3()
    ms.plot2d(color="k", ax=ax)

plt.show()
