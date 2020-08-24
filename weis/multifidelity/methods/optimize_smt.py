import numpy as np
import matplotlib.pyplot as plt
from smt.applications.mfk import MFK, NestedLHS
from smt.applications.mfkplsk import MFKPLSK

from testbed_components import simple_1D_low, simple_1D_high


# Problem set up
xlimits = np.array([[0.0, 1.0]])
xdoes = NestedLHS(nlevel=2, xlimits=xlimits)
xt_c, xt_e = xdoes(4)

# Evaluate the HF and LF functions
yt_e = simple_1D_high(xt_e)
yt_c = simple_1D_low(xt_c)

# choice of number of PLS components
ncomp = 1
sm = MFKPLSK(n_comp=ncomp, theta0=np.array(ncomp * [1.0]))

# low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(xt_c, yt_c, name=0)
# high-fidelity dataset without name
sm.set_training_values(xt_e, yt_e)

# train the model
sm.train()

x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

# query the outputs
y = sm.predict_values(x)
mse = sm.predict_variances(x)
derivs = sm.predict_derivatives(x, kx=0)

plt.figure()

plt.plot(x, simple_1D_high(x), label="reference")
plt.plot(x, y, linestyle="-.", label="mean_gp")
plt.scatter(xt_e, yt_e, marker="o", color="k", label="HF doe")
plt.scatter(xt_c, yt_c, marker="*", color="g", label="LF doe")

plt.legend(loc=0)
plt.ylim(-10, 17)
plt.xlim(-0.1, 1.1)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.show()
