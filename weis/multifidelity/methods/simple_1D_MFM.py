import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from weis.multifidelity.models.testbed_components import simple_1D_low, simple_1D_high


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(14)


x_init = np.random.rand(3)
x = x_init.copy()

for i in range(20):
    y_low = simple_1D_low(x)
    y_high = simple_1D_high(x)

    # Construct RBF interpolater for error function
    differences = y_high - y_low
    print(x)
    print(differences)
    print()
    
    try:
        e = Rbf(x, differences, epsilon=0.1)
    except:
        print("Done!")
        break

    # Create m_k = lofi + RBF
    m = lambda x: simple_1D_low(x) + e(x)

    x_plot = np.linspace(0., 1., 101)
    surrogate = m(x_plot)
    y_plot_low = simple_1D_low(x_plot)
    y_plot_high = simple_1D_high(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot_low, label='low-fidelity')
    plt.plot(x_plot, y_plot_high, label='high-fidelity')
    plt.plot(x_plot, surrogate, label='mixed surrogate')
    plt.scatter(x, y_high, color='k')
    
    plt.xlim([0., 1.])
    plt.ylim([-10, 15])
    plt.legend()
    plt.savefig(f'image_{i}.png')
    

    x0 = x[-1]
    res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=[(0., 1.)])
    x_new = res.x
    x = np.hstack((x, x_new))
    
