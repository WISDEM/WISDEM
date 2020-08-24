import numpy as np
from time import time
from weis.multifidelity.models.run_functions import CCBlade, FullCCBlade


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

s = time()
for i in range(1):
    desvars = {'blade.opt_var.chord_opt_gain' : np.array([1.2, 1.09, 1.1, 1.2, 1.3])}
    model_low = FullCCBlade(desvars)

    print(model_low.run(desvars['blade.opt_var.chord_opt_gain']))
times = [time() - s]

s = time()
for i in range(1):
    desvars = {'blade.opt_var.chord_opt_gain' : np.array([1.2, 1.09, 1.1, 1.2, 1.3])}
    model_low = CCBlade(desvars, n_span=10)

    print(model_low.run(desvars['blade.opt_var.chord_opt_gain']))
    exit()
times.append(time() - s)

print(times)