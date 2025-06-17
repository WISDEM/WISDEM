import matplotlib.pyplot as plt

from wisdem.glue_code.runWISDEM import load_wisdem

refturb, _, _ = load_wisdem("outputs/refturb_output.pkl")
xs = refturb["blade.outer_shape_bem.s_default"]
ys = refturb["rotorse.rp.powercurve.ax_induct_regII"]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax.plot(xs, ys)
ax.set_xlabel("Blade Nondimensional Span [-]")
ax.set_ylabel("Axial Induction [-]")
plt.show()
