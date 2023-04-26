from pyxdsm.XDSM import OPT, FUNC, XDSM, SOLVER

x = XDSM()

# Instantiate on-diagonal blocks
# order of args goes: object name, type of object, string for block on XDSM
x.add_system("opt", OPT, r"\text{Optimizer}")
x.add_system("rotor", FUNC, r"\text{RotorSE}")
x.add_system("drive", FUNC, r"\text{DrivetrainSE}")
x.add_system("tower", FUNC, r"\text{TowerSE}")
x.add_system("CSM", FUNC, r"\text{NREL CSM}")
x.add_system("BOS", FUNC, r"\text{LandBOSSE}")
x.add_system("costs", FUNC, r"\text{Plant\_FinanceSE}")

# Feed-forward connections; from, to, name for connections
x.connect("opt", "rotor", r"\text{Blade design variables}")
x.connect("rotor", "drive", r"\text{Performance, loads}")
x.connect("rotor", "BOS", r"\text{Rotor and blade mass}")
x.connect("drive", "tower", (r"\text{Forces, moments,}", r"\text{mass properties}"))
x.connect("tower", "CSM", r"\text{Tower mass}")
x.connect("drive", "CSM", r"\text{Drivetrain mass}")
x.connect("rotor", "CSM", r"\text{Rotor mass}")
x.connect("CSM", "BOS", r"\text{Turbine mass and costs}")
x.connect("BOS", "costs", r"\text{BOS costs}")

# Feed-backward connections
x.connect("drive", "rotor", r"\text{Efficiency}")
x.connect("costs", "opt", r"\text{Cost and profit values}")
x.connect("tower", "opt", (r"\text{Tower stress and}", r"\text{buckling values}"))
x.connect("rotor", "opt", (r"\text{Blade tip deflection,}", r"\text{blade strains}"))

# Outputs on the left-hand side
# x.add_output('opt', 'x^*, z^*', side='left')

x.add_process(["opt", "rotor", "drive", "tower", "CSM", "BOS", "costs", "opt"])

# Compile latex and write pdf
x.write("xdsm_wisdem")
