# MoorProps
# start of a script to hold the functions that hold the mooring line and anchor property info
# just simple functions for now, can import functions to other scripts if necessary
# very much in progress
# MoorProps started 4-26-21, getLineProps first initialized back in Sept/Oct 2020


import numpy as np
import wisdem.moorpy as mp


def getLineProps(dmm, type="chain", stud="studless", source="Orcaflex-altered", name=""):
    """getLineProps version 3.2: Restructuring v3.1 to 'Orcaflex-original' and 'Orcaflex-altered'

    Motivation: The existing public, and NREL-internal references for mooring line component property
    data are either proprietary, or unreliable and incomplete. The best way to derive new equations
    as a function of diameter is to use data from mooring line manufacturer's catalogs. Once developed,
    these new equations will serve as an updated version to compare against existing expressions.

    The goal is to have NREL's own library of mooring line property equations, but more research is needed.
    The original Orcaflex equations are the best we have right now and have been altered to include
    a quadratic chain MBL equation rather than a cubic, to avoid negative MBLs.
    Also, different cost models are put in to the altered version to avoid the Equimar data. Many sources
    researched for cost data, coefficients used are close to the old NREL internal data, but still an approximation.

    For more info, see the Mooring Component Properties Word doc.

    - This function requires at least one input: the line diameter in millimeters.
    - The rest of the inputs are optional: describe the desired type of line (chain, polyester, wire, etc.),
    the type of chain (studless or studlink), the source of data (Orcaflex-original or altered), or a name identifier
    - The function will output a MoorPy linetype object

    """

    if source == "Orcaflex-original":
        d = dmm / 1000  # orcaflex uses meters https://www.orcina.com/webhelp/OrcaFlex/

        if type == "chain":
            c = 1.96e4  # grade 2=1.37e4; grade 3=1.96e4; ORQ=2.11e4; R4=2.74e4
            MBL = c * d ** 2 * (44 - 80 * d) * 1000  # [N]  The same for both studless and studlink
            if stud == "studless":
                massden = 19.9 * d ** 2 * 1000  # [kg/m]
                EA = 0.854e8 * d ** 2 * 1000  # [N]
                d_vol = 1.8 * d  # [m]
            elif stud == "studlink" or stud == "stud":
                massden = 21.9 * d ** 2 * 1000  # [kg/m]
                EA = 1.010e8 * d ** 2 * 1000  # [N]
                d_vol = 1.89 * d  # [m]
            else:
                raise ValueError("getLineProps error: Choose either studless or stud chain type ")

        elif type == "nylon":
            massden = 0.6476 * d ** 2 * 1000  # [kg/m]
            EA = 1.18e5 * d ** 2 * 1000  # [N]
            MBL = 139357 * d ** 2 * 1000  # [N] for wet nylon line, 163950d^2 for dry nylon line
            d_vol = 0.85 * d  # [m]
        elif type == "polyester":
            massden = 0.7978 * d ** 2 * 1000  # [kg/m]
            EA = 1.09e6 * d ** 2 * 1000  # [N]
            MBL = 170466 * d ** 2 * 1000  # [N]
            d_vol = 0.86 * d  # [m]
        elif type == "polypropylene":
            massden = 0.4526 * d ** 2 * 1000  # [kg/m]
            EA = 1.06e6 * d ** 2 * 1000  # [N]
            MBL = 105990 * d ** 2 * 1000  # [N]
            d_vol = 0.80 * d  # [m]
        elif type == "wire-fiber" or type == "fiber":
            massden = 3.6109 * d ** 2 * 1000  # [kg/m]
            EA = 3.67e7 * d ** 2 * 1000  # [N]
            MBL = 584175 * d ** 2 * 1000  # [N]
            d_vol = 0.82 * d  # [m]
        elif type == "wire-wire" or type == "wire" or type == "IWRC":
            massden = 3.9897 * d ** 2 * 1000  # [kg/m]
            EA = 4.04e7 * d ** 2 * 1000  # [N]
            MBL = 633358 * d ** 2 * 1000  # [N]
            d_vol = 0.80 * d  # [m]
        else:
            raise ValueError("getLineProps error: Linetype not valid. Choose from given rope types or chain ")

        # cost
        # Derived from Equimar graph: https://tethys.pnnl.gov/sites/default/files/publications/EquiMar_D7.3.2.pdf
        if type == "chain":
            cost = (0.21 * (MBL / 9.81 / 1000)) * 1.29  # [$/m]
        elif type == "nylon" or type == "polyester" or type == "polypropylene":
            cost = (0.235 * (MBL / 9.81 / 1000)) * 1.29  # [$/m]
        elif type == "wire" or type == "wire-wire" or type == "IWRC" or type == "fiber" or type == "wire-fiber":
            cost = (0.18 * (MBL / 9.81 / 1000) + 90) * 1.29  # [$/m]
        else:
            raise ValueError("getLineProps error: Linetype not valid. Choose from given rope types or chain ")

    elif source == "Orcaflex-altered":
        d = dmm / 1000  # orcaflex uses meters https://www.orcina.com/webhelp/OrcaFlex/

        if type == "chain":
            c = 2.74e4  # grade 2=1.37e4; grade 3=1.96e4; ORQ=2.11e4; R4=2.74e4
            MBL = (
                (371360 * d ** 2 + 51382.72 * d) * (c / 2.11e4) * 1000
            )  # this is a fit quadratic term to the cubic MBL equation. No negatives
            if stud == "studless":
                massden = 19.9 * d ** 2 * 1000  # [kg/m]
                EA = 0.854e8 * d ** 2 * 1000  # [N]
                d_vol = 1.8 * d  # [m]
            elif stud == "studlink" or stud == "stud":
                massden = 21.9 * d ** 2 * 1000  # [kg/m]
                EA = 1.010e8 * d ** 2 * 1000  # [N]
                d_vol = 1.89 * d  # [m]
            else:
                raise ValueError("getLineProps error: Choose either studless or stud chain type ")

            # cost = 2.5*massden   # a ballpark for R4 chain
            # cost = (0.58*MBL/1000/9.81) - 87.6          # [$/m] from old NREL-internal
            # cost = 3.0*massden     # rough value similar to old NREL-internal
            cost = 2.585 * massden  # [($/kg)*(kg/m)=($/m)]
            # cost = 0.0

        elif type == "nylon":
            massden = 0.6476 * d ** 2 * 1000  # [kg/m]
            EA = 1.18e5 * d ** 2 * 1000  # [N]
            MBL = 139357 * d ** 2 * 1000  # [N] for wet nylon line, 163950d^2 for dry nylon line
            d_vol = 0.85 * d  # [m]
            cost = (0.42059603 * MBL / 1000 / 9.81) + 109.5  # [$/m] from old NREL-internal
        elif type == "polyester":
            massden = 0.7978 * d ** 2 * 1000  # [kg/m]
            EA = 1.09e6 * d ** 2 * 1000  # [N]
            MBL = 170466 * d ** 2 * 1000  # [N]
            d_vol = 0.86 * d  # [m]

            # cost = (0.42059603*MBL/1000/9.81) + 109.5   # [$/m] from old NREL-internal
            # cost = 1.1e-4*MBL               # rough value similar to old NREL-internal
            cost = 0.162 * (MBL / 9.81 / 1000)  # [$/m]

        elif type == "polypropylene":
            massden = 0.4526 * d ** 2 * 1000  # [kg/m]
            EA = 1.06e6 * d ** 2 * 1000  # [N]
            MBL = 105990 * d ** 2 * 1000  # [N]
            d_vol = 0.80 * d  # [m]
            cost = (0.42059603 * MBL / 1000 / 9.81) + 109.5  # [$/m] from old NREL-internal
        elif type == "wire-fiber" or type == "fiber":
            massden = 3.6109 * d ** 2 * 1000  # [kg/m]
            EA = 3.67e7 * d ** 2 * 1000  # [N]
            MBL = 584175 * d ** 2 * 1000  # [N]
            d_vol = 0.82 * d  # [m]
            cost = 0.53676471 * MBL / 1000 / 9.81  # [$/m] from old NREL-internal
        elif type == "wire-wire" or type == "wire" or type == "IWRC":
            massden = 3.9897 * d ** 2 * 1000  # [kg/m]
            EA = 4.04e7 * d ** 2 * 1000  # [N]
            MBL = 633358 * d ** 2 * 1000  # [N]
            d_vol = 0.80 * d  # [m]
            # cost = MBL * 900./15.0e6
            # cost = (0.33*MBL/1000/9.81) + 139.5         # [$/m] from old NREL-internal
            cost = 5.6e-5 * MBL  # rough value similar to old NREL-internal
        else:
            raise ValueError("getLineProps error: Linetype not valid. Choose from given rope types or chain ")

    elif source == "NREL":
        """
        getLineProps v3.1 used to have old NREL-internal equations here as a placeholder, but they were not trustworthy.
         - The chain equations used data from Vicinay which matched OrcaFlex data. The wire rope equations matched OrcaFlex well,
           the synthetic rope equations did not

        The idea is to have NREL's own library of mooring line property equations, but more research needs to be done.
        The 'OrcaFlex-altered' source version is a start and can change name to 'NREL' in the future, but it is
        still 'OrcaFlex-altered' because most of the equations are from OrcaFlex, which is the best we have right now.

        Future equations need to be optimization proof = no negative numbers anywhere (need to write an interpolation function)
        Can add new line types as well, such as Aramid or HMPE
        """
        pass

    # Set up a main identifier for the linetype. Useful for things like "chain_bot" or "chain_top"
    if name == "":
        typestring = f"{type}{dmm:.0f}"
    else:
        typestring = name

    notes = f"made with getLineProps - source: {source}"

    return mp.LineType(typestring, d_vol, massden, EA, MBL=MBL, cost=cost, notes=notes, input_type=type, input_d=dmm)


def getAnchorProps(fx, fz, type="drag-embedment", display=0):
    """ Calculates anchor required capacity and cost based on specified loadings and anchor type"""

    # for now this is based on API RP-2SK guidance for static analysis of permanent mooring systems
    # fx and fz are horizontal and vertical load components assumed to come from a dynamic (or equivalent) analysis.

    # mooring line tenLimit specified in yaml and inversed for a SF in constraints
    # take the line forces on the anchor and give 20% consideration for dynamic loads on lines
    # coefficients in front of fx and fz in each anchorType are the SF for that anchor for quasi-static (pages 30-31 of RP-2SK)

    # scale QS loads by 20% to approximate dynamic loads
    fx = 1.2 * fx
    fz = 1.2 * fz

    # note: capacity is measured here in kg force

    euros2dollars = 1.18  # the number of dollars there currently are in a euro (3-31-21)

    if type == "drag-embedment":
        capacity_x = 1.5 * fx / 9.81

        fzCost = 0
        # if fz > 0:
        #    fzCost = 1e3*fz
        #    if display > 0:  print('WARNING: Nonzero vertical load specified for a drag embedment anchor.')

        anchorMatCost = 0.188 * capacity_x + fzCost  # material cost
        anchorInstCost = 163548 * euros2dollars  # installation cost
        anchorDecomCost = 228967 * euros2dollars  # decommissioning cost

    elif type == "suction":
        capacity_x = 1.6 * fx / 9.81
        capacity_z = 2.0 * fz / 9.81
        capacity = np.linalg.norm([capacity_x, capacity_z])  # overall capacity, assuming in any direction for now
        anchorMatCost = 1.08 * capacity  # material cost
        anchorInstCost = 179331 * euros2dollars  # installation cost
        anchorDecomCost = 125532 * euros2dollars  # decommissioning cost

    elif type == "plate":
        capacity_x = 2.0 * fx / 9.81
        capacity_z = 2.0 * fz / 9.81
        capacity = np.linalg.norm([capacity_x, capacity_z])  # overall capacity, assuming in any direction for now
        raise ValueError("plate anchors not yet supported")

    elif type == "micropile":  # note: no API guidance on this one
        capacity_x = 2.0 * fx / 9.81
        capacity_z = 2.0 * fz / 9.81
        capacity = np.linalg.norm([capacity_x, capacity_z])  # overall capacity, assuming in any direction for now
        anchorMatCost = (
            200000 * 1.2 / 500000
        ) * capacity  # [(Euros*($/Euros)/kg)*kg] linear interpolation of material cost
        anchorInstCost = 0  # installation cost
        anchorDecomCost = 0  # decommissioning cost

    elif type == "SEPLA":  # cross between suction and plate
        capacity_x = 2.0 * fx / 9.81
        capacity_z = 2.0 * fz / 9.81
        capacity = np.linalg.norm([capacity_x, capacity_z])  # overall capacity, assuming in any direction for now
        anchorMatCost = 0.45 * capacity  # material cost
        anchorInstCost = 0  # installation cost
        anchorDecomCost = 0  # decommissioning cost

    else:
        raise ValueError(f"getAnchorProps received an unsupported anchor type ({type})")

    # mooring line sizing:  Tension limit for QS: 50% MBS.  Or FOS = 2

    return anchorMatCost, anchorInstCost, anchorDecomCost  # [USD]
