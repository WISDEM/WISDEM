Introduction
------------
.. bibliography:: references.bib

JacketSE is a module that allows for the analysis of preliminary designs of 3 and 4-legged jackets and towers.
Coupled to an optimizer, it will render the design that achieves the objective, usually set as minimum overall mass, while satisfying both geometric, modal, and load criteria.
The jackets can be seen with with piles either driven through the legs, or fastened to pile-sleeves at the bottom of the structure.
Pile heads can be located above sea-bed, in that case the connection to the legs is expected at the pile heads.
The embedded pile lengths and stiffness characteristics are calculated via simplistic semiempirical laws.
Hydrodynamic loads are calculated on the legs only, but a generous safety factor is included to account for contributions from other members.
The first order Airy theory and Morison's equation are used to assess a pseudo-static maximum wave load.
Wind drag loads are calculated on the tower and portions of the legs above mean sea level.
Hydrostatic loads are ignored.
Modal analysis is performed ignoring contributions of added mass or flooded members.

The tool can size piles, legs and brace diameters and wall thickness, batter angle, and tower base and top diameter, wall thickness schedules and tapering height (height above which tower diameter and thickness are tapered).
It also allows for parametric investigations and sensitivity analyses of both external factors and geometric variables that may drive the characteristics of the structure, thereby illustrating their impact on the mass, stiffness, blade/support clearance, strength, reliability, and expected costs.
Within WISDEM, JacketSE allows for the full gamut of component investigations to arrive at a minimum LCOE wind turbine and power plant layout.

JacketSE is based on a modular code framework and primarily consists of the following submodules: a geometry-definition module,
a finite element model (a modified version of `Frame3DD <https://frame3dd.sourceforge.net/>`_, a soil-pile-interaction module, a code checker module (based on API, GL, and Eurocode guidelines), and an optimization module.
More details on the code can be found at `WISDEM <https://github.com/WISDEM/>`_, and :cite:`damiani13`, and :ref:`theory`.
