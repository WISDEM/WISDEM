.. _documentation-label:

.. currentmodule:: wisdem.lcoe.lcoe_csm_assembly

How WISDEM works
================

Introduction
------------

Full wind plants are comprised of multiple subsystems with varying degrees of technical complexity and many interfaces between many stakeholders. A systems engineering approach can transcend these subsystem boundaries and rigid interfaces to identify lower cost and higher performing designs that could not otherwise be achieved by focusing on individual components. The same approach also enables full system cost-benefit tradeoff and sensitivity studies when evaluating new component or logistical innovations. The Wind-plant Integrated System Design and Engineering Model (WISDEM) is an opensource software package that aimes to meet these challenges and empower researcher to meet the following objects:

- Apply multidisciplinary analysis and optimization (MDAO) to engineering and cost models in an open framework to enable full wind turbine and plant system analysis
- Integrate technology or logistic innovations into the turbine and plant design through full system cost-benefit tradeoffs and sensitivity analyses
- Promote collaborative research and analysis among national laboratories, industry, and academia

.. figure:: /images/wisdem/workflow.*
    :align: center

Software Implementation
-----------------------

WISDEM is written in Python using `OpenMDAO <https://openmdao.org/>`_ to manage data flow between analysis blocks and to specify the workflow when performing an analysis or optimization.  WISDEM consists of a collection of physics and cost models for different components, at different fidelity levels, that can be combined together to answer system level research questions. All WISDEM models are steady-state and computational efficiency has always represented an important goal during the development of WISDEM to support wide explorations of the solution space.


.. figure:: /images/wisdem/WISDEM_Overview2.*
    :align: center

The models composing WISDEM are all integrated (glued) into a single workflow. The order of the models in the workflow follows the load path, therefore going from the rotor, to the nacelle, to the tower. Lastly, the cost models are called.

Rotor
~~~~~~~~~~~~~~~~

WISDEM simulates the rotor with steady-state models. The rotor aerodynamics is solved with the blade element momentum model :ref:`ccblade`, the elastic properties of the composite blades are obtained running the cross sectional solver :ref:`precomp`, and the deformations are obtained running the Timoshenko beam solver `Frame3DD <http://frame3dd.sourceforge.net/>`_. A regulation trajectory is implemented maximizing the rotor performance until rated power is reached and imposing constant power above rated wind speed. An optional constraint on maximum blade tip speed is respected maximizing power performance while keeping constant rotor speed. The annual energy production is computed here, whereas the ultimate loads are estimated by running a CCBlade simulation at rated pitch and rotor speed values and at a wind speed corresponding to the peak of the three-sigma gust for the extreme turbulence model. This approach to estimate loads is known to be somewhat over-conservative, but it is capable of capturing the relative trends and it is suitable to run iterative optimization loop on standard hardware in just a few minutes, offering to the designer the chance of a wide exploration of the solution space.

Drivetrain
~~~~~~~~~~~~~~~~

WISDEM supports both geared and directdrive wind turbine configurations and the engineering models described in :ref:`drivese` are used to design the various nacelle components. Simplified analytical relations are adopted to size the pitch system, the hub, the main bearings, the low speed shaft, a three-stage gearbox, the high speed shaft, the generator, the bedplate, the nacelle cover, a transformer, the yaw bearings and motors, and auxiliary systems. The components are designed assuming a desired value of gearbox ratio and overhang, whereas the loading comes from the rotor models described above. 

Tower
~~~~~~~~~~~~~~~~

The tower is modelled as a sequence of conical hollow cylinders made of steel. The tower is modeled as an elastic beam with a point mass at the top within `Frame3DD <http://frame3dd.sourceforge.net/>`_. The mass and center of gravity of the rotor-nacelle-assembly together with the loads at tower top are fed to the model, which computes values of maximum stresses, natural frequencies, and buckling limits. The ratio of wall thickness to outer diameter and the rate of change of wall thickness along tower height are also computed and can be constrained during the design optimizations.

Cost analysis
~~~~~~~~~~~~~~~~

Over the years the National Renewable Energy Laboratory has released multiple models that estimate the costs of wind energy.
This work combines together a detailed blade cost model :ref:`bcm`, a model to estimate the costs of the other wind turbine components and the overall turbine capital costs :ref:`nrelcsm`, and the financial model to compute the levelized cost of energy :ref:`financese`.
While the costs of some components, mostly in the nacelle, are estimated via semi-empirical relations tuned on historical data that get updated every few years, the blade cost model adopts a bottom up approach and estimates the total costs of a blade as the sum of variable and fixed costs. The former are made of materials and direct labor costs, while the latter are the costs from overhead, building, tooling, equipment, maintenance, and capital. The model simulates the whole manufacturing process and it has been tuned to estimate the costs of blades in the range of 30 to 100 meters in length. Lastly, the financial model is updated yearly and computes the levelized cost of energy modeling an entire wind farm.

Users should be made aware that the absolute values of costs certainly suffer a wide band of uncertainty, but the hope of the authors is that the models are able to capture the relative trends sufficiently well.


