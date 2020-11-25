.. _openmdao_tutorial-label:
  
4. OpenMDAO Examples
--------------------

WISDEM can be run through the yaml-input files if the intention is to do a full turbine and LCOE roll-up.  Sometimes though, a user might just want to evaluate or optimize a single component within WISDEM.  This can also be done through the yaml-input files, and some of the examples for the tower, monopile, and drivetrain show how that is accomplished.  However, for other modules it may be simpler to interface with WISDEM directly with a python script.  Since WISDEM is built on top of the `OpenMDAO <http://openmdao.org>`__ library, this tutorial is a cursory introduction into OpenMDAO syntax and problem structure.

OpenMDAO serves to
connect the various components of turbine models into a cohesive whole
that can be optimized in systems engineering problems. WISDEM uses
OpenMDAO to build up modular **components** and **groups** of components to
represent a wind turbine. Fortunately, OpenMDAO already provides some
excellent training examples on their `website <http://openmdao.org>`__.

.. contents:: List of Examples
   :depth: 2


Tutorial 1: Betz Limit
======================

This tutorial is based on the OpenMDAO example, `Optimizing an Actuator
Disk Model to Find Betz Limit for Wind
Turbines <http://openmdao.org/twodocs/versions/latest/examples/betz_limit/betz.html>`__,
which we have extracted and added some additional commentary. The aim of
this tutorial is to summarize the key points you’ll use to create basic
WISDEM models. For those interested in WISDEM development, getting
comfortable with all of the core OpenMDAO training examples is strongly
encouraged.

A classic problem of wind energy engineering is the Betz limit. This is
the theoretical upper limit of power that can be extracted from wind by
an idealized rotor. While a full explanation is beyond the scope of this
tutorial, it is explained in many places online and in textbooks. One
such explanation is on Wikipedia
https://en.wikipedia.org/wiki/Betz%27s_law .

Problem formulation
*******************

According to the Betz limit, the maximum power a turbine can extract
from wind is:

.. math::  C_p = \frac{16}{27} \approx 0.593

Where :math:`C_p` is the coefficient of power.

We will compute this limit using OpenMDAO by optimizing the power
coefficient as a function of the induction factor (ratio of rotor plane
velocity to freestream velocity) and a model of an idealized rotor using
an actuator disk.

Here is our actuator disc:

.. figure:: /images/openmdao/actuator_disc.png
   :alt: actuator disc

   Actuator disc

Where :math:`V_u` freestream air velocity, upstream of rotor, :math:`V_r` is
air velocity at rotor exit plane and :math:`V_d` is slipstream air velocity
downstream of rotor, all measured in :math:`\frac{m}{s}`.

There are few other variables we’ll have:

-  :math:`a`: Induced Velocity Factor
-  **Area**: Rotor disc area in :math:`m^2`
-  **thrust**: Thrust produced by the rotor in N
-  :math:`C_t`: Thrust coefficient
-  **power**: Power produced by rotor in *W*
-  :math:`\rho`: Air density in :math:`kg /m^3`

Before we start in on the source code, let’s look at a few key snippets
of the code

OpenMDAO implementation
***********************

First we need to import OpenMDAO

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Import
    :end-before: # --

Now we can make an :code:`ActuatorDisc` class that models the actuator disc
theory for the optimization.  This is derived from an OpenMDAO class

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Specific
    :end-before: # -- end

The class declaration, :code:`class ActuatorDisc(om.ExplicitComponent):`
shows that our class, :code:`ActuatorDisc` inherits from the
:code:`ExplicitComponent` class in OpenMDAO. In WISDEM, 99% of all coded
components are of the :code:`ExplicitComponent` class, so this is the most
fundamental building block to get accustomed to. Other types of
components are described in the OpenMDAO docs
`here <http://openmdao.org/twodocs/versions/latest/_srcdocs/packages/openmdao.components.html>`__.

The :code:`ExplicitComponent` class provides a template for the user to: -
Declare their input and output variables in the :code:`setup` method -
Calculate the outputs from the inputs in the :code:`compute` method. In an
optimization loop, this is called at every iteration. - Calculate
analytical gradients of outputs with respect to inputs in the
:code:`compute_partials` method.

The variable declarations take the form of :code:`self.add_input` or
:code:`self.add_output` where a variable name and default/initial vaue is
assigned. The value declaration also tells the OpenMDAO internals about
the size and shape for any vector or multi-dimensional variables. Other
optional keywords that can help with code documentation and model
consistency are :code:`units=` and :code:`desc=`.

Working with analytical derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need to tell OpenMDAO which derivatives will need to be computed.
That happens in the following lines:

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Declare
    :end-before: # --

Note that lines like :code:`self.declare_partials('Vr', ['a', 'Vu'])`
references both the derivatives :math:`\partial `V_r /
:math:`\partial `a` and :math:`\partial `V_r /
:math:`\partial `V_u`.

The Jacobian in which we provide solutions to the derivatives is

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Derivatives
    :end-before: # -- end

In OpenMDAO, multiple components can be connected together inside of a
Group. There will be some other new elements to review, so let’s take a
look:

Betz Group:
~~~~~~~~~~~

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Optional
    :end-before: # --

The :code:`Betz` class derives from the OpenMDAO :code:`Group` class, which is
typically the top-level class that is used in an analysis. The OpenMDAO
:code:`Group` class allows you to cluster models in hierarchies. We can put
multiple components in groups. We can also put other groups in groups.

Components are added to groups with the :code:`self.add_subsystem` command,
which has two primary arguments. The first is the string name to call
the subsystem that is added and the second is the component or sub-group
class instance. A common optional argument is :code:`promotes=`, which
elevates the input/output variable string names to the top-level
namespace. The :code:`Betz` group shows examples where the :code:`promotes=` can
be passed a list of variable string names or the :code:`'*'` wildcard to
mean all input/output variables.

The first subsystem that is added is an :code:`IndepVarComp`, which are the
independent variables of the problem. Subsystem inputs that are not tied
to other subsystem outputs should be connected to an independent
variables. For optimization problems, design variables must be part of
an :code:`IndepVarComp`. In the Betz problem, we have :code:`a`, :code:`Area`,
:code:`rho`, and :code:`Vu`. Note that they are promoted to the top level
namespace, otherwise we would have to access them by :code:`'indeps.x'` and
:code:`'indeps.z'`.

The next subsystem that is added is an instance of the component we
created above:

.. code:: python

   self.add_subsystem('a_disk', ActuatorDisc(), promotes=['a', 'Area', 'rho', 'Vu'])

The :code:`promotes=` can also serve to connect variables. In OpenMDAO, two
variables with the same string name in the same namespace are
automatically connected. By promoting the same variable string names as
in the :code:`IndepCarComp`, they are automatically connected. For variables
that are not connected in this way, explicit connect statements are
required, which is demonstrated in the next tutorial. ## Let’s optimize
our system!

Even though we have all the pieces in a :code:`Group`, we still need to put
them into a :code:`Problem` to be executed. The :code:`Problem` instance is
where we can assign design variables, objective functions, and
constraints. It is also how the user interacts with the :code:`Group` to set
initial conditions and interrogate output values.

First, we instantiate the :code:`Problem` and assign an instance of :code:`Betz`
to be the root model:

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Instantiate
    :end-before: # --

Next we assign an optimization driver to the problem instance. If we
only wanted to evaluate the model once and not optimize, then a driver
is not needed:

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Specify
    :end-before: # --

With the optimization driver in place, we can assign design variables,
objective(s), and constraints. Any IndepVarComp can be a design variable
and any model output can be an objective or constraint.

We want to maximize the objective, but OpenMDAO will want to minimize it
as it is consistent with the standard optimization problem statement. So
we minimize the negative to find the maximum. Note that :code:`Cp` is not
promoted from :code:`a_disk`. Therefore we must reference it with
:code:`a_disk.Cp`

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Assign
    :end-before: # --

Now we can run the optimization:

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Execute
    :end-before: # --

.. code:: console

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -0.5925925906659251
                Iterations: 5
                Function evaluations: 6
                Gradient evaluations: 5
    Optimization Complete
    -----------------------------------


Finally, the result:
********************

Above, we see a summary of the steps in our optimization.  Next, we print out the values we
care about and list all of the inputs and outputs that are problem used.

.. literalinclude:: /../examples/04_openmdao/betz_limit.py
    :start-after: # Display
    :end-before: # --


.. code:: console

    Coefficient of power Cp =  [0.59259259]
    Induction factor a = [0.33335528]
    Rotor disc Area = [1.] m^2
    4 Input(s) in 'model'
    ---------------------

    varname   value
    --------  ------------
    top
      a_disk
        a     [0.33335528]
        Area  [1.]
        rho   [1.225]
        Vu    [10.]


    10 Explicit Output(s) in 'model'
    --------------------------------

    varname     value
    ----------  --------------
    top
      indeps
        a       [0.33335528]
        Area    [1.]
        rho     [1.225]
        Vu      [10.]
      a_disk
        Vr      [6.6664472]
        Vd      [3.33289439]
        Ct      [0.88891815]
        Cp      [0.59259259]
        power   [362.96296178]
        thrust  [54.44623668]


    0 Implicit Output(s) in 'model'
    -------------------------------



And there we have it. This matched the Betz limit of:

.. math::  C_p = \frac{16}{27} \approx 0.593



Tutorial 2. The Sellar Problem
==============================

This tutorial is based on the OpenMDAO example, `Sellar - A
Two-Discipline Problem with a Nonlinear
Solver <http://openmdao.org/twodocs/versions/latest/basic_guide/sellar.html>`__,
which we have extracted and added some additional commentary. The aim of
this tutorial is to summarize the key points needed to create or understand basic
WISDEM models. For those interested in WISDEM development, getting
comfortable with all of the core OpenMDAO training examples is strongly
encouraged.

Problem formulation
*******************

The Sellar problem are a couple of components (what Wikipedia calls
models) that are simple equations. There is an objective to optimize and
a couple of constraints to follow.

.. figure:: /images/openmdao/sellar_xdsm.png
   :alt: Sellar XDSM

   Sellar XDSM

This is an XDSM diagram that is used to describe the problem and
optimization setups. For more reference on this notation and general
reference for multidisciplinary design analysis and optimization (MDAO),
see:

-  `Problem formulation section of multidisciplinary design optimization
   on
   Wikipedia <https://en.wikipedia.org/wiki/Multidisciplinary_design_optimization#Problem_formulation>`__:
   Read the definitions for *design variables*, *constraints*,
   *objectives* and *models*.

-  `Lambe and Martins: Extensions to the Design Structure Matrix for the
   Description of Multidisciplinary Design, Analysis, and Optimization
   Processes <http://mdolab.engin.umich.edu/content/extensions-design-structure-matrix>`__:
   Read section 2 “Terminology and Notation” for further explanation of
   *design variables*, *discipline analysis*, *response variables*,
   *target variables* and *coupling variables*. Read section 4 about
   XDSM diagrams that describe MDO processes.

OpenMDAO implementation
***********************

First we need to import OpenMDAO

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Import
    :end-before: # --

Let’s build *Discipline 1* first. On the XDSM diagram, notice the
parallelograms connected to *Discipline 1* by thick grey lines. These
are variables pertaining to the *Discipline 1* component.

-  :math:`\mathbf{z}`: An input. Since the components :math:`z_1,
   z_2` can form a vector, we call the variable :code:`z` in the code and
   initialize it to :math:`(0, 0)` with :code:`np.zeros(2)`. Note that
   components of :math:`\mathbf{z}` are found in 3 of the
   white :math:`\mathbf{z}` parallelograms connected to
   multiple components and the objective, so this is a global design
   variable.

-  :math:`x`: An input. A local design variable for Discipline 1. Since it
   isn’t a vector, we just initialize it as a float.

-  :math:`y_2`: An input. This is a coupling variable coming from an output
   on *Discipline 2*

-  :math:`y_1`: An output. This is a coupling variable going to an input on
   *Discipline 2*

Let’s take a look at the *Discipline 1* component and break it down
piece by piece. ### Discipline 1

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Create
    :end-before: # -- end Discipline 1

The class declaration, :code:`class SellarDis1(om.ExplicitComponent):` shows
that our class, :code:`SellarDis1` inherits from the :code:`ExplicitComponent`
class in OpenMDAO. In WISDEM, 99% of all coded components are of the
:code:`ExplicitComponent` class, so this is the most fundamental building
block to get accustomed to. Keen observers will notice that the *Sellar
Problem* has implicitly defined variables that will need to be
addressed, but that is addressed below. Other types of components are
described in the OpenMDAO docs
`here <http://openmdao.org/twodocs/versions/latest/_srcdocs/packages/openmdao.components.html>`__.

The :code:`ExplicitComponent` class provides a template for the user to: -
Declare their input and output variables in the :code:`setup` method -
Calculate the outputs from the inputs in the :code:`compute` method. In an
optimization loop, this is called at every iteration. - Calculate
analytical gradients of outputs with respect to inputs in the
:code:`compute_partials` method. This is absent from the *Sellar Problem*.

The variable declarations take the form of :code:`self.add_input` or
:code:`self.add_output` where a variable name and default/initial vaue is
assigned. The value declaration also tells the OpenMDAO internals about
the size and shape for any vector or multi-dimensional variables. Other
optional keywords that can help with code documentation and model
consistency are :code:`units=` and :code:`desc=`.

Finally :code:`self.declare_partials('*', '*', method='fd')` tell OpenMDAO
to use finite difference to compute the partial derivative of the
outputs with respect to the inputs. OpenMDAO provides many finite
difference capabilities including: - Forward and backward differencing -
Central differencing for second-order accurate derivatives -
Differencing in the complex domain which can offer improved accuracy for
the models that support it

Now lets take a look at *Discipline 2*.

-  :math:`\mathbf{z}`: An input comprised of :math:`z_1, z_2`.
-  :math:`y_2`: An output. This is a coupling variable going to an input on
   *Discipline 1*
-  :math:`y_1`: An input. This is a coupling variable coming from an output
   on *Discipline 1*

Discipline 2
~~~~~~~~~~~~

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Now
    :end-before: # -- end Discipline 2

In OpenMDAO, multiple components can be connected together inside of a
Group. There will be some other new elements to review, so let’s take a
look:

Sellar Group:
~~~~~~~~~~~~~

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Assemble
    :end-before: # -- end Group

The :code:`SellarMDA` class derives from the OpenMDAO :code:`Group` class,
which is typically the top-level class that is used in an analysis. The
OpenMDAO :code:`Group` class allows you to cluster models in hierarchies. We
can put multiple components in groups. We can also put other groups in
groups.

Components are added to groups with the :code:`self.add_subsystem` command,
which has two primary arguments. The first is the string name to call
the subsystem that is added and the second is the component or sub-group
class instance. A common optional argument is :code:`promotes=`, which
elevates the input/output variable string names to the top-level
namespace. The :code:`SellarMDA` group shows examples where the
:code:`promotes=` can be passed a list of variable string names or the
:code:`'*'` wildcard to mean all input/output variables.

The first subsystem that is added is an :code:`IndepVarComp`, which are the
independent variables of the problem. Subsystem inputs that are not tied
to other subsystem outputs should be connected to an independent
variables. For optimization problems, design variables must be part of
an :code:`IndepVarComp`. In the Sellar problem, we have :code:`x` and :code:`z`.
Note that they are promoted to the top level namespace, otherwise we
would have to access them by :code:`'indeps.x'` and :code:`'indeps.z'`.

The next subsystems that are added are instances of the components we
created above:

.. code:: python

   self.add_subsystem('d1', SellarDis1(), promotes=['y1', 'y2'])
   self.add_subsystem('d2', SellarDis2(), promotes=['y1', 'y2'])

The :code:`promotes=` can also serve to connect variables. In OpenMDAO, two
variables with the same string name in the same namespace are
automatically connected. By promoting :code:`y1` and :code:`y2` in both :code:`d1`
and :code:`d2`, they are automatically connected. For variables that are not
connected in this way, explicit connect statements are required such as:

.. code:: python

   self.connect('x', ['d1.x','d2.x'])
   self.connect('z', ['d1.z','d2.z'])

These statements connect the :code:`IndepVarComp` versions of :code:`x` and
:code:`z` to the :code:`d1` and :code:`d2` versions. Note that if :code:`x` and :code:`z`
could easily have been promoted in :code:`d1` and :code:`d2` too, which would
have made these connect statements unnecessary, but including them is
instructive.

The next statement, :code:`self.nonlinear_solver = om.NonlinearBlockGS()`,
handles the required internal iteration between :code:`y1` and :code:`y2` is our
two components. OpenMDAO is able to identify a *cycle* between
input/output variables and requires the user to specify a solver to
handle the nested iteration loop. WISDEM does its best to avoid cycles.

Finally, we have a series of three subsystems that use instances of the
OpenMDAO :code:`ExecComp` component. This is a useful way to defining an
:code:`ExplicitComponent` inline, without having to create a whole new
class. OpenMDAO is able to parse the string expression and populate the
:code:`setup` and :code:`compute` methods automatically. This technique is used
to create our objective function and two constraint functions directly:

.. code:: python

   self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                             z=np.array([0.0, 0.0]), x=0.0),
                      promotes=['x', 'z', 'y1', 'y2', 'obj'])
   self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
   self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

Let’s optimize our system!
**************************

Even though we have all the pieces in a :code:`Group`, we still need to put
them into a :code:`Problem` to be executed. The :code:`Problem` instance is
where we can assign design variables, objective functions, and
constraints. It is also how the user interacts with the :code:`Group` to set
initial conditions and interrogate output values.

First, we instantiate the :code:`Problem` and assign an instance of
:code:`SellarMDA` to be the root model:

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Instantiate
    :end-before: # --

Next we assign an optimization :code:`driver` to the problem instance. If we
only wanted to evaluate the model once and not optimize, then a
:code:`driver` is not needed:

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Specify
    :end-before: # --

With the optimization driver in place, we can assign design variables,
objective(s), and constraints. Any :code:`IndepVarComp` can be a design
variable and any model output can be an objective or constraint.

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Assign
    :end-before: # --

Now we are ready for to ask OpenMDAO to setup the model, to use finite
differences for gradient approximations, and to run the driver:

.. literalinclude:: /../examples/04_openmdao/sellar.py
    :start-after: # Execute
    :end-before: # --

.. code:: console

    NL: NLBGS Converged in 7 iterations
    NL: NLBGS Converged in 0 iterations
    NL: NLBGS Converged in 3 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 8 iterations
    NL: NLBGS Converged in 3 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 9 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 5 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 9 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 5 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 8 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 5 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 5 iterations
    NL: NLBGS Converged in 4 iterations
    NL: NLBGS Converged in 5 iterations
    NL: NLBGS Converged in 4 iterations
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 3.183393951735934
                Iterations: 6
                Function evaluations: 6
                Gradient evaluations: 6
    Optimization Complete
    -----------------------------------
    minimum found at
    0.0
    [1.97763888e+00 2.83540724e-15]
    minimum objective
    3.183393951735934
