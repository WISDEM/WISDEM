Theory
======

The solution process begins by evaluating the two continuous analytical catenary equations for each element based on :math:`l` and :math:`h` values obtained through node displacement relationships. 
An element is defined as the component connecting two adjacent nodes together. 
Once the element fairlead (:math:`H`, :math:`V` ) and anchor (:math:`H_a`, :math:`V_a`) values are known at the element level, the forces are transformed from the local :math:`x_i z_i` frame into the global :math:`XYZ` coordinate system. 
The force contribution at each element's anchor and fairlead is added to the corresponding node it attaches to. 

The force-balance equation is evaluated for each node, as follows:

.. math::
   \scriptsize{\left \{ \mathbf{F} \right \}_{X}^{j} = \sum^{\text{Element i at Node } j}_{i=1} \left [ H_{i}\cos(\alpha_{i}) \right ]-F_{X_{j}}^{ext} =0}
  
   \scriptsize{\left \{ \mathbf{F} \right \}_{Y}^{j} = \sum^{\text{Element $i$ at Node } j}_{i=1} \left [ H_{i}\sin(\alpha_{i}) \right ]-F_{Y_{j}}^{ext} =0}
  
   \scriptsize{\left \{ \mathbf{F} \right \}_{Z}^{j} = \sum^{\text{Element $i$ at Node } j}_{i=1} \left [ V_{i} \right ]-F_{Z_{j}}^{ext} + M_{j}g - \rho g B_{j} =0}

Node forces are found based on the connectivity geometry between element and external forces applied at the boundary conditions. 
This :ref:`is initiated by defining a series <exploded_3d>` of :math:`\mathcal{F}_i` local frames at the origin in which the individual line elements are expressed in. 
Frame :math:`\mathcal{F}_0` is an arbitrary global axis, but it is usually observed as the vessel reference origin.

.. _exploded_3d:

.. figure:: nstatic/3dProfileExploded2.png
    :align: center
    :width: 60%

    Fig. 2

    .. raw:: html

	<font size="2"><center><i><b>
	Exploded 3D multisegemented line.
	</b></i></center></font>

.. Note::
   Simplistic way to think of MAP++'s dichotomy between nodes and elements:
   Nodes define the force at connection points. 
   Elements define the mooring geometry.

Clearly, this process requires two distinct sets of equations, one of which must be solved within the other routine, to find the static cable configuration. 
The first set of equations are the force{balance relationships in three directions for each node; the second set of equations are the catenary functions proportional to the number of elements. 
Interactions between solves is captured in the :ref:`flowchart below to summarize the solve procedure <nested_flow>`. This method was first proposed in :cite:`peyrot1979`.

.. _nested_flow:

.. figure:: nstatic/nested_flowchart.png
    :align: center
    :width: 60%

    Fig. 3

    .. raw:: html

	<font size="2"><center><i><b>Partitioned approach to solve the multi-segmented, quasi-static problem.</b></i></center></font>

Line Theory
-----------

Free--Hanging Line
~~~~~~~~~~~~~~~~~~

The equations used to describe the shape of a suspended chain illustrated in :ref:`single_line` have been derived in numerous works :cite:`irvine1992`. 
For completeness, a summary of the governing equations used inside the MSQS model are presented. 
Given a set of line properties, the line geometry can be expressed as a function of the forces exerted at the end of the line:

.. math::
   \scriptsize{x\left ( s \right ) = \frac{H}{\omega}\left \{ \ln\left [ \frac{V_{a} + \omega s}{H} + \sqrt{1 + \left ( \frac{V_{a} + \omega s}{H} \right )^{2}} \right ] - \ln \left [ \frac{V_{a}}{H} + \sqrt{1 + \left ( \frac{V_{a}}{H} \right )^{2} } \right ] \right \} + \frac{Hs}{EA}}

.. math::
   \scriptsize{z \left ( s \right ) = \frac{H}{\omega} \left [ \sqrt{ 1 + \left ( \frac{V_{a} + \omega s}{H} \right )^{2} } - \sqrt{ 1 + \left ( \frac{V_{a} }{H} \right )^{2} } \right ] + \frac{1}{EA}\left ( V_{a} s + \frac{\omega s^{2}}{2} \right )}

where:

.. math::
   \scriptsize{\omega = gA\left ( \rho_{\text{cable}}-\rho \right )}

and :math:`x` and :math:`z` are coordinate axes in the local (element) frame, :ref:`exploded_3d`. 
The following substitution can be made for :math:`V_a` in the above equations:

.. math::
   \scriptsize{H_{a} = H}

.. math::
   \scriptsize{V_{a} = V-\omega L}

which simply states the decrease in the vertical anchor force component is proportional to the mass of the suspended line. 
The equations for :math:`x(s)` and :math:`z(s)` both describe the catenary profile provided all entries on the right side of the equations are known. 
However, in practice, the force terms :math:`H` and :math:`V` are sought, and the known entity is the fairlead excursion dimensions, :math:`l` and :math:`l`. 
In this case, the forces :math:`H` and :math:`V` are found by simultaneously solving the following two equations:

.. _eq_horizontal:

.. math:: 
   \scriptsize{l = \frac{H}{\omega} \left [  \ln\left ( \frac{V}{H} +\sqrt{1+\left ( \frac{V}{H} \right )^{2}}\right )- \ln\left ( \frac{V-\omega L}{H} + \sqrt{1+ \left ( \frac{V-\omega L}{H}  \right )^{2}}\right ) \right ] + \frac{HL}{EA}}

.. math::
   \scriptsize{h = \frac{H}{\omega} \left [ \sqrt{1 + \left ( \frac{V}{H} \right )^{2} } - \sqrt{1 + \left ( \frac{V - \omega L}{H} \right )^{2} } \right ] + \frac{1}{EA}\left ( VL - \frac{\omega L^{2}}{2} \right )}

.. _single_line:

.. figure:: nstatic/singleLineDefinition.png
    :align: center
    :width: 60%

    Fig. 4

    .. raw:: html

	<font size="2"><center><i><b>
	Single line definitions for a hanging catenary.
	</b></i></center></font>
		     
Line Touching the Bottom
~~~~~~~~~~~~~~~~~~~~~~~~
The solution for the line in contact with a bottom boundary is found by continuing :math:`x(s)` and :math:`z(s)` beyond the seabed touch--down point :math:`s=L_{B}`.
Integration constants are added to ensure continuity of boundary conditions between equations:

.. math::

   \scriptsize{ x\left ( s \right ) = 
   \left\{\begin{matrix}
   s & \text{if } 0 \leq s \leq x_{0}
   \\ 
   \\ 
   s + \frac{C_{B}\omega}{2EA}\left [ s^{2} - 2x_{0}s + x_{0}\lambda \right ] & \text{if } x_{0}  < s \leq L_{B} 
   \\ 
   \\ 
   \begin{matrix}
   L_{B} + \frac{H}{\omega} \ln \left [ \frac{\omega\left ( s-L_{B} \right )}{H} + \sqrt{1 + \left ( \frac{\omega\left ( s-L_{B} \right )}{H} \right )^{2}} \right ] + \frac{Hs}{EA} +\frac{C_{B}\omega}{2EA}\left [ x_{0}\lambda - L_{B}^{2} \right ]
   \end{matrix} & \text{if } L_{B} < s \leq L 
   \\ 
   \end{matrix}\right.}

where :math:`\lambda` is:

.. math::
   \scriptsize{\lambda = \left\{\begin{matrix}
   L_{B} - \frac{H}{C_{B}\omega} & \text{if } x_{0} > 0
   \\ 
   \\ 
   0 &\text{otherwise }
   \end{matrix}\right.}
  
Between the range :math:`0\leq s \leq L_{B}`, the vertical height is zero since the line is resting on the seabed and forces can only occur parallel to the horizontal plane. 
This produces:

.. math::
   \scriptsize{z\left ( s \right ) = \left\{\begin{matrix}
   0 & \text{if } 0 \leq s \leq L_{B}
   \\ 
   \\
   \frac{H}{\omega}\left [ \sqrt{1 + \left ( \frac{\omega \left ( s - L_{B} \right )}{H} \right )^{2} } - 1\right ] + \frac{\omega \left ( s - L_{B} \right )^{2} }{2EA} & \text{if } L_{B} < s \leq L
   \end{matrix}\right.}

The equations above produce the mooring line profile as a function of :math:`s`. 
Ideally, a closed--form solution for :math:`l` and :math:`h` is sought to permit simultaneous solves for :math:`H` and :math:`V`, analogous to the freely--hanging chase in the previous section. 
This is obtained by substituting :math:`s=L` to give:

.. math::
   \scriptsize{l = L_{B} + \left (\frac{H}{\omega}  \right ) \ln\left [ \frac{V}{H} + \sqrt{1+\left ( \frac{V}{H} \right )^{2}} \right ] + \frac{HL}{EA} + \frac{C_{B}\omega}{2EA}\left [ x_{0}\lambda - L_{B}^{2} \right ]}

.. math::
   \scriptsize{h = \frac{H}{\omega}\left [ \sqrt{1 + \left (  \frac{V}{H} \right )^{2} } - 1 \right ] + \frac{V^{2}}{2EA\omega}}

Finally, a useful quantity that is often evaluated is the tension as a function of :math:`s` along the line. 
This is given using:

.. math::
   \scriptsize{T_{e} \left ( s \right ) = \left\{\begin{matrix}
   \text{MAX} \left [ H+C_{B}\omega \left ( s-L_{B} \right ) \;,\; 0 \right ] & \text{if }0 \leq s\leq L_{B}
   \\
   \\
   \sqrt{H^{2}+\left [ \omega\left ( s-L_{B} \right ) \right ]^{2}} &\text{if } L_{B} < s \leq L
   \end{matrix}\right.}

.. figure:: nstatic/singleLineDefinition2.png
    :align: center
    :width: 70%

    Fig. 5
    
    .. raw:: html

	<font size="2"><center><i><b>
	Single line definitions for a catenary touching a bottom boundary with friction.
	</b></i></center></font>


Vessel
------

Reference Origin
~~~~~~~~~~~~~~~~

.. math::
   \mathbf{R} = \begin{bmatrix} \cos\psi \cos\theta   & \cos\psi \sin\theta \sin\phi - \sin\psi \cos\phi & \cos\psi  \sin\theta \cos\phi   + \sin\psi \sin\phi    
   \\ 
   \sin\psi \cos\theta & \sin\phi \sin\theta \sin\phi + \cos\psi \cos\phi & \sin\psi \sin\theta \cos\phi - \cos\psi \sin\phi   
   \\ 
   -\sin \theta   & \cos\theta \sin\phi   & \cos\theta \cos\phi   \end{bmatrix}



.. figure:: nstatic/vessel.png
    :align: center
    :width: 90%

    Fig. 6
    
    .. raw:: html

	<font size="2"><center><i><b>
	Vessel kinematic breakdown to describe fairlead position relative to the origin.
	</b></i></center></font>
