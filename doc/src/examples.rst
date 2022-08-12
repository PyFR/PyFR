.. highlight:: none

********
Examples
********

Test cases are available in the
`PyFR-Test-Cases <https://github.com/PyFR/PyFR-Test-Cases>` 
repository. It is important to note, however, that these examples 
are all relatively small 2D simulations and, as such, are *not* 
suitable for scalability or performance studies.

Euler Equations
===============

2D Euler Vortex
---------------

Proceed with the following steps to run a parallel 2D Euler vortex
simulation on a structured mesh:

1. Navigate to the ``PyFR-Test-Cases/2d-euler-vortex`` directory::

        cd PyFR-Test-Cases/2d-euler-vortex

2. Run pyfr to convert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``2d-euler-vortex.pyfrm``::

        pyfr import 2d-euler-vortex.msh 2d-euler-vortex.pyfrm

3. Run pyfr to partition the PyFR mesh file into two pieces::

        pyfr partition 2 2d-euler-vortex.pyfrm .

4. Run pyfr to solve the Euler equations on the mesh, generating a
   series of PyFR solution files called ``2d-euler-vortex*.pyfrs``::

        mpiexec -n 2 pyfr run -b cuda -p 2d-euler-vortex.pyfrm 2d-euler-vortex.ini

5. Run pyfr on the solution file ``2d-euler-vortex-100.0.pyfrs``
   converting it into an unstructured VTK file called
   ``2d-euler-vortex-100.0.vtu``::

        pyfr export 2d-euler-vortex.pyfrm 2d-euler-vortex-100.0.pyfrs 2d-euler-vortex-100.0.vtu

6. Visualise the unstructured VTK file in `Paraview
   <http://www.paraview.org/>`_

.. figure:: ../fig/2d-euler-vortex/2d-euler-vortex.png
   :width: 450px
   :figwidth: 450px
   :alt: euler vortex
   :align: center

   Colour map of density distribution at 100 time units.

Compressible Navier--Stokes Equations
=====================================

2D Couette Flow
---------------

Proceed with the following steps to run a serial 2D Couette flow
simulation on a mixed unstructured mesh:

1. Navigate to the ``PyFR-Test-Cases/2d-couette-flow`` directory::

        cd PyFR-Test-Cases/2d-couette-flow

2. Run pyfr to covert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``2d-couette-flow.pyfrm``::

        pyfr import 2d-couette-flow.msh 2d-couette-flow.pyfrm

3. Run pyfr to solve the Navier-Stokes equations on the mesh,
   generating a series of PyFR solution files called
   ``2d-couette-flow-*.pyfrs``::

        pyfr run -b cuda -p 2d-couette-flow.pyfrm 2d-couette-flow.ini

4. Run pyfr on the solution file ``2d-couette-flow-040.pyfrs``
   converting it into an unstructured VTK file called
   ``2d-couette-flow-040.vtu``::

        pyfr export 2d-couette-flow.pyfrm 2d-couette-flow-040.pyfrs 2d-couette-flow-040.vtu

5. Visualise the unstructured VTK file in `Paraview
   <http://www.paraview.org/>`_

.. figure:: ../fig/2d-couette-flow/2d-couette-flow.png
   :width: 450px
   :figwidth: 450px
   :alt: couette flow
   :align: center

   Colour map of steady-state density distribution.

Incompressible Navier--Stokes Equations
=======================================

2D Incompressible Cylinder Flow
-------------------------------

Proceed with the following steps to run a serial 2D incompressible cylinder
flow simulation on a mixed unstructured mesh:

1. Navigate to the ``PyFR-Test-Cases/2d-inc-cylinder`` directory::

        cd PyFR-Test-Cases/2d-inc-cylinder
        
2. Run pyfr to covert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``2d-inc-cylinder.pyfrm``::

        pyfr import 2d-inc-cylinder.msh 2d-inc-cylinder.pyfrm

3. Run pyfr to solve the incompressible Navier-Stokes equations on the mesh,
   generating a series of PyFR solution files called
   ``2d-inc-cylinder-*.pyfrs``::

        pyfr run -b cuda -p 2d-inc-cylinder.pyfrm 2d-inc-cylinder.ini

4. Run pyfr on the solution file ``2d-inc-cylinder-75.00.pyfrs``
   converting it into an unstructured VTK file called
   ``2d-inc-cylinder-75.00.vtu``::

        pyfr export 2d-inc-cylinder.pyfrm 2d-inc-cylinder-75.00.pyfrs 2d-inc-cylinder-75.00.vtu

5. Visualise the unstructured VTK file in `Paraview
   <http://www.paraview.org/>`_

.. figure:: ../fig/2d-inc-cylinder/2d-inc-cylinder.png
   :width: 450px
   :figwidth: 450px
   :alt: couette flow
   :align: center

   Colour map of velocity magnitude distribution at 75 time units.
