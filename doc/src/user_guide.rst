.. highlight:: none

**********
User Guide
**********

For information on how to install PyFR see :ref:`installation`.

.. _running-pyfr:

Running PyFR
============

PyFR |release| uses three distinct file formats:

1. ``.ini`` --- configuration file
2. ``.pyfrm`` --- mesh file
3. ``.pyfrs`` --- solution file

The following commands are available from the ``pyfr`` program:

1. ``pyfr import`` --- convert a `Gmsh
   <http:http://geuz.org/gmsh/>`_ .msh file into a PyFR .pyfrm file.

   Example::

        pyfr import mesh.msh mesh.pyfrm

2. ``pyfr partition`` --- partition or repartition an existing mesh and
   associated solution files.

   Example::

       pyfr partition 2 mesh.pyfrm solution.pyfrs outdir/

   Here, the newly partitioned mesh and solution are placed into the
   directory `outdir`.  Multiple solutions can be provided.
   Time-average files can also be partitioned, too.

   For mixed grids one must include the ``-e`` flag followed by weights
   for each element type, or the ``balanced`` argument. Further details
   can be found in the :ref:`performance guide <perf mixed grids>`.

3. ``pyfr run`` --- start a new PyFR simulation. Example::

        pyfr run mesh.pyfrm configuration.ini

4. ``pyfr restart`` --- restart a PyFR simulation from an existing
   solution file. Example::

        pyfr restart mesh.pyfrm solution.pyfrs

   It is also possible to restart with a different configuration file.
   Example::

        pyfr restart mesh.pyfrm solution.pyfrs configuration.ini

5. ``pyfr export`` --- convert a PyFR ``.pyfrs`` file into an
   unstructured VTK ``.vtu`` or ``.pvtu`` file. If a ``-k`` flag is
   provided with an integer argument then ``.pyfrs`` elements are
   converted to high-order VTK cells which are exported, where the
   order of the VTK cells is equal to the value of the integer
   argument. Example::

        pyfr export -k 4 mesh.pyfrm solution.pyfrs solution.vtu

   If a ``-d`` flag is provided with an integer argument then
   ``.pyfrs`` elements are subdivided into linear VTK cells which are
   exported, where the number of sub-divisions is equal to the value of
   the integer argument. Example::

        pyfr export -d 4 mesh.pyfrm solution.pyfrs solution.vtu

   If no flags are provided then ``.pyfrs`` elements are converted to
   high-order VTK cells which are exported, where the order of the
   cells is equal to the order of the solution data in the ``.pyfrs``
   file.

   By default all of the fields in the ``.pyfrs`` file will be
   exported. If only a specific field is desired this can be specified
   with the ``-f`` flag; for example ``-f density -f velocity`` will
   only export the *density* and *velocity* fields.

PyFR can be run in parallel. To do so prefix ``pyfr`` with
``mpiexec -n <cores/devices>``. Note that the mesh must be
pre-partitioned, and the number of cores or devices must be equal to
the number of partitions.

.. _configuration-file:

Configuration File (.ini)
=========================

The .ini configuration file parameterises the simulation. It is written
in the `INI <http://en.wikipedia.org/wiki/INI_file>`_ format.
Parameters are grouped into sections. The roles of each section and
their associated parameters are described below. Note that both ``;`` and
``#`` may be used as comment characters.  Additionally, all parameter
values support environment variable expansion.

Backends
--------

These sections detail how the solver will be configured for a range of
different hardware platforms. If a hardware specific backend section is omitted,
then PyFR will fall back to built-in default settings.

.. toctree::
   :maxdepth: 3

   backends/backend.rst
   backends/backend-cuda.rst
   backends/backend-hip.rst
   backends/backend-metal.rst
   backends/backend-opencl.rst
   backends/backend-openmp.rst

Systems
-------

These sections setup and control the physical system being solved, as well as
charateristics of the spatial and temporal schemes to be used.

.. toctree::
   :maxdepth: 3

        - ``scheme`` --- time-integration scheme

           ``backward-euler`` | ``sdirk33`` | ``sdirk43``

        - ``pseudo-scheme`` --- pseudo time-integration scheme

           ``euler`` | ``rk34`` | ``rk4`` | ``rk45`` | ``tvd-rk3`` | ``vermeire``

        - ``tstart`` --- initial time

           *float*

        - ``tend`` --- final time

           *float*

        - ``dt`` --- time-step

           *float*

        - ``controller`` --- time-step controller

           ``none``

        - ``pseudo-dt`` --- pseudo time-step

           *float*

        - ``pseudo-niters-max`` --- minimum number of iterations

           *int*

        - ``pseudo-niters-min`` --- maximum number of iterations

           *int*

        - ``pseudo-resid-tol`` --- pseudo residual tolerance

           *float*

        - ``pseudo-resid-norm`` --- pseudo residual norm

           ``uniform`` | ``l2``

        - ``pseudo-controller`` --- pseudo time-step controller

           ``none`` | ``local-pi``

           where

           ``local-pi`` only works with ``rk34`` and ``rk45`` and
           requires

            - ``atol`` --- absolute error tolerance

               *float*

            - ``safety-fact`` --- safety factor for pseudo time-step
              size adjustment (suitable range 0.80-0.95)

               *float*

            - ``min-fact`` --- minimum factor by which the local
              pseudo time-step can change between iterations
              (suitable range 0.98-0.998)

               *float*

            - ``max-fact`` --- maximum factor by which the local
              pseudo time-step can change between iterations
              (suitable range 1.001-1.01)

               *float*

            - ``pseudo-dt-min-mult`` --- minimum permissible
              local pseudo time-step given as a
              multiplier of ``pseudo-dt`` (suitable range 0.001-1.0)

               *float*

            - ``pseudo-dt-max-mult`` --- maximum permissible
              local pseudo time-step given as a
              multiplier of ``pseudo-dt`` (suitable range 2.0-5.0)

               *float*

2. ``dt-adjust-min-fact`` --- minimum allowed factor by which the 
   time-step modified by controller can be further changed to 
   satisfy the constraints set by the target time
   (suitable range 0.5-0.99)

    *float*

2. ``dt-adjust-max-fact`` --- maximum allowed factor by which the 
   time-step modified by controller can be further changed to 
   satisfy the constraints set by the target time
    (suitable range 1.0-1.1)

    *float*

Example::

    [solver-time-integrator]
    formulation = std
    scheme = rk45
    controller = pi
    tstart = 0.0
    tend = 10.0
    dt = 0.001
    atol = 0.00001
    rtol = 0.00001
    errest-norm = l2
    safety-fact = 0.9
    min-fact = 0.3
    max-fact = 2.5
    dt-adjust-min-fact = 0.99
    dt-adjust-max-fact = 1.01

[solver-dual-time-integrator-multip]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises multi-p for dual time-stepping with

1. ``pseudo-dt-fact`` --- factor by which the pseudo time-step size
   changes between multi-p levels:

    *float*

2. ``cycle`` --- nature of a single multi-p cycle:

    ``[(order, nsteps), (order, nsteps), ... (order, nsteps)]``

    where ``order`` in the first and last bracketed pair must be the
    overall polynomial order used for the simulation, ``order`` can
    only change by one between subsequent bracketed pairs, and
    ``nsteps`` is a non-negative rational number.

Example::

    [solver-dual-time-integrator-multip]
    pseudo-dt-fact = 2.3
    cycle = [(3, 0.1), (2, 0.1), (1, 0.2), (0, 1.4), (1, 1.1), (2, 1.1), (3, 4.5)]

[solver-interfaces]
^^^^^^^^^^^^^^^^^^^

Parameterises the interfaces with

1. ``riemann-solver`` --- type of Riemann solver:

    ``rusanov`` | ``hll`` | ``hllc`` | ``roe`` | ``roem`` | ``exact``

    where

    ``hll`` | ``hllc`` | ``roe`` | ``roem`` | ``exact`` do not work with
    ``ac-euler`` | ``ac-navier-stokes``

2. ``ldg-beta`` --- beta parameter used for LDG:

    *float*

3. ``ldg-tau`` --- tau parameter used for LDG:

    *float*

Example::

    [solver-interfaces]
    riemann-solver = rusanov
    ldg-beta = 0.5
    ldg-tau = 0.1

[solver-entropy-filter]
^^^^^^^^^^^^^^^^^^^^^^^

Parameterises entropy filter for shock capturing with

1. ``d-min`` --- minimum allowable density:

    *float*

2. ``p-min`` --- minimum allowable pressure:

    *float*

3. ``e-tol`` --- entropy tolerance:

    *float*

Example::

    [solver-entropy-filter]
    d-min = 1e-6
    p-min = 1e-6
    e-tol = 1e-6

[solver-artificial-viscosity]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises artificial viscosity for shock capturing with

1. ``max-artvisc`` --- maximum artificial viscosity:

    *float*

2. ``s0`` --- sensor cut-off:

    *float*

3. ``kappa`` --- sensor range:

    *float*

Example::

    [solver-artificial-viscosity]
    max-artvisc = 0.01
    s0 = 0.01
    kappa = 5.0

[soln-filter]
^^^^^^^^^^^^^

Parameterises an exponential solution filter with

1. ``nsteps`` --- apply filter every ``nsteps``:

    *int*

2. ``alpha`` --- strength of filter:

    *float*

3. ``order`` --- order of filter:

    *int*

4. ``cutoff`` --- cutoff frequency below which no filtering is applied:

    *int*

Example::

    [soln-filter]
    nsteps = 10
    alpha = 36.0
    order = 16
    cutoff = 1

Boundary and Initial Conditions
-------------------------------

These sections allow users to set the boundary and initial
conditions of calculations.

.. toctree::
   :maxdepth: 3

   boundary-initial-conditions/soln-bcs.rst
   boundary-initial-conditions/soln-ics.rst

Nodal Point Sets
----------------

Solution point sets must be specified for each element type that is used and
flux point sets must be specified for each interface type that is used. If
anti-aliasing is enabled then quadrature point sets for each element and
interface type that is used must also be specified. For example, a 3D mesh
comprised only of prisms requires a solution point set for prism elements and
flux point sets for quadrilateral and triangular interfaces.

.. toctree::
   :maxdepth: 3

   nodal-point-sets/solver-interfaces-line.rst
   nodal-point-sets/solver-interfaces-quad.rst
   nodal-point-sets/solver-interfaces-tri.rst
   nodal-point-sets/solver-elements-quad.rst
   nodal-point-sets/solver-elements-tri.rst
   nodal-point-sets/solver-elements-hex.rst
   nodal-point-sets/solver-elements-tet.rst
   nodal-point-sets/solver-elements-pri.rst
   nodal-point-sets/solver-elements-pyr.rst

Plugins
-------

Plugins allow for powerful additional functionality to be swapped
in and out. There are two classes of plugin available; solution
plugins which are prefixed by ``soln-`` and solver plugins which
are prefixed by ``solver-``. It is possible to create multiple
instances of the same solution plugin by appending a suffix, for
example::

    [soln-plugin-writer]
    ...

    [soln-plugin-writer-2]
    ...

    [soln-plugin-writer-three]
    ...

Certain plugins also expose functionality via a CLI, which can
be invoked independently of a PyFR run.

Solution Plugins
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 3

   plugins/soln-plugin-ascent.rst
   plugins/soln-plugin-dtstats.rst
   plugins/soln-plugin-fluid-force.rst
   plugins/soln-plugin-fwh.rst
   plugins/soln-plugin-integrate.rst
   plugins/soln-plugin-nancheck.rst
   plugins/soln-plugin-pseudostats.rst
   plugins/soln-plugin-residual.rst
   plugins/soln-plugin-sampler.rst
   plugins/soln-plugin-tavg.rst
   plugins/soln-plugin-writer.rst

Solver Plugins
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 3

   plugins/solver-plugin-source.rst
   plugins/solver-plugin-turbulence.rst

Regions
^^^^^^^

Certain plugins are capable of performing operations on a subset of the
elements inside the domain. One means of constructing these element
subsets is through parameterised regions. Note that an element is
considered part of a region if *any* of its nodes are found to be
contained within the region. Supported regions:

Rectangular cuboid ``box(x0, x1)``
  A rectangular cuboid defined by two diametrically opposed vertices.
  Valid in both 2D and 3D.

Conical frustum ``conical_frustum(x0, x1, r0, r1)``
  A conical frustum whose end caps are at *x0* and *x1* with radii
  *r0* and *r1*, respectively. Only valid in 3D.

Cone ``cone(x0, x1, r)``
  A cone of radius *r* whose centre-line is defined by *x0* and *x1*.
  Equivalent to ``conical_frustum(x0, x1, r, 0)``. Only valid in 3D.

Cylinder ``cylinder(x0, x1, r)``
  A circular cylinder of radius *r* whose centre-line is defined by
  *x0* and *x1*. Equivalent to ``conical_frustum(x0, x1, r, r)``.
  Only valid in 3D.

Cartesian ellipsoid ``ellipsoid(x0, a, b, c)``
  An ellipsoid centred at *x0* with Cartesian coordinate axes whose
  extents in the *x*, *y*, and *z* directions are given by *a*, *b*,
  and *c*, respectively. Only valid in 3D.

Sphere ``sphere(x0, r)``
  A sphere centred at *x0* with a radius of *r*. Equivalent to
  ``ellipsoid(x0, r, r, r)``. Only valid in 3D.

All region shapes also support rotation.  In 2D this is accomplished by
passing a trailing `rot=angle` argument where `angle` is a rotation
angle in degrees; for example ``box((-5, 2), (2, 0), rot=30)``.
In 3D the syntax is `rot=(phi, theta, psi)` and corresponds to a
sequence of Euler angles in the so-called *ZYX convention*.  Region
expressions can also be added and subtracted together  arbitrarily.
For example
``box((-10, -10, -10), (10, 10, 10)) - sphere((0, 0, 0), 3)`` will
result in a cube-shaped region with a sphere cut out of the middle.

Additional Information
----------------------

The :ref:`INI<configuration-file>` file format is very versatile. A feature that
can be useful in defining initial conditions is the substitution feature and
this is demonstrated in the :ref:`soln-plugin-integrate` plugin example.

To prevent situations where you have solutions files for unknown
configurations, the contents of the ``.ini`` file are added as an attribute
to ``.pyfrs`` files. These files use the HDF5 format and can be
straightforwardly probed with tools such as h5dump.

In several places within the ``.ini`` file expressions may be used. As well as
the constant ``pi``, expressions containing the following functions are
supported:

1. ``+, -, *, /`` --- basic arithmetic

2. ``sin, cos, tan`` --- basic trigonometric functions (radians)

3. ``asin, acos, atan, atan2`` --- inverse trigonometric functions

4. ``exp, log`` --- exponential and the natural logarithm

5. ``tanh`` --- hyperbolic tangent

6. ``pow`` --- power, note ``**`` is not supported

7. ``sqrt`` --- square root

8. ``abs`` --- absolute value

9. ``min, max`` --- two variable minimum and maximum functions,
   arguments can be arrays
