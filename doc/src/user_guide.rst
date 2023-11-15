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

   For mixed grids it is usually necessary to provide weights for each
   element type.  Further details can be found in the
   :ref:`performance guide <perf mixed grids>`.

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

Running in Parallel
-------------------

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

The backend sections detail how the solver will be configured for a range of
different hardware platforms. If a hardware specific backend section is omitted,
then PyFR will fall back to built-in default settings.

[backend]
^^^^^^^^^

Parameterises the backend with

1. ``precision`` --- number precision:

    ``single`` | ``double``

2. ``rank-allocator`` --- MPI rank allocator:

    ``linear`` | ``random``

3. ``collect-wait-times`` --- If to track MPI request wait times or not:

    ``True`` | ``False``

4. ``collect-wait-times-len`` --- Size of the wait time history buffer:

     *int*

Example::

    [backend]
    precision = double
    rank-allocator = linear

[backend-cuda]
^^^^^^^^^^^^^^

Parameterises the CUDA backend with

1. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``round-robin`` | ``local-rank`` | ``uuid``

2. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``cuda-aware``

3. ``cflags`` --- additional NVIDIA realtime compiler (``nvrtc``) flags:

    *string*

Example::

    [backend-cuda]
    device-id = round-robin
    mpi-type = standard

[backend-hip]
^^^^^^^^^^^^^

Parameterises the HIP backend with

1. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``local-rank`` | ``uuid``

2. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``hip-aware``

Example::

    [backend-hip]
    device-id = local-rank
    mpi-type = standard

[backend-opencl]
^^^^^^^^^^^^^^^^

Parameterises the OpenCL backend with

1. ``platform-id`` --- for selecting platform id:

    *int* | *string*

2. ``device-type`` --- for selecting what type of device(s) to run on:

    ``all`` | ``cpu`` | ``gpu`` | ``accelerator``

3. ``device-id`` --- for selecting which device(s) to run on:

    *int* | *string* | ``local-rank`` | ``uuid``

4. ``gimmik-max-nnz`` --- cutoff for GiMMiK in terms of the number of
   non-zero entires in a constant matrix:

     *int*

Example::

    [backend-opencl]
    platform-id = 0
    device-type = gpu
    device-id = local-rank
    gimmik-max-nnz = 512

[backend-openmp]
^^^^^^^^^^^^^^^^

Parameterises the OpenMP backend with

1. ``cc`` --- C compiler:

    *string*

2. ``cflags`` --- additional C compiler flags:

    *string*

3. ``alignb`` --- alignment requirement in bytes; must be a power of
   two and at least 32:

    *int*

4. ``schedule`` --- OpenMP loop scheduling scheme:

    ``static`` | ``dynamic`` | ``dynamic, n`` | ``guided`` | ``guided, n``

    where *n* is a positive integer.

Example::

    [backend-openmp]
    cc = gcc

Systems
-------

These sections of the input file setup and control the physical system being
solved, as well as charateristics of the spatial and temporal schemes to be
used.

[constants]
^^^^^^^^^^^

Sets constants used in the simulation

1. ``gamma`` --- ratio of specific heats for ``euler`` |
   ``navier-stokes``:

    *float*

2. ``mu`` --- dynamic viscosity for ``navier-stokes``:

    *float*

3. ``nu`` --- kinematic viscosity for ``ac-navier-stokes``:

    *float*

4. ``Pr`` --- Prandtl number for ``navier-stokes``:

    *float*

5. ``cpTref`` --- product of specific heat at constant pressure and
   reference temperature for ``navier-stokes`` with Sutherland's Law:

   *float*

6. ``cpTs`` --- product of specific heat at constant pressure and
   Sutherland temperature for ``navier-stokes`` with Sutherland's Law:

   *float*

7. ``ac-zeta`` --- artificial compressibility factor for ``ac-euler`` |
   ``ac-navier-stokes``

   *float*

Other constant may be set by the user which can then be used throughout the
``.ini`` file.

Example::

    [constants]
    ; PyFR Constants
    gamma = 1.4
    mu = 0.001
    Pr = 0.72

    ; User Defined Constants
    V_in = 1.0
    P_out = 20.0

[solver]
^^^^^^^^

Parameterises the solver with

1. ``system`` --- governing system:

    ``euler`` | ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``

    where

    ``euler`` requires

        - ``shock-capturing`` --- shock capturing scheme:

          ``none`` | ``entropy-filter``

    ``navier-stokes`` requires

        - ``viscosity-correction`` --- viscosity correction:

          ``none`` | ``sutherland``

        - ``shock-capturing`` --- shock capturing scheme:

          ``none`` | ``artificial-viscosity`` | ``entropy-filter``

2. ``order`` --- order of polynomial solution basis:

    *int*

3. ``anti-alias`` --- type of anti-aliasing:

    ``flux`` | ``surf-flux`` | ``flux, surf-flux``

Example::

    [solver]
    system = navier-stokes
    order = 3
    anti-alias = flux
    viscosity-correction = none
    shock-capturing = entropy-filter

[solver-time-integrator]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the time-integration scheme used by the solver with

1. ``formulation`` --- formulation:

    ``std`` | ``dual``

    where

    ``std`` requires

        - ``scheme`` --- time-integration scheme

           ``euler`` | ``rk34`` | ``rk4`` | ``rk45`` | ``tvd-rk3``

        - ``tstart`` --- initial time

           *float*

        - ``tend`` --- final time

           *float*

        - ``dt`` --- time-step

           *float*

        - ``controller`` --- time-step controller

           ``none`` | ``pi``

           where

           ``pi`` only works with ``rk34`` and ``rk45`` and requires

            - ``atol`` --- absolute error tolerance

               *float*

            - ``rtol`` --- relative error tolerance

               *float*

            - ``errest-norm`` --- norm to use for estimating the error

               ``uniform`` | ``l2``

            - ``safety-fact`` --- safety factor for step size adjustment
              (suitable range 0.80-0.95)

               *float*

            - ``min-fact`` --- minimum factor by which the time-step can
              change between iterations (suitable range 0.1-0.5)

               *float*

            - ``max-fact`` --- maximum factor by which the time-step can
              change between iterations (suitable range 2.0-6.0)

               *float*

            - ``dt-max`` --- maximum permissible time-step

               *float*

    ``dual`` requires

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

            - ``pseudo-dt-max-mult`` --- maximum permissible
              local pseudo time-step given as a
              multiplier of ``pseudo-dt`` (suitable range 2.0-5.0)

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

[soln-bcs-*name*]
^^^^^^^^^^^^^^^^^

Parameterises constant, or if available space (x, y, [z]) and time (t)
dependent, boundary condition labelled *name* in the .pyfrm file with

1. ``type`` --- type of boundary condition:

    ``ac-char-riem-inv`` | ``ac-in-fv`` | ``ac-out-fp`` | ``char-riem-inv`` |
    ``no-slp-adia-wall`` | ``no-slp-isot-wall`` | ``no-slp-wall`` |
    ``slp-adia-wall`` | ``slp-wall`` | ``sub-in-frv`` |
    ``sub-in-ftpttang`` | ``sub-out-fp`` | ``sup-in-fa`` |
    ``sup-out-fn``

    where

    ``ac-char-riem-inv`` only works with ``ac-euler`` |
    ``ac-navier-stokes`` and requires

        - ``ac-zeta`` --- artificial compressibility factor for boundary
          (increasing ``ac-zeta`` makes the boundary less reflective
          allowing larger deviation from the target state)

           *float*

        - ``niters`` --- number of Newton iterations

           *int*

        - ``p`` --- pressure

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*


    ``ac-in-fv`` only works with ``ac-euler`` | ``ac-navier-stokes`` and
    requires

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

    ``ac-out-fp`` only works with ``ac-euler`` | ``ac-navier-stokes`` and
    requires

        - ``p`` --- pressure

           *float* | *string*

    ``char-riem-inv`` only works with ``euler`` | ``navier-stokes`` and
    requires

        - ``rho`` --- density

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

        - ``p`` --- static pressure

           *float* | *string*

    ``no-slp-adia-wall`` only works with ``navier-stokes``

    ``no-slp-isot-wall`` only works with ``navier-stokes`` and requires

        - ``u`` --- x-velocity of wall

           *float*

        - ``v`` --- y-velocity of wall

           *float*

        - ``w`` --- z-velocity of wall

           *float*

        - ``cpTw`` --- product of specific heat capacity at constant
          pressure and temperature of wall

           *float*

    ``no-slp-wall`` only works with ``ac-navier-stokes`` and requires

        - ``u`` --- x-velocity of wall

           *float*

        - ``v`` --- y-velocity of wall

           *float*

        - ``w`` --- z-velocity of wall

           *float*

    ``slp-adia-wall`` only works with ``euler`` | ``navier-stokes``

    ``slp-wall`` only works with ``ac-euler`` | ``ac-navier-stokes``

    ``sub-in-frv`` only works with ``navier-stokes`` and
    requires

        - ``rho`` --- density

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

    ``sub-in-ftpttang`` only works with ``navier-stokes``
    and requires

        - ``pt`` --- total pressure

           *float*

        - ``cpTt`` --- product of specific heat capacity at constant
          pressure and total temperature

           *float*

        - ``theta`` --- azimuth angle (in degrees) of inflow measured
          in the x-y plane relative to the positive x-axis

           *float*

        - ``phi`` --- inclination angle (in degrees) of inflow measured
          relative to the positive z-axis

           *float*

    ``sub-out-fp`` only works with ``navier-stokes`` and
    requires

        - ``p`` --- static pressure

           *float* | *string*

    ``sup-in-fa`` only works with ``euler`` | ``navier-stokes`` and
    requires

        - ``rho`` --- density

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

        - ``p`` --- static pressure

           *float* | *string*

    ``sup-out-fn`` only works with ``euler`` | ``navier-stokes``

Example::

    [soln-bcs-bcwallupper]
    type = no-slp-isot-wall
    cpTw = 10.0
    u = 1.0

Simple periodic boundary conditions are supported; however, their behaviour
is not controlled through the ``.ini`` file, instead it is handled at
the mesh generation stage. Two faces may be taged with
``periodic_x_l`` and ``periodic_x_r``, where ``x`` is a unique
identifier for the pair of boundaries. Currently, only periodicity in a
single cardinal direction is supported, for example, the planes
``(x,y,0)``` and ``(x,y,10)``.

[soln-ics]
^^^^^^^^^^

Parameterises space (x, y, [z]) dependent initial conditions with

1. ``rho`` --- initial density distribution for ``euler`` |
   ``navier-stokes``:

    *string*

2. ``u`` --- initial x-velocity distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

3. ``v`` --- initial y-velocity distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

4. ``w`` --- initial z-velocity distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

5. ``p`` --- initial static pressure distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

Example::

    [soln-ics]
    rho = 1.0
    u = x*y*sin(y)
    v = z
    w = 1.0
    p = 1.0/(1.0+x)

Nodal Point Sets
----------------

Solution point sets must be specified for each element type that is used and
flux point sets must be specified for each interface type that is used. If
anti-aliasing is enabled then quadrature point sets for each element and
interface type that is used must also be specified. For example, a 3D mesh
comprised only of prisms requires a solution point set for prism elements and
flux point set for quadrilateral and triangular interfaces.

[solver-interfaces-line{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the line interfaces, or if -mg-p\ *order* is suffixed the
line interfaces at multi-p level *order*, with

1. ``flux-pts`` --- location of the flux points on a line interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing on a
   line interface:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing on a
   line interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

Example::

    [solver-interfaces-line]
    flux-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre

[solver-interfaces-tri{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the triangular interfaces, or if -mg-p\ *order* is
suffixed the triangular interfaces at multi-p level *order*, with

1. ``flux-pts`` --- location of the flux points on a triangular
   interface:

    ``williams-shunn``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing on a
   triangular interface:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing on a
   triangular interface:

    ``williams-shunn`` | ``witherden-vincent``

Example::

    [solver-interfaces-tri]
    flux-pts = williams-shunn
    quad-deg = 10
    quad-pts = williams-shunn

[solver-interfaces-quad{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the quadrilateral interfaces, or if -mg-p\ *order* is
suffixed the quadrilateral interfaces at multi-p level *order*, with

1. ``flux-pts`` --- location of the flux points on a quadrilateral
   interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing on a
   quadrilateral interface:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing on a
   quadrilateral interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-interfaces-quad]
    flux-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre

[solver-elements-tri{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the triangular elements, or if -mg-p\ *order* is suffixed
the triangular elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a triangular
   element:

    ``williams-shunn``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   triangular element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   triangular element:

    ``williams-shunn`` | ``witherden-vincent``

Example::

    [solver-elements-tri]
    soln-pts = williams-shunn
    quad-deg = 10
    quad-pts = williams-shunn

[solver-elements-quad{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the quadrilateral elements, or if -mg-p\ *order* is
suffixed the quadrilateral elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a quadrilateral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   quadrilateral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   quadrilateral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-elements-quad]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre

[solver-elements-hex{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the hexahedral elements, or if -mg-p\ *order* is suffixed
the hexahedral elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a hexahedral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   hexahedral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   hexahedral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-elements-hex]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre

[solver-elements-tet{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the tetrahedral elements, or if -mg-p\ *order* is suffixed
the tetrahedral elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a tetrahedral
   element:

    ``shunn-ham``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   tetrahedral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   tetrahedral element:

    ``shunn-ham`` | ``witherden-vincent``

Example::

    [solver-elements-tet]
    soln-pts = shunn-ham
    quad-deg = 10
    quad-pts = shunn-ham

[solver-elements-pri{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the prismatic elements, or if -mg-p\ *order* is suffixed
the prismatic elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a prismatic
   element:

    ``williams-shunn~gauss-legendre`` |
    ``williams-shunn~gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   prismatic element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   prismatic element:

    ``williams-shunn~gauss-legendre`` |
    ``williams-shunn~gauss-legendre-lobatto`` | ``witherden-vincent``

Example::

    [solver-elements-pri]
    soln-pts = williams-shunn~gauss-legendre
    quad-deg = 10
    quad-pts = williams-shunn~gauss-legendre

[solver-elements-pyr{-mg-p\ *order*}]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the pyramidal elements, or if -mg-p\ *order* is suffixed
the pyramidal elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a pyramidal
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   pyramidal element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   pyramidal element:

    ``witherden-vincent``

Example::

    [solver-elements-pyr]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = witherden-vincent

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

[soln-plugin-writer]
^^^^^^^^^^^^^^^^^^^^
Periodically write the solution to disk in the pyfrs format.
Parameterised with

1. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

2. ``basedir`` --- relative path to directory where outputs will be
   written:

    *string*

3. ``basename`` --- pattern of output names:

    *string*

4. ``post-action`` --- command to execute after writing the file:

    *string*

5. ``post-action-mode`` --- how the post-action command should be
   executed:

    ``blocking`` | ``non-blocking``

4. ``region`` --- region to be written, specified as either the
   entire domain using ``*``, a combination of the geometric shapes
   specified in :ref:`regions`, or a sub-region of elements that have
   faces on a specific domain boundary via the name of the domain
   boundary:

    ``*`` | ``shape(args, ...)`` | *string*

Example::

    [soln-plugin-writer]
    dt-out = 0.01
    basedir = .
    basename = files-{t:.2f}
    post-action = echo "Wrote file {soln} at time {t} for mesh {mesh}."
    post-action-mode = blocking
    region = box((-5, -5, -5), (5, 5, 5))

[soln-plugin-fluidforce-*name*]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Periodically integrates the pressure and viscous stress on the boundary
labelled ``name`` and writes out the resulting force and moment (if requested)
vectors to a CSV file. Parameterised with

1. ``nsteps`` --- integrate every ``nsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

4. ``morigin`` --- origin used to compute moments (optional):

    ``(x, y, [z])``

5. ``quad-deg-{etype}`` --- degree of quadrature rule for fluid force
   integration, optionally this can be specified for different element types:

    *int*

6. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

Example::

    [soln-plugin-fluidforce-wing]
    nsteps = 10
    file = wing-forces.csv
    header = true
    quad-deg = 6
    morigin = (0.0, 0.0, 0.5)

[soln-plugin-nancheck]
^^^^^^^^^^^^^^^^^^^^^^

Periodically checks the solution for NaN values. Parameterised with

1. ``nsteps`` --- check every ``nsteps``:

    *int*

Example::

    [soln-plugin-nancheck]
    nsteps = 10

[soln-plugin-residual]
^^^^^^^^^^^^^^^^^^^^^^

Periodically calculates the residual and writes it out to a CSV file.
Parameterised with

1. ``nsteps`` --- calculate every ``nsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

4. ``norm`` --- sets the degree and calculates an :math:`L_p` norm,
    default is ``2``:

    *float* | ``inf``

Example::

    [soln-plugin-residual]
    nsteps = 10
    file = residual.csv
    header = true
    norm = inf

[soln-plugin-dtstats]
^^^^^^^^^^^^^^^^^^^^^^

Write time-step statistics out to a CSV file. Parameterised with

1. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-dtstats]
    flushsteps = 100
    file = dtstats.csv
    header = true

[soln-plugin-pseudostats]
^^^^^^^^^^^^^^^^^^^^^^^^^

Write pseudo-step convergence history out to a CSV file. Parameterised
with

1. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-pseudostats]
    flushsteps = 100
    file = pseudostats.csv
    header = true

[soln-plugin-sampler]
^^^^^^^^^^^^^^^^^^^^^

Periodically samples specific points in the volume and writes them out
to a CSV file. Parameterised with

1. ``nsteps`` --- sample every ``nsteps``:

    *int*

2. ``samp-pts`` --- list of points to sample:

    ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]``

3. ``format`` --- output variable format:

    ``primitive`` | ``conservative``

4. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

5. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-sampler]
    nsteps = 10
    samp-pts = [(1.0, 0.7, 0.0), (1.0, 0.8, 0.0)]
    format = primitive
    file = point-data.csv
    header = true

[soln-plugin-tavg]
^^^^^^^^^^^^^^^^^^

Time average quantities. Parameterised with

1. ``nsteps`` --- accumulate the average every ``nsteps`` time steps:

    *int*

2. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

3. ``tstart`` --- time at which to start accumulating average data:

    *float*

4. ``mode`` --- output file accumulation mode:

    ``continuous`` | ``windowed``

    In continuous mode each output file contains average data from
    ``tstart`` until the current time. In windowed mode each output
    file only contains average data for the most recent ``dt-out`` time
    units. The default is ``windowed``.

5. ``std-mode`` --- standard deviation reporting mode:

    ``summary`` | ``all``

    If to output full standard deviation fields or just summary
    statistics.  In lieu of a complete field, summary instead reports
    the maximum and average standard deviation for each field. The
    default is ``summary`` with ``all`` doubling the size of the
    resulting files.

6. ``basedir`` --- relative path to directory where outputs will be
   written:

    *string*

7. ``basename`` --- pattern of output names:

    *string*

8. ``precision`` --- output file number precision:

    ``single`` | ``double``

    The default is ``single``. Note that this only impacts the output,
    with statistic accumulation *always* being performed in double
    precision.

9. ``region`` --- region to be written, specified as either the
   entire domain using ``*``, a combination of the geometric shapes
   specified in :ref:`regions`, or a sub-region of elements that have
   faces on a specific domain boundary via the name of the domain
   boundary:

    ``*`` | ``shape(args, ...)`` | *string*

10. ``avg``-*name* --- expression to time average, written as a
    function of the primitive variables and gradients thereof;
    multiple expressions, each with their own *name*, may be specified:

    *string*

11. ``fun-avg``-*name* --- expression to compute at file output time,
    written as a function of any ordinary average terms; multiple
    expressions, each with their own *name*, may be specified:

    *string*

Example::

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 2.0
    mode = windowed
    basedir = .
    basename = files-{t:06.2f}

    avg-u = u
    avg-v = v
    avg-uu = u*u
    avg-vv = v*v
    avg-uv = u*v

    fun-avg-upup = uu - u*u
    fun-avg-vpvp = vv - v*v
    fun-avg-upvp = uv - u*v

.. _integrate-plugin:

[soln-plugin-integrate]
^^^^^^^^^^^^^^^^^^^^^^^

Integrate quantities over the compuational domain. Parameterised with:

1. ``nsteps`` --- calculate the integral every ``nsteps`` time steps:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

4. ``quad-deg`` --- degree of quadrature rule (optional):

    *int*

5. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

6. ``norm`` --- sets the degree and calculates an :math:`L_p` norm,
    otherwise standard integration is performed:

    *float* | ``inf`` | ``none``

7. ``region`` --- region to integrate, specified as either the
   entire domain using ``*`` or a combination of the geometric shapes
   specified in :ref:`regions`:

    ``*`` | ``shape(args, ...)``

8. ``int``-*name* --- expression to integrate, written as a function of
   the primitive variables and gradients thereof, the physical coordinates
   [x, y, [z]] and/or the physical time [t]; multiple expressions,
   each with their own *name*, may be specified:

    *string*

Example::

    [soln-plugin-integrate]
    nsteps = 50
    file = integral.csv
    header = true
    quad-deg = 9
    vor1 = (grad_w_y - grad_v_z)
    vor2 = (grad_u_z - grad_w_x)
    vor3 = (grad_v_x - grad_u_y)

    int-E = rho*(u*u + v*v + w*w)
    int-enst = rho*(%(vor1)s*%(vor1)s + %(vor2)s*%(vor2)s + %(vor3)s*%(vor3)s)

[solver-plugin-source]
^^^^^^^^^^^^^^^^^^^^^^

Injects solution, space (x, y, [z]), and time (t) dependent
source terms with

1. ``rho`` --- density source term for ``euler`` | ``navier-stokes``:

    *string*

2. ``rhou`` --- x-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

3. ``rhov`` --- y-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

4. ``rhow`` --- z-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

5. ``E`` --- energy source term for ``euler`` | ``navier-stokes``
   :

    *string*

6. ``p`` --- pressure source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

7. ``u`` --- x-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

8. ``v`` --- y-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

9. ``w`` --- w-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

Example::

    [solver-plugin-source]
    rho = t
    rhou = x*y*sin(y)
    rhov = z*rho
    rhow = 1.0
    E = 1.0/(1.0+x)

[solver-plugin-turbulence]
^^^^^^^^^^^^^^^^^^^^^^^^^^

Injects synthetic eddies into a region of the domain. Parameterised with

1. ``avg-rho`` --- average free-stream density:

    *float*

2. ``avg-u`` --- average free-stream velocity magnitude:

    *float*

3. ``avg-mach`` --- averge free-stream Mach number:

    *float*

4. ``turbulence-intensity`` --- percentage turbulence intensity:

    *float*

5. ``turbulence-length-scale`` --- turbulent length scale:

    *float*

6. ``sigma`` --- standard deviation of Gaussian sythetic eddy profile:

    *float*

7. ``centre`` --- centre of plane on which synthetic eddies are injected:

    (*float*, *float*, *float*)

8. ``y-dim`` --- y-dimension of plane:

    *float*

9. ``z-dim`` --- z-dimension of plane:

    *float*

10. ``rot-axis`` --- axis about which plane is rotated:

    (*float*, *float*, *float*)

11. ``rot-angle`` --- angle in degrees that plane is rotated:

    *float*

Example::

    [solver-plugin-turbulence]
    avg-rho = 1.0
    avg-u = 1.0
    avg-mach = 0.2
    turbulence-intensity = 1.0
    turbulence-length-scale = 0.075
    sigma = 0.7
    centre = (0.15, 2.0, 2.0)
    y-dim = 3.0
    z-dim = 3.0
    rot-axis = (0, 0, 1)
    rot-angle = 0.0

Regions
-------

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

Region expressions can also be added and subtracted together
arbitrarily.  For example
``box((-10, -10, -10), (10, 10, 10)) - sphere((0, 0, 0), 3)`` will
result in a cube-shaped region with a sphere cut out of the middle.

Additional Information
----------------------

The :ref:`INI<configuration-file>` file format is very versatile. A feature that
can be useful in defining initial conditions is the substitution feature and
this is demonstrated in the :ref:`integrate-plugin` example.

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
