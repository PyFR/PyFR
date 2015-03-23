.. highlightlang:: none

**********
User Guide
**********

Getting Started
===============

Downloading the Source
----------------------

PyFR can be obtained `here <http://www.pyfr.org/download.php>`_.

Dependencies
------------

Overview
^^^^^^^^

PyFR |release| has a hard dependency on Python 3.3+ and the following
Python packages:

1. `mako <http://www.makotemplates.org/>`_
2. `mpi4py <http://mpi4py.scipy.org/>`_ >= 1.3
3. `mpmath <http://code.google.com/p/mpmath/>`_ >= 0.18
4. `numpy <http://www.numpy.org/>`_ >= 1.8
5. `pytools <https://pypi.python.org/pypi/pytools>`_ >= 2014.3

To run PyFR |release| in parallel it is also necessary to have one of
the following installed:

1. `metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`_ >= 5.0
2. `scotch <http://www.labri.fr/perso/pelegrin/scotch/>`_ >= 6.0

PyFR |release| does not currently support Microsoft Windows.

CUDA Backend
^^^^^^^^^^^^

The CUDA backend targets NVIDIA GPUs with a compute capability of 2.0
or greater. The backend requires:

1. `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 4.2
2. `pycuda <http://mathema.tician.de/software/pycuda/>`_ >= 2011.2

OpenCL Backend
^^^^^^^^^^^^^^

The OpenCL backend targets a range of accelerators including GPUs from
AMD and NVIDIA. The backend requires:

1. OpenCL
2. `pyopencl <http://mathema.tician.de/software/pyopencl/>`_ >= 2013.2
3. `clBLAS <https://github.com/clMathLibraries/clBLAS>`_

OpenMP Backend
^^^^^^^^^^^^^^

The OpenMP backend targets multi-core CPUs. The backend requires:

1. GCC >= 4.7
2. A BLAS library compiled as a shared library
   (e.g. `OpenBLAS <http://www.openblas.net/>`_)

Installation
------------

Before running PyFR |release| it is first necessary to either install
the software using the provided ``setup.py`` installer or add the root
PyFR directory to ``PYTHONPATH`` using::

    user@computer ~/PyFR$ export PYTHONPATH=.:$PYTHONPATH

Running PyFR
============

Overview
--------

PyFR |release| uses three distinct file formats:

1. ``.ini`` --- configuration file
2. ``.pyfrm`` --- mesh file
3. ``.pyfrs`` --- solution file

Mesh
----

``pyfr mesh`` is for pre-processing. The following sub-tools are
available:

1. ``pyfr mesh convert`` --- convert a `Gmsh
   <http:http://geuz.org/gmsh/>`_ .msh file into a PyFR .pyfrm file.

   Example::

        pyfr mesh convert mesh.msh mesh.pyfrm

2. ``pyfr mesh partition`` --- partition an existing mesh and
   associated solution files.

   Example::

       pyfr mesh partition 2 mesh.pyfrm solution.pyfrs .

For full details invoke::

    pyfr mesh [sub-tool] --help

Sim
---

Overview
^^^^^^^^

``pyfr sim`` is the solver. The following sub-tools are available:

1. ``pyfr sim run`` --- start a new PyFR simulation. Example::

        pyfr sim run mesh.pyfrm configuration.ini

2. ``pyfr sim restart`` --- restart a PyFR simulation from an existing
   solution file. Example::

        pyfr sim restart mesh.pyfrm solution.pyfrs

For full details invoke::

    pyfr sim [sub-tool] --help

Running in Parallel
^^^^^^^^^^^^^^^^^^^

``pyfr sim`` can be run in parallel. To do so prefix ``pyfr sim`` with
``mpirun -n <cores/devices>``. Note that the mesh must be
pre-partitioned, and the number of cores or devices must be equal to
the number of partitions.

Postp
----------

``pyfr postp`` is for post-processing. The following sub-tools are
available:

1. ``pyfr postp convert`` --- convert a PyFR .pyfrs file into an
   unstructured VTK .vtu file. Example::

        pyfr postp convert mesh.pyfrm solution.pyfrs solution.vtu

2. ``pyfr postp pack`` --- swap between the pyfr-dir and pyfr-file
   format. Example::

        pyfr postp pack solution_directory.pyfrs solution_file.pyfrs

3. ``pyfr postp time-avg`` --- time-average a series of PyFR solution
   files. Example::

        pyfr postp time-avg average.pyfrs t1.pyfrs t2.pyfrs t3.pyfrs

4. ``pyfr postp unpack`` --- swap between the pyfr-file and pyfr-dir
   format. Example::

        pyfr postp unpack solution_file.pyfrs solution_directory.pyfrs

For full details invoke::

    pyfr postp [sub-tool] --help

Configuration File (.ini)
-------------------------

Overview
^^^^^^^^

The .ini configuration file parameterizes the simulation. It is written
in the `INI <http://en.wikipedia.org/wiki/INI_file>`_ format.
Parameters are grouped into sections. The roles of each section and
their associated parameters are described below.

[backend]
^^^^^^^^^

Parameterises the backend with

1. ``precision`` --- number precision:

    ``single`` | ``double``

2. ``rank-allocator`` --- MPI rank allocator:

    ``linear``

Example::

    [backend]
    precision = double
    rank-allocator = linear

[backend-cuda]
^^^^^^^^^^^^^^

Parameterises the CUDA backend with

1. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``round-robin`` | ``local-rank``

Example::

    [backend-cuda]
    device-id = round-robin

[backend-opencl]
^^^^^^^^^^^^^^^^

Parameterises the OpenCL backend with

1. ``platform-id`` --- for selecting platform id:

    *int* | *string*

2. ``device-type`` --- for selecting what type of device(s) to run on:

    ``all`` | ``cpu`` | ``gpu`` | ``accelerator``

3. ``device-id`` --- for selecting which device(s) to run on:

    *int* | *string* | ``local-rank``

Example::

    [backend-opencl]
    platform-id = 0
    device-type = gpu
    device-id = local-rank

[backend-openmp]
^^^^^^^^^^^^^^^^

Parameterises the OpenMP backend with

1. ``cc`` --- C compiler

    *string*

2. ``cblas`` --- path to shared C BLAS library

    *string*

3. ``cblas-type`` --- type of BLAS library

    ``serial`` | ``parallel``

Example::

    [backend-openmp]
    cc = gcc
    cblas= example/path/libBLAS.dylib
    cblas-type = parallel

[constants]
^^^^^^^^^^^

Sets constants used in the simulation with

1. ``gamma`` --- ratio of specific heats

    *float*

2. ``mu`` --- dynamic viscosity

    *float*

3. ``Pr`` --- Prandtl number

    *float*

4. ``cpTref`` --- product of specific heat at constant pressure and
   reference temperature for Sutherland's Law

   *float*

5. ``cpTs`` --- product of specific heat at constant pressure and
   Sutherland temperature for Sutherland's Law

   *float*

Example::

    [constants]
    gamma = 1.4
    mu = 0.001
    Pr = 0.72

[solver]
^^^^^^^^

Parameterises the solver with

1. ``system`` --- governing system:

    ``euler`` | ``navier-stokes``

2. ``order`` --- order of polynomial solution basis

    *int*

3. ``anti-alias`` --- type of anti-aliasing:

    ``flux`` | ``surf-flux`` | ``div-flux`` | ``flux, surf-flux`` |
    ``flux, div-flux`` | ``surf-flux, div-flux`` |
    ``flux, surf-flux, div-flux``

4. ``viscosity-correction`` --- viscosity correction

    ``none`` | ``sutherland``

Example::

    [solver]
    system = navier-stokes
    order = 3
    anti-alias = flux
    viscosity-correction = none

[solver-time-integrator]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the time-integration scheme used by the solver with

1. ``scheme`` --- time-integration scheme:

    ``euler`` | ``rk34`` | ``rk4`` | ``rk45``

2. ``t0`` --- initial time

    *float*

3. ``dt`` --- time-step

    *float*

4. ``controller`` --- time-step size controller:

    ``none`` | ``pi``

    where

    ``pi`` only works with ``rk34`` and ``rk45`` and requires

        - ``atol`` --- absolute error tolerance

           *float*

        - ``rtol`` --- relative error tolerance

           *float*

        - ``safety-fact`` --- safety factor for step size adjustment
          (suitable range 0.80-0.95)

           *float*

        - ``min-fact`` --- minimum factor that the time-step can change
          between iterations (suitable range 0.1-0.5)

           *float*

        - ``max-fact`` --- maximum factor that the time-step can change
          between iterations (suitable range 2.0-6.0)

           *float*

Example::

    [solver-time-integrator]
    scheme = rk45
    controller = pi
    t0 = 0.0
    dt = 0.001
    atol = 0.00001
    rtol = 0.00001
    safety-fact = 0.9
    min-fact = 0.3
    max-fact = 2.5

[solver-interfaces]
^^^^^^^^^^^^^^^^^^^

Parameterises the interfaces with

1. ``riemann-solver`` --- type of Riemann solver:

    ``rusanov`` | ``hll`` | ``hllc`` | ``roe``

2. ``ldg-beta`` --- beta parameter used for LDG

    *float*

3. ``ldg-tau`` --- tau parameter used for LDG

    *float*

Example::

    [solver-interfaces]
    riemann-solver = rusanov
    ldg-beta = 0.5
    ldg-tau = 0.1

[solver-interfaces-line]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the line interfaces with

1. ``flux-pts`` --- location of the flux points on a line interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

Example::

    [solver-interfaces-line]
    flux-pts = gauss-legendre

[solver-interfaces-tri]
^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the triangular interfaces with

1. ``flux-pts`` --- location of the flux points on a triangular
   interface:

    ``williams-shunn``

Example::

    [solver-interfaces-tri]
    flux-pts = williams-shunn

[solver-interfaces-quad]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the quadrilateral interfaces with

1. ``flux-pts`` --- location of the flux points on a quadrilateral
   interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

Example::

    [solver-interfaces-quad]
    flux-pts = gauss-legendre

[solver-elements-tri]
^^^^^^^^^^^^^^^^^^^^^

Parameterises the triangular elements with

1. ``soln-pts`` --- location of the solution points in a triangular
   element:

    ``williams-shunn``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   triangular element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   triangular element:

    ``williams-shunn``

Example::

    [solver-elements-tri]
    soln-pts = williams-shunn
    quad-deg = 10
    quad-pts = williams-shunn

[solver-elements-quad]
^^^^^^^^^^^^^^^^^^^^^^

Parameterises the quadrilateral elements with

1. ``soln-pts`` --- location of the solution points in a quadrilateral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   quadrilateral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   quadrilateral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

Example::

    [solver-elements-quad]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre

[solver-elements-hex]
^^^^^^^^^^^^^^^^^^^^^

Parameterises the hexahedral elements with

1. ``soln-pts`` --- location of the solution points in a hexahedral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   hexahedral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   hexahedral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

Example::

    [solver-elements-hex]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre

[solver-elements-tet]
^^^^^^^^^^^^^^^^^^^^^

Parameterises the tetrahedral elements with

1. ``soln-pts`` --- location of the solution points in a tetrahedral
   element:

    ``shunn-ham``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   tetrahedral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   tetrahedral element:

    ``shunn-ham``

Example::

    [solver-elements-tet]
    soln-pts = shunn-ham
    quad-deg = 10
    quad-pts = shunn-ham

[solver-elements-pri]
^^^^^^^^^^^^^^^^^^^^^

Parameterises the prismatic elements with

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
    ``williams-shunn~gauss-legendre-lobatto``

Example::

    [solver-elements-pri]
    soln-pts = williams-shunn~gauss-legendre
    quad-deg = 10
    quad-pts = williams-shunn~gauss-legendre

[solver-elements-pyr]
^^^^^^^^^^^^^^^^^^^^^

Parameterises the pyramidal elements with

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

[solver-source-terms]
^^^^^^^^^^^^^^^^^^^^^

Parameterises space (x, y, [z]) and time (t) dependent source terms with

1. ``rho`` --- density source term

    *string*

2. ``rhou`` --- x-momentum source term

    *string*

3. ``rhov`` --- y-momentum source term

    *string*

4. ``rhow`` --- z-momentum source term

    *string*

5. ``E`` --- energy source term

    *string*

Example::

    [solver-source-terms]
    rho = t
    rhou = x*y*sin(y)
    rhov = z
    rhow = 1.0
    E = 1.0/(1.0+x)

[soln-output]
^^^^^^^^^^^^^

Parameterises the output with

1. ``format`` --- format of the outputs:

    ``pyfrs-file`` | ``pyfrs-dir``

2. ``basedir`` --- relative path to directory where outputs will be
   written

    *string*

3. ``basename`` --- pattern of output names

    *string*

4. ``times`` --- times at which outputs will be dumped

    ``range(`` *float* ``,`` *float* ``,`` *int* ``)``

Example::

    [soln-output]
    format = pyfrs-file
    basedir = .
    basename = files_%(t).2f
    times = range(0, 1, 11)

[soln-filter]
^^^^^^^^^^^^^

Parameterises an exponential solution filter with

1. ``freq`` --- frequency at which filter is applied:

    *int*

2. ``alpha`` --- strength of filter:

    *float*

3. ``order`` --- order of filter:

    *int*

4. ``cutoff`` --- cutoff frequency below which no filtering is applied:

    *int*

[soln-bcs-name]
^^^^^^^^^^^^^^^

Parameterises boundary condition labelled :code:`name` in the .pyfrm
file with

1. ``type`` --- type of boundary condition:

    ``char-riem-inv`` | ``no-slp-adia-wall`` | ``no-slp-isot-wall`` |
    ``sub-in-frv`` | ``sub-in-ftpttang`` | ``sub-out-fp`` |
    ``sup-in-fa`` | ``sup-out-fn``

    where

    ``char-riem-inv`` requires

        - ``rho`` --- density

           *float*

        - ``u`` --- x-velocity

           *float*

        - ``v`` --- y-velocity

           *float*

        - ``w`` --- z-velocity

           *float*

        - ``p`` --- static pressure

           *float*

    ``no-slp-isot-wall`` requires

        - ``u`` --- x-velocity of wall

           *float*

        - ``v`` --- y-velocity of wall

           *float*

        - ``w`` --- z-velocity of wall

           *float*

        - ``cpTw`` --- product of specific heat capacity at constant
          pressure and temperature of wall

           *float*

    ``sub-in-frv`` requires

        - ``rho`` --- density

           *float*

        - ``u`` --- x-velocity

           *float*

        - ``v`` --- y-velocity

           *float*

        - ``w`` --- z-velocity

           *float*

    ``sub-in-ftpttang`` requires

        - ``pt`` --- total pressure

           *float*

        - ``cpTt`` --- product of specific heat capacity at constant
          pressure and total temperature

           *float*

        - ``theta`` --- azimuth angle of inflow measured in
          the x-y plane relative to the global positive x-axis

           *float*

        - ``phi`` --- inclination angle of inflow measured
          relative to the global positive z-axis

           *float*

    ``sub-out-fp`` requires

        - ``p`` --- static pressure

           *float*

    ``sup-in-fa`` requires

        - ``rho`` --- density

           *float*

        - ``u`` --- x-velocity

           *float*

        - ``v`` --- y-velocity

           *float*

        - ``w`` --- z-velocity

           *float*

        - ``p`` --- static pressure

           *float*

Example::

    [soln-bcs-bcwallupper]
    type = no-slp-isot-wall
    cpTw = 10.0
    u = 1.0

[soln-ics]
^^^^^^^^^^

Parameterises space (x, y, [z]) dependent initial conditions with

1. ``rho`` --- initial density distribution

    *string*

2. ``u`` --- initial x-velocity distribution

    *string*

3. ``v`` --- initial y-velocity distribution

    *string*

4. ``w`` --- initial z-velocity distribution

    *string*

5. ``p`` --- initial static pressure distribution

    *string*

Example::

    [soln-ics]
    rho = 1.0
    u = x*y*sin(y)
    v = z
    w = 1.0
    p = 1.0/(1.0+x)

Example --- 2D Couette Flow
===========================

Proceed with the following steps to run a serial 2D Couette flow
simulation on a mixed unstructured mesh:

1. Create a working directory called ``couette_flow_2d/``

2. Copy the configuration file
   ``PyFR/examples/couette_flow_2d/couette_flow_2d.ini`` into
   ``couette_flow_2d/``

3. Copy the `Gmsh <http:http://geuz.org/gmsh/>`_ mesh file
   ``PyFR/examples/couette_flow_2d/couette_flow_2d.msh`` into
   ``couette_flow_2d/``

4. Run pyfr mesh to covert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``couette_flow_2d.pyfrm``::

        pyfr mesh convert couette_flow_2d.msh couette_flow_2d.pyfrm

5. Run pyfr sim to solve the Navier-Stokes equations on the mesh,
   generating a series of PyFR solution files called
   ``couette_flow_2d-*.pyfrs``::

        pyfr sim -p run couette_flow_2d.pyfrm couette_flow_2d.ini

6. Run pyfr postp on the solution file ``couette_flow_2d_4.00.pyfrs``
   converting it into an unstructured VTK file called
   ``couette_flow_2d_4.00.vtu``. Note that in order to visualise the
   high-order data, each high-order element is sub-divided into smaller
   linear elements. The level of sub-division is controlled by the
   integer at the end of the command::

        pyfr postp convert couette_flow_2d.pyfrm couette_flow_2d_4.00.pyfrs couette_flow_2d_4.00.vtu -d 4

7. Visualise the unstructured VTK file in `Paraview
   <http://www.paraview.org/>`_

.. figure:: ../fig/couette_flow_2d/couette_flow_2d.png
   :width: 450px
   :figwidth: 450px
   :alt: couette flow
   :align: center

   Colour map of steady-state density distribution.

Example --- 2D Euler Vortex
===========================

Proceed with the following steps to run a parallel 2D Euler vortex
simulation on a structured mesh:

1. Create a working directory called ``euler_vortex_2d/``

2. Copy the configuration file
   ``PyFR/examples/euler_vortex_2d/euler_vortex_2d.ini`` into
   ``euler_vortex_2d/``

3. Copy the `Gmsh <http:http://geuz.org/gmsh/>`_ file
   ``PyFR/examples/euler_vortex_2d/euler_vortex_2d.msh`` into
   ``euler_vortex_2d/``

4. Run pyfr mesh to convert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``euler_vortex_2d.pyfrm``::

        pyfr mesh convert euler_vortex_2d.msh euler_vortex_2d.pyfrm

5. Run pyfrmesh to partition the PyFR mesh file into two pieces::

        pyfr mesh partition 2 euler_vortex_2d.pyfrm .

6. Run pyfr sim to solve the Euler equations on the mesh, generating a
   series of PyFR solution files called ``euler_vortex_2d*.pyfrs``::

        mpirun -n 2 pyfr sim -p run euler_vortex_2d.pyfrm euler_vortex_2d.ini

7. Run pyfr postp on the solution file ``euler_vortex_2d_100.0.pyfrs``
   converting it into an unstructured VTK file called
   ``euler_vortex_2d_100.0.vtu``. Note that in order to visualise the
   high-order data, each high-order element is sub-divided into smaller
   linear elements. The level of sub-division is controlled by the
   integer at the end of the command::

        pyfr postp convert euler_vortex_2d.pyfrm euler_vortex_2d-100.0.pyfrs euler_vortex_2d_100.0.vtu -d 4

8. Visualise the unstructured VTK file in `Paraview
   <http://www.paraview.org/>`_

.. figure:: ../fig/euler_vortex_2d/euler_vortex_2d.png
   :width: 450px
   :figwidth: 450px
   :alt: euler vortex
   :align: center

   Colour map of density distribution at 100 time units.
