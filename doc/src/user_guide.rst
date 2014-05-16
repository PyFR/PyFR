**********
User Guide
**********

Getting Started
===============

Downloading the Source
----------------------

PyFR can be obtained `here <http://www.pyfr.org/download.php>`_

Dependencies
------------

Overview
^^^^^^^^

PyFR |release| has a hard dependency on Python 2.7. PyFR |release| does
not currently support Microsoft Windows. To run PyFR |release| it is
necessary to install the following Python packages:

1. `mako <http://www.makotemplates.org/>`_
2. `mpi4py <http://mpi4py.scipy.org/>`_ >= 1.3
3. `numpy <http://www.numpy.org/>`_ >= 1.8
4. `mpmath <http://code.google.com/p/mpmath/>`_ >= 0.18

OpenMP Backend
^^^^^^^^^^^^^^

The OpenMP backend targets multi-core CPUs. The backend requires:

1. GCC >= 4.7
2. A BLAS library compiled as a shared library
   (e.g. `OpenBLAS <http://www.openblas.net/>`_)

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

Installation
------------

Before running PyFR |release| it is first necessary to either install
the software using the provided ``setup.py`` installer or add the root
PyFR directory to ``PYTHONPATH``::

    user@computer ~/PyFR$ export PYTHONPATH=.:$PYTHONPATH
    user@computer ~/PyFR$ python pyfr/scripts/pyfr-sim --help

Running PyFR
============

Overview
--------

PyFR |release| consists of three separate tools:

1. ``pyfr-mesh`` --- for pre-processing
2. ``pyfr-sim`` --- the solver
3. ``pyfr-postp`` --- for post-processing

PyFR |release| uses three distinct file formats:

1. ``.ini`` --- configuration file
2. ``.pyfrm`` --- mesh file
3. ``.pyfrs`` --- solution file

PyFR-Mesh
---------

``pyfr-pmesh`` is for pre-processing. The following sub-tools are
available:

1. ``pyfr-mesh convert`` --- Convert a `Gmsh
   <http:http://geuz.org/gmsh/>`_ .msh file into a PyFR .pyfrm file.
   Example::

        pyfr-mesh convert mesh.msh mesh.pyfrm

For full details invoke::

    pyfr-mesh [sub-tool] --help

PyFR-Sim
--------

Overview
^^^^^^^^

``pyfr-sim`` is the solver. The following sub-tools are available:

1. ``pyfr-sim run`` --- Start a new PyFR simulation. Example::

        pyfr-sim run mesh.pyfrm configuration.ini
        
2. ``pyfr-sim restart`` --- Restart a PyFR simulation from an existing 
   solution file. Example::

        pyfr-sim restart mesh.pyfrm solution.pyfrs

For full details invoke::

    pyfr-sim [sub-tool] --help

Running in Parallel
^^^^^^^^^^^^^^^^^^^

``pyfr-sim`` can be run in parallel. To do so prefix ``pyfr-sim`` with
``mpirun -n <cores/devices>``. Note that the mesh must be
pre-partitioned, and the number of cores or devices must be equal to
the number of partitions.

PyFR-PostP
----------

``pyfr-postp`` is for post-processing. The following sub-tools are
available:

1. ``pyfr-postp convert`` --- Convert a PyFR .pyfrs file into an
   unstructured VTK .vtu file. Example::

        pyfr-postp convert mesh.pyfrm solution.pyfrs solution.vtu divide

2. ``pyfr-postp pack`` --- Swap between the pyfr-dir and pyfr-file
   format. Example::

        pyfr-postp pack solution_directory.pyfrs solution_file.pyfrs

3. ``pyfr-postp time-avg`` --- Time-average a series of PyFR solution
   files. Example::

        pyfr-postp time-avg average.pyfrs t1.pyfrs t2.pyfrs t3.pyfrs

4. ``pyfr-postp unpack`` --- Swap between the pyfr-file and pyfr-dir
   format. Example::

        pyfr-postp unpack solution_file.pyfrs solution_directory.pyfrs

For full details invoke::

    pyfr-postp [sub-tool] --help

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

Parameterizes the backend. Options:

1. ``precision`` --- number precision:

    ``single | double``

2. ``rank-allocator`` --- MPI rank allocator:

    ``linear``

Example::

    [backend]
    precision = double
    rank-allocator = linear

[backend-openmp]
^^^^^^^^^^^^^^^^

Parameterizes the OpenMP backend. Options:

1. ``cc`` --- C compiler

2. ``cblas-st`` --- path to shared single-threaded C BLAS library

3. ``cblas-mt`` --- path to shared multi-threaded C BLAS library

Example::

    [backend-openmp]
    cc = gcc
    cblas-mt = example/path/libBLAS.dylib

[constants]
^^^^^^^^^^^

Sets constants used in the simulation. Options:

1. ``gamma`` --- ratio of specific heats

2. ``mu`` --- dynamic viscosity

3. ``Pr`` --- Prandlt number

Example::

    [constants]
    gamma = 1.4
    mu = 0.001
    Pr = 0.72

[solver]
^^^^^^^^

Parameterizes the solver. Options:

1. ``system`` --- governing system:

    ``euler | navier-stokes``

2. ``order`` --- order of polynomial solution basis

Example::

    [solver]
    system = navier-stokes
    order = 3

[solver-time-integrator]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterizes the time-integration scheme used by the solver. Options:

1. ``scheme`` --- time-integration scheme:

    ``euler | rk4 | rk45 | dopri5``

2. ``controller`` --- time-step size controller:

    ``none``

3. ``t0`` --- initial time

4. ``dt`` --- time-step

Example::

    [solver-time-integrator]
    scheme = rk4
    controller = none
    t0 = 0.0
    dt = 0.001

[solver-interfaces]
^^^^^^^^^^^^^^^^^^^

Parameterizes the interfaces. Options:

1. ``riemann-solver`` --- Riemann solver:

    ``rusanov``

2. ``ldg-beta`` --- beta parameter used for LDG

3. ``ldg-tau`` --- tau parameter used for LDG

Example::

    [solver-interfaces]
    riemann-solver = rusanov
    ldg-beta = 0.5
    ldg-tau = 0.1

[solver-interfaces-line]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterizes the line interfaces. Options:

1. ``flux-pts`` --- location of the flux points on a line interface:

    ``gauss-legendre | gauss-legendre-lobatto``

Example::

    [solver-interfaces-line]
    flux-pts = gauss-legendre

[solver-interfaces-tri]
^^^^^^^^^^^^^^^^^^^^^^^

Parameterizes the triangular interfaces. Options:

1. ``flux-pts`` --- location of the flux points on a triangular
   interface:

    ``williams-shunn``

Example::

    [solver-interfaces-tri]
    flux-pts = williams-shunn

[solver-interfaces-quad]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterizes the quadrilateral interfaces. Options:

1. ``flux-pts`` --- location of the flux points on a quadrilateral
   interface:

    ``gauss-legendre | gauss-legendre-lobatto``

Example::

    [solver-interfaces-quad]
    flux-pts = gauss-legendre

[solver-elements-tri]
^^^^^^^^^^^^^^^^^^^^^

Parameterizes the triangular elements. Options:

1. ``soln-pts`` --- location of the solution points in a triangular
   element:

    ``williams-shunn``

Example::

    [solver-elements-tri]
    soln-pts = williams-shunn

[solver-elements-quad]
^^^^^^^^^^^^^^^^^^^^^^

Parameterizes the quadrilateral elements. Options:

1. ``soln-pts`` --- location of the solution points in a quadrilateral
   element:

    ``gauss-legendre | gauss-legendre-lobatto``

Example::

    [solver-elements-quad]
    soln-pts = gauss-legendre

[solver-elements-hex]
^^^^^^^^^^^^^^^^^^^^^

Parameterizes the hexahedral elements. Options:

1. ``soln-pts`` --- location of the solution points in a hexahedral
   element:

    ``gauss-legendre | gauss-legendre-lobatto``

Example::

    [solver-elements-hex]
    soln-pts = gauss-legendre

[solver-elements-tet]
^^^^^^^^^^^^^^^^^^^^^

Parameterizes the tetrahedral elements. Options:

1. ``soln-pts`` --- location of the solution points in a tetrahedral
   element:

    ``shunn-ham``

Example::

    [solver-elements-tet]
    soln-pts = shunn-ham

[solver-elements-pri]
^^^^^^^^^^^^^^^^^^^^^

Parameterizes the prismatic elements. Options:

1. ``soln-pts`` --- location of the solution points in a prismatic
   element:

    ``williams-shunn~gauss-legendre | 
    williams-shunn~gauss-legendre-lobatto``

Example::

    [solver-elements-pri]
    soln-pts = williams-shunn~gauss-legendre

[soln-output]
^^^^^^^^^^^^^^^

Parameterizes the solver output. Options:

1. ``format`` --- format of the outputs:

    ``pyfrs-file | pyfrs-dir``

2. ``basedir`` --- relative path to directory where outputs will be
   written

3. ``basename`` --- pattern of output names

4. ``times`` --- times at which outputs will be dumped

Example::

    [soln-output]
    format = pyfrs-file
    basedir = .
    basename = files_%(t).2f
    times = range(0, 1, 11)

[soln-bcs-{$NAME}]
^^^^^^^^^^^^^^^^^^

Parameterizes boundary condition labelled {$NAME} in the .pyfrm file.
Options:

1. ``type`` --- type of boundary condition:

    ``no-slp-adia-wall | no-slp-aisot-wall | sub-in-frv | 
    sub-in-ftpttang | sub-out-fp | sup-in-fa | sup-out-fn``

    where

    ``no-slp-isot-wall`` requires

        - ``cpTw`` --- product of specific heat capacity at constant
          pressure and temperature of wall

        - ``u`` --- x-velocity of wall

        - ``v`` --- y-velocity of wall

        - ``w`` --- z-velocity of wall

    ``sub-in-frv`` requires

        - ``rho`` --- density

        - ``u`` --- x-velocity

        - ``v`` --- y-velocity

        - ``w`` --- z-velocity

    ``sub-in-ftpttang`` requires

        - ``pt`` --- total pressure

        - ``cpTt`` --- product of specific heat capacity at constant
          pressure and total temperature

        - ``theta`` --- azimuth angle of inflow (in degrees) measured in
          the x-y plane relative to the global positive x-axis

        - ``phi`` --- inclination angle of inflow (in degrees) measured
          relative to the global positive z-axis

    ``sub-out-fp`` requires

        - ``p`` --- static pressure

    ``sup-in-fa`` requires

        - ``rho`` --- density

        - ``u`` --- x-velocity

        - ``v`` --- y-velocity

        - ``w`` --- z-velocity

        - ``p`` --- static pressure

Example::

    [soln-bcs-bcwallupper]
    type = no-slp-isot-wall
    cpTw = 10.0
    u = 1.0

[soln-ics]
^^^^^^^^^^

Parameterizes the initial conditions. Options:

1. ``rho`` --- initial density distribution

2. ``u`` --- initial x-velocity distribution

3. ``v`` --- initial y-velocity distribution

4. ``w`` --- initial z-velocity distribution

5. ``p`` --- initial static pressure distribution

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

4. Run pyfr-mesh to covert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``couette_flow_2d.pyfrm``::

        pyfr-mesh convert couette_flow_2d.msh couette_flow_2d.pyfrm

5. Run pyfr-sim to solve the Navier-Stokes equations on the mesh,
   generating a series of PyFR solution files called
   ``couette_flow_2d-*.pyfrs``::

        pyfr-sim -p run couette_flow_2d.pyfrm couette_flow_2d.ini

6. Run pyfr-postp on the solution file ``couette_flow_2d_4.00.pyfrs``
   converting it into an unstructured VTK file called
   ``couette_flow_2d_4.00.vtu``. Note that in order to visualise the
   high-order data, each high-order element is sub-divided into smaller
   linear elements. The level of sub-division is controlled by the 
   integer at the end of the command::

        pyfr-postp convert couette_flow_2d.pyfrm couette_flow_2d_4.00.pyfrs couette_flow_2d_4.00.vtu divide -d 4

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

3. Copy the partitioned `Gmsh <http:http://geuz.org/gmsh/>`_ file
   ``PyFR/examples/euler_vortex_2d/euler_vortex_2d.msh`` into
   ``euler_vortex_2d/``

4. Run pyfr-mesh to convert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``euler_vortex_2d.pyfrm``::

        pyfr-mesh convert euler_vortex_2d.msh euler_vortex_2d.pyfrm

5. Run pyfr-sim to solve the Euler equations on the mesh, generating a
   series of PyFR solution files called ``euler_vortex_2d*.pyfrs``::

        mpirun -n 2 pyfr-sim -p run euler_vortex_2d.pyfrm euler_vortex_2d.ini

6. Run pyfr-postp on the solution file ``euler_vortex_2d_100.0.pyfrs``
   converting it into an unstructured VTK file called
   ``euler_vortex_2d_100.0.vtu``. Note that in order to visualise the
   high-order data, each high-order element is sub-divided into smaller
   linear elements. The level of sub-division is controlled by the 
   integer at the end of the command::

        pyfr-postp convert euler_vortex_2d.pyfrm euler_vortex_2d-100.0.pyfrs euler_vortex_2d_100.0.vtu divide -d 4

7. Visualise the unstructured VTK file in `Paraview
   <http://www.paraview.org/>`_

.. figure:: ../fig/euler_vortex_2d/euler_vortex_2d.png
   :width: 450px
   :figwidth: 450px
   :alt: euler vortex
   :align: center

   Colour map of density distribution at 100 time units.
