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

1. `h5py <http://www.h5py.org/>`_ >= 2.5
2. `mako <http://www.makotemplates.org/>`_ >= 1.0.0
3. `mpi4py <http://mpi4py.scipy.org/>`_ >= 1.3
4. `numpy <http://www.numpy.org/>`_ >= 1.8
5. `pytools <https://pypi.python.org/pypi/pytools>`_ >= 2014.3

Note that due to a bug in `numpy <http://www.numpy.org/>`_ PyFR is not
compatible with 32-bit Python distributions.

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
2. `pyopencl <http://mathema.tician.de/software/pyopencl/>`_
   >= 2015.2.4
3. `clBLAS <https://github.com/clMathLibraries/clBLAS>`_

OpenMP Backend
^^^^^^^^^^^^^^

The OpenMP backend targets multi-core CPUs. The backend requires:

1. GCC >= 4.7
2. A BLAS library compiled as a shared library
   (e.g. `OpenBLAS <http://www.openblas.net/>`_)

Running in Parallel
^^^^^^^^^^^^^^^^^^^

To partition meshes for running in parallel it is also necessary to
have one of the following partitioners installed:

1. `metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`_ >= 5.0
2. `scotch <http://www.labri.fr/perso/pelegrin/scotch/>`_ >= 6.0

Installation
------------

Before running PyFR |release| it is first necessary to either install
the software using the provided ``setup.py`` installer or add the root
PyFR directory to ``PYTHONPATH`` using::

    user@computer ~/PyFR$ export PYTHONPATH=.:$PYTHONPATH

To manage installation of Python dependencies we strongly recommend
using `pip <https://pypi.python.org/pypi/pip>`_ and
`virtualenv <https://pypi.python.org/pypi/virtualenv>`_.

Running PyFR
============

Overview
--------

PyFR |release| uses three distinct file formats:

1. ``.ini`` --- configuration file
2. ``.pyfrm`` --- mesh file
3. ``.pyfrs`` --- solution file

The following commands are available from the ``pyfr`` program:

1. ``pyfr import`` --- convert a `Gmsh
   <http:http://geuz.org/gmsh/>`_ .msh file into a PyFR .pyfrm file.

   Example::

        pyfr import mesh.msh mesh.pyfrm

2. ``pyfr partition`` --- partition an existing mesh and
   associated solution files.

   Example::

       pyfr partition 2 mesh.pyfrm solution.pyfrs .

3. ``pyfr run`` --- start a new PyFR simulation. Example::

        pyfr run mesh.pyfrm configuration.ini

4. ``pyfr restart`` --- restart a PyFR simulation from an existing
   solution file. Example::

        pyfr restart mesh.pyfrm solution.pyfrs

5. ``pyfr export`` --- convert a PyFR .pyfrs file into an
   unstructured VTK .vtu or .pvtu file. Example::

        pyfr export mesh.pyfrm solution.pyfrs solution.vtu

Running in Parallel
^^^^^^^^^^^^^^^^^^^

``pyfr`` can be run in parallel. To do so prefix ``pyfr`` with
``mpirun -n <cores/devices>``. Note that the mesh must be
pre-partitioned, and the number of cores or devices must be equal to
the number of partitions.

Configuration File (.ini)
-------------------------

Overview
^^^^^^^^

The .ini configuration file parameterises the simulation. It is written
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

1. ``cc`` --- C compiler:

    *string*

2. ``cflags`` --- Additional C compiler flags:

    *string*

3. ``cblas`` --- path to shared C BLAS library:

    *string*

4. ``cblas-type`` --- type of BLAS library:

    ``serial`` | ``parallel``

Example::

    [backend-openmp]
    cc = gcc
    cblas= example/path/libBLAS.dylib
    cblas-type = parallel

[constants]
^^^^^^^^^^^

Sets constants used in the simulation with

1. ``gamma`` --- ratio of specific heats:

    *float*

2. ``mu`` --- dynamic viscosity:

    *float*

3. ``Pr`` --- Prandtl number:

    *float*

4. ``cpTref`` --- product of specific heat at constant pressure and
   reference temperature for Sutherland's Law:

   *float*

5. ``cpTs`` --- product of specific heat at constant pressure and
   Sutherland temperature for Sutherland's Law:

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

2. ``order`` --- order of polynomial solution basis:

    *int*

3. ``anti-alias`` --- type of anti-aliasing:

    ``flux`` | ``surf-flux`` | ``div-flux`` | ``flux, surf-flux`` |
    ``flux, div-flux`` | ``surf-flux, div-flux`` |
    ``flux, surf-flux, div-flux``

4. ``viscosity-correction`` --- viscosity correction:

    ``none`` | ``sutherland``

5. ``shock-capturing`` --- shock capturing scheme:

    ``none`` | ``artificial-viscosity``

Example::

    [solver]
    system = navier-stokes
    order = 3
    anti-alias = flux
    viscosity-correction = none
    shock-capturing = artificial-viscosity

[solver-time-integrator]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the time-integration scheme used by the solver with

1. ``scheme`` --- time-integration scheme:

    ``euler`` | ``rk34`` | ``rk4`` | ``rk45`` | ``tvd-rk3``

2. ``tstart`` --- initial time:

    *float*

3. ``tend`` --- final time:

    *float*

4. ``dt`` --- time-step:

    *float*

5. ``controller`` --- time-step size controller:

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
    tstart = 0.0
    tend = 10.0
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

    ``rusanov`` | ``hll`` | ``hllc`` | ``roe`` | ``roem``

2. ``ldg-beta`` --- beta parameter used for LDG:

    *float*

3. ``ldg-tau`` --- tau parameter used for LDG:

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

[solver-interfaces-tri]
^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the triangular interfaces with

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

[solver-interfaces-quad]
^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises the quadrilateral interfaces with

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

    ``williams-shunn`` | ``witherden-vincent``

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

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

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

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

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

    ``shunn-ham`` | ``witherden-vincent``

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
    ``williams-shunn~gauss-legendre-lobatto`` | ``witherden-vincent``

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

Parameterises solution, space (x, y, [z]), and time (t) dependent
source terms with

1. ``rho`` --- density source term:

    *string*

2. ``rhou`` --- x-momentum source term:

    *string*

3. ``rhov`` --- y-momentum source term:

    *string*

4. ``rhow`` --- z-momentum source term:

    *string*

5. ``E`` --- energy source term:

    *string*

Example::

    [solver-source-terms]
    rho = t
    rhou = x*y*sin(y)
    rhov = z*rho
    rhow = 1.0
    E = 1.0/(1.0+x)

[solver-artificial-viscosity]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameterises artificial viscosity for shock capturing with

1. ``max-amu`` --- maximum artificial viscosity:

    *float*

2. ``s0`` --- sensor cut-off:

    *float*

3. ``kappa`` --- sensor range:

    *float*

Example::

    [solver-artificial-viscosity]
    max-amu = 0.01
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

Example::

    [soln-plugin-writer]
    dt-out = 0.01
    basedir = .
    basename = files-{t:.2f}

[soln-plugin-fluidforce-name]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Periodically integrates the pressure and viscous stress on the boundary
labelled ``name`` and writes out the resulting force vectors to a CSV
file. Parameterised with

1. ``nsteps`` --- integrate every ``nsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-fluidforce-wing]
    nsteps = 10
    file = wing-forces.csv
    header = true

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

Example::

    [soln-plugin-residual]
    nsteps = 10
    file = residual.csv
    header = true

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
    format = primative
    file = point-data.csv
    header = true

[soln-plugin-tavg]
^^^^^^^^^^^^^^^^^^^^^^

Time average quantities. Parameterised with

1. ``nsteps`` --- accumulate the average every ``nsteps`` time steps:

    *int*

2. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

3. ``basedir`` --- relative path to directory where outputs will be
   written:

    *string*

4. ``basename`` --- pattern of output names:

    *string*

5. ``avg-name`` --- expression as a function of the primitive variables,
   time (t), and space (x, y, [z]) to time average; multiple
   expressions, each with their own *name*, may be specified:

    *string*

Example::

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 2.0
    basedir = .
    basename = files-{t:06.2f}

    avg-p = p
    avg-p2 = p*p
    avg-vel = sqrt(u*u + v*v)

[soln-bcs-name]
^^^^^^^^^^^^^^^

Parameterises constant, or if available space (x, y, [z]) and time (t)
dependent, boundary condition labelled :code:`name` in the .pyfrm file
with

1. ``type`` --- type of boundary condition:

    ``char-riem-inv`` | ``no-slp-adia-wall`` | ``no-slp-isot-wall`` |
    ``slp-adia-wall`` | ``sub-in-frv`` | ``sub-in-ftpttang`` |
    ``sub-out-fp`` | ``sup-in-fa`` | ``sup-out-fn``

    where

    ``char-riem-inv`` requires

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

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

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

           *float* | *string*

    ``sup-in-fa`` requires

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

Example::

    [soln-bcs-bcwallupper]
    type = no-slp-isot-wall
    cpTw = 10.0
    u = 1.0

[soln-ics]
^^^^^^^^^^

Parameterises space (x, y, [z]) dependent initial conditions with

1. ``rho`` --- initial density distribution:

    *string*

2. ``u`` --- initial x-velocity distribution:

    *string*

3. ``v`` --- initial y-velocity distribution:

    *string*

4. ``w`` --- initial z-velocity distribution:

    *string*

5. ``p`` --- initial static pressure distribution:

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

4. Run pyfr to covert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``couette_flow_2d.pyfrm``::

        pyfr import couette_flow_2d.msh couette_flow_2d.pyfrm

5. Run pyfr to solve the Navier-Stokes equations on the mesh,
   generating a series of PyFR solution files called
   ``couette_flow_2d-*.pyfrs``::

        pyfr run -b cuda -p couette_flow_2d.pyfrm couette_flow_2d.ini

6. Run pyfr on the solution file ``couette_flow_2d-040.pyfrs``
   converting it into an unstructured VTK file called
   ``couette_flow_2d-040.vtu``. Note that in order to visualise the
   high-order data, each high-order element is sub-divided into smaller
   linear elements. The level of sub-division is controlled by the
   integer at the end of the command::

        pyfr export couette_flow_2d.pyfrm couette_flow_2d-040.pyfrs couette_flow_2d-040.vtu -d 4

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

4. Run pyfr to convert the `Gmsh <http:http://geuz.org/gmsh/>`_
   mesh file into a PyFR mesh file called ``euler_vortex_2d.pyfrm``::

        pyfr import euler_vortex_2d.msh euler_vortex_2d.pyfrm

5. Run pyfr to partition the PyFR mesh file into two pieces::

        pyfr partition 2 euler_vortex_2d.pyfrm .

6. Run pyfr to solve the Euler equations on the mesh, generating a
   series of PyFR solution files called ``euler_vortex_2d*.pyfrs``::

        mpirun -n 2 pyfr run -b cuda -p euler_vortex_2d.pyfrm euler_vortex_2d.ini

7. Run pyfr on the solution file ``euler_vortex_2d-100.0.pyfrs``
   converting it into an unstructured VTK file called
   ``euler_vortex_2d-100.0.vtu``. Note that in order to visualise the
   high-order data, each high-order element is sub-divided into smaller
   linear elements. The level of sub-division is controlled by the
   integer at the end of the command::

        pyfr export euler_vortex_2d.pyfrm euler_vortex_2d-100.0.pyfrs euler_vortex_2d-100.0.vtu -d 4

8. Visualise the unstructured VTK file in `Paraview
   <http://www.paraview.org/>`_

.. figure:: ../fig/euler_vortex_2d/euler_vortex_2d.png
   :width: 450px
   :figwidth: 450px
   :alt: euler vortex
   :align: center

   Colour map of density distribution at 100 time units.
