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

PyFR currently has a hard dependency on Python 2.7.  PyFR does not currently support Microsoft Windows. To run PyFR it is necessary to install the following Python packages:

1. `mako <http://www.makotemplates.org/>`_
2. `mpi4py <http://mpi4py.scipy.org/>`_ >= 1.3
3. `numpy <http://www.numpy.org/>`_ >= 1.8
4. `mpmath <http://code.google.com/p/mpmath/>`_ >= 0.18

CUDA Backend
^^^^^^^^^^^^

The CUDA backend targets NVIDIA GPUs with a compute capability of 2.0 or
greater. The backend requires:

1. `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 4.2
2. `pycuda <http://mathema.tician.de/software/pycuda/>`_ >= 2011.2

OpenMP Backend
^^^^^^^^^^^^^^
The OpenMP backend targets multi-core CPUs. The backend requires:

1. GCC >= 4.7
2. A BLAS library compiled as a shared library (e.g. `OpenBLAS <http://www.openblas.net/>`_)

Installation
------------

Before running PyFR it is first necessary to either install PyFR using the provided ``setup.py`` installer or add the root PyFR directory to
``PYTHONPATH``::

  user@computer ~/PyFR$ export PYTHONPATH=.:$PYTHONPATH
  user@computer ~/PyFR$ python pyfr/scripts/pyfr-sim --help

Running PyFR
============

Overview
--------

PyFR consists of three separate tools:

1. ``pyfr-mesh`` --- for pre-processing
2. ``pyfr-sim`` --- the solver
3. ``pyfr-postp`` --- for post-processing

PyFR-Mesh
---------

``pyfr-pmesh`` is for pre-processing. The following sub-tools are available:

1. ``pyfr-mesh convert`` --- Convert a `Gmsh <http:http://geuz.org/gmsh/>`_ mesh file into the PyFR format. Example::

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
    
2. ``pyfr-sim restart`` --- Restart a PyFR simulation from an existing solution file. Example::

        pyfr-sim restart mesh.pyfrm solution.pyfrs

For full details invoke:: 

    pyfr-sim [sub-tool] --help        
        
Running in Parallel
^^^^^^^^^^^^^^^^^^^

``pyfr-sim`` can be run in parallel. To do so prefix ``pyfr-sim`` with ``mpirun -n <cores/devices>``. Note that the mesh must be pre-partitioned, and the number of cores or devices must be equal to the number of partitions.

PyFR-PostP
----------

``pyfr-postp`` is for post-processing. The following sub-tools are available:

1. ``pyfr-postp convert`` --- Convert a PyFR solution file into an unstructured VTK file. Example::

        pyfr-postp convert mesh.pyfrm solution.pyfrs solution.vtu divide
        
2. ``pyfr-postp pack`` --- Swap between the pyfr-dir and pyfr-file format. Example::

        pyfr-postp pack solution_directory.pyfrs solution_file.pyfrs
        
3. ``pyfr-postp time-avg`` --- Time-average a series of PyFR solution files. Example::

        pyfr-postp time-avg average.pyfrs t1.pyfrs t2.pyfrs t3.pyfrs
        
4. ``pyfr-postp unpack`` --- Swap between the pyfr-file and pyfr-dir format. Example::

        pyfr-postp unpack solution_file.pyfrs solution_directory.pyfrs

For full details invoke:: 

    pyfr-postp [sub-tool] --help        
        
Example - 2D Couette Flow
=========================

Proceed with the following steps to run a serial 2D Couette flow simulation on a mixed unstructured mesh:

1. Create a working directory called ``couette_flow_2d/``

2. Copy the configuration file ``PyFR/examples/couette_flow_2d/couette_flow_2d.ini`` into ``couette_flow_2d/``

3. Copy the `Gmsh <http:http://geuz.org/gmsh/>`_ mesh file ``PyFR/examples/couette_flow_2d/couette_flow_2d.msh`` into ``couette_flow_2d/``

4. Run pyfr-mesh to covert the `Gmsh <http:http://geuz.org/gmsh/>`_ mesh file into a PyFR mesh file called ``couette_flow_2d.pyfrm``::

    pyfr-mesh convert couette_flow_2d.msh couette_flow_2d.pyfrm

5. Run pyfr-sim to solve the Navier-Stokes equations on the mesh, generating a series of PyFR solution files called ``couette_flow_2d-*.pyfrs``::

    pyfr-sim -p run couette_flow_2d.pyfrm couette_flow_2d.ini

6. Run pyfr-postp on the solution file ``couette_flow_2d_4.00.pyfrs`` converting it into an unstructured VTK file called ``couette_flow_2d_4.00.vtu``. Note that in order to visualise the high-order data, each high-order element is sub-divided into smaller linear elements. The level of sub-division is controlled by the integer at the end of the command::

    pyfr-postp convert couette_flow_2d.pyfrm couette_flow_2d_4.00.pyfrs couette_flow_2d_4.00.vtu divide -d 4

7. Visualise the unstructured VTK file in `Paraview <http://www.paraview.org/>`_

.. figure:: ../fig/couette_flow_2d/couette_flow_2d.png
   :width: 450px
   :figwidth: 450px
   :alt: couette flow
   :align: center

   Colour map of steady-state density distribution.

Example - 2D Euler Vortex
=========================

Proceed with the following steps to run a parallel 2D Euler vortex simulation on a structured mesh:

1. Create a working directory called ``euler_vortex_2d/``

2. Copy the configuration file ``PyFR/examples/euler_vortex_2d/euler_vortex_2d.ini`` into ``euler_vortex_2d/``

3. Copy the partitioned `Gmsh <http:http://geuz.org/gmsh/>`_ file ``PyFR/examples/euler_vortex_2d/euler_vortex_2d.msh`` into ``euler_vortex_2d/``

4. Run pyfr-mesh to convert the `Gmsh <http:http://geuz.org/gmsh/>`_ mesh file into a PyFR mesh file called ``euler_vortex_2d.pyfrm``::

    pyfr-mesh convert euler_vortex_2d.msh euler_vortex_2d.pyfrm

5. Run pyfr-sim to solve the Euler equations on the mesh, generating a series of PyFR solution files called ``euler_vortex_2d*.pyfrs``::

    mpirun -n 2 pyfr-sim -p run euler_vortex_2d.pyfrm euler_vortex_2d.ini

6. Run pyfr-postp on the solution file ``euler_vortex_2d_100.0.pyfrs`` converting it into an unstructured VTK file called ``euler_vortex_2d_100.0.vtu``. Note that in order to visualise the high-order data, each high-order element is sub-divided into smaller linear elements. The level of sub-division is controlled by the integer at the end of the command::

    pyfr-postp convert euler_vortex_2d.pyfrm euler_vortex_2d-100.0.pyfrs euler_vortex_2d_100.0.vtu divide -d 4

7. Visualise the unstructured VTK file in `Paraview <http://www.paraview.org/>`_

.. figure:: ../fig/euler_vortex_2d/euler_vortex_2d.png
   :width: 450px
   :figwidth: 450px
   :alt: euler vortex
   :align: center

   Colour map of density distribution at 100 time units.

