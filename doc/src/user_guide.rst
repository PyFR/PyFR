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

PyFR currently has a hard depndency on Python 2.7.  PyFR does not currently
support Microsoft Windows system. To run PyFR it is necessary to install the
following Python packages:

  - `mako <http://www.makotemplates.org/>`_,
  - `mpi4py <http://mpi4py.scipy.org/>`_ >= 1.3,
  - `numpy <http://www.numpy.org/>`_ >= 1.6,
  - `sympy <http://sympy.org/>`_ >= 0.7.3


CUDA Backend
^^^^^^^^^^^^

The CUDA backend targets NVIDIA GPUs with a compute capability of 2.0 or
later.  This requires CUDA 4.2 or later to be installed and functioning
on the system along with the PyCUDA wrapper.

  - `pycuda <http://mathema.tician.de/software/pycuda/>`_ >= 2011.2

OpenMP Backend
^^^^^^^^^^^^^^

  - GCC >= 4.7
  - A BLAS library compiled as a shared library,
    e.g, `OpenBLAS <http://www.openblas.net/>`_.

Installation
------------

Before running PyFR it is first necessary to
either install PyFR using the provided ``setup.py`` installer or add the
root PyFR directory to
``PYTHONPATH``::

  user@computer ~/PyFR$ export PYTHONPATH=.:$PYTHONPATH
  user@computer ~/PyFR$ python pyfr/scripts/pyfr-sim --help

Running PyFR-Mesh
=================

Overview
--------

Command Line Arguments
----------------------

    -p        Show a progress bar
    -n N      Check for NaNs every N steps
    --progress        Show a progress bar
    --nansweep N      Check for NaNs every N steps

Running PyFR-Sim
================

Overview
--------

Configuration File Format (.ini)
--------------------------------

Command Line Arguments
----------------------

    -p        Show a progress bar
    -n N      Check for NaNs every N steps
    --progress        Show a progress bar
    --nansweep N      Check for NaNs every N steps

Running PyFR-Postp
==================

Overview
--------

Command Line Arguments
----------------------

    -p        Show a progress bar
    -n N      Check for NaNs every N steps
    --progress        Show a progress bar
    --nansweep N      Check for NaNs every N steps
    
2D Couette Flow
===============

Proceed with the following steps to run a 2D Couette Flow simulation:

1. Create a working directory called ``couette_flow/``
2. Copy the file ``PyFR/examples/couette_flow/couette_2d.ini`` into ``couette_flow/``
3. Copy the file ``PyFR/examples/couette_flow/couette_2d_mixed.msh`` into ``couette_flow/``
4. Run pyfr-mesh to covert the mixed quadrilateral-triangular mesh into PyFR-format called ``couette_flow_2d_mixed.pyfrm``

    ``pyfr-mesh convert couette_2d_mixed.msh couette_2d_mixed.pyfrm``

5. Run pyfr-sim to solve the Navier-Stokes equations on the mesh, generating a series of solution files called ``couette_2d-*.pyfrs``

    ``pyfr-sim -p run couette_2d_mixed.pyfrm couette_2d.ini``

6. Run pyfr-postp to generate a series of VTU files called ``couette_2d_mixed-*.vtu``

    ``pyfr-postp convert couette_2d_mixed.pyfrm couette_2d-*.pyfrs couette_2d_mixed-*.vtu``

7. Visualise the VTU files in `Paraview <http://www.paraview.org/>`_

.. figure:: ../fig/couette_flow/couette_flow_2d_steady_state.png
   :width: 450px
   :figwidth: 450px
   :alt: cylinder flow
   :align: center

   Colour map of steady-state density distribution.
    
3D Euler Vortex
===============

Proceed with the following steps to run a 3D Euler vortex simulation:

1. Create a working directory called ``euler_vortex/``
2. Copy the file ``PyFR/examples/euler_vortex/euler_vortex.ini`` into ``euler_vortex/``
3. Run pyfr-mesh to generate a hexahedral mesh with a single partition called ``euler_vortex.pyfrm``

    ``pyfr-mesh .... euler_vortex.pyfrm``

4. Run pyfr-sim to solve Euler's equations on the mesh, generating a series of solution files called ``euler_vortex_*.pyfrs``

    ``pyfr-sim -p run euler_vortex.pyfrm euler_vortex.ini``

5. Run pyfr-postp to generate a series of VTK files called ``euler_vortex_*.vtu``

    ``pyfr-postp .... euler_vortex.pyfrs``

6. Visualise the VTK files in `Paraview <http://www.paraview.org/>`_

.. figure:: ../fig/euler_vortex/euler_vortex.jpg
   :width: 450px
   :figwidth: 450px
   :alt: cylinder flow
   :align: center

   Colour map of density.

3D Cylinder Flow
================

Proceed with the following steps to run a 3D cylinder flow simulation:

1. Create a working directory called ``cylinder_flow/``
2. Copy the file ``PyFR/examples/cylinder_flow/cylinder_flow.ini`` into ``cylinder_flow/``
3. Copy the file ``PyFR/examples/cylinder_flow/cylinder_flow.msh`` into ``cylinder_flow/``
4. Run pyfr-mesh to generate a four partition hexahedral mesh called ``cylinder_flow.pyfrm``

    ``pyfr-mesh .... cylinder_flow.pyfrm``

4. Run pyfr-sim on four nodes to solve the compressible Navier-Stokes equations on the mesh, generating a series of solution files called ``cylinder_flow_*.pyfrs``

    ``mpirun -n 4 pyfr-sim -p run cylinder_flow.pyfrm cylinder_flow.ini``

5. Run pyfr-postp to generate a series of VTK files called ``cylinder_flow_*.vtu``

    ``pyfr-postp .... cylinder_flow.pyfrs``

6. Visualise the VTK files in `Paraview <http://www.paraview.org/>`_

.. figure:: ../fig/cylinder_flow/cylinder_flow.jpg
   :width: 450px
   :figwidth: 450px
   :alt: cylinder flow
   :align: center

   Iso-surfaces of Q-criterion coloured by velocity magnitude.    
