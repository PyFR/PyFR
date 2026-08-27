************
Installation
************

Quick-start
===========

PyFR |release| can be installed using
`pip <https://pypi.python.org/pypi/pip>`_ and
`venv <https://docs.python.org/3/library/venv.html>`_, as shown in the
quick-start guides below.

macOS
-----

It is assumed that the Xcode Command Line Tools and
`Homebrew <https://brew.sh/>`_ are already installed. Follow the steps
below to setup the OpenMP backend on macOS:

#. Install GCC and MPI:

   .. code-block:: shell

       brew install gcc open-mpi

#. Download and install libxsmm and set the library path:

   .. code-block:: shell

       git clone https://github.com/libxsmm/libxsmm.git
       cd libxsmm
       make -j4
       export PYFR_XSMM_LIBRARY_PATH=`pwd`/lib/libxsmm.dylib

#. Make a venv and activate it:

   .. code-block:: shell

       python3.12 -m venv pyfr-venv
       source pyfr-venv/bin/activate

#. Install PyFR:

   .. code-block:: shell

       pip install pyfr

#. Add the following to your :ref:`configuration-file`:

   .. code-block:: ini

       [backend-openmp]
       cc = gcc-15

The compiler must support OpenMP 5.1.

Ubuntu
------

Follow the steps below to setup the OpenMP backend on Ubuntu:

#. Install Python and MPI:

   .. code-block:: shell

       sudo apt install build-essential git python3 python3-pip python3-venv
       sudo apt install libopenmpi-dev openmpi-bin

#. Download and install libxsmm and set the library path:

   .. code-block:: shell

       git clone https://github.com/libxsmm/libxsmm.git
       cd libxsmm
       make -j4
       export PYFR_XSMM_LIBRARY_PATH=`pwd`/lib/libxsmm.so

#. Make a venv and activate it:

   .. code-block:: shell

       python3 -m venv pyfr-venv
       source pyfr-venv/bin/activate

#. Install PyFR:

   .. code-block:: shell

       pip install pyfr

These instructions target Ubuntu 26.04.

.. _compile-from-source:

Compiling from source
=====================

PyFR can be obtained
`here <https://github.com/PyFR/PyFR/tree/develop>`_.  To install the
software from source, run the following from the repository root:

.. code-block:: shell

    pip install .

When installing from source, we strongly recommend using
`pip <https://pypi.python.org/pypi/pip>`_ and
`venv <https://docs.python.org/3/library/venv.html>`_ to manage the
Python dependencies.

Dependencies
------------

PyFR |release| has a hard dependency on Python 3.12+ and the following
Python packages:

#. `boostree <https://github.com/PyFR/Boostree>`_ >= 0.3.0
#. `gimmik <https://github.com/PyFR/GiMMiK>`_ >= 4.0
#. `h5py <https://www.h5py.org/>`_ >= 2.10
#. `mako <https://www.makotemplates.org/>`_ >= 1.0.0
#. `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ >= 4.0
#. `numpy <https://www.numpy.org/>`_ >= 2.4.2
#. `platformdirs <https://pypi.org/project/platformdirs/>`_ >= 2.2.0

In addition an MPI library supporting version 4 of the MPI standard is
required.

.. _install cuda backend:

CUDA Backend
^^^^^^^^^^^^

The CUDA backend targets NVIDIA GPUs with a compute capability of 3.5
or greater. The backend requires:

#. `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 11.4

HIP Backend
^^^^^^^^^^^

The HIP backend targets AMD GPUs which are supported by the ROCm stack.
The backend requires:

#. `ROCm <https://docs.amd.com/>`_ >= 7.14
#. `rocBLAS <https://github.com/ROCm/rocBLAS>`_ >= 5.5.0

Metal Backend
^^^^^^^^^^^^^

The Metal backend targets Apple silicon GPUs. The backend requires:

#. `pyobjc-framework-Metal <https://pyobjc.readthedocs.io/en/latest>`_ >= 12.0

OpenCL Backend
^^^^^^^^^^^^^^

The OpenCL backend targets a range of accelerators including GPUs from
AMD, Intel, and NVIDIA. The backend requires:

#. OpenCL >= 2.1
#. Optionally `CLBlast <https://github.com/CNugteren/CLBlast>`_
#. Optionally `TinyTC <https://intel.github.io/tiny-tensor-compiler/>`_
   >= 0.3.1

.. note::

   When running on NVIDIA GPUs the OpenCL backend may terminate with a
   segmentation fault after the simulation has finished.  This is due
   to a long-standing bug in how the NVIDIA OpenCL implementation
   handles sub-buffers.  As it occurs during the termination phase ---
   after all data has been written out to disk --- the issue does *not*
   impact the functionality or correctness of PyFR.

.. _install openmp backend:

OpenMP Backend
^^^^^^^^^^^^^^

The OpenMP backend targets multi-core x86-64 and ARM CPUs. The backend
requires:

#. GCC >= 12.0 or another C compiler with OpenMP 5.1 support
#. `libxsmm <https://github.com/libxsmm/libxsmm>`_ >= 2.0.0

In order for PyFR to find libxsmm it must be located in a directory
which is on the library search path.  Alternatively, the path can be
specified explicitly by exporting the environment variable
``PYFR_XSMM_LIBRARY_PATH=/path/to/libxsmm.so``.

Parallel
^^^^^^^^

PyFR includes a baseline partitioner which requires no external
libraries.  The following optional partitioners are also supported:

#. `METIS <https://github.com/KarypisLab/METIS>`_ >= 5.2
#. `SCOTCH <https://www.labri.fr/perso/pelegrin/scotch/>`_ >= 7.0
#. `KaHIP <https://kahip.github.io/>`_ >= 3.24

In order for PyFR to find these libraries they must be located in a
directory which is on the library search path.  Alternatively, the
paths can be specified explicitly by exporting environment
variables e.g. ``PYFR_METIS_LIBRARY_PATH=/path/to/libmetis.so``,
``PYFR_SCOTCH_LIBRARY_PATH=/path/to/libscotch.so``, and/or
``PYFR_KAHIP_LIBRARY_PATH=/path/to/libkahip.so``.

Ascent
^^^^^^

To run the :ref:`soln-plugin-ascent` plugin, MPI, VTK-m, and Conduit are required.
VTK-m is a supplementary VTK library, and Conduit is a library that implements
the data classes used in Ascent. Detailed information on compilation and installation
of `Conduit <https://llnl-conduit.readthedocs.io>`_ and `Ascent <https://ascent.readthedocs.io>`_ can
be found in the respective documentation. Ascent must be version >=0.9.0.
When compiling Ascent a renderer must be selected to be compiled, currently
PyFR only supports the VTK-h option that comes with Ascent. The paths to the
libraries may need to be set as an environment variable. For example, on linux
you will need:

.. code-block:: shell

    PYFR_CONDUIT_LIBRARY_PATH=/path/to/libconduit.so
    PYFR_ASCENT_MPI_LIBRARY_PATH=/path/to/libascent_mpi.so

Currently the plugin requires that Ascent and Conduit are 64-bit, this is
default when compiling in most cases.
