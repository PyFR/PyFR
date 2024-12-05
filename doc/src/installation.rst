.. highlight:: none

************
Installation
************

Quick-start
===========

PyFR |release| can be installed using
`pip <https://pypi.python.org/pypi/pip>`_ and
`virtualenv <https://pypi.python.org/pypi/virtualenv>`_, as shown in the
quick-start guides below.

macOS
-----

It is assumed that the Xcode Command Line Tools and
`Homebrew <https://brew.sh/>`_ are already installed. Follow the steps
below to setup the OpenMP backend on macOS:

#. Install MPI::

        brew install mpi4py

#. Download and install libxsmm and set the library path::

        git clone https://github.com/libxsmm/libxsmm.git
        cd libxsmm
        make -j4 STATIC=0 BLAS=0
        export PYFR_XSMM_LIBRARY_PATH=`pwd`/lib/libxsmm.dylib

#. Make a venv and activate it::

        python3.12 -m venv pyfr-venv
        source pyfr-venv/bin/activate

#. Install PyFR::

        pip install pyfr

#. Add the following to your :ref:`configuration-file`::

        [backend-openmp]
        cc = gcc-13

Note the version of the compiler which must support the ``openmp``
flag. This has been tested on macOS 13.6.2 with an Apple M1 Max.

Ubuntu
------

Follow the steps below to setup the OpenMP backend on Ubuntu:

#. Install Python and MPI::

        sudo apt install python3 python3-pip libopenmpi-dev openmpi-bin
        pip3 install virtualenv

#. Download and install libxsmm and set the library path::

        git clone https://github.com/libxsmm/libxsmm.git
        cd libxsmm
        make -j4 STATIC=0 BLAS=0
        export PYFR_XSMM_LIBRARY_PATH=`pwd`/lib/libxsmm.so

#. Make a virtualenv and activate it::

        python3 -m virtualenv pyfr-venv
        source pyfr-venv/bin/activate

#. Install PyFR::

        pip install pyfr

This has been tested on Ubuntu 22.04.

.. _compile-from-source:

Compiling from source
=====================

PyFR can be obtained
`here <https://github.com/PyFR/PyFR/tree/master>`_.  To install the
software from source, use the provided ``setup.py`` installer or add
the root PyFR directory to ``PYTHONPATH`` using::

    user@computer ~/PyFR$ export PYTHONPATH=.:$PYTHONPATH

When installing from source, we strongly recommend using
`pip <https://pypi.python.org/pypi/pip>`_ and
`virtualenv <https://pypi.python.org/pypi/virtualenv>`_ to manage the
Python dependencies.

Dependencies
------------

PyFR |release| has a hard dependency on Python 3.11+ and the following
Python packages:

#. `gimmik <https://github.com/PyFR/GiMMiK>`_ >= 3.1.1
#. `h5py <https://www.h5py.org/>`_ >= 2.10
#. `mako <https://www.makotemplates.org/>`_ >= 1.0.0
#. `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ >= 4.0
#. `numpy <https://www.numpy.org/>`_ >= 1.26.4
#. `platformdirs <https://pypi.org/project/platformdirs/>`_ >= 2.2.0
#. `pytools <https://pypi.python.org/pypi/pytools>`_ >= 2016.2.1
#. `rtree <https://pypi.org/project/Rtree/>`_ >= 1.0.1

In addition an MPI library supporting version 4 of the MPI standard is
required.

.. _install cuda backend:

CUDA Backend
^^^^^^^^^^^^

The CUDA backend targets NVIDIA GPUs with a compute capability of 3.0
or greater. The backend requires:

#. `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 11.4

HIP Backend
^^^^^^^^^^^

The HIP backend targets AMD GPUs which are supported by the ROCm stack.
The backend requires:

#. `ROCm <https://docs.amd.com/>`_ >= 6.0.0
#. `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ >=
   4.0.0

Metal Backend
^^^^^^^^^^^^^

The Metal backend targets Apple silicon GPUs. The backend requires:

#. `pyobjc-framework-Metal <https://pyobjc.readthedocs.io/en/latest>`_ >= 9.0

OpenCL Backend
^^^^^^^^^^^^^^

The OpenCL backend targets a range of accelerators including GPUs from
AMD, Intel, and NVIDIA. The backend requires:

#. OpenCL >= 2.1
#. Optionally `CLBlast <https://github.com/CNugteren/CLBlast>`_
#. Optionally `TinyTC <https://intel.github.io/tiny-tensor-compiler/>`_
   >= 0.3.1

Note that when running on NVIDIA GPUs the OpenCL backend may terminate
with a segmentation fault after the simulation has finished.  This is
due to a long-standing bug in how the NVIDIA OpenCL implementation
handles sub-buffers.  As it occurs during the termination phase—after
all data has been written out to disk—the issue does *not* impact the
functionality or correctness of PyFR.

.. _install openmp backend:

OpenMP Backend
^^^^^^^^^^^^^^

The OpenMP backend targets multi-core x86-64 and ARM CPUs. The backend
requires:

#. GCC >= 12.0 or another C compiler with OpenMP 5.1 support
#. `libxsmm <https://github.com/hfp/libxsmm>`_ >= commit
   bf5313db8bf2edfc127bb715c36353e610ce7c04 in the ``main`` branch
   compiled as a shared library (STATIC=0) with BLAS=0.

In order for PyFR to find libxsmm it must be located in a directory
which is on the library search path.  Alternatively, the path can be
specified explicitly by exporting the environment variable
``PYFR_XSMM_LIBRARY_PATH=/path/to/libxsmm.so``.

Parallel
^^^^^^^^

To partition meshes for running in parallel it is also necessary to
have one of the following partitioners installed:

#. `METIS <http://glaros.dtc.umn.edu/gkhome/views/metis>`_ >= 5.2
#. `SCOTCH <https://www.labri.fr/perso/pelegrin/scotch/>`_ >= 7.0
#. `KaHIP <https://kahip.github.io/>`_ >= 3.10

In order for PyFR to find these libraries they must be located in a
directory which is on the library search path.  Alternatively, the
paths can be specified explicitly by exporting environment
variables e.g. ``PYFR_METIS_LIBRARY_PATH=/path/to/libmetis.so`` and/or
``PYFR_SCOTCH_LIBRARY_PATH=/path/to/libscotch.so``.

Ascent
^^^^^^

To run the :ref:`soln-plugin-ascent` plugin, MPI, VTK-m, and Conduit are required.
VTK-m is a supplimentary VTK library, and Conduit is a library that implements
the data classes used in Ascent. Detailed information on compilation and installation
of `Conduit <https://llnl-conduit.readthedocs.io>`_ and `Ascent <https://ascent.readthedocs.io>`_ can
be found in the respective documentation. Ascent must be version >=0.9.0.
When compiling Ascent a renderer must be selected to be compiled, currently
PyFR only supports the VTK-h option that comes with Ascent. The paths to the
libraries may need to be set as an environment variable. For example, on linux
you will need::

    PYFR_CONDUIT_LIBRARY_PATH=/path/to/libconduit.so
    PYFR_ASCENT_MPI_LIBRARY_PATH=/path/to/libascent_mpi.so

Currently the plugin requires that Ascent and Conduit are 64-bit, this is
default when compiling in most cases.
