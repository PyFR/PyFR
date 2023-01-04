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

1. Install MPI::

        brew install mpi4py

2. Install METIS and set the library path::

        brew install metis
        export PYFR_METIS_LIBRARY_PATH=/opt/homebrew/lib/libmetis.dylib

3. Download and install libxsmm and set the library path::

        git clone git@github.com:libxsmm/libxsmm.git
        cd libxsmm
        make -j4 STATIC=0 BLAS=0
        export PYFR_XSMM_LIBRARY_PATH=`pwd`/lib/libxsmm.dylib

4. Make a venv and activate it::

        python3.10 -m venv pyfr-venv
        source pyfr-venv/bin/activate

5. Install PyFR::

        pip install pyfr

6. Add the following to your :ref:`configuration-file`::

        [backend-openmp]
        cc = gcc-12

Note the version of the compiler which must support the ``openmp``
flag. This has been tested on macOS 12.5 with an Apple M1 Max.

Ubuntu
------

Follow the steps below to setup the OpenMP backend on Ubuntu:

1. Install Python and MPI::

        sudo apt install python3 python3-pip libopenmpi-dev openmpi-bin
        pip3 install virtualenv

2. Install METIS::

        sudo apt install metis libmetis-dev

3. Download and install libxsmm and set the library path::

        git clone git@github.com:libxsmm/libxsmm.git
        cd libxsmm
        make -j4 STATIC=0 BLAS=0
        export PYFR_XSMM_LIBRARY_PATH=`pwd`/lib/libxsmm.so

4. Make a virtualenv and activate it::

        python3 -m virtualenv pyfr-venv
        source pyfr-venv/bin/activate

5. Install PyFR::

        pip install pyfr

This has been tested on Ubuntu 20.04.

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

PyFR |release| has a hard dependency on Python 3.9+ and the following
Python packages:

1. `gimmik <https://github.com/PyFR/GiMMiK>`_ >= 3.0
2. `h5py <https://www.h5py.org/>`_ >= 2.10
3. `mako <https://www.makotemplates.org/>`_ >= 1.0.0
4. `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ >= 3.0
5. `numpy <https://www.numpy.org/>`_ >= 1.20
6. `platformdirs <https://pypi.org/project/platformdirs/>`_ >= 2.2.0
7. `pytools <https://pypi.python.org/pypi/pytools>`_ >= 2016.2.1

Note that due to a bug in NumPy, PyFR is not compatible with 32-bit
Python distributions.

.. _install cuda backend:

CUDA Backend
^^^^^^^^^^^^

The CUDA backend targets NVIDIA GPUs with a compute capability of 3.0
or greater. The backend requires:

1. `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 11.4

HIP Backend
^^^^^^^^^^^

The HIP backend targets AMD GPUs which are supported by the ROCm stack.
The backend requires:

1. `ROCm <https://docs.amd.com/>`_ >= 5.2.0
2. `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ >=
   2.41.0

OpenCL Backend
^^^^^^^^^^^^^^

The OpenCL backend targets a range of accelerators including GPUs from
AMD, Intel, and NVIDIA. The backend requires:

1. OpenCL >= 2.1
2. Optionally `CLBlast <https://github.com/CNugteren/CLBlast>`_

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

1. GCC >= 12.0 or another C compiler with OpenMP 5.1 support
2. `libxsmm <https://github.com/hfp/libxsmm>`_ >= commit
   0db15a0da13e3d9b9e3d57b992ecb3384d2e15ea compiled as a shared
   library (STATIC=0) with BLAS=0.

In order for PyFR to find libxsmm it must be located in a directory
which is on the library search path.  Alternatively, the path can be
specified explicitly by exporting the environment variable
``PYFR_XSMM_LIBRARY_PATH=/path/to/libxsmm.so``.

Parallel
^^^^^^^^

To partition meshes for running in parallel it is also necessary to
have one of the following partitioners installed:

1. `METIS <http://glaros.dtc.umn.edu/gkhome/views/metis>`_ >= 5.0
2. `SCOTCH <http://www.labri.fr/perso/pelegrin/scotch/>`_ >= 6.0

In order for PyFR to find these libraries they must be located in a
directory which is on the library search path.  Alternatively, the
paths can be specified explicitly by exporting the environment
variables ``PYFR_METIS_LIBRARY_PATH=/path/to/libmetis.so`` and/or
``PYFR_SCOTCH_LIBRARY_PATH=/path/to/libscotch.so``.
