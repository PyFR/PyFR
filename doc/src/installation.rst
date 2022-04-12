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

Alternatively, PyFR |release| can be installed from
`source <https://github.com/PyFR/PyFR/tree/master>`_, see
:ref:`compile-from-source`.

macOS
-----

We recommend using the package manager `homebrew <https://brew.sh/>`_.
Open the terminal and install the dependencies with the following
commands::

    brew install python3 open-mpi metis
    pip3 install virtualenv

For visualisation of results, either install ParaView from the command
line::

    brew cask install paraview

or download the app from the ParaView
`website <https://www.paraview.org/>`_. Then create a virtual
environment and activate it::

    virtualenv --python=python3 ENV3
    source ENV3/bin/activate

Finally, install PyFR with `pip <https://pypi.python.org/pypi/pip>`_
in the virtual environment::

    pip install pyfr

This concludes the installation. In order to run PyFR with the OpenMP
backend (see :ref:`running-pyfr`), use the following settings in the
:ref:`configuration-file`::

    [backend-openmp]
    cc = gcc-8

Note the version of the compiler which must support the ``openmp``
flag. This has been tested on macOS 11.6 for ARM and Intel CPUs.

Ubuntu
------

Open the terminal and install the dependencies with the following
commands::

    sudo apt install python3 python3-pip libopenmpi-dev openmpi-bin
    sudo apt install metis libmetis-dev
    pip3 install virtualenv

For visualisation of results, either install ParaView from the command
line::

    sudo apt install paraview

or download the app from the ParaView
`website <https://www.paraview.org/>`_.  Then create a virtual
environment and activate it::

    python3 -m virtualenv pyfr-venv
    source pyfr-venv/bin/activate

Finally, install PyFR with
`pip <https://pypi.python.org/pypi/pip>`_ in the virtual environment::

    pip install pyfr

This concludes the installation.

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

1. `gimmik <https://github.com/PyFR/GiMMiK>`_ >= 2.3
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

1. `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 8.0

HIP Backend
^^^^^^^^^^^

The HIP backend targets AMD GPUs which are supported by the ROCm stack.
The backend requires:

1. `ROCm <https://rocmdocs.amd.com/en/latest/>`_ >= 4.5.0
2. `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ >=
   2.41.0

OpenCL Backend
^^^^^^^^^^^^^^

The OpenCL backend targets a range of accelerators including GPUs from
AMD, Intel, and NVIDIA. The backend requires:

1. OpenCL
2. Optionally `CLBlast <https://github.com/CNugteren/CLBlast>`_

Note that when running on NVIDIA GPUs the OpenCL backend terminate with
a segmentation fault after the simulation has finished.  This is due
to a long-standing bug in how the NVIDIA OpenCL implementation handles
sub-buffers.  As it occurs during the termination phase—after all data
has been written out to disk—the issue does *not* impact the
functionality or correctness of PyFR.

.. _install openmp backend:

OpenMP Backend
^^^^^^^^^^^^^^

The OpenMP backend targets multi-core CPUs. The backend requires:

1. GCC >= 4.9 or another C compiler with OpenMP support
2. `libxsmm <https://github.com/hfp/libxsmm>`_ >= commit
   14b6cea61376653b2712e3eefa72b13c5e76e421 compiled as a shared
   library (STATIC=0) with BLAS=0 and CODE_BUF_MAXSIZE=262144

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
