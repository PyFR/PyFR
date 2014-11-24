###############
Developer Guide
###############

************
Introduction
************

A detailed developer guide is provided below


.. _devg-internal-file-formats:

*********************
Internal File Formats
*********************

Mesh Format (.pyfrm)
====================

Solution Format (.pyfrs)
========================

**********
Unit Tests
**********

****
Code
****

Backend Interface
=================

All of the backends in PyFR implement a common interface.  This
interface is based around the `factory method pattern
<http://en.wikipedia.org/wiki/Factory_method_pattern>`_ and consists
of a ``Backend`` class (the factory) and a series of types (products
of the factory).  Broadly speaking these types can be placed into one
of three categories: matrices, kernels and queues.

All algorithms in PyFR are implemented according to the following
procedure.  Firstly a series of matrices are allocated and populated
with suitable initial values. Next a set of kernels are defined to
operate these matrices.  Then one or more queues are allocated to
execute these kernels.  Finally, these queues are populated with the
relevant kernels and run.  As an example of this process consider
matrix multiplication::

  import numpy as np

  # Take be to be a concrete Backend instance
  be = get_a_backend()

  # Allocate two 1024 by 1024 matrices filled with random numbers
  m1 = be.const_matrix(np.random.rand(1024, 1024))
  m2 = be.const_matrix(np.random.rand(1024, 1024))

  # Allocate a third empty matrix
  m3 = be.matrix(1024, 1024)

  # Prepare a kernel to multiply m1 by m2 putting the result in m3
  mmul = be.kernel('mul', m1, m2, out=m3)

  # Allocate a queue
  q = be.queue()

  # Populate the queue with `mmul` and run it
  q % [mmul()]

.. COMMENTOUT-ALTEREDPATH autoclass:: pyfr.backends.base.Backend
    :members:

.. autoclass:: pyfr.backends.base.Matrix
    :members:
    :inherited-members:

.. autoclass:: pyfr.backends.base.ConstMatrix
    :members:
    :inherited-members:

.. COMMENTOUT-ALTEREDPATH autoclass:: pyfr.backends.base.SparseMatrix
    :members:
    :inherited-members:

.. autoclass:: pyfr.backends.base.XchgMatrix
    :members:
    :inherited-members:

.. autoclass:: pyfr.backends.base.MatrixBank()
    :members:
    :inherited-members:

.. autoclass:: pyfr.backends.base.View
    :members:

.. autoclass:: pyfr.backends.base.XchgView
    :members:

.. COMMENTOUT-ALTEREDPATH autoclass:: pyfr.backends.base.Kernel
    :members:

.. autoclass:: pyfr.backends.base.Queue
    :members: __lshift__, __mod__

Readers
=======

Functionality for PyFR to read internal and external file formats is
provided here.

:ref:`devg-internal-file-formats` have two base structures; file and
directory. These structures allow for convenience and performance
respectively, but require different treatments. It is the purpose of
the :py:mod:`pyfr.readers.native` module to present a common and
convenient interface for general use in PyFR.

Native
------

.. automodule:: pyfr.readers.native
    :members:

Gmsh
----

Utilities
---------

Documentation for the utility modules contained in the readers folder.

Node Maps
^^^^^^^^^

.. automodule:: pyfr.readers.nodemaps
    :members:
