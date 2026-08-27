***************
Developer Guide
***************

This section provides a map of the PyFR source tree.  Detailed API
documentation is intentionally omitted; the implementation and the base
classes it defines are the authoritative references.

Runtime
========

The ``pyfr`` entry point is :func:`pyfr.__main__.main`.  For ``run`` and
``restart``, :func:`pyfr.__main__._process_common` performs the following
steps:

#. Initialise MPI.
#. Load the mesh and, for a restart, the solution.
#. Construct the selected backend.
#. Construct the solver.
#. Run the solver.

Solver construction is handled by :func:`pyfr.solvers.get_solver`.  It
selects the system named in the ``[solver]`` configuration section and
combines it with the requested time integrator.

Source Tree
===========

The principal packages are:

``pyfr.backends``
    Compute backends and their kernel providers.

``pyfr.integrators``
    Explicit and implicit time integrators, controllers, and steppers.

``pyfr.solvers``
    Governing systems, elements, interfaces, and boundary conditions.

``pyfr.readers`` and ``pyfr.writers``
    Mesh, solution, and visualisation I/O.

``pyfr.plugins``
    Runtime, command-line, and post-processing plugins.

``pyfr.partitioners``
    Mesh partitioning implementations.

Extension Model
===============

PyFR selects implementations from subclasses using identifying class
attributes such as ``name``.  New implementations should derive from the
appropriate base class and follow an existing neighbouring implementation.
Configuration keys and user-visible behaviour must be documented in the
corresponding user-guide section.

Kernel Templates
================

Backend and solver kernels are generated from Mako templates.  Backend
templates provide implementation-specific primitives, while solver
templates express numerical operations in terms of those primitives.

PyFR-Mako Kernels
-----------------

A directly invocable pointwise kernel is declared with
``pyfr:kernel``:

.. code-block:: none

    <%pyfr:kernel name='kernel_name' ndim='1'
                  u='in fpdtype_t'
                  f='out fpdtype_t'>
        f = u;
    </%pyfr:kernel>

Each argument specification has the form
``[intent] [attribute] [reduce(op)] data-type[dimensions]``.  The
supported intents are ``in``, ``inout``, and ``out``.  Attributes
include ``broadcast``, ``broadcast-row``, ``broadcast-col``, ``mpi``,
``scalar``, and ``view``.  Reductions support ``min``, ``max``, and
``sum``.  The implementation in
:mod:`pyfr.backends.base.generator` defines the exact constraints.

PyFR-Mako Macros
----------------

Reusable template fragments are declared with ``pyfr:macro``:

.. code-block:: none

    <%pyfr:macro name='copy' params='src, dst'>
        dst = src;
    </%pyfr:macro>

Macros are expanded inside kernels using:

.. code-block:: none

    ${pyfr.expand('copy', 'u', 'f')};

Normal Mako expression substitution, conditionals, and loops are
evaluated while rendering the template.  Thus ``${expression}``
substitutes a Python expression, while ``% if`` and ``% for`` select or
generate source before the backend compiler sees it.  Existing templates
under ``pyfr/solvers`` and ``pyfr/backends`` are the reference for
supported usage.
