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

Kernels whose argument list depends on template parameters can build a
dictionary of specifications in a Mako code block and pass it via the
``kargs`` attribute:

.. code-block:: none

    <% kargs = {f'x{i}': 'in fpdtype_t' for i in range(nv)} %>
    <%pyfr:kernel name='kernel_name' ndim='1' kargs='${kargs}'
                  f='out fpdtype_t'>
        f = ${' + '.join(f'x{i}' for i in range(nv))};
    </%pyfr:kernel>

Differentiation Regions
^^^^^^^^^^^^^^^^^^^^^^^

A block of code may be differentiated in forward mode by prefixing it
with a ``#pragma pyfr tangent(...)`` or a ``#pragma pyfr jacobian(...)``
directive whose arguments bind the names used by the block to the
storage for their derivatives: ``y -> dy`` names where the derivative
of an output is written and ``x <- dx`` names where the derivative of
an input is read.  A ``tangent`` region computes the directional
derivative of each output from those of the inputs:

.. code-block:: none

    fpdtype_t f[${nvars}], df[${nvars}];
    #pragma pyfr tangent(f -> df, u <- du)
    ${pyfr.expand('flux', 'u', 'f')};

A ``jacobian`` region takes a single output followed by its array
inputs, each bound to the tile which receives the derivatives of every
entry of the output with respect to every entry of that input:

.. code-block:: none

    fpdtype_t f[${nvars}], jfu[${nvars}][${nvars}];
    #pragma pyfr jacobian(f, u -> jfu)
    ${pyfr.expand('flux', 'u', 'f')};

Bound storage is declared before the region with the type of its
primal: a tangent has the extents of its output and a tile those of
the output followed by the number of entries of its input in row major
order.  A region writes only the derivatives it computes, so storage
may be shared by the branches of a conditional or accumulated by a
loop around the region; an output is a constant on entry unless it
also binds its storage as an input.  Conditionals are differentiated
branch by branch with the condition held constant.  Loops are allowed
in ``tangent`` regions only, the output of a ``jacobian`` region may
not also be an input and regions may not nest.

Statements in a nested ``#pragma pyfr nodiff`` block are copied
without differentiation and the names they write are constants, which
keeps piecewise constant quantities such as limiter switches out of
the derivative.

Where a function is not differentiable ``fabs`` differentiates to
``copysign(1, x)``, ``fmin`` and ``fmax`` follow their first argument
on ties, ``sign``, comparisons and casts to integer types are
constants, ``?:`` differentiates only its branches and ``%`` is not
differentiable.

Requesting a kernel with ``tangent={'u': 'du', ...}`` wraps its body
in a ``tangent`` region: each mapped argument gains a twin of the same
specification under the given name, the twins of ``in`` and ``inout``
arguments are read, those of ``out`` and ``inout`` arguments written,
and every output must be mapped.  Mapped arguments must be floating
point and the kernel may have no reduction arguments.

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
