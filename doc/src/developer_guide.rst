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
``scalar``, ``tiled``, and ``view``.  Reductions support ``min``,
``max``, and ``sum``.  The implementation in
:mod:`pyfr.backends.base.generator` defines the exact constraints.

A ``reduce(op)`` argument is assigned to like any output and the
contributions of the items are combined with ``op``: a ``view`` into
the entries it refers to, a ``broadcast-col`` into one value per
column, and a ``broadcast`` into a single vector.  The caller
initialises an ``inout`` reduction with the identity of ``op``.

A ``tiled`` argument is a matrix per item of a one dimensional kernel,
indexed as ``m[r][c]`` and bound to a matrix from ``tiled_matrix``;
the backend chooses how the tiles are laid out.

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

Directives
^^^^^^^^^^

A ``#pragma pyfr`` directive applies to the block or single statement
which follows it:

.. code-block:: none

    #pragma pyfr unroll(${nvars})
    for (int i = 0; i < ${nvars}; i++)
        f[i] = u[i]*${factor};

``unroll(n)`` and ``simd`` are hints which each backend lowers to the
equivalent of its compiler or drops.  ``fp-precise`` has its block
evaluated as written, free of reassociation and contraction.  Any
other directive is emitted as written.

Cooperative Kernels
^^^^^^^^^^^^^^^^^^^

A one dimensional kernel may assign a group of lanes to each item of its
iteration range with ``coop='N'``, ``N`` being the number of lanes the
body can use on one item.  In the body ``@lidx`` and ``@lsz`` give the
index of the lane and the size of the group, ``@sync();`` is a barrier
every lane must reach, and storage shared by the group is declared at
the top level with the ``shared`` qualifier:

.. code-block:: none

    <%pyfr:kernel name='kernel_name' ndim='1' coop='${str(npts)}'
                  u='inout fpdtype_t[${str(npts)}][${str(nvars)}]'
                  vdm='in broadcast fpdtype_t[${str(npts)}][${str(npts)}]'>
        shared fpdtype_t s_u[${npts}][${nvars}];
        for (int i = @lidx; i < ${npts*nvars}; i += @lsz)
            s_u[i / ${nvars}][i % ${nvars}] = u[i / ${nvars}][i % ${nvars}];
        @sync();
        for (int i = @lidx; i < ${npts*nvars}; i += @lsz)
        {
            fpdtype_t a = 0.0;
            for (int k = 0; k < ${npts}; k++)
                a += vdm[i / ${nvars}][k]*s_u[k][i % ${nvars}];
            u[i / ${nvars}][i % ${nvars}] = a;
        }
    </%pyfr:kernel>

Shared storage is limited to the ``shared_budget`` of the backend.  On
the OpenMP backend a cooperative body whose control flow does not
depend on the item, and which writes no broadcast argument, is
vectorised across the items of a SoA group.  Broadcast arguments may
have a single dimension, in which case they are indexed directly and
the matrix must have a single row.

``coop='fit'`` has the backend place as many lane-groups in a block as
fit the budget, one item each.  Loads and ``@sync();`` are then carried
out by the whole block, so a barrier or reduction can not sit in a
conditional.  The OpenMP backend ignores the grouping.

Shared storage may instead be loaded from an ``in`` or ``inout``
per-item, broadcast or tiled array argument with
``shared s = @load(a);``, which takes the extents and, unless written
``shared T s = @load(a);``, the type of ``a``.  The load may also be
an expression of several arguments, ``shared s = @load(a*(x0 + x1));``,
evaluated entry by entry with NumPy broadcasting and of type
``fpdtype_t`` unless given.  Loaded storage is used only after the
``@sync();`` which follows it, so the staging loop above is
``shared s_u = @load(u);``.

``@store(y, expr);`` likewise has the block write an ``out`` or
``inout`` per-item array argument, ``expr`` being an expression of
shared arrays and arguments which may include ``y`` itself, as in
``@store(y, b*y + osc*ys);``.  The storage a store reads may not be
written between the ``@sync();`` before the store and the one after it.
Only a top-level ``@sync();`` counts as the barrier a load or store
requires; one nested in a loop or conditional does not.

A block of one lane, or one whose copies of this storage exceed the
budget, works on the arguments in place of it where it can.  A block of
several lanes does so for the storage it stores first, and for the
storage it loads only if its copies still exceed the budget.  Storage
which is loaded once and only read is read through the expression it
loads, and storage which is stored once, to an argument of its extents
and type which the body does not otherwise name, is that argument.  An
argument a kernel stores to must therefore not be bound to the same
matrix as one whose entries it loads.

A two dimensional kernel may stage a ``broadcast-col`` array argument
in storage shared by its block with the same load; the block fills the
storage, with a barrier of its own, before walking its rows and the
body reads it in place of the argument.

``@dot`` initialises a declaration with the dot product of row ``r``
of a two dimensional argument or shared array ``m`` and a vector
``v``, accumulated in the declared type ``T``:

``T a = @dot(m[r], v);``
    The product of the row with ``v``.

``T a[n] = @dot(m[r], v);``
    The products of the row with the ``n`` vectors interleaved in
    ``v`` as ``v[k*n + j]`` or ``v[k][j]``.

``T a[n] = @dot(w*m[r], v);``
    Either of the above with the row scaled entry by entry by the
    weights ``w``, of extent ``[ncol]``.

``T a[p][n] = @dot(w*m[r], v);``
    The same for each row in turn of weights of extents ``[p][ncol]``.

Operands of type ``bf16`` are converted entry by entry; ``T`` cannot
be ``bf16``.  The row of a tiled argument is walked tile by tile, which
is quickest for dot products that open the body of a loop from
``@lidx`` in steps of ``@lsz``.  Together these constructs apply a
matrix per item, staging
the output so the block writes its items together:

.. code-block:: none

    <%pyfr:kernel name='kernel_name' ndim='1' coop='fit'
                  x='in fpdtype_t[${str(npts)}][${str(nvars)}]'
                  minv='in tiled fpdtype_t[${str(n)}][${str(n)}]'
                  y='out fpdtype_t[${str(npts)}][${str(nvars)}]'>
        shared xs = @load(x);
        shared fpdtype_t ys[${npts}][${nvars}];
        @sync();
        for (int r = @lidx; r < ${n}; r += @lsz)
        {
            fpdtype_t yr = @dot(minv[r], xs);
            ys[r / ${nvars}][r % ${nvars}] = yr;
        }
        @sync();
        @store(y, ys);
    </%pyfr:kernel>

Values held by each lane of a group are combined across the group with
a ``#pragma pyfr allreduce(a, b, ...)`` block, where ``a, b, ...`` are
scalar variables declared in the body and, within the block,
``@partner(a)`` is the value of ``a`` held by the partner lane; the
block folds the partner values into ``a`` and ``b``, and afterwards
every lane holds the combined value of each variable.  A sum is
``a += @partner(a);`` and a minimum ``a = fmin(a, @partner(a));``, and
several variables are combined together when one decides the others:

.. code-block:: none

    fpdtype_t g = ${fpdtype_max};
    int imin = ${npts};
    for (int i = @lidx; i < ${npts}; i += @lsz)
        if (u[i][0] < g)
        {
            g = u[i][0];
            imin = i;
        }
    #pragma pyfr allreduce(g, imin)
    {
        fpdtype_t g_o = @partner(g);
        int imin_o = @partner(imin);
        if (g_o < g)
        {
            g = g_o;
            imin = imin_o;
        }
    }

Like ``@sync();`` a reduction must be reached by every lane of the
group.  The exchange is a butterfly between the lanes through shared
storage, counted against the budget, with a barrier either side of
each exchange.

Within a ``tangent`` or ``jacobian`` region, and so in a kernel
requested with ``tangent``, shared storage can not depend on the inputs
of the region, and a call such as ``@load`` can name neither what
depends on them nor an output.

The lanes of an item are modelled by
:class:`~pyfr.backends.base.generator.lanes.LaneGroup` and, for a two
dimensional kernel, :class:`~pyfr.backends.base.generator.lanes.RowLanes`
or the :class:`~pyfr.backends.base.generator.lanes.StagedRowLanes` of
the GPU backends, which size the group, hoist the shared declarations
and lower the lane intrinsics, loads and reductions; the kernel
generators render what the lanes supply.

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
