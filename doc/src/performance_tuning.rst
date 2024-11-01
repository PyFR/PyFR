.. highlight:: none

******************
Performance Tuning
******************

The following sections contain best practices for *tuning* the
performance of PyFR.  Note, however, that it is typically not worth
pursuing the advice in this section until a simulation is working
acceptably and generating the desired results.

.. _perf openmp backend:

OpenMP Backend
==============

AVX-512
-------

When running on an AVX-512 capable CPU Clang and GCC will, by default,
only make use of 256-bit vectors.  Given that the kernels in PyFR
benefit meaningfully from longer vectors it is desirable to override
this behaviour.  This can be accomplished through the ``cflags`` key
as::

        [backend-openmp]
        cflags = -mprefer-vector-width=512

Cores vs. threads
-----------------

PyFR does not typically derive any benefit from SMT.  As such the number
of OpenMP threads should be chosen to be equal to the number of physical
cores.

Loop Scheduling
---------------

By default PyFR employs static scheduling for loops, with work being
split evenly across cores.  For systems with a single type of core this
is usually the right choice.  However, on heterogeneous systems it
typically results in load imbalance.  Here, it can be beneficial to
experiment with the *guided* and *dynamic* loop schedules as::

        [backend-openmp]
        schedule = dynamic, 5

MPI processes vs. OpenMP threads
--------------------------------

When using the OpenMP backend it is recommended to employ *one MPI rank
per NUMA zone*.  For most systems each socket represents its own NUMA
zone.  Thus, on a two socket system it is suggested to run PyFR with two
MPI ranks, with each process being bound to a single socket.  The
specifics of how to accomplish this depend on both the job scheduler and
MPI distribution.

Asynchronous MPI progression
----------------------------

The parallel scalability of the OpenMP backend depends *heavily* on MPI
having support for asynchronous progression; that is to say the ability
for non-blocking send and receive requests to complete *without* the
need for the host application to make explicit calls into MPI routines.
A lack of support for asynchronous progression prevents PyFR from being
able to overlap computation with communication.

.. _perf cuda backend:

CUDA Backend
============

CUDA-aware MPI
--------------

PyFR is capable of taking advantage of CUDA-aware MPI.  This enables
CUDA device pointers to be directly to passed MPI routines.  Under the
right circumstances this can result in improved performance for
simulations which are near the strong scaling limit.  Assuming mpi4py
has been built against an MPI distribution which is CUDA-aware this
functionality can be enabled through the ``mpi-type`` key as::

        [backend-cuda]
        mpi-type = cuda-aware

.. _perf hip backend:

HIP Backend
===========

HIP-aware MPI
-------------

PyFR is capable of taking advantage of HIP-aware MPI.  This enables HIP
device pointers to be directly to passed MPI routines.  Under the right
circumstances this can result in improved performance for simulations
which are near the strong scaling limit.  Assuming mpi4py has been built
against an MPI distribution which is HIP-aware this functionality can be
enabled through the ``mpi-type`` key as::

        [backend-hip]
        mpi-type = hip-aware

Partitioning
============

METIS vs SCOTCH vs KaHIP
------------------------

The partitioning module in PyFR includes support for both METIS, SCOTCH,
and KaHIP.  All three usually result in high-quality decompositions.
However, for long running simulations on complex geometries it may be
worth partitioning a grid with each and observing which decomposition
performs best.

.. _perf mixed grids:

Mixed grids
-----------

When running PyFR in parallel on mixed element grids it is necessary
to take some additional care when partitioning the grid.  A good domain
decomposition is one where each partition contains the same amount of
computational work.  For grids with a single element type the amount of
computational work is very well approximated by the number of elements
assigned to a partition.  Thus the goal is simply to ensure that all of
the partitions have roughly the same number of elements.  However, when
considering mixed grids this relationship begins to break down since the
computational cost of one element type can be appreciably more than that
of another.

There are two main solutions to this problem.  The first is to require
that each partition contain the same number of elements of each type.
For example, if partitioning a mesh with 500 quadrilaterals and
1,500 triangles into two parts, then a sensible goal is to aim for
each domain to have 250 quadrilaterals and 750 triangles.  Irrespective
of what the relative performance differential between the element types
is, both partitions will have near identical amounts of work.  In PyFR
this is known as the *balanced* approach and can be requested via::

    pyfr partition add -e balanced ...

This approach typically works well when the number of partitions is
small.  However, for larger partition counts it can become difficult to
achieve such a balance whilst simultaneously minimising communication
volume.  Thus, in order to obtain a good decomposition a secondary
approach is required in which each type of element in the domain is
assigned a *weight*.  Element types which are more computationally
intensive are assigned a larger weight than those that are less
intensive.  Through this mechanism the total cost of each partition can
remain balanced.  Unfortunately, the relative cost of different element
types depends on a variety of factors, including:

 - The polynomial order.
 - If anti-aliasing is enabled in the simulation, and if so, to what
   extent.
 - The hardware which the simulation will be run on.

Weights can be specified when partitioning the mesh as
``-e shape:weight``.  For example, if on a particular system a
quadrilateral is found to be 50% more expensive than a triangle this
can be specified as::

        pyfr partition add -e quad:3 -e tri:2 ...

If precise profiling data is not available regarding the performance of
each element type in a given configuration a helpful rule of thumb is
to under-weight the dominant element type in the domain.  For example,
if a domain is 90% tetrahedra and 10% prisms then, absent any
additional information about the relative performance of tetrahedra and
prisms, a safe choice is to assume the prisms are appreciably *more*
expensive than the tetrahedra.

Detecting load imbalances
-------------------------

PyFR includes code for monitoring the amount of time each rank spends
waiting for MPI transfers to complete.  This can be used, among other
things, to detect load imbalances.  Such imbalances are typically
observed on mixed-element grids with an incorrect weighting factor.
Wait time tracking can be enabled as::

        [backend]
        collect-wait-times = true

with the resulting statistics being recorded in the
``[backend-wait-times]`` section of the ``/stats`` object which is
included in all PyFR solution files.  This can be extracted as::

        h5dump -d /stats -b --output=stats.ini soln.pyfrs

Note that the number of graphs depends on the system, and not all graphs
initiate MPI requests.  The average amount of time each rank spends
waiting for MPI requests per right hand side evaluation can be obtained
by vertically summing all of the ``-median`` fields together.

There exists an inverse relationship between the amount of computational
work a rank has to perform and the amount of time it spends waiting for
MPI requests to complete.  Hence, ranks which spend comparatively less
time waiting than their peers are likely to be overloaded, whereas those
which spend comparatively more time waiting are likely to be
underloaded.  This information can then be used to explicitly re-weight
the partitions and/or the per-element weights.

Scaling
=======

The general recommendation when running PyFR in parallel is to aim for a
parallel efficiency of :math:`\epsilon \simeq 0.8` with the parallel
efficiency being defined as:

.. math::

  \epsilon = \frac{1}{N}\frac{T_1}{T_N},

where :math:`N` is the number of ranks, :math:`T_1` is the simulation
time with one rank, and :math:`T_N` is the simulation time with
:math:`N` ranks.  This represents a reasonable trade-off between the
overall time-to-solution and efficient resource utilisation.

Plugins
=======

A common source of performance issues is running plugins too frequently.
PyFR records the amount of time spent in plugins in the
``[solver-time-integrator]`` section of the ``/stats`` object which is
included in all PyFR solution files.  This can be extracted as::

    h5dump -d /stats -b --output=stats.ini soln.pyfrs

Here, the *common* field contains the amount of time spent obtaining
properties which are not directly attributable to any specific plugin.
Examples include fetching the solution, computing its gradient, and
computing its time derivative.  The *other* field accounts for time
spent in unnamed plugins such as the progress bar.

Given the time steps taken by PyFR are typically much smaller than those
associated with the underlying physics there is seldom any benefit to
running integration and/or time average accumulation plugins more
frequently than once every 50 steps.  Further, when running with
adaptive time stepping there is no need to run the NaN check plugin.
For simulations with fixed time steps, it is not recommended to run the
NaN check plugin more frequently than once every 10 steps.

Start-up Time
=============

The start-up time required by PyFR can be reduced by ensuring that
Python is compiled from source with profile guided optimisations (PGO)
which can be enabled by passing ``--enable-optimizations --with-lto`` to
the ``configure`` script.

It is also important that NumPy be configured to use an optimised
BLAS/LAPACK distribution.  Further details can be found in the `NumPy
building from source <https://numpy.org/devdocs/user/building.html>`_
guide.
