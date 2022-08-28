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

PyFR does not typically derive any benefit from SMT.  As such the
number of OpenMP threads should be chosen to be equal to the number of
physical cores.

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
zone.  Thus, on a two socket system it is suggested to run PyFR with
two MPI ranks, with each process being bound to a single socket.  The
specifics of how to accomplish this depend on both the job scheduler
and MPI distribution.

.. _perf cuda backend:

CUDA Backend
============

CUDA-aware MPI
--------------

PyFR is capable of taking advantage of CUDA-aware MPI.  This enables
CUDA device pointers to be directly to passed MPI routines.  Under the
right circumstances this can result in improved performance for
simulations which are near the strong scaling limit.  Assuming
mpi4py has been built against an MPI distribution which is CUDA-aware
this functionality can be enabled through the ``mpi-type`` key as::

        [backend-cuda]
        mpi-type = cuda-aware

.. _perf hip backend:

HIP Backend
===========

HIP-aware MPI
-------------

PyFR is capable of taking advantage of HIP-aware MPI.  This enables
HIP device pointers to be directly to passed MPI routines.  Under the
right circumstances this can result in improved performance for
simulations which are near the strong scaling limit.  Assuming
mpi4py has been built against an MPI distribution which is HIP-aware
this functionality can be enabled through the ``mpi-type`` key as::

        [backend-hip]
        mpi-type = hip-aware

Partitioning
============

METIS vs SCOTCH
---------------

The partitioning module in PyFR includes support for both METIS and
SCOTCH.  Both usually result in high-quality decompositions.  However,
for long running simulations on complex geometries it may be worth
partitioning a grid with both and observing which decomposition
performs best.

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

Thus in order to obtain a good decomposition it is necessary to assign
a weight to each type of element in the domain.  Element types which
are more computationally intensive should be assigned a larger weight
than those that are less intensive.  Unfortunately, the relative cost
of different element types depends on a variety of factors, including:

 - The polynomial order.
 - If anti-aliasing is enabled in the simulation, and if so, to what
   extent.
 - The hardware which the simulation will be run on.

Weights can be specified when partitioning the mesh as
``-e shape:weight``.  For example, if on a particular system a
quadrilateral is found to be 50% more expensive than a triangle this
can be specified as::

        pyfr partition -e quad:3 -e tri:2 ...

If precise profiling data is not available regarding the performance of
each element type in a given configuration a helpful rule of thumb is
to under-weight the dominant element type in the domain.  For example,
if a domain is 90% tetrahedra and 10% prisms then, absent any
additional information about the relative performance of tetrahedra and
prisms, a safe choice is to assume the prisms are appreciably *more*
expensive than the tetrahedra.

Scaling
=======

The general recommendation when running PyFR in parallel is to aim for
a parallel efficiency of :math:`\epsilon \simeq 0.8` with the parallel
efficiency being defined as:

.. math::

  \epsilon = \frac{1}{N}\frac{T_1}{T_N},

where :math:`N` is the number of ranks, :math:`T_1` is the simulation
time with one rank, and :math:`T_N` is the simulation time with
:math:`N` ranks.  This represents a reasonable trade-off between the
overall time-to-solution and efficient resource utilisation.

Parallel I/O
============

PyFR incorporates support for parallel file I/O via HDF5 and will use it
automatically where available.  However, for this work several
prerequisites must be satisfied:

 - HDF5 must be explicitly compiled with support for parallel I/O.
 - The mpi4py Python module *must* be compiled against the same MPI
   distribution as HDF5.  A version mismatch here can result in subtle
   and difficult to diagnose errors.
 - The h5py Python module *must* be built with support for parallel
   I/O.

After completing this process it is highly recommended to verify
everything is working by trying the
`h5py parallel hdf5 example <https://docs.h5py.org/en/stable/mpi.html#using-parallel-hdf5-from-h5py>`_.

Start-up Time
=============

The start-up time required by PyFR can be reduced by ensuring that
Python is compiled from source with profile guided optimisations (PGO)
which can be enabled by passing ``--enable-optimizations`` to the
``configure`` script.

It is also important that NumPy be configured to use an optimized
BLAS/LAPACK distribution.  Further details can be found in the
`NumPy building from source <https://numpy.org/devdocs/user/building.html>`_
guide.

If the point sampler plugin is being employed with a large number of
sample points it is further recommended to install SciPy.
