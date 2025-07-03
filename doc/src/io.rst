.. highlight:: none

********
Disk I/O
********

Solution File Writing
^^^^^^^^^^^^^^^^^^^^^

Plugins in PyFR which write ``.pyfrs`` solution files support several
different modes of operation. Specifically, file write operations can
be *serial* or *parallel* and can be performed *synchronously* or
*asynchronously*. These modes have implications for both performance
and robustness. Irrespective of what mode is employed the writes
themselves are *always* performed using Python's I/O routines in lieu
of MPI I/O.

In *serial* mode all data is written out by the root rank.  As such
this method is not suitable for large-scale simulations with practical
I/O rates being limited to a few gigabytes per second. This
limitation can be overcome by the use of *parallel* writing where
each rank writes its own data into the output file. However, for this
deliver any benefits in terms of performance it is necessary for the
file to reside on a parallel file system. Indeed, issuing parallel
I/O operations on a non-parallel file system, such as NFS, frequently
results in data corruption. Thus, PyFR will only enable parallel I/O
when it detects that the output directory resides on a parallel file
system.

Currently, the only such supported file system is Lustre, which PyFR
is capable of both detecting and automatically configuring. When
running PyFR on a Lustre system there is no need to manually specify
the stripe size or the OST count; instead PyFR will automatically
configure these when creating the output file.

The *synchronous* and *asynchronous* writing modes determine if file
I/O is allowed to overlap with computation. This can greatly improve
performance on systems with limited I/O bandwidth since the
simulation is no longer stalled waiting for a file write operation to
complete. Asynchronous writing is enabled by default with an
approximate timeout of 60 seconds for writes to complete. After this
timeout has elapsed PyFR will explicitly wait for the file write to
finish before proceed with the next time step.

The downside of asynchronous writing is that should the simulation
terminate abnormally while the write operation is in progress the
output file will be corrupted. Given this, it is possible to disable
asynchronous writing by setting the timeout to be zero. Further
details can be found in the documentation for the relevant plugins.

VTU export
^^^^^^^^^^

When exporting solution files to ``.vtu`` format, PyFR uses MPI I/O
for parallel operations. However, this approach can cause problems if
the output directory is located on a non-parallel file system (such
as NFS) and MPI ranks are distributed across multiple compute nodes.
Under these conditions, concurrent write operations from different
nodes may corrupt the output file.

To avoid this issue there are two options:

Single-node execution
    Restrict all of the ranks to a single node to eliminate
    cross-node I/O conflicts.

PVTU format
    Export to ``.pvtu`` format instead, which avoids shared file I/O
    and hence does not suffer from cross-node I/O issues.
