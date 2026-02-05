*********************
[soln-plugin-sampler]
*********************

Periodically samples specific points in the volume and writes them out
to a CSV or HDF5 file.  Parameterised with

#. ``nsteps`` --- sample every ``nsteps``:

    *int*

#. ``samp-pts`` --- list of points to sample or a *named* point set:

    ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]`` | *name*

#. ``format`` --- output variable format:

    ``primitive`` | ``conservative``

#. ``sample-gradients`` --- if to sample gradient information or not:

    *boolean*

#. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

#. ``file-format`` --- type of file to output:

    ``csv`` | ``hdf5``

#. ``file-dataset`` --- for HDF5 output the dataset to write into:

    *string*

#. ``file-header`` --- for CSV output to output a header row or not:

    *boolean*

Example::

    [soln-plugin-sampler]
    nsteps = 10
    samp-pts = [(1.0, 0.7, 0.0), (1.0, 0.8, 0.0)]
    format = primitive
    file = point-data.csv
    file-header = true

This plugin also exposes functionality via a CLI. The following
functions are available

-  ``pyfr sampler add`` --- preprocesses and adds a set of points to a
   mesh.  This command can be run under MPI.  Example::

     pyfr sampler add mesh.pyfrm mypoints.csv

-  ``pyfr sampler list`` --- lists the named point sets in a mesh.
   Example::

     pyfr sampler list mesh.pyfrm

-  ``pyfr sampler dump`` --- dumps the locations of all points in a
   named point set.  Example::

     pyfr sampler dump mesh.pyfrm mypoints

-  ``pyfr sampler remove`` --- removes a named point set from a mesh.
   Example::

     pyfr sampler remove mesh.pyfrm mypoints

-  ``pyfr sampler sample`` --- samples a solution file.  This command
   can be run in parallel using ``mpiexec -np n``.  Example::

     pyfr sampler sample --pts=mypoints.csv mesh.pyfrm soln.pyfrs
