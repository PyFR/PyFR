*********************
[soln-plugin-sampler]
*********************

Periodically samples specific points in the volume and writes them out
to a CSV file. Parameterised with

1. ``nsteps`` --- sample every ``nsteps``:

    *int*

2. ``samp-pts`` --- list of points to sample or a *named* point set:

    ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]`` | *name*

3. ``format`` --- output variable format:

    ``primitive`` | ``conservative``

4. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

5. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-sampler]
    nsteps = 10
    samp-pts = [(1.0, 0.7, 0.0), (1.0, 0.8, 0.0)]
    format = primitive
    file = point-data.csv
    header = true

This plugin also exposes functionality via a CLI. The following
functions are available

#. ``pyfr sampler add`` --- preprocesses and adds a set of points to a
   mesh.  This command can be run under MPI.

   Example::

     pyfr sampler add mesh.pyfrm mypoints.csv

#. ``pyfr sampler list`` --- lists the named point sets in a mesh.

   Example::

     pyfr sampler list mesh.pyfrm

#. ``pyfr sampler dump`` --- dumps the locations of all points in a
   named point set.

   Example::

     pyfr sampler dump mesh.pyfrm mypoints

#. ``pyfr sampler remove`` --- removes a named point set from a mesh.

   Example::

     pyfr sampler remove mesh.pyfrm mypoints
