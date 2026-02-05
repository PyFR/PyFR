*******************************
[soln-plugin-fluidforce-*name*]
*******************************

Periodically integrates the pressure and viscous stress on the boundary
labelled ``name`` and writes out the resulting force and moment (if
requested) vectors to a CSV or HDF5 file.  Parameterised with

#. ``nsteps`` --- integrate every ``nsteps``:

    *int*

#. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

#. ``file-format`` --- output file type (defaults to CSV):

    ``csv`` | ``hdf5``

#. ``file-header`` --- for CSV output if to write a header row or not:

    *boolean*

#. ``file-dataset`` --- for HDF5 output where in the HDF5 to write the
   data:

    *string*

#. ``morigin`` --- origin used to compute moments (optional):

    ``(x, y, [z])``

#. ``quad-deg-{etype}`` --- degree of quadrature rule for fluid force
   integration, optionally this can be specified for different element
   types:

    *int*

#. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

Example::

    [soln-plugin-fluidforce-wing]
    nsteps = 10
    file = wing-forces.h5
    file-dataset = /forces
    quad-deg = 6
    morigin = (0.0, 0.0, 0.5)
