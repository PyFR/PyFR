*****************
[soln-plugin-fwh]
*****************

Use the Ffowcs Williams--Hawkings equation to approximate the far
field noise generated within a uniformly moving medium.  The
formulation is a convective time-domain one which evaluates all
surface sources at a common emission time; as such it assumes the
integration surface is acoustically compact.  Note that in two
dimensions the three dimensional free-space Green's function is
employed as an approximation.

#. ``dt`` --- time step between samples:

    *float*

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

#. ``file-reset`` --- if to clear any existing output when starting a
   new simulation; restarts always append:

    *boolean*

#. ``surface`` --- a region the surface of which is sampled for the
   FWH solver; either a boundary as ``bc/name`` or a combination of
   the geometric shapes specified in :ref:`user_guide:regions`:

   ``bc/name`` | ``shape(args, ...)``

#. ``quad-deg`` --- degree of surface quadrature rule:

    *int*

#. ``quad-pts-{etype}`` --- name of surface quadrature rule (optional):

    *string*

#. ``observer-pts`` --- the observation points in the far field at
   which noise is approximated; these may not lie on the surface
   itself:

   ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]``

#. ``rho, u, v, (w), p`` --- the constant far field properties of the
   flow, with the far field sound speed following from the ideal gas
   law:

    *float*

Example:

.. code-block:: ini

    [soln-plugin-fwh]
    file = fwh.csv
    file-header = true
    surface = box((1, -5), (10, 5))
    quad-deg = 6
    dt = 1e-2
    observer-pts = [(1, 10), (1, 30), (1, 100), (1, 300)]

    rho = 1
    u = 1
    v = 0
    p = 10
