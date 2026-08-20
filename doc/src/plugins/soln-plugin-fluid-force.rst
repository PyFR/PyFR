*******************************
[soln-plugin-fluidforce-*name*]
*******************************

Periodically integrates the pressure and viscous stress on the boundary
labelled ``name`` and writes out the resulting force and moment (if
requested) vectors to a CSV or HDF5 file.  When reference conditions are
supplied the non-dimensional force coefficients are also written out.
Here ``name`` may also be a brace enumeration of the form
``{le, te, ps, ss}``, in which case the pressure and viscous stress are
integrated over every named boundary as though they formed a single
surface.  Parameterised with

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

#. ``file-reset`` --- if to clear any existing output when starting a
   new simulation; restarts always append:

    *boolean*

#. ``morigin`` --- origin used to compute moments (optional):

    ``(x, y, [z])``

#. ``area-ref`` --- reference area used to non-dimensionalise the force
   into coefficients (optional).  In two dimensions this is a reference
   length (the chord, taken per unit span).  Coefficients are only
   computed when this is given, in which case ``rho-inf`` and ``u-inf``
   must be available in ``[constants]``.  Either may be overridden here
   in the plugin section:

    *float*

#. ``drag-dir`` --- unit (or unnormalised) drag direction; required when
   ``area-ref`` is given:

    ``(x, y, [z])``

#. ``lift-dir`` --- lift direction.  Required in three dimensions;
   in two dimensions it defaults to ``drag-dir`` rotated by
   :math:`+90^{\circ}`.  Must be orthogonal to ``drag-dir``:

    ``(x, y, [z])``

#. ``len-ref`` --- reference length used to non-dimensionalise the
   moment into coefficients (optional).  Moment coefficients are only
   computed when both ``morigin`` and this are given:

    *float*

#. ``quad-deg-{etype}`` --- degree of quadrature rule for fluid force
   integration, optionally this can be specified for different element
   types:

    *int*

#. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

#. ``publish-as`` --- expose computed force components under a name
   for use by triggers (optional):

    *string*

    Published fields: ``px``, ``py`` (pressure forces), ``vx``, ``vy``
    (viscous forces), and the *z* components in 3D.  When coefficients
    are enabled ``cd`` and ``cl`` (plus ``cs`` in 3D) are also published,
    along with the moment coefficients ``cm`` in 2D or ``cmr``, ``cmp``
    and ``cmy`` (roll, pitch and yaw) in 3D.

The force coefficients are formed from the *total* (pressure plus
viscous) force :math:`\mathbf{F}` as :math:`c_d = \mathbf{F}\cdot
\mathbf{\hat{d}}/q_\infty`, :math:`c_l = \mathbf{F}\cdot\mathbf{\hat{l}}
/q_\infty` and, in 3D, :math:`c_s = \mathbf{F}\cdot\mathbf{\hat{s}}
/q_\infty`, where :math:`q_\infty = \tfrac{1}{2}\rho_\infty u_\infty^2
A_\mathrm{ref}` and the side direction :math:`\mathbf{\hat{s}} =
\mathbf{\hat{l}}\times\mathbf{\hat{d}}`.  When ``len-ref`` is also given
the moment coefficients follow as :math:`\mathbf{M}\cdot\mathbf{\hat{n}}
/(q_\infty l_\mathrm{ref})`, where :math:`\mathbf{M}` is the total moment
and the roll, pitch and yaw axes :math:`\mathbf{\hat{n}}` are
:math:`\mathbf{\hat{d}}`, :math:`\mathbf{\hat{s}}` and
:math:`\mathbf{\hat{l}}` respectively.  They are appended to the output
after the force and moment vectors.

Example:

.. code-block:: ini

    [constants]
    rho-inf = 1.0
    u-inf = 1.0

    [soln-plugin-fluidforce-wing]
    nsteps = 10
    file = wing-forces.h5
    file-dataset = /forces
    quad-deg = 6
    morigin = (0.0, 0.0, 0.5)
    area-ref = 0.5
    len-ref = 0.1
    drag-dir = (1.0, 0.0, 0.0)
    lift-dir = (0.0, 0.0, 1.0)
