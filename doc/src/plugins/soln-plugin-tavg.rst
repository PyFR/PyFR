******************
[soln-plugin-tavg]
******************

Time average quantities. Parameterised with

#. ``nsteps`` --- accumulate the average every ``nsteps`` time steps:

    *int*

#. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

#. ``mode`` --- output file accumulation mode:

    ``continuous`` | ``windowed``

    In continuous mode each output file contains average data from
    ``tstart`` up until the time at which the file is written. In windowed
    mode each output file only contains average data for the preceding
    ``dt-out`` time units. The default is ``windowed``. Average data files
    obtained using the windowed mode can be accumulated after-the-fact using
    the CLI.

#. ``basedir`` --- relative path to directory where outputs will be
   written:

    *string*

#. ``basename`` --- pattern of output names:

    *string*

#. ``precision`` --- output file number precision:

    ``single`` | ``double``

    The default is ``single``. Note that this only impacts the output,
    with statistic accumulation *always* being performed in double
    precision.

#. ``region`` --- region to be averaged, specified as either the entire
   domain using ``*``, a combination of the geometric shapes specified
   in :ref:`user_guide:regions`, or a sub-region of elements that have faces on a
   specific domain boundary via the name of the domain boundary:

    ``*`` | ``shape(args, ...)`` | *string*

#. ``region-type`` --- if to average all of the elements contained inside
   the region or only those which are on its surface:

    ``volume`` | ``surface``

#. ``region-expand`` --- how many layers to grow the region by:

    *int*

#. ``avg``-*name* --- expression to time average, written as a function
   of the primitive variables and gradients thereof; multiple
   expressions, each with their own *name*, may be specified:

    *string*

#. ``fun-avg``-*name* --- derived expression written as a function of
   any ordinary average terms; multiple expressions, each with their
   own *name*, may be specified:

    *string*

    Derived expressions are not stored in the output files; they are
    evaluated on demand at export time from the averaged fields,
    which they are exact functions of.

#. ``stats`` --- statistics packages to accumulate; a comma separated
   list of set names, optionally with arguments, and/or individual
   quantity names:

    *string*

    The plugin automatically selects the averaged fields needed by
    the requested quantities and deduplicates them against any
    ``avg-*`` expressions.  Derived quantities are finalised during
    ``pyfr export``, which enables the required processing (including
    spatial differentiation of the averages for budget terms and
    similar) automatically.

    Related sets may be selected together using brace enumeration
    and wildcard patterns; as such ``budget-{uu, k}`` is equivalent
    to ``budget-uu, budget-k`` while ``budget-*`` selects every set
    with the prefix ``budget-``.  Sets which extend a base set
    include it by definition; for example ``scales-exact`` contains
    all of ``scales``.

Statistics sets
===============

For the Euler and Navier--Stokes systems the following sets are
available:

- ``mean`` --- mean primitive fields.
- ``moments-n`` --- central moments up to order *n* between 2 and 4:
  Reynolds stresses, rms values, and, at higher orders, skewness and
  flatness.  Sets are cumulative, so ``moments-3`` includes all of
  ``moments-2``.  By default all velocity components and pressure are
  included; per-field sets restrict this, with cross moments selected
  explicitly, as in ``moments-2-{u, v, uv}``.
- ``tke`` --- turbulent kinetic energy; ``tke-anisotropy``
  additionally exports the anisotropy tensor and its invariants.
- ``acoustic`` --- time-averaged acoustic intensity components
  :math:`I_i = \langle p'u_i'\rangle` and magnitude, along with the
  acoustic energy density
  :math:`\langle p'^2\rangle/2\rho c^2 + \rho\langle|u'|^2\rangle/2`
  formed from the mean density and sound speed.  These are the
  classic quiescent-medium expressions; in the presence of a strong
  mean flow they omit the mean-flow transport contributions.
- ``heat-flux`` --- temperature statistics and turbulent heat fluxes
  normalised by the wall state; requires the ``cpTw`` constant.
- ``favre`` --- Favre (density-weighted) mean velocities and turbulent
  mass fluxes; ``favre-2`` adds the Favre stresses and turbulent
  kinetic energy and ``favre-heat-flux`` the Favre temperature
  statistics and heat fluxes.

The Navier--Stokes system additionally provides:

- ``budget-<component>`` --- Reynolds stress transport budgets.
  Components are stress pairs (``uu``, ``uv``, ...) or ``k`` for the
  turbulent kinetic energy budget, with ``budget-*`` selecting every
  budget.  Each budget comprises production, convection, turbulent
  transport, pressure transport, pressure strain (pressure dilatation
  for ``k``), viscous diffusion, dissipation, and a closure residual.
- ``scales`` --- dissipation rate and the derived Kolmogorov, Taylor,
  and integral scales, along with grid-resolution ratios;
  ``scales-exact`` upgrades the dissipation estimate with the
  transposed-gradient contraction.
- ``compressibility`` --- solenoidal and dilatational dissipation,
  pressure dilatation, turbulent Mach number, density rms, and the
  isentropic pressure-density correlation.
- ``vgt`` --- velocity-gradient tensor invariants: means and rms
  values of the instantaneous invariants together with the invariants
  of the mean velocity gradient.
- ``wall`` --- mean wall quantities (mean traction vector,
  ``tau-wall``, ``cf``, ``u-tau``, ``yplus``) from runtime-accumulated
  mu-weighted velocity gradients; finalised on boundary exports.
  ``wall-heat`` extends this with the mean wall heat flux ``q-wall``,
  formed from the mu-weighted gradients of :math:`c_p T` so no
  temperature constant is needed.

The nominal cost of each set is tabulated below in terms of the
number of accumulated fields, which determines the runtime cost and
the size of the average files, and the number of additional derived
quantities which are finalised on export and appear in the output
alongside the accumulated fields.  Counts are for each set on its
own; as sets share underlying moments the cost of a combination is
typically less than the sum of its parts.

================  =========  ==========  =========  ==========
Set               2D fields  2D derived  3D fields  3D derived
================  =========  ==========  =========  ==========
mean                      4           0          5           0
moments-2                 7           6         11          10
moments-3                10           9         15          14
moments-4                13          12         19          18
tke                       4           1          6           1
tke-anisotropy            5           9          9          11
acoustic                  9           4         12           5
heat-flux                 6           3          8           4
favre                     5           4          7           6
favre-2                   8           8         13          13
favre-heat-flux           9           8         12          11
budget-uu                18           8         24           8
budget-k                 28           8         51           8
budget-*                 34          32         72          56
scales                   12          11         20          11
scales-exact             13          12         21          12
vgt                       6           2         13           4
wall                      6           7         11           8
wall-heat                 8           8         14           9
compressibility          18           8         25           8
================  =========  ==========  =========  ==========

Individual quantity names may also be given directly, with all
prerequisite moments selected automatically.  Example:

.. code-block:: ini

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 25.0
    mode = windowed
    basedir = .
    basename = tavg-{t:.2f}

    stats = budget-k, scales-exact, moments-2, skew-p

.. note::

   Statistic accumulation is performed on the backend; in double
   precision where supported and otherwise in single precision with
   compensated summation.  The instantaneous solution and products
   thereof are, however, evaluated at the working precision of the
   backend.  Budget closure therefore degrades with single precision
   backends, and ``precision = double`` is recommended when budgets
   are requested.  Spatial derivatives of averaged quantities taken
   at export time use element-local differentiation of the lifted
   average.

Example:

.. code-block:: ini

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 2.0
    mode = windowed
    basedir = .
    basename = files-{t:06.2f}

    avg-u = u
    avg-v = v
    avg-uu = u*u
    avg-vv = v*v
    avg-uv = u*v

    fun-avg-upup = uu - u*u
    fun-avg-vpvp = vv - v*v
    fun-avg-upvp = uv - u*v

This plugin also exposes functionality via a CLI. The following
functions are available

- ``pyfr tavg merge`` --- average together multiple time average files
  into a single time average file. The averaging times are read from the
  file and do not need to be evenly spaced in time.  Passing
  ``-r``/``--report`` additionally treats each input window as an
  independent batch and reports the relative standard error of every
  merged field along with every derived quantity which does not
  involve spatial derivatives; this indicates how statistically
  converged the averages, and any turbulence statistics computed from
  them, are.  Example:

  .. code-block:: ini

      pyfr tavg merge avg-1.00.pyfrs avg-2.00.pyfrs avg-10.00.pyfrs merged_avg.pyfrs
