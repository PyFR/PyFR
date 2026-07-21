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

#. ``file-reset`` --- if to clear any existing output when starting a
   new simulation; restarts always append:

    *boolean*

#. ``file-header`` --- for CSV output to output a header row or not:

    *boolean*

Example:

.. code-block:: ini

    [soln-plugin-sampler]
    nsteps = 10
    samp-pts = [(1.0, 0.7, 0.0), (1.0, 0.8, 0.0)]
    format = primitive
    file = point-data.csv
    file-header = true

This plugin also exposes functionality via a CLI. The following
functions are available

-  ``pyfr sampler add`` --- preprocesses and adds a set of points to a
   mesh.  This command can be run under MPI.  Example:

   .. code-block:: ini

       pyfr sampler add mesh.pyfrm mypoints.csv

-  ``pyfr sampler list`` --- lists the named point sets in a mesh.
   Example:

   .. code-block:: ini

       pyfr sampler list mesh.pyfrm

-  ``pyfr sampler dump`` --- dumps the locations of all points in a
   named point set.  Example:

   .. code-block:: ini

       pyfr sampler dump mesh.pyfrm mypoints

-  ``pyfr sampler remove`` --- removes a named point set from a mesh.
   Example:

   .. code-block:: ini

       pyfr sampler remove mesh.pyfrm mypoints

-  ``pyfr sampler sample`` --- samples a solution file.  This command
   can be run in parallel using ``mpiexec -np n``.  Example:

   .. code-block:: ini

       pyfr sampler sample --pts=mypoints.csv mesh.pyfrm soln.pyfrs

   Derived fields can be appended to the output by passing one or more
   ``--postproc name`` flags (requires ``-f primitive`` for solution
   files).  On time-averaged files the mean fields stand in for the
   primitives, enabling quantities such as the pressure coefficient of
   the mean flow.  Plugin parameters are read from the solution file's
   embedded config, or from an alternative INI file supplied via
   ``--cfg``.  See :doc:`postproc-plugins` for the list of available
   plugins.  Example:

   .. code-block:: ini

       pyfr sampler sample --pts=mypoints.csv -f primitive \
           --postproc mach --postproc vorticity \
           mesh.pyfrm soln.pyfrs

   Time-averaged files may also be sampled; as with exports, the
   file's default post-processing plugins run automatically, so the
   stored moments are supplemented with all derived statistics from
   the file's statistics specification (Reynolds stresses, rms
   values, budget terms, and so on), evaluated at the sample points.
   Statistics which require gradients of the averaged fields have
   these synthesised from the stored moments; this requires a
   solution of order two or above.  As the sample points are
   arbitrary this can be used to extract wake traverses, boundary
   layer profiles, and spanwise distributions from fully
   three-dimensional configurations.  Example:

   .. code-block:: ini

       pyfr sampler sample --pts=traverse.csv mesh.pyfrm tavg.pyfrs

The following commands post-process HDF5 time series written by the
plugin.  All take ``-d`` to select the dataset (required for HDF5
input), ``--tstart``/``--tend`` windowing, and write tab-separated
data to standard output.

The estimators operate directly on the recorded, potentially
non-uniform, sample times: no resampling onto a uniform grid is
performed, so adaptive time stepping introduces no interpolation
bias, statistics are weighted by the time each sample represents,
and analysis windows containing data gaps are discarded.

Wherever a field name is accepted, tabulated derived quantities
(``cp``, ``mach``, ``vorticity``, ...) may also be given; they are
evaluated over the sampled primitives on the fly.  Inline
expressions over the recorded and tabulated fields are likewise
accepted, as in ``-f "0.6*u + 0.8*v"``; this permits, for example,
velocity components to be projected onto arbitrary directions.
In signal pair arguments a trailing ``@POINT`` selects the sample
point and applies to the expression as a whole; hence
``-a "0.6*u + 0.8*v@5"`` is the projected velocity at point five.
Points may not be mixed within a single expression.

In addition to HDF5 history files the commands accept CSV
histories: both the point form written by the sampler and FWH
plugins and plain ``t`` plus named-column histories such as those
from the ``fluidforce`` and ``integrate`` plugins, enabling, for
example, a PSD of the lift coefficient.  As CSV files do not embed
a configuration, ``-c`` may be used to supply one when tabulated
derived quantities are requested.

-  ``pyfr sampler stats`` --- time-weighted mean and rms along with
   the minimum and maximum per point and field, with ``-f`` and ``-p``
   selecting fields and point indices.  With ``--ci`` the mean gains
   an autocorrelation-aware confidence interval following `Bergmann et
   al. <https://doi.org/10.1115/1.4052402>`__: the standard error uses
   an effective sample count from the integrated autocorrelation
   timescale, with the level set by ``--level``.  Passing a relative
   error target via ``--target`` adds the averaging time required to
   attain it, which is useful when planning how much longer to run a
   simulation.  A warning is emitted when a dominant tone (vortex
   shedding, say) is present, as deterministic periodic content
   inflates the estimated interval; second moments converge much more
   slowly than means and their intervals are correspondingly wider.
   Example:

   .. code-block:: ini

       pyfr sampler stats probes.h5 -d probes -f u,cp --tstart 50
       pyfr sampler stats forces.csv -f px --ci --target 0.01 --tstart 50

-  ``pyfr sampler transient`` --- detects the end of the initial
   transient of each selected signal with the marginal standard error
   rule.  The maximum truncation time over all signals is written to
   standard output, with the per-point and per-field breakdown on
   standard error, so the result composes directly into the
   ``--tstart`` of the other commands and excludes startup
   contamination from their statistics.  Note that the rule cannot
   anticipate dynamics which have yet to begin, such as the onset of
   shedding after a quiet spell, and that drifts on timescales
   comparable to the series length escape it.  Example:

   .. code-block:: ini

       pyfr sampler stats forces.csv -f px \
           --tstart=`pyfr sampler transient forces.csv -f px,py`

-  ``pyfr sampler tavg`` --- evaluates statistics packages (see
   :doc:`soln-plugin-tavg`) over the time axis, reporting all
   derivative-free quantities per point; ``--stats`` may be repeated
   to evaluate several packages.  Example:

   .. code-block:: ini

       pyfr sampler tavg probes.h5 -d probes --stats tke --stats moments-2 --tstart 50

-  ``pyfr sampler psd`` --- power spectral density via Welch averaging
   of Hann-tapered quadrature nonuniform DFTs; ``-w`` sets the
   analysis window duration in time units and ``-o`` the overlap
   fraction.  Each window only reports frequencies below its local
   Nyquist rate.  Example:

   .. code-block:: ini

       pyfr sampler psd probes.h5 -d probes -f v -p 2 --tstart 50 -w 5.0

-  ``pyfr sampler tone`` --- frequency and amplitude of the dominant
   tone within a band via a floating-mean least-squares fit; as the
   sampling is non-uniform, tones above the mean-rate Nyquist
   frequency remain identifiable.  Example:

   .. code-block:: ini

       pyfr sampler tone probes.h5 -d probes -f v -p 2 --fmin 1 --fmax 4

-  ``pyfr sampler autocorr`` --- autocorrelation function and integral
   time scale via fuzzy slotting with local normalisation;
   ``--maxlag`` sets the lag range and ``--nslots`` the resolution.
   Example:

   .. code-block:: ini

       pyfr sampler autocorr probes.h5 -d probes -f u -p 0 --maxlag 10

-  ``pyfr sampler structfun`` --- temporal structure functions
   :math:`S_n(\tau) = \langle |x(t + \tau) - x(t)|^n \rangle` of the
   requested orders (``-n``, default ``2,3``), estimated directly
   from sample pairs binned by lag; ``--signed`` uses raw rather than
   absolute increments, as appropriate for the third-order law.
   Under Taylor's hypothesis time lags map to streamwise separations.
   Example:

   .. code-block:: ini

       pyfr sampler structfun probes.h5 -d probes -f u -p 0 -n 2,3,4 --tstart 50

-  ``pyfr sampler xcorr`` --- slotted cross-correlation between two
   signals given as ``FIELD[@POINT]``, reporting the peak correlation
   and its lag; useful for convection velocities.  Example:

   .. code-block:: ini

       pyfr sampler xcorr probes.h5 -d probes -a v@0 -b v@2 --maxlag 10

-  ``pyfr sampler coherence`` --- magnitude-squared coherence and
   cross-spectrum phase between two signals; the coherence is
   corrected for the finite window-count bias of the Welch estimate.
   Example:

   .. code-block:: ini

       pyfr sampler coherence probes.h5 -d probes -a v@0 -b v@2 -w 10.0

-  ``pyfr sampler two-point`` --- two-point correlation of a field
   against a reference point (``-r``, default 0) across a set of
   points, reported as a function of separation distance.  Each row
   gives the zero-lag correlation coefficient, the peak coefficient
   over all lags, and the lag of the peak; the correlation length,
   the integral of the zero-lag coefficient over separation up to its
   first zero crossing, is reported as a comment.  With
   ``--coherence`` the bias-corrected magnitude-squared coherence is
   instead reported per frequency and separation, together with the
   e-folding length of an exponential decay fit over separation.
   Examples:

   .. code-block:: ini

       pyfr sampler two-point probes.h5 -d probes -f u -r 0 --maxlag 10
       pyfr sampler two-point probes.h5 -d probes -f p --coherence -w 10.0
