********
Triggers
********

Triggers extend the plugin system with conditional execution.  Instead
of running on a fixed schedule, plugins can be gated by trigger
conditions---for example, starting time averaging only after the flow
has settled, or writing a snapshot when a NaN is detected.

Triggers are defined in ``[trigger-name]`` sections and referenced by
name from plugin sections.  Trigger state is included in solution
checkpoint files, so restarts preserve trigger history.

Concepts
========

Trigger Modes
-------------

Each trigger has a *mode* that controls how its raw condition is
translated into an active/inactive state:

``latch``
  Once the condition becomes true the trigger stays active permanently.
  This is the default.

``level``
  The trigger is active only while the condition is true.

``edge``
  The trigger fires for a single evaluation cycle on the rising edge
  (false |rarr| true transition).

Plugin Options
--------------

Every plugin supports the following optional trigger-related keys:

``enabled``
  *boolean* --- if ``false`` the plugin is never called.

``trigger``
  *string* --- name(s) of trigger(s) that control this plugin.
  Use ``&`` to require **all** triggers (AND) and ``|`` to require
  **any** trigger (OR).  Examples: ``trigger = settled & lift``,
  ``trigger = timeout | converged``.

``trigger-action``
  ``activate`` | ``gate``

  With ``activate`` (the default) the plugin remains dormant until the
  trigger condition is met, after which it runs on its normal schedule
  permanently.  With ``gate`` the plugin runs only while the trigger
  condition is met.

``trigger-write``
  *string* --- name of a trigger; when active the plugin's
  ``trigger_write`` method is called (if it has one) to perform a
  non-destructive snapshot.  This does not affect the plugin's normal
  output schedule.

``trigger-set``
  *string* --- name of a trigger to fire programmatically when the
  plugin detects a condition (e.g. NaN detection).

``publish-as``
  *string* --- expose the plugin's scalar outputs under the given name
  so that expression and steady-state triggers can reference them.

Trigger Types
=============

[trigger-*name*] type = manual
-------------------------------

A trigger that is never true on its own.  It can only be activated
programmatically by a plugin via ``trigger-set``.  Parameterised with:

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-nan-detected]
    type = manual
    mode = edge

[trigger-*name*] type = time
----------------------------

Becomes true once the simulation time reaches a threshold.
Parameterised with:

#. ``t`` --- simulation time at which to fire:

    *float*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-flow-developed]
    type = time
    t = 5.0

[trigger-*name*] type = wallclock
---------------------------------

Becomes true once wall-clock time since the start of the run reaches a
threshold.  Useful for graceful shutdown before a queue limit.
Parameterised with:

#. ``t`` --- wall-clock seconds since the run started:

    *float*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-wall-limit]
    type = wallclock
    t = 82800
    mode = edge

[trigger-*name*] type = signal
------------------------------

Becomes true when a Unix signal is received.  The signal is blocked on
startup and consumed non-destructively via ``sigpending``/``sigwait``.
Only the root MPI rank listens for the signal; the result is broadcast
to all ranks.  Parameterised with:

#. ``signal`` --- signal name (e.g. ``USR1``, ``SIGUSR1``):

    *string*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-dump]
    type = signal
    signal = USR1
    mode = edge

[trigger-*name*] type = file
-----------------------------

Becomes true when a file exists on disk.  Only evaluated on the root
rank.  Parameterised with:

#. ``path`` --- path to the sentinel file:

    *string*

#. ``watch`` --- what to monitor:

    ``exists`` | ``mtime`` | ``ctime`` | ``atime``

    With ``exists`` (the default) the trigger fires when the file is
    present.  With ``mtime``, ``ctime``, or ``atime`` the trigger
    fires whenever the corresponding timestamp changes, producing a
    single-cycle pulse per change.  This allows retriggering by simply
    touching the file rather than deleting and recreating it.  The
    last-seen timestamp is included in checkpoint files.

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-stop]
    type = file
    path = /scratch/job/please_stop
    mode = latch

[trigger-*name*] type = expression
-----------------------------------

Evaluates a mathematical expression against integrator scalars (``t``,
``dt``, ``step``) and published plugin values.  Parameterised with:

#. ``condition`` --- condition of the form ``expr cmp threshold``
   where *cmp* is one of ``<``, ``<=``, ``>``, ``>=``:

    *string*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Published values are referenced as ``publisher.field``, where
*publisher* is the ``publish-as`` name of a plugin and *field* is one
of its output fields.

Example:

.. code-block:: ini

    [trigger-high-drag]
    type = expression
    condition = forces.px > 100.0

[trigger-*name*] type = field
-----------------------------

Evaluates a field expression over the domain (or a region), reduces it,
and compares against a threshold.  This is a collective operation
across all MPI ranks.  Parameterised with:

#. ``condition`` --- condition of the form
   ``reduction(expr) cmp threshold`` where *reduction* is one of
   ``min``, ``max``, ``sum``, ``avg``, ``l2norm`` and *cmp* is one of
   ``<``, ``<=``, ``>``, ``>=``:

    *string*

#. ``nsteps`` --- evaluate every *nsteps* steps:

    *int*

#. ``region`` --- region to evaluate over (default ``*``):

    ``*`` | ``shape(args, ...)``

#. ``quad-deg`` --- quadrature degree (optional):

    *int*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-pressure-spike]
    type = field
    condition = max(p) > 1e6
    region = box((-1, -1), (1, 1))
    nsteps = 5

[trigger-*name*] type = point
------------------------------

Samples the solution at one or more discrete probe points, evaluates
an expression in primitive variables, reduces over the points, and
compares against a threshold.  This reuses the point sampling
infrastructure from the sampler plugin.  Parameterised with:

#. ``condition`` --- condition of the form
   ``reduction(expr) cmp threshold`` where *reduction* is one of
   ``min``, ``max``, ``avg`` and *cmp* is one of
   ``<``, ``<=``, ``>``, ``>=``:

    *string*

#. ``pts`` --- probe point coordinates:

    ``[(x, y), ...]`` | ``[(x, y, z), ...]``

#. ``nsteps`` --- evaluate every *nsteps* steps:

    *int*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-probe-temp]
    type = point
    condition = max(p / rho) > 350.0
    pts = [(0.5, 0.0), (1.0, 0.0)]
    nsteps = 20

[trigger-*name*] type = steady
-------------------------------

Monitors a published scalar value and fires when it has become steady
according to a criterion.  Parameterised with:

#. ``source`` --- published value to monitor, as
   ``publisher.field``:

    *string*

#. ``criterion`` --- steadiness metric:

    ``range`` | ``gradient`` | ``std`` | ``mser``

    ``range`` fires when (max |minus| min) / |mean| < tolerance.
    ``gradient`` fires when the normalised slope over the window is
    below tolerance.  ``std`` fires when std / |mean| < tolerance.
    ``mser`` detects the end of the initial transient with the
    marginal standard error rule of `Bergmann et al.
    <https://doi.org/10.1115/1.4052402>`__ applied to the full
    published history, firing once the detected transient end is
    at least ``margin`` old and the estimate has stopped moving.

#. ``window`` --- number of published samples in the sliding window
   (``range``, ``gradient`` and ``std`` criteria):

    *int*

#. ``tolerance`` --- convergence threshold, relative to the mean
   (``range``, ``gradient`` and ``std`` criteria):

    *float*

#. ``margin`` --- stability horizon for the ``mser`` criterion; the
   transient-end estimate must be this old and must have varied by
   less than a quarter of this over it.  Choose a value spanning
   several periods of the slowest expected dynamics, since a quiet
   spell before their onset is indistinguishable from stationarity:

    *float*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-transient-over]
    type = steady
    source = forces.px
    criterion = mser
    margin = 10.0

[trigger-*name*] type = meanci
------------------------------

Fires when the time-averaged mean of a published value is known to a
target accuracy.  The standard error accounts for autocorrelation in
the series via the integrated timescale, following `Bergmann et al.
<https://doi.org/10.1115/1.4052402>`__; the resulting interval matches
the offline ``pyfr sampler stats --ci`` command.  Parameterised with:

#. ``source`` --- published value to monitor, as
   ``publisher.field``:

    *string*

#. ``tolerance`` --- target half-width of the confidence interval:

    *float*

#. ``relative`` --- if the tolerance is relative to the mean:

    ``true`` | ``false``

#. ``level`` --- confidence level; defaults to 0.68:

    *float*

#. ``min-neff`` --- minimum effective sample count before the
   interval is trusted; defaults to 32:

    *float*

#. ``after`` --- optional trigger to wait for; only samples published
   after it becomes active enter the estimate.  Chain this behind an
   ``mser`` steady trigger so startup transients do not bias the mean:

    *string*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

    Note that deterministic periodic content, such as vortex
    shedding, inflates the estimated interval, and that a verdict may
    revert as new samples arrive; use ``edge`` to fire once on each
    upward transition.

Example:

.. code-block:: ini

    [trigger-cd-converged]
    type = meanci
    source = forces.px
    after = transient-over
    tolerance = 0.005
    relative = true

[trigger-*name*] type = duration
--------------------------------

Becomes true a fixed time or number of steps after another trigger
fires.  Parameterised with:

#. ``after`` --- name of the trigger to wait for:

    *string*

#. ``duration`` --- time to wait after the *after* trigger fires:

    *float*

#. ``steps`` --- steps to wait (alternative to ``duration``):

    *int*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-warm-up-done]
    type = duration
    after = flow-developed
    duration = 10.0

[trigger-*name*] type = all
---------------------------

Logical AND of multiple triggers.  Parameterised with:

#. ``triggers`` --- space-separated list of trigger names:

    *string*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-ready]
    type = all
    triggers = flow-developed cd-converged

[trigger-*name*] type = any
---------------------------

Logical OR of multiple triggers.  Parameterised with:

#. ``triggers`` --- space-separated list of trigger names:

    *string*

#. ``mode`` --- trigger mode:

    ``latch`` | ``level`` | ``edge``

Example:

.. code-block:: ini

    [trigger-should-stop]
    type = any
    triggers = wall-limit dump-requested

Examples
========

Write a Snapshot Before NaN Abort
---------------------------------

When NaN values are detected, fire a trigger that causes a second
writer instance to dump the solution before the solver aborts.  This
gives a file that can be inspected to diagnose the instability.

.. code-block:: ini

    [soln-plugin-nancheck]
    nsteps = 10
    trigger-set = nan-detected

    [trigger-nan-detected]
    type = manual
    mode = edge

    [soln-plugin-writer]
    dt-out = 1.0
    basedir = .
    basename = soln-{t:.4f}

    [soln-plugin-writer-crashdump]
    basedir = .
    basename = crash-{t:.4f}
    trigger = nan-detected
    trigger-action = gate

Here the nancheck plugin fires ``nan-detected`` on the step a NaN
appears.  The ``writer-crashdump`` instance has no ``dt-out`` and is
gated by the trigger, so it writes exactly once---immediately before
the abort.

Dump Solution and Time Averages on a Signal
--------------------------------------------

Send ``SIGUSR1`` to the running process to trigger an immediate dump
of both the instantaneous solution and the current time-average
accumulation, without interrupting the normal output schedule.

.. code-block:: ini

    [trigger-dump]
    type = signal
    signal = USR1
    mode = edge

    [soln-plugin-writer]
    dt-out = 5.0
    basedir = .
    basename = soln-{t:.4f}
    trigger-write = dump

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 50.0
    basedir = .
    basename = tavg-{t:.4f}
    avg-u = u
    avg-v = v
    trigger-write = dump

With ``trigger-write`` the plugins continue their normal periodic
output; when the signal arrives, each plugin additionally performs a
non-destructive snapshot.

On-Demand Snapshot via File Touch
----------------------------------

Use a file trigger with ``watch = mtime`` to dump a snapshot each time
a sentinel file is touched.  Because the trigger fires only when the
timestamp changes, each ``touch`` produces exactly one write:

.. code-block:: ini

    [trigger-snap-now]
    type = file
    path = /tmp/pyfr-snap
    watch = mtime

    [soln-plugin-writer]
    dt-out = 5.0
    basedir = .
    basename = soln-{t:.4f}
    trigger-write = snap-now

Run ``touch /tmp/pyfr-snap`` at any time to get an immediate snapshot;
touch it again later for another.  The normal ``dt-out`` schedule is
unaffected.  Alternatively, use ``watch = exists`` (the default) with
``mode = edge`` for a delete-and-recreate workflow.

Start Time Averaging When the Flow Settles
-------------------------------------------

Use the fluid force plugin to publish the drag coefficient, monitor it
with a steady-state trigger, and only begin accumulating time averages
once the drag has converged.

.. code-block:: ini

    [soln-plugin-fluidforce-cylinder]
    nsteps = 10
    file = forces.csv
    file-header = true
    publish-as = forces

    [trigger-cd-converged]
    type = steady
    source = forces.px
    window = 200
    tolerance = 0.01
    criterion = range

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 20.0
    basedir = .
    basename = tavg-{t:.4f}
    avg-u = u
    avg-v = v
    avg-uu = u*u
    avg-vv = v*v
    avg-uv = u*v
    fun-avg-upup = uu - u*u
    fun-avg-vpvp = vv - v*v
    fun-avg-upvp = uv - u*v
    trigger = cd-converged
    trigger-action = activate

The time averaging plugin stays dormant until the drag converges, then
runs permanently on its normal schedule.

Graceful Shutdown Before Queue Limit
-------------------------------------

Write a final checkpoint and stop the simulation cleanly 30 minutes
before an HPC queue's wall-clock limit (assuming a 24-hour allocation).

.. code-block:: ini

    [trigger-wall-limit]
    type = wallclock
    t = 82800
    mode = edge

    [soln-plugin-writer-final]
    basedir = .
    basename = checkpoint-{t:.4f}
    trigger = wall-limit
    trigger-action = gate

Conditional Field Monitoring
----------------------------

Fire a trigger when the maximum pressure anywhere in a region exceeds
a threshold---useful for detecting shocks or instabilities.

.. code-block:: ini

    [trigger-pressure-spike]
    type = field
    condition = max(p) > 1.5e5
    region = box((-2, -2), (2, 2))
    nsteps = 20

    [soln-plugin-writer-spike]
    basedir = .
    basename = spike-{t:.4f}
    trigger = pressure-spike
    trigger-action = gate

Composite: Start Averaging After Vortex Shedding Onset
-------------------------------------------------------

For a cylinder flow, wait for the initial transient to pass and for
vortex shedding to develop before starting time averaging.  A time
trigger prevents early activation from impulsive-start force spikes,
while an expression trigger detects the onset of lift oscillation.

.. code-block:: ini

    [soln-plugin-fluidforce-wall]
    nsteps = 10
    quad-deg = 9
    file = forces.csv
    publish-as = forces

    [trigger-settled]
    type = time
    t = 10.0

    [trigger-lift]
    type = expression
    condition = abs(forces.py) > 0.01
    mode = level

    [trigger-shedding]
    type = all
    triggers = settled lift
    mode = latch

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 20.0
    basedir = .
    basename = tavg-{t:.4f}
    avg-u = u
    avg-v = v
    avg-uu = u*u
    avg-vv = v*v
    avg-uv = u*v
    fun-avg-upup = uu - u*u
    fun-avg-vpvp = vv - v*v
    fun-avg-upvp = uv - u*v
    trigger = shedding
    trigger-action = activate

Statistical Averaging and Convergence
-------------------------------------

The heuristics above can be replaced with the statistical methods of
`Bergmann et al. <https://doi.org/10.1115/1.4052402>`__: the marginal
standard error rule detects the end of the initial transient, after
which time averaging begins, and the run emits a final
gradient-bearing snapshot once the mean drag is known to within 0.5%
at 68% confidence.  Published force histories are
serialised into snapshots, so both estimates resume exactly across
restarts.

.. code-block:: ini

    [soln-plugin-fluidforce-wall]
    nsteps = 50
    quad-deg = 6
    file = forces.csv
    publish-as = forces

    [trigger-transient-over]
    type = steady
    criterion = mser
    source = forces.px
    margin = 10.0

    [trigger-drag-converged]
    type = meanci
    source = forces.px
    after = transient-over
    tolerance = 0.005
    relative = true
    mode = edge

    [soln-plugin-tavg]
    nsteps = 100
    dt-out = 10.0
    basedir = .
    basename = tavg-{t:.2f}
    mode = continuous
    stats = mean, moments-2, wall
    trigger = transient-over
    trigger-action = activate

    [soln-plugin-writer-final]
    basedir = .
    basename = final-{t:.2f}
    write-gradients = true
    trigger = drag-converged
    trigger-action = gate

.. |rarr| unicode:: U+2192
.. |minus| unicode:: U+2212
.. |mean| unicode:: U+007C mean U+007C
