*********************
[soln-plugin-dtstats]
*********************

Write time-step statistics out to a CSV file. For implicit integrators,
an optional stage file can be specified to record per-stage convergence
information (Newton iterations, GMRES iterations, residuals).

Parameterised with

#. ``file`` --- output file path for step statistics; should the file
   already exist it will be appended to:

    *string*

   Each row records the step number ``n``, the time ``t`` at the start
   of the step, the step size ``dt``, the ``action`` taken, the wall
   time ``wtime`` of the attempt, and the ``error`` estimate, if any.
   The action is ``accept`` or ``reject`` for steps judged by the
   controller and, for implicit integrators whose nonlinear solver
   failed to converge, ``retry`` if the step is to be retried with a
   rebuilt preconditioner or ``fail`` if the step size is to be
   reduced.

#. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

#. ``header`` --- if to output a header row or not:

    *boolean*

#. ``file-reset`` --- if to clear any existing output when starting a
   new simulation; restarts always append:

    *boolean*

#. ``stage-file`` --- output file path for per-stage convergence
   statistics (implicit integrators only); if not specified, no stage
   file is written:

    *string*

   Each row records, for one stage of one step, the nonlinear
   iteration count, residual norms and convergence status; the Krylov
   tolerance, retries, matrix-vector products and preconditioner
   applications; the right-hand side evaluations of the stage and the
   wall time of the solve; and the state of the preconditioner,
   including whether it was built during the stage, the wall time of
   that build, and the drift of :math:`\gamma \Delta t` since it was
   built.  Columns which do not
   apply to the nonlinear solver in use are left empty.

Example:

.. code-block:: ini

    [soln-plugin-dtstats]
    file = dtstats.csv
    flushsteps = 100
    ; For implicit integrators, optionally output stage convergence info
    stage-file = dtstats-stages.csv
