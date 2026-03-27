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

#. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

#. ``header`` --- if to output a header row or not:

    *boolean*

#. ``stage-file`` --- output file path for per-stage convergence
   statistics (implicit integrators only); if not specified, no stage
   file is written:

    *string*

Example:

.. code-block:: ini

    [soln-plugin-dtstats]
    file = dtstats.csv
    flushsteps = 100
    ; For implicit integrators, optionally output stage convergence info
    stage-file = dtstats-stages.csv
