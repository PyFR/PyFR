*************************
[soln-plugin-pseudostats]
*************************

Write pseudo-step convergence history out to a CSV file. Parameterised
with

#. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

#. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

#. ``file-header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-pseudostats]
    flushsteps = 100
    file = pseudostats.csv
    file-header = true
