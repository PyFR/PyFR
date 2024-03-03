*************************
[soln-plugin-pseudostats]
*************************

Write pseudo-step convergence history out to a CSV file. Parameterised
with

1. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-pseudostats]
    flushsteps = 100
    file = pseudostats.csv
    header = true
