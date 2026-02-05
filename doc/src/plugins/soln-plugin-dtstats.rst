*********************
[soln-plugin-dtstats]
*********************

Write time-step statistics out to a CSV file. Parameterised with

#. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

#. ``file`` --- output file path; should the file already exist it will
   be appended to:

    *string*

#. ``file-header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-dtstats]
    flushsteps = 100
    file = dtstats.csv
    file-header = true
