**********************
[soln-plugin-residual]
**********************

Periodically calculates the residual and writes it out to a CSV file.
Parameterised with

#. ``nsteps`` --- calculate every ``nsteps``:

    *int*

#. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

#. ``file-header`` --- if to output a header row or not:

    *boolean*

#. ``norm`` --- sets the degree and calculates an :math:`L_p` norm,
    default is ``2``:

    *float* | ``inf``

Example::

    [soln-plugin-residual]
    nsteps = 10
    file = residual.csv
    file-header = true
    norm = inf
