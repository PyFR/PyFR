*******************************
[soln-plugin-fluidforce-*name*]
*******************************

Periodically integrates the pressure and viscous stress on the boundary
labelled ``name`` and writes out the resulting force and moment (if requested)
vectors to a CSV file. Parameterised with

1. ``nsteps`` --- integrate every ``nsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

4. ``morigin`` --- origin used to compute moments (optional):

    ``(x, y, [z])``

5. ``quad-deg-{etype}`` --- degree of quadrature rule for fluid force
   integration, optionally this can be specified for different element types:

    *int*

6. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

Example::

    [soln-plugin-fluidforce-wing]
    nsteps = 10
    file = wing-forces.csv
    header = true
    quad-deg = 6
    morigin = (0.0, 0.0, 0.5)
