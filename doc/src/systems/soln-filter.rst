*************
[soln-filter]
*************

Parameterises an exponential solution filter with

1. ``nsteps`` --- apply filter every ``nsteps``:

    *int*

2. ``alpha`` --- strength of filter:

    *float*

3. ``order`` --- order of filter:

    *int*

4. ``cutoff`` --- cutoff frequency below which no filtering is applied:

    *int*

Example::

    [soln-filter]
    nsteps = 10
    alpha = 36.0
    order = 16
    cutoff = 1
