***********************
[solver-entropy-filter]
***********************

Parameterises entropy filter for shock capturing with

1. ``d-min`` --- minimum allowable density:

    *float*

2. ``p-min`` --- minimum allowable pressure:

    *float*

3. ``e-tol`` --- entropy tolerance:

    *float*

Example::

    [solver-entropy-filter]
    d-min = 1e-6
    p-min = 1e-6
    e-tol = 1e-6
