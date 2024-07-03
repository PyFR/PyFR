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

4. ``e-func`` --- type of entropy functional:

    ``physical`` | ``numerical``

5. ``niters`` --- number of filter strength solver iterations:

    *int*

Example::

    [solver-entropy-filter]
    d-min = 1e-6
    p-min = 1e-6
    e-tol = 1e-6
    e-func = physical
    niters = 2

Used in the following Examples:

1. :ref:`2D Double Mach Reflection`

2. :ref:`2D Viscous Shock Tube`
