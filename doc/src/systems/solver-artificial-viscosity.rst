*****************************
[solver-artificial-viscosity]
*****************************

Parameterises artificial viscosity for shock capturing with

1. ``max-artvisc`` --- maximum artificial viscosity:

    *float*

2. ``s0`` --- sensor cut-off:

    *float*

3. ``kappa`` --- sensor range:

    *float*

Example::

    [solver-artificial-viscosity]
    max-artvisc = 0.01
    s0 = 0.01
    kappa = 5.0
