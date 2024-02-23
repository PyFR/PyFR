************************************
[solver-dual-time-integrator-multip]
************************************

Parameterises multi-p for dual time-stepping with

1. ``pseudo-dt-fact`` --- factor by which the pseudo time-step size
   changes between multi-p levels:

    *float*

2. ``cycle`` --- nature of a single multi-p cycle:

    ``[(order, nsteps), (order, nsteps), ... (order, nsteps)]``

    where ``order`` in the first and last bracketed pair must be the
    overall polynomial order used for the simulation, ``order`` can
    only change by one between subsequent bracketed pairs, and
    ``nsteps`` is a non-negative rational number.

Example::

    [solver-dual-time-integrator-multip]
    pseudo-dt-fact = 2.3
    cycle = [(3, 0.1), (2, 0.1), (1, 0.2), (0, 1.4), (1, 1.1), (2, 1.1), (3, 4.5)]

Used in the following Examples:

1. :ref:`2D Incompressible Cylinder Flow`