*******************
[solver-interfaces]
*******************

Parameterises the interfaces with

1. ``riemann-solver`` --- type of Riemann solver:

    ``rusanov`` | ``hll`` | ``hllc`` | ``roe`` | ``roem`` | ``exact``

    where

    ``hll`` | ``hllc`` | ``roe`` | ``roem`` | ``exact`` do not work with
    ``ac-euler`` | ``ac-navier-stokes``

2. ``ldg-beta`` --- beta parameter used for LDG:

    *float*

3. ``ldg-tau`` --- tau parameter used for LDG:

    *float*

Example::

    [solver-interfaces]
    riemann-solver = rusanov
    ldg-beta = 0.5
    ldg-tau = 0.1
