****************************************
[solver-interfaces-quad{-mg-p\ *order*}]
****************************************

Parameterises the quadrilateral interfaces, or if -mg-p\ *order* is
suffixed the quadrilateral interfaces at multi-p level *order*, with

1. ``flux-pts`` --- location of the flux points on a quadrilateral
   interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing on a
   quadrilateral interface:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing on a
   quadrilateral interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-interfaces-quad]
    flux-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre
