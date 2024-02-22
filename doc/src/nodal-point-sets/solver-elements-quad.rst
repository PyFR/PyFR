**************************************
[solver-elements-quad{-mg-p\ *order*}]
**************************************

Parameterises the quadrilateral elements, or if -mg-p\ *order* is
suffixed the quadrilateral elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a quadrilateral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   quadrilateral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   quadrilateral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-elements-quad]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre
