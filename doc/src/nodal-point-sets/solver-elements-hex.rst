*************************************
[solver-elements-hex{-mg-p\ *order*}]
*************************************

Parameterises the hexahedral elements, or if -mg-p\ *order* is suffixed
the hexahedral elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a hexahedral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   hexahedral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   hexahedral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-elements-hex]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre
